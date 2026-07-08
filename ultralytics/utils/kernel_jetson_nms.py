# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Jetson Nano optimized CUDA NMS kernel (Phase 2 reference implementation).

Provides a custom CUDA non-maximum suppression kernel targeting the Jetson Nano's Maxwell architecture (sm_53). The
kernel is lazy-compiled on-device via `torch.utils.cpp_extension.load_inline` on first call and registered with the
kernel dispatcher so the inference pipeline can transparently adopt it when running on Jetson Nano hardware.

Design choices for Jetson Nano (Maxwell, 128 CUDA cores, 4GB shared CPU/GPU memory):
- Single-block kernel with parallel IoU: thread 0 picks the next surviving box, then ALL threads cooperatively
  compute IoU against that box and suppress in parallel. This leverages the 128-core GPU for the O(N) inner loop
  of each suppression round, while keeping the O(K) outer loop (K = kept boxes) sequential.
- Shared memory for boxes: avoids repeated global memory reads during the IoU computation.
- FP32 throughout: Maxwell lacks INT8 tensor cores and efficient FP16 compute, so FP32 is the native fast path.
- The keep-count compaction is done on-GPU (atomicAdd) to avoid a CPU sync round-trip per call.

When CUDA is unavailable or compilation fails (e.g. no nvcc on the device), the dispatcher's fallback returns None and
callers use the existing `TorchNMS.nms` / `torchvision.ops.nms` path — no regression on unsupported hardware.

Examples:
    >>> from ultralytics.utils.kernel_jetson_nms import jetson_nms, register_jetson_nms
    >>> register_jetson_nms()  # register with dispatcher; no-op if CUDA unavailable
"""

from __future__ import annotations

import functools
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.kernel_dispatch import register_kernel

# CUDA C++ source for the NMS kernel. Compiled via load_inline targeting the current device's architecture.
# Algorithm: sort by score descending (done in PyTorch before the kernel), then iteratively pick the top surviving box
# and suppress all remaining boxes with IoU > threshold. The single-block design uses shared memory for boxes.
# All threads cooperate on the IoU computation for each suppression round — thread 0 selects the next box, then
# every thread checks a subset of remaining boxes and writes suppression flags in parallel.
_CUDA_NMS_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

// Parallel single-block NMS kernel for small N (YOLO post-filter regime, N <= 2700).
// boxes: (N, 4) float32 xyxy, sorted by score descending.
// keep:  (K,) int64, filled with kept indices (K <= N, compacted via atomicAdd counter).
// iou_threshold: float32 scalar.
__global__ void nms_kernel(const float* __restrict__ boxes,
                           int64_t* __restrict__ keep,
                           int* __restrict__ keep_count,
                           float iou_threshold,
                           int n) {
    extern __shared__ float s_data[];  // 4*n floats (boxes) + n ints (suppressed flags)

    float* s_boxes = s_data;
    int* s_suppressed = (int*)(s_data + n * 4);  // int array for coalesced access

    int tid = threadIdx.x;
    int blockDim_x = blockDim.x;

    // Load boxes into shared memory (4 floats per box, coalesced)
    for (int i = tid; i < n * 4; i += blockDim_x) {
        s_boxes[i] = boxes[i];
    }
    // Initialize suppressed flags to 0
    for (int i = tid; i < n; i += blockDim_x) {
        s_suppressed[i] = 0;
    }
    if (tid == 0) *keep_count = 0;
    __syncthreads();

    // Thread 0 drives the selection loop; all threads cooperate on suppression
    for (int i = 0; i < n; i++) {
        // Thread 0 checks if box i is suppressed; if not, it's the next kept box
        __shared__ bool s_is_suppressed;
        if (tid == 0) {
            s_is_suppressed = (s_suppressed[i] != 0);
        }
        __syncthreads();

        if (s_is_suppressed) continue;

        // Thread 0 writes the kept index and increments the counter
        if (tid == 0) {
            int idx = atomicAdd(keep_count, 1);
            keep[idx] = i;
        }

        // All threads cooperate to suppress boxes j > i with high IoU
        float ix1 = s_boxes[i * 4 + 0];
        float iy1 = s_boxes[i * 4 + 1];
        float ix2 = s_boxes[i * 4 + 2];
        float iy2 = s_boxes[i * 4 + 3];
        float iarea = (ix2 - ix1) * (iy2 - iy1);
        if (iarea <= 0.0f) {
            __syncthreads();
            continue;
        }

        // Each thread checks a subset of boxes j = i+1..n-1
        for (int j = i + 1 + tid; j < n; j += blockDim_x) {
            if (s_suppressed[j]) continue;
            float jx1 = s_boxes[j * 4 + 0];
            float jy1 = s_boxes[j * 4 + 1];
            float jx2 = s_boxes[j * 4 + 2];
            float jy2 = s_boxes[j * 4 + 3];

            float xx1 = fmaxf(ix1, jx1);
            float yy1 = fmaxf(iy1, jy1);
            float xx2 = fminf(ix2, jx2);
            float yy2 = fminf(iy2, jy2);

            float w = fmaxf(0.0f, xx2 - xx1);
            float h = fmaxf(0.0f, yy2 - yy1);
            float inter = w * h;
            float jarea = (jx2 - jx1) * (jy2 - jy1);
            float iou = inter / (iarea + jarea - inter + 1e-8f);
            if (iou > iou_threshold) {
                s_suppressed[j] = 1;
            }
        }
        __syncthreads();
    }
}

// Host wrapper: boxes must be (N, 4) float32 on CUDA, sorted by score descending.
// Returns (K,) int64 tensor of kept indices (compacted, no sentinels).
at::Tensor nms(at::Tensor boxes, double iou_threshold) {
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N, 4)");
    TORCH_CHECK(boxes.scalar_type() == at::kFloat, "boxes must be float32");

    int n = boxes.size(0);
    auto keep = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(boxes.device()));
    auto keep_count = at::zeros({1}, at::TensorOptions().dtype(at::kInt).device(boxes.device()));

    if (n == 0) return keep;

    // Shared memory: 4*n floats (boxes) + n ints (suppressed flags) = n * (16 + 4) = 20*n bytes.
    // 48KB / 20 = ~2457 boxes. Use 2300 as a conservative cap.
    const int MAX_N = 2300;
    TORCH_CHECK(n <= MAX_N, "nms: N exceeds shared-memory cap (", MAX_N, "); use torchvision fallback for N > ", MAX_N);

    size_t shared_mem_bytes = (size_t)n * 4 * sizeof(float) + (size_t)n * sizeof(int);

    // Use 256 threads for parallel IoU computation
    nms_kernel<<<1, 256, shared_mem_bytes>>>(
        boxes.data_ptr<float>(),
        keep.data_ptr<int64_t>(),
        keep_count.data_ptr<int>(),
        (float)iou_threshold,
        n
    );

    // Read keep_count from GPU (single int — minimal sync) and truncate
    int k = keep_count.item<int>();
    return keep.narrow(0, 0, k).clone();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms", &nms, "Jetson Nano NMS (CUDA)");
}
"""


@functools.lru_cache(maxsize=1)
def _compile_jetson_nms():
    """Lazy-compile the CUDA NMS kernel via load_inline on first call.

    Returns the compiled module with an `nms(boxes, iou_threshold)` callable, or raises if compilation fails.
    """
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="jetson_nms",
        cpp_sources=[""],  # no C++ source needed; PYBIND11_MODULE is in the CUDA source
        cuda_sources=[_CUDA_NMS_SOURCE],
        functions=None,  # don't auto-generate binding — we provide PYBIND11_MODULE in the CUDA source
        verbose=False,
    )


# Maximum N the single-block kernel can handle (limited by 48KB shared memory: 20*N bytes < 48KB).
# The Python wrapper falls back to torchvision for N above this threshold.
_MAX_KERNEL_N = 2300


def jetson_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Run the Jetson Nano CUDA NMS kernel on sorted boxes.

    Falls back to torchvision.ops.nms when N exceeds the shared-memory cap or when the custom kernel
    raises at runtime, so callers always get a correct result.

    Args:
        boxes (torch.Tensor): (N, 4) float32 xyxy boxes on CUDA.
        scores (torch.Tensor): (N,) confidence scores on CUDA.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        (torch.Tensor): (K,) int64 indices of kept boxes, sorted by score descending.
    """
    # Sort by score descending (PyTorch handles this efficiently on GPU), then call the CUDA kernel
    order = scores.argsort(descending=True)
    sorted_boxes = boxes[order].contiguous().float()

    if sorted_boxes.shape[0] > _MAX_KERNEL_N:
        # Fall back to torchvision for large N (shared memory can't hold the boxes)
        import torchvision  # scoped as slow import

        return order[torchvision.ops.nms(sorted_boxes, scores[order], iou_threshold)]

    module = _compile_jetson_nms()
    try:
        keep_local = module.nms(sorted_boxes, float(iou_threshold))  # indices into the sorted array
    except Exception as e:
        LOGGER.warning(f"kernel_jetson_nms: CUDA kernel failed ({e}), falling back to torchvision")
        import torchvision  # scoped as slow import

        return order[torchvision.ops.nms(sorted_boxes, scores[order], iou_threshold)]
    # Map back to original indices
    return order[keep_local.to(order.device)]


def register_jetson_nms(priority: int = 10) -> None:
    """Register the Jetson Nano NMS kernel with the global dispatcher.

    Targets CUDA devices with compute capability >= 5.3 (Maxwell, Jetson Nano). The kernel is lazy-compiled on first
    call, so registration itself is cheap and safe on non-CUDA hosts (it just records the entry; resolution only
    returns it when a matching CUDA device is detected at runtime).

    Args:
        priority (int): Dispatcher priority; higher wins among matching kernels. Default 10 puts this above a
            potential generic CUDA kernel at priority 0 but below any hand-tuned per-arch kernel.
    """
    if not torch.cuda.is_available():
        LOGGER.debug("kernel_jetson_nms: CUDA unavailable, registration is a no-op (fallback will be used)")
        return

    register_kernel(
        operation="nms",
        backend="cuda",
        implementation=jetson_nms,
        priority=priority,
        min_compute_capability=(5, 3),  # Jetson Nano is the floor; covers all CUDA GPUs Maxwell+
        hardware=None,  # not arch-restricted beyond the CC floor — works on T4/RTX too for testing
    )
    LOGGER.info("kernel_jetson_nms: registered Jetson Nano CUDA NMS kernel (sm_53+, priority=%d)", priority)

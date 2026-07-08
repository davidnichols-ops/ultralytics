# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Jetson Nano optimized CUDA NMS kernel (Phase 2 reference implementation).

Provides a custom CUDA non-maximum suppression kernel targeting the Jetson Nano's Maxwell architecture (sm_53). The
kernel is lazy-compiled on-device via `torch.utils.cpp_extension.load_inline` on first call and registered with the
kernel dispatcher so the inference pipeline can transparently adopt it when running on Jetson Nano hardware.

Design choices for Jetson Nano (Maxwell, 128 CUDA cores, 4GB shared CPU/GPU memory):
- Single-block kernel using shared memory: YOLO post-filter box counts are small (typically <1000, often <100), so a
  single thread block with shared-memory boxes avoids the global-memory traffic and launch overhead that hurts
  low-core-count GPUs. This trades scalability on huge N (not a YOLO post-filter scenario) for low latency on small N.
- FP32 throughout: Maxwell lacks INT8 tensor cores and efficient FP16 compute, so FP32 is the native fast path.
- No dynamic parallelism: Maxwell (sm_53) supports it poorly; the iterative suppress loop runs in a single block.

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
# and suppress all remaining boxes with IoU > threshold. The single-block design uses shared memory for boxes and
# a bitmask for the "suppressed" state, keeping global memory traffic to one read of boxes + one write of keep indices.
_CUDA_NMS_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

// Single-block NMS kernel for small N (YOLO post-filter regime, N <= 4096).
// boxes: (N, 4) float32 xyxy, sorted by score descending.
// keep:  (N,) int64, filled with kept indices (-1 sentinel after the last valid entry).
// iou_threshold: float32 scalar.
__global__ void nms_kernel(const float* __restrict__ boxes,
                           int64_t* __restrict__ keep,
                           float iou_threshold,
                           int n) {
    extern __shared__ float s_boxes[];  // 4 * n floats

    int tid = threadIdx.x;
    // Load boxes into shared memory (4 floats per box, coalesced across threads)
    for (int i = tid; i < n * 4; i += blockDim.x) {
        s_boxes[i] = boxes[i];
    }
    __syncthreads();

    // Suppressed flags: one byte per box, in shared memory via a separate allocation passed as dynamic shared
    // We use a simple approach: each thread checks IoU against the current "best" box.
    // Thread 0 drives the iterative selection loop.
    if (tid == 0) {
        int keep_count = 0;
        // Simple suppressed array in shared memory (reuse tail of s_boxes after the 4*n floats)
        // We allocated 4*n + n bytes conceptually; cast the tail to char*
        char* suppressed = (char*)(s_boxes + n * 4);
        for (int i = 0; i < n; i++) suppressed[i] = 0;

        for (int i = 0; i < n; i++) {
            if (suppressed[i]) continue;
            keep[keep_count++] = i;  // keep original index (caller sorted, so i is the rank)
            if (keep_count >= n) break;

            float ix1 = s_boxes[i * 4 + 0];
            float iy1 = s_boxes[i * 4 + 1];
            float ix2 = s_boxes[i * 4 + 2];
            float iy2 = s_boxes[i * 4 + 3];
            float iarea = (ix2 - ix1) * (iy2 - iy1);
            if (iarea <= 0.0f) continue;

            for (int j = i + 1; j < n; j++) {
                if (suppressed[j]) continue;
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
                    suppressed[j] = 1;
                }
            }
        }
        // Sentinel: fill remaining slots with -1
        for (int k = keep_count; k < n; k++) keep[k] = -1;
    }
}

// Host wrapper: boxes must be (N, 4) float32 on CUDA, sorted by score descending.
// Returns (keep_count,) int64 tensor of kept indices (compact, no sentinels).
// Named `nms` so load_inline's auto-generated pybind binding (functions=["nms"]) finds it.
at::Tensor nms(at::Tensor boxes, double iou_threshold) {
    TORCH_CHECK(boxes.is_cuda(), "boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "boxes must be (N, 4)");
    TORCH_CHECK(boxes.scalar_type() == at::kFloat, "boxes must be float32");

    int n = boxes.size(0);
    auto keep = at::empty({n}, at::TensorOptions().dtype(at::kLong).device(boxes.device()));

    if (n == 0) return keep;

    // Cap N to avoid exceeding shared memory on Maxwell (48KB per block on sm_53).
    // 4*n floats (boxes) + n bytes (suppressed) = 5*n bytes. 48KB / 5 = ~9600 boxes max.
    // For YOLO post-filter this is never hit; fall back to torchvision for pathological N.
    const int MAX_N = 4096;  // conservative cap for shared memory + register pressure
    TORCH_CHECK(n <= MAX_N, "nms_forward: N exceeds single-block cap; use torchvision fallback for N > ", MAX_N);

    // Shared memory: 4*n floats for boxes + n bytes for suppressed flags
    size_t shared_mem_bytes = 4 * n * sizeof(float) + n * sizeof(char);

    nms_kernel<<<1, 256, shared_mem_bytes>>>(
        boxes.data_ptr<float>(),
        keep.data_ptr<int64_t>(),
        (float)iou_threshold,
        n
    );

    // Compact: remove -1 sentinels by scanning and truncating
    // (Simple approach: find the first -1 and narrow the tensor)
    auto keep_cpu = keep.cpu();
    auto keep_acc = keep_cpu.accessor<int64_t, 1>();
    int keep_count = 0;
    for (int i = 0; i < n; i++) {
        if (keep_acc[i] >= 0) keep_count++;
        else break;
    }
    return keep.narrow(0, 0, keep_count).clone();
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
        cpp_sources=["at::Tensor nms(at::Tensor boxes, double iou_threshold);"],
        cuda_sources=[_CUDA_NMS_SOURCE],
        functions=["nms"],
        verbose=False,
    )


def jetson_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Run the Jetson Nano CUDA NMS kernel on sorted boxes.

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

    module = _compile_jetson_nms()
    keep_local = module.nms(sorted_boxes, float(iou_threshold))  # indices into the sorted array
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

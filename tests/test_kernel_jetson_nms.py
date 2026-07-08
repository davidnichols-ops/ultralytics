# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tests for the Jetson Nano CUDA NMS kernel and its dispatcher registration.

CUDA-dependent tests (compilation, correctness vs torchvision, benchmarking) are skipped on CPU-only runners and are
exercised by the Kaggle/Colab kernel that runs on a T4 GPU. Locally we verify the registration logic, fallback
behavior, and the PyTorch-side sort+index-mapping wrapper with mocked capabilities.
"""

import pytest
import torch

from ultralytics.utils.kernel_dispatch import (
    DeviceCapabilities,
    KernelEntry,
    KernelRegistry,
    dispatcher,
    resolve_kernel,
)
from ultralytics.utils.kernel_jetson_nms import (
    _CUDA_NMS_SOURCE,
    jetson_nms,
    register_jetson_nms,
)


def _jetson_nano_caps():
    """DeviceCapabilities fixture for a Jetson Nano (Maxwell, sm_53)."""
    return DeviceCapabilities(
        type="cuda",
        vendor="nvidia",
        name="NVIDIA Jetson Nano",
        compute_capability=(5, 3),
        arch="Maxwell",
        supports_fp16=True,
        supports_int8=False,  # Maxwell lacks INT8 tensor cores
        supports_bf16=False,
    )


def _t4_caps():
    """DeviceCapabilities fixture for a Colab/Kaggle T4 (Turing, sm_75)."""
    return DeviceCapabilities(
        type="cuda",
        vendor="nvidia",
        name="NVIDIA Tesla T4",
        compute_capability=(7, 5),
        arch="Turing",
        supports_fp16=True,
        supports_int8=True,
        supports_bf16=False,
    )


def _cpu_caps():
    """DeviceCapabilities fixture for a CPU (fallback path)."""
    return DeviceCapabilities(type="cpu", vendor="intel", name="Intel Core i7", supports_fp16=True, supports_int8=True)


def test_cuda_source_contains_nms_kernel():
    """The embedded CUDA source defines the nms_kernel, host wrapper, and pybind binding."""
    assert "nms_kernel" in _CUDA_NMS_SOURCE
    assert "at::Tensor nms(" in _CUDA_NMS_SOURCE  # host wrapper
    assert "PYBIND11_MODULE" in _CUDA_NMS_SOURCE  # self-contained binding (no auto-generation)
    assert "extern __shared__" in _CUDA_NMS_SOURCE  # shared-memory single-block design


def test_register_jetson_nms_noop_without_cuda(monkeypatch):
    """register_jetson_nms is a no-op (no registration) when CUDA is unavailable."""
    dispatcher.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    register_jetson_nms()
    assert dispatcher.entries() == []
    dispatcher.clear()


def test_register_jetson_nms_with_cuda(monkeypatch):
    """register_jetson_nms registers a kernel entry when CUDA is available."""
    dispatcher.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    register_jetson_nms(priority=10)
    entries = dispatcher.entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.operation == "nms"
    assert entry.backend == "cuda"
    assert entry.implementation is jetson_nms
    assert entry.priority == 10
    assert entry.min_compute_capability == (5, 3)
    dispatcher.clear()


def test_resolve_jetson_nms_on_jetson_nano(monkeypatch):
    """The dispatcher resolves the Jetson Nano kernel on a Jetson Nano capability."""
    dispatcher.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    register_jetson_nms()
    fn = resolve_kernel("nms", _jetson_nano_caps())
    assert fn is jetson_nms
    dispatcher.clear()


def test_resolve_jetson_nms_on_t4(monkeypatch):
    """The kernel also resolves on a T4 (sm_75 >= sm_53 floor) — used for Colab/Kaggle testing."""
    dispatcher.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    register_jetson_nms()
    fn = resolve_kernel("nms", _t4_caps())
    assert fn is jetson_nms
    dispatcher.clear()


def test_resolve_jetson_nms_not_on_cpu(monkeypatch):
    """The dispatcher returns None (PyTorch fallback) on a CPU capability."""
    dispatcher.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    register_jetson_nms()
    assert resolve_kernel("nms", _cpu_caps()) is None
    dispatcher.clear()


def test_jetson_nms_below_compute_capability_floor():
    """A kernel with min_compute_capability=(5,3) does not match a pre-Maxwell GPU (sm_50)."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", jetson_nms, min_compute_capability=(5, 3)))
    old_caps = DeviceCapabilities(
        type="cuda", vendor="nvidia", name="GTX 750", compute_capability=(5, 0), arch="Maxwell"
    )
    assert reg.resolve("nms", old_caps) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to compile and run the kernel")
def test_jetson_nms_correctness_vs_torchvision():
    """On a CUDA host, the Jetson Nano kernel matches torchvision NMS output for a small box set."""
    import torchvision

    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30], [50, 50, 60, 60], [2, 2, 12, 12]],
        dtype=torch.float32,
        device="cuda",
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5], dtype=torch.float32, device="cuda")
    iou_threshold = 0.5

    expected = torchvision.ops.nms(boxes, scores, iou_threshold)
    got = jetson_nms(boxes, scores, iou_threshold)

    # Both should keep the same boxes (order may differ in tie cases, so compare sets)
    assert set(expected.tolist()) == set(got.tolist())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to compile and run the kernel")
def test_jetson_nms_empty_input():
    """The kernel handles empty input without crashing."""
    boxes = torch.empty((0, 4), dtype=torch.float32, device="cuda")
    scores = torch.empty((0,), dtype=torch.float32, device="cuda")
    got = jetson_nms(boxes, scores, 0.5)
    assert got.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to compile and run the kernel")
def test_jetson_nms_no_overlaps():
    """With no overlapping boxes, all boxes are kept."""
    boxes = torch.tensor([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cuda")
    got = jetson_nms(boxes, scores, 0.5)
    assert set(got.tolist()) == {0, 1, 2}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required to compile and run the kernel")
def test_jetson_nms_all_overlapping():
    """With all boxes overlapping above threshold, only the highest-score box is kept."""
    boxes = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32, device="cuda")
    got = jetson_nms(boxes, scores, 0.5)
    assert got.tolist() == [0]

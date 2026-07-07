# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.utils.kernel_dispatch import (
    DeviceCapabilities,
    KernelEntry,
    KernelRegistry,
    detect_device,
    register_kernel,
    resolve_kernel,
)


def _cuda_caps(cc=(8, 9), name="NVIDIA GeForce RTX 4090", arch="Ada Lovelace"):
    """Build a CUDA DeviceCapabilities fixture for matching tests (no GPU required)."""
    return DeviceCapabilities(
        type="cuda",
        vendor="nvidia",
        name=name,
        compute_capability=cc,
        arch=arch,
        supports_fp16=True,
        supports_int8=cc >= (7, 0),
        supports_bf16=cc >= (8, 0),
    )


def _cpu_caps(name="Intel Core i7"):
    """Build a CPU DeviceCapabilities fixture for fallback tests."""
    return DeviceCapabilities(type="cpu", vendor="intel", name=name, supports_fp16=True, supports_int8=True)


def test_detect_device_cpu():
    """detect_device('cpu') returns a CPU-flavored capability with a non-empty name."""
    caps = detect_device("cpu")
    assert caps.type == "cpu"
    assert caps.name
    assert caps.supports_fp16  # CPU FP16 always available via PyTorch fallback


def test_detect_device_auto_no_crash():
    """detect_device() with no argument resolves the current device without raising."""
    caps = detect_device()
    assert caps.type in {"cuda", "cpu", "mps", "npu"}


def test_register_and_resolve_basic():
    """A registered kernel resolves on a matching device and returns None on a non-matching one."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "cuda_nms", priority=1))
    assert reg.resolve("nms", _cuda_caps())() == "cuda_nms"
    assert reg.resolve("nms", _cpu_caps()) is None  # no CUDA kernel on CPU -> PyTorch fallback signal


def test_resolve_unknown_operation_returns_none():
    """Resolving an operation with no registrations returns None (fallback)."""
    assert KernelRegistry().resolve("nonexistent_op", _cuda_caps()) is None


def test_priority_selects_highest():
    """Among multiple matching kernels, the highest priority implementation wins."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "low", priority=1))
    reg.register(KernelEntry("nms", "cuda", lambda: "high", priority=10))
    assert reg.resolve("nms", _cuda_caps())() == "high"


def test_priority_tie_breaks_by_registration_order():
    """Equal-priority registrations resolve to the last-registered implementation (override semantics)."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "first", priority=5))
    reg.register(KernelEntry("nms", "cuda", lambda: "second", priority=5))
    assert reg.resolve("nms", _cuda_caps())() == "second"


def test_min_compute_capability_filters():
    """min_compute_capability excludes devices below the requirement and admits those at or above."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "ampere_plus", min_compute_capability=(8, 6)))
    assert reg.resolve("nms", _cuda_caps(cc=(8, 9)))() == "ampere_plus"
    assert reg.resolve("nms", _cuda_caps(cc=(7, 5))) is None  # Turing below Ampere floor


def test_precision_constraint():
    """A precision requirement excludes devices that lack that precision support."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "int8", precision="int8"))
    assert reg.resolve("nms", _cuda_caps(cc=(8, 9)))() == "int8"  # Ada supports INT8
    assert reg.resolve("nms", _cuda_caps(cc=(6, 1))) is None  # Pascal lacks INT8 tensor cores


def test_hardware_arch_constraint():
    """The hardware constraint matches by architecture codename."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "ada_only", hardware="Ada Lovelace"))
    assert reg.resolve("nms", _cuda_caps(arch="Ada Lovelace"))() == "ada_only"
    assert reg.resolve("nms", _cuda_caps(arch="Ampere")) is None


def test_register_overrides_same_operation_backend():
    """Re-registering the same operation+backend replaces the prior entry rather than duplicating."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "v1", priority=1))
    reg.register(KernelEntry("nms", "cuda", lambda: "v2", priority=1))
    assert len([e for e in reg.entries() if e.operation == "nms" and e.backend == "cuda"]) == 1
    assert reg.resolve("nms", _cuda_caps())() == "v2"


def test_clear_removes_all_entries():
    """clear() empties the registry so subsequent resolves fall back to None."""
    reg = KernelRegistry()
    reg.register(KernelEntry("nms", "cuda", lambda: "x"))
    reg.clear()
    assert reg.entries() == []
    assert reg.resolve("nms", _cuda_caps()) is None


def test_global_dispatcher_register_and_resolve():
    """The module-level register_kernel/resolve_kernel convenience functions work end to end."""
    from ultralytics.utils.kernel_dispatch import dispatcher

    dispatcher.clear()
    register_kernel("nms", "cuda", lambda: "global_cuda_nms", priority=2)
    assert resolve_kernel("nms", _cuda_caps())() == "global_cuda_nms"
    assert resolve_kernel("nms", _cpu_caps()) is None
    dispatcher.clear()


def test_matches_device_capabilities_helper():
    """DeviceCapabilities.matches enforces backend, hardware, and compute-capability constraints together."""
    caps = _cuda_caps(cc=(8, 9), arch="Ada Lovelace")
    assert caps.matches("cuda", None, None)
    assert not caps.matches("cpu", None, None)
    assert caps.matches("cuda", "Ada Lovelace", (8, 6))
    assert not caps.matches("cuda", "Ampere", None)
    assert not caps.matches("cuda", None, (9, 0))  # Ada (8.9) below Hopper floor (9.0)


def test_detect_device_cuda_when_available():
    """When CUDA is available, detect_device('cuda:0') returns a CUDA capability with a compute capability tuple."""
    if not torch.cuda.is_available():
        return  # skip on CPU-only runners; CI's GPU job covers this path
    caps = detect_device("cuda:0")
    assert caps.type == "cuda"
    assert caps.vendor == "nvidia"
    assert caps.compute_capability is not None
    assert caps.compute_capability >= (5, 0)

"""RK1828 platform profile (RKNN3 PCIe coprocessor).

RK1828 is a PCIe endpoint accelerator (RISC-V + NPU, ~5GB) attached to a host
SoC (typically RK3588, e.g. Radxa ROCK 5T).  It is NOT the host SoC: the CPU
topology fields below describe the *host* and are informational only —
``is_coprocessor=True`` flags that affinity/CPU-mask logic must skip them
(see ``platform/base.py``).  The device is addressed by ``device_id`` (its
PCIe endpoint BDF) rather than by NPU core masks.
"""

from .base import PlatformProfile

RK1828 = PlatformProfile(
    name="rk1828",
    npu_cores=1,                   # single EP NPU; addressed via device_id
    npu_memory_limit_mb=5120,      # ~5GB on-module memory
    # Host RK3588 topology (informational only; skipped — is_coprocessor=True)
    cpu_big_cores=[4, 5, 6, 7],    # A76
    cpu_mid_cores=[],
    cpu_little_cores=[0, 1, 2, 3], # A55
    cpu_mask_big=0xF0,
    cpu_mask_all=0xFF,
    default_rkllm_domain=1,
    device_id="0001:11:00.0",
    is_coprocessor=True,
)

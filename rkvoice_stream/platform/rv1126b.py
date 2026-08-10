"""RV1126B platform profile (single-core NPU host SoC).

RV1126B is a low-power quad-core host SoC with a SINGLE NPU core. Unlike the
multi-core rk3576/rk3588 parts, its NPU has no core-mask concept: RKNNLite must
be initialised with a plain ``init_runtime()`` (no ``core_mask=``) — passing a
mask such as ``NPU_CORE_0`` errors on this platform. Backends detect this via
``npu_cores == 1`` (and ``is_coprocessor is False``, distinguishing it from the
PCIe RK1828 which is also single-core but addressed by ``device_id``).
"""

from .base import PlatformProfile

RV1126B = PlatformProfile(
    name="rv1126b",
    npu_cores=1,                   # single NPU core; no core-mask concept
    npu_memory_limit_mb=256,       # small on-SoC NPU working set
    # Quad Cortex-A7 host topology (informational).
    cpu_big_cores=[],
    cpu_mid_cores=[],
    cpu_little_cores=[0, 1, 2, 3],
    cpu_mask_big=0x0,
    cpu_mask_all=0xF,
    default_rkllm_domain=1,
)

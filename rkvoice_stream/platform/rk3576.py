"""RK3576 platform profile."""

from .base import PlatformProfile

RK3576 = PlatformProfile(
    name="rk3576",
    npu_cores=2,
    npu_memory_limit_mb=180,
    cpu_big_cores=[6, 7],          # A72
    cpu_mid_cores=[4, 5],          # A55
    cpu_little_cores=[0, 1, 2, 3], # A55
    cpu_mask_big=0xC0,             # CPU6+7
    cpu_mask_all=0xFF,
    default_rkllm_domain=1,
)

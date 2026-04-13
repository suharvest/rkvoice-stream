"""RK3588 platform profile."""

from .base import PlatformProfile

RK3588 = PlatformProfile(
    name="rk3588",
    npu_cores=3,
    npu_memory_limit_mb=512,
    cpu_big_cores=[4, 5, 6, 7],    # A76
    cpu_mid_cores=[],
    cpu_little_cores=[0, 1, 2, 3], # A55
    cpu_mask_big=0xF0,             # CPU4-7
    cpu_mask_all=0xFF,
    default_rkllm_domain=1,
)

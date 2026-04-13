"""Platform configuration for Rockchip NPU devices."""

from .base import PlatformProfile
from .rk3576 import RK3576
from .rk3588 import RK3588

PLATFORMS = {
    "rk3576": RK3576,
    "rk3588": RK3588,
}


def get_platform(name: str) -> PlatformProfile:
    """Get platform profile by name."""
    p = PLATFORMS.get(name.lower())
    if p is None:
        raise ValueError(
            f"Unknown platform: {name!r}. Available: {list(PLATFORMS.keys())}"
        )
    return p

"""Platform configuration for Rockchip NPU devices."""

from .base import PlatformProfile
from .rk3576 import RK3576
from .rk3588 import RK3588
from .rk1828 import RK1828
from .rv1126b import RV1126B

PLATFORMS = {
    "rk3576": RK3576,
    "rk3588": RK3588,
    "rk1828": RK1828,
    "rv1126b": RV1126B,
}


def get_platform(name: str) -> PlatformProfile:
    """Get platform profile by name."""
    p = PLATFORMS.get(name.lower())
    if p is None:
        raise ValueError(
            f"Unknown platform: {name!r}. Available: {list(PLATFORMS.keys())}"
        )
    return p


# Platform-aware RKNNLite init helper (imported late to avoid a cycle:
# runtime.py imports get_platform from this module).
from .runtime import init_runtime_for_platform, platform_is_single_core  # noqa: E402

__all__ = [
    "PlatformProfile",
    "PLATFORMS",
    "get_platform",
    "init_runtime_for_platform",
    "platform_is_single_core",
]

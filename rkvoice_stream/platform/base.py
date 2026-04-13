"""Base platform profile dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlatformProfile:
    """Hardware profile for a Rockchip NPU platform.

    Used by capability.py for conflict detection and by backends
    for selecting appropriate NPU core masks and CPU affinity.
    """

    name: str
    npu_cores: int
    npu_memory_limit_mb: int

    # CPU topology
    cpu_big_cores: list[int] = field(default_factory=list)
    cpu_mid_cores: list[int] = field(default_factory=list)
    cpu_little_cores: list[int] = field(default_factory=list)

    # CPU masks for RKLLM enabled_cpus_mask
    cpu_mask_big: int = 0
    cpu_mask_all: int = 0xFF

    # Default RKLLM domain (1 to avoid conflict with RKNN domain 0)
    default_rkllm_domain: int = 1

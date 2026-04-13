"""Service capability and conflict detection.

Checks whether ASR and TTS backends can coexist on the same device
based on their NPU resource requirements and the platform limits.

Used at startup (fail-fast) and at runtime via /capabilities API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResourceProfile:
    """NPU resource requirements declared by a backend."""

    npu_domain: int = 0              # 0 or 1
    npu_cores: list[str] = field(default_factory=lambda: ["CORE_AUTO"])
    uses_rkllm: bool = False         # RKLLM occupies NPU scheduler
    npu_memory_mb: int = 0           # Estimated NPU memory usage
    label: str = ""                  # Human-readable name


def check_compatibility(
    asr_profile: Optional[ResourceProfile],
    tts_profile: Optional[ResourceProfile],
    platform_name: str = "rk3576",
) -> dict:
    """Check whether ASR and TTS can coexist.

    Returns:
        {
            "compatible": bool,
            "mode": "parallel" | "sequential" | "exclusive",
            "warnings": [str],
            "errors": [str],
        }
    """
    from rkvoice_stream.platform import get_platform

    platform = get_platform(platform_name)
    warnings: list[str] = []
    errors: list[str] = []

    if asr_profile is None or tts_profile is None:
        return {
            "compatible": True,
            "mode": "single",
            "warnings": [],
            "errors": [],
        }

    # Check NPU memory
    total_mem = asr_profile.npu_memory_mb + tts_profile.npu_memory_mb
    if total_mem > platform.npu_memory_limit_mb:
        errors.append(
            f"NPU memory exceeds limit: {total_mem}MB > {platform.npu_memory_limit_mb}MB "
            f"({asr_profile.label}={asr_profile.npu_memory_mb}MB + "
            f"{tts_profile.label}={tts_profile.npu_memory_mb}MB)"
        )

    # Check domain conflict
    if asr_profile.npu_domain == tts_profile.npu_domain:
        if asr_profile.uses_rkllm or tts_profile.uses_rkllm:
            warnings.append(
                f"RKLLM and RKNN share NPU domain {asr_profile.npu_domain}. "
                f"Set base_domain_id=1 for RKLLM to isolate."
            )

    # Check core overlap
    asr_cores = set(asr_profile.npu_cores)
    tts_cores = set(tts_profile.npu_cores)
    overlap = asr_cores & tts_cores - {"CORE_AUTO"}
    if overlap:
        warnings.append(
            f"NPU core overlap: {overlap}. ASR and TTS must run sequentially."
        )

    # Determine mode
    if errors:
        mode = "exclusive"
        compatible = False
    elif overlap or (asr_profile.uses_rkllm and tts_profile.uses_rkllm):
        mode = "sequential"
        compatible = True
    else:
        mode = "parallel"
        compatible = True

    return {
        "compatible": compatible,
        "mode": mode,
        "warnings": warnings,
        "errors": errors,
    }


def check_on_startup(
    asr_profile: Optional[ResourceProfile],
    tts_profile: Optional[ResourceProfile],
    platform_name: str = "rk3576",
) -> None:
    """Run at startup. Logs warnings, raises on errors."""
    import logging

    logger = logging.getLogger(__name__)
    result = check_compatibility(asr_profile, tts_profile, platform_name)

    for w in result["warnings"]:
        logger.warning("Capability check: %s", w)

    if result["errors"]:
        for e in result["errors"]:
            logger.error("Capability check: %s", e)
        raise RuntimeError(
            f"Service configuration conflict: {'; '.join(result['errors'])}"
        )

    logger.info(
        "Capability check passed: mode=%s, warnings=%d",
        result["mode"], len(result["warnings"]),
    )

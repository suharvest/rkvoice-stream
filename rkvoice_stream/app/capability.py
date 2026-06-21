"""Service capability and conflict detection.

Checks whether speech backends (ASR / TTS / AudioLLM) can coexist on the same
device based on their NPU resource requirements and the platform limits.

Used at startup (fail-fast) and at runtime via /capabilities API.

Device-id dimension (Phase 2)
-----------------------------
Each backend declares which *device* it runs on via ``ResourceProfile.device_id``.
A PCIe coprocessor such as the RK1828 (``is_coprocessor=True``, addressed by its
PCIe BDF, e.g. ``"0001:11:00.0"``) is a **separate** device from the host SoC's
NPU: it has its own ~5GB memory budget and its own core/domain namespace, so it
does NOT contend with host-SoC backends. Backends are bucketed by ``device_id``;
the memory limit and core/domain conflict checks run **per device**.

Backward compatibility: when every profile has ``device_id=None`` (the RK3576 /
RK3588 single-SoC case) all profiles fall into one host bucket and the behaviour
is byte-for-byte identical to the original single-platform model.
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
    # Device this backend runs on. ``None`` = the host SoC NPU (RK3576/88).
    # A PCIe coprocessor (RK1828) sets its device_id (PCIe BDF), making it a
    # separate resource bucket that does not contend with the host.
    device_id: Optional[str] = None


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

    # ── Device-id dimension ──────────────────────────────────────────────
    # If the two backends live on different devices (e.g. ASR on the host SoC,
    # TTS on the RK1828 PCIe coprocessor), they share no NPU memory budget and
    # no core/domain namespace → unconditionally parallel-compatible.
    if asr_profile.device_id != tts_profile.device_id:
        return {
            "compatible": True,
            "mode": "parallel",
            "warnings": [],
            "errors": [],
        }

    # Same device: pick that device's memory limit. ``device_id=None`` means the
    # host SoC (``platform_name``); a coprocessor uses its own platform limit.
    mem_limit = _device_memory_limit(asr_profile.device_id, platform)

    # Check NPU memory
    total_mem = asr_profile.npu_memory_mb + tts_profile.npu_memory_mb
    if total_mem > mem_limit:
        errors.append(
            f"NPU memory exceeds limit: {total_mem}MB > {mem_limit}MB "
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


def _device_memory_limit(device_id: Optional[str], host_platform) -> int:
    """Memory budget (MB) for a device bucket.

    ``device_id=None`` → the host SoC NPU (``host_platform``). A non-None id is a
    PCIe coprocessor: look it up among the registered platforms by matching
    ``PlatformProfile.device_id``; fall back to the host limit if unknown.
    """
    if device_id is None:
        return host_platform.npu_memory_limit_mb

    from rkvoice_stream.platform import PLATFORMS

    for prof in PLATFORMS.values():
        if prof.device_id == device_id:
            return prof.npu_memory_limit_mb
    return host_platform.npu_memory_limit_mb


def check_resources(
    profiles: list[ResourceProfile],
    platform_name: str = "rk3576",
) -> dict:
    """Device-aware conflict check for an arbitrary set of backends.

    Buckets ``profiles`` by ``device_id`` and runs an independent budget per
    device: each device's resident backends must fit that device's memory limit,
    and core/domain overlaps are evaluated only *within* a device. Cross-device
    backends (host SoC vs RK1828 EP) never contend.

    Returns::

        {
            "compatible": bool,
            "devices": {
                <device_key>: {
                    "mode": "single" | "parallel" | "sequential" | "exclusive",
                    "total_memory_mb": int,
                    "memory_limit_mb": int,
                    "labels": [str],
                    "warnings": [str],
                    "errors": [str],
                },
                ...
            },
            "warnings": [str],   # flattened
            "errors": [str],     # flattened
        }
    """
    from rkvoice_stream.platform import get_platform

    host_platform = get_platform(platform_name)

    buckets: dict[Optional[str], list[ResourceProfile]] = {}
    for p in profiles:
        if p is None:
            continue
        buckets.setdefault(p.device_id, []).append(p)

    devices: dict[str, dict] = {}
    all_warnings: list[str] = []
    all_errors: list[str] = []
    compatible = True

    for device_id, dev_profiles in buckets.items():
        key = device_id if device_id is not None else "host"
        limit = _device_memory_limit(device_id, host_platform)
        total_mem = sum(p.npu_memory_mb for p in dev_profiles)
        labels = [p.label or "?" for p in dev_profiles]
        warnings: list[str] = []
        errors: list[str] = []

        if total_mem > limit:
            parts = " + ".join(
                f"{p.label or '?'}={p.npu_memory_mb}MB" for p in dev_profiles
            )
            errors.append(
                f"[{key}] NPU memory exceeds limit: {total_mem}MB > {limit}MB ({parts})"
            )

        # Domain conflict: any RKLLM sharing a domain with another resident model.
        domains: dict[int, list[ResourceProfile]] = {}
        for p in dev_profiles:
            domains.setdefault(p.npu_domain, []).append(p)
        for domain, members in domains.items():
            if len(members) > 1 and any(m.uses_rkllm for m in members):
                warnings.append(
                    f"[{key}] RKLLM and RKNN share NPU domain {domain}. "
                    f"Set base_domain_id=1 for RKLLM to isolate."
                )

        # Core overlap: any explicit core used by more than one resident model.
        core_users: dict[str, int] = {}
        for p in dev_profiles:
            for c in set(p.npu_cores) - {"CORE_AUTO"}:
                core_users[c] = core_users.get(c, 0) + 1
        overlap = {c for c, n in core_users.items() if n > 1}
        if overlap:
            warnings.append(
                f"[{key}] NPU core overlap: {overlap}. Backends must run sequentially."
            )

        rkllm_count = sum(1 for p in dev_profiles if p.uses_rkllm)
        if errors:
            mode = "exclusive"
            compatible = False
        elif len(dev_profiles) <= 1:
            mode = "single"
        elif overlap or rkllm_count > 1:
            mode = "sequential"
        else:
            mode = "parallel"

        devices[key] = {
            "mode": mode,
            "total_memory_mb": total_mem,
            "memory_limit_mb": limit,
            "labels": labels,
            "warnings": warnings,
            "errors": errors,
        }
        all_warnings.extend(warnings)
        all_errors.extend(errors)

    return {
        "compatible": compatible,
        "devices": devices,
        "warnings": all_warnings,
        "errors": all_errors,
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

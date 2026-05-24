#!/usr/bin/env python3
"""Verify RK3576 performance-mode sysfs settings for MOSS profiling.

This is intentionally read-only. It mirrors the intent of Rockchip's
``rknn-llm/scripts/fix_freq_rk3576.sh`` without writing to sysfs: production
profiling should not mix ondemand CPU/NPU/DDR governors with latency claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CPU_POLICIES = ("policy0", "policy4")
DEFAULT_CPUS = tuple(range(8))
DEFAULT_NPU = "27700000.npu"
DEFAULT_GPU = "27800000.gpu"
GOVERNOR_OK = {"userspace", "performance"}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except PermissionError as exc:
        return f"<permission denied: {exc}>"


def _parse_freqs(text: str | None) -> list[int]:
    if not text:
        return []
    values: list[int] = []
    for item in text.split():
        try:
            values.append(int(item))
        except ValueError:
            continue
    return values


def _check_freq_block(
    *,
    label: str,
    governor_path: Path,
    current_path: Path,
    available_path: Path,
    set_path: Path | None = None,
) -> dict[str, Any]:
    governor = _read_text(governor_path)
    current_text = _read_text(current_path)
    available_text = _read_text(available_path)
    available = _parse_freqs(available_text)
    current = int(current_text) if current_text and current_text.isdigit() else None
    max_available = max(available) if available else None
    errors: list[str] = []

    if governor is None:
        errors.append(f"{label}.governor missing: {governor_path}")
    elif governor not in GOVERNOR_OK:
        errors.append(f"{label}.governor={governor!r}, expected one of {sorted(GOVERNOR_OK)}")
    if current is None:
        errors.append(f"{label}.cur_freq missing or invalid: {current_path}")
    if max_available is None:
        errors.append(f"{label}.available_frequencies missing or invalid: {available_path}")
    if current is not None and max_available is not None and current < max_available:
        errors.append(f"{label}.cur_freq={current} below max_available={max_available}")

    remediation: list[str] = []
    if max_available is not None:
        remediation.append(f"echo userspace > {governor_path}")
        if set_path is not None:
            remediation.append(f"echo {max_available} > {set_path}")
    return {
        "label": label,
        "passed": not errors,
        "errors": errors,
        "governor": governor,
        "cur_freq": current,
        "available_frequencies": available,
        "max_available": max_available,
        "paths": {
            "governor": str(governor_path),
            "cur_freq": str(current_path),
            "available_frequencies": str(available_path),
            "set_freq": str(set_path) if set_path else None,
        },
        "remediation": remediation,
    }


def _check_cpuidle(sysfs_root: Path, cpus: tuple[int, ...]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    remediation: list[str] = []
    for cpu in cpus:
        path = sysfs_root / "devices/system/cpu" / f"cpu{cpu}" / "cpuidle/state1/disable"
        value = _read_text(path)
        item = {"cpu": cpu, "path": str(path), "value": value, "passed": value == "1"}
        items.append(item)
        if value is None:
            errors.append(f"cpu{cpu}.cpuidle.state1.disable missing: {path}")
        elif value != "1":
            errors.append(f"cpu{cpu}.cpuidle.state1.disable={value!r}, expected '1'")
            remediation.append(f"echo 1 > {path}")
    return {"passed": not errors, "errors": errors, "items": items, "remediation": remediation}


def verify_performance_mode(
    *,
    sysfs_root: Path = Path("/sys"),
    include_gpu: bool = False,
    check_cpuidle: bool = True,
) -> dict[str, Any]:
    blocks: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for policy in DEFAULT_CPU_POLICIES:
        base = sysfs_root / "devices/system/cpu/cpufreq" / policy
        blocks[f"cpu_{policy}"] = _check_freq_block(
            label=f"cpu.{policy}",
            governor_path=base / "scaling_governor",
            current_path=base / "scaling_cur_freq",
            available_path=base / "scaling_available_frequencies",
            set_path=base / "scaling_setspeed",
        )
    devfreq = sysfs_root / "class/devfreq"
    blocks["npu"] = _check_freq_block(
        label="npu",
        governor_path=devfreq / DEFAULT_NPU / "governor",
        current_path=devfreq / DEFAULT_NPU / "cur_freq",
        available_path=devfreq / DEFAULT_NPU / "available_frequencies",
        set_path=devfreq / DEFAULT_NPU / "userspace/set_freq",
    )
    blocks["ddr"] = _check_freq_block(
        label="ddr",
        governor_path=devfreq / "dmc/governor",
        current_path=devfreq / "dmc/cur_freq",
        available_path=devfreq / "dmc/available_frequencies",
        set_path=devfreq / "dmc/userspace/set_freq",
    )
    if include_gpu:
        blocks["gpu"] = _check_freq_block(
            label="gpu",
            governor_path=devfreq / DEFAULT_GPU / "governor",
            current_path=devfreq / DEFAULT_GPU / "cur_freq",
            available_path=devfreq / DEFAULT_GPU / "available_frequencies",
            set_path=devfreq / DEFAULT_GPU / "userspace/set_freq",
        )

    cpuidle = _check_cpuidle(sysfs_root, DEFAULT_CPUS) if check_cpuidle else None
    for block in blocks.values():
        errors.extend(block["errors"])
    if cpuidle:
        errors.extend(cpuidle["errors"])

    remediation = {
        "note": "Requires root. Review before applying; this verifier never writes sysfs.",
        "commands": [cmd for block in blocks.values() for cmd in block["remediation"]]
        + (cpuidle["remediation"] if cpuidle else []),
        "source_reference": "/Users/harvest/project/rknn-llm/scripts/fix_freq_rk3576.sh",
    }
    return {
        "passed": not errors,
        "errors": errors,
        "sysfs_root": str(sysfs_root),
        "governor_ok": sorted(GOVERNOR_OK),
        "blocks": blocks,
        "cpuidle": cpuidle,
        "remediation": remediation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--include-gpu", action="store_true")
    parser.add_argument("--no-check-cpuidle", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_performance_mode(
        sysfs_root=args.sysfs_root,
        include_gpu=args.include_gpu,
        check_cpuidle=not args.no_check_cpuidle,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

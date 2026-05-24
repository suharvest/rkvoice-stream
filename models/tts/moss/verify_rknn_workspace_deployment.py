#!/usr/bin/env python3
"""Verify the persistent RKNN artifact workspace deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from models.tts.moss.prepare_rknn_workspace import _fstab_entry
from models.tts.moss.verify_rknn_artifact_workspace import DEFAULT_MIN_FREE_MB, verify_workspace


DEFAULT_MOUNT_POINT = Path("/mnt/rknn-workspace")
DEFAULT_WORKSPACE = DEFAULT_MOUNT_POINT / "moss-rknn-workspace"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _parse_mounts(text: str) -> list[dict[str, str]]:
    mounts: list[dict[str, str]] = []
    for raw in text.splitlines():
        parts = raw.split()
        if len(parts) < 3:
            continue
        mounts.append({"source": parts[0], "target": parts[1], "fstype": parts[2]})
    return mounts


def _fstab_has_entry(text: str, entry: str) -> bool:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == entry:
            return True
    return False


def verify_deployment(
    *,
    workspace: Path = DEFAULT_WORKSPACE,
    mount_point: Path = DEFAULT_MOUNT_POINT,
    label: str = "RKNN_WS",
    fstab_file: Path = Path("/etc/fstab"),
    mounts_file: Path = Path("/proc/mounts"),
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_fstab_entry = _fstab_entry(label, mount_point)
    fstab_text = _read_text(fstab_file)
    fstab_entry_present = _fstab_has_entry(fstab_text, expected_fstab_entry)
    if not fstab_entry_present:
        errors.append(f"missing fstab entry: {expected_fstab_entry}")

    mounts = _parse_mounts(_read_text(mounts_file))
    mounted = next((item for item in mounts if item["target"] == str(mount_point)), None)
    if mounted is None:
        errors.append(f"mount point is not mounted: {mount_point}")
    elif mounted["fstype"] != "ext4":
        errors.append(f"mount point fstype={mounted['fstype']!r}, expected 'ext4'")

    workspace_report = verify_workspace(
        workspace=workspace,
        min_free_mb=min_free_mb,
    )
    errors.extend(workspace_report["errors"])

    return {
        "passed": not errors,
        "errors": errors,
        "workspace": str(workspace),
        "mount_point": str(mount_point),
        "label": label,
        "fstab_file": str(fstab_file),
        "fstab_entry": expected_fstab_entry,
        "fstab_entry_present": fstab_entry_present,
        "mounts_file": str(mounts_file),
        "mounted": mounted,
        "workspace_report": workspace_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--mount-point", type=Path, default=DEFAULT_MOUNT_POINT)
    parser.add_argument("--label", default="RKNN_WS")
    parser.add_argument("--fstab-file", type=Path, default=Path("/etc/fstab"))
    parser.add_argument("--mounts-file", type=Path, default=Path("/proc/mounts"))
    parser.add_argument("--min-free-mb", type=int, default=DEFAULT_MIN_FREE_MB)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_deployment(
        workspace=args.workspace,
        mount_point=args.mount_point,
        label=args.label,
        fstab_file=args.fstab_file,
        mounts_file=args.mounts_file,
        min_free_mb=args.min_free_mb,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

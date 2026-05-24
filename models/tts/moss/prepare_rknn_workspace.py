#!/usr/bin/env python3
"""Prepare a non-root RKNN artifact workspace.

Default mode is dry-run. Destructive setup requires an explicit confirmation
string because it may partition and format a block device.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


CONFIRMATION = "FORMAT_RKNN_WORKSPACE"
REQUIRED_COMMANDS = ("parted", "partprobe", "mkfs.ext4", "mkdir", "mount", "chmod")
FSTAB_CONFIRM_MARKER = "# rkvoice-stream rknn workspace"


def _missing_commands(commands: tuple[str, ...] = REQUIRED_COMMANDS) -> list[str]:
    return [command for command in commands if shutil.which(command) is None]


def _run(cmd: list[str], *, execute: bool) -> dict[str, Any]:
    if not execute:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": "", "executed": False}
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "executed": True,
    }


def _read_mounts(path: Path = Path("/proc/mounts")) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _partition_path(device: Path, index: int = 1) -> Path:
    name = device.name
    suffix = f"p{index}" if name.startswith(("mmcblk", "nvme")) else str(index)
    return device.with_name(f"{name}{suffix}")


def _is_block_device(path: Path) -> bool:
    try:
        return stat.S_ISBLK(os.stat(path).st_mode)
    except OSError:
        return False


def _device_guard(device: Path, mount_point: Path, mounts_text: str, *, execute: bool = False) -> list[str]:
    errors: list[str] = []
    if not str(device).startswith("/dev/"):
        errors.append(f"device must be under /dev, got {device}")
    if execute and not _is_block_device(device):
        errors.append(f"device is not an existing block device: {device}")
    if str(device) in mounts_text:
        errors.append(f"device is already mounted: {device}")
    if str(mount_point) in mounts_text:
        errors.append(f"mount point is already mounted: {mount_point}")
    if mount_point == Path("/") or mount_point == Path("/home") or str(mount_point).startswith("/home/cat"):
        errors.append(f"mount point must be a non-home workspace path, got {mount_point}")
    return errors


def _fstab_entry(label: str, mount_point: Path) -> str:
    return f"LABEL={label} {mount_point} ext4 defaults,nofail,noatime 0 2"


def _write_fstab_entry(fstab_file: Path, entry: str) -> dict[str, Any]:
    text = _read_text(fstab_file)
    if entry in text:
        return {
            "cmd": ["write_fstab_entry", str(fstab_file), entry],
            "returncode": 0,
            "stdout": "entry already present",
            "stderr": "",
            "executed": True,
        }
    with fstab_file.open("a", encoding="utf-8") as f:
        if text and not text.endswith("\n"):
            f.write("\n")
        f.write(f"{FSTAB_CONFIRM_MARKER}\n{entry}\n")
    return {
        "cmd": ["write_fstab_entry", str(fstab_file), entry],
        "returncode": 0,
        "stdout": "entry appended",
        "stderr": "",
        "executed": True,
    }


def build_plan(device: Path, mount_point: Path, label: str) -> list[list[str]]:
    partition = _partition_path(device)
    return [
        ["parted", "-s", str(device), "mklabel", "gpt"],
        ["parted", "-s", str(device), "mkpart", "primary", "ext4", "0%", "100%"],
        ["partprobe", str(device)],
        ["mkfs.ext4", "-F", "-L", label, str(partition)],
        ["mkdir", "-p", str(mount_point)],
        ["mount", str(partition), str(mount_point)],
        ["mkdir", "-p", str(mount_point / "moss-rknn-workspace")],
        ["chmod", "0777", str(mount_point / "moss-rknn-workspace")],
    ]


def prepare_workspace(
    *,
    device: Path = Path("/dev/mmcblk1"),
    mount_point: Path = Path("/mnt/rknn-workspace"),
    label: str = "RKNN_WS",
    execute: bool = False,
    confirm: str = "",
    mounts_file: Path = Path("/proc/mounts"),
    fstab_file: Path = Path("/etc/fstab"),
    persist_fstab: bool = True,
) -> dict[str, Any]:
    mounts_text = _read_mounts(mounts_file)
    partition = _partition_path(device)
    errors = _device_guard(device, mount_point, mounts_text, execute=execute)
    missing_commands = _missing_commands()
    for command in missing_commands:
        errors.append(f"required command not found: {command}")
    requires_confirmation = True
    if execute and confirm != CONFIRMATION:
        errors.append(f"--confirm must equal {CONFIRMATION!r} when --execute is used")
    commands = build_plan(device, mount_point, label)
    fstab_entry = _fstab_entry(label, mount_point)
    fstab_already_configured = fstab_entry in _read_text(fstab_file)
    if persist_fstab:
        commands.append(["write_fstab_entry", str(fstab_file), fstab_entry])
    results = []
    if execute and not errors:
        for cmd in commands:
            if cmd and cmd[0] == "write_fstab_entry":
                try:
                    result = _write_fstab_entry(fstab_file, fstab_entry)
                except OSError as exc:
                    result = {
                        "cmd": cmd,
                        "returncode": 1,
                        "stdout": "",
                        "stderr": str(exc),
                        "executed": True,
                    }
            else:
                result = _run(cmd, execute=True)
            results.append(result)
            if result["returncode"] != 0:
                errors.append(f"command failed: {' '.join(cmd)}")
                break
    else:
        results = [_run(cmd, execute=False) for cmd in commands]
    return {
        "passed": not errors,
        "errors": errors,
        "execute": execute,
        "requires_confirmation": requires_confirmation,
        "confirmation": CONFIRMATION,
        "device": str(device),
        "partition": str(partition),
        "mount_point": str(mount_point),
        "label": label,
        "persist_fstab": persist_fstab,
        "fstab_file": str(fstab_file),
        "fstab_entry": fstab_entry,
        "fstab_already_configured": fstab_already_configured,
        "missing_commands": missing_commands,
        "commands": commands,
        "results": results,
        "workspace": str(mount_point / "moss-rknn-workspace"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, default=Path("/dev/mmcblk1"))
    parser.add_argument("--mount-point", type=Path, default=Path("/mnt/rknn-workspace"))
    parser.add_argument("--label", default="RKNN_WS")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--fstab-file", type=Path, default=Path("/etc/fstab"))
    parser.add_argument("--no-persist-fstab", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = prepare_workspace(
        device=args.device,
        mount_point=args.mount_point,
        label=args.label,
        execute=args.execute,
        confirm=args.confirm,
        fstab_file=args.fstab_file,
        persist_fstab=not args.no_persist_fstab,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

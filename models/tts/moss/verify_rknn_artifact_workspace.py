#!/usr/bin/env python3
"""Validate a workspace before writing large RKNN experiment artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MIN_FREE_MB = 2048


def _disk_summary(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_mb": usage.free // (1024 * 1024),
        "used_mb": usage.used // (1024 * 1024),
        "total_mb": usage.total // (1024 * 1024),
    }


def _same_filesystem(left: Path, right: Path) -> bool:
    try:
        return left.stat().st_dev == right.stat().st_dev
    except OSError:
        return False


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def _nearest_existing_parent(path: Path) -> Path | None:
    cursor = path
    while not cursor.exists():
        parent = cursor.parent
        if parent == cursor:
            return None
        cursor = parent
    return cursor


def verify_workspace(
    *,
    workspace: Path,
    root_path: Path = Path("/"),
    home_dir: Path = Path.home(),
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
    allow_root_filesystem: bool = False,
    require_writable: bool = True,
) -> dict[str, Any]:
    workspace = workspace.resolve(strict=False)
    root_path = root_path.resolve(strict=False)
    home_dir = home_dir.resolve(strict=False)
    existing = _nearest_existing_parent(workspace)
    errors: list[str] = []

    if existing is None:
        errors.append(f"no existing parent for workspace: {workspace}")
        disk = None
        same_as_root = None
    else:
        disk = _disk_summary(existing)
        same_as_root = _same_filesystem(existing, root_path)
        if require_writable and not workspace.exists():
            errors.append(f"workspace does not exist: {workspace}")
        elif require_writable and not workspace.is_dir():
            errors.append(f"workspace is not a directory: {workspace}")
        elif require_writable:
            probe = workspace / ".rkvoice_workspace_write_probe"
            try:
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
            except OSError as exc:
                errors.append(f"workspace is not writable: {workspace}: {exc}")
        if _is_under(workspace, home_dir):
            errors.append(f"workspace is under home directory; move RKNN artifacts off root storage: {workspace}")
        if same_as_root and not allow_root_filesystem:
            errors.append(f"workspace is on the root filesystem: {workspace}")
        if int(disk["free_mb"]) < min_free_mb:
            errors.append(f"workspace.free_mb={disk['free_mb']} below required {min_free_mb}")

    return {
        "passed": not errors,
        "errors": errors,
        "workspace": str(workspace),
        "existing_parent": str(existing) if existing is not None else None,
        "root_path": str(root_path),
        "home_dir": str(home_dir),
        "same_filesystem_as_root": same_as_root,
        "allow_root_filesystem": allow_root_filesystem,
        "require_writable": require_writable,
        "min_free_mb": min_free_mb,
        "disk": disk,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--root-path", type=Path, default=Path("/"))
    parser.add_argument("--home-dir", type=Path, default=Path.home())
    parser.add_argument("--min-free-mb", type=int, default=DEFAULT_MIN_FREE_MB)
    parser.add_argument("--allow-root-filesystem", action="store_true")
    parser.add_argument("--no-require-writable", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_workspace(
        workspace=args.workspace,
        root_path=args.root_path,
        home_dir=args.home_dir,
        min_free_mb=args.min_free_mb,
        allow_root_filesystem=args.allow_root_filesystem,
        require_writable=not args.no_require_writable,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

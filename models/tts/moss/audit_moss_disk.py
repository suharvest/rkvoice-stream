#!/usr/bin/env python3
"""Report MOSS-related disk usage on RK devices.

The default mode is read-only. It identifies large temporary MOSS artifacts and
home-directory experiment/model folders so release preflight failures can be
resolved deliberately instead of by ad hoc `du` output. Cleanup is limited to
temporary candidates and requires an explicit confirmation string.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_TMP_PATTERNS = ("moss*", "rkvoice-pycache")
DEFAULT_HOME_PATTERNS = ("moss*", "*rknn*", "*tts*", "*onnx*")
DELETE_CONFIRMATION = "DELETE_MOSS_TMP_CANDIDATES"
PROTECTED_HOME_NAMES = {
    "moss-onnx-baseline",
    "rknn-venv",
    "sherpa-onnx-paraformer-zh-2023-09-14",
}
ARCHIVE_CANDIDATE_MARKERS = (
    "probe",
    "export",
    "bundle",
    "test",
    ".tar",
    ".onnx",
    ".rknn",
)
BLOCK_SIZE_BYTES = 512


def _disk_summary(path: Path) -> dict[str, int | str]:
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


def _mounted_sources(mounts_file: Path = Path("/proc/mounts")) -> set[str]:
    sources: set[str] = set()
    try:
        for line in mounts_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                sources.add(parts[0])
    except OSError:
        pass
    return sources


def _device_has_filesystem(device: Path, blkid_dir: Path | None = None) -> bool:
    if blkid_dir is None:
        return False
    return (blkid_dir / device.name).exists()


def _scan_block_devices(
    *,
    sys_block_dir: Path = Path("/sys/block"),
    dev_dir: Path = Path("/dev"),
    mounts_file: Path = Path("/proc/mounts"),
    blkid_dir: Path | None = None,
    min_size_mb: int = 1024,
) -> list[dict[str, Any]]:
    mounted = _mounted_sources(mounts_file)
    candidates: list[dict[str, Any]] = []
    if not sys_block_dir.exists():
        return candidates
    for entry in sorted(sys_block_dir.iterdir()):
        name = entry.name
        if name.startswith(("loop", "ram", "zram")):
            continue
        size_path = entry / "size"
        try:
            blocks = int(size_path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            continue
        size_mb = blocks * BLOCK_SIZE_BYTES // (1024 * 1024)
        if size_mb < min_size_mb:
            continue
        device = dev_dir / name
        child_partitions = sorted(
            child.name for child in entry.iterdir() if child.name.startswith(name + "p")
        )
        is_mounted = str(device) in mounted or any(str(dev_dir / part) in mounted for part in child_partitions)
        has_fs = _device_has_filesystem(device, blkid_dir)
        if not is_mounted:
            candidates.append(
                {
                    "device": str(device),
                    "size_mb": size_mb,
                    "mounted": False,
                    "has_partitions": bool(child_partitions),
                    "partitions": child_partitions,
                    "filesystem_detected": has_fs,
                    "suggested_action": "prepare_as_non_root_workspace_after_explicit_approval",
                    "requires_destructive_setup": not has_fs and not child_partitions,
                }
            )
    candidates.sort(key=lambda item: int(item["size_mb"]), reverse=True)
    return candidates


def _path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() or item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _collect(root: Path, patterns: tuple[str, ...], min_mb: int) -> list[dict[str, Any]]:
    seen: set[Path] = set()
    items: list[dict[str, Any]] = []
    if not root.exists():
        return items
    for pattern in patterns:
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            size_bytes = _path_size(path)
            size_mb = size_bytes / (1024 * 1024)
            if size_mb < min_mb:
                continue
            items.append(
                {
                    "path": str(path),
                    "size_mb": round(size_mb, 3),
                    "kind": "dir" if path.is_dir() else "file",
                }
            )
    items.sort(key=lambda item: float(item["size_mb"]), reverse=True)
    return items


def _classify_home_candidate(path_text: str) -> dict[str, str]:
    name = Path(path_text).name
    if name in PROTECTED_HOME_NAMES:
        return {
            "classification": "protect",
            "reason": "known production/runtime dependency; do not delete for MOSS cleanup",
            "suggested_action": "keep",
        }
    if any(marker in name for marker in ARCHIVE_CANDIDATE_MARKERS):
        return {
            "classification": "review_archive_or_move",
            "reason": "experiment/export/probe artifact; review, archive, or move off root disk",
            "suggested_action": "archive_or_move_off_root_after_review",
        }
    return {
        "classification": "review",
        "reason": "matches MOSS/RKNN/TTS/ONNX pattern but is not in the protected allowlist",
        "suggested_action": "manual_review",
    }


def _migration_priority(item: dict[str, Any]) -> str:
    if item["classification"] == "protect":
        return "protected"
    size_mb = float(item["size_mb"])
    if item["classification"] == "review_archive_or_move" and size_mb >= 500:
        return "high"
    if item["classification"] == "review_archive_or_move" and size_mb >= 200:
        return "medium"
    if item["classification"] == "review_archive_or_move":
        return "low"
    return "review"


def _classify_home_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified = []
    for item in candidates:
        enriched = dict(item)
        enriched.update(_classify_home_candidate(str(item["path"])))
        enriched["migration_priority"] = _migration_priority(enriched)
        enriched["safe_to_delete_without_review"] = False
        classified.append(enriched)
    return classified


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _delete_tmp_candidates(tmp_dir: Path, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    deleted: list[dict[str, Any]] = []
    errors: list[str] = []
    root = tmp_dir.resolve()
    for item in candidates:
        path = Path(str(item["path"]))
        if not _is_under(path, root):
            errors.append(f"refusing to delete outside tmp_dir: {path}")
            continue
        try:
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            else:
                errors.append(f"candidate disappeared before delete: {path}")
                continue
            deleted.append(item)
        except OSError as exc:
            errors.append(f"failed to delete {path}: {exc}")
    return deleted, errors


def audit_disk(
    *,
    root_path: Path = Path("/"),
    tmp_dir: Path = Path("/tmp"),
    home_dir: Path = Path.home(),
    min_candidate_mb: int = 100,
    delete_tmp_candidates: bool = False,
    confirm_delete: str = "",
    scan_block_devices: bool = True,
    sys_block_dir: Path = Path("/sys/block"),
    dev_dir: Path = Path("/dev"),
    mounts_file: Path = Path("/proc/mounts"),
    blkid_dir: Path | None = None,
) -> dict[str, Any]:
    tmp_candidates = _collect(tmp_dir, DEFAULT_TMP_PATTERNS, min_candidate_mb)
    home_candidates = _classify_home_candidates(_collect(home_dir, DEFAULT_HOME_PATTERNS, min_candidate_mb))
    deleted_candidates: list[dict[str, Any]] = []
    delete_errors: list[str] = []
    if delete_tmp_candidates:
        if confirm_delete != DELETE_CONFIRMATION:
            delete_errors.append(
                f"delete requested but --confirm-delete must equal {DELETE_CONFIRMATION!r}"
            )
        else:
            deleted_candidates, delete_errors = _delete_tmp_candidates(tmp_dir, tmp_candidates)
    disk = _disk_summary(root_path)
    tmp_disk = _disk_summary(tmp_dir)
    home_disk = _disk_summary(home_dir)
    block_candidates = (
        _scan_block_devices(
            sys_block_dir=sys_block_dir,
            dev_dir=dev_dir,
            mounts_file=mounts_file,
            blkid_dir=blkid_dir,
        )
        if scan_block_devices
        else []
    )
    return {
        "disk": disk,
        "tmp_disk": tmp_disk,
        "home_disk": home_disk,
        "tmp_candidates_affect_root_disk": _same_filesystem(tmp_dir, root_path),
        "home_candidates_affect_root_disk": _same_filesystem(home_dir, root_path),
        "unmounted_block_device_candidates": block_candidates,
        "min_candidate_mb": min_candidate_mb,
        "tmp_candidates": tmp_candidates,
        "home_candidates": home_candidates,
        "home_protected_mb": round(
            sum(float(item["size_mb"]) for item in home_candidates if item["classification"] == "protect"),
            3,
        ),
        "home_review_archive_or_move_mb": round(
            sum(
                float(item["size_mb"])
                for item in home_candidates
                if item["classification"] == "review_archive_or_move"
            ),
            3,
        ),
        "home_migration_plan": [
            item
            for item in home_candidates
            if item["classification"] == "review_archive_or_move"
        ],
        "candidate_total_mb": round(
            sum(float(item["size_mb"]) for item in tmp_candidates + home_candidates),
            3,
        ),
        "delete_performed": bool(deleted_candidates),
        "deleted_candidates": deleted_candidates,
        "delete_errors": delete_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", type=Path, default=Path("/"))
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--home-dir", type=Path, default=Path.home())
    parser.add_argument("--min-candidate-mb", type=int, default=100)
    parser.add_argument(
        "--delete-tmp-candidates",
        action="store_true",
        help="delete only the reported tmp candidates; never deletes home candidates",
    )
    parser.add_argument(
        "--confirm-delete",
        default="",
        help=f"must equal {DELETE_CONFIRMATION!r} when --delete-tmp-candidates is used",
    )
    parser.add_argument("--no-scan-block-devices", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = audit_disk(
        root_path=args.root_path,
        tmp_dir=args.tmp_dir,
        home_dir=args.home_dir,
        min_candidate_mb=args.min_candidate_mb,
        delete_tmp_candidates=args.delete_tmp_candidates,
        confirm_delete=args.confirm_delete,
        scan_block_devices=not args.no_scan_block_devices,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

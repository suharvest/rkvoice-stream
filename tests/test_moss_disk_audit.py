"""Unit checks for the MOSS disk usage audit helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_disk_module():
    path = ROOT / "models" / "tts" / "moss" / "audit_moss_disk.py"
    spec = importlib.util.spec_from_file_location("audit_moss_disk", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_moss_disk_audit_reports_large_candidates(monkeypatch, tmp_path):
    module = _load_disk_module()
    tmp_dir = tmp_path / "tmp"
    home_dir = tmp_path / "home"
    tmp_dir.mkdir()
    home_dir.mkdir()
    (tmp_dir / "moss-edge-slices").mkdir()
    (tmp_dir / "moss-edge-slices" / "slice.bin").write_bytes(b"x" * 2048)
    (tmp_dir / "unrelated").mkdir()
    (tmp_dir / "unrelated" / "data.bin").write_bytes(b"x" * 4096)
    (home_dir / "moss-rknn-probe").mkdir()
    (home_dir / "moss-rknn-probe" / "model.bin").write_bytes(b"x" * 3072)

    class _Usage:
        total = 1024 * 1024
        used = 512 * 1024
        free = 512 * 1024

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: _Usage())

    report = module.audit_disk(root_path=tmp_path, tmp_dir=tmp_dir, home_dir=home_dir, min_candidate_mb=0)

    tmp_paths = {item["path"] for item in report["tmp_candidates"]}
    home_paths = {item["path"] for item in report["home_candidates"]}
    assert str(tmp_dir / "moss-edge-slices") in tmp_paths
    assert str(tmp_dir / "unrelated") not in tmp_paths
    assert str(home_dir / "moss-rknn-probe") in home_paths
    probe = next(item for item in report["home_candidates"] if item["path"] == str(home_dir / "moss-rknn-probe"))
    assert probe["classification"] == "review_archive_or_move"
    assert probe["suggested_action"] == "archive_or_move_off_root_after_review"
    assert probe["migration_priority"] == "low"
    assert probe["safe_to_delete_without_review"] is False
    assert report["delete_performed"] is False
    assert report["candidate_total_mb"] > 0
    assert report["disk"]["path"] == str(tmp_path)
    assert report["tmp_disk"]["path"] == str(tmp_dir)
    assert report["home_disk"]["path"] == str(home_dir)
    assert report["tmp_candidates_affect_root_disk"] is True
    assert report["home_candidates_affect_root_disk"] is True


def test_moss_disk_audit_classifies_protected_home_candidates(tmp_path):
    module = _load_disk_module()
    tmp_dir = tmp_path / "tmp"
    home_dir = tmp_path / "home"
    tmp_dir.mkdir()
    home_dir.mkdir()
    protected = home_dir / "moss-onnx-baseline"
    protected.mkdir()
    (protected / "model.bin").write_bytes(b"x" * 2048)
    candidate = home_dir / "moss-rknn-probe"
    candidate.mkdir()
    (candidate / "model.bin").write_bytes(b"x" * 2048)

    report = module.audit_disk(root_path=tmp_path, tmp_dir=tmp_dir, home_dir=home_dir, min_candidate_mb=0)
    by_path = {item["path"]: item for item in report["home_candidates"]}

    assert by_path[str(protected)]["classification"] == "protect"
    assert by_path[str(protected)]["suggested_action"] == "keep"
    assert by_path[str(protected)]["migration_priority"] == "protected"
    assert by_path[str(candidate)]["classification"] == "review_archive_or_move"
    assert report["home_migration_plan"][0]["path"] == str(candidate)
    assert report["home_protected_mb"] > 0
    assert report["home_review_archive_or_move_mb"] > 0


def test_moss_disk_audit_prioritizes_large_home_migration_candidates(tmp_path):
    module = _load_disk_module()
    tmp_dir = tmp_path / "tmp"
    home_dir = tmp_path / "home"
    tmp_dir.mkdir()
    home_dir.mkdir()
    high = home_dir / "moss-official-bundle"
    high.mkdir()
    (high / "data.bin").write_bytes(b"x")
    with (high / "data.bin").open("r+b") as f:
        f.truncate(600 * 1024 * 1024)
    medium = home_dir / "encoder.rknn"
    medium.write_bytes(b"x")
    with medium.open("r+b") as f:
        f.truncate(250 * 1024 * 1024)

    report = module.audit_disk(root_path=tmp_path, tmp_dir=tmp_dir, home_dir=home_dir, min_candidate_mb=0)
    by_path = {item["path"]: item for item in report["home_candidates"]}

    assert by_path[str(high)]["migration_priority"] == "high"
    assert by_path[str(medium)]["migration_priority"] == "medium"


def test_moss_disk_audit_reports_unmounted_block_device_candidates(tmp_path):
    module = _load_disk_module()
    sys_block = tmp_path / "sys" / "block"
    dev = tmp_path / "dev"
    sys_block.mkdir(parents=True)
    dev.mkdir()
    root_dev = sys_block / "mmcblk0"
    root_dev.mkdir()
    (root_dev / "size").write_text(str(58 * 1024 * 1024 * 1024 // module.BLOCK_SIZE_BYTES), encoding="utf-8")
    (root_dev / "mmcblk0p3").mkdir()
    extra = sys_block / "mmcblk1"
    extra.mkdir()
    (extra / "size").write_text(str(115 * 1024 * 1024 * 1024 // module.BLOCK_SIZE_BYTES), encoding="utf-8")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"{dev / 'mmcblk0p3'} / ext4 rw 0 0\n", encoding="utf-8")

    report = module.audit_disk(
        root_path=tmp_path,
        tmp_dir=tmp_path,
        home_dir=tmp_path,
        min_candidate_mb=0,
        sys_block_dir=sys_block,
        dev_dir=dev,
        mounts_file=mounts,
    )

    candidates = report["unmounted_block_device_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["device"] == str(dev / "mmcblk1")
    assert candidates[0]["size_mb"] >= 115 * 1024 - 1
    assert candidates[0]["requires_destructive_setup"] is True


def test_moss_disk_audit_default_mode_is_read_only():
    module = _load_disk_module()

    assert module.audit_disk()["delete_performed"] is False


def test_moss_disk_audit_delete_requires_confirmation(tmp_path):
    module = _load_disk_module()
    tmp_dir = tmp_path / "tmp"
    home_dir = tmp_path / "home"
    tmp_dir.mkdir()
    home_dir.mkdir()
    candidate = tmp_dir / "moss-edge-slices"
    candidate.mkdir()
    (candidate / "slice.bin").write_bytes(b"x" * 2048)

    report = module.audit_disk(
        root_path=tmp_path,
        tmp_dir=tmp_dir,
        home_dir=home_dir,
        min_candidate_mb=0,
        delete_tmp_candidates=True,
    )

    assert candidate.exists()
    assert report["delete_performed"] is False
    assert "confirm-delete" in report["delete_errors"][0]


def test_moss_disk_audit_delete_only_tmp_candidates(tmp_path):
    module = _load_disk_module()
    tmp_dir = tmp_path / "tmp"
    home_dir = tmp_path / "home"
    tmp_dir.mkdir()
    home_dir.mkdir()
    tmp_candidate = tmp_dir / "moss-edge-slices"
    tmp_candidate.mkdir()
    (tmp_candidate / "slice.bin").write_bytes(b"x" * 2048)
    home_candidate = home_dir / "moss-rknn-probe"
    home_candidate.mkdir()
    (home_candidate / "model.bin").write_bytes(b"x" * 2048)

    report = module.audit_disk(
        root_path=tmp_path,
        tmp_dir=tmp_dir,
        home_dir=home_dir,
        min_candidate_mb=0,
        delete_tmp_candidates=True,
        confirm_delete=module.DELETE_CONFIRMATION,
    )

    assert not tmp_candidate.exists()
    assert home_candidate.exists()
    assert report["delete_performed"] is True
    assert report["delete_errors"] == []
    assert report["deleted_candidates"][0]["path"] == str(tmp_candidate)

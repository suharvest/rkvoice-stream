"""Unit checks for persistent RKNN workspace deployment verification."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "verify_rknn_workspace_deployment.py"
    spec = importlib.util.spec_from_file_location("verify_rknn_workspace_deployment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rknn_workspace_deployment_rejects_missing_mount_and_fstab(tmp_path):
    module = _load_module()
    workspace = tmp_path / "mnt" / "rknn-workspace" / "moss-rknn-workspace"
    fstab = tmp_path / "fstab"
    mounts = tmp_path / "mounts"
    fstab.write_text("", encoding="utf-8")
    mounts.write_text("", encoding="utf-8")

    report = module.verify_deployment(
        workspace=workspace,
        mount_point=tmp_path / "mnt" / "rknn-workspace",
        fstab_file=fstab,
        mounts_file=mounts,
        min_free_mb=0,
    )

    assert report["passed"] is False
    assert report["fstab_entry_present"] is False
    assert report["mounted"] is None
    assert any("missing fstab entry" in error for error in report["errors"])
    assert any("mount point is not mounted" in error for error in report["errors"])
    assert any("workspace does not exist" in error for error in report["errors"])


def test_rknn_workspace_deployment_accepts_prepared_mount(monkeypatch, tmp_path):
    module = _load_module()
    mount_point = tmp_path / "mnt" / "rknn-workspace"
    workspace = mount_point / "moss-rknn-workspace"
    workspace.mkdir(parents=True)
    fstab = tmp_path / "fstab"
    mounts = tmp_path / "mounts"
    entry = module._fstab_entry("RKNN_WS", mount_point)
    fstab.write_text(entry + "\n", encoding="utf-8")
    mounts.write_text(f"/dev/mmcblk1p1 {mount_point} ext4 rw,noatime 0 0\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "verify_workspace",
        lambda **kwargs: {
            "passed": True,
            "errors": [],
            "workspace": str(kwargs["workspace"]),
            "same_filesystem_as_root": False,
            "disk": {"free_mb": 100000},
        },
    )

    report = module.verify_deployment(
        workspace=workspace,
        mount_point=mount_point,
        fstab_file=fstab,
        mounts_file=mounts,
        min_free_mb=2048,
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["fstab_entry_present"] is True
    assert report["mounted"] == {"source": "/dev/mmcblk1p1", "target": str(mount_point), "fstype": "ext4"}


def test_rknn_workspace_deployment_rejects_wrong_fstype(monkeypatch, tmp_path):
    module = _load_module()
    mount_point = tmp_path / "mnt" / "rknn-workspace"
    workspace = mount_point / "moss-rknn-workspace"
    workspace.mkdir(parents=True)
    fstab = tmp_path / "fstab"
    mounts = tmp_path / "mounts"
    fstab.write_text(module._fstab_entry("RKNN_WS", mount_point) + "\n", encoding="utf-8")
    mounts.write_text(f"tmpfs {mount_point} tmpfs rw 0 0\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "verify_workspace",
        lambda **kwargs: {"passed": True, "errors": [], "workspace": str(kwargs["workspace"])},
    )

    report = module.verify_deployment(
        workspace=workspace,
        mount_point=mount_point,
        fstab_file=fstab,
        mounts_file=mounts,
        min_free_mb=0,
    )

    assert report["passed"] is False
    assert any("fstype='tmpfs'" in error for error in report["errors"])

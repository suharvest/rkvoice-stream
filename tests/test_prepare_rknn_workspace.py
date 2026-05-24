"""Unit checks for RKNN workspace preparation safeguards."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "prepare_rknn_workspace.py"
    spec = importlib.util.spec_from_file_location("prepare_rknn_workspace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_rknn_workspace_defaults_to_dry_run(monkeypatch, tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    fstab = tmp_path / "fstab"
    mounts.write_text("", encoding="utf-8")
    fstab.write_text("", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=False,
        mounts_file=mounts,
        fstab_file=fstab,
    )

    assert report["passed"] is True
    assert report["execute"] is False
    assert report["requires_confirmation"] is True
    assert report["missing_commands"] == []
    assert report["workspace"] == "/mnt/rknn-workspace/moss-rknn-workspace"
    assert report["partition"] == "/dev/mmcblk1p1"
    assert report["persist_fstab"] is True
    assert report["fstab_file"] == str(fstab)
    assert report["fstab_entry"] == "LABEL=RKNN_WS /mnt/rknn-workspace ext4 defaults,nofail,noatime 0 2"
    assert ["write_fstab_entry", str(fstab), report["fstab_entry"]] in report["commands"]
    assert all(result["executed"] is False for result in report["results"])
    assert report["commands"][0] == ["parted", "-s", "/dev/mmcblk1", "mklabel", "gpt"]
    assert ["mkfs.ext4", "-F", "-L", "RKNN_WS", "/dev/mmcblk1p1"] in report["commands"]


def test_prepare_rknn_workspace_builds_partition_paths_by_device_family():
    module = _load_module()

    assert module._partition_path(Path("/dev/mmcblk1")) == Path("/dev/mmcblk1p1")
    assert module._partition_path(Path("/dev/nvme0n1")) == Path("/dev/nvme0n1p1")
    assert module._partition_path(Path("/dev/sda")) == Path("/dev/sda1")


def test_prepare_rknn_workspace_execute_requires_confirmation(tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    mounts.write_text("", encoding="utf-8")

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=True,
        confirm="",
        mounts_file=mounts,
    )

    assert report["passed"] is False
    assert any("confirm" in error for error in report["errors"])
    assert all(result["executed"] is False for result in report["results"])


def test_prepare_rknn_workspace_execute_requires_existing_block_device(monkeypatch, tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    mounts.write_text("", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(module, "_is_block_device", lambda path: False)

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk9"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=True,
        confirm=module.CONFIRMATION,
        mounts_file=mounts,
    )

    assert report["passed"] is False
    assert any("not an existing block device" in error for error in report["errors"])
    assert all(result["executed"] is False for result in report["results"])


def test_prepare_rknn_workspace_can_append_persistent_fstab_entry(monkeypatch, tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    fstab = tmp_path / "fstab"
    mounts.write_text("", encoding="utf-8")
    fstab.write_text("# base\n", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(module, "_is_block_device", lambda path: True)

    def _fake_run(cmd, *, execute):
        return {"cmd": cmd, "returncode": 0, "stdout": "", "stderr": "", "executed": execute}

    monkeypatch.setattr(module, "_run", _fake_run)

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=True,
        confirm=module.CONFIRMATION,
        mounts_file=mounts,
        fstab_file=fstab,
    )

    assert report["passed"] is True
    assert report["fstab_already_configured"] is False
    assert report["results"][-1]["cmd"][0] == "write_fstab_entry"
    assert report["results"][-1]["stdout"] == "entry appended"
    text = fstab.read_text(encoding="utf-8")
    assert module.FSTAB_CONFIRM_MARKER in text
    assert report["fstab_entry"] in text


def test_prepare_rknn_workspace_skips_duplicate_fstab_entry(monkeypatch, tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    fstab = tmp_path / "fstab"
    entry = "LABEL=RKNN_WS /mnt/rknn-workspace ext4 defaults,nofail,noatime 0 2"
    mounts.write_text("", encoding="utf-8")
    fstab.write_text(entry + "\n", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda command: f"/usr/bin/{command}")
    monkeypatch.setattr(module, "_is_block_device", lambda path: True)
    monkeypatch.setattr(
        module,
        "_run",
        lambda cmd, *, execute: {"cmd": cmd, "returncode": 0, "stdout": "", "stderr": "", "executed": execute},
    )

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=True,
        confirm=module.CONFIRMATION,
        mounts_file=mounts,
        fstab_file=fstab,
    )

    assert report["passed"] is True
    assert report["fstab_already_configured"] is True
    assert report["results"][-1]["stdout"] == "entry already present"
    assert fstab.read_text(encoding="utf-8").count(entry) == 1


def test_prepare_rknn_workspace_reports_missing_commands(monkeypatch, tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    mounts.write_text("", encoding="utf-8")
    monkeypatch.setattr(module.shutil, "which", lambda command: None if command == "parted" else f"/usr/bin/{command}")

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=False,
        mounts_file=mounts,
    )

    assert report["passed"] is False
    assert report["missing_commands"] == ["parted"]
    assert "required command not found: parted" in report["errors"]


def test_prepare_rknn_workspace_rejects_home_mount_point(tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    mounts.write_text("", encoding="utf-8")

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/home/cat/rknn-workspace"),
        execute=False,
        mounts_file=mounts,
    )

    assert report["passed"] is False
    assert any("non-home workspace" in error for error in report["errors"])


def test_prepare_rknn_workspace_rejects_mounted_device(tmp_path):
    module = _load_module()
    mounts = tmp_path / "mounts"
    mounts.write_text("/dev/mmcblk1 /mnt/rknn-workspace ext4 rw 0 0\n", encoding="utf-8")

    report = module.prepare_workspace(
        device=Path("/dev/mmcblk1"),
        mount_point=Path("/mnt/rknn-workspace"),
        execute=False,
        mounts_file=mounts,
    )

    assert report["passed"] is False
    assert any("already mounted" in error for error in report["errors"])

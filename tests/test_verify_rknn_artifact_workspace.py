"""Unit checks for RKNN artifact workspace preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "verify_rknn_artifact_workspace.py"
    spec = importlib.util.spec_from_file_location("verify_rknn_artifact_workspace", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rknn_artifact_workspace_rejects_home_path(tmp_path):
    module = _load_module()
    home = tmp_path / "home" / "cat"
    workspace = home / "moss-rknn-probe"
    workspace.mkdir(parents=True)

    report = module.verify_workspace(
        workspace=workspace,
        root_path=tmp_path,
        home_dir=home,
        min_free_mb=0,
    )

    assert report["passed"] is False
    assert report["same_filesystem_as_root"] is True
    assert any("under home directory" in error for error in report["errors"])
    assert any("root filesystem" in error for error in report["errors"])


def test_rknn_artifact_workspace_can_allow_root_filesystem_for_small_local_tests(tmp_path):
    module = _load_module()
    workspace = tmp_path / "work"
    workspace.mkdir()

    report = module.verify_workspace(
        workspace=workspace,
        root_path=tmp_path,
        home_dir=tmp_path / "home",
        min_free_mb=0,
        allow_root_filesystem=True,
    )

    assert report["passed"] is True
    assert report["errors"] == []


def test_rknn_artifact_workspace_requires_existing_directory(tmp_path):
    module = _load_module()

    report = module.verify_workspace(
        workspace=tmp_path / "missing",
        root_path=tmp_path,
        home_dir=tmp_path / "home",
        min_free_mb=0,
        allow_root_filesystem=True,
    )

    assert report["passed"] is False
    assert any("workspace does not exist" in error for error in report["errors"])

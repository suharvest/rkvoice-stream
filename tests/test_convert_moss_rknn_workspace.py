"""Unit checks for MOSS RKNN conversion workspace safeguards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "convert_moss_rknn.py"
    spec = importlib.util.spec_from_file_location("convert_moss_rknn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_moss_rknn_workspace_check_rejects_out_dir_outside_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = tmp_path / "home" / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": True, "errors": [], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=0)

    assert report["passed"] is False
    assert any("out_dir must be under RKNN workspace" in error for error in report["errors"])


def test_convert_moss_rknn_workspace_check_rejects_unprepared_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = workspace / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": False, "errors": ["workspace does not exist"], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=0)

    assert report["passed"] is False
    assert "workspace does not exist" in report["errors"]


def test_convert_moss_rknn_workspace_check_accepts_prepared_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = workspace / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": True, "errors": [], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=2048)

    assert report["passed"] is True
    assert report["errors"] == []


def test_convert_moss_rknn_blocks_before_missing_onnx_check_when_workspace_preflight_fails(
    monkeypatch, tmp_path
):
    module = _load_module()
    calls = []

    def _fake_check(**kwargs):
        calls.append(kwargs)
        return {"passed": False, "errors": ["workspace does not exist"]}

    out_dir = tmp_path / "workspace" / "moss-rknn"
    monkeypatch.setattr(module, "check_output_workspace", _fake_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_moss_rknn.py",
            "--onnx-bundle",
            str(tmp_path / "missing-bundle"),
            "--out-dir",
            str(out_dir),
            "--only",
            "sampler",
            "--require-rknn-workspace",
            "--rknn-workspace",
            str(tmp_path / "workspace"),
            "--rknn-workspace-min-free-mb",
            "0",
        ],
    )

    try:
        module.main()
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("workspace preflight failure must stop RKNN conversion")

    assert "RKNN workspace preflight failed" in error
    assert "workspace does not exist" in error
    assert calls[0]["out_dir"] == out_dir
    assert calls[0]["workspace"] == tmp_path / "workspace"
    assert not out_dir.exists()

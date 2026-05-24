"""Unit checks for the MOSS production service runner."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = ROOT / "models" / "tts" / "moss" / "run_moss_production_server.py"
    spec = importlib.util.spec_from_file_location("run_moss_production_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_runner_builds_uvicorn_command():
    runner = _load_runner()

    command = runner.build_command(python="/venv/bin/python", host="127.0.0.1", port=8624)

    assert command == [
        "/venv/bin/python",
        "-m",
        "uvicorn",
        "rkvoice_stream.app.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8624",
    ]


def test_runner_sets_config_env_without_dropping_existing_values():
    runner = _load_runner()

    env = runner.production_env(Path("configs/rk3576-moss-ort-stream.yaml"), {"PATH": "/bin"})

    assert env["CONFIG"] == "configs/rk3576-moss-ort-stream.yaml"
    assert env["PATH"] == "/bin"


def test_runner_dry_run_writes_audited_start_command(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"

    def _no_exec(*args, **kwargs):
        raise AssertionError("dry-run must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--dry-run",
            "--host",
            "127.0.0.1",
            "--port",
            "8624",
            "--python",
            "/venv/bin/python",
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["passed"] is True
    assert report["production_default"] == "moss_ort"
    assert report["service"]["dry_run"] is True
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}
    assert report["service"]["command"] == [
        "/venv/bin/python",
        "-m",
        "uvicorn",
        "rkvoice_stream.app.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8624",
    ]


def test_runner_blocks_start_when_preflight_fails(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"

    class _Usage:
        total = 1024 * 1024 * 1024
        used = 900 * 1024 * 1024
        free = 124 * 1024 * 1024

    def _no_exec(*args, **kwargs):
        raise AssertionError("failed preflight must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(runner.audit.__globals__["shutil"], "disk_usage", lambda path: _Usage())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--min-root-free-mb",
            "256",
            "--disk-path",
            str(tmp_path),
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["passed"] is False
    assert any(error.startswith("disk.free_mb=124 below required 256") for error in report["errors"])


def test_runner_forwards_entrypoint_evidence_to_release_audit(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"
    bad_entrypoint = tmp_path / "bad-entrypoint.json"
    bad_entrypoint.write_text(
        json.dumps(
            {
                "gates": {"passed": True, "errors": []},
                "entrypoint": {
                    "runner": "/tmp/not-the-production-runner.py",
                    "config": "configs/rk3576-moss-ort-stream.yaml",
                },
                "health": {
                    "tts_info": {
                        "backend": "moss_ort",
                        "model_dir": "/opt/tts/models/moss-tts-nano-onnx",
                        "manifest": {"validated": True, "required_artifacts": 11},
                        "profile": {
                            "codec_streaming": True,
                            "codec_full_loaded": False,
                            "codec_batch_frames": 3,
                            "codec_async": False,
                            "cache_voice_prefix": False,
                        },
                        "hybrid": {"enabled": False},
                    }
                },
                "health_after": {
                    "tts_info": {
                        "streaming_stats": {
                            "requests": 2,
                            "completed": 2,
                            "errors": 0,
                            "active": 0,
                            "chunks": 16,
                        }
                    }
                },
                "tts_stream": {"first_payload_ms": 1000},
                "dialogue": {"first_payload_ms": 1000, "max_payload_gap_ms": 500, "binary_chunks": 8},
            }
        ),
        encoding="utf-8",
    )

    def _no_exec(*args, **kwargs):
        raise AssertionError("failed entrypoint evidence must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--entrypoint-evidence",
            str(bad_entrypoint),
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["checks"]["production_entrypoint"]["passed"] is False
    assert any("production entrypoint runner=" in error for error in report["errors"])


def test_runner_forwards_required_rknn_workspace_to_release_audit(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"
    workspace = tmp_path / "missing-rknn-workspace"

    def _no_exec(*args, **kwargs):
        raise AssertionError("failed RKNN workspace preflight must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--require-rknn-workspace",
            "--rknn-workspace",
            str(workspace),
            "--rknn-workspace-min-free-mb",
            "0",
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["checks"]["rknn_workspace"]["passed"] is False
    assert any("workspace does not exist" in error for error in report["errors"])


def test_runner_forwards_required_rknn_workspace_deployment_to_release_audit(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"
    workspace = tmp_path / "missing-rknn-workspace"
    mount_point = tmp_path / "mnt-rknn-workspace"

    def _no_exec(*args, **kwargs):
        raise AssertionError("failed RKNN workspace deployment preflight must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--require-rknn-workspace-deployment",
            "--rknn-workspace",
            str(workspace),
            "--rknn-workspace-mount-point",
            str(mount_point),
            "--rknn-workspace-min-free-mb",
            "0",
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["checks"]["rknn_workspace_deployment"]["passed"] is False
    assert any("mount point is not mounted" in error for error in report["errors"])


def test_runner_validate_artifacts_rejects_missing_model_dir(monkeypatch, tmp_path):
    runner = _load_runner()
    report_path = tmp_path / "runner.json"

    config = tmp_path / "rk3576-moss-ort-stream.yaml"
    config.write_text(
        """
asr:
  backend: null
tts:
  backend: moss_ort
  model_dir: /missing/moss
  manifest: moss-ort-manifest.json
  sample_rate: 48000
  channels: 2
  threads: 6
  prefill_threads: 8
  decode_threads: 5
  codec_threads: 5
  prefill_seq: 0
  max_new_frames: 20
  codec_streaming: 1
  codec_batch_frames: 3
  cache_voice_prefix: 0
  warmup_text: "你好"
  voice: Junhao
  seed: 314
  allow_deterministic_fallback: 0
""",
        encoding="utf-8",
    )

    def _no_exec(*args, **kwargs):
        raise AssertionError("failed artifact preflight must not exec uvicorn")

    monkeypatch.setattr(os, "execvpe", _no_exec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_moss_production_server.py",
            "--dry-run",
            "--validate-artifacts",
            "--config",
            str(config),
            "--json-out",
            str(report_path),
        ],
    )

    rc = runner.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 2
    assert report["passed"] is False
    assert report["checks"]["ort_artifacts"]["passed"] is False
    assert any("tts.model_dir does not exist: /missing/moss" in error for error in report["errors"])
    remediation = report["checks"]["ort_artifacts"]["remediation"]
    assert "models/tts/moss/prepare_moss_ort_deployment.py" in remediation["dry_run"]
    assert "--execute" in remediation["execute_requires_confirmation"]
    assert "INSTALL_MOSS_ORT_DEPLOYMENT" in remediation["execute_requires_confirmation"]

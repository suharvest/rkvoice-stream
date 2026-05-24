"""Unit checks for the MOSS release audit gate."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    path = ROOT / "models" / "tts" / "moss" / "audit_moss_release.py"
    spec = importlib.util.spec_from_file_location("audit_moss_release", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_release_audit_passes_current_checked_in_contract():
    audit_module = _load_audit_module()

    report = audit_module.audit()

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["production_default"] == "moss_ort"
    assert report["rknn_status"] == "production_candidate_requires_manifest_evidence"
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_evidence"]["passed"] is True
    assert report["checks"]["roundtrip_evidence"]["passed"] is True
    assert report["checks"]["production_entrypoint"]["passed"] is True
    assert report["checks"]["rknn_candidate_config"]["passed"] is True


def test_release_audit_rejects_experimental_rknn_candidate_config(tmp_path):
    audit_module = _load_audit_module()
    bad_rknn = tmp_path / "rk3576-moss-rknn-stream.yaml"
    bad_rknn.write_text(
        """
asr:
  backend: disabled
tts:
  backend: moss_rknn
  model_dir: /opt/tts/models/moss-tts-nano-rknn
  worker_bin: /opt/rkvoice-workers/moss_rknn_worker
  manifest: moss-rknn-manifest.json
  sample_rate: 48000
  channels: 2
  max_seq_len: 1024
  chunk_frames: 4
  require_production_default: 0
""",
        encoding="utf-8",
    )

    report = audit_module.audit(rknn_config=bad_rknn)

    assert report["passed"] is False
    assert any("RKNN config tts.require_production_default=0" in error for error in report["errors"])


def test_release_audit_rejects_stale_or_slow_evidence(tmp_path):
    audit_module = _load_audit_module()
    source = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-ort-production-current-rerun.json"
    evidence = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    evidence["service_streaming"]["tts_stream"]["first_payload_ms"] = 2000
    bad_evidence = tmp_path / "bad-evidence.json"
    bad_evidence.write_text(json.dumps(evidence), encoding="utf-8")

    report = audit_module.audit(evidence=bad_evidence)

    assert report["passed"] is False
    assert any("service.tts_stream.first_payload_ms=2000 exceeds 1500" in error for error in report["errors"])


def test_release_audit_rejects_bad_production_entrypoint_evidence(tmp_path):
    audit_module = _load_audit_module()
    source = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-production-entrypoint-profile.json"
    evidence = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    evidence["health_after"]["tts_info"]["streaming_stats"]["errors"] = 1
    evidence["entrypoint"]["runner"] = "/tmp/not-the-production-runner.py"
    bad_evidence = tmp_path / "bad-entrypoint-evidence.json"
    bad_evidence.write_text(json.dumps(evidence), encoding="utf-8")

    report = audit_module.audit(entrypoint_evidence=bad_evidence)

    assert report["passed"] is False
    assert report["checks"]["production_entrypoint"]["passed"] is False
    assert any("production entrypoint streaming_stats.errors=1" in error for error in report["errors"])
    assert any("production entrypoint runner=" in error for error in report["errors"])


def test_release_audit_can_check_root_disk_space(monkeypatch, tmp_path):
    audit_module = _load_audit_module()

    class _Usage:
        total = 1024 * 1024 * 1024
        used = 900 * 1024 * 1024
        free = 124 * 1024 * 1024

    monkeypatch.setattr(
        audit_module.shutil,
        "disk_usage",
        lambda path: _Usage(),
    )

    report = audit_module.audit(min_root_free_mb=256, disk_path=tmp_path)

    assert report["passed"] is False
    assert report["checks"]["root_disk"]["free_mb"] == 124
    assert any("disk.free_mb=124 below required 256" in error for error in report["errors"])


def test_release_audit_can_validate_ort_artifacts_and_reject_missing_model_dir(tmp_path):
    audit_module = _load_audit_module()
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

    report = audit_module.audit(ort_config=config, validate_ort_artifacts=True)

    assert report["passed"] is False
    assert report["checks"]["ort_artifacts"]["passed"] is False
    assert any("tts.model_dir does not exist: /missing/moss" in error for error in report["errors"])
    remediation = report["checks"]["ort_artifacts"]["remediation"]
    assert "models/tts/moss/prepare_moss_ort_deployment.py" in remediation["dry_run"]
    assert "--execute" in remediation["execute_requires_confirmation"]
    assert "INSTALL_MOSS_ORT_DEPLOYMENT" in remediation["execute_requires_confirmation"]


def test_release_audit_can_require_rknn_workspace(tmp_path):
    audit_module = _load_audit_module()
    workspace = tmp_path / "moss-rknn-workspace"

    report = audit_module.audit(require_rknn_workspace=True, rknn_workspace=workspace, rknn_workspace_min_free_mb=0)

    assert report["passed"] is False
    assert report["checks"]["rknn_workspace"]["passed"] is False
    assert any("workspace does not exist" in error for error in report["errors"])


def test_release_audit_accepts_prepared_non_root_rknn_workspace(tmp_path):
    audit_module = _load_audit_module()
    workspace = tmp_path / "external" / "moss-rknn-workspace"

    def _pass_workspace(**kwargs):
        assert kwargs["workspace"] == workspace
        assert kwargs["min_free_mb"] == 0
        return {"passed": True, "errors": [], "workspace": str(workspace)}

    audit_module.verify_rknn_workspace = _pass_workspace

    report = audit_module.audit(
        require_rknn_workspace=True,
        rknn_workspace=workspace,
        rknn_workspace_min_free_mb=0,
    )

    assert report["checks"]["rknn_workspace"]["passed"] is True


def test_release_audit_can_require_persistent_rknn_workspace_deployment(tmp_path):
    audit_module = _load_audit_module()
    workspace = tmp_path / "external" / "moss-rknn-workspace"
    mount_point = tmp_path / "external"

    def _fail_deployment(**kwargs):
        assert kwargs["workspace"] == workspace
        assert kwargs["mount_point"] == mount_point
        assert kwargs["min_free_mb"] == 0
        return {"passed": False, "errors": ["mount point is not mounted"], "workspace": str(workspace)}

    audit_module.verify_rknn_workspace_deployment = _fail_deployment

    report = audit_module.audit(
        require_rknn_workspace_deployment=True,
        rknn_workspace=workspace,
        rknn_workspace_mount_point=mount_point,
        rknn_workspace_min_free_mb=0,
    )

    assert report["passed"] is False
    assert report["checks"]["rknn_workspace_deployment"]["passed"] is False
    assert "mount point is not mounted" in report["errors"]

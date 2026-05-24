"""Evidence locks for RK3576 canonical MOSS ORT deployment."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"
DESTINATION = "/opt/tts/models/moss-tts-nano-onnx"
SOURCE = "/home/cat/moss-onnx-baseline"


def test_rk3576_moss_ort_canonical_symlink_was_deployed():
    report = json.loads((EVIDENCE_DIR / "rk3576-moss-ort-deployment-execute.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["execute"] is True
    assert report["deployed"] is True
    assert report["mode"] == "symlink"
    assert report["source"] == SOURCE
    assert report["destination"] == DESTINATION
    assert report["source_manifest"]["required_artifacts"] == 11
    assert report["commands"] == [
        ["mkdir", "-p", "/opt/tts/models"],
        ["ln", "-s", SOURCE, DESTINATION],
    ]
    assert all(result["executed"] is True and result["returncode"] == 0 for result in report["results"])


def test_rk3576_moss_release_audit_passes_with_canonical_artifacts():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-release-audit-ort-artifact-pass.json").read_text(encoding="utf-8")
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_evidence"]["passed"] is True
    assert report["checks"]["roundtrip_evidence"]["passed"] is True
    assert report["checks"]["rknn_candidate_config"]["passed"] is True
    artifacts = report["checks"]["ort_artifacts"]
    assert artifacts["passed"] is True
    assert artifacts["summary"]["model_dir"] == DESTINATION
    assert artifacts["summary"]["target_platform"] == "rk3576"
    assert artifacts["summary"]["required_artifacts"] == 11


def test_rk3576_moss_production_runner_preflight_passes_with_canonical_artifacts():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-production-server-opt-artifact-pass.json").read_text(encoding="utf-8")
    )

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["checks"]["ort_artifacts"]["summary"]["model_dir"] == DESTINATION
    assert report["service"]["dry_run"] is True
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}
    assert report["service"]["command"][-2:] == ["--port", "8621"]


def test_rk3576_moss_canonical_service_profile_is_streaming_and_production_safe():
    report = json.loads((EVIDENCE_DIR / "rk3576-moss-canonical-profile.json").read_text(encoding="utf-8"))

    assert report["gates"]["passed"] is True
    assert report["gates"]["errors"] == []
    assert report["health"]["tts_info"]["model_dir"] == DESTINATION
    assert report["health"]["tts_info"]["manifest"]["validated"] is True
    assert report["health"]["tts_info"]["manifest"]["required_artifacts"] == 11
    assert report["health"]["tts_info"]["profile"]["codec_streaming"] is True
    assert report["health"]["tts_info"]["profile"]["codec_full_loaded"] is False
    assert report["health_after"]["tts_info"]["streaming_stats"]["requests"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["completed"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["errors"] == 0
    assert report["health_after"]["tts_info"]["streaming_stats"]["active"] == 0
    assert report["tts_stream"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["max_payload_gap_ms"] <= 1500
    assert report["dialogue"]["binary_chunks"] >= 7

"""Evidence lock for RK3576 release audit ORT artifact gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-release-audit-ort-artifact-gate.json"


def test_rk3576_release_audit_ort_artifact_gate_reports_missing_canonical_model_dir():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_evidence"]["passed"] is True
    assert report["checks"]["roundtrip_evidence"]["passed"] is True
    assert report["checks"]["rknn_candidate_config"]["passed"] is True
    assert report["checks"]["ort_artifacts"]["passed"] is False
    assert "tts.model_dir does not exist: /opt/tts/models/moss-tts-nano-onnx" in report["errors"]
    remediation = report["checks"]["ort_artifacts"]["remediation"]
    assert "models/tts/moss/prepare_moss_ort_deployment.py" in remediation["dry_run"]
    assert "--execute" in remediation["execute_requires_confirmation"]
    assert "INSTALL_MOSS_ORT_DEPLOYMENT" in remediation["execute_requires_confirmation"]

"""Evidence lock for RK3576 MOSS deployment path preflight."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-production-server-opt-artifact-gate.json"


def test_rk3576_canonical_opt_model_path_is_not_deployed_yet():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_artifacts"]["passed"] is False
    assert "tts.model_dir does not exist: /opt/tts/models/moss-tts-nano-onnx" in report["errors"]
    remediation = report["checks"]["ort_artifacts"]["remediation"]
    assert "models/tts/moss/prepare_moss_ort_deployment.py" in remediation["dry_run"]
    assert "--execute" in remediation["execute_requires_confirmation"]
    assert "INSTALL_MOSS_ORT_DEPLOYMENT" in remediation["execute_requires_confirmation"]
    assert report["service"]["dry_run"] is True

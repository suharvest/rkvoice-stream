"""Evidence lock for RK3576 canonical MOSS ORT deployment dry-run."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-ort-deployment-dry-run.json"


def test_rk3576_moss_ort_deployment_dry_run_is_safe_and_validated():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["execute"] is False
    assert report["deployed"] is False
    assert report["source"] == "/home/cat/moss-onnx-baseline"
    assert report["destination"] == "/opt/tts/models/moss-tts-nano-onnx"
    assert report["source_manifest"]["required_artifacts"] == 11
    assert report["commands"] == [
        ["mkdir", "-p", "/opt/tts/models"],
        ["ln", "-s", "/home/cat/moss-onnx-baseline", "/opt/tts/models/moss-tts-nano-onnx"],
    ]
    assert all(result["executed"] is False for result in report["results"])

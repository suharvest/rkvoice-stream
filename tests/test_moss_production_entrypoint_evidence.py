"""Evidence lock for RK3576 MOSS production entrypoint verification."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"


def test_rk3576_production_entrypoint_preflight_passes():
    report = json.loads((EVIDENCE_DIR / "rk3576-moss-production-entrypoint-preflight.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_artifacts"]["passed"] is True
    assert report["checks"]["ort_artifacts"]["summary"]["model_dir"] == "/opt/tts/models/moss-tts-nano-onnx"
    assert report["service"]["dry_run"] is False
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}
    assert report["service"]["command"] == [
        "/home/cat/rknn-venv/bin/python",
        "-m",
        "uvicorn",
        "rkvoice_stream.app.server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8629",
    ]


def test_rk3576_production_entrypoint_streaming_profile_passes():
    report = json.loads((EVIDENCE_DIR / "rk3576-moss-production-entrypoint-profile.json").read_text(encoding="utf-8"))

    assert report["gates"]["passed"] is True
    assert report["gates"]["errors"] == []
    assert report["entrypoint"]["runner"].endswith("models/tts/moss/run_moss_production_server.py")
    assert report["entrypoint"]["config"] == "configs/rk3576-moss-ort-stream.yaml"
    assert report["health"]["tts_info"]["model_dir"] == "/opt/tts/models/moss-tts-nano-onnx"
    assert report["health"]["tts_info"]["manifest"]["validated"] is True
    assert report["health_after"]["tts_info"]["streaming_stats"]["requests"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["completed"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["errors"] == 0
    assert report["tts_stream"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["max_payload_gap_ms"] <= 1500
    assert report["dialogue"]["binary_chunks"] >= 7


def test_rk3576_production_entrypoint_log_has_clean_shutdown():
    text = (EVIDENCE_DIR / "rk3576-moss-production-entrypoint.log").read_text(encoding="utf-8")

    assert "Application startup complete." in text
    assert "Uvicorn running on http://127.0.0.1:8629" in text
    assert "Application shutdown complete." in text
    assert "Finished server process" in text
    assert "Traceback" not in text
    assert "SystemExit" not in text
    assert "ERROR" not in text

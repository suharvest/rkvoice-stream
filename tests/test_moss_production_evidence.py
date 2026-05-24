"""Regression checks for checked-in MOSS RK3576 production evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-ort-production-current-rerun.json"
ROUNDTRIP_EVIDENCE = (
    ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-ort-production-current-rerun-roundtrip.json"
)


def _load_json(path: Path) -> dict:
    assert path.exists(), f"missing production evidence: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rk3576_moss_ort_production_evidence_passes_all_gates():
    report = _load_json(EVIDENCE)

    assert report["summary"]["passed"] is True
    assert report["summary"]["errors"] == []
    assert report["summary"]["checks"] == {
        "artifact_manifest": True,
        "service_streaming": True,
        "backend_stage": True,
        "roundtrip_quality": True,
    }


def test_rk3576_moss_ort_production_evidence_locks_safe_runtime_profile():
    report = _load_json(EVIDENCE)
    profile = report["profile"]
    runtime = report["service_streaming"]["health"]["tts_info"]["profile"]
    hybrid = report["service_streaming"]["health"]["tts_info"]["hybrid"]

    assert profile["backend"] == "moss_ort"
    assert profile["manifest"] == "moss-ort-manifest.json"
    assert profile["voice"] == "Junhao"
    assert profile["seed"] == 314
    assert profile["threads"] == 6
    assert profile["session_threads"] == {"prefill": 8, "decode": 5, "sampler": 6, "codec": 5}
    assert profile["prefill_seq"] == 0
    assert profile["codec_streaming"] == 1
    assert profile["codec_batch_frames"] == 3

    assert runtime["codec_streaming"] is True
    assert runtime["codec_full_loaded"] is False
    assert runtime["codec_batch_frames"] == 3
    assert runtime["codec_async"] is False
    assert runtime["cache_voice_prefix"] is False
    assert hybrid["enabled"] is False


def test_rk3576_moss_ort_production_evidence_meets_latency_and_quality_contract():
    report = _load_json(EVIDENCE)
    service = report["service_streaming"]
    backend = report["backend_stage"]["summary"]
    roundtrip = report["roundtrip_quality"]

    assert service["tts_stream"]["first_payload_ms"] <= 1500
    assert service["dialogue"]["first_payload_ms"] <= 1500
    assert service["dialogue"]["max_payload_gap_ms"] <= 1500
    assert service["dialogue"]["binary_chunks"] >= 7
    assert backend["ttfa_ms"] <= 1500
    assert backend["prefill_ms"] <= 1200
    assert backend["max_codec_ms"] <= 170
    assert backend["audio_frames"] >= 20
    assert roundtrip["avg_cer"] <= 0.5
    assert roundtrip["max_cer"] <= 1.0
    assert roundtrip["min_rms"] >= 0.02
    assert roundtrip["max_ttfa_ms"] <= 1500
    assert roundtrip["max_codec_ms"] <= 170


def test_rk3576_roundtrip_evidence_matches_production_profile_summary():
    report = _load_json(EVIDENCE)
    roundtrip = _load_json(ROUNDTRIP_EVIDENCE)

    assert roundtrip["gates"]["passed"] is True
    assert roundtrip["avg_cer"] == report["roundtrip_quality"]["avg_cer"]
    assert roundtrip["max_cer"] == report["roundtrip_quality"]["max_cer"]
    assert roundtrip["min_rms"] == report["roundtrip_quality"]["min_rms"]
    assert roundtrip["max_ttfa_ms"] == report["roundtrip_quality"]["max_ttfa_ms"]
    assert roundtrip["max_codec_ms"] == report["roundtrip_quality"]["max_codec_ms"]

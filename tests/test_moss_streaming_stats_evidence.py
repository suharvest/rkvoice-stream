"""Evidence lock for RK3576 MOSS streaming-stats service gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-service-streaming-stats-profile.json"


def test_rk3576_moss_service_streaming_stats_profile_passes():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["gates"]["passed"] is True
    assert report["gates"]["errors"] == []
    assert report["health"]["tts_info"]["streaming_stats"]["errors"] == 0
    assert report["health_after"]["tts_info"]["streaming_stats"]["requests"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["completed"] == 2
    assert report["health_after"]["tts_info"]["streaming_stats"]["errors"] == 0
    assert report["health_after"]["tts_info"]["streaming_stats"]["active"] == 0
    assert report["health_after"]["tts_info"]["streaming_stats"]["chunks"] >= 14
    assert report["tts_stream"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["first_payload_ms"] <= 1500
    assert report["dialogue"]["max_payload_gap_ms"] <= 1500
    assert report["dialogue"]["binary_chunks"] >= 7

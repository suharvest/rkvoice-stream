"""Evidence lock for RK3576 sampler RKNN promotion gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-sampler-sequential-per-block-rknn-promotion.json"


def test_rk3576_sampler_per_block_rknn_is_not_service_integratable_yet():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["mlps_mode"] == "per_block"
    assert report["runs"] == 1
    assert report["token_equal"] == 0
    assert report["continue_equal"] == 1
    assert report["gates"]["token_parity"] is False
    assert report["gates"]["continue_parity"] is True
    assert report["gates"]["mlp_parity"] is True
    assert report["gates"]["passed"] is False
    assert report["latency_ms"]["rknn_mlps_avg"] < 80
    assert report["latency_ms"]["split_total_avg"] > report["latency_ms"]["full_ort_avg"]
    assert report["promotion"]["allow_service_integration"] is False
    assert any(error.startswith("token parity failed") for error in report["promotion"]["errors"])
    assert any("below required" in error for error in report["promotion"]["errors"])

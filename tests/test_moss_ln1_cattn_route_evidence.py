"""Evidence lock for the coarser MOSS ln1+cattn RKNN route."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def test_all_layer_ln1_cattn_s320_builds_and_runs_on_rk3576():
    build = _load("wsl2-moss-ln1-cattn-s320-all-build.json")
    runtime = _load("rk3576-moss-ln1-cattn-s320-all-runtime-probe.json")

    assert build["preset"] == "ln1_cattn"
    assert build["passed"] is True
    assert build["layers"] == list(range(12))
    assert all(item["returncode"] == 0 for item in build["results"])
    assert all(item["report_json"]["rknn_build"]["status"] == "OK" for item in build["results"])

    assert runtime["summary"] == {"total": 12, "ok": 12, "crash": 0, "timeout": 0, "missing": 0}
    assert all(item["status"] == "OK" for item in runtime["results"])
    assert all(item["outputs"][0]["shape"] == [1, 320, 2304] for item in runtime["results"])
    assert all(item["outputs"][0]["finite"] is True for item in runtime["results"])


def test_all_layer_ln1_cattn_s320_parity_and_island_speedup_pass():
    parity = _load("rk3576-moss-ln1-cattn-s320-all-parity.json")
    summary = parity["summary"]

    assert parity["passed"] is True
    assert summary["layers_total"] == 12
    assert summary["reports"] == 12
    assert summary["max_rel_l2"] < 0.002
    assert summary["min_cosine"] > 0.99999
    assert summary["sum_rknn_avg_ms"] < summary["sum_ort_avg_ms"]
    assert summary["speedup"] > 1.8


def test_ln1_cattn_integrated_prefill_improves_the_route_but_still_needs_service_gates():
    baseline = _load("rk3576-moss-hybrid-prefill-ln2-mlp-baseline.json")
    cattn = _load("rk3576-moss-hybrid-prefill-ln2-mlp-cattn.json")
    ln1_cattn = _load("rk3576-moss-hybrid-prefill-ln2-mlp-ln1-cattn.json")

    assert ln1_cattn["use_ln1_cattn"] is True
    assert ln1_cattn["gates"]["passed"] is True
    assert ln1_cattn["outputs"]["global_hidden"]["finite"] is True
    assert ln1_cattn["outputs"]["global_hidden"]["rel_l2"] < 0.005
    assert ln1_cattn["outputs"]["global_hidden"]["cosine"] > 0.99999
    assert ln1_cattn["outputs"]["kv_max_rel_l2"] < 0.006
    assert ln1_cattn["outputs"]["kv_min_cosine"] > 0.99998

    baseline_ms = baseline["timings_ms"]["hybrid_prefill_ms"]
    cattn_ms = cattn["timings_ms"]["hybrid_prefill_ms"]
    ln1_cattn_ms = ln1_cattn["timings_ms"]["hybrid_prefill_ms"]
    assert ln1_cattn_ms < cattn_ms < baseline_ms
    assert baseline_ms / ln1_cattn_ms > 1.15
    assert cattn_ms / ln1_cattn_ms > 1.07


def test_ln1_cattn_service_smoke_streams_audio_on_rk3576():
    smoke = _load("rk3576-moss-service-smoke-ln1-cattn.json")
    summary = smoke["summary"]
    first = summary["first_meta"]

    assert smoke["gates"]["passed"] is True
    assert summary["chunks"] >= 2
    assert summary["audio_frames"] >= 2
    assert summary["ttfa_ms"] < 1300
    assert summary["hybrid_prefill_ms"] < 1050
    assert first["mode"] == "text_hybrid_rknn"
    assert first["hybrid"]["layers"][0]["attention_kind"] == "rknn_ln1_cattn_suffix_ort"
    assert all(layer["mlp_kind"] == "rknn_ln2_mlp" for layer in first["hybrid"]["layers"])


def test_ln1_cattn_production_service_profile_streams_but_is_not_promoted_over_ort():
    hybrid = _load("rk3576-moss-service-profile-ln1-cattn-production.json")
    ort = _load("rk3576-moss-canonical-profile.json")

    assert hybrid["gates"]["passed"] is True
    assert hybrid["gates"]["errors"] == []
    assert hybrid["health"]["tts_info"]["manifest"]["validated"] is True
    assert hybrid["health"]["tts_info"]["profile"]["voice"] == "Junhao"
    assert hybrid["health"]["tts_info"]["profile"]["seed"] == 314
    assert hybrid["health"]["tts_info"]["profile"]["codec_batch_frames"] == 3
    assert hybrid["health"]["tts_info"]["hybrid"]["enabled"] is True
    assert hybrid["health"]["tts_info"]["hybrid"]["strict"] is True
    assert hybrid["health"]["tts_info"]["hybrid"]["split"] == "ln1_cattn"
    assert hybrid["health"]["tts_info"]["hybrid"]["layers"] == list(range(12))
    assert hybrid["health_after"]["tts_info"]["streaming_stats"]["requests"] == 2
    assert hybrid["health_after"]["tts_info"]["streaming_stats"]["completed"] == 2
    assert hybrid["health_after"]["tts_info"]["streaming_stats"]["errors"] == 0

    assert hybrid["tts_stream"]["first_payload_ms"] <= 1500
    assert hybrid["dialogue"]["first_payload_ms"] <= 1500
    assert hybrid["dialogue"]["max_payload_gap_ms"] <= 1500
    assert hybrid["dialogue"]["binary_chunks"] >= 7

    # Under the current short Junhao production prompt, RKNN prefill offload is
    # service-safe but not a service-level win over the canonical full-ORT route.
    assert hybrid["tts_stream"]["first_payload_ms"] > ort["tts_stream"]["first_payload_ms"]
    assert hybrid["dialogue"]["wall_ms"] > ort["dialogue"]["wall_ms"]

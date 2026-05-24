"""Evidence locks for RK3576 MOSS backend-stage timing."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def test_backend_stage_profiles_cover_same_production_streaming_shape():
    ort = _load("rk3576-moss-backend-stage-ort-junhao314-frames20.json")
    hybrid = _load("rk3576-moss-backend-stage-ln1-cattn-junhao314-frames20.json")

    for report in (ort, hybrid):
        assert report["text"] == "你好"
        assert report["seed"] == 314
        assert report["max_new_frames"] == 20
        assert report["gates"]["passed"] is True
        assert report["gates"]["errors"] == []
        assert report["summary"]["chunks"] == 8
        assert report["summary"]["audio_frames"] == 20
        assert report["summary"]["total_samples"] == 76800
        assert report["summary"]["first_meta"]["codec_batch_frames"] == 1
        assert report["summary"]["max_codec_ms"] <= 180
        assert report["summary"]["ttfa_ms"] <= 1500


def test_ln1_cattn_stage_profile_identifies_prefill_handoff_as_the_blocker():
    ort = _load("rk3576-moss-backend-stage-ort-junhao314-frames20.json")
    hybrid = _load("rk3576-moss-backend-stage-ln1-cattn-junhao314-frames20.json")

    ort_summary = ort["summary"]
    hybrid_summary = hybrid["summary"]
    first = hybrid_summary["first_meta"]
    hybrid_meta = first["hybrid"]
    layers = hybrid_meta["layers"]

    assert first["mode"] == "text_hybrid_rknn"
    assert hybrid_summary["hybrid_prefill_ms"] == hybrid_meta["hybrid_prefill_ms"]
    assert len(layers) == 12
    assert all(layer["attention_kind"] == "rknn_ln1_cattn_suffix_ort" for layer in layers)
    assert all(layer["mlp_kind"] == "rknn_ln2_mlp" for layer in layers)

    attention_total = sum(float(layer["attention_ms"]) for layer in layers)
    mlp_total = sum(float(layer["mlp_ms"]) for layer in layers)
    layer_total = sum(float(layer["layer_ms"]) for layer in layers)

    assert 700 <= attention_total <= 750
    assert 190 <= mlp_total <= 210
    assert 920 <= layer_total <= 950
    assert mlp_total < attention_total / 3

    # On the short Junhao production prompt, the current coarse RKNN prefill
    # route is service-safe but slower than the tuned full-ORT backend path.
    assert hybrid_summary["prefill_ms"] > ort_summary["prefill_ms"]
    assert hybrid_summary["ttfa_ms"] > ort_summary["ttfa_ms"]
    assert hybrid_summary["wall_ms"] > ort_summary["wall_ms"]

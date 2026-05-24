"""Evidence lock for integrated MOSS cattn RKNN hybrid prefill."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def test_ln2_mlp_baseline_prefill_passes_hidden_and_kv_gates():
    report = _load("rk3576-moss-hybrid-prefill-ln2-mlp-baseline.json")

    assert report["use_cattn"] is False
    assert report["seq_len"] == 320
    assert report["actual_len"] == 297
    assert report["gates"]["passed"] is True
    assert report["outputs"]["global_hidden"]["finite"] is True
    assert report["outputs"]["global_hidden"]["rel_l2"] < 0.003
    assert report["outputs"]["global_hidden"]["cosine"] > 0.99999
    assert report["outputs"]["kv_max_rel_l2"] < 0.004
    assert report["outputs"]["kv_min_cosine"] > 0.99999
    assert report["timings_ms"]["hybrid_prefill_ms"] < report["timings_ms"]["full_prefill_target_ms"]


def test_cattn_integrated_prefill_passes_accuracy_and_has_net_speedup():
    baseline = _load("rk3576-moss-hybrid-prefill-ln2-mlp-baseline.json")
    cattn = _load("rk3576-moss-hybrid-prefill-ln2-mlp-cattn.json")

    assert cattn["use_cattn"] is True
    assert cattn["gates"]["passed"] is True
    assert cattn["outputs"]["global_hidden"]["finite"] is True
    assert cattn["outputs"]["global_hidden"]["rel_l2"] < 0.003
    assert cattn["outputs"]["global_hidden"]["cosine"] > 0.99999
    assert cattn["outputs"]["kv_max_rel_l2"] < 0.004
    assert cattn["outputs"]["kv_min_cosine"] > 0.99999

    base_ms = baseline["timings_ms"]["hybrid_prefill_ms"]
    cattn_ms = cattn["timings_ms"]["hybrid_prefill_ms"]
    assert cattn_ms < base_ms
    assert base_ms / cattn_ms > 1.05


def test_cattn_integrated_prefill_exposes_remaining_handoff_cost():
    cattn = _load("rk3576-moss-hybrid-prefill-ln2-mlp-cattn.json")
    layers = cattn["timings_ms"]["layers"]

    assert len(layers) == 12
    assert all(item["ln1_ms"] is not None for item in layers)
    assert all(item["cattn_rknn_ms"] is not None for item in layers)
    assert all(item["attention_suffix_ms"] is not None for item in layers)

    cattn_rknn_ms = sum(item["cattn_rknn_ms"] for item in layers)
    suffix_ms = sum(item["attention_suffix_ms"] for item in layers)
    ln1_ms = sum(item["ln1_ms"] for item in layers)

    assert cattn_rknn_ms > 200.0
    assert suffix_ms > cattn_rknn_ms
    assert suffix_ms + ln1_ms > 500.0

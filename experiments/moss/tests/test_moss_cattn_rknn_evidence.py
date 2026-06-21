"""Evidence lock for the MOSS fused qkv projection RKNN island."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"


def _load(name: str) -> dict:
    return json.loads((EVIDENCE_DIR / name).read_text(encoding="utf-8"))


def test_block0_cattn_s320_builds_and_runs_on_rk3576():
    build = _load("wsl2-moss-cattn-s320-build.json")
    runtime = _load("rk3576-moss-cattn-s320-runtime-probe.json")

    assert build["preset"] == "cattn"
    assert build["inputs"] == ["/ln_1/LayerNormalization_output_0"]
    assert build["outputs"] == ["/c_attn/Add_output_0"]
    assert build["rknn_build"]["status"] == "OK"
    assert build["rknn_build"]["size_bytes"] < 5_000_000

    assert runtime["summary"] == {"total": 1, "ok": 1, "crash": 0, "timeout": 0, "missing": 0}
    result = runtime["results"][0]
    assert result["status"] == "OK"
    assert result["outputs"][0]["shape"] == [1, 320, 2304]
    assert result["outputs"][0]["finite"] is True


def test_block0_cattn_s320_parity_and_latency_are_promising_but_not_service_proof():
    parity = _load("rk3576-moss-cattn-s320-parity.json")
    metrics = parity["outputs"][0]
    latency = parity["latency_ms"]

    assert parity["gates"]["passed"] is True
    assert metrics["rel_l2"] < 0.001
    assert metrics["cosine"] > 0.99999
    assert latency["rknn_avg"] < latency["ort_avg"]
    assert latency["repeat"] >= 8

    # This is an isolated projection island. Service promotion still requires
    # all-layer integration, hidden/logit parity, and streaming quality gates.
    assert parity["input_names"] == ["/ln_1/LayerNormalization_output_0"]


def test_all_layer_cattn_s320_builds_and_runs_on_rk3576():
    build = _load("wsl2-moss-cattn-s320-all-build.json")
    runtime = _load("rk3576-moss-cattn-s320-all-runtime-probe.json")

    assert build["passed"] is True
    assert build["layers"] == list(range(12))
    assert len(build["results"]) == 12
    assert all(item["returncode"] == 0 for item in build["results"])
    assert all(item["report_json"]["rknn_build"]["status"] == "OK" for item in build["results"])

    assert runtime["summary"] == {"total": 12, "ok": 12, "crash": 0, "timeout": 0, "missing": 0}
    assert all(item["status"] == "OK" for item in runtime["results"])
    assert all(item["outputs"][0]["shape"] == [1, 320, 2304] for item in runtime["results"])
    assert all(item["outputs"][0]["finite"] is True for item in runtime["results"])


def test_all_layer_cattn_s320_parity_and_island_speedup_pass():
    parity = _load("rk3576-moss-cattn-s320-all-parity.json")
    summary = parity["summary"]

    assert parity["passed"] is True
    assert summary["layers_total"] == 12
    assert summary["reports"] == 12
    assert summary["max_rel_l2"] < 0.001
    assert summary["min_cosine"] > 0.99999
    assert summary["sum_rknn_avg_ms"] < summary["sum_ort_avg_ms"]
    assert summary["speedup"] > 1.5

"""Regression checks for MOSS RKLLM runtime evidence on RK3576."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"


def _load_json(name: str) -> dict:
    path = EVIDENCE_DIR / name
    assert path.exists(), f"missing RKLLM evidence: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rkllm_smoke_audio_is_not_production_accuracy_evidence():
    smoke = _load_json("rk3576-moss-rkllm-stream-smoke.json")
    hidden = _load_json("rk3576-moss-rkllm-hidden-vs-onnx-s8.json")

    assert smoke["passed"] is True
    assert smoke["audio_finite"] is True
    assert smoke["audio_rms"] > 0.02

    assert hidden["passed"] is False
    assert hidden["prefill_metrics"]["rel_l2"] > hidden["thresholds"]["max_prefill_rel_l2"]
    assert hidden["decode_metrics"]["rel_l2"] > hidden["thresholds"]["max_decode_rel_l2"]
    assert hidden["prefill_metrics"]["cosine"] < hidden["thresholds"]["min_cosine"]
    assert hidden["decode_metrics"]["cosine"] < hidden["thresholds"]["min_cosine"]


def test_rkllm_runtime_probe_must_be_followed_by_hidden_parity():
    runtime = _load_json("rk3576-moss-rkllm-folded-runtime-probe.json")
    hidden = _load_json("rk3576-moss-rkllm-hidden-vs-onnx-s8.json")

    assert runtime["loaded"] is True
    assert runtime["prefill_ok"] is True
    assert runtime["decode_ok"] is True

    assert hidden["rkllm_model"] == runtime["model_path"]
    assert hidden["passed"] is False
    assert hidden["prefill_metrics"]["rel_l2"] >= 1.0
    assert hidden["decode_metrics"]["rel_l2"] >= 1.0


def test_rkllm_token_input_is_not_a_moss_accuracy_fix():
    token = _load_json("rk3576-moss-rkllm-token-hidden-s8.json")

    assert token["passed"] is False
    assert token["prefill_metrics"]["rel_l2"] > token["thresholds"]["max_rel_l2"]
    assert token["prefill_metrics"]["cosine"] < token["thresholds"]["min_cosine"]
    assert token["prefill_metrics"]["rel_l2"] >= 1.0


def test_rkllm_embed_flash_zero_is_not_a_moss_accuracy_fix():
    default = _load_json("rk3576-moss-rkllm-hidden-vs-onnx-s8.json")
    embed_flash0 = _load_json("rk3576-moss-rkllm-hidden-vs-onnx-s8-embedflash0.json")

    assert embed_flash0["passed"] is False
    assert embed_flash0["prefill_metrics"]["rel_l2"] >= 1.0
    assert embed_flash0["decode_metrics"]["rel_l2"] >= 1.0
    assert abs(
        embed_flash0["prefill_metrics"]["rel_l2"]
        - default["prefill_metrics"]["rel_l2"]
    ) < 1.0e-4


def test_rmsnorm_variant_does_not_explain_rkllm_moss_mismatch():
    variants = _load_json("wsl2-moss-hf-variants-vs-rkllm-s8.json")

    assert variants["variants"]["hf_original"]["vs_onnx"]["rel_l2"] < 1.0e-4
    assert variants["best_vs_rkllm"] == "hf_rmsnorm"
    assert variants["variants"]["hf_rmsnorm"]["vs_rkllm"]["rel_l2"] >= 1.0
    assert variants["variants"]["hf_rmsnorm"]["vs_rkllm"]["cosine"] < 0.2

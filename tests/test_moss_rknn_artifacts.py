"""MOSS RKNN artifact contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rkvoice_stream.backends.tts.moss_rknn import MossArtifactError, validate_moss_artifacts
from models.tts.moss.extract_moss_rknn_island import _preset_spec
from models.tts.moss.verify_moss_sampler_text_head_mlps_split import (
    LN2_OUTPUTS,
    MLP_OUTPUTS,
    TEXT_HEAD_IN,
    TEXT_HEAD_OUT,
    _mlp_node_prefixes,
)
from models.tts.moss.verify_moss_sampler_sequential_mlps_split import (
    _mlp_prefixes_for,
    _parse_layer_spec,
    _promotion_decision,
    _resolve_mlp_rknn_paths,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_simple_manifest(tmp_path, *, production_default: bool = False, full_evidence: bool = False) -> None:
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    quality_status: dict[str, object] = {"production_default": production_default}
    if full_evidence:
        quality_status["production_evidence"] = {
            "passed": True,
            "checks": {
                "artifact_manifest": True,
                "service_streaming": True,
                "backend_stage": True,
                "roundtrip_quality": True,
            },
        }
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "quality_status": quality_status,
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validate_moss_artifacts_accepts_manifest_with_hashes(tmp_path):
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "channels": 1,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "production_gates": {
            "max_ttfa_ms": 500,
            "max_rtf": 0.75,
            "max_asr_cer": 0.15,
        },
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_artifacts(tmp_path)
    assert parsed["target_platform"] == "rk3576"


def test_validate_moss_artifacts_accepts_production_default_with_full_evidence(tmp_path):
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "quality_status": {
            "production_default": True,
            "production_evidence": {
                "passed": True,
                "checks": {
                    "artifact_manifest": True,
                    "service_streaming": True,
                    "backend_stage": True,
                    "roundtrip_quality": True,
                },
            },
        },
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_artifacts(tmp_path)

    assert parsed["quality_status"]["production_default"] is True


def test_validate_moss_artifacts_rejects_production_default_without_evidence(tmp_path):
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "quality_status": {"production_default": True},
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossArtifactError, match="production_evidence.passed=true"):
        validate_moss_artifacts(tmp_path)


def test_validate_moss_artifacts_can_require_production_default(tmp_path):
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "quality_status": {"production_default": False},
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossArtifactError, match="requires quality_status.production_default=true"):
        validate_moss_artifacts(tmp_path, require_production_default=True)


def test_verify_moss_rknn_artifacts_cli_requires_production_default(tmp_path):
    _write_simple_manifest(tmp_path, production_default=False)
    script = ROOT / "models" / "tts" / "moss" / "verify_moss_rknn_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-dir",
            str(tmp_path),
            "--require-production-default",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "requires quality_status.production_default=true" in result.stderr


def test_verify_moss_rknn_artifacts_cli_accepts_full_production_evidence(tmp_path):
    _write_simple_manifest(tmp_path, production_default=True, full_evidence=True)
    script = ROOT / "models" / "tts" / "moss" / "verify_moss_rknn_artifacts.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model-dir",
            str(tmp_path),
            "--require-production-default",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert "MOSS_RKNN_ARTIFACTS_OK" in result.stdout
    assert "production_default=True" in result.stdout


def test_validate_moss_artifacts_rejects_incomplete_production_evidence(tmp_path):
    artifact = tmp_path / "tokenizer.model"
    artifact.write_bytes(b"tokenizer")
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [
            {
                "path": "tokenizer.model",
                "required": True,
                "size_bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
            }
        ],
        "quality_status": {
            "production_default": True,
            "production_evidence": {
                "passed": True,
                "checks": {
                    "artifact_manifest": True,
                    "service_streaming": True,
                    "backend_stage": False,
                    "roundtrip_quality": True,
                },
            },
        },
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossArtifactError, match="requires production_evidence checks"):
        validate_moss_artifacts(tmp_path)


def test_validate_moss_artifacts_rejects_missing_required_file(tmp_path):
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "artifacts": [{"path": "missing.rknn", "required": True}],
    }
    (tmp_path / "moss-rknn-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossArtifactError, match="Missing required"):
        validate_moss_artifacts(tmp_path)


def test_attn_residual_preset_matches_hybrid_split_contract():
    block0 = _preset_spec("attn_residual", 0)
    block7 = _preset_spec("attn_residual", 7)

    assert block0["inputs"] == ["/Add_15_output_0", "attention_mask"]
    assert block0["outputs"] == ["/Add_19_output_0", "present_key_0", "present_value_0"]
    assert block0["input_shapes"] == {
        "/Add_15_output_0": [1, 32, 768],
        "attention_mask": [1, 32],
    }
    assert block7["inputs"] == ["/Mul_58_output_0", "attention_mask"]
    assert block7["outputs"] == ["/Add_54_output_0", "present_key_7", "present_value_7"]


def test_ln2_and_mlp_presets_support_mlp_only_split_contract():
    ln2 = _preset_spec("ln2", 0)
    mlp = _preset_spec("mlp", 0)
    fc_out = _preset_spec("fc_out", 0)

    assert ln2["inputs"] == ["/Add_19_output_0"]
    assert ln2["outputs"] == ["/ln_2/LayerNormalization_output_0"]
    assert mlp["inputs"] == ["/ln_2/LayerNormalization_output_0"]
    assert mlp["outputs"] == ["/mlp/fc_out/Add_output_0"]
    assert fc_out["inputs"] == ["/mlp/act/Mul_3_output_0"]
    assert fc_out["outputs"] == ["/mlp/fc_out/Add_output_0"]
    assert fc_out["shape"] == [1, 32, 3072]


def test_prefill_edge_presets_match_hybrid_split_contract():
    embedding = _preset_spec("embedding_prefix", 0)
    final_norm = _preset_spec("final_norm", 0)

    assert embedding["inputs"] == ["input_ids"]
    assert embedding["outputs"] == ["/Add_15_output_0"]
    assert embedding["input_shapes"] == {"input_ids": [1, 32, 17]}
    assert final_norm["inputs"] == ["/Mul_88_output_0"]
    assert final_norm["outputs"] == ["/ln_f/LayerNormalization_output_0"]
    assert final_norm["input_shapes"] == {
        "/Mul_88_output_0": [1, 32, 768],
    }


def test_sampler_island_presets_match_sampler_graph_contract():
    audio0 = _preset_spec("sampler_audio_head", 0)
    audio9 = _preset_spec("sampler_audio_head", 9)
    audio_all = _preset_spec("sampler_audio_heads", 0)
    text = _preset_spec("sampler_text_lm_head", 0)
    fc_in = _preset_spec("sampler_fc_in_act", 16)
    fc_out = _preset_spec("sampler_fc_out", 16)
    mlp = _preset_spec("sampler_mlp", 16)
    mlps = _preset_spec("sampler_mlps", 0)

    assert audio0["inputs"] == ["/Gather_25_output_0"]
    assert audio0["outputs"] == ["/audio_lm_heads.0/MatMul_output_0"]
    assert audio0["shape"] == [1, 768]
    assert audio9["inputs"] == ["/Gather_151_output_0"]
    assert audio9["outputs"] == ["/audio_lm_heads.9/MatMul_output_0"]
    assert len(audio_all["inputs"]) == 16
    assert len(audio_all["outputs"]) == 16
    assert audio_all["inputs"][15] == "/Gather_235_output_0"
    assert audio_all["outputs"][15] == "/audio_lm_heads.15/MatMul_output_0"
    assert audio_all["input_shapes"]["/Gather_25_output_0"] == [1, 768]
    assert text["inputs"] == ["/Gather_11_output_0"]
    assert text["outputs"] == ["/text_lm_head/MatMul_output_0"]
    assert fc_in["inputs"] == ["/ln_2_16/LayerNormalization_output_0"]
    assert fc_in["outputs"] == ["/mlp/act_16/Mul_3_output_0"]
    assert fc_out["inputs"] == ["/mlp/act_16/Mul_3_output_0"]
    assert fc_out["outputs"] == ["/mlp/fc_out_16/Add_output_0"]
    assert fc_in["shape"] == [1, 1, 768]
    assert fc_out["shape"] == [1, 1, 3072]
    assert mlp["inputs"] == ["/ln_2_16/LayerNormalization_output_0"]
    assert mlp["outputs"] == ["/mlp/fc_out_16/Add_output_0"]
    assert mlp["shape"] == [1, 1, 768]
    assert len(mlps["inputs"]) == 17
    assert len(mlps["outputs"]) == 17
    assert mlps["inputs"][0] == "/ln_2/LayerNormalization_output_0"
    assert mlps["inputs"][16] == "/ln_2_16/LayerNormalization_output_0"
    assert mlps["outputs"][0] == "/mlp/fc_out/Add_output_0"
    assert mlps["outputs"][16] == "/mlp/fc_out_16/Add_output_0"


def test_sampler_text_head_mlps_split_contract_names():
    prefixes = _mlp_node_prefixes()

    assert TEXT_HEAD_IN == "/Gather_11_output_0"
    assert TEXT_HEAD_OUT == "/text_lm_head/MatMul_output_0"
    assert LN2_OUTPUTS[0] == "/ln_2/LayerNormalization_output_0"
    assert LN2_OUTPUTS[16] == "/ln_2_16/LayerNormalization_output_0"
    assert MLP_OUTPUTS[0] == "/mlp/fc_out/Add_output_0"
    assert MLP_OUTPUTS[16] == "/mlp/fc_out_16/Add_output_0"
    assert "/mlp/fc_in/" in prefixes
    assert "/mlp/act_16/" in prefixes
    assert "/mlp/fc_out_16/" in prefixes


def test_sampler_sequential_mlps_split_contract_names():
    assert _mlp_prefixes_for(0) == ("/mlp/fc_in/", "/mlp/act/", "/mlp/fc_out/")
    assert _mlp_prefixes_for(16) == ("/mlp/fc_in_16/", "/mlp/act_16/", "/mlp/fc_out_16/")


def test_sampler_sequential_split_resolves_per_block_mlp_artifacts(tmp_path):
    for index in range(17):
        (tmp_path / f"moss_sampler_mlp{index}.fp16.rk3576.rknn").write_bytes(b"rknn")

    paths = _resolve_mlp_rknn_paths(None, tmp_path)

    assert paths is not None
    assert len(paths) == 17
    assert paths[0].name == "moss_sampler_mlp0.fp16.rk3576.rknn"
    assert paths[16].name == "moss_sampler_mlp16.fp16.rk3576.rknn"


def test_sampler_sequential_split_rejects_missing_per_block_mlp_artifacts(tmp_path):
    (tmp_path / "moss_sampler_mlp0.fp16.rk3576.rknn").write_bytes(b"rknn")

    with pytest.raises(FileNotFoundError, match="block 1"):
        _resolve_mlp_rknn_paths(None, tmp_path)


def test_sampler_sequential_split_parses_layer_specs():
    assert _parse_layer_spec("all") == set(range(17))
    assert _parse_layer_spec("0,2,5-7") == {0, 2, 5, 6, 7}

    with pytest.raises(ValueError, match=r"\[17\]"):
        _parse_layer_spec("17")


def test_sampler_sequential_promotion_blocks_token_drift_even_when_mlp_parity_passes():
    decision = _promotion_decision(
        {
            "runs": 4,
            "token_equal": 2,
            "continue_equal": 4,
            "gates": {"token_parity": False, "continue_parity": True, "mlp_parity": True},
            "mlps": {"max_rel_l2": 0.0032},
            "latency_ms": {"full_ort_avg": 87.0, "split_total_avg": 70.0},
        },
        min_speedup=1.05,
    )

    assert decision["allow_service_integration"] is False
    assert any(error.startswith("token parity failed") for error in decision["errors"])


def test_sampler_sequential_promotion_requires_service_worthy_speedup():
    decision = _promotion_decision(
        {
            "runs": 4,
            "token_equal": 4,
            "continue_equal": 4,
            "gates": {"token_parity": True, "continue_parity": True, "mlp_parity": True},
            "mlps": {"max_rel_l2": 0.001},
            "latency_ms": {"full_ort_avg": 87.0, "split_total_avg": 90.0},
        },
        min_speedup=1.05,
    )

    assert decision["allow_service_integration"] is False
    assert decision["speedup"] == pytest.approx(87.0 / 90.0)
    assert any("below required" in error for error in decision["errors"])


def test_sampler_sequential_promotion_allows_fast_exact_split():
    decision = _promotion_decision(
        {
            "runs": 4,
            "token_equal": 4,
            "continue_equal": 4,
            "gates": {"token_parity": True, "continue_parity": True, "mlp_parity": True},
            "mlps": {"max_rel_l2": 0.001},
            "latency_ms": {"full_ort_avg": 100.0, "split_total_avg": 80.0},
        },
        min_speedup=1.05,
    )

    assert decision == {
        "allow_service_integration": True,
        "min_speedup": 1.05,
        "speedup": 1.25,
        "errors": [],
    }

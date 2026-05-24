"""MOSS ORT artifact contract tests."""

from __future__ import annotations

import hashlib
import json

import pytest

from rkvoice_stream import _apply_tts_env
from rkvoice_stream.backends.tts.moss_ort import (
    MossORTArtifactError,
    default_moss_hybrid_artifacts,
    default_moss_hybrid_fc_split_artifacts,
    default_moss_hybrid_ln1_cattn_artifacts,
    default_moss_hybrid_mlp_only_artifacts,
    default_moss_ort_artifacts,
    validate_moss_hybrid_artifacts,
    validate_moss_ort_artifacts,
)


def _write_required_files(root, overrides: dict[str, bytes] | None = None) -> None:
    overrides = overrides or {}
    for name in default_moss_ort_artifacts(require_streaming_codec=True):
        (root / name).write_bytes(overrides.get(name, name.encode("utf-8")))


def _manifest_for(root) -> dict:
    artifacts = []
    for name in default_moss_ort_artifacts(require_streaming_codec=True):
        data = (root / name).read_bytes()
        artifacts.append(
            {
                "path": name,
                "required": True,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "model_id": "moss-tts-nano-onnx",
        "target_platform": "rk3576",
        "sample_rate": 48000,
        "channels": 2,
        "streaming_required": True,
        "artifacts": artifacts,
        "production_gates": {
            "max_tts_first_payload_ms": 1500,
            "max_dialogue_first_payload_ms": 1500,
            "max_tts_wall_ms": 2000,
            "max_dialogue_wall_ms": 2000,
            "max_avg_cer": 0.5,
            "max_cer": 1.0,
            "min_rms": 0.02,
        },
    }


def test_validate_moss_ort_artifacts_accepts_hash_manifest(tmp_path):
    _write_required_files(tmp_path)
    manifest = _manifest_for(tmp_path)
    (tmp_path / "moss-ort-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_ort_artifacts(tmp_path)

    assert parsed["model_id"] == "moss-tts-nano-onnx"
    assert parsed["streaming_required"] is True


def test_validate_moss_ort_artifacts_rejects_sha_mismatch(tmp_path):
    _write_required_files(tmp_path)
    manifest = _manifest_for(tmp_path)
    manifest["artifacts"][0]["sha256"] = "0" * 64
    (tmp_path / "moss-ort-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossORTArtifactError, match="sha256 mismatch"):
        validate_moss_ort_artifacts(tmp_path)


def test_validate_moss_ort_artifacts_rejects_streaming_manifest_without_decode_step(tmp_path):
    _write_required_files(tmp_path)
    manifest = _manifest_for(tmp_path)
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != "moss_audio_tokenizer_decode_step.onnx"
    ]
    (tmp_path / "moss-ort-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossORTArtifactError, match="streaming manifest missing"):
        validate_moss_ort_artifacts(tmp_path)


def test_moss_ort_config_maps_manifest_env(monkeypatch):
    monkeypatch.delenv("MOSS_ORT_MANIFEST", raising=False)

    _apply_tts_env(
        {
            "backend": "moss_ort",
            "model_dir": "/models/moss",
            "manifest": "moss-ort-manifest.json",
        }
    )

    assert __import__("os").environ["MOSS_ORT_MANIFEST"] == "moss-ort-manifest.json"


def _write_hybrid_files(root, seq_len: int = 320, target: str = "rk3576") -> None:
    for name in default_moss_hybrid_artifacts(seq_len, target):
        (root / name).write_bytes(name.encode("utf-8"))


def _hybrid_manifest_for(root, seq_len: int = 320, target: str = "rk3576") -> dict:
    artifacts = []
    for name in default_moss_hybrid_artifacts(seq_len, target):
        data = (root / name).read_bytes()
        artifacts.append(
            {
                "path": name,
                "required": True,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "model_id": "moss-tts-nano-hybrid-rknn",
        "target_platform": target,
        "seq_len": seq_len,
        "split": "prefill_ln2_mlp",
        "artifacts": artifacts,
        "quality_status": {"production_default": False},
    }


def _write_fc_out_hybrid_files(root, layers: set[int], seq_len: int = 320, target: str = "rk3576") -> None:
    for name in (
        f"moss_embedding_prefix.s{seq_len}.onnx",
        f"moss_final_norm.s{seq_len}.onnx",
        *[f"moss_block{layer}_attn_residual.s{seq_len}.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_ln2.s{seq_len}.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s{seq_len}.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s{seq_len}.fp16.{target}.rknn" for layer in layers],
    ):
        (root / name).write_bytes(name.encode("utf-8"))


def _hybrid_manifest_for_names(root, names: list[str], *, split: str, layers: set[int]) -> dict:
    artifacts = []
    for name in names:
        data = (root / name).read_bytes()
        artifacts.append(
            {
                "path": name,
                "required": True,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "model_id": "moss-tts-nano-hybrid-rknn",
        "target_platform": "rk3576",
        "seq_len": 320,
        "split": f"prefill_{split}",
        "rknn_layers": sorted(layers),
        "artifacts": artifacts,
        "quality_status": {"production_default": False},
    }


def test_validate_moss_hybrid_artifacts_accepts_hash_manifest(tmp_path):
    _write_hybrid_files(tmp_path)
    manifest = _hybrid_manifest_for(tmp_path)
    (tmp_path / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_hybrid_artifacts(tmp_path, seq_len=320, target="rk3576")

    assert parsed["split"] == "prefill_ln2_mlp"
    assert parsed["quality_status"]["production_default"] is False


def test_validate_moss_hybrid_artifacts_rejects_production_default_claim(tmp_path):
    _write_hybrid_files(tmp_path)
    manifest = _hybrid_manifest_for(tmp_path)
    manifest["quality_status"]["production_default"] = True
    (tmp_path / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossORTArtifactError, match="production_default=true"):
        validate_moss_hybrid_artifacts(tmp_path, seq_len=320, target="rk3576")


def test_validate_moss_hybrid_artifacts_rejects_missing_required_entry(tmp_path):
    _write_hybrid_files(tmp_path)
    manifest = _hybrid_manifest_for(tmp_path)
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != "moss_block11_ln2_mlp.s320.fp16.rk3576.rknn"
    ]
    (tmp_path / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossORTArtifactError, match="hybrid manifest missing"):
        validate_moss_hybrid_artifacts(tmp_path, seq_len=320, target="rk3576")


def test_validate_moss_hybrid_artifacts_accepts_fc_out_layer_manifest(tmp_path):
    layers = {0, 1, 4, 5, 6}
    _write_fc_out_hybrid_files(tmp_path, layers)
    names = [
        f"moss_embedding_prefix.s320.onnx",
        f"moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_ln2.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s320.fp16.rk3576.rknn" for layer in layers],
    ]
    manifest = _hybrid_manifest_for_names(tmp_path, names, split="fc_out_only", layers=layers)
    (tmp_path / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_hybrid_artifacts(
        tmp_path,
        seq_len=320,
        target="rk3576",
        split="fc_out_only",
        layers=layers,
    )

    assert parsed["split"] == "prefill_fc_out_only"
    assert parsed["rknn_layers"] == [0, 1, 4, 5, 6]


def test_validate_moss_hybrid_artifacts_rejects_fc_out_layer_mismatch(tmp_path):
    layers = {0, 1, 4}
    _write_fc_out_hybrid_files(tmp_path, layers)
    names = [
        f"moss_embedding_prefix.s320.onnx",
        f"moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_ln2.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s320.fp16.rk3576.rknn" for layer in layers],
    ]
    manifest = _hybrid_manifest_for_names(tmp_path, names, split="fc_out_only", layers=layers)
    (tmp_path / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(MossORTArtifactError, match="rknn_layers mismatch"):
        validate_moss_hybrid_artifacts(
            tmp_path,
            seq_len=320,
            target="rk3576",
            split="fc_out_only",
            layers={0, 1, 4, 5, 6},
        )


def test_validate_moss_hybrid_artifacts_accepts_dual_root_fc_out_manifest(tmp_path):
    artifact_dir = tmp_path / "base"
    rknn_dir = tmp_path / "fc"
    artifact_dir.mkdir()
    rknn_dir.mkdir()
    layers = {0, 1, 4, 5, 6}
    for name in (
        f"moss_embedding_prefix.s320.onnx",
        f"moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
    ):
        (artifact_dir / name).write_bytes(name.encode("utf-8"))
    for name in (
        *[f"moss_block{layer}_ln2.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        (rknn_dir / name).write_bytes(name.encode("utf-8"))

    artifacts = []
    for name in (
        f"moss_embedding_prefix.s320.onnx",
        f"moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
    ):
        data = (artifact_dir / name).read_bytes()
        artifacts.append({"path": name, "required": True, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    for name in (
        *[f"moss_block{layer}_ln2.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        data = (rknn_dir / name).read_bytes()
        artifacts.append(
            {
                "root": "rknn_dir",
                "path": name,
                "required": True,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "model_id": "moss-tts-nano-hybrid-rknn",
        "target_platform": "rk3576",
        "seq_len": 320,
        "split": "prefill_fc_out_only",
        "rknn_layers": sorted(layers),
        "artifacts": artifacts,
        "quality_status": {"production_default": False},
    }
    (artifact_dir / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_hybrid_artifacts(
        artifact_dir,
        seq_len=320,
        target="rk3576",
        split="fc_out_only",
        layers=layers,
        rknn_dir=rknn_dir,
    )

    assert parsed["split"] == "prefill_fc_out_only"


def test_validate_moss_hybrid_artifacts_accepts_dual_root_ln1_cattn_manifest(tmp_path):
    artifact_dir = tmp_path / "base"
    rknn_dir = tmp_path / "ln1-cattn"
    artifact_dir.mkdir()
    rknn_dir.mkdir()
    layers = set(range(12))
    for name in (
        "moss_embedding_prefix.s320.onnx",
        "moss_final_norm.s320.onnx",
    ):
        (artifact_dir / name).write_bytes(name.encode("utf-8"))
    for name in (
        *[f"moss_block{layer}_attn_after_cattn.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_ln1_cattn.s320.fp16.rk3576.rknn" for layer in layers],
        *[f"moss_block{layer}_ln2_mlp.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        (rknn_dir / name).write_bytes(name.encode("utf-8"))

    artifacts = []
    for name in ("moss_embedding_prefix.s320.onnx", "moss_final_norm.s320.onnx"):
        data = (artifact_dir / name).read_bytes()
        artifacts.append({"path": name, "required": True, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    for name in (
        *[f"moss_block{layer}_attn_after_cattn.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_ln1_cattn.s320.fp16.rk3576.rknn" for layer in layers],
        *[f"moss_block{layer}_ln2_mlp.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        data = (rknn_dir / name).read_bytes()
        artifacts.append(
            {
                "root": "rknn_dir",
                "path": name,
                "required": True,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "model_id": "moss-tts-nano-hybrid-rknn",
        "target_platform": "rk3576",
        "seq_len": 320,
        "split": "prefill_ln1_cattn",
        "rknn_layers": sorted(layers),
        "artifacts": artifacts,
        "quality_status": {"production_default": False},
    }
    (artifact_dir / "moss-hybrid-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    parsed = validate_moss_hybrid_artifacts(
        artifact_dir,
        seq_len=320,
        target="rk3576",
        split="ln1_cattn",
        layers=layers,
        rknn_dir=rknn_dir,
    )

    assert parsed["split"] == "prefill_ln1_cattn"


def test_mlp_only_artifact_contract_includes_ln2_and_mlp_files():
    artifacts = default_moss_hybrid_mlp_only_artifacts(seq_len=320, target="rk3576")

    assert "moss_block0_ln2.s320.onnx" in artifacts
    assert "moss_block0_mlp.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block11_ln2.s320.onnx" in artifacts
    assert "moss_block11_mlp.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block0_ln2_mlp.s320.fp16.rk3576.rknn" not in artifacts


def test_fc_out_artifact_contract_names():
    artifacts = default_moss_hybrid_fc_split_artifacts(seq_len=320, target="rk3576", split="fc_out_only")

    assert "moss_block0_ln2.s320.onnx" in artifacts
    assert "moss_block0_fc_in_act.s320.onnx" in artifacts
    assert "moss_block0_fc_out.s320.fp16.rk3576.rknn" in artifacts


def test_ln1_cattn_artifact_contract_names():
    artifacts = default_moss_hybrid_ln1_cattn_artifacts(seq_len=320, target="rk3576")

    assert "moss_block0_attn_after_cattn.s320.onnx" in artifacts
    assert "moss_block0_ln1_cattn.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block0_ln2_mlp.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block0_attn_residual.s320.onnx" not in artifacts


def test_moss_ort_config_maps_hybrid_manifest_env(monkeypatch):
    monkeypatch.delenv("MOSS_ORT_HYBRID_MANIFEST", raising=False)

    _apply_tts_env(
        {
            "backend": "moss_ort",
            "model_dir": "/models/moss",
            "hybrid_manifest": "moss-hybrid-manifest.json",
        }
    )

    assert __import__("os").environ["MOSS_ORT_HYBRID_MANIFEST"] == "moss-hybrid-manifest.json"

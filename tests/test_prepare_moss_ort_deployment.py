"""Unit checks for canonical MOSS ORT deployment preparation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "prepare_moss_ort_deployment.py"
    spec = importlib.util.spec_from_file_location("prepare_moss_ort_deployment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_valid_bundle(root: Path) -> None:
    artifacts = [
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "codec_browser_onnx_meta.json",
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
        "moss_audio_tokenizer_decode_shared.data",
    ]
    for name in artifacts:
        (root / name).write_bytes(b"x")
    manifest_items = []
    for name in artifacts:
        path = root / name
        manifest_items.append({"path": name, "required": True, "size_bytes": path.stat().st_size})
    (root / "moss-ort-manifest.json").write_text(
        json.dumps(
            {
                "model_id": "moss-tts-nano-onnx",
                "target_platform": "rk3576",
                "sample_rate": 48000,
                "channels": 2,
                "streaming_required": True,
                "artifacts": manifest_items,
            }
        ),
        encoding="utf-8",
    )


def _opt_test_destination(tmp_path: Path) -> Path:
    return Path("/opt/tts/models") / f"moss-tts-nano-onnx-test-{tmp_path.name}"


def test_prepare_moss_ort_deployment_defaults_to_dry_run(tmp_path):
    module = _load_module()
    source = tmp_path / "bundle"
    source.mkdir()
    _write_valid_bundle(source)
    destination = _opt_test_destination(tmp_path)

    report = module.prepare_deployment(
        source=source,
        destination=destination,
    )

    assert report["passed"] is True
    assert report["execute"] is False
    assert report["deployed"] is False
    assert report["source_manifest"]["required_artifacts"] == 11
    assert all(result["executed"] is False for result in report["results"])
    assert report["commands"][-1] == ["ln", "-s", str(source.resolve()), str(destination)]


def test_prepare_moss_ort_deployment_rejects_missing_source(tmp_path):
    module = _load_module()

    report = module.prepare_deployment(
        source=tmp_path / "missing",
        destination=Path("/opt/tts/models/moss-tts-nano-onnx"),
    )

    assert report["passed"] is False
    assert any("source artifact validation failed" in error for error in report["errors"])


def test_prepare_moss_ort_deployment_execute_requires_confirmation(tmp_path):
    module = _load_module()
    source = tmp_path / "bundle"
    source.mkdir()
    _write_valid_bundle(source)
    destination = _opt_test_destination(tmp_path)

    report = module.prepare_deployment(
        source=source,
        destination=destination,
        execute=True,
        confirm="",
    )

    assert report["passed"] is False
    assert any("confirm" in error for error in report["errors"])
    assert all(result["executed"] is False for result in report["results"])


def test_prepare_moss_ort_deployment_rejects_existing_wrong_destination(tmp_path):
    module = _load_module()
    source = tmp_path / "bundle"
    source.mkdir()
    _write_valid_bundle(source)
    destination = tmp_path / "moss-tts-nano-onnx"
    destination.write_text("occupied", encoding="utf-8")

    report = module.prepare_deployment(source=source, destination=destination)

    assert report["passed"] is False
    assert any("destination already exists" in error for error in report["errors"])
    assert any("destination must be under /opt/tts/models" in error for error in report["errors"])

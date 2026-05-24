from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "assess_moss_rkllm_reuse.py"
    spec = importlib.util.spec_from_file_location("assess_moss_rkllm_reuse", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_moss_bundle(root: Path) -> None:
    for name in (
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "tokenizer.model",
    ):
        (root / name).write_bytes(b"x")
    meta = {
        "model_config": {
            "row_width": 17,
            "hidden_size": 768,
            "global_layers": 12,
            "global_heads": 12,
            "local_layers": 1,
            "vocab_size": 16384,
            "audio_codebook_sizes": [1024] * 16,
        }
    }
    (root / "tts_browser_onnx_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_moss_onnx_only_bundle_is_custom_rkllm_bridge_candidate(tmp_path):
    module = _load_module()
    _write_moss_bundle(tmp_path)

    report = module.assess(tmp_path)

    assert report["verdict"] == "custom_hf_bridge_candidate"
    assert report["official_rkllm_first"] is True
    assert report["direct_builtin_rkllm"]["ready"] is False
    assert report["custom_hf_bridge"]["candidate"] is True
    assert report["custom_hf_bridge"]["needs_onnx_weight_reconstruction"] is True
    assert "RKLLM_INPUT_EMBED" in report["custom_hf_bridge"]["runtime_api_to_reuse"]
    assert report["fallback_order"][0] == "official RKLLM custom bridge"


def test_builtin_hf_weights_can_be_direct_rkllm_export(tmp_path):
    module = _load_module()
    _write_moss_bundle(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    (tmp_path / "model.safetensors").write_bytes(b"x")

    report = module.assess(tmp_path)

    assert report["verdict"] == "direct_builtin_export_ready"
    assert report["direct_builtin_rkllm"]["ready"] is True
    assert report["direct_builtin_rkllm"]["detected_family"] == "qwen3"


def test_reports_rkllm_custom_demo_availability(tmp_path):
    module = _load_module()
    _write_moss_bundle(tmp_path)
    rkllm_repo = tmp_path / "rknn-llm"
    custom = rkllm_repo / "rkllm-toolkit" / "examples" / "custom_demo"
    custom.mkdir(parents=True)
    for name in ("configuration_custom.py", "modeling_custom.py", "config.json"):
        (custom / name).write_text("# stub\n", encoding="utf-8")

    report = module.assess(tmp_path, rkllm_repo)

    assert report["custom_hf_bridge"]["rkllm_custom_demo"]["available"] is True

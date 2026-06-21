from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_moss_rkllm_custom_model.py"
    spec = importlib.util.spec_from_file_location("prepare_moss_rkllm_custom_model", path)
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
            "audio_pad_token_id": 1024,
            "pad_token_id": 3,
            "im_start_token_id": 4,
            "im_end_token_id": 5,
            "audio_assistant_slot_token_id": 9,
        }
    }
    (root / "tts_browser_onnx_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _write_custom_demo(root: Path) -> Path:
    custom = root / "rkllm-toolkit" / "examples" / "custom_demo"
    custom.mkdir(parents=True)
    for name in (
        "configuration_custom.py",
        "modeling_custom.py",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        (custom / name).write_text("{}\n" if name.endswith(".json") else "# stub\n", encoding="utf-8")
    return custom


def test_prepare_moss_rkllm_custom_model_writes_scaffold(tmp_path):
    module = _load_module()
    model_dir = tmp_path / "moss"
    model_dir.mkdir()
    _write_moss_bundle(model_dir)
    custom = _write_custom_demo(tmp_path / "rknn-llm")
    out_dir = tmp_path / "out"

    report = module.prepare(model_dir, custom, out_dir)

    config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    mapping = json.loads((out_dir / "moss_rkllm_weight_map.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is False
    assert config["hidden_size"] == 768
    assert config["num_hidden_layers"] == 12
    assert config["moss_rkllm"]["row_width"] == 17
    assert config["moss_rkllm"]["preferred_runtime_output"] == "RKLLM_INFER_GET_LAST_HIDDEN_LAYER"
    assert mapping["ready_for_rkllm_export"] is False
    assert any(row["source"] == "core.model.transformer.h.0.attn.c_attn.bias" for row in mapping["weight_map"])
    assert (out_dir / "modeling_custom.py").exists()


def test_prepare_moss_rkllm_custom_model_dry_run_does_not_write(tmp_path):
    module = _load_module()
    model_dir = tmp_path / "moss"
    model_dir.mkdir()
    _write_moss_bundle(model_dir)
    custom = _write_custom_demo(tmp_path / "rknn-llm")
    out_dir = tmp_path / "out"

    report = module.prepare(model_dir, custom, out_dir, dry_run=True)

    assert report["dry_run"] is True
    assert report["config"]["moss_rkllm"]["audio_columns"] == list(range(1, 17))
    assert not out_dir.exists()

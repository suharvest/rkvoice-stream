from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "write_moss_rkllm_custom_code.py"
    spec = importlib.util.spec_from_file_location("write_moss_rkllm_custom_code", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_write_custom_code_updates_scaffold(tmp_path):
    module = _load_module()
    config = {
        "hidden_size": 8,
        "vocab_size": 16,
        "moss_rkllm": {
            "row_width": 17,
            "audio_codebook_sizes": [4] * 16,
            "audio_pad_token_id": 4,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    report = module.write_custom_code(tmp_path)

    updated = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert report["ready_for_hf_load_probe"] is True
    assert report["supports_row_width"] == 17
    assert updated["auto_map"]["AutoModelForCausalLM"] == "modeling_custom.CustomForCausalLM"
    assert "class CustomForCausalLM" in (tmp_path / "modeling_custom.py").read_text(encoding="utf-8")
    assert "class CustomConfig" in (tmp_path / "configuration_custom.py").read_text(encoding="utf-8")

"""Unit checks for MOSS RKNN island extraction workspace safeguards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "extract_moss_rknn_island.py"
    spec = importlib.util.spec_from_file_location("extract_moss_rknn_island", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_moss_rknn_island_imports_workspace_defaults():
    module = _load_module()

    assert module.DEFAULT_RKNN_WORKSPACE == Path("/mnt/rknn-workspace/moss-rknn-workspace")
    assert module.DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB >= 2048


def test_extract_moss_rknn_island_has_narrow_attention_qkv_projection_preset():
    module = _load_module()

    assert "cattn" in module.PRESET_CHOICES
    assert "ln1_cattn" in module.PRESET_CHOICES

    first = module._preset_spec("cattn", 0)
    later = module._preset_spec("cattn", 3)
    fused = module._preset_spec("ln1_cattn", 4)

    assert first["inputs"] == ["/ln_1/LayerNormalization_output_0"]
    assert first["outputs"] == ["/c_attn/Add_output_0"]
    assert first["shape"] == [1, 32, 768]
    assert first["case"] == "island_float"

    assert later["inputs"] == ["/ln_1_3/LayerNormalization_output_0"]
    assert later["outputs"] == ["/c_attn_3/Add_output_0"]

    assert fused["inputs"] == ["/Mul_40_output_0"]
    assert fused["outputs"] == ["/c_attn_4/Add_output_0"]
    assert fused["shape"] == [1, 32, 768]
    assert fused["case"] == "island_float"


def test_extract_moss_rknn_island_has_cattn_cpu_boundaries():
    module = _load_module()

    ln1 = module._preset_spec("ln1", 0)
    suffix = module._preset_spec("attn_after_cattn", 2)

    assert ln1["inputs"] == ["/Mul_16_output_0"]
    assert ln1["outputs"] == ["/ln_1/LayerNormalization_output_0"]
    assert ln1["case"] == "onnx_cpu"

    assert suffix["inputs"] == [
        "/c_attn_2/Add_output_0",
        "/Mul_28_output_0",
        "attention_mask",
    ]
    assert suffix["outputs"] == ["/Add_29_output_0", "present_key_2", "present_value_2"]
    assert suffix["input_shapes"]["/c_attn_2/Add_output_0"] == [1, 32, 2304]


def test_extract_moss_rknn_island_blocks_before_writing_when_workspace_preflight_fails(monkeypatch, tmp_path):
    module = _load_module()
    calls = []

    def _fake_check(**kwargs):
        calls.append(kwargs)
        return {"passed": False, "errors": ["workspace does not exist"]}

    monkeypatch.setattr(module, "check_output_workspace", _fake_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "extract_moss_rknn_island.py",
            "--onnx",
            str(tmp_path / "missing.onnx"),
            "--out-dir",
            str(tmp_path / "out"),
            "--preset",
            "ln2_mlp",
            "--require-rknn-workspace",
            "--rknn-workspace",
            str(tmp_path / "workspace"),
            "--rknn-workspace-min-free-mb",
            "0",
        ],
    )

    try:
        module.main()
    except RuntimeError as exc:
        error = str(exc)
    else:
        raise AssertionError("workspace preflight failure must stop island extraction")

    assert "RKNN workspace preflight failed" in error
    assert calls[0]["out_dir"] == tmp_path / "out"
    assert not (tmp_path / "out").exists()

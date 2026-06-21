from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_moss_onnx_to_rkllm_state.py"
    spec = importlib.util.spec_from_file_location("convert_moss_onnx_to_rkllm_state", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_onnx(path: Path, hidden: int = 4, inter: int = 8, layers: int = 1) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["x"], ["y"])
    inits = [
        numpy_helper.from_array(np.zeros((16, hidden), dtype=np.float32), "core.model.transformer.wte.weight"),
        numpy_helper.from_array(np.zeros((1024, hidden), dtype=np.float32), "core.model.audio_embeddings.0.weight"),
        numpy_helper.from_array(np.zeros((hidden,), dtype=np.float32), "core.model.transformer.h.0.ln_1.bias"),
        numpy_helper.from_array(np.arange(hidden * hidden * 3, dtype=np.float32).reshape(hidden, hidden * 3), "onnx::MatMul_1"),
        numpy_helper.from_array(np.arange(hidden * 3, dtype=np.float32), "core.model.transformer.h.0.attn.c_attn.bias"),
        numpy_helper.from_array(np.zeros((hidden, hidden), dtype=np.float32), "onnx::MatMul_2"),
        numpy_helper.from_array(np.zeros((hidden,), dtype=np.float32), "core.model.transformer.h.0.attn.c_proj.bias"),
        numpy_helper.from_array(np.zeros((hidden,), dtype=np.float32), "core.model.transformer.h.0.ln_2.bias"),
        numpy_helper.from_array(np.zeros((hidden, inter), dtype=np.float32), "onnx::MatMul_3"),
        numpy_helper.from_array(np.zeros((inter,), dtype=np.float32), "core.model.transformer.h.0.mlp.fc_in.bias"),
        numpy_helper.from_array(np.zeros((inter, hidden), dtype=np.float32), "onnx::MatMul_4"),
        numpy_helper.from_array(np.zeros((hidden,), dtype=np.float32), "core.model.transformer.h.0.mlp.fc_out.bias"),
        numpy_helper.from_array(np.zeros((hidden,), dtype=np.float32), "core.model.transformer.ln_f.bias"),
    ]
    for idx in range(1, 16):
        inits.append(numpy_helper.from_array(np.zeros((1024, hidden), dtype=np.float32), f"core.model.audio_embeddings.{idx}.weight"))
    graph = helper.make_graph([node], "g", [x], [y], initializer=inits)
    onnx.save(helper.make_model(graph), path)


def _write_scaffold(root: Path, hidden: int = 4, inter: int = 8) -> None:
    root.mkdir()
    config = {"hidden_size": hidden, "num_hidden_layers": 1}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    mapping = {
        "anonymous_onnx_weight_sources_complete": True,
        "anonymous_onnx_weight_sources": [
            {"layer": 0, "logical": "attn.c_attn.weight", "source": "onnx::MatMul_1", "shape_matches": True},
            {"layer": 0, "logical": "attn.c_proj.weight", "source": "onnx::MatMul_2", "shape_matches": True},
            {"layer": 0, "logical": "mlp.fc_in.weight", "source": "onnx::MatMul_3", "shape_matches": True},
            {"layer": 0, "logical": "mlp.fc_out.weight", "source": "onnx::MatMul_4", "shape_matches": True},
        ],
    }
    (root / "moss_rkllm_weight_map.json").write_text(json.dumps(mapping), encoding="utf-8")


def test_convert_state_splits_qkv_and_transposes_linear_weights(tmp_path):
    module = _load_module()
    model_dir = tmp_path / "moss"
    model_dir.mkdir()
    _write_onnx(model_dir / "moss_tts_prefill.onnx")
    scaffold = tmp_path / "scaffold"
    _write_scaffold(scaffold)

    report = module.convert_state(model_dir, scaffold, write=True)

    assert report["missing_count"] == 0
    assert report["tensor_count"] > 0
    assert (scaffold / "model.safetensors").exists()
    from safetensors.torch import load_file

    tensors = load_file(str(scaffold / "model.safetensors"))
    assert tuple(tensors["model.layers.0.self_attn.q_proj.weight"].shape) == (4, 4)
    assert tuple(tensors["model.layers.0.mlp.fc_in.weight"].shape) == (8, 4)
    assert tuple(tensors["model.layers.0.mlp.fc_out.weight"].shape) == (4, 8)
    assert "model.layers.0.input_layernorm.weight" in tensors

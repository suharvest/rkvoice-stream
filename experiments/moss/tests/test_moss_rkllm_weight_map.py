from __future__ import annotations

import importlib.util
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "inspect_moss_rkllm_weight_map.py"
    spec = importlib.util.spec_from_file_location("inspect_moss_rkllm_weight_map", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_onnx(path: Path, initializers) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [x], [y], initializer=initializers)
    model = helper.make_model(graph)
    onnx.save(model, path)


def test_inspect_detects_transformer_weight_classes(tmp_path):
    module = _load_module()
    init = [
        helper.make_tensor("wte.weight", TensorProto.FLOAT, [32, 8], [0.0] * 256),
        helper.make_tensor("blocks.0.c_attn.weight", TensorProto.FLOAT, [8, 24], [0.0] * 192),
        helper.make_tensor("blocks.0.c_proj.weight", TensorProto.FLOAT, [8, 8], [0.0] * 64),
        helper.make_tensor("blocks.0.fc_in.weight", TensorProto.FLOAT, [8, 32], [0.0] * 256),
        helper.make_tensor("blocks.0.fc_out.weight", TensorProto.FLOAT, [32, 8], [0.0] * 256),
        helper.make_tensor("ln_f.weight", TensorProto.FLOAT, [8], [1.0] * 8),
    ]
    for name in ("moss_tts_prefill.onnx", "moss_tts_decode_step.onnx", "moss_tts_local_fixed_sampled_frame.onnx"):
        _write_onnx(tmp_path / name, init)

    report = module.inspect(tmp_path)

    assert report["rkllm_bridge_feasibility"]["can_reconstruct_from_onnx_initializers"] is True
    counts = report["inventories"]["prefill"]["class_counts"]
    assert counts["embedding"] == 1
    assert counts["attention_qkv"] == 1
    assert counts["mlp_out"] == 1

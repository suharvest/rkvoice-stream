from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "build_moss_codec_suffix_islands.py"
    spec = importlib.util.spec_from_file_location("build_moss_codec_suffix_islands", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_suffix_graph() -> onnx.ModelProto:
    nodes = []
    inputs = [
        helper.make_tensor_value_info("attn0", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("res0", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("attn1", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("res1", TensorProto.FLOAT, [1, 4, 256]),
    ]
    initializers = [
        helper.make_tensor("w0", TensorProto.FLOAT, [256, 256], [0.0] * (256 * 256)),
        helper.make_tensor("w1", TensorProto.FLOAT, [256, 256], [0.0] * (256 * 256)),
        helper.make_tensor("scale0", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("scale1", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("ffn0", TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor("ffn1", TensorProto.FLOAT, [1], [1.0]),
    ]
    for layer, out_proj_index in enumerate((16, 17)):
        attn = f"attn{layer}"
        res = f"res{layer}"
        suffix = "" if layer == 0 else f"_{layer}"
        nodes.extend(
            [
                helper.make_node("MatMul", [attn, f"w{layer}"], [f"out{layer}"], name=f"/out_proj_{out_proj_index}/MatMul"),
                helper.make_node("Mul", [f"scale{layer}", f"out{layer}"], [f"ls1_{layer}"], name=f"/layer_scale_1{suffix}/Mul"),
                helper.make_node("Add", [res, f"ls1_{layer}"], [f"add_mid_{layer}"], name=f"/Add_mid{suffix}"),
                helper.make_node("Identity", [f"add_mid_{layer}"], [f"ffn_mid_{layer}"], name=f"/ffn_stub{suffix}"),
                helper.make_node("Mul", [f"ffn{layer}", f"ffn_mid_{layer}"], [f"ls2_{layer}"], name=f"/layer_scale_2{suffix}/Mul"),
                helper.make_node("Add", [f"add_mid_{layer}", f"ls2_{layer}"], [f"layer_out_{layer}"], name=f"/Add_out{suffix}"),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "suffix_scan",
        inputs,
        [helper.make_tensor_value_info("layer_out_1", TensorProto.FLOAT, [1, 4, 256])],
        initializers,
    )
    return helper.make_model(graph)


def test_discovers_codec_suffix_outproj_ffn_boundaries():
    module = _load_module()

    specs = module.discover_codec_suffix_specs(_make_suffix_graph())

    assert [spec.layer for spec in specs] == [0, 1]
    assert specs[0].out_proj_node == "/out_proj_16/MatMul"
    assert specs[0].inputs == ["attn0", "res0"]
    assert specs[0].output == "layer_out_0"
    assert specs[1].out_proj_node == "/out_proj_17/MatMul"
    assert specs[1].inputs == ["attn1", "res1"]
    assert specs[1].output == "layer_out_1"


def test_parse_codec_suffix_layers_range():
    module = _load_module()

    assert module._parse_layers("0,2-4,7") == [0, 2, 3, 4, 7]

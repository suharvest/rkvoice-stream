from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "build_moss_codec_middle_bridges.py"
    spec = importlib.util.spec_from_file_location("build_moss_codec_middle_bridges", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_middle_graph() -> onnx.ModelProto:
    nodes = []
    inputs = [
        helper.make_tensor_value_info("hidden0", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("hidden1", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("attn_offset_0", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("attn_cached_keys_0", TensorProto.FLOAT, [1, 4, 500, 64]),
        helper.make_tensor_value_info("attn_cached_values_0", TensorProto.FLOAT, [1, 4, 500, 64]),
        helper.make_tensor_value_info("attn_cached_positions_0", TensorProto.INT64, [1, 500]),
        helper.make_tensor_value_info("attn_offset_1", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("attn_cached_keys_1", TensorProto.FLOAT, [1, 4, 500, 64]),
        helper.make_tensor_value_info("attn_cached_values_1", TensorProto.FLOAT, [1, 4, 500, 64]),
        helper.make_tensor_value_info("attn_cached_positions_1", TensorProto.INT64, [1, 500]),
    ]
    initializers = [
        helper.make_tensor("norm_w0", TensorProto.FLOAT, [256], [1.0] * 256),
        helper.make_tensor("norm_b0", TensorProto.FLOAT, [256], [0.0] * 256),
        helper.make_tensor("norm_w1", TensorProto.FLOAT, [256], [1.0] * 256),
        helper.make_tensor("norm_b1", TensorProto.FLOAT, [256], [0.0] * 256),
        helper.make_tensor("proj_w0", TensorProto.FLOAT, [256, 768], [0.0] * (256 * 768)),
        helper.make_tensor("proj_w1", TensorProto.FLOAT, [256, 768], [0.0] * (256 * 768)),
        helper.make_tensor("out_w0", TensorProto.FLOAT, [256, 256], [0.0] * (256 * 256)),
        helper.make_tensor("out_w1", TensorProto.FLOAT, [256, 256], [0.0] * (256 * 256)),
        helper.make_tensor("shape0", TensorProto.INT64, [5], [1, 4, 3, 4, 64]),
        helper.make_tensor("shape1", TensorProto.INT64, [5], [1, 4, 3, 4, 64]),
        helper.make_tensor("one0", TensorProto.INT64, [1], [1]),
        helper.make_tensor("one1", TensorProto.INT64, [1], [1]),
    ]
    for layer, suffix in enumerate(("", "_1")):
        nodes.extend(
            [
                helper.make_node(
                    "LayerNormalization",
                    [f"hidden{layer}", f"norm_w{layer}", f"norm_b{layer}"],
                    [f"norm{layer}"],
                    name=f"/norm1{suffix}/LayerNormalization",
                ),
                helper.make_node(
                    "MatMul",
                    [f"norm{layer}", f"proj_w{layer}"],
                    [f"qkv_flat{layer}"],
                    name=f"/in_proj{suffix}/MatMul",
                ),
                helper.make_node("Reshape", [f"qkv_flat{layer}", f"shape{layer}"], [f"qkv_rs{layer}"], name=f"/Reshape{suffix}"),
                helper.make_node("Transpose", [f"qkv_rs{layer}"], [f"qkv{layer}"], name=f"/Transpose{suffix}", perm=[2, 0, 3, 1, 4]),
                helper.make_node("Identity", [f"qkv{layer}"], [f"attn{layer}"], name=f"/attn_stub{suffix}"),
                helper.make_node("Identity", [f"attn_cached_keys_{layer}"], [f"attn_cached_keys_out_{layer}"], name=f"/key_stub{suffix}"),
                helper.make_node("Identity", [f"attn_cached_values_{layer}"], [f"attn_cached_values_out_{layer}"], name=f"/value_stub{suffix}"),
                helper.make_node("Identity", [f"attn_cached_positions_{layer}"], [f"attn_cached_positions_out_{layer}"], name=f"/pos_stub{suffix}"),
                helper.make_node("Add", [f"attn_offset_{layer}", f"one{layer}"], [f"attn_offset_out_{layer}"], name=f"/offset_stub{suffix}"),
                helper.make_node("MatMul", [f"attn{layer}", f"out_w{layer}"], [f"out{layer}"], name=f"/out_proj_{16 + layer}/MatMul"),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "middle_scan",
        inputs,
        [helper.make_tensor_value_info("out1", TensorProto.FLOAT, [3, 1, 4, 4, 256])],
        initializers,
    )
    return helper.make_model(graph)


def test_discovers_codec_middle_attention_boundaries():
    module = _load_module()

    specs = module.discover_codec_middle_specs(_make_middle_graph())

    assert [spec.layer for spec in specs] == [0, 1]
    assert specs[0].front_output == "qkv0"
    assert specs[0].inputs == [
        "qkv0",
        "attn_offset_0",
        "attn_cached_keys_0",
        "attn_cached_values_0",
        "attn_cached_positions_0",
    ]
    assert specs[0].outputs == [
        "attn0",
        "attn_offset_out_0",
        "attn_cached_keys_out_0",
        "attn_cached_values_out_0",
        "attn_cached_positions_out_0",
    ]
    assert specs[1].front_output == "qkv1"
    assert specs[1].outputs[0] == "attn1"


def test_parse_codec_middle_layers_range():
    module = _load_module()

    assert module._parse_layers("0,2-4,7") == [0, 2, 3, 4, 7]

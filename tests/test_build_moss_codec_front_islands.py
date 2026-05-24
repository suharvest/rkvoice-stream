from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "build_moss_codec_front_islands.py"
    spec = importlib.util.spec_from_file_location("build_moss_codec_front_islands", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_front_graph() -> onnx.ModelProto:
    nodes = []
    inputs = [
        helper.make_tensor_value_info("hidden0", TensorProto.FLOAT, [1, 4, 256]),
        helper.make_tensor_value_info("hidden1", TensorProto.FLOAT, [1, 4, 256]),
    ]
    initializers = [
        helper.make_tensor("norm_w0", TensorProto.FLOAT, [256], [1.0] * 256),
        helper.make_tensor("norm_b0", TensorProto.FLOAT, [256], [0.0] * 256),
        helper.make_tensor("norm_w1", TensorProto.FLOAT, [256], [1.0] * 256),
        helper.make_tensor("norm_b1", TensorProto.FLOAT, [256], [0.0] * 256),
        helper.make_tensor("proj_w0", TensorProto.FLOAT, [256, 768], [0.0] * (256 * 768)),
        helper.make_tensor("proj_w1", TensorProto.FLOAT, [256, 768], [0.0] * (256 * 768)),
        helper.make_tensor("shape0", TensorProto.INT64, [5], [1, 4, 3, 4, 64]),
        helper.make_tensor("shape1", TensorProto.INT64, [5], [1, 4, 3, 4, 64]),
    ]
    for layer, suffix in enumerate(("", "_1")):
        hidden = f"hidden{layer}"
        nodes.extend(
            [
                helper.make_node(
                    "LayerNormalization",
                    [hidden, f"norm_w{layer}", f"norm_b{layer}"],
                    [f"norm{layer}"],
                    name=f"/norm1{suffix}/LayerNormalization",
                ),
                helper.make_node(
                    "MatMul",
                    [f"norm{layer}", f"proj_w{layer}"],
                    [f"qkv_flat{layer}"],
                    name=f"/in_proj{suffix}/MatMul",
                ),
                helper.make_node(
                    "Reshape",
                    [f"qkv_flat{layer}", f"shape{layer}"],
                    [f"qkv_reshape{layer}"],
                    name=f"/Reshape{suffix}",
                ),
                helper.make_node(
                    "Transpose",
                    [f"qkv_reshape{layer}"],
                    [f"qkv{layer}"],
                    name=f"/Transpose{suffix}",
                    perm=[2, 0, 3, 1, 4],
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "front_scan",
        inputs,
        [helper.make_tensor_value_info("qkv1", TensorProto.FLOAT, [3, 1, 4, 4, 64])],
        initializers,
    )
    return helper.make_model(graph)


def test_discovers_codec_front_norm1_qkv_boundaries():
    module = _load_module()

    specs = module.discover_codec_front_specs(_make_front_graph())

    assert [spec.layer for spec in specs] == [0, 1]
    assert specs[0].in_proj_node == "/in_proj/MatMul"
    assert specs[0].input == "hidden0"
    assert specs[0].output == "qkv0"
    assert specs[1].in_proj_node == "/in_proj_1/MatMul"
    assert specs[1].input == "hidden1"
    assert specs[1].output == "qkv1"


def test_parse_codec_front_layers_range():
    module = _load_module()

    assert module._parse_layers("0,2-4,7") == [0, 2, 3, 4, 7]

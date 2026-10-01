from pathlib import Path
import sys

import onnx
import pytest
from onnx import TensorProto, helper

from extract_piper_dp_conditioner import extract


def _model(path: Path, unsafe: bool = False, no_mask: bool = False):
    inputs = [helper.make_tensor_value_info("tokens", TensorProto.FLOAT, [1, 192, 4]),
              helper.make_tensor_value_info("mask_src", TensorProto.FLOAT, [1, 1, 4])]
    mask_name = "mask_src" if no_mask else "/enc_p/Cast_1_output_0"
    nodes = [
        helper.make_node("Identity", ["tokens"], ["/enc_p/encoder/Mul_2_output_0"], name="/enc_p/encoder/Mul_2"),
        *([] if no_mask else [helper.make_node("Identity", ["mask_src"], ["/enc_p/Cast_1_output_0"], name="/enc_p/Cast_1")]),
        helper.make_node("Conv", ["/enc_p/encoder/Mul_2_output_0", "w"], ["pre"], name="/dp/pre/Conv"),
        helper.make_node("Mul", ["pre", mask_name], ["cond"], name="/dp/convs/Mul"),
        helper.make_node("Identity", ["cond"], ["projected"], name="/dp/proj/Identity"),
    ]
    # The conditioning tensor reaches the flow through a post-convs projection.
    flow_input = "projected"
    if unsafe:
        nodes.append(helper.make_node("RandomNormalLike", ["cond"], ["random"], name="/dp/RandomNormalLike"))
        flow_input = "random"
    nodes.append(helper.make_node("RandomNormalLike", ["mask_src"], ["latent"], name="/dp/latent"))
    nodes.append(helper.make_node("Add", [flow_input, "latent"], ["flow_out"], name="/dp/flows.7/Split"))
    graph = helper.make_graph(nodes, "dp", inputs,
                              [helper.make_tensor_value_info("flow_out", TensorProto.FLOAT, [1, 192, 4])])
    graph.initializer.append(helper.make_tensor("w", TensorProto.FLOAT, [192, 192, 1], [0.0] * (192 * 192)))
    graph.value_info.extend([
        helper.make_tensor_value_info("pre", TensorProto.FLOAT, [1, 192, 4]),
        helper.make_tensor_value_info("cond", TensorProto.FLOAT, [1, 192, 4]),
        helper.make_tensor_value_info("projected", TensorProto.FLOAT, [1, 192, 4]),
    ])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)]), path)


def test_extracts_only_conditioner_closure(tmp_path):
    source = tmp_path / "model.onnx"; _model(source)
    manifest = extract(source, tmp_path / "out")
    sub = onnx.load(tmp_path / "out/dp_conditioner.onnx")
    assert manifest["conditioning_output"] == "projected"
    assert {x["name"] for x in manifest["inputs"]} == {"/enc_p/encoder/Mul_2_output_0", "/enc_p/Cast_1_output_0"}
    assert all(x["dtype"] == TensorProto.FLOAT for x in manifest["inputs"])
    assert {n.name for n in sub.graph.node} == {"/dp/pre/Conv", "/dp/convs/Mul", "/dp/proj/Identity"}
    assert all("/dp/flows." not in n.name for n in sub.graph.node)


def test_rejects_random_dependency(tmp_path):
    source = tmp_path / "model.onnx"; _model(source, unsafe=True)
    with pytest.raises(RuntimeError, match="unsafe node"):
        extract(source, tmp_path / "out")


def test_rejects_missing_mask_boundary(tmp_path):
    source = tmp_path / "model.onnx"; _model(source, no_mask=True)
    with pytest.raises(RuntimeError, match="missing explicit hidden/mask boundaries"):
        extract(source, tmp_path / "out")

"""Unit checks for MOSS RKNN conversion workspace safeguards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "convert_moss_rknn.py"
    spec = importlib.util.spec_from_file_location("convert_moss_rknn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_convert_moss_rknn_workspace_check_rejects_out_dir_outside_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = tmp_path / "home" / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": True, "errors": [], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=0)

    assert report["passed"] is False
    assert any("out_dir must be under RKNN workspace" in error for error in report["errors"])


def test_convert_moss_rknn_workspace_check_rejects_unprepared_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = workspace / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": False, "errors": ["workspace does not exist"], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=0)

    assert report["passed"] is False
    assert "workspace does not exist" in report["errors"]


def test_convert_moss_rknn_workspace_check_accepts_prepared_workspace(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    out_dir = workspace / "moss-rknn"

    monkeypatch.setattr(
        module,
        "verify_rknn_workspace",
        lambda **kwargs: {"passed": True, "errors": [], "workspace": str(kwargs["workspace"])},
    )

    report = module.check_output_workspace(out_dir=out_dir, workspace=workspace, min_free_mb=2048)

    assert report["passed"] is True
    assert report["errors"] == []


def test_convert_moss_rknn_parses_disable_rule_list():
    module = _load_module()

    assert module.parse_string_list("") == []
    assert module.parse_string_list("merge_conv_channel_inner_perm") == ["merge_conv_channel_inner_perm"]
    assert module.parse_string_list("a, b,,c ") == ["a", "b", "c"]


def test_prepare_codec_onnx_rewrites_xor_false_as_identity(tmp_path):
    module = _load_module()
    input_path = tmp_path / "codec.onnx"
    output_path = tmp_path / "codec.fixed.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["false_const"],
                name="/Constant_455",
                value=helper.make_tensor("false", TensorProto.BOOL, [], [False]),
            ),
            helper.make_node("Xor", inputs=["mask", "false_const"], outputs=["out"], name="/Xor"),
        ],
        "codec_xor_false",
        [helper.make_tensor_value_info("mask", TensorProto.BOOL, [1])],
        [helper.make_tensor_value_info("out", TensorProto.BOOL, [1])],
    )
    model = helper.make_model(graph)
    onnx.save_model(model, str(input_path))

    module.prepare_codec_onnx(input_path, output_path)

    fixed = onnx.load(str(output_path), load_external_data=False)
    rewritten = [node for node in fixed.graph.node if node.output and node.output[0] == "out"][0]
    assert rewritten.op_type == "Identity"
    assert list(rewritten.input) == ["mask"]


def test_prepare_codec_onnx_moves_input_cast_to_int64_to_input_dtype(tmp_path):
    module = _load_module()
    input_path = tmp_path / "codec.onnx"
    output_path = tmp_path / "codec.fixed.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Cast", inputs=["attn_offset_0"], outputs=["offset_i64"], name="/Cast_53", to=TensorProto.INT64),
            helper.make_node("Add", inputs=["offset_i64", "offset_const"], outputs=["out"], name="/Add"),
        ],
        "codec_input_cast_i64",
        [helper.make_tensor_value_info("attn_offset_0", TensorProto.INT32, [1])],
        [helper.make_tensor_value_info("out", TensorProto.INT64, [1])],
        [helper.make_tensor("offset_const", TensorProto.INT64, [1], [0])],
    )
    model = helper.make_model(graph)
    onnx.save_model(model, str(input_path))

    module.prepare_codec_onnx(input_path, output_path)

    fixed = onnx.load(str(output_path), load_external_data=False)
    fixed_input = fixed.graph.input[0].type.tensor_type
    rewritten = [node for node in fixed.graph.node if node.output and node.output[0] == "offset_i64"][0]
    assert fixed_input.elem_type == TensorProto.INT64
    assert rewritten.op_type == "Identity"
    assert list(rewritten.input) == ["attn_offset_0"]


def test_prepare_codec_onnx_does_not_rewrite_input_cast_to_float(tmp_path):
    module = _load_module()
    input_path = tmp_path / "codec.onnx"
    output_path = tmp_path / "codec.fixed.onnx"
    graph = helper.make_graph(
        [helper.make_node("Cast", inputs=["attn_offset_0"], outputs=["offset_f32"], name="/rope/Cast_4", to=TensorProto.FLOAT)],
        "codec_input_cast_float",
        [helper.make_tensor_value_info("attn_offset_0", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("offset_f32", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph)
    onnx.save_model(model, str(input_path))

    module.prepare_codec_onnx(input_path, output_path)

    fixed = onnx.load(str(output_path), load_external_data=False)
    kept = fixed.graph.node[0]
    assert kept.op_type == "Cast"
    assert kept.attribute[0].i == TensorProto.FLOAT


def test_convert_moss_rknn_blocks_before_missing_onnx_check_when_workspace_preflight_fails(
    monkeypatch, tmp_path
):
    module = _load_module()
    calls = []

    def _fake_check(**kwargs):
        calls.append(kwargs)
        return {"passed": False, "errors": ["workspace does not exist"]}

    out_dir = tmp_path / "workspace" / "moss-rknn"
    monkeypatch.setattr(module, "check_output_workspace", _fake_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_moss_rknn.py",
            "--onnx-bundle",
            str(tmp_path / "missing-bundle"),
            "--out-dir",
            str(out_dir),
            "--only",
            "sampler",
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
        raise AssertionError("workspace preflight failure must stop RKNN conversion")

    assert "RKNN workspace preflight failed" in error
    assert "workspace does not exist" in error
    assert calls[0]["out_dir"] == out_dir
    assert calls[0]["workspace"] == tmp_path / "workspace"
    assert not out_dir.exists()

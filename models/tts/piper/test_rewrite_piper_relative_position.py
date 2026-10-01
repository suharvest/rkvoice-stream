from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from rewrite_piper_relative_position import rewrite


def _numpy_chain(p):
    b, h, s, _ = p.shape
    x = np.pad(p, ((0, 0), (0, 0), (0, 0), (0, s - 1)))
    x = np.pad(x.reshape(b, h, -1), ((0, 0), (0, 0), (s, 0)))
    return x.reshape(b, h, s, 2 * s)[..., 1:]


def _numpy_gather(p):
    b, h, s, _ = p.shape
    q = np.zeros((b, h, s, 2 * s - 1), dtype=p.dtype)
    for i in range(s):
        for k in range(2 * s - 1):
            j = k + i - (s - 1)
            if 0 <= j < s:
                q[:, :, i, k] = p[:, :, i, j]
    return q


@pytest.mark.parametrize("s", [2, 3, 4, 128])
def test_relative_math_equivalence(s):
    p = np.random.default_rng(s).normal(size=(2, 3, s, s)).astype(np.float32)
    a, b = _numpy_chain(p), _numpy_gather(p)
    assert a.shape == b.shape
    assert np.array_equal(a, b)
    assert np.max(np.abs(a - b)) == 0


def _make_model(path: Path, layers: int = 6, s: int = 3):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, s, s])
    outputs = []
    nodes = []
    initializers = []
    for layer in range(layers):
        base = f"/enc_p/encoder/attn_layers.{layer}"
        soft = f"{base}/Softmax_output_0"
        p0 = f"{base}/Pad_output_0"; flat = f"{base}/Flatten_output_0"
        p1 = f"{base}/Pad_1_output_0"; resh = f"{base}/Reshape_output_0"
        target = f"{base}/Slice_8_output_0"
        nodes.append(helper.make_node("Softmax", ["x"], [soft], name=f"{base}/Softmax", axis=-1))
        initializers.extend([
            numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, s - 1], np.int64), name=f"p{layer}_pads"),
            numpy_helper.from_array(np.array([0, 0, s, 0, 0, 0], np.int64), name=f"p{layer}_pads_left"),
            numpy_helper.from_array(np.array([1, 2, s, 2 * s], np.int64), name=f"p{layer}_shape"),
            numpy_helper.from_array(np.array([1, 2, -1], np.int64), name=f"p{layer}_shape3"),
            numpy_helper.from_array(np.array([0, 0, 0, 1], np.int64), name=f"p{layer}_starts"),
            numpy_helper.from_array(np.array([1, 2, s, 2 * s], np.int64), name=f"p{layer}_ends"),
            numpy_helper.from_array(np.array([0, 1, 2, 3], np.int64), name=f"p{layer}_axes"),
        ])
        nodes.extend([
            helper.make_node("Pad", [soft, f"p{layer}_pads"], [p0], name=f"{base}/Pad"),
            helper.make_node("Reshape", [p0, f"p{layer}_shape3"], [flat], name=f"{base}/Flatten"),
            helper.make_node("Pad", [flat, f"p{layer}_pads_left"], [p1], name=f"{base}/Pad_1"),
            helper.make_node("Reshape", [p1, f"p{layer}_shape"], [resh], name=f"{base}/Reshape"),
            helper.make_node("Slice", [resh, f"p{layer}_starts", f"p{layer}_ends", f"p{layer}_axes"], [target], name=f"{base}/Slice_8"),
        ])
        outputs.append(helper.make_tensor_value_info(target, TensorProto.FLOAT, [1, 2, s, 2 * s - 1]))
    graph = helper.make_graph(nodes, "relative", [x], outputs, initializer=initializers)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)], ir_version=8), path)


def _add_matmuls(path: Path, dynamic_b: bool = False, s: int = 3):
    model = onnx.load(path)
    l = 2 * s - 1
    for layer in range(6):
        target = f"/enc_p/encoder/attn_layers.{layer}/Slice_8_output_0"
        b_name = f"B{layer}"
        b = np.zeros((1, 1, l, 2), dtype=np.float32)
        b[0, 0, [0, s - 1, l - 1], :] = np.arange(6, dtype=np.float32).reshape(3, 2) + layer
        if dynamic_b:
            model.graph.input.append(helper.make_tensor_value_info(b_name, TensorProto.FLOAT, [1, 1, l, 2]))
        else:
            model.graph.initializer.append(numpy_helper.from_array(b, name=b_name))
        output = f"/enc_p/encoder/attn_layers.{layer}/MatMul_3_output_0"
        model.graph.node.append(helper.make_node("MatMul", [target, b_name], [output], name=f"/enc_p/encoder/attn_layers.{layer}/MatMul_3"))
        model.graph.output.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, [1, 2, s, 2]))
    onnx.save(model, path)


def _make_shape_derived_model(path: Path, s: int = 3):
    model = onnx.load(path)
    x = model.graph.input[0]
    x.type.tensor_type.shape.dim[0].ClearField("dim_value")
    x.type.tensor_type.shape.dim[0].dim_param = "tokens"
    model.graph.initializer.append(numpy_helper.from_array(
        np.array([1, 2, s, s], dtype=np.int64), name="derived_shape"))
    model.graph.node.insert(0, helper.make_node(
        "Reshape", [x.name, "derived_shape"], ["derived_x"], name="derived_reshape"))
    for node in model.graph.node:
        if node.op_type == "Softmax":
            node.input[0] = "derived_x"
    onnx.save(model, path)


def _wrap_first_b_static(path: Path, s: int = 3):
    model = onnx.load(path)
    b_name = "B0"
    model.graph.initializer.remove(next(init for init in model.graph.initializer if init.name == b_name))
    base = np.zeros((s, 2), dtype=np.float32)
    base[[0, 1, 2][:s], :] = np.arange(min(s, 3) * 2, dtype=np.float32).reshape(min(s, 3), 2) + 1
    model.graph.initializer.append(numpy_helper.from_array(base, name="emb_rel_v"))
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([1, 0, 1, 0], dtype=np.int64), name="emb_pads"),
        numpy_helper.from_array(np.array([0, 0], dtype=np.int64), name="emb_starts"),
        numpy_helper.from_array(np.array([2 * s - 1, 2], dtype=np.int64), name="emb_ends"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="emb_axes"),
        numpy_helper.from_array(np.array([0, 1], dtype=np.int64), name="emb_unsq_axes"),
    ])
    pad = "/enc_p/encoder/attn_layers.0/emb_pad"
    sliced = "/enc_p/encoder/attn_layers.0/emb_slice"
    unsq = "/enc_p/encoder/attn_layers.0/emb_unsqueeze"
    chain = [
        helper.make_node("Pad", ["emb_rel_v", "emb_pads"], [pad], name=pad),
        helper.make_node("Slice", [pad, "emb_starts", "emb_ends", "emb_axes"], [sliced], name=sliced),
        helper.make_node("Unsqueeze", [sliced, "emb_unsq_axes"], [unsq], name=unsq),
    ]
    mm = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/MatMul_3"))
    mm.input[1] = unsq
    index = next(i for i, node in enumerate(model.graph.node) if node.name == mm.name)
    nodes = list(model.graph.node)
    nodes[index:index] = chain
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    onnx.save(model, path)


def test_rewrites_all_six_layers_and_matches_ort(tmp_path):
    source = tmp_path / "source.onnx"; output = tmp_path / "rewritten.onnx"
    _make_model(source)
    manifest = rewrite(source, output, expected_layers=6, seq_len=3)
    assert len(manifest["rewritten_layers"]) == 6
    rewritten = onnx.load(output)
    assert not any(node.op_type == "Pad" for node in rewritten.graph.node)
    assert sum(node.op_type == "Gather" for node in rewritten.graph.node) == 6
    a = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(4).normal(size=(1, 2, 3, 3)).astype(np.float32)
    for left, right in zip(a.run(None, {"x": x}), b.run(None, {"x": x})):
        assert np.array_equal(left, right)


def test_rejects_missing_layer(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source, layers=5)
    with pytest.raises(RuntimeError, match="expected 6"):
        rewrite(source, tmp_path / "out.onnx", expected_layers=6)


def test_rejects_nonmatching_static_bucket(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source, layers=6, s=3)
    with pytest.raises(RuntimeError, match="unsupported attention shape"):
        rewrite(source, tmp_path / "out.onnx", expected_layers=6, seq_len=4)


def test_rejects_matmul_instead_of_relative_chain(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    pad = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/Pad"))
    pad.op_type = "MatMul"
    pad.input[:] = [pad.input[0], "bad_weight"]
    model.graph.initializer.append(
        numpy_helper.from_array(np.eye(3, dtype=np.float32), name="bad_weight")
    )
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="valid relative-position Slice chains"):
        rewrite(source, tmp_path / "out.onnx")


def test_rejects_wrong_slice_parameters(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    starts = next(init for init in model.graph.initializer if init.name == "p0_starts")
    starts.CopyFrom(numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64), name="p0_starts"))
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="unsupported relative Slice parameters"):
        rewrite(source, tmp_path / "out.onnx")


def test_accepts_normalized_slice_parameters(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    max_end = np.iinfo(np.int64).max
    for layer in range(6):
        for name, values in (
            (f"p{layer}_starts", [1]),
            (f"p{layer}_ends", [max_end]),
            (f"p{layer}_axes", [3]),
        ):
            init = next(value for value in model.graph.initializer if value.name == name)
            init.CopyFrom(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name))
    manifest = rewrite(source, tmp_path / "out.onnx", seq_len=3)
    assert len(manifest["rewritten_layers"]) == 6


def test_rejects_normalized_slice_wrong_start(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    init = next(value for value in model.graph.initializer if value.name == "p0_starts")
    init.CopyFrom(numpy_helper.from_array(np.array([2], dtype=np.int64), name="p0_starts"))
    ends = next(value for value in model.graph.initializer if value.name == "p0_ends")
    ends.CopyFrom(numpy_helper.from_array(np.array([np.iinfo(np.int64).max], dtype=np.int64), name="p0_ends"))
    axes = next(value for value in model.graph.initializer if value.name == "p0_axes")
    axes.CopyFrom(numpy_helper.from_array(np.array([3], dtype=np.int64), name="p0_axes"))
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="unsupported relative Slice parameters"):
        rewrite(source, tmp_path / "out.onnx")


def test_rejects_float16(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    model.graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    for value in model.graph.output:
        value.type.tensor_type.elem_type = TensorProto.FLOAT16
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="unsupported Softmax dtype"):
        rewrite(source, tmp_path / "out.onnx")


@pytest.mark.parametrize("mode", ["edge", "reflect"])
def test_rejects_nonconstant_pad_mode(tmp_path, mode):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    pad = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/Pad"))
    pad.attribute.append(helper.make_attribute("mode", mode))
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="unsupported Pad mode"):
        rewrite(source, tmp_path / "out.onnx")


def test_rejects_nonzero_pad_constant(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    pad = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/Pad"))
    pad.input.append("p0_constant")
    model.graph.initializer.append(
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="p0_constant")
    )
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match="nonzero Pad constant_value"):
        rewrite(source, tmp_path / "out.onnx")


def test_preserves_shared_old_chain_node(tmp_path):
    source = tmp_path / "source.onnx"; output = tmp_path / "rewritten.onnx"
    _make_model(source)
    model = onnx.load(source)
    shared = "/enc_p/encoder/attn_layers.0/Pad_1_output_0"
    identity = "/enc_p/encoder/attn_layers.0/shared_identity"
    model.graph.node.append(helper.make_node("Identity", [shared], [identity], name=identity))
    model.graph.output.append(helper.make_tensor_value_info(identity, TensorProto.FLOAT, [1, 2, 12]))
    onnx.save(model, source)
    rewrite(source, output)
    rewritten = onnx.load(output)
    assert any(node.name.endswith("attn_layers.0/Pad_1") for node in rewritten.graph.node)


@pytest.mark.parametrize("s", [2, 4])
def test_compact_relative_value_matches_ort_and_keeps_slice(tmp_path, s):
    source = tmp_path / "source.onnx"; output = tmp_path / "compact.onnx"
    _make_model(source, s=s); _add_matmuls(source, s=s)
    manifest = rewrite(source, output, compact_relative_value=True, seq_len=s)
    assert manifest["mode"] == "compact-relative-value"
    assert all(row["K"] == [0, s - 1, 2 * s - 2] for row in manifest["rewritten_layers"])
    original = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    compact = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(9).normal(size=(1, 2, s, s)).astype(np.float32)
    for left, right in zip(original.run(None, {"x": x}), compact.run(None, {"x": x})):
        np.testing.assert_allclose(left, right, rtol=0, atol=1e-6)
    graph = onnx.load(output).graph
    assert sum(node.op_type == "Slice" for node in graph.node) == 6
    assert sum(node.op_type == "Gather" for node in graph.node) == 6


def test_compact_rejects_dynamic_b(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source); _add_matmuls(source, dynamic_b=True)
    with pytest.raises(RuntimeError, match="must be static initializer or Constant"):
        rewrite(source, tmp_path / "compact.onnx", compact_relative_value=True)


def test_shape_derived_attention_metadata_is_accepted(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    _make_shape_derived_model(source)
    manifest = rewrite(source, tmp_path / "out.onnx", seq_len=3)
    assert len(manifest["rewritten_layers"]) == 6


def test_dynamic_attention_shape_is_rejected(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    dim.ClearField("dim_value")
    dim.dim_param = "dynamic_batch"
    onnx.save(model, source)
    with pytest.raises(RuntimeError, match=r"needs static \[B,H,S,S\] shape"):
        rewrite(source, tmp_path / "out.onnx")


def test_shape_only_identity_parameter_is_supported(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    pad = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/Pad"))
    init = next(value for value in model.graph.initializer if value.name == "p0_pads")
    model.graph.initializer.remove(init)
    model.graph.initializer.append(numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, 0, 2], dtype=np.int64), name="p0_pads_base"))
    pad_index = next(index for index, node in enumerate(model.graph.node) if node.name == pad.name)
    model.graph.node.insert(pad_index, helper.make_node("Identity", ["p0_pads_base"], ["p0_pads"], name="static_pads_identity"))
    onnx.save(model, source)
    from rewrite_piper_relative_position import rewrite as local_rewrite
    manifest = local_rewrite(source, tmp_path / "out.onnx", expected_layers=6, seq_len=3)
    assert len(manifest["rewritten_layers"]) == 6


def test_shape_only_transpose_parameter_is_supported(tmp_path):
    source = tmp_path / "source.onnx"
    _make_model(source)
    model = onnx.load(source)
    pad = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/Pad"))
    init = next(value for value in model.graph.initializer if value.name == "p0_pads")
    model.graph.initializer.remove(init)
    desired = np.array([0, 0, 0, 0, 0, 0, 0, 2], dtype=np.int64)
    model.graph.initializer.append(numpy_helper.from_array(desired.reshape(4, 2).T, name="p0_pads_base"))
    model.graph.initializer.append(numpy_helper.from_array(np.array([1, 0], dtype=np.int64), name="p0_transpose_perm"))
    pad_index = next(index for index, node in enumerate(model.graph.node) if node.name == pad.name)
    model.graph.node.insert(pad_index, helper.make_node(
        "Transpose", ["p0_pads_base"], ["p0_pads_transposed"],
        name="p0_pads_transpose", perm=[1, 0]))
    model.graph.node.insert(pad_index + 1, helper.make_node(
        "Reshape", ["p0_pads_transposed", "p0_pad_shape"], ["p0_pads"],
        name="p0_pads_reshape"))
    model.graph.initializer.append(numpy_helper.from_array(np.array([8], dtype=np.int64), name="p0_pad_shape"))
    onnx.save(model, source)
    manifest = rewrite(source, tmp_path / "out.onnx", expected_layers=6, seq_len=3)
    assert len(manifest["rewritten_layers"]) == 6


def test_compact_static_emb_rel_v_pad_slice_unsqueeze(tmp_path):
    source = tmp_path / "source.onnx"; output = tmp_path / "compact.onnx"
    _make_model(source); _add_matmuls(source); _wrap_first_b_static(source)
    manifest = rewrite(source, output, compact_relative_value=True, seq_len=3)
    assert manifest["rewritten_layers"][0]["B_source"].endswith("emb_unsqueeze")


def test_compact_static_emb_rel_v_opset11_unsqueeze_attribute(tmp_path):
    source = tmp_path / "source.onnx"; output = tmp_path / "compact.onnx"
    _make_model(source); _add_matmuls(source); _wrap_first_b_static(source)
    model = onnx.load(source)
    model.opset_import[0].version = 11
    unsq = next(node for node in model.graph.node if node.name.endswith("attn_layers.0/emb_unsqueeze"))
    del unsq.input[1:]
    unsq.attribute.append(helper.make_attribute("axes", [0, 1]))
    onnx.checker.check_model(model)
    onnx.save(model, source)
    manifest = rewrite(source, output, compact_relative_value=True, seq_len=3)
    assert manifest["rewritten_layers"][0]["B_source"].endswith("emb_unsqueeze")

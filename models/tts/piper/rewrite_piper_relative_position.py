#!/usr/bin/env python3
"""Replace Piper attention absolute→relative position slice chains.

The replacement preserves the Slice output tensor name and uses a static
Gather index table for a fixed token bucket. It does not bake tokens, masks, or
random values. The source graph must contain exactly the requested attention
layers, each with one Softmax→Slice dependency and static ``[1,H,S,S]`` shape.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnx.reference import ReferenceEvaluator


def _shape(v):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in v.type.tensor_type.shape.dim]


def _attention_shape(values, output, layer):
    info = values.get(output)
    dims = None if info is None else _shape(info)
    if dims is None or len(dims) != 4 or not all(isinstance(x, int) and x > 0 for x in dims):
        raise RuntimeError(f"attention layer {layer} needs static [B,H,S,S] shape, got {dims}")
    return dims


def _infer_shapes(model):
    model = shape_inference.infer_shapes(model, data_prop=True)
    values = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    unresolved = []
    for node in model.graph.node:
        if node.op_type != "Softmax" or "/attn_layers." not in node.name:
            continue
        info = values.get(node.output[0])
        dims = None if info is None else _shape(info)
        if dims is None or len(dims) != 4 or not all(isinstance(x, int) and x > 0 for x in dims):
            unresolved.append(node.output[0])
    if unresolved:
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        except ImportError as exc:
            raise RuntimeError(
                f"attention shape remains dynamic for {', '.join(unresolved)}; "
                "onnxruntime symbolic shape inference is unavailable"
            ) from exc
        try:
            model = SymbolicShapeInference.infer_shapes(
                model, auto_merge=False, guess_output_rank=False
            )
        except Exception as exc:
            raise RuntimeError(
                f"attention shape remains dynamic for {', '.join(unresolved)}"
            ) from exc
        values = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    return model, values


def _const_ints(model, name):
    array = _static_array(model, name, set())
    if not np.issubdtype(array.dtype, np.integer):
        raise RuntimeError(f"parameter {name} is not integer constant")
    return array.astype(np.int64).reshape(-1).tolist()


def _static_array(model, name, stack):
    if name in stack:
        raise RuntimeError(f"cyclic static parameter dependency at {name}")
    stack.add(name)
    try:
        for init in model.graph.initializer:
            if init.name == name:
                return numpy_helper.to_array(init)
        for node in model.graph.node:
            if node.op_type == "Constant" and node.output and node.output[0] == name:
                tensor = next((a.t for a in node.attribute if a.name == "value"), None)
                if tensor is not None:
                    return numpy_helper.to_array(tensor)
        producers = {out: node for node in model.graph.node for out in node.output}
        node = producers.get(name)
        if node is None or not node.input:
            raise RuntimeError(f"parameter {name} must be static shape-only data")
        if node.op_type == "Shape":
            values = {v.name: v for v in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output)}
            value = values.get(node.input[0])
            if value is None:
                raise RuntimeError(f"Shape input {node.input[0]} lacks static metadata")
            dims = _shape(value)
            if not all(isinstance(dim, int) and dim > 0 for dim in dims):
                raise RuntimeError(f"Shape input {node.input[0]} is dynamic")
            start = next((a.i for a in node.attribute if a.name == "start"), 0)
            end = next((a.i for a in node.attribute if a.name == "end"), len(dims))
            return np.asarray(dims[start:end], dtype=np.int64)
        allowed = {"Identity", "Cast", "Gather", "Concat", "Unsqueeze", "Transpose", "Pad", "Slice",
                   "Reshape", "Add", "Sub", "Mul", "Div"}
        if node.op_type not in allowed:
            raise RuntimeError(f"unsupported dynamic/static parameter op {node.op_type} for {name}")
        arrays = {value: _static_array(model, value, stack) for value in node.input if value}
        inputs = [helper.make_tensor_value_info(value, helper.np_dtype_to_tensor_dtype(array.dtype), list(array.shape))
                  for value, array in arrays.items()]
        graph = helper.make_graph(
            [node], "static_parameter", inputs,
            [helper.make_tensor_value_info(name, helper.np_dtype_to_tensor_dtype(next(iter(arrays.values())).dtype), None)],
        )
        evaluator_model = helper.make_model(
            graph,
            opset_imports=[copy.deepcopy(opset) for opset in model.opset_import],
            ir_version=model.ir_version,
        )
        evaluator = ReferenceEvaluator(evaluator_model)
        return np.asarray(evaluator.run(None, arrays)[0])
    finally:
        stack.remove(name)


def _const_is_zero(model, name):
    return bool(np.all(_static_array(model, name, set()) == 0))


def _resolve_reshape_shape(spec, input_shape, allowzero=False):
    resolved = []
    infer_at = None
    for index, dim in enumerate(spec):
        if dim == 0 and not allowzero:
            if index >= len(input_shape):
                raise RuntimeError("Reshape zero-copy index exceeds input rank")
            resolved.append(input_shape[index])
        elif dim == -1:
            if infer_at is not None:
                raise RuntimeError("Reshape has multiple -1 dimensions")
            infer_at = index
            resolved.append(-1)
        elif dim > 0:
            resolved.append(dim)
        else:
            raise RuntimeError(f"unsupported Reshape dimension {dim}")
    input_size = int(np.prod(input_shape, dtype=np.int64))
    known_size = int(np.prod([x for x in resolved if x != -1], dtype=np.int64))
    if infer_at is not None:
        if known_size == 0 or input_size % known_size:
            raise RuntimeError("Reshape -1 is incompatible with input size")
        resolved[infer_at] = input_size // known_size
    elif known_size != input_size:
        raise RuntimeError("Reshape changes tensor element count")
    return resolved


def _check_reshape(model, node, input_shape, expected):
    if len(node.input) != 2:
        raise RuntimeError(f"unsupported Reshape inputs in {node.name}")
    spec = _const_ints(model, node.input[1])
    allowzero = next((a.i for a in node.attribute if a.name == "allowzero"), 0)
    actual = _resolve_reshape_shape(spec, input_shape, bool(allowzero))
    if actual != expected:
        raise RuntimeError(f"unsupported Reshape shape in {node.name}: {spec} -> {actual}")


def _check_relative_slice(model, node, rank, dims, s):
    if len(node.input) < 3:
        raise RuntimeError(f"unsupported relative Slice parameters in {node.name}")
    starts = _const_ints(model, node.input[1])
    ends = _const_ints(model, node.input[2])
    axes = _const_ints(model, node.input[3]) if len(node.input) > 3 else list(range(len(starts)))
    steps = _const_ints(model, node.input[4]) if len(node.input) > 4 else [1] * len(starts)
    if len(starts) != len(ends) or len(axes) != len(starts) or len(steps) != len(starts):
        raise RuntimeError(f"unsupported relative Slice parameters in {node.name}")
    normalized = [(0, dim) for dim in dims]
    seen = set()
    int64_min, int64_max = -(1 << 63), (1 << 63) - 1
    for start, end, axis, step in zip(starts, ends, axes, steps):
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank or axis in seen or step != 1:
            raise RuntimeError(f"unsupported relative Slice parameters in {node.name}")
        seen.add(axis)
        dim = dims[axis]
        start = max(0, min(dim, start + dim if start < 0 else start))
        if end == int64_max:
            end = dim
        elif end == int64_min:
            end = 0
        else:
            end = max(0, min(dim, end + dim if end < 0 else end))
        normalized[axis] = (start, end)
    expected = [(0, dims[0]), (0, dims[1]), (0, dims[2]), (1, 2 * s)]
    if normalized != expected:
        raise RuntimeError(f"unsupported relative Slice parameters in {node.name}")


def _previous(producers, value, expected):
    node = producers.get(value)
    while node is not None and node.op_type == "Identity":
        value = node.input[0]
        node = producers.get(value)
    if node is None or node.op_type != expected:
        actual = None if node is None else node.op_type
        raise RuntimeError(f"expected {expected} before {value}, found {actual}")
    return node, value


def _check_zero_pad(model, node):
    mode = next((a.s.decode() for a in node.attribute if a.name == "mode"), "constant")
    if mode != "constant":
        raise RuntimeError(f"unsupported Pad mode in {node.name}: {mode}")
    if len(node.input) > 2 and node.input[2]:
        if not _const_is_zero(model, node.input[2]):
            raise RuntimeError(f"nonzero Pad constant_value in {node.name}")


def _match_chain(model, soft, target, values):
    producers = {out: n for n in model.graph.node for out in n.output}
    if len(target.input) < 1:
        raise RuntimeError(f"Slice {target.name} has no data input")
    reshape2, value = _previous(producers, target.input[0], "Reshape")
    soft_shape = _attention_shape(values, soft.output[0], soft.name)
    b, h, s, _ = soft_shape
    _check_reshape(model, reshape2, [1, h, 2 * s * s], [1, h, s, 2 * s])
    pad2, value = _previous(producers, reshape2.input[0], "Pad")
    _check_zero_pad(model, pad2)
    if len(pad2.input) < 2 or _const_ints(model, pad2.input[1]) != [0, 0, s, 0, 0, 0]:
        raise RuntimeError(f"unsupported left Pad parameters in {pad2.name}")
    reshape1, value = _previous(producers, pad2.input[0], "Reshape")
    _check_reshape(model, reshape1, [1, h, s, 2 * s - 1], [1, h, s * (2 * s - 1)])
    pad1, value = _previous(producers, reshape1.input[0], "Pad")
    _check_zero_pad(model, pad1)
    if len(pad1.input) < 2 or _const_ints(model, pad1.input[1]) != [0, 0, 0, 0, 0, 0, 0, s - 1]:
        raise RuntimeError(f"unsupported right Pad parameters in {pad1.name}")
    if pad1.input[0] not in soft.output:
        raise RuntimeError(f"relative chain for {target.name} does not originate at its Softmax")
    _check_relative_slice(model, target, 4, [1, h, s, 2 * s], s)
    out_shape = _shape(values.get(target.output[0])) if target.output[0] in values else None
    if out_shape != [1, h, s, 2 * s - 1]:
        raise RuntimeError(f"unsupported relative Slice output shape in {target.name}: {out_shape}")
    target_info = values.get(target.output[0])
    if target_info is None or target_info.type.tensor_type.elem_type != TensorProto.FLOAT:
        dtype = None if target_info is None else target_info.type.tensor_type.elem_type
        raise RuntimeError(f"unsupported relative Slice dtype in {target.name}: {dtype}")
    return {pad1.name, reshape1.name, pad2.name, reshape2.name, target.name}


def _find_layers(model, values):
    producers = {out: n for n in model.graph.node for out in n.output}
    consumers = {}
    for n in model.graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    found = {}
    for soft in model.graph.node:
        if soft.op_type != "Softmax" or "/attn_layers." not in soft.name:
            continue
        layer = soft.name.split("/attn_layers.", 1)[1].split("/", 1)[0]
        if not layer.isdigit():
            continue
        soft_info = values.get(soft.output[0])
        if soft_info is None:
            raise RuntimeError(f"missing shape metadata for {soft.output[0]}")
        _attention_shape(values, soft.output[0], layer)
        if soft_info.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(
                f"unsupported Softmax dtype for layer {layer}: "
                f"{soft_info.type.tensor_type.elem_type}"
            )
        queue = list(soft.output); seen = set(); slices = []
        while queue:
            value = queue.pop(0)
            if value in seen:
                continue
            seen.add(value)
            for node in consumers.get(value, []):
                if node.op_type == "Slice":
                    slices.append(node)
                    continue
                # Only walk the known Pad/Reshape/Slice plumbing. A MatMul or
                # random branch is never a candidate relative-position chain.
                if node.op_type not in {"Identity", "Pad", "Reshape", "Slice"}:
                    continue
                queue.extend(node.output)
        valid = []
        errors = []
        for target in slices:
            try:
                _match_chain(model, soft, target, values)
            except RuntimeError as exc:
                errors.append(str(exc))
                continue
            valid.append(target)
        if len(valid) != 1:
            semantic_errors = [error for error in errors if error.startswith((
                "unsupported Pad mode", "nonzero Pad constant_value",
                "unsupported relative Slice parameters"))]
            if not valid and semantic_errors:
                target = slices[errors.index(semantic_errors[0])]
                raise RuntimeError(f"{target.name}: {semantic_errors[0]}")
            if errors:
                details = "; ".join(
                    f"{target.name}: {error}" for target, error in zip(slices, errors)
                )
            else:
                details = "no Softmax-reachable Slice candidates"
            raise RuntimeError(
                f"attention layer {layer} has {len(valid)} valid relative-position Slice chains; {details}"
            )
        if layer in found:
            raise RuntimeError(f"duplicate Softmax layer {layer}")
        found[layer] = (soft, valid[0])
    return found


def _closure_between(model, soft, target):
    producers = {out: n for n in model.graph.node for out in n.output}
    keep = set(); stack = list(target.input)
    while stack:
        value = stack.pop()
        if value in soft.output:
            continue
        node = producers.get(value)
        if node is None or node.name in keep:
            continue
        keep.add(node.name)
        stack.extend(node.input)
    return keep


def _prune_replaced_chain(nodes, remove_names, graph_outputs):
    """Drop the selected chain even when its nodes consume one another."""
    remaining = list(nodes)
    while True:
        used = {value for node in remaining for value in node.input}
        dead = {
            node.name for node in remaining
            if node.name in remove_names
            and not any(output in used or output in graph_outputs for output in node.output)
        }
        if not dead:
            return remaining
        remaining = [node for node in remaining if node.name not in dead]


def _static_tensor(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init), f"initializer:{name}"
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] == name:
            tensor = next((a.t for a in node.attribute if a.name == "value"), None)
            if tensor is not None:
                return numpy_helper.to_array(tensor), f"Constant:{node.name or name}"
    raise RuntimeError(f"MatMul B {name} must be static initializer or Constant")


def _compact_layer(model, soft, target, values, nodes, layer):
    consumers = [node for node in nodes if target.output[0] in node.input]
    matmuls = [node for node in consumers if node.op_type == "MatMul"]
    if len(matmuls) != 1 or len(consumers) != 1:
        raise RuntimeError(f"layer {layer} relative Slice must have exactly one MatMul consumer")
    matmul = matmuls[0]
    try:
        b_array = _static_array(model, matmul.input[1], set())
    except RuntimeError as exc:
        raise RuntimeError(
            f"MatMul B {matmul.input[1]} must be static initializer or Constant/shape-only closure"
        ) from exc
    b_source = f"static-closure:{matmul.input[1]}"
    if b_array.dtype != np.float32 or b_array.ndim != 4:
        raise RuntimeError(f"layer {layer} B must be FLOAT32 rank-4 [1,1,2S-1,D], got {b_array.shape}")
    shape = _shape(values[soft.output[0]])
    _, h, s, _ = shape
    if list(b_array.shape[:2]) != [1, 1] or list(b_array.shape[2:3]) != [2 * s - 1]:
        raise RuntimeError(f"layer {layer} B shape must be [1,1,{2*s-1},D], got {b_array.shape}")
    rows = np.any(b_array[0, 0] != 0, axis=1)
    k = np.flatnonzero(rows).astype(np.int64)
    if k.size == 0:
        raise RuntimeError(f"layer {layer} B has no nonzero relative rows")
    width = int(k.size)
    index = np.full((s, width), s * s, dtype=np.int64)
    for i in range(s):
        for q, relative_row in enumerate(k):
            j = int(relative_row) + i - (s - 1)
            if 0 <= j < s:
                index[i, q] = i * s + j
    prefix = f"piper_compact_relative_{layer}"
    b_name = f"{prefix}_B"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([1, h, s * s], np.int64), name=f"{prefix}_shape"),
        numpy_helper.from_array(np.zeros((1, h, 1), np.float32), name=f"{prefix}_zero"),
        numpy_helper.from_array(index, name=f"{prefix}_indices"),
        numpy_helper.from_array(b_array[:, :, k, :], name=b_name),
    ])
    compact = f"{prefix}_value"
    replacement = [
        helper.make_node("Reshape", [soft.output[0], f"{prefix}_shape"], [f"{prefix}_flat"], name=f"{prefix}_reshape"),
        helper.make_node("Concat", [f"{prefix}_flat", f"{prefix}_zero"], [f"{prefix}_padded"], axis=2, name=f"{prefix}_pad"),
        helper.make_node("Gather", [f"{prefix}_padded", f"{prefix}_indices"], [compact], axis=2, name=f"{prefix}_gather"),
    ]
    insert = next(i for i, node in enumerate(nodes) if node.name == matmul.name)
    nodes[insert:insert] = replacement
    matmul.input[0] = compact
    matmul.input[1] = b_name
    return {"layer": int(layer), "softmax": soft.output[0], "slice": target.output[0],
            "matmul": matmul.name, "compact_tensor": compact, "K": k.tolist(),
            "B_shape": list(b_array.shape), "compact_B_shape": list(b_array[:, :, k, :].shape),
            "B_source": b_source, "output_shape": [1, h, s, width]}


def rewrite(input_path: Path, output_path: Path, expected_layers: int = 6,
            seq_len: int | None = None, compact_relative_value: bool = False) -> dict:
    model, values = _infer_shapes(onnx.load(str(input_path), load_external_data=False))
    onnx.checker.check_model(model)
    layers = _find_layers(model, values)
    if len(layers) != expected_layers:
        raise RuntimeError(f"expected {expected_layers} attention layers, found {len(layers)}")
    replacements = []
    remove_names = set()
    new_nodes = list(model.graph.node)
    for layer in sorted(layers, key=int):
        soft, target = layers[layer]
        shape = values.get(soft.output[0])
        if shape is None:
            raise RuntimeError(f"missing shape metadata for {soft.output[0]}")
        dims = _shape(shape)
        if shape.type.tensor_type.elem_type != TensorProto.FLOAT:
            raise RuntimeError(f"unsupported Softmax dtype for layer {layer}: {shape.type.tensor_type.elem_type}")
        if len(dims) != 4 or not all(isinstance(x, int) and x > 0 for x in dims):
            raise RuntimeError(f"attention layer {layer} needs static [B,H,S,S] shape, got {dims}")
        b, h, s0, s1 = dims
        if b != 1 or s0 != s1 or (seq_len is not None and s0 != seq_len):
            raise RuntimeError(f"unsupported attention shape for layer {layer}: {dims}")
        _match_chain(model, soft, target, values)
        if compact_relative_value:
            replacements.append(_compact_layer(model, soft, target, values, new_nodes, layer))
            continue
        L = 2 * s0 - 1
        idx = np.full((s0, L), s0 * s0, dtype=np.int64)
        for i in range(s0):
            for k in range(L):
                j = k + i - (s0 - 1)
                if 0 <= j < s0:
                    idx[i, k] = i * s0 + j
        prefix = f"piper_relative_{layer}"
        model.graph.initializer.extend([
            numpy_helper.from_array(np.array([1, h, s0 * s0], np.int64), name=f"{prefix}_shape"),
            numpy_helper.from_array(np.zeros((1, h, 1), np.float32), name=f"{prefix}_zero"),
            numpy_helper.from_array(idx, name=f"{prefix}_indices"),
        ])
        replacement = [
            helper.make_node("Reshape", [soft.output[0], f"{prefix}_shape"], [f"{prefix}_flat"], name=f"{prefix}_reshape"),
            helper.make_node("Concat", [f"{prefix}_flat", f"{prefix}_zero"], [f"{prefix}_padded"], axis=2, name=f"{prefix}_pad"),
            helper.make_node("Gather", [f"{prefix}_padded", f"{prefix}_indices"], [target.output[0]], axis=2, name=f"{prefix}_gather"),
        ]
        insert = next(i for i, n in enumerate(new_nodes) if target.name == n.name)
        new_nodes[insert:insert + 1] = replacement
        remove_names.update(_closure_between(model, soft, target))
        remove_names.discard(soft.name)
        replacements.append({"layer": int(layer), "softmax": soft.output[0], "target": target.output[0],
                             "input_shape": dims, "output_shape": [1, h, s0, L],
                             "removed_nodes": sorted(_closure_between(model, soft, target))})
    if not compact_relative_value:
        graph_outputs = {value.name for value in model.graph.output}
        new_nodes = _prune_replaced_chain(new_nodes, remove_names, graph_outputs)
    del model.graph.node[:]; model.graph.node.extend(new_nodes)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    manifest = {"format": 1, "source": str(input_path), "expected_layers": expected_layers,
                "mode": "compact-relative-value" if compact_relative_value else "dense-relative-value",
                "rewritten_layers": replacements,
                "constraints": "static bucket; no token/mask/random baking; static B only"}
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--compact-relative-value", action="store_true")
    a = p.parse_args()
    print(json.dumps(rewrite(a.input, a.output, a.layers, a.seq_len, a.compact_relative_value), indent=2))


if __name__ == "__main__":
    main()

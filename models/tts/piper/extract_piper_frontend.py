#!/usr/bin/env python3
"""Extract Piper's dependency-closed text encoder for RKNN experiments.

This deliberately does not apply the older RKNN surgery passes.  It keeps the
real token/mask dependency graph intact and emits a manifest describing every
boundary tensor.  The default cut is the encoder output consumed by the
duration predictor: mask, encoder hidden state, and the two split tensors.

Example::

  python extract_piper_frontend.py --input model.onnx --output-dir out \
      --seq-len 128 --check

The generated ``text_encoder.onnx`` is the prefix and ``remainder.onnx`` is
the original graph after that prefix.  Both are fixed to the requested token
bucket while the mask remains an explicit input, so padding values are not
baked into the graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference
from onnx.utils import Extractor

try:
    from extract_piper_dp_conditioner import (
        REJECT_OPS as DP_REJECT_OPS,
        _find_boundaries as _find_dp_boundaries,
        _find_conditioning as _find_dp_conditioning,
    )
except ImportError:  # pragma: no cover - package-style import
    from .extract_piper_dp_conditioner import (
        REJECT_OPS as DP_REJECT_OPS,
        _find_boundaries as _find_dp_boundaries,
        _find_conditioning as _find_dp_conditioning,
    )

DEFAULT_OUTPUTS = (
    "/enc_p/Cast_1_output_0",
    "/enc_p/encoder/Mul_2_output_0",
    "/enc_p/Split_output_0",
    "/enc_p/Split_output_1",
    "/enc_p/encoder/Unsqueeze_output_0",
)


def _shape(v) -> list[int | str]:
    out = []
    for d in v.type.tensor_type.shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") else d.dim_param)
    return out

def _dtype(v) -> str:
    return TensorProto.DataType.Name(v.type.tensor_type.elem_type).lower()


def _replace_mask_input(model: onnx.ModelProto, seq_len: int) -> None:
    """Expose the real encoder mask without deleting unrelated Range nodes."""
    target = "/enc_p/Cast_1_output_0"
    producers = [n for n in model.graph.node if target in n.output]
    if not producers:
        raise ValueError(f"mask tensor {target!r} is absent")
    if len(producers[0].output) != 1:
        raise ValueError(f"mask producer {producers[0].name!r} has multiple outputs")
    mask = helper.make_tensor_value_info("x_mask", TensorProto.FLOAT, [1, 1, seq_len])
    if not any(i.name == mask.name for i in model.graph.input):
        model.graph.input.append(mask)
    for node in model.graph.node:
        for i, value in enumerate(node.input):
            if value == target:
                node.input[i] = "x_mask"
    # Keep the public boundary name stable.  The original producer is removed
    # from this output path and replaced by an explicit identity, while its
    # other inputs remain available for any independently reachable branch.
    kept = [n for n in model.graph.node if target not in n.output]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model.graph.node.append(helper.make_node("Identity", ["x_mask"], [target],
                                             name="piper_frontend_mask_boundary"))


def _ensure_boundary_metadata(model: onnx.ModelProto, names: tuple[str, ...],
                              seq_len: int) -> None:
    known = {v.name for v in model.graph.value_info}
    known.update(v.name for v in model.graph.input)
    known.update(v.name for v in model.graph.output)
    defaults = {
        "/enc_p/Cast_1_output_0": (TensorProto.FLOAT, [1, 1, seq_len]),
        "/enc_p/encoder/Mul_2_output_0": (TensorProto.FLOAT, [1, 192, seq_len]),
        "/enc_p/Split_output_0": (TensorProto.FLOAT, [1, 192, seq_len]),
        "/enc_p/Split_output_1": (TensorProto.FLOAT, [1, 192, seq_len]),
        "/enc_p/encoder/Unsqueeze_output_0": (TensorProto.FLOAT, [1, 1, 1, seq_len]),
    }
    for name in names:
        if name not in known and name in defaults:
            dtype, shape = defaults[name]
            model.graph.value_info.append(helper.make_tensor_value_info(name, dtype, shape))


def _prune_unused_inputs(model: onnx.ModelProto) -> None:
    used = {value for node in model.graph.node for value in node.input}
    kept = [value for value in model.graph.input if value.name in used]
    del model.graph.input[:]
    model.graph.input.extend(kept)


def _conditioner_output(model: onnx.ModelProto) -> tuple[str, set[str]]:
    """Find and validate the deterministic DP conditioner output.

    The DP extractor owns the graph-specific boundary selection.  This
    wrapper additionally checks the selected closure for all random/flow and
    spline-like nodes before allowing it into the frontend prefix.
    """
    model = shape_inference.infer_shapes(model)
    boundaries = _find_dp_boundaries(model)
    _flow_name, candidates = _find_dp_conditioning(model, boundaries)
    if len(candidates) != 1:
        raise ValueError(f"DP conditioner requires one selected output, got {candidates}")
    selected = candidates[0]
    producer = {out: node for node in model.graph.node for out in node.output}
    stack = [selected]
    seen = set()
    safe_nodes = set()
    while stack:
        value = stack.pop()
        if value in seen or value in boundaries:
            continue
        seen.add(value)
        node = producer.get(value)
        if node is None:
            continue
        name = node.name.lower()
        if (node.op_type in DP_REJECT_OPS or node.op_type.startswith("Random")
                or node.name.startswith("/dp/flows.") or "spline" in name):
            raise ValueError(f"unsafe node in DP conditioner closure: {node.name} ({node.op_type})")
        safe_nodes.add(node.name)
        stack.extend(node.input)
    return selected, safe_nodes


def _tensor_metadata(model: onnx.ModelProto, name: str) -> dict:
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        if value.name == name:
            return {"name": name, "dtype": _dtype(value), "shape": _shape(value)}
    raise ValueError(f"missing shape/dtype metadata for conditioner tensor {name!r}")


def extract(input_path: Path, output_dir: Path, seq_len: int,
            outputs: tuple[str, ...], check: bool,
            include_dp_conditioner: bool = False) -> dict:
    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    all_outputs = {o for n in model.graph.node for o in n.output}
    missing = [name for name in outputs if name not in all_outputs]
    if missing:
        raise ValueError(f"requested boundary tensors are missing: {missing}")
    _replace_mask_input(model, seq_len)
    _ensure_boundary_metadata(model, outputs, seq_len)
    conditioner_output = None
    if include_dp_conditioner:
        conditioner_output, conditioner_nodes = _conditioner_output(model)
        model = shape_inference.infer_shapes(model)
        if conditioner_output in outputs:
            raise ValueError(f"DP conditioner duplicates an existing boundary: {conditioner_output}")
        outputs = tuple(outputs) + (conditioner_output,)
        # The metadata must be available for both Extractor and the manifest;
        # fail closed instead of inventing runtime shapes or dtypes.
        _tensor_metadata(shape_inference.infer_shapes(model), conditioner_output)
    graph_inputs = [i.name for i in model.graph.input]

    # A fixed token bucket is required by RKNN.  Keep lengths/scales as real
    # inputs; only the token dimension is pinned here.
    for inp in model.graph.input:
        if inp.name == "input":
            dims = inp.type.tensor_type.shape.dim
            if len(dims) == 2:
                dims[0].dim_value = 1; dims[0].ClearField("dim_param")
                dims[1].dim_value = seq_len; dims[1].ClearField("dim_param")

    out_dir = output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_path = out_dir / "text_encoder.onnx"
    remainder_path = out_dir / "remainder.onnx"
    prefix = Extractor(model).extract_model(graph_inputs, list(outputs))
    # Remainder consumes the exact closure outputs and retains original inputs
    # that are still used by DP (e.g. scales and input_lengths).
    remainder_inputs = list(dict.fromkeys(list(outputs) + graph_inputs))
    remainder = Extractor(model).extract_model(remainder_inputs,
                                                [o.name for o in model.graph.output])
    if len(remainder.graph.output) != 2:
        raise ValueError("frontend split expects an encoder graph with exactly two outputs: z and y_mask")
    remainder_shapes = [_shape(o) for o in remainder.graph.output]
    if any(len(shape) != 3 for shape in remainder_shapes):
        raise ValueError(f"frontend remainder outputs must be rank-3, got {remainder_shapes}")
    if remainder_shapes[0][1] not in (192, "192") or remainder_shapes[1][1] not in (1, "1"):
        raise ValueError(f"frontend remainder output shapes must be z=[1,192,T], y_mask=[1,1,T], got {remainder_shapes}")
    for sub in (prefix, remainder):
        _prune_unused_inputs(sub)
        onnx.checker.check_model(sub)
    prefix_names = {n.name for n in prefix.graph.node}
    remainder_names = {n.name for n in remainder.graph.node}
    if any(name.startswith("/dp") for name in prefix_names):
        if not include_dp_conditioner:
            raise RuntimeError("prefix closure contains duration predictor nodes")
        if any(name.startswith("/dp") and name not in conditioner_nodes for name in prefix_names):
            raise RuntimeError("prefix closure contains non-conditioner duration predictor nodes")
    if include_dp_conditioner and conditioner_output not in {o.name for o in prefix.graph.output}:
        raise RuntimeError("prefix is missing the requested DP conditioner output")
    if any(name.startswith("/enc_p/encoder") for name in remainder_names):
        raise RuntimeError("remainder closure recomputes text encoder nodes")
    onnx.save(prefix, str(prefix_path)); onnx.save(remainder, str(remainder_path))
    conditioner_io = None
    if include_dp_conditioner:
        conditioner_io = {
            "inputs": [_tensor_metadata(prefix, name) for name in (
                "/enc_p/encoder/Mul_2_output_0", "/enc_p/Cast_1_output_0")],
            "output": _tensor_metadata(prefix, conditioner_output),
        }
    manifest = {
        "format": 1, "source": str(input_path), "seq_len": seq_len,
        "mask_input": "x_mask", "bucket": {"input": [1, seq_len]},
        "prefix": {"path": prefix_path.name, "inputs": [{"name": i.name, "dtype": _dtype(i), "shape": _shape(i)} for i in prefix.graph.input],
                   "outputs": [{"name": o.name, "shape": _shape(o)} for o in prefix.graph.output]},
        "remainder": {"path": remainder_path.name, "inputs": [i.name for i in remainder.graph.input],
                      "outputs": [o.name for o in remainder.graph.output],
                      "output_shapes": [_shape(o) for o in remainder.graph.output],
                      "output_semantic": ["z", "y_mask"]},
        "boundary_outputs": list(outputs),
        "dp_conditioner": {"enabled": include_dp_conditioner,
                            "stage": "prefix" if include_dp_conditioner else None,
                            "tensor": (_tensor_metadata(prefix, conditioner_output)
                                       if include_dp_conditioner else None),
                            "io": conditioner_io},
        "notes": "Dependency-closed ORT split; no Erf replacement, constant noise, or Range baking.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if check:
        print(json.dumps({"prefix_nodes": len(prefix.graph.node),
                          "remainder_nodes": len(remainder.graph.node),
                          "boundary_outputs": list(outputs)}, indent=2))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--output", action="append", dest="outputs")
    p.add_argument("--include-dp-conditioner", action="store_true",
                   help="include the validated deterministic DP conditioner in the NPU prefix")
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if a.seq_len <= 0: p.error("--seq-len must be positive")
    extract(a.input, a.output_dir, a.seq_len, tuple(a.outputs or DEFAULT_OUTPUTS), a.check,
            a.include_dp_conditioner)


if __name__ == "__main__":
    main()

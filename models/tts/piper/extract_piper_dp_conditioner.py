#!/usr/bin/env python3
"""Extract the deterministic Piper DP conditioner for an isolated RKNN test.

The cut is found from the first ``/dp/flows.*`` consumer by tracing every input:
the conditioning branch is selected when its reverse closure reaches
``/dp/pre`` or ``/dp/convs``. Other flow inputs (for example latent noise) are
ignored. Random, spline-indexing, or flow nodes are rejected only when they are
also in the selected conditioning closure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import shape_inference
from onnx.utils import Extractor

REJECT_OPS = {"RandomNormalLike", "RandomUniformLike", "NonZero", "GatherND", "ScatterND", "CumSum"}


def _shape(v):
    return [d.dim_value if d.HasField("dim_value") else d.dim_param
            for d in v.type.tensor_type.shape.dim]


def _dtype(v):
    return v.type.tensor_type.elem_type


def _find_boundaries(model):
    names = {v.name for v in model.graph.value_info}
    names.update(v.name for v in model.graph.input)
    wanted = ("/enc_p/encoder/Mul_2_output_0", "/enc_p/Cast_1_output_0")
    missing = [x for x in wanted if x not in names]
    if missing:
        raise RuntimeError(f"missing explicit hidden/mask boundaries: {missing}")
    return set(wanted)


def _find_conditioning(model, boundaries):
    producer = {out: node for node in model.graph.node for out in node.output}
    for node in model.graph.node:
        if not node.name.startswith("/dp/flows."):
            continue
        candidates = []
        for name in node.input:
            stack = [name]; seen = set(); has_conditioner = False; unsafe = None
            while stack:
                value = stack.pop()
                if value in seen or value in boundaries:
                    continue
                seen.add(value)
                upstream = producer.get(value)
                if upstream is None:
                    continue
                if upstream.op_type in REJECT_OPS or upstream.name.startswith("/dp/flows."):
                    unsafe = upstream
                    if any((producer.get(inp) is not None and
                            (producer[inp].name.startswith("/dp/pre/") or
                             producer[inp].name.startswith("/dp/convs/")))
                           for inp in upstream.input):
                        has_conditioner = True
                    continue
                if upstream.name.startswith("/dp/pre/") or upstream.name.startswith("/dp/convs/"):
                    has_conditioner = True
                stack.extend(upstream.input)
            if has_conditioner:
                if unsafe is not None:
                    raise RuntimeError(f"unsafe node in conditioner boundary: {unsafe.name} ({unsafe.op_type})")
                candidates.append(name)
        if candidates:
            return node.name, candidates
    raise RuntimeError("could not find a /dp/pre or /dp/convs tensor entering a flow")


def _prune_inputs(model):
    used = {value for node in model.graph.node for value in node.input}
    kept = [value for value in model.graph.input if value.name in used]
    del model.graph.input[:]
    model.graph.input.extend(kept)


def _profile_conditioner(profile_path: Path) -> dict:
    data = json.loads(profile_path.read_text())
    totals = {"events": 0, "us": 0}
    by_prefix = {}
    for event in data:
        if event.get("cat") != "Node":
            continue
        name = event.get("name", "")
        if not (name.startswith("/dp/pre/") or name.startswith("/dp/convs/")):
            continue
        prefix = "/dp/pre" if name.startswith("/dp/pre/") else "/dp/convs"
        by_prefix.setdefault(prefix, {"events": 0, "us": 0})
        by_prefix[prefix]["events"] += 1
        by_prefix[prefix]["us"] += event.get("dur", 0)
        totals["events"] += 1
        totals["us"] += event.get("dur", 0)
    return {"profile": str(profile_path), "by_prefix": by_prefix, "total": totals,
            "note": "ORT profiled node time; not an NPU speed estimate"}


def extract(input_path: Path, output_dir: Path, profile_path: Path | None = None) -> dict:
    model = shape_inference.infer_shapes(onnx.load(str(input_path)))
    onnx.checker.check_model(model)
    boundaries = _find_boundaries(model)
    flow_node, candidates = _find_conditioning(model, boundaries)
    producer = {out: node for node in model.graph.node for out in node.output}
    selected = []
    for tensor in candidates:
        node = producer[tensor]
        # Walk only the selected tensor's dependency closure and reject unsafe
        # nodes. Extractor performs the actual closure extraction below.
        stack = [tensor]; seen = set()
        while stack:
            name = stack.pop()
            if name in seen or name in boundaries:
                continue
            seen.add(name)
            n = producer.get(name)
            if n is None:
                continue
            if n.op_type in REJECT_OPS or n.name.startswith("/dp/flows."):
                raise RuntimeError(f"unsafe node in conditioner closure: {n.name} ({n.op_type})")
            stack.extend(n.input)
        selected.append(tensor)
    # A single conditioning output is required for the first compile test.
    if len(selected) != 1:
        raise RuntimeError(f"flow node {flow_node} has {len(selected)} conditioner candidates: {selected}")
    sub = Extractor(model).extract_model(sorted(boundaries), selected)
    _prune_inputs(sub)
    onnx.checker.check_model(sub)
    actual_inputs = {i.name for i in sub.graph.input}
    if actual_inputs != boundaries:
        raise RuntimeError(f"conditioner inputs must be exactly hidden+mask boundaries, got {sorted(actual_inputs)}")
    if any(n.name.startswith("/enc_p/") or n.name.startswith("/dp/flows.") for n in sub.graph.node):
        raise RuntimeError("conditioner extraction retained enc_p or flow nodes")
    if not any(n.name.startswith("/dp/pre/") for n in sub.graph.node) or not any(n.name.startswith("/dp/convs/") for n in sub.graph.node):
        raise RuntimeError("conditioner extraction did not retain both /dp/pre and /dp/convs")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(sub, str(output_dir / "dp_conditioner.onnx"))
    output = sub.graph.output[0]
    manifest = {
        "format": 1, "source": str(input_path), "flow_consumer": flow_node,
        "boundary_inputs": sorted(boundaries),
        "conditioning_output": output.name, "inputs": [
            {"name": i.name, "dtype": _dtype(i), "shape": _shape(i)} for i in sub.graph.input],
        "output": {"name": output.name, "shape": _shape(output)},
        "constraints": {"forbid_ops": sorted(REJECT_OPS), "forbid_namespace": "/dp/flows.*",
                        "quantization": "FP16 only; no baked random or Range"},
    }
    if profile_path:
        manifest["profile_evidence"] = _profile_conditioner(profile_path)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--profile", type=Path)
    a = p.parse_args()
    print(json.dumps(extract(a.input, a.output_dir, a.profile), indent=2))


if __name__ == "__main__":
    main()

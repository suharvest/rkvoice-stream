#!/usr/bin/env python3
"""Inspect ONNX nodes and tensor shapes by substring.

This helper is intentionally generic because Kokoro split work depends on
exact internal tensor names exported by different ONNX variants.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


def _dims(value_info: onnx.ValueInfoProto) -> list[int | str]:
    dims: list[int | str] = []
    for dim in value_info.type.tensor_type.shape.dim:
        dims.append(dim.dim_value or dim.dim_param or "?")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--contains", action="append", required=True)
    parser.add_argument("--context", type=int, default=0)
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    shapes = {}
    for coll in (model.graph.input, model.graph.value_info, model.graph.output):
        for item in coll:
            shapes[item.name] = _dims(item)

    wanted = args.contains
    hits: set[int] = set()
    nodes = list(model.graph.node)
    for idx, node in enumerate(nodes):
        names = [node.name, node.op_type, *node.input, *node.output]
        if any(pattern in name for pattern in wanted for name in names):
            start = max(0, idx - args.context)
            end = min(len(nodes), idx + args.context + 1)
            hits.update(range(start, end))

    for idx in sorted(hits):
        node = nodes[idx]
        print(
            json.dumps(
                {
                    "idx": idx,
                    "op": node.op_type,
                    "name": node.name,
                    "inputs": [
                        {"name": name, "shape": shapes.get(name)}
                        for name in node.input
                        if name
                    ],
                    "outputs": [
                        {"name": name, "shape": shapes.get(name)}
                        for name in node.output
                        if name
                    ],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

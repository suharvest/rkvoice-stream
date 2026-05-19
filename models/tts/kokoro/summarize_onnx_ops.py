#!/usr/bin/env python3
"""Summarize ONNX op counts and large intermediate tensors."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx


def _dims(value_info) -> list[int]:
    return [int(dim.dim_value or 0) for dim in value_info.type.tensor_type.shape.dim]


def _numel(dims: list[int]) -> int:
    total = 1
    for dim in dims:
        total *= dim or 1
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--large-numel", type=int, default=500_000)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    model = onnx.load(str(args.model))
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {}
    for coll in (model.graph.input, model.graph.value_info, model.graph.output):
        for item in coll:
            shapes[item.name] = _dims(item)

    op_counts = collections.Counter(node.op_type for node in model.graph.node)
    large = []
    for idx, node in enumerate(model.graph.node):
        outs = []
        for out in node.output:
            dims = shapes.get(out, [])
            outs.append({"name": out, "shape": dims, "numel": _numel(dims)})
        max_numel = max((item["numel"] for item in outs), default=0)
        if max_numel >= args.large_numel:
            large.append(
                {
                    "idx": idx,
                    "op": node.op_type,
                    "name": node.name,
                    "max_numel": max_numel,
                    "outputs": outs,
                }
            )

    result = {
        "nodes": len(model.graph.node),
        "op_counts": dict(op_counts.most_common()),
        "large_threshold": args.large_numel,
        "large_count": len(large),
        "large_by_op": dict(collections.Counter(item["op"] for item in large).most_common()),
        "large_top": large[: args.top],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

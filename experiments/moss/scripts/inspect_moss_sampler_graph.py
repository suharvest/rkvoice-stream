#!/usr/bin/env python3
"""Inspect MOSS sampler ONNX graph boundaries for RKNN island candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import onnx


def _value_info(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    tensors = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for value in tensors:
        t = value.type.tensor_type
        dims: list[int | str] = []
        if t.HasField("shape"):
            for dim in t.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                elif dim.HasField("dim_param"):
                    dims.append(str(dim.dim_param))
                else:
                    dims.append("?")
        info[value.name] = {"elem_type": int(t.elem_type), "shape": dims}
    for init in model.graph.initializer:
        info.setdefault(init.name, {"elem_type": int(init.data_type), "shape": list(init.dims)})
    return info


def _node_row(node: onnx.NodeProto, value_info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": node.name,
        "op_type": node.op_type,
        "inputs": [{"name": name, **value_info.get(name, {})} for name in node.input],
        "outputs": [{"name": name, **value_info.get(name, {})} for name in node.output],
    }


def _match_nodes(model: onnx.ModelProto, patterns: list[str]) -> list[onnx.NodeProto]:
    lowered = [pattern.lower() for pattern in patterns]
    matches = []
    for node in model.graph.node:
        haystack = " ".join([node.name, node.op_type, *node.input, *node.output]).lower()
        if any(pattern in haystack for pattern in lowered):
            matches.append(node)
    return matches


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    return consumers


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.output:
            producers[name] = node
    return producers


def _walk_consumers(
    start: str,
    consumers: dict[str, list[onnx.NodeProto]],
    value_info: dict[str, dict[str, Any]],
    max_depth: int,
    max_nodes: int,
) -> list[dict[str, Any]]:
    queue: deque[tuple[str, int]] = deque([(start, 0)])
    seen_nodes: set[str] = set()
    rows: list[dict[str, Any]] = []
    while queue and len(rows) < max_nodes:
        value, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for node in consumers.get(value, []):
            key = node.name or "|".join(node.output)
            if key in seen_nodes:
                continue
            seen_nodes.add(key)
            row = _node_row(node, value_info)
            row["depth"] = depth + 1
            rows.append(row)
            for output in node.output:
                queue.append((output, depth + 1))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--patterns",
        default="/mlp/fc_in,/mlp/fc_out,/text_lm_head,/lm_head,TopK,Softmax",
        help="Comma-separated case-insensitive node/value patterns to report.",
    )
    parser.add_argument("--walk-from", default="", help="Optional tensor name to walk forward from.")
    parser.add_argument("--walk-depth", type=int, default=3)
    parser.add_argument("--max-walk-nodes", type=int, default=80)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    info = _value_info(model)
    patterns = [item.strip() for item in args.patterns.split(",") if item.strip()]
    matches = _match_nodes(model, patterns)
    consumers = _consumer_map(model)
    producers = _producer_map(model)

    result: dict[str, Any] = {
        "onnx": str(args.onnx),
        "inputs": [{"name": item.name, **info.get(item.name, {})} for item in model.graph.input],
        "outputs": [{"name": item.name, **info.get(item.name, {})} for item in model.graph.output],
        "node_count": len(model.graph.node),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node).most_common()),
        "patterns": patterns,
        "matches": [_node_row(node, info) for node in matches],
    }
    if args.walk_from:
        producer = producers.get(args.walk_from)
        result["walk_from"] = {
            "tensor": args.walk_from,
            "producer": _node_row(producer, info) if producer is not None else None,
            "consumers": _walk_consumers(args.walk_from, consumers, info, args.walk_depth, args.max_walk_nodes),
        }

    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

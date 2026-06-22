#!/usr/bin/env python3
"""Inspect MOSS codec decode-step ONNX graph for RKNN conversion blockers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import onnx


def _shape_from_value(value: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    dims: list[int | str] = []
    if tensor_type.HasField("shape"):
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            elif dim.HasField("dim_param"):
                dims.append(str(dim.dim_param))
            else:
                dims.append("?")
    return {"elem_type": int(tensor_type.elem_type), "shape": dims}


def _value_info(model: onnx.ModelProto) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        info[value.name] = _shape_from_value(value)
    for init in model.graph.initializer:
        info.setdefault(
            init.name,
            {
                "elem_type": int(init.data_type),
                "shape": list(init.dims),
                "initializer": True,
            },
        )
    return info


def _graph_value_info(graph: onnx.GraphProto) -> dict[str, dict[str, Any]]:
    model = onnx.ModelProto()
    model.graph.CopyFrom(graph)
    return _value_info(model)


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output in node.output:
            producers[output] = node
    return producers


def _consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    return consumers


def _node_summary(node: onnx.NodeProto, info: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": node.name,
        "op_type": node.op_type,
        "inputs": [{"name": name, **info.get(name, {})} for name in node.input],
        "outputs": [{"name": name, **info.get(name, {})} for name in node.output],
    }


def _upstream_ops(
    tensor: str,
    producers: dict[str, onnx.NodeProto],
    *,
    max_depth: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    queue: deque[tuple[str, int]] = deque([(tensor, 0)])
    seen: set[str] = set()
    while queue:
        name, depth = queue.popleft()
        if depth >= max_depth:
            continue
        producer = producers.get(name)
        if producer is None:
            continue
        key = producer.name or "|".join(producer.output)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "depth": depth + 1,
                "name": producer.name,
                "op_type": producer.op_type,
                "outputs": list(producer.output),
            }
        )
        for input_name in producer.input:
            queue.append((input_name, depth + 1))
    return rows


def _inspect_graph(graph: onnx.GraphProto, graph_name: str) -> dict[str, Any]:
    info = _graph_value_info(graph)
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    matmuls: list[dict[str, Any]] = []
    matmul_rank_counts: Counter[tuple[int | None, int | None]] = Counter()
    risky_matmuls: list[dict[str, Any]] = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        input_shapes = [info.get(name, {}).get("shape") for name in node.input]
        ranks = tuple(len(shape) if isinstance(shape, list) else None for shape in input_shapes)
        matmul_rank_counts[ranks] += 1
        row = {
            **_node_summary(node, info),
            "input_ranks": list(ranks),
            "upstream": {
                node.input[0]: _upstream_ops(node.input[0], producers, max_depth=3) if node.input else [],
                node.input[1]: _upstream_ops(node.input[1], producers, max_depth=3) if len(node.input) > 1 else [],
            },
            "consumer_ops": [
                {
                    "name": consumer.name,
                    "op_type": consumer.op_type,
                    "outputs": list(consumer.output),
                }
                for output in node.output
                for consumer in consumers.get(output, [])
            ],
        }
        matmuls.append(row)
        if any(rank == 3 for rank in ranks) or ranks[0] != ranks[1]:
            risky_matmuls.append(row)

    if_nodes: list[dict[str, Any]] = []
    bool_logic_nodes: list[dict[str, Any]] = []
    for node in graph.node:
        if node.op_type not in {"Equal", "Xor", "Not", "And", "Or", "If"}:
            continue
        bool_logic_nodes.append(
            {
                **_node_summary(node, info),
                "upstream": {name: _upstream_ops(name, producers, max_depth=4) for name in node.input},
                "consumer_ops": [
                    {
                        "name": consumer.name,
                        "op_type": consumer.op_type,
                        "outputs": list(consumer.output),
                    }
                    for output in node.output
                    for consumer in consumers.get(output, [])
                ],
            }
        )
    for node in graph.node:
        if node.op_type != "If":
            continue
        branches = []
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                branches.append(
                    {
                        "name": attr.name,
                        "node_count": len(attr.g.node),
                        "op_counts": dict(Counter(child.op_type for child in attr.g.node).most_common()),
                    }
                )
        if_nodes.append({**_node_summary(node, info), "branches": branches})

    return {
        "name": graph_name,
        "node_count": len(graph.node),
        "op_counts": dict(Counter(node.op_type for node in graph.node).most_common()),
        "inputs": [{"name": value.name, **info.get(value.name, {})} for value in graph.input],
        "outputs": [{"name": value.name, **info.get(value.name, {})} for value in graph.output],
        "matmul_rank_counts": {str(key): value for key, value in matmul_rank_counts.items()},
        "matmul_count": len(matmuls),
        "risky_matmul_count": len(risky_matmuls),
        "risky_matmuls": risky_matmuls,
        "matmuls": matmuls,
        "if_count": len(if_nodes),
        "if_nodes": if_nodes,
        "bool_logic_nodes": bool_logic_nodes,
    }


def _subgraphs(model: onnx.ModelProto) -> list[tuple[str, onnx.GraphProto]]:
    graphs: list[tuple[str, onnx.GraphProto]] = []
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                graphs.append((f"{node.name or node.op_type}.{attr.name}", attr.g))
    return graphs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--include-all-matmuls", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    top = _inspect_graph(model.graph, "main")
    if not args.include_all_matmuls:
        top.pop("matmuls", None)
    subgraphs = []
    for name, graph in _subgraphs(model):
        item = _inspect_graph(graph, name)
        if not args.include_all_matmuls:
            item.pop("matmuls", None)
        subgraphs.append(item)
    report = {
        "onnx": str(args.onnx),
        "main": top,
        "subgraphs": subgraphs,
        "summary": {
            "subgraph_count": len(subgraphs),
            "total_if_nodes": top["if_count"] + sum(item["if_count"] for item in subgraphs),
            "total_risky_matmuls": top["risky_matmul_count"] + sum(item["risky_matmul_count"] for item in subgraphs),
            "main_risky_matmuls": top["risky_matmul_count"],
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

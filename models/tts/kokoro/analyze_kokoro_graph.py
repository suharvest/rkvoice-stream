#!/usr/bin/env python3
"""Inspect Kokoro fixed ONNX graph for RKNN hybrid split points."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import onnx


def _dims(value_info) -> list[str | int]:
    shape = value_info.type.tensor_type.shape
    return [d.dim_value or d.dim_param or "?" for d in shape.dim]


def _group(name: str) -> str:
    if name.startswith("/"):
        parts = name.split("/")
        return "/" + parts[1] if len(parts) > 1 else "/"
    return "<root>"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=Path)
    parser.add_argument("--needle", default="text_encoder")
    args = parser.parse_args()

    model = onnx.load(str(args.onnx))
    producer = {}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node
        for inp in node.input:
            if inp:
                consumers[inp].append(node)

    vi = {}
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in coll:
            vi[item.name] = item

    print("== inputs ==")
    for item in model.graph.input:
        print(item.name, _dims(item), item.type.tensor_type.elem_type)
    print("== outputs ==")
    for item in model.graph.output:
        print(item.name, _dims(item), item.type.tensor_type.elem_type)

    print("== node groups ==")
    group_counts = Counter(_group(n.name) for n in model.graph.node)
    for name, count in group_counts.most_common():
        print(f"{name:40s} {count}")

    print(f"== nodes matching {args.needle!r} ==")
    for node in model.graph.node:
        if args.needle in node.name:
            print(node.op_type, node.name)
            print("  in ", list(node.input))
            print("  out", list(node.output))
            for out in node.output:
                if out in vi:
                    print("   shape", out, _dims(vi[out]))
                for c in consumers.get(out, []):
                    if _group(c.name) != _group(node.name):
                        print("   cross-consumer", c.op_type, c.name)

    print("== cross-group tensors ==")
    seen = set()
    for tensor, prod in producer.items():
        prod_group = _group(prod.name)
        for cons in consumers.get(tensor, []):
            cons_group = _group(cons.name)
            if prod_group != cons_group and (prod_group == "/text_encoder" or cons_group == "/text_encoder"):
                key = (tensor, prod.name, cons.name)
                if key in seen:
                    continue
                seen.add(key)
                shape = _dims(vi[tensor]) if tensor in vi else ["?"]
                print(f"{tensor} {shape}")
                print(f"  {prod_group}: {prod.op_type} {prod.name}")
                print(f"  {cons_group}: {cons.op_type} {cons.name}")

    print("== external inputs by group ==")
    for target_group in sorted(set(_group(n.name) for n in model.graph.node)):
        seen = set()
        rows = []
        for node in model.graph.node:
            if _group(node.name) != target_group:
                continue
            for inp in node.input:
                if not inp or inp in seen:
                    continue
                prod = producer.get(inp)
                prod_group = _group(prod.name) if prod is not None else "GRAPH/INIT"
                if prod_group == target_group:
                    continue
                seen.add(inp)
                rows.append((
                    inp,
                    _dims(vi[inp]) if inp in vi else ["?"],
                    prod_group,
                    prod.name if prod is not None else "",
                ))
        if rows:
            print(f"-- {target_group} --")
            for tensor, shape, prod_group, prod_name in rows[:80]:
                print(f"{tensor} {shape} from {prod_group} {prod_name}")
            if len(rows) > 80:
                print(f"... {len(rows) - 80} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

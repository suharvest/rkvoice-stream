#!/usr/bin/env python3
"""Inspect MOSS prefill ONNX tensor names needed for hybrid RKNN splitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx


def _suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


def _layer_tensors(layer: int) -> dict[str, str]:
    suffix = _suffix(layer)
    return {
        "block_input": f"/Add_{14 + layer * 5}_output_0" if layer else "/Add_14_output_0",
        "attn_residual": f"/Add_{19 + layer * 5}_output_0",
        "mlp_output": f"/mlp/fc_out{suffix}/Add_output_0",
        "block_output": f"/Add_{20 + layer * 5}_output_0",
    }


def inspect_graph(path: Path, context: int, tensors: list[str] | None = None) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    nodes = list(model.graph.node)
    producers = {out: idx for idx, node in enumerate(nodes) for out in node.output}
    consumers: dict[str, list[int]] = {}
    for idx, node in enumerate(nodes):
        for name in node.input:
            consumers.setdefault(name, []).append(idx)

    layers = []
    for layer in range(12):
        tensors = _layer_tensors(layer)
        layer_report: dict[str, Any] = {"layer": layer, "tensors": tensors, "anchors": {}}
        anchor_indices = sorted(
            {
                producers[name]
                for name in tensors.values()
                if name in producers
            }
            | {
                idx
                for name in tensors.values()
                for idx in consumers.get(name, [])
            }
        )
        snippets = []
        seen = set()
        for idx in anchor_indices:
            for j in range(max(0, idx - context), min(len(nodes), idx + context + 1)):
                if j in seen:
                    continue
                seen.add(j)
                node = nodes[j]
                snippets.append(
                    {
                        "index": j,
                        "name": node.name,
                        "op_type": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                    }
                )
        for key, name in tensors.items():
            layer_report["anchors"][key] = {
                "producer": producers.get(name),
                "consumers": consumers.get(name, []),
            }
        layer_report["snippets"] = snippets
        layers.append(layer_report)

    tensor_reports = {}
    for name in tensors or []:
        anchor_indices = sorted({producers[name]} if name in producers else set()) + consumers.get(name, [])
        seen = set()
        snippets = []
        for idx in sorted(set(anchor_indices)):
            for j in range(max(0, idx - context), min(len(nodes), idx + context + 1)):
                if j in seen:
                    continue
                seen.add(j)
                node = nodes[j]
                snippets.append(
                    {
                        "index": j,
                        "name": node.name,
                        "op_type": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                    }
                )
        tensor_reports[name] = {
            "producer": producers.get(name),
            "consumers": consumers.get(name, []),
            "snippets": snippets,
        }

    return {
        "onnx": str(path),
        "num_nodes": len(nodes),
        "inputs": [item.name for item in model.graph.input],
        "outputs": [item.name for item in model.graph.output],
        "tensors": tensor_reports,
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = inspect_graph(args.onnx, args.context, args.tensor)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

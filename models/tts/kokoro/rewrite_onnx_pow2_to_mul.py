#!/usr/bin/env python3
"""Rewrite ONNX Pow(x, 2) nodes to Mul(x, x)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _scalar_initializer_values(model: onnx.ModelProto) -> dict[str, float]:
    values: dict[str, float] = {}
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.size == 1:
            values[init.name] = float(arr.reshape(()))
    return values


def _constant_node_values(model: onnx.ModelProto) -> dict[str, float]:
    values: dict[str, float] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = numpy_helper.to_array(attr.t)
            if arr.size == 1:
                values[node.output[0]] = float(arr.reshape(()))
    return values


def rewrite(model: onnx.ModelProto) -> int:
    scalar_values = _scalar_initializer_values(model)
    scalar_values.update(_constant_node_values(model))

    rewritten = 0
    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.op_type == "Pow" and len(node.input) >= 2:
            exponent = scalar_values.get(node.input[1])
            if exponent is not None and np.isclose(exponent, 2.0):
                new_nodes.append(
                    helper.make_node(
                        "Mul",
                        inputs=[node.input[0], node.input[0]],
                        outputs=list(node.output),
                        name=f"{node.name}_pow2_mul" if node.name else "",
                    )
                )
                rewritten += 1
                continue
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--infer-shapes", action="store_true")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    rewritten = rewrite(model)
    if args.infer_shapes:
        model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    print(f"rewritten={rewritten}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

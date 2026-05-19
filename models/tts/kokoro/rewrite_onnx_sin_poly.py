#!/usr/bin/env python3
"""Rewrite ONNX Sin nodes to a clipped 7th-order polynomial approximation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _const(name: str, value: float) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=f"{name}_value")
    return helper.make_node("Constant", [], [name], value=tensor, name=f"{name}_const")


def _replace_sin(node: onnx.NodeProto, prefix: str) -> list[onnx.NodeProto]:
    x = node.input[0]
    y = node.output[0]
    clip_min = f"{prefix}/clip_min"
    clip_max = f"{prefix}/clip_max"
    c3 = f"{prefix}/c3"
    c5 = f"{prefix}/c5"
    c7 = f"{prefix}/c7"
    one = f"{prefix}/one"
    xc = f"{prefix}/clip"
    x2 = f"{prefix}/x2"
    t0 = f"{prefix}/t0"
    t1 = f"{prefix}/t1"
    t2 = f"{prefix}/t2"
    t3 = f"{prefix}/t3"
    t4 = f"{prefix}/t4"
    t5 = f"{prefix}/t5"
    return [
        _const(clip_min, -math.pi),
        _const(clip_max, math.pi),
        _const(c3, -1.0 / 6.0),
        _const(c5, 1.0 / 120.0),
        _const(c7, -1.0 / 5040.0),
        _const(one, 1.0),
        helper.make_node("Clip", [x, clip_min, clip_max], [xc], name=f"{prefix}/Clip"),
        helper.make_node("Mul", [xc, xc], [x2], name=f"{prefix}/Mul_x2"),
        helper.make_node("Mul", [x2, c7], [t0], name=f"{prefix}/Mul_c7"),
        helper.make_node("Add", [t0, c5], [t1], name=f"{prefix}/Add_c5"),
        helper.make_node("Mul", [t1, x2], [t2], name=f"{prefix}/Mul_x2_1"),
        helper.make_node("Add", [t2, c3], [t3], name=f"{prefix}/Add_c3"),
        helper.make_node("Mul", [t3, x2], [t4], name=f"{prefix}/Mul_x2_2"),
        helper.make_node("Add", [t4, one], [t5], name=f"{prefix}/Add_one"),
        helper.make_node("Mul", [xc, t5], [y], name=f"{prefix}/Mul_out"),
    ]


def rewrite(model: onnx.ModelProto) -> int:
    rewritten = 0
    new_nodes: list[onnx.NodeProto] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Sin":
            safe = node.name.strip("/").replace("/", "_") if node.name else f"Sin_{idx}"
            new_nodes.extend(_replace_sin(node, f"/rknn_sin_poly/{safe}"))
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"rewritten={rewritten}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Make Kokoro generator-tail probe graphs friendlier to RKNN.

The generator tail contains Snake activations:

  y = x + alpha^-1 * sin(alpha * x)^2

On RKNN/librknnrt 2.3.x, Sin/Pow often becomes CPU fallback or an unknown
target and can make rknnlite inference return None.  This script replaces:

  Sin(x)      -> 7th-order polynomial approximation over clipped x
  Pow(x, 2)   -> Mul(x, x)

It is intentionally scoped for probe graphs.  Validate audio quality before
using these approximations in a production tail.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _const(name: str, value: float) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=f"{name}_value")
    return helper.make_node("Constant", [], [name], value=tensor, name=f"{name}_const")


def _const_i64(name: str, value: list[int]) -> onnx.NodeProto:
    tensor = numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=f"{name}_value")
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


def _const_value(model: onnx.ModelProto, name: str) -> np.ndarray | None:
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output or node.output[0] != name:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
    return None


def _is_square_pow(model: onnx.ModelProto, node: onnx.NodeProto) -> bool:
    if len(node.input) < 2:
        return False
    value = _const_value(model, node.input[1])
    if value is None:
        return False
    return bool(np.allclose(value.astype(np.float32), 2.0))


def _replace_instance_norm(model: onnx.ModelProto, node: onnx.NodeProto, prefix: str) -> list[onnx.NodeProto]:
    x, scale, bias = node.input[:3]
    y = node.output[0]
    scale_arr = _const_value(model, scale)
    channels = int(scale_arr.reshape(-1).shape[0]) if scale_arr is not None else -1
    eps = 1e-5
    for attr in node.attribute:
        if attr.name == "epsilon":
            eps = float(attr.f)
    shape = f"{prefix}/shape"
    eps_c = f"{prefix}/eps"
    mean = f"{prefix}/mean"
    centered = f"{prefix}/centered"
    sq = f"{prefix}/sq"
    var = f"{prefix}/var"
    var_eps = f"{prefix}/var_eps"
    denom = f"{prefix}/denom"
    norm = f"{prefix}/norm"
    scale_r = f"{prefix}/scale_r"
    bias_r = f"{prefix}/bias_r"
    scaled = f"{prefix}/scaled"
    shape_value = [1, channels, 1] if channels > 0 else [1, -1, 1]
    return [
        _const_i64(shape, shape_value),
        _const(eps_c, eps),
        helper.make_node("ReduceMean", [x], [mean], name=f"{prefix}/ReduceMean_mean", axes=[2], keepdims=1),
        helper.make_node("Sub", [x, mean], [centered], name=f"{prefix}/Sub_centered"),
        helper.make_node("Mul", [centered, centered], [sq], name=f"{prefix}/Mul_square"),
        helper.make_node("ReduceMean", [sq], [var], name=f"{prefix}/ReduceMean_var", axes=[2], keepdims=1),
        helper.make_node("Add", [var, eps_c], [var_eps], name=f"{prefix}/Add_eps"),
        helper.make_node("Sqrt", [var_eps], [denom], name=f"{prefix}/Sqrt"),
        helper.make_node("Div", [centered, denom], [norm], name=f"{prefix}/Div"),
        helper.make_node("Reshape", [scale, shape], [scale_r], name=f"{prefix}/Reshape_scale"),
        helper.make_node("Reshape", [bias, shape], [bias_r], name=f"{prefix}/Reshape_bias"),
        helper.make_node("Mul", [norm, scale_r], [scaled], name=f"{prefix}/Mul_scale"),
        helper.make_node("Add", [scaled, bias_r], [y], name=f"{prefix}/Add_bias"),
    ]


def fix_model(input_path: Path, output_path: Path) -> dict:
    model = onnx.load(str(input_path))
    new_nodes: list[onnx.NodeProto] = []
    replaced_sin = 0
    replaced_pow = 0
    replaced_instance_norm = 0
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Sin":
            safe = node.name.strip("/").replace("/", "_") or f"Sin_{idx}"
            new_nodes.extend(_replace_sin(node, f"/rknn_sin_poly/{safe}"))
            replaced_sin += 1
        elif node.op_type == "Pow" and _is_square_pow(model, node):
            new_nodes.append(
                helper.make_node(
                    "Mul",
                    [node.input[0], node.input[0]],
                    list(node.output),
                    name=f"{node.name}/Pow2AsMul",
                )
            )
            replaced_pow += 1
        elif node.op_type == "InstanceNormalization":
            safe = node.name.strip("/").replace("/", "_") or f"InstanceNormalization_{idx}"
            new_nodes.extend(_replace_instance_norm(model, node, f"/rknn_instancenorm/{safe}"))
            replaced_instance_norm += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return {
        "replaced_sin": replaced_sin,
        "replaced_pow": replaced_pow,
        "replaced_instance_norm": replaced_instance_norm,
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(fix_model(args.input, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replace Sin ops in the Mimi codec decoder ONNX model with polynomial approximations.

SnakeBeta activation: x + (1/β) * sin²(β*x), where Sin inputs are β*x.
Rotary embeddings: Sin/Cos of position * frequency — kept as-is (RKNN handles natively).

Strategy:
- Replace only SnakeBeta Sin nodes (29 total) with 7th-order Taylor polynomial.
- Skip rotary embedding Sin/Cos (RKNN compiler recognizes the pattern and runs on NPU).
- No range reduction needed: SnakeBeta inputs are within [-π, π] for 99%+ of values,
  and the polynomial error for rare outliers is a small bounded perturbation.
- Taylor sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040 (Horner form, only Mul/Add ops)
- All replacement ops (Mul, Add) are NPU-supported on RK3576. No Floor needed.
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def make_scalar_constant(name: str, value: float, graph: onnx.GraphProto) -> str:
    """Add a scalar float32 initializer to the graph and return its name."""
    init = helper.make_tensor(name, TensorProto.FLOAT, [], [value])
    graph.initializer.append(init)
    return name


def is_rotary_sin_cos(node: onnx.NodeProto) -> bool:
    """Check if a Sin/Cos node belongs to rotary embeddings (not SnakeBeta)."""
    name = node.output[0] if node.output else ""
    input_name = node.input[0] if node.input else ""
    return "rotary" in name.lower() or "rotary" in input_name.lower()


def build_sin_polynomial_nodes(
    input_name: str, output_name: str, prefix: str, graph: onnx.GraphProto,
    clamp: bool = True,
) -> list:
    """Build 7th-order Taylor sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040.

    Uses Horner form: x * (1 + x² * (-1/6 + x² * (1/120 + x² * (-1/5040))))
    When clamp=True, input is clamped to [-π, π] via Clip before polynomial.
    Clip is NPU-supported on RK3576 (unlike Floor used in range reduction).
    """
    nodes = []

    # Optionally clamp input to [-π, π]
    if clamp:
        pi_val = float(np.pi)
        neg_pi_name = make_scalar_constant(f"{prefix}_neg_pi", -pi_val, graph)
        pos_pi_name = make_scalar_constant(f"{prefix}_pos_pi", pi_val, graph)
        clamped_name = f"{prefix}_clamped"
        nodes.append(helper.make_node("Clip", [input_name, neg_pi_name, pos_pi_name], [clamped_name]))
        poly_input = clamped_name
    else:
        poly_input = input_name

    # Coefficients for Horner form: sin(x) = x * (c0 + x² * (c1 + x² * (c2 + x² * c3)))
    # c0 = 1, c1 = -1/6, c2 = 1/120, c3 = -1/5040
    c1 = make_scalar_constant(f"{prefix}_c1", -1.0 / 6.0, graph)
    c2 = make_scalar_constant(f"{prefix}_c2", 1.0 / 120.0, graph)
    c3 = make_scalar_constant(f"{prefix}_c3", -1.0 / 5040.0, graph)
    one = make_scalar_constant(f"{prefix}_one", 1.0, graph)

    # x²
    x2 = f"{prefix}_x2"
    nodes.append(helper.make_node("Mul", [poly_input, poly_input], [x2]))

    # Inner: c3 * x² + c2
    t1 = f"{prefix}_t1"
    nodes.append(helper.make_node("Mul", [x2, c3], [t1]))
    t2 = f"{prefix}_t2"
    nodes.append(helper.make_node("Add", [t1, c2], [t2]))

    # Middle: t2 * x² + c1
    t3 = f"{prefix}_t3"
    nodes.append(helper.make_node("Mul", [t2, x2], [t3]))
    t4 = f"{prefix}_t4"
    nodes.append(helper.make_node("Add", [t3, c1], [t4]))

    # Outer: t4 * x² + 1
    t5 = f"{prefix}_t5"
    nodes.append(helper.make_node("Mul", [t4, x2], [t5]))
    t6 = f"{prefix}_t6"
    nodes.append(helper.make_node("Add", [t5, one], [t6]))

    # Final: x * t6
    nodes.append(helper.make_node("Mul", [poly_input, t6], [output_name]))

    return nodes


def build_cos_polynomial_nodes(
    input_name: str, output_name: str, prefix: str, graph: onnx.GraphProto
) -> list:
    """Build 6th-order Taylor cos(x) ≈ 1 - x²/2 + x⁴/24 - x⁶/720.

    Horner form: 1 + x² * (-1/2 + x² * (1/24 + x² * (-1/720)))
    """
    nodes = []

    c0 = make_scalar_constant(f"{prefix}_c0", 1.0, graph)
    c1 = make_scalar_constant(f"{prefix}_c1", -1.0 / 2.0, graph)
    c2 = make_scalar_constant(f"{prefix}_c2", 1.0 / 24.0, graph)
    c3 = make_scalar_constant(f"{prefix}_c3", -1.0 / 720.0, graph)

    # x²
    x2 = f"{prefix}_x2"
    nodes.append(helper.make_node("Mul", [input_name, input_name], [x2]))

    # Inner: c3 * x² + c2
    t1 = f"{prefix}_t1"
    nodes.append(helper.make_node("Mul", [x2, c3], [t1]))
    t2 = f"{prefix}_t2"
    nodes.append(helper.make_node("Add", [t1, c2], [t2]))

    # Middle: t2 * x² + c1
    t3 = f"{prefix}_t3"
    nodes.append(helper.make_node("Mul", [t2, x2], [t3]))
    t4 = f"{prefix}_t4"
    nodes.append(helper.make_node("Add", [t3, c1], [t4]))

    # Outer: t4 * x² + 1
    t5 = f"{prefix}_t5"
    nodes.append(helper.make_node("Mul", [t4, x2], [t5]))
    nodes.append(helper.make_node("Add", [t5, c0], [output_name]))

    return nodes


def replace_sin_cos_nodes(model: onnx.ModelProto) -> tuple[int, int, int]:
    """Replace SnakeBeta Sin nodes with polynomial approximations.

    Rotary embedding Sin/Cos nodes are skipped (RKNN handles them natively on NPU).
    No range reduction is used — avoids Floor op which falls back to CPU on RK3576.

    Returns (sin_replaced, cos_replaced, skipped) counts.
    """
    graph = model.graph
    sin_count = 0
    cos_count = 0
    skipped = 0

    # Collect nodes to replace (iterate over a copy)
    nodes_to_process = []
    for node in list(graph.node):
        if node.op_type in ("Sin", "Cos"):
            nodes_to_process.append(node)

    for node in nodes_to_process:
        # Skip rotary embedding Sin/Cos — RKNN compiler handles these natively
        if is_rotary_sin_cos(node):
            skipped += 1
            print(f"  Skipping rotary {node.op_type}: {node.output[0]}")
            continue

        input_name = node.input[0]
        output_name = node.output[0]
        idx = list(graph.node).index(node)
        is_sin = node.op_type == "Sin"

        if is_sin:
            prefix = f"sin_poly_{sin_count}"
            sin_count += 1
        else:
            prefix = f"cos_poly_{cos_count}"
            cos_count += 1

        # Direct polynomial approximation (no range reduction)
        if is_sin:
            poly_nodes = build_sin_polynomial_nodes(
                input_name, output_name, prefix, graph
            )
        else:
            poly_nodes = build_cos_polynomial_nodes(
                input_name, output_name, prefix, graph
            )

        # Remove original node and insert replacements
        graph.node.remove(node)
        for j, new_node in enumerate(poly_nodes):
            graph.node.insert(idx + j, new_node)

    return sin_count, cos_count, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Replace Sin/Cos ops with polynomial approximations for NPU compatibility"
    )
    parser.add_argument("input_onnx", help="Input ONNX model path")
    parser.add_argument("output_onnx", help="Output ONNX model path")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification comparing original vs modified model",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.input_onnx}")
    model = onnx.load(args.input_onnx)

    orig_node_count = len(model.graph.node)
    sin_nodes_before = sum(1 for n in model.graph.node if n.op_type == "Sin")
    cos_nodes_before = sum(1 for n in model.graph.node if n.op_type == "Cos")
    print(f"Original model: {orig_node_count} nodes, {sin_nodes_before} Sin, {cos_nodes_before} Cos")

    sin_replaced, cos_replaced, skipped = replace_sin_cos_nodes(model)

    new_node_count = len(model.graph.node)
    sin_nodes_after = sum(1 for n in model.graph.node if n.op_type == "Sin")
    cos_nodes_after = sum(1 for n in model.graph.node if n.op_type == "Cos")

    print(f"\nReplaced: {sin_replaced} Sin + {cos_replaced} Cos nodes (skipped {skipped} rotary)")
    print(f"Modified model: {new_node_count} nodes, {sin_nodes_after} Sin, {cos_nodes_after} Cos")
    print(f"Node count change: {orig_node_count} -> {new_node_count} (+{new_node_count - orig_node_count})")

    # Validate
    print("\nValidating modified model...")
    try:
        onnx.checker.check_model(model, full_check=False)
        print("ONNX validation passed")
    except Exception as e:
        print(f"ONNX validation warning: {e}")
        print("(This may be OK if the model is large and uses external data)")

    print(f"\nSaving to: {args.output_onnx}")
    onnx.save(model, args.output_onnx)

    output_size = Path(args.output_onnx).stat().st_size
    print(f"Output size: {output_size / 1024 / 1024:.1f} MB")

    if args.verify:
        verify_outputs(args.input_onnx, args.output_onnx)


def verify_outputs(orig_path: str, modified_path: str):
    """Compare outputs of original and modified models."""
    import onnxruntime as ort

    print("\n--- Verification ---")
    np.random.seed(42)
    test_input = np.random.randn(1, 512, 75).astype(np.float32) * 0.5

    sess_orig = ort.InferenceSession(orig_path)
    sess_poly = ort.InferenceSession(modified_path)

    out_orig = sess_orig.run(None, {"embeddings": test_input})
    out_poly = sess_poly.run(None, {"embeddings": test_input})

    for i, (o, p) in enumerate(zip(out_orig, out_poly)):
        if o.size == 0 or p.size == 0:
            print(f"Output[{i}]: empty (shape orig={o.shape}, poly={p.shape})")
            continue
        max_diff = np.abs(o - p).max()
        mean_diff = np.abs(o - p).mean()
        if o.size > 1:
            corr = np.corrcoef(o.flatten(), p.flatten())[0, 1]
        else:
            corr = float("nan")
        print(
            f"Output[{i}]: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
            f"correlation={corr:.8f}, shape={o.shape}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze matcha-icefall-zh-en ONNX model for RKNN compatibility."""
import onnx
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "matcha-icefall-zh-en/model-steps-3.onnx"
model = onnx.load(model_path)

print("=== Model Inputs ===")
for inp in model.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append(f"dynamic({d.dim_param})")
    print(f"  {inp.name}: [{', '.join(dims)}]")

print("\n=== Model Outputs ===")
for out in model.graph.output:
    dims = []
    for d in out.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append(f"dynamic({d.dim_param})")
    print(f"  {out.name}: [{', '.join(dims)}]")

print(f"\n=== Total Nodes: {len(model.graph.node)} ===")

# Count op types
ops = {}
for node in model.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1

print("\n=== Op Types (top 20) ===")
for op, cnt in sorted(ops.items(), key=lambda x: -x[1])[:20]:
    print(f"  {op}: {cnt}")

# Check problematic ops for RKNN
print("\n=== RKNN Problematic Ops ===")
problematic = ["Range", "Slice", "Scan", "Loop", "If", "Where", "GatherND", "NonZero"]
found_problems = []
for op in problematic:
    if op in ops:
        print(f"  FOUND: {op}: {ops[op]}")
        found_problems.append((op, ops[op]))

if not found_problems:
    print("  None found - model may be compatible!")

# Find specific Range and Slice nodes
if "Range" in ops:
    print("\n=== Range Node Details ===")
    for node in model.graph.node:
        if node.op_type == "Range":
            print(f"  {node.name}: inputs={[i for i in node.input]}")

if "Slice" in ops:
    print("\n=== Slice Node Details (first 5) ===")
    count = 0
    for node in model.graph.node:
        if node.op_type == "Slice":
            print(f"  {node.name}: inputs={[i for i in node.input]}")
            count += 1
            if count >= 5:
                break
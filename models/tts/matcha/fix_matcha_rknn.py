#!/usr/bin/env python3
"""Fix Matcha TTS acoustic model ONNX for RKNN conversion (v2, probe-first).

This script applies a 5-step ONNX graph surgery pipeline to make
model-steps-3.onnx (from sherpa-onnx matcha-icefall-zh-baker) compatible
with RKNN conversion.

Key insight: RKNN does not support dynamic output shapes. The model's mel
output length depends on x_length (token count) via the duration predictor.
We bake the model at a specific x_len, producing a fixed-size mel output.
At runtime, the caller trims to the actual needed length.

Strategy (probe-first):
  1. Probe the ORIGINAL model at target x_len to capture all intermediate
     tensor shapes and values (Range outputs, RandomNormalLike noise shape)
  2. Run onnxsim with fixed input shapes
  3. Replace problematic ops using pre-probed shapes (avoids cascading
     shape corruption from patching the modified graph)

Surgery steps:
  Step 1: onnxsim with fixed shapes (SEQ_LEN, x_length=[1])
  Step 2: Replace Range nodes with pre-probed constant tensors
  Step 3: Replace Ceil with Neg(Floor(Neg(x))) [RKNN CPU op unsupported]
  Step 4: Fix Slice_2 dynamic ends with probed constant
  Step 5: Replace RandomNormalLike with pre-probed fixed noise (seed=42)

Recommended bucket sizes for RK3576 NPU:
  --seq-len  80 --x-len  64  ->  ~599 mel frames (~9.6s audio), 53MB, ~390ms
  --seq-len 160 --x-len 140  -> ~1278 mel frames (~20s audio), 60MB, ~900ms

Note: seq_len=256/x_len=224 produces ~2033 mel frames but exceeds RK3576
NPU's SDP attention memory limit, causing inference failure. Use seq_len<=160.

Usage:
  python fix_matcha_rknn.py --input model-steps-3.onnx --output model-fixed.onnx

Then convert with RKNN toolkit2:
  from rknn.api import RKNN
  rknn = RKNN()
  rknn.config(target_platform='rk3576', optimization_level=0, float_dtype='float16')
  rknn.load_onnx(model='model-fixed.onnx')
  rknn.build(do_quantization=False)
  rknn.export_rknn('matcha-acoustic.rknn')

The vocoder (vocos-vocoder.rknn) expects mel input of shape [1,80,2048].
Pad the mel output from the acoustic model before feeding to vocoder.
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort
from collections import Counter

# Default: medium bucket, ~600 mel frames, fits RK3576 NPU
SEQ_LEN = 80
X_LEN = 64


def probe_original(model_path: str, seq_len: int, x_len: int):
    """Probe the original model to get all dynamic tensor shapes/values.

    Returns:
        probed: dict mapping tensor name -> numpy array
        test_inputs: dict of model inputs used for probing
        ref_mel_shape: shape of the reference mel output
    """
    model = onnx.load(model_path)

    # Find all tensors we need to probe
    targets = {}
    for n in model.graph.node:
        if n.op_type == 'Range':
            targets[n.output[0]] = 'range'
        elif n.op_type == 'RandomNormalLike':
            targets[n.output[0]] = 'rnl'

    # Probe with ValueInfoProto (let ORT infer types)
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in targets:
        vi = onnx.ValueInfoProto()
        vi.name = name
        m_probe.graph.output.append(vi)

    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])

    x = np.zeros((1, seq_len), dtype=np.int64)
    x[0, :x_len] = (np.arange(x_len) % 100) + 1
    test_inputs = {
        'x': x,
        'x_length': np.array([x_len], dtype=np.int64),
        'noise_scale': np.array([0.667], dtype=np.float32),
        'length_scale': np.array([1.0], dtype=np.float32),
    }

    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)

    results = {}
    for name, val in zip([o.name for o in m_probe.graph.output], all_out):
        if name in targets:
            results[name] = val
            kind = targets[name]
            print(f"  Probed {kind} {name}: shape={val.shape}, dtype={val.dtype}")

    mel = all_out[0]
    print(f"  Reference mel: shape={mel.shape}")
    return results, test_inputs, mel.shape


def load_and_simplify(input_path: str, seq_len: int) -> onnx.ModelProto:
    """Step 1: Load and simplify with onnxsim."""
    import onnxsim
    model = onnx.load(input_path)
    print(f"  Original nodes: {len(model.graph.node)}")
    simplified, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={
            'x': [1, seq_len],
            'x_length': [1],
            'noise_scale': [1],
            'length_scale': [1],
        },
    )
    print(f"  Simplified: ok={ok}, nodes={len(simplified.graph.node)}")
    return simplified


def fix_range_nodes(model: onnx.ModelProto, probed_values: dict) -> onnx.ModelProto:
    """Step 2: Replace Range nodes with pre-probed constant values."""
    new_nodes = []
    count = 0
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Range' and n.output[0] in probed_values:
            arr = probed_values[n.output[0]]
            const = helper.make_node(
                'Constant', inputs=[], outputs=[n.output[0]],
                name=f'const_range_{i}',
                value=numpy_helper.from_array(arr, name=n.output[0])
            )
            new_nodes.insert(0, const)
            print(f"  Replaced Range {n.output[0]}: shape={arr.shape}")
            count += 1
        elif n.op_type == 'Range':
            print(f"  WARNING: Unprobed Range {n.output[0]}")
            new_nodes.append(n)
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  Total: {count} Range nodes replaced")
    return model


def fix_ceil_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    """Step 3: Replace Ceil(x) with Neg(Floor(Neg(x))).

    RKNN runtime doesn't support Ceil op on CPU.
    Mathematically equivalent: ceil(x) = -floor(-x)
    """
    new_nodes = []
    count = 0
    for n in model.graph.node:
        if n.op_type == 'Ceil':
            x_in, y_out = n.input[0], n.output[0]
            neg1 = f'{y_out}__neg1'
            flr = f'{y_out}__floor'
            new_nodes.extend([
                helper.make_node('Neg', [x_in], [neg1], name=f'{n.name}_neg1'),
                helper.make_node('Floor', [neg1], [flr], name=f'{n.name}_floor'),
                helper.make_node('Neg', [flr], [y_out], name=f'{n.name}_neg2'),
            ])
            count += 1
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  Replaced {count} Ceil nodes")
    return model


def fix_dynamic_slice_ends(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 4: Fix Slice nodes with dynamic index inputs.

    The dynamic end comes from the mel frame count computation.
    We probe the current model to get the value and bake it as a constant.
    """
    init_names = {init.name for init in model.graph.initializer}
    const_names = {out for n in model.graph.node if n.op_type == 'Constant' for out in n.output}
    static_names = init_names | const_names
    graph_input_names = {inp.name for inp in model.graph.input}

    dynamic_tensors = {}
    for n in model.graph.node:
        if n.op_type == 'Slice':
            for i in range(1, len(n.input)):
                inp = n.input[i]
                if inp and inp not in static_names and inp not in graph_input_names:
                    dynamic_tensors[inp] = None

    if not dynamic_tensors:
        print("  No dynamic Slice index inputs found")
        return model

    # Probe current model state
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in dynamic_tensors:
        vi = helper.make_tensor_value_info(name, TensorProto.INT64, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    try:
        sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
        all_out = sess.run(None, test_inputs)
        for name, val in zip([o.name for o in m_probe.graph.output], all_out):
            if name in dynamic_tensors:
                dynamic_tensors[name] = val
                print(f"  Probed {name} = {val}")
    except Exception as e:
        print(f"  Probe failed: {e}")
        return model
    finally:
        os.unlink(tmp)

    # Replace dynamic tensors with constants
    out_to_node = {out: n for n in model.graph.node for out in n.output}
    tensor_consumers = {}
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                tensor_consumers.setdefault(inp, []).append(n)

    nodes_to_remove = set()
    for name, val in dynamic_tensors.items():
        if val is None:
            continue
        model.graph.initializer.append(
            numpy_helper.from_array(val.astype(np.int64), name=name))
        prod = out_to_node.get(name)
        consumers = tensor_consumers.get(name, [])
        if prod and len(prod.output) == 1 and all(c.op_type in ('Slice',) for c in consumers):
            nodes_to_remove.add(id(prod))
            print(f"  Removed {prod.op_type} {prod.name}")

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return model


def fix_random_normal_like(model: onnx.ModelProto, probed_values: dict) -> onnx.ModelProto:
    """Step 5: Replace RandomNormalLike with fixed constant noise (seed=42).

    ODE noise is baked at compile time (deterministic).
    RKNN runtime doesn't support RandomNormalLike on CPU.
    Uses pre-probed shapes from the original model to avoid shape corruption.
    """
    rn_nodes = [n for n in model.graph.node if n.op_type == 'RandomNormalLike']
    if not rn_nodes:
        print("  No RandomNormalLike nodes found")
        return model

    rng = np.random.default_rng(42)
    nodes_to_remove = set()
    const_nodes = []

    for rn in rn_nodes:
        name = rn.output[0]
        if name in probed_values:
            shape = probed_values[name].shape
            noise = rng.standard_normal(size=shape).astype(np.float32)
            const_nodes.append(helper.make_node(
                'Constant', inputs=[], outputs=[name],
                name=f'const_{rn.name}',
                value=numpy_helper.from_array(noise, name=name)
            ))
            nodes_to_remove.add(id(rn))
            print(f"  Replaced RandomNormalLike {name}: shape={shape}")
        else:
            print(f"  WARNING: no probed shape for {name}")

    new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    for cn in const_nodes:
        new_nodes.insert(0, cn)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Fix Matcha TTS ONNX for RKNN conversion (probe-first approach)')
    parser.add_argument('--input', default='~/matcha-data/model-steps-3.onnx',
                        help='Input ONNX model path')
    parser.add_argument('--output', default='~/matcha-data/model-steps-3-rknn-ready.onnx',
                        help='Output fixed ONNX model path')
    parser.add_argument('--x-len', type=int, default=X_LEN,
                        help=f'Target x_length for baking (default: {X_LEN})')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN,
                        help=f'Max phoneme sequence length (default: {SEQ_LEN})')
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    seq_len = args.seq_len
    x_len = args.x_len
    if x_len >= seq_len:
        x_len = seq_len - 1

    print(f"=== Matcha ONNX RKNN Fix Pipeline (v2, probe-first) ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"SEQ_LEN={seq_len}, X_LEN={x_len}")

    # Step 0: Probe ORIGINAL model for intermediate shapes
    print("\n[0/5] Probing original model...")
    probed, test_inputs, ref_mel_shape = probe_original(input_path, seq_len, x_len)

    # Step 1: onnxsim
    print("\n[1/5] onnxsim simplification...")
    model = load_and_simplify(input_path, seq_len)

    # Step 2: Range nodes (use pre-probed values)
    print("\n[2/5] Replacing Range nodes...")
    model = fix_range_nodes(model, probed)

    # Step 3: Ceil ops
    print("\n[3/5] Replacing Ceil ops...")
    model = fix_ceil_ops(model)

    # Step 4: Dynamic Slice ends (probe current model state)
    print("\n[4/5] Fixing dynamic Slice index inputs...")
    model = fix_dynamic_slice_ends(model, test_inputs)

    # Step 5: RandomNormalLike (use pre-probed shapes)
    print("\n[5/5] Replacing RandomNormalLike...")
    model = fix_random_normal_like(model, probed)

    # Shape inference
    print("\nRunning shape inference...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("  OK")
    except Exception as e:
        print(f"  Warning: {e}")

    for o in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"  Output: {o.name} shape={dims}")

    # ORT verify
    print("\nVerifying with ORT...")
    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        out = sess.run(None, test_inputs)
        mel = out[0]
        print(f"  x_len={x_len}: mel shape={mel.shape}, RMS={float(np.sqrt(np.mean(mel**2))):.4f}")
        print(f"  Reference shape: {ref_mel_shape}")
        if mel.shape == ref_mel_shape:
            print("  Shape MATCH!")
        else:
            print(f"  Shape MISMATCH: got {mel.shape} expected {ref_mel_shape}")

        # Compare with original
        sess_orig = ort.InferenceSession(input_path, providers=['CPUExecutionProvider'])
        mel_orig = sess_orig.run(None, test_inputs)[0]
        N = min(mel.shape[2], mel_orig.shape[2])
        m1, m2 = mel_orig[:, :, :N], mel[:, :, :N]
        cos = np.sum(m1 * m2) / (np.linalg.norm(m1) * np.linalg.norm(m2) + 1e-8)
        print(f"  Cosine similarity vs original (first {N} frames): {cos:.6f}")

        # Test variable x_lengths
        print("\n  Variable x_length test:")
        for xl in [4, 15, 50, 100, x_len]:
            if xl >= seq_len:
                continue
            x = np.zeros((1, seq_len), dtype=np.int64)
            x[0, :xl] = (np.arange(xl) % 100) + 1
            ti = {
                'x': x,
                'x_length': np.array([xl], dtype=np.int64),
                'noise_scale': np.array([0.667], dtype=np.float32),
                'length_scale': np.array([1.0], dtype=np.float32),
            }
            try:
                out = sess.run(None, ti)
                rms = float(np.sqrt(np.mean(out[0] ** 2)))
                print(f"    x_len={xl:3d}: shape={out[0].shape}, rms={rms:.4f}")
            except Exception as e:
                print(f"    x_len={xl:3d}: FAILED - {e}")

        print("\n  ORT OK")
    except Exception as e:
        print(f"  ORT FAIL: {e}")
        sys.exit(1)

    # Check for remaining problematic ops
    for op in ['Range', 'Ceil', 'RandomNormalLike']:
        n = sum(1 for node in model.graph.node if node.op_type == op)
        if n:
            print(f"  WARNING: {n} {op} nodes remain")

    # Save
    onnx.save(model, output_path)
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({sz:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()

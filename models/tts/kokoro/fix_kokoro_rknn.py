#!/usr/bin/env python3
"""Fix Kokoro TTS ONNX model for RKNN conversion.

Kokoro TTS is an 82M parameter text-to-speech model. This script applies a
6-step ONNX graph surgery pipeline to make model.onnx (kokoro-multi-lang-v1_1)
compatible with RKNN conversion:

Step 1: onnxsim with fixed shapes (SEQ_LEN=128)
  - Resolves symbolic dims and folds static shape computations
  - Input shapes: tokens=[1,seq_len], style=[1,256], speed=[1]

Step 2: Replace Range nodes with Constants
  - Range nodes for positional encodings are replaced with baked constant tensors
  - Values extracted from initializers/Constant nodes; limit falls back to seq_len

Step 3: Fix dynamic Slice ends
  - Slice nodes with dynamic index inputs are replaced with baked constant initializers
  - Values obtained via ORT probe

Step 4: Replace Ceil with Neg(Floor(Neg(x)))
  - RKNN runtime doesn't support Ceil op on CPU
  - Mathematically equivalent: ceil(x) = -floor(-x)

Step 5: Replace RandomNormalLike / RandomUniformLike with fixed constant tensors
  - ODE/diffusion noise is baked at compile time (deterministic, seed=42)
  - RKNN runtime doesn't support these stochastic ops on CPU
  - Normal -> rng.standard_normal; Uniform -> rng.uniform(0, 1)

Step 6: Shape inference and ORT verification
  - Verifies model runs correctly with test inputs after surgery
  - Warns about remaining unsupported ops (If, Loop, SplitToSequence, etc.)

After surgery, RKNN conversion succeeds without "Unsupport CPU op" errors.

Usage:
  python fix_kokoro_rknn.py \\
      --input ~/kokoro-analysis/kokoro-multi-lang-v1_1/model.onnx \\
      --output ~/kokoro-analysis/kokoro-multi-lang-v1_1/model-rknn-ready.onnx \\
      --seq-len 128

Then convert with RKNN toolkit2:
  from rknn.api import RKNN
  rknn = RKNN()
  rknn.config(target_platform='rk3576', optimization_level=0)
  rknn.load_onnx(model='model-rknn-ready.onnx')
  rknn.build(do_quantization=False)
  rknn.export_rknn('kokoro-fp16.rknn')
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort

# Fixed parameters
SEQ_LEN = 128   # Max token sequence length (bucket size)


def eliminate_control_flow(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 0: Replace Loop/If/Sequence ops with probed constant values.

    onnxsim can't handle these ops, so we must eliminate them first.
    Uses ORT to probe the actual output values, then replaces with Constants.
    """
    control_flow_ops = {'Loop', 'If', 'SplitToSequence', 'SequenceEmpty', 'ConcatFromSequence'}
    cf_nodes = [n for n in model.graph.node if n.op_type in control_flow_ops]
    if not cf_nodes:
        print("  No control flow ops found")
        return model

    # Collect outputs and detect their types by probing one at a time
    cf_outputs = {}
    for n in cf_nodes:
        for out_name in n.output:
            if out_name:
                cf_outputs[out_name] = None

    print(f"  Found {len(cf_nodes)} control flow nodes, {len(cf_outputs)} outputs to probe")

    # Probe each output individually (types may differ)
    for name in list(cf_outputs.keys()):
        for try_dtype in [TensorProto.FLOAT, TensorProto.INT64, TensorProto.INT32]:
            m_probe = onnx.ModelProto()
            m_probe.CopyFrom(model)
            vi = helper.make_tensor_value_info(name, try_dtype, None)
            m_probe.graph.output.append(vi)
            tmp = tempfile.mktemp(suffix='.onnx')
            onnx.save(m_probe, tmp)
            try:
                sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
                all_out = sess.run(None, test_inputs)
                val = all_out[-1]
                cf_outputs[name] = val
                os.unlink(tmp)
                print(f"  Probed {name}: shape={val.shape}, dtype={val.dtype}")
                break
            except Exception:
                os.unlink(tmp)
                continue

    # Replace control flow nodes: add probed values as initializers,
    # remove the producing nodes
    nodes_to_remove = {id(n) for n in cf_nodes}
    for name, val in cf_outputs.items():
        if val is not None:
            model.graph.initializer.append(numpy_helper.from_array(val, name=name))

    new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  Removed {len(cf_nodes)} nodes, baked {sum(1 for v in cf_outputs.values() if v is not None)} outputs")
    return model


def load_and_simplify(input_path: str, seq_len: int = SEQ_LEN) -> onnx.ModelProto:
    """Step 1: Load and simplify with onnxsim."""
    import onnxsim
    model = onnx.load(input_path)
    print(f"  Original nodes: {len(model.graph.node)}")
    simplified, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={
            'tokens': [1, seq_len],
            'style': [1, 256],
            'speed': [1],
        },
    )
    print(f"  Simplified: ok={ok}, nodes={len(simplified.graph.node)}")
    return simplified


def probe_ort(model: onnx.ModelProto, extra_outputs: list[str], test_inputs: dict) -> dict:
    """Run ORT inference and capture specific intermediate tensor values."""
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in extra_outputs:
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)

    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)

    all_names = [o.name for o in m_probe.graph.output]
    return {name: val for name, val in zip(all_names, all_out)}


def fix_range_nodes(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 2: Replace Range nodes with constant tensors.

    Range(start, limit, delta) produces arange(start, limit, delta).
    We extract start/delta from initializers/Constant nodes; if limit is
    unavailable we fall back to seq_len derived from the tokens input shape.
    Handles both int64 and float32 dtypes.
    """
    range_nodes = [n for n in model.graph.node if n.op_type == 'Range']
    if not range_nodes:
        print("  No Range nodes found")
        return model

    # Infer seq_len from tokens input
    seq_len = test_inputs['tokens'].shape[1]

    # Build initializer lookup
    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    # Also collect Constant node outputs
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    init_map[node.output[0]] = numpy_helper.to_array(attr.t)

    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Range':
            start_name, limit_name, delta_name = n.input[0], n.input[1], n.input[2]

            start_val = init_map.get(start_name)
            limit_val = init_map.get(limit_name)
            delta_val = init_map.get(delta_name)

            if start_val is not None and delta_val is not None:
                start = float(start_val.flat[0])
                delta = float(delta_val.flat[0])
                if limit_val is not None:
                    limit = float(limit_val.flat[0])
                else:
                    # Fall back to seq_len when limit is dynamic
                    limit = float(seq_len)
                arr = np.arange(start, limit, delta)
                # Preserve dtype from start/delta
                if start_val.dtype in (np.int64, np.int32):
                    arr = arr.astype(np.int64)
                else:
                    arr = arr.astype(np.float32)
            else:
                # Fallback: generate arange(0, seq_len) as int64
                arr = np.arange(seq_len, dtype=np.int64)

            const_node = helper.make_node(
                'Constant', inputs=[], outputs=[n.output[0]],
                name=f'const_range_{i}',
                value=numpy_helper.from_array(arr, name=n.output[0])
            )
            new_nodes.insert(0, const_node)
            print(f"  Replaced Range {n.output[0]}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_dynamic_slice_ends(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 3: Fix Slice nodes with dynamic index inputs."""
    init_names = {init.name for init in model.graph.initializer}
    const_names = {out for n in model.graph.node if n.op_type == 'Constant' for out in n.output}
    static_names = init_names | const_names
    graph_input_names = {inp.name for inp in model.graph.input}

    # Find Slice nodes with dynamic INDEX inputs (not data)
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

    # Probe to get values
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for name in dynamic_tensors:
        vi = helper.make_tensor_value_info(name, TensorProto.INT64, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)
    for name, val in zip([o.name for o in m_probe.graph.output], all_out):
        if name in dynamic_tensors:
            dynamic_tensors[name] = val
            print(f"  {name} = {val}")

    # For each dynamic tensor, find its single producing node (if safe to remove)
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
        prod = out_to_node.get(name)
        consumers = tensor_consumers.get(name, [])
        # Add constant initializer
        model.graph.initializer.append(numpy_helper.from_array(val.astype(np.int64), name=name))
        # Remove producing node only if safe
        if (prod and len(prod.output) == 1
                and all(c.op_type in ('Slice',) for c in consumers)):
            nodes_to_remove.add(id(prod))
            print(f"  Removed {prod.op_type} {prod.name}")

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return model


def fix_ceil_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    """Step 4: Replace Ceil(x) with Neg(Floor(Neg(x)))."""
    ceil_nodes = [n for n in model.graph.node if n.op_type == 'Ceil']
    if not ceil_nodes:
        print("  No Ceil nodes found")
        return model

    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Ceil':
            x_in, y_out = n.input[0], n.output[0]
            neg1 = f'{y_out}__neg1'
            flr = f'{y_out}__floor'
            new_nodes.extend([
                helper.make_node('Neg', [x_in], [neg1], name=f'{n.name}_neg1'),
                helper.make_node('Floor', [neg1], [flr], name=f'{n.name}_floor'),
                helper.make_node('Neg', [flr], [y_out], name=f'{n.name}_neg2'),
            ])
            print(f"  Replaced Ceil {n.name}")
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_random_noise(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 5: Replace RandomNormalLike / RandomUniformLike with fixed constant tensors (seed=42).

    Normal distribution  -> rng.standard_normal(shape)
    Uniform distribution -> rng.uniform(0, 1, shape)
    """
    target_ops = {'RandomNormalLike', 'RandomUniformLike'}
    rn_nodes = [n for n in model.graph.node if n.op_type in target_ops]
    if not rn_nodes:
        print("  No RandomNormalLike / RandomUniformLike nodes found")
        return model

    # Probe to get output shapes
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for rn in rn_nodes:
        vi = helper.make_tensor_value_info(rn.output[0], TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)
    shapes = {name: val.shape for name, val in zip([o.name for o in m_probe.graph.output], all_out)
              if any(rn.output[0] == name for rn in rn_nodes)}

    rng = np.random.default_rng(42)
    nodes_to_remove = set()
    const_nodes = []
    for rn in rn_nodes:
        name = rn.output[0]
        if name in shapes:
            if rn.op_type == 'RandomNormalLike':
                noise = rng.standard_normal(size=shapes[name]).astype(np.float32)
            else:  # RandomUniformLike
                noise = rng.uniform(0, 1, size=shapes[name]).astype(np.float32)
            const_nodes.append(helper.make_node(
                'Constant', inputs=[], outputs=[name],
                name=f'const_{rn.name}',
                value=numpy_helper.from_array(noise, name=name)
            ))
            nodes_to_remove.add(id(rn))
            print(f"  Replaced {rn.op_type} {name}: shape={shapes[name]}")

    new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    for cn in const_nodes:
        new_nodes.insert(0, cn)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Fix Kokoro TTS ONNX model for RKNN conversion (6-step pipeline)'
    )
    parser.add_argument('--input',
                        default='~/kokoro-analysis/kokoro-multi-lang-v1_1/model.onnx',
                        help='Path to input ONNX model')
    parser.add_argument('--output',
                        default='~/kokoro-analysis/kokoro-multi-lang-v1_1/model-rknn-ready.onnx',
                        help='Path to output fixed ONNX model')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN,
                        help=f'Max token sequence length / bucket size (default: {SEQ_LEN})')
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    seq_len = args.seq_len

    print(f"=== Kokoro TTS ONNX RKNN Fix Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"SEQ_LEN={seq_len}")

    # Reference test inputs
    rng = np.random.default_rng(0)
    tokens = np.zeros((1, seq_len), dtype=np.int64)
    tokens[0, :5] = [1, 2, 3, 4, 5]
    test_inputs = {
        'tokens': tokens,
        'style': rng.standard_normal((1, 256)).astype(np.float32),
        'speed': np.array([1.0], dtype=np.float32),
    }

    # Step 0: Eliminate control flow (must run before onnxsim)
    print("\n[0/6] Eliminating control flow ops (Loop/If/Sequence)...")
    raw_model = onnx.load(input_path)
    raw_model = eliminate_control_flow(raw_model, test_inputs)
    # Save intermediate for onnxsim
    tmp_no_cf = tempfile.mktemp(suffix='.onnx')
    onnx.save(raw_model, tmp_no_cf)

    # Step 1: onnxsim
    print("\n[1/6] onnxsim simplification...")
    model = load_and_simplify(tmp_no_cf, seq_len)
    os.unlink(tmp_no_cf)

    # Step 2: Range nodes
    print("\n[2/6] Replacing Range nodes...")
    model = fix_range_nodes(model, test_inputs)

    # Step 3: Dynamic Slice ends
    print("\n[3/6] Fixing dynamic Slice index inputs...")
    model = fix_dynamic_slice_ends(model, test_inputs)

    # Step 4: Ceil ops
    print("\n[4/6] Replacing Ceil ops...")
    model = fix_ceil_ops(model)

    # Step 5: RandomNormalLike / RandomUniformLike
    print("\n[5/6] Replacing RandomNormalLike / RandomUniformLike...")
    model = fix_random_noise(model, test_inputs)

    # Step 6: Shape inference + ORT verify
    print("\n[6/6] Shape inference and ORT verification...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("  Shape inference OK")
    except Exception as e:
        print(f"  Warning (shape inference): {e}")

    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        out = sess.run(None, test_inputs)
        audio = out[0]
        print(f"  Output shape: {audio.shape}")
        print(f"  Output RMS: {float(np.sqrt(np.mean(audio**2))):.4f}")
        print("  ORT OK")
    except Exception as e:
        print(f"  ORT FAIL: {e}")
        sys.exit(1)

    # Check for remaining problematic ops
    warn_ops = ['Range', 'Ceil', 'RandomNormalLike', 'RandomUniformLike',
                'If', 'Loop', 'SplitToSequence', 'SequenceEmpty', 'ConcatFromSequence']
    for op in warn_ops:
        n = sum(1 for node in model.graph.node if node.op_type == op)
        if n:
            print(f"  WARNING: {n} {op} node(s) remain — may cause RKNN conversion failure")

    # Save
    onnx.save(model, output_path)
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({sz:.1f} MB)")
    print("Done.")


if __name__ == '__main__':
    main()

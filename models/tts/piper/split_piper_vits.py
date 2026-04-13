#!/usr/bin/env python3
"""Split Piper VITS into encoder (CPU) + flow_decoder (NPU) sub-models.

The Piper VITS architecture has three logical stages:
  1. Text Encoder + Duration Predictor + Length Regulator  (dynamic shapes)
  2. Flow (invertible 1x1 convolutions)                    (fixed shapes)
  3. HiFi-GAN Decoder                                      (fixed shapes)

Stages 2+3 can run on NPU with fixed mel_len, while stage 1 must stay on
CPU due to dynamic sequence lengths from the Length Regulator.

Split point: the tensors fed into the Flow module.

Usage:
  python split_piper_vits.py \\
      --input en_US-lessac-medium.onnx \\
      --output-dir /tmp/piper-split/en_US/

  python split_piper_vits.py \\
      --input en_US-lessac-medium.onnx \\
      --output-dir /tmp/piper-split/en_US/ \\
      --mel-len 256 \\
      --verify

Output:
  encoder.onnx         -- Encoder+DP+LengthRegulator (dynamic shapes, ORT CPU)
  flow_decoder.onnx    -- Flow+Decoder (fixed shapes, for RKNN conversion)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Auto-detect split point tensors
# ---------------------------------------------------------------------------

def _find_split_tensors(model) -> tuple[str, str]:
    """Find the tensor names at the encoder/flow boundary.

    Strategy:
    1. Look for the first node in the /flow/ namespace.
    2. Its inputs are the split-point tensors (z and y_mask).
    3. If /flow/ namespace not found, fall back to known names.

    Returns (z_tensor_name, y_mask_tensor_name).
    """
    # Build map: tensor_name -> producing node
    producer = {}
    for node in model.graph.node:
        for out in node.output:
            producer[out] = node

    # Find first node whose name starts with /flow/
    flow_nodes = [n for n in model.graph.node if n.name.startswith("/flow/")]

    if flow_nodes:
        first_flow = flow_nodes[0]
        # The first flow node typically has 2 inputs: z and y_mask
        # z shape: (1, 192, mel_len), y_mask shape: (1, 1, mel_len)
        inputs = list(first_flow.input)
        if len(inputs) >= 2:
            print(f"  Auto-detected split point from first /flow/ node: {first_flow.name}")
            print(f"    Input tensors: {inputs}")
            return inputs[0], inputs[1]

    # Fallback: try known tensor names from common Piper models
    known_z = ["/Add_output_0", "/enc_p/Add_output_0"]
    known_mask = ["/Cast_2_output_0", "/enc_p/Cast_2_output_0"]

    graph_tensors = set()
    for node in model.graph.node:
        for out in node.output:
            graph_tensors.add(out)

    for z_name in known_z:
        for mask_name in known_mask:
            if z_name in graph_tensors and mask_name in graph_tensors:
                print(f"  Using known split tensors: z={z_name}, mask={mask_name}")
                return z_name, mask_name

    # Last resort: search for the Add node that feeds into /flow/
    # by tracing backwards from /dec/ or /flow/ inputs
    dec_nodes = [n for n in model.graph.node
                 if n.name.startswith("/dec/") or n.name.startswith("/flow/")]
    if dec_nodes:
        # Find the Conv node in decoder, trace its input chain
        for node in dec_nodes:
            if "conv_pre" in node.name or "Conv" in node.op_type:
                # This node's input comes from flow output, not useful
                continue

    raise RuntimeError(
        "Could not auto-detect split point tensors. "
        "Please specify --z-tensor and --mask-tensor manually. "
        "Hint: look for the inputs to the first /flow/ node in the ONNX graph."
    )


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_model(onnx_path: str, output_dir: str,
                z_tensor: str | None = None,
                mask_tensor: str | None = None,
                mel_len: int = 256) -> tuple[str, str]:
    """Split Piper VITS ONNX into encoder + flow_decoder.

    Returns (encoder_path, flow_decoder_path).
    """
    import onnx
    from onnx.utils import Extractor

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {onnx_path}")
    model = onnx.load(onnx_path)
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")

    # Detect split tensors
    if z_tensor and mask_tensor:
        print(f"  Using user-specified split tensors: z={z_tensor}, mask={mask_tensor}")
    else:
        z_tensor, mask_tensor = _find_split_tensors(model)

    # Get original input/output names
    input_names = [i.name for i in model.graph.input]
    output_names = [o.name for o in model.graph.output]

    # --- Extract encoder sub-model ---
    print("\nExtracting encoder sub-model...")
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    e = Extractor(model)
    encoder = e.extract_model(input_names, [z_tensor, mask_tensor])
    onnx.save(encoder, encoder_path)
    enc_size = os.path.getsize(encoder_path) / (1024 * 1024)
    print(f"  Saved: {encoder_path} ({enc_size:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in encoder.graph.input]}")
    print(f"  Outputs: {[o.name for o in encoder.graph.output]}")

    # --- Extract flow+decoder sub-model ---
    print("\nExtracting flow_decoder sub-model...")
    flow_decoder_path = os.path.join(output_dir, "flow_decoder.onnx")
    e2 = Extractor(model)
    flow_decoder = e2.extract_model([z_tensor, mask_tensor], output_names)
    onnx.save(flow_decoder, flow_decoder_path)
    fd_size = os.path.getsize(flow_decoder_path) / (1024 * 1024)
    print(f"  Saved: {flow_decoder_path} ({fd_size:.1f} MB)")
    print(f"  Inputs:  {[i.name for i in flow_decoder.graph.input]}")
    print(f"  Outputs: {[o.name for o in flow_decoder.graph.output]}")

    # --- Simplify flow_decoder with fixed shapes ---
    print(f"\nSimplifying flow_decoder with fixed shapes (mel_len={mel_len})...")
    try:
        import onnxsim

        # Determine channel dim from z tensor
        # Default: z=(1,192,mel_len), y_mask=(1,1,mel_len)
        z_shape = {z_tensor: [1, 192, mel_len]}
        mask_shape = {mask_tensor: [1, 1, mel_len]}
        input_shapes = {**z_shape, **mask_shape}

        fd_model = onnx.load(flow_decoder_path)
        fd_simplified, ok = onnxsim.simplify(
            fd_model,
            input_shapes=input_shapes,
        )
        if ok:
            onnx.save(fd_simplified, flow_decoder_path)
            fd_size2 = os.path.getsize(flow_decoder_path) / (1024 * 1024)
            print(f"  Simplified OK: {fd_size2:.1f} MB, nodes={len(fd_simplified.graph.node)}")
        else:
            print("  WARNING: onnxsim simplification failed, keeping unsimplified version")
    except ImportError:
        print("  WARNING: onnxsim not installed, skipping simplification")

    return encoder_path, flow_decoder_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_split(onnx_path: str, encoder_path: str, flow_decoder_path: str,
                 seq_len: int = 30) -> bool:
    """Verify split correctness by comparing original vs split pipeline output.

    Both run on ORT CPU. Returns True if correlation > 0.99.
    """
    import onnx
    import onnxruntime as ort

    print(f"\nVerifying split correctness...")
    print(f"  Original:      {onnx_path}")
    print(f"  Encoder:       {encoder_path}")
    print(f"  Flow+Decoder:  {flow_decoder_path}")

    # Prepare test inputs
    tokens = np.zeros((1, seq_len), dtype=np.int64)
    tokens[0, :5] = [1, 15, 42, 7, 23]  # arbitrary phoneme IDs
    lengths = np.array([5], dtype=np.int64)
    scales = np.array([0.667, 1.0, 0.8], dtype=np.float32)

    base_inputs = {
        'input': tokens,
        'input_lengths': lengths,
        'scales': scales,
    }

    # Check if original model needs 'sid' input
    orig_model = onnx.load(onnx_path)
    orig_input_names = {i.name for i in orig_model.graph.input}
    if 'sid' in orig_input_names:
        base_inputs['sid'] = np.array([0], dtype=np.int64)
    del orig_model

    # Run original model
    print("  Running original model...")
    sess_orig = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    orig_out = sess_orig.run(None, base_inputs)
    orig_audio = orig_out[0].flatten()
    print(f"    Output shape: {orig_out[0].shape}, samples: {len(orig_audio)}")

    # Run encoder
    print("  Running encoder...")
    sess_enc = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])
    enc_input_names = [i.name for i in sess_enc.get_inputs()]
    enc_inputs = {name: base_inputs[name] for name in enc_input_names if name in base_inputs}
    enc_out = sess_enc.run(None, enc_inputs)
    z = enc_out[0]
    y_mask = enc_out[1]
    print(f"    z shape: {z.shape}, y_mask shape: {y_mask.shape}")

    # Run flow+decoder (no padding needed on CPU, shapes are dynamic in ORT)
    print("  Running flow_decoder...")
    sess_fd = ort.InferenceSession(flow_decoder_path, providers=['CPUExecutionProvider'])
    fd_input_names = [i.name for i in sess_fd.get_inputs()]
    fd_inputs = {}
    for name in fd_input_names:
        if name == enc_out[0].shape:  # won't match, just enumerate
            pass
    # Map by position: first input = z, second = y_mask
    fd_inputs[fd_input_names[0]] = z
    if len(fd_input_names) > 1:
        fd_inputs[fd_input_names[1]] = y_mask
    fd_out = sess_fd.run(None, fd_inputs)
    split_audio = fd_out[0].flatten()
    print(f"    Output shape: {fd_out[0].shape}, samples: {len(split_audio)}")

    # Compare
    min_len = min(len(orig_audio), len(split_audio))
    if min_len == 0:
        print("  ERROR: zero-length output!")
        return False

    orig_trimmed = orig_audio[:min_len]
    split_trimmed = split_audio[:min_len]

    corr = np.corrcoef(orig_trimmed, split_trimmed)[0, 1]
    max_diff = np.max(np.abs(orig_trimmed - split_trimmed))
    rms_diff = np.sqrt(np.mean((orig_trimmed - split_trimmed) ** 2))

    print(f"\n  Correlation:  {corr:.6f}")
    print(f"  Max diff:     {max_diff:.6f}")
    print(f"  RMS diff:     {rms_diff:.6f}")
    print(f"  Len original: {len(orig_audio)}, split: {len(split_audio)}")

    ok = corr > 0.99
    print(f"\n  {'PASS' if ok else 'FAIL'}: correlation {'>' if ok else '<='} 0.99")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Split Piper VITS into encoder (CPU) + flow_decoder (NPU)"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to original Piper VITS ONNX model")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: same dir as input)")
    parser.add_argument("--mel-len", type=int, default=256,
                        help="Fixed mel length for flow_decoder (default: 256)")
    parser.add_argument("--z-tensor", default=None,
                        help="Override z tensor name (auto-detected if omitted)")
    parser.add_argument("--mask-tensor", default=None,
                        help="Override y_mask tensor name (auto-detected if omitted)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify split by comparing with original on ORT CPU")
    parser.add_argument("--seq-len", type=int, default=30,
                        help="Sequence length for verification (default: 30)")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(input_path)

    print("=" * 60)
    print("Piper VITS Model Splitter")
    print(f"  Input:      {input_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Mel len:    {args.mel_len}")
    print("=" * 60)

    encoder_path, fd_path = split_model(
        input_path, output_dir,
        z_tensor=args.z_tensor,
        mask_tensor=args.mask_tensor,
        mel_len=args.mel_len,
    )

    if args.verify:
        ok = verify_split(input_path, encoder_path, fd_path, seq_len=args.seq_len)
        if not ok:
            sys.exit(1)

    print("\nDone.")
    print(f"  encoder.onnx:       {encoder_path}")
    print(f"  flow_decoder.onnx:  {fd_path}")


if __name__ == "__main__":
    main()

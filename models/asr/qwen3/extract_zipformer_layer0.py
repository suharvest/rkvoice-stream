#!/usr/bin/env python3
"""
Extract Zipformer encoder layer 0 weights and generate reference data.

This is Phase 0a of the MTE engine development.
"""

import numpy as np
import onnxruntime as ort
import json
from pathlib import Path

MODEL_DIR = Path.home() / "models" / "sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16"
OUTPUT_DIR = Path(__file__).parent / "mte_reference" / "zipformer_layer0"

def analyze_encoder():
    """Analyze encoder ONNX structure."""
    encoder_path = MODEL_DIR / "encoder-epoch-99-avg-1.onnx"
    print(f"Loading encoder: {encoder_path}")

    sess = ort.InferenceSession(str(encoder_path))
    print("\n=== Encoder Inputs ===")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")

    print("\n=== Encoder Outputs ===")
    for out in sess.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    return sess

def extract_layer0_weights(sess, output_dir):
    """Extract layer 0 weights from encoder."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use onnx library to inspect weights
    import onnx
    encoder_path = MODEL_DIR / "encoder-epoch-99-avg-1.onnx"
    onnx_model = onnx.load(str(encoder_path))

    print("\n=== Initializers (Weights) ===")
    all_weights = {}

    for init in onnx_model.graph.initializer:
        name = init.name
        arr = onnx.numpy_helper.to_array(init)
        all_weights[name] = arr

    # Print layer 0 specific weights
    layer0_prefix = "encoder.encoders.0.layers.0"
    print(f"\n=== Layer 0 weights (prefix: {layer0_prefix}) ===")
    layer0_weights = {}
    for name, arr in all_weights.items():
        if name.startswith(layer0_prefix) or 'encoder_embed' in name:
            print(f"  {name}: {arr.shape} {arr.dtype}")
            layer0_weights[name] = arr

    # Also extract MatMul weights (stored as onnx::MatMul_N)
    print("\n=== MatMul Weights (onnx::MatMul_*) ===")
    matmul_weights = {}
    for name, arr in all_weights.items():
        if name.startswith("onnx::MatMul"):
            print(f"  {name}: {arr.shape} {arr.dtype}")
            matmul_weights[name] = arr

    # Extract self_attn in_proj weight
    print("\n=== Self-attention in_proj weights ===")
    for name, arr in all_weights.items():
        if 'self_attn.in_proj' in name and 'weight' not in name.lower():
            # These are the MatMul weights for Q/K/V projection
            print(f"  {name}: {arr.shape} {arr.dtype}")

    # Save all weights to npz
    np.savez(output_dir / "all_weights.npz", **all_weights)
    np.savez(output_dir / "layer0_weights.npz", **layer0_weights)

    # Save metadata about layer 0 structure
    layer0_info = {
        "prefix": layer0_prefix,
        "encoder_dim": 256,
        "attention_dim": 192,
        "ffn_dim": 768,
        "num_heads": 8,  # 192 / 24 = 8
        "head_dim": 24,
        "conv_kernel": 31,
        "weights": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                    for k, v in layer0_weights.items()}
    }
    with open(output_dir / "layer0_info.json", "w") as f:
        json.dump(layer0_info, f, indent=2)

    print(f"\n  Saved {len(all_weights)} total weights, {len(layer0_weights)} layer0 weights")
    print(f"  Output: {output_dir}")

    return layer0_weights

def generate_reference_data(sess, output_dir):
    """Generate reference input/output for layer 0 validation."""
    output_dir = Path(output_dir)

    # Get input info
    inputs = sess.get_inputs()
    print("\n=== Generating Reference Data ===")

    # Zipformer encoder inputs:
    # - x: [batch, T, 80] - fbank features (T=39 for typical chunk)
    # - cached_len_N: [2, batch] - int64, streaming position
    # - cached_avg_N: [2, batch, 256] - float32
    # - cached_key_N: [2, L, batch, 192] - float32 (L varies)
    # - cached_val_N: [2, L, batch, 96] - float32
    # - cached_val2_N: [2, L, batch, 96] - float32
    # - cached_conv1/2_N: [2, batch, 256, 30] - float32

    batch_size = 1
    T = 39  # Typical chunk size for streaming
    feat_dim = 80

    # Create dummy input
    x = np.random.randn(batch_size, T, feat_dim).astype(np.float32) * 0.1

    # Build input dict with proper shapes
    input_dict = {"x": x}

    # Cache lengths for each stack (5 stacks: 0-4)
    for i in range(5):
        input_dict[f"cached_len_{i}"] = np.zeros((2, batch_size), dtype=np.int64)

    # Cache avg for each stack
    for i in range(5):
        input_dict[f"cached_avg_{i}"] = np.zeros((2, batch_size, 256), dtype=np.float32)

    # Cache key/val/val2 for each stack (L varies)
    cache_sizes = [64, 32, 16, 8, 32]  # L for each stack
    for i, L in enumerate(cache_sizes):
        input_dict[f"cached_key_{i}"] = np.zeros((2, L, batch_size, 192), dtype=np.float32)
        input_dict[f"cached_val_{i}"] = np.zeros((2, L, batch_size, 96), dtype=np.float32)
        input_dict[f"cached_val2_{i}"] = np.zeros((2, L, batch_size, 96), dtype=np.float32)

    # Cache conv for each stack
    for i in range(5):
        input_dict[f"cached_conv1_{i}"] = np.zeros((2, batch_size, 256, 30), dtype=np.float32)
        input_dict[f"cached_conv2_{i}"] = np.zeros((2, batch_size, 256, 30), dtype=np.float32)

    print(f"\n  Input dict keys: {len(input_dict)} tensors")

    # Run inference
    try:
        outputs = sess.run(None, input_dict)
        print(f"\n  Outputs: {len(outputs)} tensors")

        # Save reference data
        save_dict = {"input_x": x}
        for k, v in input_dict.items():
            if k != "x":
                save_dict[f"input_{k}"] = v
        for i, out in enumerate(outputs):
            save_dict[f"output_{i}"] = out
            print(f"    Output {i}: shape={out.shape}, dtype={out.dtype}")

        np.savez(output_dir / "reference_io.npz", **save_dict)

        # Save metadata
        meta = {
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                       for k, v in input_dict.items()},
            "outputs": {str(i): {"shape": list(outputs[i].shape), "dtype": str(outputs[i].dtype)}
                        for i in range(len(outputs))},
            "encoder_out_dim": 512,
            "num_stacks": 5,
        }
        with open(output_dir / "reference_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n  Saved reference data to {output_dir}")
        return True

    except Exception as e:
        print(f"\n  Error running inference: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Phase 0a: Zipformer Layer 0 Weight Extraction")
    print("=" * 60)

    # Analyze encoder
    sess = analyze_encoder()

    # Extract weights
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer0_weights = extract_layer0_weights(sess, OUTPUT_DIR / "weights")

    # Generate reference data
    generate_reference_data(sess, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Phase 0a Complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
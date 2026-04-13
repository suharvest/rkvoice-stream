#!/usr/bin/env python3
"""Quantize code_predictor FP16 weights to per-column INT4 + scale for W4A16 matmul API.

The RKNN matmul API on RK3576 (librknnrt 2.3.0) only supports per-layer scale for INT4.
We work around this by:
1. Quantizing each column independently (per-column symmetric INT4)
2. Packing two INT4 values per byte
3. Saving per-column FP32 scale vectors separately
4. At runtime: NPU computes A*B_int4 (scale=1.0), then CPU multiplies output by scales

Input:  cp_weights/ (FP16 .bin files from extract_cp_weights.py)
Output: cp_weights_w4a16/ with:
  - layer_N/proj_name.int4.bin  (packed INT4, K*N/2 bytes)
  - layer_N/proj_name.scales.bin (FP32 scales, N floats)
  - lm_heads/lm_head_N.int4.bin
  - lm_heads/lm_head_N.scales.bin
  - (norm weights and codec embeddings are copied as-is)

Run on any machine with numpy:
  python3 quantize_w4a16.py --input /home/cat/cp_weights --output /home/cat/cp_weights_w4a16
"""

import argparse
import os
import json
import numpy as np


def quantize_per_column_int4(W_fp16_path, K, N):
    """Load FP16 weight [K, N], quantize per-column to INT4, return packed + scales."""
    # Load raw FP16 binary -> [K, N]
    raw = np.fromfile(W_fp16_path, dtype=np.float16)
    assert raw.size == K * N, f"Expected {K*N} fp16 values, got {raw.size}"
    W = raw.reshape(K, N).astype(np.float32)

    # Per-column quantization: scale = max(|col|) / 7
    col_amax = np.max(np.abs(W), axis=0)  # [N]
    scales = col_amax / 7.0
    scales[scales == 0] = 1.0  # avoid div-by-zero

    # Quantize to INT4 range [-8, 7]
    W_q = np.clip(np.round(W / scales[None, :]), -8, 7).astype(np.int8)

    # Pack two INT4 values per byte (row-major order)
    # flat[i] and flat[i+1] pack into one byte: low nibble = flat[i], high nibble = flat[i+1]
    flat = W_q.flatten()
    assert flat.size % 2 == 0
    lo = flat[0::2].astype(np.uint8) & 0x0F
    hi = flat[1::2].astype(np.uint8) & 0x0F
    packed = lo | (hi << 4)

    return packed, scales.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Quantize FP16 weights to W4A16")
    parser.add_argument("--input", required=True, help="Input FP16 weight dir (cp_weights/)")
    parser.add_argument("--output", required=True, help="Output W4A16 weight dir")
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.input, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    n_layers = meta["num_layers"]   # 5
    n_steps = meta["num_steps"]     # 15
    hidden = meta["hidden_size"]    # 1024
    inter = meta["intermediate_size"]  # 3072
    vocab = meta["vocab_size"]      # 2048
    num_q = meta["num_q_heads"]     # 16
    num_kv = meta["num_kv_heads"]   # 8
    head_dim = meta["head_dim"]     # 128

    # Matmul B dimensions: [K, N] (same as in C engine)
    proj_dims = {
        "q_proj":    (hidden, num_q * head_dim),   # [1024, 2048]
        "k_proj":    (hidden, num_kv * head_dim),  # [1024, 1024]
        "v_proj":    (hidden, num_kv * head_dim),  # [1024, 1024]
        "o_proj":    (num_q * head_dim, hidden),   # [2048, 1024]
        "gate_proj": (hidden, inter),              # [1024, 3072]
        "up_proj":   (hidden, inter),              # [1024, 3072]
        "down_proj": (inter, hidden),              # [3072, 1024]
    }

    os.makedirs(args.output, exist_ok=True)
    total_orig = 0
    total_int4 = 0

    # --- Quantize transformer layer weights ---
    for l in range(n_layers):
        layer_in = os.path.join(args.input, f"layer_{l}")
        layer_out = os.path.join(args.output, f"layer_{l}")
        os.makedirs(layer_out, exist_ok=True)

        for proj_name, (K, N) in proj_dims.items():
            fp16_path = os.path.join(layer_in, f"{proj_name}.bin")
            packed, scales = quantize_per_column_int4(fp16_path, K, N)

            # Save packed INT4
            int4_path = os.path.join(layer_out, f"{proj_name}.int4.bin")
            packed.tofile(int4_path)

            # Save per-column scales (FP32)
            scales_path = os.path.join(layer_out, f"{proj_name}.scales.bin")
            scales.tofile(scales_path)

            orig_size = K * N * 2  # FP16
            int4_size = packed.nbytes + scales.nbytes
            total_orig += orig_size
            total_int4 += int4_size
            ratio = int4_size / orig_size
            print(f"  layer {l} {proj_name:10s}: [{K},{N}] -> {int4_size/1024:.0f}KB ({ratio:.1%})")

        # Copy norm weights as-is (they're FP32, used on CPU)
        for norm_name in ["input_norm.bin", "post_norm.bin", "q_norm.bin", "k_norm.bin"]:
            src = os.path.join(layer_in, norm_name)
            dst = os.path.join(layer_out, norm_name)
            data = np.fromfile(src, dtype=np.uint8)
            data.tofile(dst)

    # --- Final norm (copy as-is) ---
    src = os.path.join(args.input, "final_norm.bin")
    dst = os.path.join(args.output, "final_norm.bin")
    np.fromfile(src, dtype=np.uint8).tofile(dst)

    # --- Quantize LM heads ---
    lm_in = os.path.join(args.input, "lm_heads")
    lm_out = os.path.join(args.output, "lm_heads")
    os.makedirs(lm_out, exist_ok=True)

    K_lm, N_lm = hidden, vocab  # [1024, 2048]
    for s in range(n_steps):
        fp16_path = os.path.join(lm_in, f"lm_head_{s}.bin")
        packed, scales = quantize_per_column_int4(fp16_path, K_lm, N_lm)

        packed.tofile(os.path.join(lm_out, f"lm_head_{s}.int4.bin"))
        scales.tofile(os.path.join(lm_out, f"lm_head_{s}.scales.bin"))

        orig_size = K_lm * N_lm * 2
        int4_size = packed.nbytes + scales.nbytes
        total_orig += orig_size
        total_int4 += int4_size
        ratio = int4_size / orig_size
        print(f"  lm_head_{s}: [{K_lm},{N_lm}] -> {int4_size/1024:.0f}KB ({ratio:.1%})")

    # --- Codec embeddings (copy as-is, FP32) ---
    emb_in = os.path.join(args.input, "codec_embeddings")
    emb_out = os.path.join(args.output, "codec_embeddings")
    os.makedirs(emb_out, exist_ok=True)
    for s in range(n_steps):
        src = os.path.join(emb_in, f"codec_embed_{s}.bin")
        dst = os.path.join(emb_out, f"codec_embed_{s}.bin")
        np.fromfile(src, dtype=np.uint8).tofile(dst)
        print(f"  codec_embed_{s}: copied")

    # --- Copy metadata ---
    meta["quantization"] = "per_column_int4"
    meta["int4_packing"] = "two_per_byte_lo_hi"
    meta["scale_dtype"] = "float32"
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTotal projection weight size:")
    print(f"  FP16:  {total_orig / 1024 / 1024:.1f} MB")
    print(f"  INT4:  {total_int4 / 1024 / 1024:.1f} MB")
    print(f"  Ratio: {total_int4/total_orig:.1%}")
    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    main()

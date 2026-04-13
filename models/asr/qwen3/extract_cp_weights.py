#!/usr/bin/env python3
"""Extract code_predictor weights from HuggingFace safetensors for the C engine.

Run on wsl2-local:
  ~/qwen3-tts-export/.venv/bin/python extract_cp_weights.py \
      --model-dir ~/qwen3-tts-export/code-predictor-hf \
      --lm-heads-dir ~/qwen3-tts-export/code-predictor-lm-heads \
      --output-dir ~/qwen3-tts-export/cp_weights

Then transfer cp_weights/ to cat-remote:
  rsync -avP cp_weights/ cat-remote:/home/cat/cp_weights/
"""

import argparse
import os
import struct
import numpy as np

def save_fp16_bin(arr, path):
    """Save array as raw FP16 binary."""
    arr_fp16 = arr.astype(np.float16) if arr.dtype != np.float16 else arr
    arr_fp16.tofile(path)

def save_fp32_bin(arr, path):
    """Save array as raw FP32 binary."""
    arr_fp32 = arr.astype(np.float32) if arr.dtype != np.float32 else arr
    arr_fp32.tofile(path)

def main():
    parser = argparse.ArgumentParser(description="Extract code_predictor weights")
    parser.add_argument("--model-dir", required=True, help="HF model dir with model.safetensors")
    parser.add_argument("--lm-heads-dir", required=True, help="Dir with lm_head_*.npy and codec_embeddings/")
    parser.add_argument("--output-dir", required=True, help="Output weight dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load safetensors
    from safetensors import safe_open
    sf = safe_open(os.path.join(args.model_dir, "model.safetensors"), framework="numpy")

    # --- Transformer layer weights ---
    for layer_idx in range(5):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)
        prefix = f"model.layers.{layer_idx}"

        # Norm weights (FP32 for CPU ops)
        input_norm = sf.get_tensor(f"{prefix}.input_layernorm.weight")
        save_fp32_bin(input_norm, os.path.join(layer_dir, "input_norm.bin"))

        post_norm = sf.get_tensor(f"{prefix}.post_attention_layernorm.weight")
        save_fp32_bin(post_norm, os.path.join(layer_dir, "post_norm.bin"))

        q_norm = sf.get_tensor(f"{prefix}.self_attn.q_norm.weight")
        save_fp32_bin(q_norm, os.path.join(layer_dir, "q_norm.bin"))

        k_norm = sf.get_tensor(f"{prefix}.self_attn.k_norm.weight")
        save_fp32_bin(k_norm, os.path.join(layer_dir, "k_norm.bin"))

        # Projection weights for matmul (FP16, stored as [N, K] in safetensors)
        # rknn_matmul expects B as [K, N] (normal layout), then converts to native
        # safetensors stores as [out_features, in_features] = [N, K]
        # We need to transpose to [K, N] for rknn_matmul B matrix

        proj_names = {
            "q_proj": f"{prefix}.self_attn.q_proj.weight",   # [2048, 1024] -> B[1024, 2048]
            "k_proj": f"{prefix}.self_attn.k_proj.weight",   # [1024, 1024] -> B[1024, 1024]
            "v_proj": f"{prefix}.self_attn.v_proj.weight",   # [1024, 1024] -> B[1024, 1024]
            "o_proj": f"{prefix}.self_attn.o_proj.weight",   # [1024, 2048] -> B[2048, 1024]
            "gate_proj": f"{prefix}.mlp.gate_proj.weight",   # [3072, 1024] -> B[1024, 3072]
            "up_proj": f"{prefix}.mlp.up_proj.weight",       # [3072, 1024] -> B[1024, 3072]
            "down_proj": f"{prefix}.mlp.down_proj.weight",   # [1024, 3072] -> B[3072, 1024]
        }

        for name, key in proj_names.items():
            w = sf.get_tensor(key)  # [N, K]
            w_t = w.T.copy()  # [K, N] - contiguous for rknn
            save_fp16_bin(w_t, os.path.join(layer_dir, f"{name}.bin"))
            print(f"  layer {layer_idx} {name}: {w.shape} -> B{w_t.shape}")

    # --- Final norm weight ---
    final_norm = sf.get_tensor("model.norm.weight")
    save_fp32_bin(final_norm, os.path.join(args.output_dir, "final_norm.bin"))
    print(f"  final_norm: {final_norm.shape}")

    # --- LM heads (15 separate heads) ---
    # Each is [2048, 1024] float32 from npy; we need B = [K=1024, N=2048] FP16
    lm_dir = os.path.join(args.output_dir, "lm_heads")
    os.makedirs(lm_dir, exist_ok=True)
    for i in range(15):
        lm = np.load(os.path.join(args.lm_heads_dir, f"lm_head_{i}.npy"))  # [2048, 1024]
        lm_t = lm.T.copy()  # [1024, 2048]
        save_fp16_bin(lm_t, os.path.join(lm_dir, f"lm_head_{i}.bin"))
        print(f"  lm_head_{i}: {lm.shape} -> B{lm_t.shape}")

    # --- Codec embeddings (15 codebooks) ---
    # Each is [2048, 1024] float32; stored as FP32 for CPU lookup
    embed_dir = os.path.join(args.output_dir, "codec_embeddings")
    os.makedirs(embed_dir, exist_ok=True)
    for i in range(15):
        emb = np.load(os.path.join(args.lm_heads_dir, "codec_embeddings", f"codec_embed_{i}.npy"))
        save_fp32_bin(emb, os.path.join(embed_dir, f"codec_embed_{i}.bin"))
        print(f"  codec_embed_{i}: {emb.shape}")

    # --- Write metadata ---
    meta = {
        "num_layers": 5,
        "num_steps": 15,
        "hidden_size": 1024,
        "num_q_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 3072,
        "vocab_size": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
    }
    import json
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Compute total size
    total = 0
    for root, dirs, files in os.walk(args.output_dir):
        for fn in files:
            if fn.endswith('.bin'):
                total += os.path.getsize(os.path.join(root, fn))
    print(f"\nTotal weight size: {total / 1024 / 1024:.1f} MB")
    print(f"Done. Weights saved to {args.output_dir}")

if __name__ == "__main__":
    main()

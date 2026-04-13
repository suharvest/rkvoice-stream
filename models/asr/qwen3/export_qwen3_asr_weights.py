#!/usr/bin/env python3
"""Export Qwen3-ASR-0.6B weights for matmul decoder.

This script exports the decoder weights from HuggingFace format to the
binary format expected by the C matmul decoder.

Expected output structure:
    output_dir/
    ├── config.json           # Model config
    ├── embeddings.bin        # [vocab_size, hidden_dim] FP32
    ├── lm_head.bin           # [vocab_size, hidden_dim] FP32 (optional if tied)
    └── layer_XX/             # Per-layer weights
        ├── input_norm.bin    # [hidden_dim] FP32
        ├── post_norm.bin     # [hidden_dim] FP32
        ├── q_proj.bin        # [hidden_dim, num_q_heads*head_dim] FP16
        ├── k_proj.bin        # [hidden_dim, num_kv_heads*head_dim] FP16
        ├── v_proj.bin        # [hidden_dim, num_kv_heads*head_dim] FP16
        ├── o_proj.bin        # [num_q_heads*head_dim, hidden_dim] FP16
        ├── gate_proj.bin     # [hidden_dim, ffn_dim] FP16
        ├── up_proj.bin       # [hidden_dim, ffn_dim] FP16
        └── down_proj.bin     # [ffn_dim, hidden_dim] FP16

Usage:
    # On a machine with the HF model downloaded
    python export_qwen3_asr_weights.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --output-dir ./qwen3-matmul

    # Then rsync to device:
    rsync -avP ./qwen3-matmul/ device:/home/cat/models/qwen3-matmul/
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def save_fp32(arr: np.ndarray, path: str):
    """Save array as raw FP32 binary."""
    arr = arr.astype(np.float32)
    arr.tofile(path)
    print(f"  Saved {path}: {arr.shape}")


def save_fp16(arr: np.ndarray, path: str):
    """Save array as raw FP16 binary."""
    arr = arr.astype(np.float16)
    arr.tofile(path)
    print(f"  Saved {path}: {arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-ASR weights")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--quant", default="int4", choices=["fp16", "int8", "int4"],
                        help="Quantization type (for config.json)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}...")
    try:
        from safetensors import safe_open
        sf_path = os.path.join(args.model, "model.safetensors")
        if os.path.exists(sf_path):
            sf = safe_open(sf_path, framework="numpy")
            use_safetensors = True
            print(f"  Using safetensors: {sf_path}")
        else:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype="auto", device_map="cpu"
            )
            state_dict = model.state_dict()
            use_safetensors = False
            print(f"  Using transformers model")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install: pip install safetensors transformers")
        return

    def get_tensor(name: str) -> np.ndarray:
        if use_safetensors:
            return sf.get_tensor(name).numpy()
        else:
            return state_dict[name].numpy()

    # Read config
    config_path = os.path.join(args.model, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            hf_config = json.load(f)
    else:
        # Default Qwen2.5-0.5B config
        hf_config = {
            "hidden_size": 896,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "intermediate_size": 4864,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": True,
        }

    hidden_dim = hf_config.get("hidden_size", 896)
    num_q_heads = hf_config.get("num_attention_heads", 14)
    num_kv_heads = hf_config.get("num_key_value_heads", 2)
    head_dim = hidden_dim // num_q_heads
    ffn_dim = hf_config.get("intermediate_size", 4864)
    num_layers = hf_config.get("num_hidden_layers", 24)
    vocab_size = hf_config.get("vocab_size", 151936)
    rms_eps = hf_config.get("rms_norm_eps", 1e-6)
    rope_theta = hf_config.get("rope_theta", 1000000.0)
    tie_embeddings = hf_config.get("tie_word_embeddings", True)

    print(f"\nModel config:")
    print(f"  hidden_dim: {hidden_dim}")
    print(f"  num_q_heads: {num_q_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  ffn_dim: {ffn_dim}")
    print(f"  num_layers: {num_layers}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  tie_embeddings: {tie_embeddings}")

    # Save config.json
    config = {
        "name": "qwen3-asr",
        "hidden_dim": hidden_dim,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_len": 4096,
        "rms_eps": rms_eps,
        "rope_theta": rope_theta,
        "tie_word_embeddings": tie_embeddings,
        "quant_type": args.quant,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved config.json")

    # Export embeddings
    print(f"\nExporting embeddings...")
    try:
        embed = get_tensor("model.embed_tokens.weight")
        save_fp32(embed, os.path.join(args.output_dir, "embeddings.bin"))
    except Exception as e:
        print(f"  Warning: Could not export embeddings: {e}")
        print(f"  Creating random embeddings for testing...")
        embed = np.random.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        save_fp32(embed, os.path.join(args.output_dir, "embeddings.bin"))

    # Export lm_head (if not tied)
    if not tie_embeddings:
        print(f"\nExporting lm_head...")
        try:
            lm_head = get_tensor("lm_head.weight")  # [vocab_size, hidden_dim]
            save_fp32(lm_head, os.path.join(args.output_dir, "lm_head.bin"))
        except Exception as e:
            print(f"  Warning: Could not export lm_head: {e}")

    # Export per-layer weights
    print(f"\nExporting {num_layers} layers...")
    for layer_idx in range(num_layers):
        layer_dir = os.path.join(args.output_dir, f"layer_{layer_idx:02d}")
        os.makedirs(layer_dir, exist_ok=True)
        prefix = f"model.layers.{layer_idx}"

        print(f"\n  Layer {layer_idx}:")

        # Norm weights (FP32)
        try:
            input_norm = get_tensor(f"{prefix}.input_layernorm.weight")
            save_fp32(input_norm, os.path.join(layer_dir, "input_norm.bin"))
        except Exception as e:
            print(f"    Warning: input_norm not found: {e}")

        try:
            post_norm = get_tensor(f"{prefix}.post_attention_layernorm.weight")
            save_fp32(post_norm, os.path.join(layer_dir, "post_norm.bin"))
        except Exception as e:
            print(f"    Warning: post_norm not found: {e}")

        # Projection weights (FP16, transposed for matmul B matrix)
        proj_specs = {
            "q_proj": (hidden_dim, num_q_heads * head_dim),
            "k_proj": (hidden_dim, num_kv_heads * head_dim),
            "v_proj": (hidden_dim, num_kv_heads * head_dim),
            "o_proj": (num_q_heads * head_dim, hidden_dim),
            "gate_proj": (hidden_dim, ffn_dim),
            "up_proj": (hidden_dim, ffn_dim),
            "down_proj": (ffn_dim, hidden_dim),
        }

        for proj_name, (K, N) in proj_specs.items():
            try:
                # HF stores as [N, K], we need [K, N] for matmul B
                w = get_tensor(f"{prefix}.self_attn.{proj_name}.weight")
                if proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    w = get_tensor(f"{prefix}.mlp.{proj_name}.weight")

                w_t = w.T.copy()  # [K, N]
                save_fp16(w_t, os.path.join(layer_dir, f"{proj_name}.bin"))
            except Exception as e:
                print(f"    Warning: {proj_name} not found: {e}")

    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(args.output_dir):
        for fn in files:
            if fn.endswith('.bin'):
                total_size += os.path.getsize(os.path.join(root, fn))

    print(f"\n{'='*50}")
    print(f"Export complete!")
    print(f"  Output: {args.output_dir}")
    print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"\nTo deploy to RK3576:")
    print(f"  rsync -avP {args.output_dir}/ device:/home/cat/models/qwen3-matmul/")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Quantize code_predictor FP16 weights to per-column INT8 + scale for W8A16 matmul.

Uses FP16xINT8->FP32 (type=5) matmul, which supports rknn_matmul_set_quant_params.
But we use per-column quantization with CPU scale application for better accuracy.
"""
import argparse, os, json
import numpy as np

def quantize_per_column_int8(W_fp16_path, K, N):
    raw = np.fromfile(W_fp16_path, dtype=np.float16)
    assert raw.size == K * N
    W = raw.reshape(K, N).astype(np.float32)
    col_amax = np.max(np.abs(W), axis=0)
    scales = col_amax / 127.0
    scales[scales == 0] = 1.0
    W_q = np.clip(np.round(W / scales[None, :]), -128, 127).astype(np.int8)
    return W_q, scales.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.input, "meta.json")) as f:
        meta = json.load(f)

    n_layers = meta["num_layers"]
    n_steps = meta["num_steps"]
    hidden = meta["hidden_size"]
    inter = meta["intermediate_size"]
    vocab = meta["vocab_size"]
    num_q = meta["num_q_heads"]
    num_kv = meta["num_kv_heads"]
    head_dim = meta["head_dim"]

    proj_dims = {
        "q_proj": (hidden, num_q * head_dim),
        "k_proj": (hidden, num_kv * head_dim),
        "v_proj": (hidden, num_kv * head_dim),
        "o_proj": (num_q * head_dim, hidden),
        "gate_proj": (hidden, inter),
        "up_proj": (hidden, inter),
        "down_proj": (inter, hidden),
    }

    os.makedirs(args.output, exist_ok=True)
    total_orig = 0
    total_int8 = 0

    for l in range(n_layers):
        layer_out = os.path.join(args.output, f"layer_{l}")
        os.makedirs(layer_out, exist_ok=True)
        for proj_name, (K, N) in proj_dims.items():
            fp16_path = os.path.join(args.input, f"layer_{l}", f"{proj_name}.bin")
            W_q, scales = quantize_per_column_int8(fp16_path, K, N)
            W_q.tofile(os.path.join(layer_out, f"{proj_name}.int8.bin"))
            scales.tofile(os.path.join(layer_out, f"{proj_name}.scales.bin"))
            orig = K * N * 2
            int8 = W_q.nbytes + scales.nbytes
            total_orig += orig; total_int8 += int8
            print(f"  layer {l} {proj_name:10s}: [{K},{N}] -> {int8/1024:.0f}KB ({int8/orig:.1%})")
        for norm in ["input_norm.bin", "post_norm.bin", "q_norm.bin", "k_norm.bin"]:
            np.fromfile(os.path.join(args.input, f"layer_{l}", norm), dtype=np.uint8).tofile(
                os.path.join(layer_out, norm))

    np.fromfile(os.path.join(args.input, "final_norm.bin"), dtype=np.uint8).tofile(
        os.path.join(args.output, "final_norm.bin"))

    lm_out = os.path.join(args.output, "lm_heads")
    os.makedirs(lm_out, exist_ok=True)
    for s in range(n_steps):
        fp16_path = os.path.join(args.input, "lm_heads", f"lm_head_{s}.bin")
        W_q, scales = quantize_per_column_int8(fp16_path, hidden, vocab)
        W_q.tofile(os.path.join(lm_out, f"lm_head_{s}.int8.bin"))
        scales.tofile(os.path.join(lm_out, f"lm_head_{s}.scales.bin"))
        orig = hidden * vocab * 2
        int8 = W_q.nbytes + scales.nbytes
        total_orig += orig; total_int8 += int8
        print(f"  lm_head_{s}: [{hidden},{vocab}] -> {int8/1024:.0f}KB ({int8/orig:.1%})")

    emb_out = os.path.join(args.output, "codec_embeddings")
    os.makedirs(emb_out, exist_ok=True)
    for s in range(n_steps):
        np.fromfile(os.path.join(args.input, "codec_embeddings", f"codec_embed_{s}.bin"), dtype=np.uint8).tofile(
            os.path.join(emb_out, f"codec_embed_{s}.bin"))

    meta["quantization"] = "per_column_int8"
    with open(os.path.join(args.output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nFP16: {total_orig/1024/1024:.1f}MB -> INT8: {total_int8/1024/1024:.1f}MB ({total_int8/total_orig:.1%})")

if __name__ == "__main__":
    main()

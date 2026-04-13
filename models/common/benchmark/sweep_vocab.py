#!/usr/bin/env python3
"""
Sweep vocab sizes for RKLLM talker benchmark.
Creates padded HF models, converts to RKLLM.
"""
import json
import shutil
import os
import sys

VOCAB_SIZES = [8192, 16384, 32768, 65536]
BASE_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-talker-hf-4096")
EXPORT_DIR = os.path.expanduser("~/qwen3-tts-export")


def create_padded_model(vocab_size):
    out_dir = os.path.join(EXPORT_DIR, "qwen3-tts-talker-hf-{}".format(vocab_size))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    shutil.copytree(BASE_DIR, out_dir)

    # Update config.json
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["vocab_size"] = vocab_size
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Create padded vocab.json
    with open(os.path.join(BASE_DIR, "vocab.json")) as f:
        base_vocab = json.load(f)

    n_base = len(base_vocab)  # 4093
    n_needed = vocab_size - 3  # special tokens take last 3 slots

    new_vocab = dict(base_vocab)
    for i in range(n_base, n_needed):
        new_vocab["<pad_{}>".format(i)] = i

    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(new_vocab, f)

    # merges.txt stays the same
    # Update tokenizer.json
    with open(os.path.join(BASE_DIR, "tokenizer.json")) as f:
        tok = json.load(f)

    model_vocab = tok["model"]["vocab"]
    for i in range(n_base, n_needed):
        model_vocab["<pad_{}>".format(i)] = i

    tok["added_tokens"] = [
        {"id": n_needed, "content": "<|endoftext|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": n_needed + 1, "content": "<|im_start|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        {"id": n_needed + 2, "content": "<|im_end|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
    ]

    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tok, f)

    # Update tokenizer_config.json
    with open(os.path.join(out_dir, "tokenizer_config.json")) as f:
        tok_config = json.load(f)
    tok_config["vocab_size"] = vocab_size
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_config, f, indent=2)

    # Pad model weights
    from safetensors.torch import load_file, save_file
    import torch

    weights = load_file(os.path.join(BASE_DIR, "model.safetensors"))

    for key in ["model.embed_tokens.weight", "lm_head.weight"]:
        if key in weights:
            w = weights[key]
            old_size = w.shape[0]
            if old_size < vocab_size:
                pad = torch.zeros(vocab_size - old_size, w.shape[1], dtype=w.dtype)
                weights[key] = torch.cat([w, pad], dim=0)
                print("  Padded {}: {} -> {}".format(key, old_size, vocab_size))

    save_file(weights, os.path.join(out_dir, "model.safetensors"))
    print("Created padded model: {}".format(out_dir))
    return out_dir


def convert_rkllm(model_dir, vocab_size):
    output_path = os.path.join(EXPORT_DIR, "talker_v{}_w4a16_rk3576.rkllm".format(vocab_size))
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print("  RKLLM already exists: {} ({:.1f} MB)".format(output_path, size_mb))
        return output_path

    from rkllm.api import RKLLM
    llm = RKLLM()
    ret = llm.load_huggingface(model=model_dir, device="cpu")
    if ret != 0:
        print("  ERROR: load_huggingface returned {}".format(ret))
        return None

    ret = llm.build(do_quantization=True, quantized_dtype="w4a16",
                    target_platform="rk3576", num_npu_core=2,
                    max_context=1024, optimization_level=1)
    if ret != 0:
        print("  ERROR: build returned {}".format(ret))
        return None

    ret = llm.export_rkllm(output_path)
    if ret != 0:
        print("  ERROR: export returned {}".format(ret))
        return None

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("  Exported: {} ({:.1f} MB)".format(output_path, size_mb))
    return output_path


if __name__ == "__main__":
    for vs in VOCAB_SIZES:
        print("\n" + "=" * 60)
        print("Processing vocab_size = {}".format(vs))
        print("=" * 60)

        model_dir = create_padded_model(vs)
        rkllm_path = convert_rkllm(model_dir, vs)
        if rkllm_path:
            print("  SUCCESS: {}".format(rkllm_path))
        else:
            print("  FAILED for vocab_size={}".format(vs))

    print("\n\nAll conversions complete!")
    print("RKLLM files:")
    for vs in VOCAB_SIZES:
        path = os.path.join(EXPORT_DIR, "talker_v{}_w4a16_rk3576.rkllm".format(vs))
        if os.path.exists(path) and os.path.getsize(path) > 0:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print("  vocab={}: {:.1f} MB".format(vs, size_mb))

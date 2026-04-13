#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#
# Extract the Qwen3-TTS talker sub-model as a standalone HuggingFace model
# for conversion via RKLLM-Toolkit.
#
# The talker is a modified Qwen2 backbone with:
#   - vocab_size = 3072 (codec tokens, not text tokens)
#   - hidden_size = 1024
#   - Custom embedding layer (text_embed_tokens + codec tokens share space)
#
# Usage:
#   pip install qwen-tts torch transformers
#   python3 export-rkllm-talker.py \
#       --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
#       --output-dir ./qwen3-tts-talker-hf
#
# Then convert with RKLLM-Toolkit:
#   from rkllm.api import RKLLM
#   llm = RKLLM()
#   llm.load_huggingface(model='./qwen3-tts-talker-hf', device='cuda')
#   llm.build(quantized_dtype="W4A16", target_platform="RK3576", ...)
#   llm.export_rkllm("talker_w4a16_rk3576.rkllm")

import argparse
import json
import os
import shutil

import torch
from transformers import AutoConfig, AutoModel


def main():
    parser = argparse.ArgumentParser(
        description="Extract Qwen3-TTS talker as standalone HuggingFace model for RKLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3-tts-talker-hf",
        help="Output directory for the standalone talker model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for saved weights",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Register Qwen3TTS model classes so AutoModel recognizes "qwen3_tts"
    from qwen_tts.core.models import (  # noqa: F401
        Qwen3TTSConfig,
        Qwen3TTSForConditionalGeneration,
    )

    print(f"Loading Qwen3-TTS model: {args.model} ...")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    # Use Qwen3TTSForConditionalGeneration directly to avoid AutoModel lookup issues
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
    )
    model.eval()

    talker = model.talker
    talker_config = model.config.talker_config

    # Print talker architecture info
    print(f"\nTalker architecture:")
    print(f"  hidden_size:        {talker_config.hidden_size}")
    print(f"  num_hidden_layers:  {talker_config.num_hidden_layers}")
    print(f"  num_attention_heads:{talker_config.num_attention_heads}")
    print(f"  num_key_value_heads:{talker_config.num_key_value_heads}")
    print(f"  vocab_size:         {talker_config.vocab_size}")
    print(f"  intermediate_size:  {getattr(talker_config, 'intermediate_size', 'N/A')}")

    # Count parameters
    num_params = sum(p.numel() for p in talker.parameters())
    print(f"  parameters:         {num_params / 1e6:.1f}M")

    # --- Save as HuggingFace Qwen2-compatible format ---

    # 1. Build config.json that RKLLM-Toolkit can recognize as Qwen2
    config_dict = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": talker_config.hidden_size,
        "intermediate_size": getattr(talker_config, "intermediate_size", talker_config.hidden_size * 4),
        "num_hidden_layers": talker_config.num_hidden_layers,
        "num_attention_heads": talker_config.num_attention_heads,
        "num_key_value_heads": talker_config.num_key_value_heads,
        "vocab_size": talker_config.vocab_size,
        "max_position_embeddings": getattr(talker_config, "max_position_embeddings", 4096),
        "rms_norm_eps": getattr(talker_config, "rms_norm_eps", 1e-6),
        "rope_theta": getattr(talker_config, "rope_theta", 10000.0),
        "hidden_act": getattr(talker_config, "hidden_act", "silu"),
        "tie_word_embeddings": getattr(talker_config, "tie_word_embeddings", False),
        "torch_dtype": args.dtype,
        # Mark this as a TTS talker for reference
        "_qwen3_tts_talker": True,
        "_original_vocab_size": talker_config.vocab_size,
    }

    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nSaved config.json")

    # 2. Save model weights
    # The talker namespace includes code_predictor, text_projection, etc.
    # We ONLY want the core transformer: model.* (Qwen2Model) + lm_head.* (Linear)
    state_dict = {}
    skipped = []
    for name, param in talker.named_parameters():
        # Only keep model.* and lm_head.* — these form the Qwen2ForCausalLM
        if name.startswith("model.") or name.startswith("lm_head"):
            state_dict[name] = param.data
            print(f"  {name}: {list(param.shape)}")
        else:
            skipped.append(name)

    if skipped:
        print(f"\n  Skipped {len(skipped)} non-core weights: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Save in safetensors format (preferred by HuggingFace)
    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(args.output_dir, "model.safetensors"))
        print(f"\nSaved model.safetensors")
    except ImportError:
        torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
        print(f"\nSaved pytorch_model.bin (safetensors not available)")

    # 3. Save generation_config.json (optional, helps RKLLM understand the model)
    gen_config = {
        "max_new_tokens": 2048,
        "do_sample": True,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
    }
    with open(os.path.join(args.output_dir, "generation_config.json"), "w") as f:
        json.dump(gen_config, f, indent=2)

    # 4. Save metadata for reference
    metadata = {
        "source_model": args.model,
        "component": "talker",
        "description": "Qwen3-TTS talker sub-model (Qwen2 backbone for codec AR generation)",
        "input_type": "pre-computed embeddings (text_embed + codec_embed, not raw text tokens)",
        "output_type": "codec token logits (vocab_size=3072)",
        "note": "Use RKLLM_INPUT_EMBED for inference, not RKLLM_INPUT_PROMPT",
    }
    with open(os.path.join(args.output_dir, "talker_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Talker model saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Validate: python3 -c \"from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('{args.output_dir}'); print('OK', sum(p.numel() for p in m.parameters()) / 1e6, 'M params')\"")
    print(f"  2. Convert: Use RKLLM-Toolkit to build .rkllm file for RK3576")


if __name__ == "__main__":
    main()

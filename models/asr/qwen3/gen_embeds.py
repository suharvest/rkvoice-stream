#!/usr/bin/env python3
"""
Generate prefill embeddings for RKLLM E2E testing.
Runs on CPU to avoid CUDA issues.
"""
import os
import json
import numpy as np
import torch

OUT_DIR = os.path.expanduser("~/qwen3-tts-export/e2e_ref")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_TEXT = "今天天气真不错"


def main():
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
    from transformers import AutoTokenizer

    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

    print("Loading model on CPU fp32...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    talker = model.talker
    config = model.config

    # Hook talker.generate to capture inputs_embeds
    prefill_captures = []
    original_generate = talker.generate

    def hooked_generate(inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            prefill_captures.append(inputs_embeds.detach().cpu().float().numpy())
            print("  [hook] Captured prefill: shape={}".format(inputs_embeds.shape))
        return original_generate(inputs_embeds=inputs_embeds, **kwargs)

    talker.generate = hooked_generate

    # Build input IDs
    assistant_text = "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n".format(TEST_TEXT)
    input_ids = tokenizer(assistant_text, return_tensors="pt").input_ids
    print("Input IDs:", input_ids.shape)

    print("Generating (CPU fp32)...")
    with torch.no_grad():
        talker_codes_list, _ = model.generate(
            input_ids=[input_ids],
            languages=["chinese"],
            non_streaming_mode=True,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            temperature=0.9,
        )

    codes = talker_codes_list[0].cpu().numpy()
    print("Codec tokens: shape={}".format(codes.shape))
    print("First column (primary): {}".format(codes[:, 0].tolist()[:20]))

    # Save prefill embeddings
    if prefill_captures:
        emb = prefill_captures[0]
        np.save(os.path.join(OUT_DIR, "prefill_embeds.npy"), emb)
        print("Saved prefill_embeds.npy: shape={}, dtype={}".format(emb.shape, emb.dtype))

        # Also save as fp16 for RKLLM
        emb_fp16 = emb.astype(np.float16)
        np.save(os.path.join(OUT_DIR, "prefill_embeds_fp16.npy"), emb_fp16)
        print("Saved prefill_embeds_fp16.npy")
    else:
        print("WARNING: No prefill embeddings captured!")

    # Save primary codec tokens only
    primary_tokens = codes[:, 0]
    np.save(os.path.join(OUT_DIR, "primary_codec_tokens.npy"), primary_tokens)
    print("Primary tokens: {}".format(primary_tokens.tolist()))

    print("\nDone.")


if __name__ == "__main__":
    main()

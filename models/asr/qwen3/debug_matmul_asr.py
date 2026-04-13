#!/usr/bin/env python3
"""Debug script: compare encoder embedding scale vs token embedding scale,
and test matmul decoder with both pure-text and encoder-output embeddings.

Run inside Docker:
  docker exec rk3576-speech python3 /opt/tts/scripts/debug_matmul_asr.py
"""

import sys
import time
import numpy as np

sys.path.insert(0, "/opt/tts/app")

from tokenizers import Tokenizer

# ---- Load resources ----
print("=== Loading resources ===")
emb_table = np.load("/opt/asr/models/embd/decoder_hf.embed_tokens.npy", mmap_mode='r')
tok = Tokenizer.from_file("/opt/asr/models/tokenizer/tokenizer.json")
print(f"Embedding table: {emb_table.shape} dtype={emb_table.dtype}")

# Special tokens
ID_IM_START = tok.encode("<|im_start|>").ids[0]  # 151644
ID_IM_END = tok.encode("<|im_end|>").ids[0]      # 151645
ID_AUDIO_START = tok.encode("<|audio_start|>").ids[0]  # 151646
ID_AUDIO_END = tok.encode("<|audio_end|>").ids[0]      # 151647
ID_ASR_TEXT = tok.encode("<asr_text>").ids[0]

print(f"Special tokens: im_start={ID_IM_START} im_end={ID_IM_END} "
      f"audio_start={ID_AUDIO_START} audio_end={ID_AUDIO_END} asr_text={ID_ASR_TEXT}")

# ---- Token embedding statistics ----
print("\n=== Token Embedding Statistics ===")
# Sample various tokens
sample_ids = [ID_IM_START, ID_IM_END, ID_AUDIO_START, ID_AUDIO_END, ID_ASR_TEXT,
              100, 500, 1000, 5000, 8948]
for tid in sample_ids:
    e = emb_table[tid]
    print(f"  token {tid:6d}: mean={e.mean():+.6f} std={e.std():.6f} "
          f"norm={np.linalg.norm(e):.4f} min={e.min():.6f} max={e.max():.6f}")

# Overall stats for random subset
rng = np.random.RandomState(42)
rand_ids = rng.randint(0, emb_table.shape[0], 1000)
rand_embs = emb_table[rand_ids]
norms = np.linalg.norm(rand_embs, axis=1)
print(f"\n  Random 1000 tokens: norm mean={norms.mean():.4f} std={norms.std():.4f} "
      f"min={norms.min():.4f} max={norms.max():.4f}")

# ---- Encoder output statistics ----
print("\n=== Encoder Output Statistics ===")
from qwen3asr.encoder import RknnEncoder
from qwen3asr.mel import MelExtractor

encoder = RknnEncoder(
    "/opt/asr/models/encoder/rk3576",
    "/opt/asr/models/mel_filters.npy",
    sizes=[4],
)

# Generate test audio: 1s of speech-like signal
sr = 16000
t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
# Multi-frequency to simulate speech
audio = (0.3 * np.sin(2 * np.pi * 200 * t) +
         0.2 * np.sin(2 * np.pi * 500 * t) +
         0.1 * np.sin(2 * np.pi * 1000 * t))
audio = audio.astype(np.float32)

hidden, enc_ms, model_sec = encoder.encode(audio)
print(f"Encoder output: shape={hidden.shape} dtype={hidden.dtype} enc_ms={enc_ms:.1f}")
print(f"  mean={hidden.mean():+.6f} std={hidden.std():.6f} "
      f"min={hidden.min():.6f} max={hidden.max():.6f}")

enc_norms = np.linalg.norm(hidden, axis=1)
print(f"  Row norms: mean={enc_norms.mean():.4f} std={enc_norms.std():.4f} "
      f"min={enc_norms.min():.4f} max={enc_norms.max():.4f}")
print(f"  First 5 row norms: {enc_norms[:5].tolist()}")

# Check if encoder output is all zeros (NPU conflict)
if np.all(hidden == 0):
    print("\n  *** WARNING: Encoder output is ALL ZEROS! NPU conflict? ***")
    print("  Cannot proceed with decoder test.")
    encoder.release()
    sys.exit(1)

# ---- Scale comparison ----
print("\n=== Scale Comparison: Token Embeddings vs Encoder Output ===")
token_norms_mean = norms.mean()
enc_norms_mean = enc_norms.mean()
ratio = enc_norms_mean / token_norms_mean
print(f"  Token embedding norm (mean): {token_norms_mean:.4f}")
print(f"  Encoder output norm (mean):  {enc_norms_mean:.4f}")
print(f"  Ratio (encoder/token):       {ratio:.4f}")
if ratio > 5 or ratio < 0.2:
    print(f"  *** SCALE MISMATCH DETECTED: ratio={ratio:.2f} ***")
    print(f"  Token embeddings and encoder output are at very different scales!")
else:
    print(f"  Scales are comparable (ratio within 0.2-5.0 range)")

# ---- Test matmul decoder ----
print("\n=== Test Matmul Decoder ===")
import matmul_decoder as md

decoder = md.MatmulDecoder(
    model_dir="/opt/asr/models/decoder/matmul",
    max_seq_len=4096,
    quant_type="fp16",
    exec_mode="dual_core",
)

# Test 1: Pure text token prompt (known working)
print("\n--- Test 1: Pure text tokens ---")
decoder.clear_kv_cache()
# Build a simple prompt with token IDs
text_prompt = "<|im_start|>system\nYou are a helpful assistant. <|im_end|><|im_start|>user\n你好<|im_end|><|im_start|>assistant\n"
text_ids = tok.encode(text_prompt).ids
print(f"  Prompt tokens: {len(text_ids)}")

# Feed as embeddings (to test the embedding path)
t0 = time.time()
for tid in text_ids:
    _ = decoder.step_get_token(token_id=tid)

# Generate a few tokens
generated = []
prev = _  # Last token from prefill
generated.append(prev)
for i in range(20):
    prev = decoder.step_get_token(token_id=prev)
    generated.append(prev)
    if prev == ID_IM_END:
        break

text1 = tok.decode(generated)
print(f"  Generated ({len(generated)} tokens): {text1}")
print(f"  Token IDs: {generated}")

# Test 2: ASR prompt with encoder embeddings
print("\n--- Test 2: ASR prompt with encoder embeddings ---")
decoder.clear_kv_cache()

# Build ASR prompt: same as engine.build_embed()
system_text = "You are a helpful assistant. "
prefix_tokens = (
    [ID_IM_START] + tok.encode(f"system\n{system_text}").ids + [ID_IM_END]
    + [ID_IM_START] + tok.encode("user\n").ids + [ID_AUDIO_START]
)
suffix_tokens = (
    [ID_AUDIO_END]
    + tok.encode("转录：").ids
    + [ID_IM_END]
    + [ID_IM_START]
    + tok.encode("assistant\nlanguage Chinese").ids
    + [ID_ASR_TEXT]
)

n_pre = len(prefix_tokens)
n_audio = hidden.shape[0]
n_suf = len(suffix_tokens)
total = n_pre + n_audio + n_suf
print(f"  Prefix: {n_pre} tokens, Audio: {n_audio} tokens, Suffix: {n_suf} tokens, Total: {total}")

# Build full embedding array (same as engine.build_embed)
embed_dim = emb_table.shape[1]
full_embd = np.zeros((total, embed_dim), dtype=np.float32)
full_embd[:n_pre] = emb_table[prefix_tokens]
full_embd[n_pre:n_pre + n_audio] = hidden
full_embd[n_pre + n_audio:] = emb_table[suffix_tokens]

# Check norms at different positions
pre_norms = [round(float(np.linalg.norm(full_embd[i])), 4) for i in range(min(3, n_pre))]
aud_norms = [round(float(np.linalg.norm(full_embd[n_pre + i])), 4) for i in range(min(3, n_audio))]
suf_norms = [round(float(np.linalg.norm(full_embd[n_pre + n_audio + i])), 4) for i in range(min(3, n_suf))]
print(f"  Prefix embedding norms: {pre_norms}")
print(f"  Audio embedding norms (first 3): {aud_norms}")
print(f"  Suffix embedding norms: {suf_norms}")

# Feed to decoder
t0 = time.time()
last_token = None
for i in range(total):
    emb = full_embd[i]
    last_token = decoder.step_get_token(token_id=-1, embedding=emb)

prefill_ms = (time.time() - t0) * 1000
print(f"  Prefill: {prefill_ms:.0f}ms, first predicted token: {last_token} ({tok.decode([last_token]) if last_token < 151936 else '?'})")

# Generate
generated2 = [last_token]
prev = last_token
t0 = time.time()
for i in range(50):
    prev = decoder.step_get_token(token_id=prev)
    generated2.append(prev)
    if prev == ID_IM_END:
        break

gen_ms = (time.time() - t0) * 1000
text2 = tok.decode(generated2)
for tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
    text2 = text2.replace(tag, "")
print(f"  Generated ({len(generated2)} tokens, {gen_ms:.0f}ms): '{text2}'")
print(f"  Token IDs: {generated2[:20]}...")

# Test 3: Feed via token_id for prefix and suffix, embedding only for audio
print("\n--- Test 3: Token IDs for text, embedding for audio ---")
decoder.clear_kv_cache()

t0 = time.time()
# Feed prefix as token IDs
for tid in prefix_tokens:
    _ = decoder.step_get_token(token_id=tid)

# Feed audio as embeddings
for i in range(n_audio):
    _ = decoder.step_get_token(token_id=-1, embedding=hidden[i])

# Feed suffix as token IDs
for tid in suffix_tokens:
    last_token = decoder.step_get_token(token_id=tid)

prefill_ms = (time.time() - t0) * 1000
print(f"  Prefill: {prefill_ms:.0f}ms, first predicted token: {last_token} ({tok.decode([last_token]) if last_token < 151936 else '?'})")

# Generate
generated3 = [last_token]
prev = last_token
t0 = time.time()
for i in range(50):
    prev = decoder.step_get_token(token_id=prev)
    generated3.append(prev)
    if prev == ID_IM_END:
        break

gen_ms = (time.time() - t0) * 1000
text3 = tok.decode(generated3)
for tag in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
    text3 = text3.replace(tag, "")
print(f"  Generated ({len(generated3)} tokens, {gen_ms:.0f}ms): '{text3}'")
print(f"  Token IDs: {generated3[:20]}...")

# Cleanup
encoder.release()
del decoder
print("\n=== Done ===")

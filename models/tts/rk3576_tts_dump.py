#!/usr/bin/env python3
"""
RK3576 TTS Diagnostic Dumper
============================
Runs the RK3576 Qwen3-TTS pipeline and dumps intermediate tensors
for comparison against the official reference (from tts_dump_reference.py).

This script is meant to be run on the RK3576 device or in its Docker container.

Usage:
    # On RK3576 device:
    python rk3576_tts_dump.py --output-dir ./tts_rk3576_dump --seed 42

Output structure (matching tts_dump_reference.py):
    {output_dir}/{case_idx}/
        token_ids.npy          — text token IDs
        text_embeds.npy        — text embeddings
        prefill_logits.npy     — talker prefill logits
        prefill_hidden.npy     — talker prefill hidden states
        primary_codes.npy      — primary codec codes
        cp_codes.npy           — CP residual codes
        codec_sums.npy         — codec embedding sums
        audio.wav              — synthesized audio
        metadata.json          — metadata
        frames/                — per-frame intermediates (first 5 frames)
"""

import argparse
import json
import os
import struct
import sys
import wave
from typing import List, Optional

import numpy as np

# Add app directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

# Test cases matching tts_dump_reference.py
TEST_CASES = [
    ("你好", "chinese"),
    ("今天天气真不错", "chinese"),
    ("今天天气真不错，我们一起去公园散步吧。", "chinese"),
]


def write_wav(path: str, audio: np.ndarray, sr: int = 12000):
    """Write mono float32 audio [-1, 1] to a 16-bit PCM WAV file."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    n = len(pcm)
    with open(path, "wb") as f:
        data_size = n * 2
        f.write(struct.pack("<4sI4s", b"RIFF", 36 + data_size, b"WAVE"))
        f.write(struct.pack("<4sIHHIIHH", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16))
        f.write(struct.pack("<4sI", b"data", data_size))
        f.write(pcm.tobytes())


class TTSDiagnostic:
    """Diagnostic wrapper for TTSService that dumps intermediate data."""

    def __init__(self, model_dir: str, output_dir: str, seed: int = 42,
                 dump_frames: int = 5):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.seed = seed
        self.dump_frames = dump_frames

        # Import and initialize TTSService
        from tts_service import TTSService
        self.service = TTSService(model_dir)
        self.service.load()

        # Constants from tts_service.py
        self.HIDDEN_SIZE = 1024
        self.NUM_CODE_GROUPS = 16
        self.CODEC_EOS_TOKEN_ID = 2150
        self.FRAMES_PER_CHAR = 8
        self.EOS_BIAS_RAMP_FRAMES = 50
        self.EOS_BIAS_MAX = 15.0
        self.SAMPLE_RATE = 12000

    def run_case(self, case_idx: int, text: str, lang: str) -> dict:
        """Run one test case and dump intermediate data."""
        print(f"\n{'='*70}")
        print(f"Case {case_idx}: '{text}' ({lang})")
        print(f"{'='*70}")

        np.random.seed(self.seed)

        case_dir = os.path.join(self.output_dir, str(case_idx))
        frames_dir = os.path.join(case_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Get internal components
        tokenizer = self.service._tokenizer
        talker = self.service._talker
        codec_head_weight = self.service._codec_head_weight

        # ------------------------------------------------------------------
        # 1. Tokenize
        # ------------------------------------------------------------------
        formatted_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(formatted_text)
        print(f"  Token IDs: {input_ids} (len={len(input_ids)})")

        # Save token_ids - extract only text tokens (same as reference)
        # The reference saves only the text tokens, not the full formatted sequence
        text_token_ids = tokenizer.encode(text, add_special_tokens=False)
        np.save(os.path.join(case_dir, "token_ids.npy"),
                np.array(text_token_ids, dtype=np.int64))

        # ------------------------------------------------------------------
        # 2. Build prefill embeddings (official format)
        # ------------------------------------------------------------------
        prefill_embeds, tts_pad_vec = self.service._build_prefill(input_ids, language=lang)
        print(f"  Prefill embeds: {prefill_embeds.shape}")

        # Get text embeddings separately
        text_embeds = self.service._run_text_project(text_token_ids)
        np.save(os.path.join(case_dir, "text_embeds.npy"),
                text_embeds.astype(np.float32))
        print(f"  text_embeds: {text_embeds.shape}")

        # ------------------------------------------------------------------
        # 3. Talker prefill
        # ------------------------------------------------------------------
        talker.clear_kv_cache()
        result = talker.run_embed(prefill_embeds, mode=1, keep_history=0)
        hidden = result["hidden"]  # [n_prefill, 1024]
        last_hidden = hidden[-1:]  # [1, 1024]

        # Compute logits
        logits = (last_hidden @ codec_head_weight.T)[0]  # [3072]

        # Save prefill outputs
        # Note: prefill_logits shape differs from reference - we only save last position
        prefill_logits_full = (hidden @ codec_head_weight.T)  # [n_prefill, 3072]
        np.save(os.path.join(case_dir, "prefill_logits.npy"),
                prefill_logits_full[np.newaxis, :, :].astype(np.float32))
        np.save(os.path.join(case_dir, "prefill_hidden.npy"),
                hidden[np.newaxis, :, :].astype(np.float32))
        print(f"  prefill_logits: {prefill_logits_full.shape}")
        print(f"  prefill_hidden: {hidden.shape}")

        # ------------------------------------------------------------------
        # 4. AR loop with intermediate capture
        # ------------------------------------------------------------------
        all_primary_codes: List[int] = []
        all_cp_codes: List[List[int]] = []
        all_codec_sums: List[np.ndarray] = []

        text_chars = len(text.strip())
        est_frames = max(text_chars * self.FRAMES_PER_CHAR, 10)
        max_new_tokens = min(300, max(text_chars * self.FRAMES_PER_CHAR * 3, 30))
        min_new_tokens = 2

        suppress_mask = np.ones(3072, dtype=bool)
        suppress_mask[:2048] = False
        suppress_mask[self.CODEC_EOS_TOKEN_ID] = False
        suppress_indices = np.where(suppress_mask)[0]

        generated_primary_codes = []

        for step in range(max_new_tokens):
            logits_copy = logits.copy()
            logits_copy[suppress_indices] = -float("inf")
            logits_copy = self.service._apply_repetition_penalty(
                logits_copy, generated_primary_codes, penalty=1.05)

            if step < min_new_tokens:
                logits_copy[self.CODEC_EOS_TOKEN_ID] = -float("inf")

            if step >= est_frames:
                ramp = min((step - est_frames) / self.EOS_BIAS_RAMP_FRAMES, 1.0)
                logits_copy[self.CODEC_EOS_TOKEN_ID] += ramp * self.EOS_BIAS_MAX

            # Log EOS info for debugging
            if step % 25 == 0 or step < 5:
                eos_logit = logits_copy[self.CODEC_EOS_TOKEN_ID]
                argmax_id = int(np.argmax(logits_copy))
                argmax_val = logits_copy[argmax_id]
                rank = int((logits_copy > eos_logit).sum())
                print(f"  Step {step}: EOS logit={eos_logit:.3f} (rank {rank}/3072), "
                      f"argmax={argmax_id} ({argmax_val:.3f}), bias={'ON' if step >= est_frames else 'OFF'}")

            # Greedy EOS check
            if step >= min_new_tokens and np.argmax(logits_copy) == self.CODEC_EOS_TOKEN_ID:
                print(f"  EOS at step {step} (greedy)")
                break

            # Sample
            primary_code = self.service._sample_top_k(
                logits_copy, top_k=5, temperature=0.8,
                eos_id=self.CODEC_EOS_TOKEN_ID if step >= min_new_tokens else None)

            if primary_code == self.CODEC_EOS_TOKEN_ID:
                print(f"  EOS at step {step}")
                break

            generated_primary_codes.append(primary_code)
            all_primary_codes.append(primary_code)

            # Primary embedding
            primary_embed = self.service._run_codec_embed([primary_code])  # [1, 1024]

            # CP residual codes
            frame_codes = [primary_code]
            if self.service._cp_engine is not None:
                codes, codec_sum = self.service._cp_engine.run(
                    last_hidden[0], primary_embed[0], num_steps=15)
                frame_codes.extend(int(c) for c in codes)
            else:
                # Fallback RKNN path (less accurate)
                codec_sum = primary_embed[0].copy()
                for j in range(15):
                    cp_input = np.stack([last_hidden[0], codec_sum])[np.newaxis, :, :]
                    cp_logits = self.service._run_code_predictor(cp_input)
                    res_code = int(np.argmax(cp_logits[0, 0, :2048]))
                    frame_codes.append(res_code)
                    res_embed = self.service._run_code_predictor_embed(res_code, j)
                    codec_sum += res_embed

            all_cp_codes.append(frame_codes[1:])  # Save only residual codes
            all_codec_sums.append(codec_sum.copy())

            # Save per-frame data (first dump_frames only)
            if step < self.dump_frames:
                np.save(os.path.join(frames_dir, f"frame_{step}_talker_logits.npy"),
                        logits[np.newaxis, np.newaxis, :].astype(np.float32))
                np.save(os.path.join(frames_dir, f"frame_{step}_talker_hidden.npy"),
                        last_hidden[np.newaxis, :, :].astype(np.float32))
                np.save(os.path.join(frames_dir, f"frame_{step}_input_embed.npy"),
                        (tts_pad_vec + codec_sum)[np.newaxis, :, :].astype(np.float32))

            # Next talker step
            if step < len(trailing_text):
                txt_hidden = trailing_text[step]
            else:
                txt_hidden = tts_pad_vec

            next_embed = (codec_sum + txt_hidden)[np.newaxis, :]
            result = talker.run_embed(next_embed, mode=1, keep_history=1)
            last_hidden = result["hidden"][-1:]
            logits = (last_hidden @ codec_head_weight.T)[0]

            if (step + 1) % 10 == 0:
                print(f"  Frame {step + 1}, primary_code={primary_code}")

        # ------------------------------------------------------------------
        # 5. Save arrays
        # ------------------------------------------------------------------
        n_frames = len(all_primary_codes)
        print(f"  Generated {n_frames} frames ({n_frames / 12.5:.2f}s audio)")

        if n_frames == 0:
            print("  WARNING: No frames generated!")
            return {"error": "No frames generated"}

        np.save(os.path.join(case_dir, "primary_codes.npy"),
                np.array(all_primary_codes, dtype=np.int32))
        np.save(os.path.join(case_dir, "cp_codes.npy"),
                np.array(all_cp_codes, dtype=np.int32))
        np.save(os.path.join(case_dir, "codec_sums.npy"),
                np.array(all_codec_sums, dtype=np.float32))

        # ------------------------------------------------------------------
        # 6. Vocoder
        # ------------------------------------------------------------------
        print("  Running vocoder...")
        codes_array = np.array([[c] + rc for c, rc in zip(all_primary_codes, all_cp_codes)],
                               dtype=np.int64)
        audio = self.service._decode_audio(codes_array)

        wav_path = os.path.join(case_dir, "audio.wav")
        write_wav(wav_path, audio, self.SAMPLE_RATE)
        print(f"  Audio: {wav_path} ({len(audio)/self.SAMPLE_RATE:.2f}s)")

        # ------------------------------------------------------------------
        # 7. Metadata
        # ------------------------------------------------------------------
        meta = {
            "case_idx": case_idx,
            "text": text,
            "language": lang,
            "n_frames": n_frames,
            "audio_duration_s": round(len(audio) / self.SAMPLE_RATE, 3),
            "sample_rate": self.SAMPLE_RATE,
            "seed": self.seed,
            "primary_codes_sample": all_primary_codes[:10],
            "platform": "RK3576",
            "models": {
                "talker": "talker_fullvocab_fixed_w4a16_rk3576.rkllm",
                "vocoder": self.service._vocoder_name,
                "cp_engine": "C engine" if self.service._cp_engine else "RKNN fallback",
            },
        }

        with open(os.path.join(case_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {case_dir}/")
        return meta


def main():
    parser = argparse.ArgumentParser(description="RK3576 TTS diagnostic dumper")
    parser.add_argument("--model-dir", default="/opt/tts/models",
                        help="Model directory")
    parser.add_argument("--output-dir", default="./tts_rk3576_dump",
                        help="Output directory for dumped data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cases", type=int, default=3,
                        help="Number of test cases")
    parser.add_argument("--dump-frames", type=int, default=5,
                        help="Number of frames to dump detailed data for")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    diag = TTSDiagnostic(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        dump_frames=args.dump_frames,
    )

    for i, (text, lang) in enumerate(TEST_CASES[:args.cases]):
        diag.run_case(i, text, lang)

    print(f"\n{'='*70}")
    print(f"Done. Diagnostic data saved to: {args.output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
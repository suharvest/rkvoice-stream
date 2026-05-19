#!/usr/bin/env python3
"""Benchmark Paraformer full-ORT vs hybrid RKNN/CPU pipeline.

This isolates the acceleration from the RKNN encoder prefix while keeping the
same Python fbank/LFR, CIF, and ONNX Runtime decoder path for both modes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.asr.paraformer.verify_paraformer_rknn_hybrid import (
    HybridEncoder,
    make_encoder_inputs,
    read_audio,
)
from models.asr.paraformer.verify_paraformer_rknn_parity import run_decoder_ort, run_text_pipeline
from rkvoice_stream.backends.asr.paraformer_rknn import (
    CACHE_COUNT,
    CACHE_SHAPE,
    DEC_MAX_ENC_FRAMES,
    DEC_MAX_TOKENS,
    SAMPLE_RATE,
    compute_fbank,
    load_tokens,
    stack_frames,
)


def _run_pipeline(encoder: HybridEncoder, decoder, feats: np.ndarray, tokens: list[str], frames: int, mode: str):
    def enc_ort_fn(chunk: np.ndarray):
        speech, speech_len, enc_mask, cif_mask, orig_frames = make_encoder_inputs(chunk, frames)
        enc, _enc_len, alphas = encoder.run_full_ort(speech, speech_len, enc_mask, cif_mask, orig_frames)
        return enc, alphas

    def enc_hybrid_fn(chunk: np.ndarray):
        enc, _enc_len, alphas, _prefix = encoder.run_hybrid(chunk, frames)
        return enc, alphas

    def dec_cpu_fn(enc: np.ndarray, enc_len: int, ae: np.ndarray, caches: list[np.ndarray]):
        return run_decoder_ort(decoder, enc, enc_len, ae, caches, DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS)

    return run_text_pipeline(enc_hybrid_fn if mode == "hybrid" else enc_ort_fn, dec_cpu_fn, feats, tokens, frames)


def _bench_mode(
    mode: str,
    encoder: HybridEncoder,
    decoder,
    wavs: list[Path],
    tokens: list[str],
    frames: int,
    repeat: int,
    warmup: int,
) -> dict[str, Any]:
    results = []
    for wav in wavs:
        audio = read_audio(wav)
        feats = stack_frames(compute_fbank(audio))
        duration_s = len(audio) / SAMPLE_RATE
        text = ""
        for _ in range(warmup):
            text = _run_pipeline(encoder, decoder, feats, tokens, frames, mode)["text"]
        times_ms = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            text = _run_pipeline(encoder, decoder, feats, tokens, frames, mode)["text"]
            times_ms.append((time.perf_counter() - t0) * 1000)
        mean_ms = statistics.mean(times_ms)
        results.append(
            {
                "wav": str(wav),
                "duration_s": duration_s,
                "times_ms": times_ms,
                "mean_ms": mean_ms,
                "median_ms": statistics.median(times_ms),
                "rtf_mean": mean_ms / 1000.0 / max(duration_s, 1e-9),
                "text": text,
            }
        )
    total_audio_s = sum(r["duration_s"] for r in results)
    total_mean_ms = sum(r["mean_ms"] for r in results)
    return {
        "mode": mode,
        "total_audio_s": total_audio_s,
        "total_mean_ms": total_mean_ms,
        "weighted_rtf_mean": total_mean_ms / 1000.0 / max(total_audio_s, 1e-9),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prefix-rknn", type=Path, required=True)
    parser.add_argument("--prefix-onnx", type=Path, required=True)
    parser.add_argument("--suffix-onnx", type=Path, required=True)
    parser.add_argument("--full-encoder-onnx", type=Path, required=True)
    parser.add_argument("--decoder-onnx", type=Path, required=True)
    parser.add_argument("--core-mask", default="NPU_CORE_1")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--wav", type=Path, action="append", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    tokens = load_tokens(str(args.model_dir / "tokens.txt"))
    decoder = ort.InferenceSession(str(args.decoder_onnx), providers=["CPUExecutionProvider"])
    encoder = HybridEncoder(args.prefix_rknn, args.prefix_onnx, args.suffix_onnx, args.full_encoder_onnx, args.core_mask)
    try:
        ort_result = _bench_mode("ort", encoder, decoder, args.wav, tokens, args.frames, args.repeat, args.warmup)
        hybrid_result = _bench_mode("hybrid", encoder, decoder, args.wav, tokens, args.frames, args.repeat, args.warmup)
    finally:
        encoder.close()

    speedup = ort_result["weighted_rtf_mean"] / max(hybrid_result["weighted_rtf_mean"], 1e-9)
    summary = {
        "frames": args.frames,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "ort": ort_result,
        "hybrid": hybrid_result,
        "speedup_hybrid_vs_ort": speedup,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

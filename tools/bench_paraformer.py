#!/usr/bin/env python3
"""Benchmark Paraformer ASR backends on local audio files.

Examples:

  python tools/bench_paraformer.py \
    --backend paraformer_rknn \
    --model-dir /opt/asr/paraformer \
    --rknn-dir /opt/asr/paraformer/rknn \
    --encoder-mode hybrid \
    --decoder-backend cpu \
    --encoder-suffix-onnx /opt/asr/paraformer/encoder_suffix_from_block30.onnx \
    --decoder-onnx /opt/asr/paraformer/decoder-rknn.onnx \
    --wav test_wavs/0.wav --repeat 3

  python tools/bench_paraformer.py \
    --backend paraformer_sherpa \
    --model-dir /opt/asr/paraformer \
    --wav test_wavs/0.wav --repeat 3
"""

from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import soundfile as sf


def _set_env(args: argparse.Namespace) -> None:
    os.environ["ASR_BACKEND"] = args.backend
    os.environ["PARAFORMER_MODEL_DIR"] = str(args.model_dir)
    if args.rknn_dir:
        os.environ["PARAFORMER_RKNN_DIR"] = str(args.rknn_dir)
    if args.encoder_precision:
        os.environ["PARAFORMER_RKNN_ENC_PRECISION"] = args.encoder_precision
    if args.decoder_precision:
        os.environ["PARAFORMER_RKNN_DEC_PRECISION"] = args.decoder_precision
    if args.encoder_mode:
        os.environ["PARAFORMER_RKNN_ENCODER_MODE"] = args.encoder_mode
    if args.decoder_backend:
        os.environ["PARAFORMER_RKNN_DECODER"] = args.decoder_backend
    if args.encoder_suffix_onnx:
        os.environ["PARAFORMER_ENCODER_SUFFIX_ONNX"] = str(args.encoder_suffix_onnx)
    if args.decoder_onnx:
        os.environ["PARAFORMER_DECODER_ONNX"] = str(args.decoder_onnx)
    if args.encoder_core:
        os.environ["PARAFORMER_RKNN_ENC_CORE"] = args.encoder_core
    if args.decoder_core:
        os.environ["PARAFORMER_RKNN_DEC_CORE"] = args.decoder_core
    if args.num_threads:
        os.environ["PARAFORMER_NUM_THREADS"] = str(args.num_threads)


def _wav_duration_s(path: Path) -> float:
    with sf.SoundFile(path) as f:
        return float(len(f) / f.samplerate)


def _wav_bytes(path: Path) -> bytes:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _bench_one(backend, wav: Path, repeat: int, warmup: int) -> dict[str, Any]:
    audio_bytes = _wav_bytes(wav)
    duration_s = _wav_duration_s(wav)
    text = ""

    for _ in range(warmup):
        text = backend.transcribe(audio_bytes).text

    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = backend.transcribe(audio_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        text = result.text

    mean_ms = statistics.mean(times_ms)
    return {
        "wav": str(wav),
        "duration_s": duration_s,
        "repeat": repeat,
        "times_ms": times_ms,
        "mean_ms": mean_ms,
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "rtf_mean": mean_ms / 1000.0 / max(duration_s, 1e-9),
        "text": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["paraformer_rknn", "paraformer_sherpa"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rknn-dir", type=Path)
    parser.add_argument("--encoder-suffix-onnx", type=Path)
    parser.add_argument("--decoder-onnx", type=Path)
    parser.add_argument("--encoder-mode", choices=["auto", "full", "hybrid"])
    parser.add_argument("--decoder-backend", choices=["cpu", "rknn"])
    parser.add_argument("--encoder-precision", default="fp16")
    parser.add_argument("--decoder-precision", default="")
    parser.add_argument("--encoder-core", default="")
    parser.add_argument("--decoder-core", default="")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--wav", type=Path, action="append", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _set_env(args)

    from rkvoice_stream.engine.asr import create_asr

    backend = create_asr(args.backend)
    t0 = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - t0) * 1000

    results = [_bench_one(backend, wav, args.repeat, args.warmup) for wav in args.wav]
    if hasattr(backend, "cleanup"):
        backend.cleanup()

    total_audio_s = sum(r["duration_s"] for r in results)
    total_mean_ms = sum(r["mean_ms"] for r in results)
    summary = {
        "backend": args.backend,
        "model_dir": str(args.model_dir),
        "rknn_dir": str(args.rknn_dir) if args.rknn_dir else "",
        "encoder_mode": args.encoder_mode or "",
        "decoder_backend": args.decoder_backend or "",
        "preload_ms": preload_ms,
        "total_audio_s": total_audio_s,
        "total_mean_ms": total_mean_ms,
        "weighted_rtf_mean": total_mean_ms / 1000.0 / max(total_audio_s, 1e-9),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

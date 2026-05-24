#!/usr/bin/env python3
"""Probe MOSS streaming codec multi-frame equivalence and latency."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_ort import MossORTBackend


def _frames(count: int) -> list[list[int]]:
    return [[(frame * 37 + q * 17 + 11) % 1024 for q in range(16)] for frame in range(count)]


def _concat_audio(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((1, 2, 0), dtype=np.float32)
    return np.concatenate(chunks, axis=-1)


def _rms(audio: np.ndarray) -> float:
    f32 = audio.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(f32 * f32))) if f32.size else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", default="moss-ort-manifest.json")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-threads", type=int)
    parser.add_argument("--decode-threads", type=int)
    parser.add_argument("--sampler-threads", type=int)
    parser.add_argument("--codec-threads", type=int)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=4)
    parser.add_argument("--max-abs-diff", type=float, default=1e-5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    os.environ["MOSS_ORT_MODEL_DIR"] = str(args.model_dir)
    os.environ["MOSS_ORT_MANIFEST"] = args.manifest
    os.environ["MOSS_ORT_THREADS"] = str(args.threads)
    os.environ["MOSS_ORT_CODEC_STREAMING"] = "1"
    os.environ["MOSS_ORT_CACHE_VOICE_PREFIX"] = "0"
    for attr, env_name in (
        ("prefill_threads", "MOSS_ORT_PREFILL_THREADS"),
        ("decode_threads", "MOSS_ORT_DECODE_THREADS"),
        ("sampler_threads", "MOSS_ORT_SAMPLER_THREADS"),
        ("codec_threads", "MOSS_ORT_CODEC_THREADS"),
    ):
        value = getattr(args, attr)
        if value is not None:
            os.environ[env_name] = str(value)

    backend = MossORTBackend()
    load_start = time.perf_counter()
    backend.preload()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    session = backend._codec_stream_session
    if session is None:
        raise RuntimeError("streaming codec session is not loaded")

    frames = _frames(args.frames)

    session.reset()
    single_chunks: list[np.ndarray] = []
    single_times: list[float] = []
    for frame in frames:
        start = time.perf_counter()
        audio, _ = session.run_frames([frame])
        single_times.append((time.perf_counter() - start) * 1000.0)
        single_chunks.append(audio)
    single = _concat_audio(single_chunks)

    session.reset()
    batch_chunks: list[np.ndarray] = []
    batch_times: list[float] = []
    for start_index in range(0, len(frames), args.batch_frames):
        group = frames[start_index : start_index + args.batch_frames]
        start = time.perf_counter()
        audio, _ = session.run_frames(group)
        batch_times.append((time.perf_counter() - start) * 1000.0)
        batch_chunks.append(audio)
    batch = _concat_audio(batch_chunks)
    backend.cleanup()

    if single.shape != batch.shape:
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float(np.max(np.abs(single - batch))) if single.size else 0.0
    result: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "frames": args.frames,
        "batch_frames": args.batch_frames,
        "load_ms": round(load_ms, 3),
        "single": {
            "shape": list(single.shape),
            "calls": len(single_times),
            "total_ms": round(sum(single_times), 3),
            "max_ms": round(max(single_times), 3) if single_times else 0.0,
            "rms": _rms(single),
        },
        "batch": {
            "shape": list(batch.shape),
            "calls": len(batch_times),
            "total_ms": round(sum(batch_times), 3),
            "max_ms": round(max(batch_times), 3) if batch_times else 0.0,
            "rms": _rms(batch),
        },
        "max_abs_diff": max_abs_diff,
    }
    result["passed"] = bool(max_abs_diff <= args.max_abs_diff)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

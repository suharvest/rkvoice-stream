#!/usr/bin/env python3
"""Probe whether MOSS token generation can overlap streaming codec work."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_ort import MossORTBackend


def _set_env(args: argparse.Namespace) -> None:
    os.environ["MOSS_ORT_MODEL_DIR"] = str(args.model_dir)
    os.environ["MOSS_ORT_MANIFEST"] = args.manifest
    os.environ["MOSS_ORT_THREADS"] = str(args.threads)
    os.environ["MOSS_ORT_CODEC_STREAMING"] = "1"
    os.environ["MOSS_ORT_CODEC_BATCH_FRAMES"] = str(args.codec_batch_frames)
    os.environ["MOSS_ORT_CACHE_VOICE_PREFIX"] = "0"
    os.environ["MOSS_ORT_WARMUP_TEXT"] = args.warmup_text
    os.environ["MOSS_ORT_VOICE"] = args.voice
    os.environ["MOSS_ORT_SEED"] = str(args.seed)
    for attr, env_name in (
        ("prefill_threads", "MOSS_ORT_PREFILL_THREADS"),
        ("decode_threads", "MOSS_ORT_DECODE_THREADS"),
        ("sampler_threads", "MOSS_ORT_SAMPLER_THREADS"),
        ("codec_threads", "MOSS_ORT_CODEC_THREADS"),
    ):
        value = getattr(args, attr)
        if value is not None:
            os.environ[env_name] = str(value)


def _concat_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _generate_frames(backend: MossORTBackend, text: str, max_new_frames: int, seed: int) -> tuple[list[list[int]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    input_ids, mode = backend._build_prefill_rows(text)
    attention_mask = (input_ids[:, :, 0] != backend._pad_token_id()).astype(np.int32)
    if not attention_mask.any():
        attention_mask[:] = 1
    prefill_names = backend._tts_meta["onnx"]["prefill_output_names"]
    prefill_outputs = backend._prefill.run(prefill_names, {"input_ids": input_ids, "attention_mask": attention_mask})
    global_hidden = prefill_outputs[0]
    kv_cache = {prefill_names[i]: prefill_outputs[i] for i in range(1, len(prefill_names))}
    prefill_ms = (time.perf_counter() - start) * 1000.0
    past_len = int(attention_mask.sum())
    current_hidden = global_hidden[:, past_len - 1, :].astype(np.float32, copy=False)
    repetition_seen_mask = np.zeros((1, 16, 1024), dtype=np.int32)
    last_frame_tokens: np.ndarray | None = None
    frames: list[list[int]] = []
    decode_ms = 0.0
    sampler_ms = 0.0
    token_start = time.perf_counter()
    for frame_index in range(max_new_frames):
        if frame_index > 0:
            assert last_frame_tokens is not None
            row = backend._make_audio_row(last_frame_tokens).reshape(1, 1, 17)
            start_decode = time.perf_counter()
            current_hidden, kv_cache = backend._decode_step(row, kv_cache, past_len)
            decode_ms += (time.perf_counter() - start_decode) * 1000.0
            past_len += 1
        start_sampler = time.perf_counter()
        should_continue, frame_token_ids = backend._sampler.run(
            None,
            {
                "global_hidden": current_hidden.astype(np.float32, copy=False),
                "repetition_seen_mask": repetition_seen_mask,
                "assistant_random_u": np.asarray([min(0.99999994, max(0.0, float(rng.random())))], dtype=np.float32),
                "audio_random_u": np.asarray(
                    [[min(0.99999994, max(0.0, float(rng.random()))) for _ in range(16)]],
                    dtype=np.float32,
                ),
            },
        )
        sampler_ms += (time.perf_counter() - start_sampler) * 1000.0
        frame = np.asarray(frame_token_ids, dtype=np.int32).reshape(1, 16)
        for index, token_id in enumerate(frame[0]):
            if 0 <= token_id < repetition_seen_mask.shape[2]:
                repetition_seen_mask[0, index, token_id] = 1
        last_frame_tokens = frame[0]
        frames.append(frame[0].astype(np.int32, copy=False).tolist())
        if int(np.asarray(should_continue).reshape(-1)[0]) == 0:
            break
    return frames, {
        "mode": mode,
        "frames": len(frames),
        "prefill_ms": round(prefill_ms, 3),
        "token_ms": round((time.perf_counter() - token_start) * 1000.0, 3),
        "decode_ms": round(decode_ms, 3),
        "sampler_ms": round(sampler_ms, 3),
    }


def _run_serial_codec(backend: MossORTBackend, frames: list[list[int]], batch_frames: int) -> tuple[list[np.ndarray], list[dict[str, Any]], float]:
    assert backend._codec_stream_session is not None
    backend._codec_stream_session.reset()
    chunks: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    pending: list[list[int]] = []
    pending_start = 0
    start_all = time.perf_counter()
    for frame_index, frame in enumerate(frames):
        if not pending:
            pending_start = frame_index
        pending.append(frame)
        target = 1 if frame_index == 0 else batch_frames
        if len(pending) < target and frame_index + 1 < len(frames):
            continue
        start = time.perf_counter()
        audio, audio_len = backend._codec_stream_session.run_frames(pending)
        codec_ms = (time.perf_counter() - start) * 1000.0
        chunks.append(backend._normalize_audio(audio, np.asarray([audio_len], dtype=np.int32)))
        metas.append({"chunk_start_index": pending_start, "chunk_index": frame_index, "batch_frames": len(pending), "codec_ms": codec_ms})
        pending = []
    return chunks, metas, (time.perf_counter() - start_all) * 1000.0


def _run_overlap_codec(backend: MossORTBackend, frames: list[list[int]], batch_frames: int, token_delay_ms: float) -> tuple[list[np.ndarray], list[dict[str, Any]], float]:
    assert backend._codec_stream_session is not None
    backend._codec_stream_session.reset()
    chunks: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    def run_codec(group: list[list[int]]) -> tuple[np.ndarray, int]:
        return backend._codec_stream_session.run_frames(group)

    start_all = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future: Future[tuple[np.ndarray, int]] | None = None
        future_meta: dict[str, Any] | None = None
        pending: list[list[int]] = []
        pending_start = 0
        for frame_index, frame in enumerate(frames):
            if token_delay_ms > 0:
                time.sleep(token_delay_ms / 1000.0)
            if not pending:
                pending_start = frame_index
            pending.append(frame)
            target = 1 if frame_index == 0 else batch_frames
            if len(pending) < target and frame_index + 1 < len(frames):
                continue
            if future is not None and future_meta is not None:
                wait_start = time.perf_counter()
                audio, audio_len = future.result()
                future_meta["wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
                chunks.append(backend._normalize_audio(audio, np.asarray([audio_len], dtype=np.int32)))
                metas.append(future_meta)
            submit_start = time.perf_counter()
            future = executor.submit(run_codec, list(pending))
            future_meta = {
                "chunk_start_index": pending_start,
                "chunk_index": frame_index,
                "batch_frames": len(pending),
                "submit_ms": (time.perf_counter() - submit_start) * 1000.0,
            }
            pending = []
        if future is not None and future_meta is not None:
            wait_start = time.perf_counter()
            audio, audio_len = future.result()
            future_meta["wait_ms"] = (time.perf_counter() - wait_start) * 1000.0
            chunks.append(backend._normalize_audio(audio, np.asarray([audio_len], dtype=np.int32)))
            metas.append(future_meta)
    return chunks, metas, (time.perf_counter() - start_all) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", default="moss-ort-manifest.json")
    parser.add_argument("--text", default="你好")
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-threads", type=int)
    parser.add_argument("--decode-threads", type=int)
    parser.add_argument("--sampler-threads", type=int)
    parser.add_argument("--codec-threads", type=int)
    parser.add_argument("--codec-batch-frames", type=int, default=4)
    parser.add_argument("--max-new-frames", type=int, default=20)
    parser.add_argument("--warmup-text", default="你好")
    parser.add_argument("--token-delay-ms", type=float, default=0.0)
    parser.add_argument("--max-abs-diff", type=float, default=1e-5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _set_env(args)
    backend = MossORTBackend()
    load_start = time.perf_counter()
    backend.preload()
    load_ms = (time.perf_counter() - load_start) * 1000.0
    try:
        frames, token_meta = _generate_frames(backend, args.text, args.max_new_frames, args.seed)
        serial_chunks, serial_meta, serial_codec_ms = _run_serial_codec(backend, frames, args.codec_batch_frames)
        serial_audio = _concat_chunks(serial_chunks)
        overlap_chunks, overlap_meta, overlap_codec_ms = _run_overlap_codec(
            backend,
            frames,
            args.codec_batch_frames,
            args.token_delay_ms,
        )
        overlap_audio = _concat_chunks(overlap_chunks)
    finally:
        backend.cleanup()

    if serial_audio.shape != overlap_audio.shape:
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float(np.max(np.abs(serial_audio - overlap_audio))) if serial_audio.size else 0.0
    result: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "load_ms": round(load_ms, 3),
        "token_generation": token_meta,
        "codec_batch_frames": args.codec_batch_frames,
        "token_delay_ms": args.token_delay_ms,
        "serial": {
            "chunks": len(serial_chunks),
            "codec_wall_ms": round(serial_codec_ms, 3),
            "audio_shape": list(serial_audio.shape),
            "metas": [{**item, "codec_ms": round(float(item["codec_ms"]), 3)} for item in serial_meta],
        },
        "overlap": {
            "chunks": len(overlap_chunks),
            "codec_wall_ms": round(overlap_codec_ms, 3),
            "audio_shape": list(overlap_audio.shape),
            "metas": [
                {
                    **item,
                    "submit_ms": round(float(item.get("submit_ms", 0.0)), 3),
                    "wait_ms": round(float(item.get("wait_ms", 0.0)), 3),
                }
                for item in overlap_meta
            ],
        },
        "estimated_serial_total_ms": round(float(token_meta["prefill_ms"]) + float(token_meta["token_ms"]) + serial_codec_ms, 3),
        "estimated_overlap_total_ms": round(float(token_meta["prefill_ms"]) + max(float(token_meta["token_ms"]), overlap_codec_ms), 3),
        "max_abs_diff": max_abs_diff,
    }
    result["passed"] = bool(max_abs_diff <= args.max_abs_diff)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify MOSS ONNX Runtime streaming baseline latency.

This is the production accuracy fallback while RKNN subgraphs are being split
and stabilized. It exercises the low-latency path needed for dialogue:

  prefill global model -> local fixed-frame sampler -> codec full decode

The script intentionally uses deterministic dummy token ids and sampler
randomness. It proves model load, tensor contracts, finite outputs, first audio
chunk shape, RMS, and TTFA for the first 80 ms / 3840-sample stereo chunk.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def make_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    start = time.perf_counter()
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    return sess, (time.perf_counter() - start) * 1000


def finite_summary(name: str, array) -> dict:
    arr = np.asarray(array)
    item = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.number) else None,
    }
    if np.issubdtype(arr.dtype, np.number) and arr.size:
        f32 = arr.astype(np.float32, copy=False)
        item["mean"] = float(f32.mean())
        item["rms"] = float(np.sqrt(np.mean(f32 * f32)))
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prefill-seq", type=int, default=32)
    parser.add_argument("--max-ttfa-ms", type=float, default=500.0)
    parser.add_argument("--min-rms", type=float, default=0.02)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    prefill_path = args.model_dir / "moss_tts_prefill.onnx"
    sampler_path = args.model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    codec_path = args.model_dir / "moss_audio_tokenizer_decode_full.onnx"
    for path in (prefill_path, sampler_path, codec_path):
        if not path.exists():
            raise FileNotFoundError(path)

    prefill, prefill_load_ms = make_session(prefill_path, args.threads)
    sampler, sampler_load_ms = make_session(sampler_path, args.threads)
    codec, codec_load_ms = make_session(codec_path, args.threads)

    input_ids = np.zeros((1, args.prefill_seq, 17), dtype=np.int32)
    attention_mask = np.ones((1, args.prefill_seq), dtype=np.int32)
    input_ids[:, :, 0] = 1

    start = time.perf_counter()
    global_hidden = prefill.run(["global_hidden"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]
    prefill_ms = (time.perf_counter() - start) * 1000
    last_hidden = global_hidden[:, -1, :].astype(np.float32, copy=False)

    sampler_inputs = {
        "global_hidden": last_hidden,
        "repetition_seen_mask": np.zeros((1, 16, 1024), dtype=np.int32),
        "assistant_random_u": np.array([0.5], dtype=np.float32),
        "audio_random_u": np.full((1, 16), 0.5, dtype=np.float32),
    }
    start = time.perf_counter()
    should_continue, frame_token_ids = sampler.run(None, sampler_inputs)
    sampler_ms = (time.perf_counter() - start) * 1000

    codec_inputs = {
        "audio_codes": frame_token_ids.reshape(1, 1, 16).astype(np.int32, copy=False),
        "audio_code_lengths": np.array([1], dtype=np.int32),
    }
    start = time.perf_counter()
    audio, audio_lengths = codec.run(None, codec_inputs)
    codec_ms = (time.perf_counter() - start) * 1000

    ttfa_ms = prefill_ms + sampler_ms + codec_ms
    audio_rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
    passed = (
        bool(np.isfinite(global_hidden).all())
        and bool(np.isfinite(audio).all())
        and bool(should_continue.reshape(-1)[0] == 1)
        and ttfa_ms <= args.max_ttfa_ms
        and audio_rms >= args.min_rms
    )

    result = {
        "model_dir": str(args.model_dir),
        "threads": args.threads,
        "load_ms": {
            "prefill": round(prefill_load_ms, 3),
            "sampler": round(sampler_load_ms, 3),
            "codec": round(codec_load_ms, 3),
        },
        "stream_ms": {
            "prefill": round(prefill_ms, 3),
            "sampler": round(sampler_ms, 3),
            "codec": round(codec_ms, 3),
            "ttfa": round(ttfa_ms, 3),
        },
        "outputs": [
            finite_summary("global_hidden", global_hidden),
            finite_summary("should_continue", should_continue),
            finite_summary("frame_token_ids", frame_token_ids),
            finite_summary("audio", audio),
            finite_summary("audio_lengths", audio_lengths),
        ],
        "gates": {
            "max_ttfa_ms": args.max_ttfa_ms,
            "min_rms": args.min_rms,
            "passed": passed,
        },
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())

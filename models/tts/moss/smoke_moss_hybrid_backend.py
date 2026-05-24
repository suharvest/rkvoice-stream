#!/usr/bin/env python3
"""Smoke-test MOSS ORT/hybrid backend streaming from environment config."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rkvoice_stream.backends.tts.moss_ort import MossORTBackend


def _float_meta(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chunk_samples(chunk: Any) -> int:
    shape = list(getattr(chunk, "shape", []))
    return int(shape[0]) if shape else 0


def _summarize(
    preload_ms: float,
    wall_ms: float,
    chunks: list[Any],
    metas: list[dict[str, Any]],
    samples_per_frame: int = 3840,
) -> dict[str, Any]:
    first_meta = metas[0] if metas else {}
    codec_values = [value for value in (_float_meta(meta, "codec_ms") for meta in metas) if value is not None]
    sampler_values = [value for value in (_float_meta(meta, "sampler_ms") for meta in metas) if value is not None]
    decode_values = [value for value in (_float_meta(meta, "decode_ms") for meta in metas) if value is not None]
    total_samples = sum(_chunk_samples(chunk) for chunk in chunks)
    return {
        "chunks": len(chunks),
        "total_samples": total_samples,
        "audio_frames": total_samples // samples_per_frame if samples_per_frame > 0 else None,
        "wall_ms": round(wall_ms, 3),
        "preload_ms": round(preload_ms, 3),
        "first_meta": first_meta or None,
        "ttfa_ms": _float_meta(first_meta, "ttfa_ms"),
        "prefill_ms": _float_meta(first_meta, "prefill_ms"),
        "hybrid_prefill_ms": (
            _float_meta(first_meta.get("hybrid", {}) if isinstance(first_meta.get("hybrid"), dict) else {}, "hybrid_prefill_ms")
        ),
        "max_codec_ms": max(codec_values) if codec_values else None,
        "max_sampler_ms": max(sampler_values) if sampler_values else None,
        "max_decode_ms": max(decode_values) if decode_values else None,
        "metas": metas,
        "chunk_shapes": [list(getattr(chunk, "shape", [])) for chunk in chunks],
    }


def _collect_gate_errors(
    summary: dict[str, Any],
    *,
    min_chunks: int = 0,
    min_audio_frames: int = 0,
    max_ttfa_ms: float = 0.0,
    max_prefill_ms: float = 0.0,
    max_codec_ms: float = 0.0,
) -> list[str]:
    errors: list[str] = []
    if min_chunks > 0 and int(summary.get("chunks") or 0) < min_chunks:
        errors.append(f"chunks={summary.get('chunks')!r} below {min_chunks}")
    if min_audio_frames > 0 and int(summary.get("audio_frames") or 0) < min_audio_frames:
        errors.append(f"audio_frames={summary.get('audio_frames')!r} below {min_audio_frames}")
    ttfa = summary.get("ttfa_ms")
    if max_ttfa_ms > 0 and ttfa is not None and float(ttfa) > max_ttfa_ms:
        errors.append(f"ttfa_ms={float(ttfa):.3f} exceeds {max_ttfa_ms:.3f}")
    prefill = summary.get("prefill_ms")
    if max_prefill_ms > 0 and prefill is not None and float(prefill) > max_prefill_ms:
        errors.append(f"prefill_ms={float(prefill):.3f} exceeds {max_prefill_ms:.3f}")
    codec = summary.get("max_codec_ms")
    if max_codec_ms > 0 and codec is not None and float(codec) > max_codec_ms:
        errors.append(f"max_codec_ms={float(codec):.3f} exceeds {max_codec_ms:.3f}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--max-new-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-chunks", type=int, default=0)
    parser.add_argument("--min-audio-frames", type=int, default=0)
    parser.add_argument("--max-ttfa-ms", type=float, default=0.0)
    parser.add_argument("--max-prefill-ms", type=float, default=0.0)
    parser.add_argument("--max-codec-ms", type=float, default=0.0)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    backend = MossORTBackend()
    t0 = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - t0) * 1000.0
    print(f"PRELOAD_MS {preload_ms:.3f}", flush=True)

    chunks = []
    metas = []
    t0 = time.perf_counter()
    try:
        for chunk, meta in backend.synthesize_stream(args.text, max_new_frames=args.max_new_frames, seed=args.seed):
            chunks.append(chunk)
            metas.append(meta)
            print(
                "CHUNK",
                len(chunks),
                list(chunk.shape),
                json.dumps(meta, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
    finally:
        backend.cleanup()
    summary = _summarize(preload_ms, (time.perf_counter() - t0) * 1000.0, chunks, metas)
    errors = _collect_gate_errors(
        summary,
        min_chunks=args.min_chunks,
        min_audio_frames=args.min_audio_frames,
        max_ttfa_ms=args.max_ttfa_ms,
        max_prefill_ms=args.max_prefill_ms,
        max_codec_ms=args.max_codec_ms,
    )
    report = {
        "text": args.text,
        "max_new_frames": args.max_new_frames,
        "seed": args.seed,
        "summary": summary,
        "gates": {
            "min_chunks": args.min_chunks,
            "min_audio_frames": args.min_audio_frames,
            "max_ttfa_ms": args.max_ttfa_ms,
            "max_prefill_ms": args.max_prefill_ms,
            "max_codec_ms": args.max_codec_ms,
            "passed": not errors,
            "errors": errors,
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.write_text(output, encoding="utf-8")
    print("DONE", json.dumps({"summary": summary, "gates": report["gates"]}, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify RK3576 MOSS ORT TTS quality with an isolated ASR roundtrip.

The full verifier intentionally runs TTS generation and ASR transcription in
separate Python processes. On 8 GB RK3576 boards, co-loading MOSS ORT sessions
and Paraformer ASR in one process can OOM even when both components pass alone.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any

DEFAULT_SENTENCES = (
    "你好",
    "欢迎使用语音服务",
    "语音识别测试一二三四五",
)


def normalize_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            old = row[j]
            row[j] = prev if ca == cb else min(prev, row[j], row[j - 1]) + 1
            prev = old
    return row[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wav_info(wav_bytes: bytes) -> dict[str, Any]:
    import numpy as np

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        raw = wav.readframes(frames)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_s": frames / sample_rate if sample_rate else 0.0,
        "rms": float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0,
    }


def parse_sentences(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_SENTENCES)
    path = Path(raw)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [item.strip() for item in raw.split("|") if item.strip()]


def _env_with_path(base_env: dict[str, str], repo_root: Path) -> dict[str, str]:
    env = dict(base_env)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not current else f"{repo_root}{os.pathsep}{current}"
    return env


def run_stage_subprocess(args: argparse.Namespace, stage: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--model-dir",
        str(args.model_dir),
        "--asr-model-dir",
        str(args.asr_model_dir),
        "--out-dir",
        str(args.out_dir),
        "--threads",
        str(args.threads),
        "--max-new-frames",
        str(args.max_new_frames),
        "--voice",
        args.voice,
        "--prefill-seq",
        str(args.prefill_seq),
        "--codec-streaming",
        str(args.codec_streaming),
        "--codec-batch-frames",
        str(args.codec_batch_frames),
    ]
    if getattr(args, "manifest", ""):
        cmd.extend(["--manifest", args.manifest])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    for attr, flag in (
        ("prefill_threads", "--prefill-threads"),
        ("decode_threads", "--decode-threads"),
        ("sampler_threads", "--sampler-threads"),
        ("codec_threads", "--codec-threads"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.warmup_text:
        cmd.extend(["--warmup-text", args.warmup_text])
    if args.sentences:
        cmd.extend(["--sentences", args.sentences])
    subprocess.run(cmd, check=True, env=_env_with_path(os.environ, repo_root))


def run_tts_stage(args: argparse.Namespace) -> None:
    os.environ["MOSS_ORT_MODEL_DIR"] = str(args.model_dir)
    os.environ["MOSS_ORT_THREADS"] = str(args.threads)
    for attr, env_name in (
        ("prefill_threads", "MOSS_ORT_PREFILL_THREADS"),
        ("decode_threads", "MOSS_ORT_DECODE_THREADS"),
        ("sampler_threads", "MOSS_ORT_SAMPLER_THREADS"),
        ("codec_threads", "MOSS_ORT_CODEC_THREADS"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_name] = str(value)
    os.environ["MOSS_ORT_MAX_NEW_FRAMES"] = str(args.max_new_frames)
    os.environ["MOSS_ORT_PREFILL_SEQ"] = str(args.prefill_seq)
    os.environ["MOSS_ORT_VOICE"] = args.voice
    os.environ["MOSS_ORT_CODEC_STREAMING"] = str(args.codec_streaming)
    os.environ["MOSS_ORT_CODEC_BATCH_FRAMES"] = str(args.codec_batch_frames)
    os.environ["MOSS_ORT_CODEC_ASYNC"] = "0"
    os.environ["MOSS_ORT_CACHE_VOICE_PREFIX"] = "0"
    os.environ["MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK"] = "0"
    os.environ["MOSS_ORT_HYBRID_RKNN"] = "0"
    os.environ["MOSS_ORT_WARMUP_TEXT"] = args.warmup_text
    if getattr(args, "manifest", ""):
        os.environ["MOSS_ORT_MANIFEST"] = args.manifest
    if args.seed is not None:
        os.environ["MOSS_ORT_SEED"] = str(args.seed)

    from rkvoice_stream.backends.tts.moss_ort import MossORTBackend

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tts = MossORTBackend()
    start = time.perf_counter()
    tts.preload()
    print(json.dumps({"event": "tts_loaded", "load_ms": round((time.perf_counter() - start) * 1000, 3)}), flush=True)

    manifest = []
    for idx, text in enumerate(parse_sentences(args.sentences), 1):
        start = time.perf_counter()
        wav_bytes, meta = tts.synthesize(text, max_new_frames=args.max_new_frames)
        wav_path = args.out_dir / f"moss_{idx}.wav"
        wav_path.write_bytes(wav_bytes)
        item = {
            "id": idx,
            "text": text,
            "wav": str(wav_path),
            "bytes": len(wav_bytes),
            "tts_ms": round((time.perf_counter() - start) * 1000, 3),
            "meta": meta,
        }
        manifest.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_asr_stage(args: argparse.Namespace) -> None:
    os.environ["PARAFORMER_MODEL_DIR"] = str(args.asr_model_dir)
    os.environ.setdefault("PARAFORMER_NUM_THREADS", "2")

    from rkvoice_stream.backends.asr.paraformer_sherpa import ParaformerSherpaBackend

    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr = ParaformerSherpaBackend()
    start = time.perf_counter()
    asr.preload()
    print(json.dumps({"event": "asr_loaded", "load_ms": round((time.perf_counter() - start) * 1000, 3)}), flush=True)

    results = []
    for item in manifest:
        wav_bytes = Path(item["wav"]).read_bytes()
        start = time.perf_counter()
        hypothesis = asr.transcribe(wav_bytes, language="Chinese").text
        result = {
            **item,
            "hypothesis": hypothesis,
            "cer": cer(item["text"], hypothesis),
            "wav_info": wav_info(wav_bytes),
            "asr_ms": round((time.perf_counter() - start) * 1000, 3),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = summarize(results)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary, ensure_ascii=False), flush=True)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "avg_cer": 1.0,
            "max_cer": 1.0,
            "min_rms": 0.0,
            "max_ttfa_ms": None,
            "max_codec_ms": None,
            "profile": {},
            "results": [],
        }
    ttfas = [float(item["meta"].get("ttfa_ms", 0.0) or 0.0) for item in results]
    codec_times = [float(item["meta"].get("codec_ms", 0.0) or 0.0) for item in results]
    return {
        "avg_cer": sum(float(item["cer"]) for item in results) / len(results),
        "max_cer": max(float(item["cer"]) for item in results),
        "min_rms": min(float(item["wav_info"]["rms"]) for item in results),
        "max_ttfa_ms": max(ttfas),
        "max_codec_ms": max(codec_times),
        "profile": {
            "backend": "moss_ort",
            "voice": results[0]["meta"].get("voice") if results[0].get("meta") else None,
        },
        "results": results,
    }


def evaluate_gates(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "avg_cer": float(summary["avg_cer"]) <= args.max_avg_cer,
        "max_cer": float(summary["max_cer"]) <= args.max_cer,
        "min_rms": float(summary["min_rms"]) >= args.min_rms,
        "max_ttfa_ms": float(summary["max_ttfa_ms"] or 0.0) <= args.max_ttfa_ms,
        "max_codec_ms": float(summary["max_codec_ms"] or 0.0) <= args.max_codec_ms,
    }
    return {
        "thresholds": {
            "max_avg_cer": args.max_avg_cer,
            "max_cer": args.max_cer,
            "min_rms": args.min_rms,
            "max_ttfa_ms": args.max_ttfa_ms,
            "max_codec_ms": args.max_codec_ms,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_all(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_stage_subprocess(args, "tts")
    run_stage_subprocess(args, "asr")

    summary = json.loads((args.out_dir / "summary.json").read_text(encoding="utf-8"))
    gates = evaluate_gates(args, summary)
    result = {
        **summary,
        "profile": {
            "backend": "moss_ort",
            "voice": args.voice,
            "seed": args.seed,
            "manifest": args.manifest or None,
            "threads": args.threads,
            "session_threads": {
                "prefill": args.prefill_threads or args.threads,
                "decode": args.decode_threads or args.threads,
                "sampler": args.sampler_threads or args.threads,
                "codec": args.codec_threads or args.threads,
            },
            "prefill_seq": args.prefill_seq,
            "codec_streaming": args.codec_streaming,
            "codec_batch_frames": args.codec_batch_frames,
        },
        "gates": gates,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if gates["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "tts", "asr"), default="all")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/moss_ort_roundtrip"))
    parser.add_argument("--sentences")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-threads", type=int)
    parser.add_argument("--decode-threads", type=int)
    parser.add_argument("--sampler-threads", type=int)
    parser.add_argument("--codec-threads", type=int)
    parser.add_argument("--max-new-frames", type=int, default=20)
    parser.add_argument("--voice", default="Lingyu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prefill-seq", type=int, default=0)
    parser.add_argument("--codec-streaming", type=int, default=1)
    parser.add_argument("--codec-batch-frames", type=int, default=1)
    parser.add_argument("--manifest", default="")
    parser.add_argument("--warmup-text", default="你好")
    parser.add_argument("--max-avg-cer", type=float, default=0.5)
    parser.add_argument("--max-cer", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.02)
    parser.add_argument("--max-ttfa-ms", type=float, default=1500.0)
    parser.add_argument("--max-codec-ms", type=float, default=120.0)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stage == "tts":
        run_tts_stage(args)
        return 0
    if args.stage == "asr":
        run_asr_stage(args)
        return 0
    return run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())

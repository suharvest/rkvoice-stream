#!/usr/bin/env python3
"""Run the MOSS production profile acceptance suite.

This verifier is intentionally a thin orchestrator over the focused checks:

1. ORT artifact manifest validation.
2. Service-level streaming latency gates.
3. Backend-stage timing gates.
4. Isolated TTS -> ASR roundtrip quality gates.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_ort_artifacts


PRODUCTION_CODEC_BATCH_FRAMES = 3


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str], *, repo_root: Path, env: dict[str, str] | None = None) -> None:
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = str(repo_root) + (os.pathsep + run_env["PYTHONPATH"] if run_env.get("PYTHONPATH") else "")
    if env:
        run_env.update(env)
    subprocess.run(cmd, cwd=repo_root, env=run_env, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _service_env(args: argparse.Namespace) -> list[str]:
    values = [
        "TTS_BACKEND=moss_ort",
        "ASR_BACKEND=disabled",
        f"MOSS_ORT_MODEL_DIR={args.model_dir}",
        f"MOSS_ORT_MANIFEST={args.manifest}",
        f"MOSS_ORT_THREADS={args.threads}",
        f"MOSS_ORT_PREFILL_SEQ={args.prefill_seq}",
        f"MOSS_ORT_MAX_NEW_FRAMES={args.service_max_new_frames}",
        "MOSS_ORT_CODEC_STREAMING=1",
        f"MOSS_ORT_CODEC_BATCH_FRAMES={args.codec_batch_frames}",
        "MOSS_ORT_LOAD_FULL_CODEC=0",
        "MOSS_ORT_CODEC_ASYNC=0",
        "MOSS_ORT_CACHE_VOICE_PREFIX=0",
        "MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK=0",
        f"MOSS_ORT_VOICE={args.voice}",
        f"MOSS_ORT_WARMUP_TEXT={args.warmup_text}",
        "MOSS_ORT_HYBRID_RKNN=0",
    ]
    if args.seed is not None:
        values.append(f"MOSS_ORT_SEED={args.seed}")
    for attr, env_name in (
        ("prefill_threads", "MOSS_ORT_PREFILL_THREADS"),
        ("decode_threads", "MOSS_ORT_DECODE_THREADS"),
        ("sampler_threads", "MOSS_ORT_SAMPLER_THREADS"),
        ("codec_threads", "MOSS_ORT_CODEC_THREADS"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            values.append(f"{env_name}={value}")
    return values


def _build_service_cmd(args: argparse.Namespace, service_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_script_dir() / "verify_moss_service_profile.py"),
        "--repo-root",
        str(args.repo_root),
        "--port",
        str(args.port),
        "--text",
        args.text,
        "--expected-backend",
        "moss_ort",
        "--expected-sample-rate",
        str(args.expected_sample_rate),
        "--min-payload-bytes",
        str(args.min_payload_bytes),
        "--max-tts-wall-ms",
        str(args.max_tts_wall_ms),
        "--max-dialogue-wall-ms",
        str(args.max_dialogue_wall_ms),
        "--max-tts-first-payload-ms",
        str(args.max_tts_first_payload_ms),
        "--max-dialogue-first-payload-ms",
        str(args.max_dialogue_first_payload_ms),
        "--max-dialogue-payload-gap-ms",
        str(args.max_dialogue_payload_gap_ms),
        "--require-manifest-validated",
        "--expected-voice",
        args.voice,
        "--expected-manifest",
        args.manifest,
        "--expected-codec-batch-frames",
        str(args.codec_batch_frames),
        "--require-production-runtime",
        "--min-dialogue-binary-chunks",
        str(args.min_dialogue_binary_chunks),
        "--startup-timeout",
        str(args.startup_timeout),
        "--request-timeout",
        str(args.request_timeout),
        "--json-out",
        str(service_json),
        "--log-file",
        str(args.log_file),
    ]
    if args.seed is not None:
        cmd.extend(["--expected-seed", str(args.seed)])
    for value in _service_env(args):
        cmd.extend(["--set-env", value])
    return cmd


def _build_roundtrip_cmd(args: argparse.Namespace, roundtrip_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_script_dir() / "verify_moss_ort_roundtrip.py"),
        "--model-dir",
        str(args.model_dir),
        "--asr-model-dir",
        str(args.asr_model_dir),
        "--out-dir",
        str(args.roundtrip_out_dir),
        "--json-out",
        str(roundtrip_json),
        "--threads",
        str(args.threads),
        "--prefill-threads",
        str(args.prefill_threads or args.threads),
        "--decode-threads",
        str(args.decode_threads or args.threads),
        "--sampler-threads",
        str(args.sampler_threads or args.threads),
        "--codec-threads",
        str(args.codec_threads or args.threads),
        "--max-new-frames",
        str(args.roundtrip_max_new_frames),
        "--voice",
        args.voice,
        "--prefill-seq",
        str(args.prefill_seq),
        "--codec-streaming",
        "1",
        "--codec-batch-frames",
        str(args.codec_batch_frames),
        "--manifest",
        args.manifest,
        "--warmup-text",
        args.warmup_text,
        "--max-avg-cer",
        str(args.max_avg_cer),
        "--max-cer",
        str(args.max_cer),
        "--min-rms",
        str(args.min_rms),
        "--max-ttfa-ms",
        str(args.max_roundtrip_ttfa_ms),
        "--max-codec-ms",
        str(args.max_codec_ms),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if args.sentences:
        cmd.extend(["--sentences", args.sentences])
    return cmd


def _build_backend_stage_cmd(args: argparse.Namespace, backend_stage_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_script_dir() / "smoke_moss_hybrid_backend.py"),
        "--text",
        args.text,
        "--max-new-frames",
        str(args.service_max_new_frames),
        "--seed",
        str(args.seed),
        "--min-audio-frames",
        str(args.min_backend_audio_frames),
        "--max-ttfa-ms",
        str(args.max_backend_ttfa_ms),
        "--max-prefill-ms",
        str(args.max_backend_prefill_ms),
        "--max-codec-ms",
        str(args.max_codec_ms),
        "--json-out",
        str(backend_stage_json),
    ]
    return cmd


def _summarize(
    *,
    artifact_manifest: dict[str, Any] | None,
    service_report: dict[str, Any] | None,
    backend_stage_report: dict[str, Any] | None,
    roundtrip_report: dict[str, Any] | None,
    errors: list[str],
) -> dict[str, Any]:
    artifact_ok = artifact_manifest is not None
    service_ok = bool((service_report or {}).get("gates", {}).get("passed"))
    backend_stage_ok = bool((backend_stage_report or {}).get("gates", {}).get("passed"))
    roundtrip_ok = bool((roundtrip_report or {}).get("gates", {}).get("passed"))
    return {
        "passed": artifact_ok and service_ok and backend_stage_ok and roundtrip_ok and not errors,
        "checks": {
            "artifact_manifest": artifact_ok,
            "service_streaming": service_ok,
            "backend_stage": backend_stage_ok,
            "roundtrip_quality": roundtrip_ok,
        },
        "errors": errors,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", default="moss-ort-manifest.json")
    parser.add_argument("--asr-model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/moss_production_profile"))
    parser.add_argument("--roundtrip-out-dir", type=Path, default=Path("/tmp/moss_production_profile_roundtrip"))
    parser.add_argument("--port", type=int, default=8624)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--sentences")
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--prefill-threads", type=int)
    parser.add_argument("--decode-threads", type=int)
    parser.add_argument("--sampler-threads", type=int)
    parser.add_argument("--codec-threads", type=int)
    parser.add_argument("--prefill-seq", type=int, default=0)
    parser.add_argument("--warmup-text", default="你好")
    parser.add_argument("--service-max-new-frames", type=int, default=20)
    parser.add_argument("--roundtrip-max-new-frames", type=int, default=20)
    parser.add_argument("--codec-batch-frames", type=int, default=PRODUCTION_CODEC_BATCH_FRAMES)
    parser.add_argument("--min-dialogue-binary-chunks", type=int, default=7)
    parser.add_argument("--expected-sample-rate", type=int, default=48000)
    parser.add_argument("--min-payload-bytes", type=int, default=30720)
    parser.add_argument("--max-tts-wall-ms", type=float, default=0.0)
    parser.add_argument("--max-dialogue-wall-ms", type=float, default=0.0)
    parser.add_argument("--max-tts-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-payload-gap-ms", type=float, default=1500.0)
    parser.add_argument("--max-avg-cer", type=float, default=0.5)
    parser.add_argument("--max-cer", type=float, default=1.0)
    parser.add_argument("--min-rms", type=float, default=0.02)
    parser.add_argument("--max-roundtrip-ttfa-ms", type=float, default=1500.0)
    parser.add_argument("--min-backend-audio-frames", type=int, default=20)
    parser.add_argument("--max-backend-ttfa-ms", type=float, default=1500.0)
    parser.add_argument("--max-backend-prefill-ms", type=float, default=1200.0)
    parser.add_argument("--max-codec-ms", type=float, default=170.0)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/moss_production_profile_uvicorn.log"))
    parser.add_argument("--json-out", type=Path)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    service_json = args.out_dir / "service_streaming.json"
    backend_stage_json = args.out_dir / "backend_stage.json"
    roundtrip_json = args.out_dir / "roundtrip_quality.json"
    errors: list[str] = []

    artifact_manifest: dict[str, Any] | None = None
    service_report: dict[str, Any] | None = None
    backend_stage_report: dict[str, Any] | None = None
    roundtrip_report: dict[str, Any] | None = None

    try:
        artifact_manifest = validate_moss_ort_artifacts(args.model_dir, args.manifest)
    except MossORTArtifactError as exc:
        errors.append(f"artifact manifest failed: {exc}")

    if artifact_manifest is not None:
        try:
            _run(_build_service_cmd(args, service_json), repo_root=args.repo_root)
            service_report = _load_json(service_json)
        except subprocess.CalledProcessError as exc:
            errors.append(f"service streaming verifier failed: exit={exc.returncode}")
            if service_json.exists():
                service_report = _load_json(service_json)

        try:
            _run(_build_backend_stage_cmd(args, backend_stage_json), repo_root=args.repo_root, env=dict(item.split("=", 1) for item in _service_env(args)))
            backend_stage_report = _load_json(backend_stage_json)
        except subprocess.CalledProcessError as exc:
            errors.append(f"backend stage verifier failed: exit={exc.returncode}")
            if backend_stage_json.exists():
                backend_stage_report = _load_json(backend_stage_json)

        try:
            _run(_build_roundtrip_cmd(args, roundtrip_json), repo_root=args.repo_root)
            roundtrip_report = _load_json(roundtrip_json)
        except subprocess.CalledProcessError as exc:
            errors.append(f"roundtrip quality verifier failed: exit={exc.returncode}")
            if roundtrip_json.exists():
                roundtrip_report = _load_json(roundtrip_json)

    report = {
        "profile": {
            "backend": "moss_ort",
            "model_dir": str(args.model_dir),
            "manifest": args.manifest,
            "voice": args.voice,
            "seed": args.seed,
            "threads": args.threads,
            "session_threads": {
                "prefill": args.prefill_threads or args.threads,
                "decode": args.decode_threads or args.threads,
                "sampler": args.sampler_threads or args.threads,
                "codec": args.codec_threads or args.threads,
            },
            "prefill_seq": args.prefill_seq,
            "codec_streaming": 1,
            "codec_batch_frames": args.codec_batch_frames,
        },
        "summary": _summarize(
            artifact_manifest=artifact_manifest,
            service_report=service_report,
            backend_stage_report=backend_stage_report,
            roundtrip_report=roundtrip_report,
            errors=errors,
        ),
        "artifact_manifest": {
            "target_platform": artifact_manifest.get("target_platform") if artifact_manifest else None,
            "sample_rate": artifact_manifest.get("sample_rate") if artifact_manifest else None,
            "channels": artifact_manifest.get("channels") if artifact_manifest else None,
            "required_artifacts": len([a for a in artifact_manifest.get("artifacts", []) if a.get("required", True)])
            if artifact_manifest
            else 0,
        },
        "service_streaming": service_report,
        "backend_stage": backend_stage_report,
        "roundtrip_quality": roundtrip_report,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_path = args.json_out or (args.out_dir / "production_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["summary"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

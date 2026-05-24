#!/usr/bin/env python3
"""Write a MOSS-TTS-Nano RKNN production manifest from a built artifact dir."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def default_required(target: str) -> list[str]:
    return [
    "tokenizer.model",
    "tts_browser_onnx_meta.json",
    "codec_browser_onnx_meta.json",
    f"moss_tts_prefill.s32.fp16.{target}.rknn",
    f"moss_tts_prefill.s64.fp16.{target}.rknn",
    f"moss_tts_prefill.s128.fp16.{target}.rknn",
    f"moss_tts_prefill.s256.fp16.{target}.rknn",
    f"moss_tts_decode_step.p1.fp16.{target}.rknn",
    f"moss_tts_decode_step.p32.fp16.{target}.rknn",
    f"moss_tts_decode_step.p64.fp16.{target}.rknn",
    f"moss_tts_decode_step.p128.fp16.{target}.rknn",
    f"moss_tts_decode_step.p256.fp16.{target}.rknn",
    f"moss_tts_decode_step.p512.fp16.{target}.rknn",
    f"moss_tts_local_fixed_sampled_frame.fp16.{target}.rknn",
    f"codec_decode_step.f1.fp16.{target}.rknn",
    f"codec_decode_step.f4.fp16.{target}.rknn",
    f"codec_decode_step.f8.fp16.{target}.rknn",
    ]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_entry(root: Path, rel: str, required: bool = True) -> dict:
    path = root / rel
    entry = {"path": rel, "required": required}
    if path.exists() and path.is_file():
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", default="moss-rknn-manifest.json")
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--artifact-set", default="")
    args = parser.parse_args()

    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": args.target,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "artifact_set": args.artifact_set,
        "artifacts": [artifact_entry(args.model_dir, rel) for rel in default_required(args.target)],
        "production_gates": {
            "max_ttfa_ms": 500,
            "max_rtf": 0.75,
            "max_asr_cer": 0.15,
            "min_non_silent_rms": 0.02,
        },
        "quality_status": {
            "production_default": False,
            "reason": "RKNN MOSS artifacts must pass service streaming, backend-stage, and ASR roundtrip gates on target hardware before promotion",
        },
    }
    out_path = args.model_dir / args.out
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

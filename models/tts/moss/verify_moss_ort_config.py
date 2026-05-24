#!/usr/bin/env python3
"""Validate the RK3576 MOSS ORT production config contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream import load_config
from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_ort_artifacts


EXPECTED = {
    "backend": "moss_ort",
    "require_backend": 1,
    "manifest": "moss-ort-manifest.json",
    "sample_rate": 48000,
    "channels": 2,
    "threads": 6,
    "prefill_threads": 8,
    "decode_threads": 5,
    "codec_threads": 5,
    "prefill_seq": 0,
    "max_new_frames": 20,
    "codec_streaming": 1,
    "codec_batch_frames": 3,
    "cache_voice_prefix": 0,
    "warmup_text": "你好",
    "voice": "Junhao",
    "seed": 314,
    "allow_deterministic_fallback": 0,
}

FORBIDDEN_EXPERIMENTAL_KEYS = {
    "hybrid_rknn",
    "hybrid_strict",
    "hybrid_seq_len",
    "hybrid_split",
    "hybrid_layers",
    "hybrid_dir",
    "hybrid_rknn_dir",
    "hybrid_manifest",
    "load_full_codec",
    "codec_async",
}


def _check_config(config: dict[str, Any], *, require_model_dir: bool = False) -> list[str]:
    errors: list[str] = []
    asr = config.get("asr") or {}
    tts = config.get("tts") or {}
    if asr.get("backend") not in {None, "disabled"}:
        errors.append(f"asr.backend must be null/disabled for TTS-only production profile, got {asr.get('backend')!r}")
    for key, expected in EXPECTED.items():
        if tts.get(key) != expected:
            errors.append(f"tts.{key}={tts.get(key)!r}, expected {expected!r}")
    for key in sorted(FORBIDDEN_EXPERIMENTAL_KEYS):
        if key in tts and tts.get(key) not in {None, 0, False, ""}:
            errors.append(f"tts.{key} is experimental and must not be enabled in the production ORT profile")
    model_dir = tts.get("model_dir")
    if not model_dir:
        errors.append("tts.model_dir is required")
    elif require_model_dir and not Path(str(model_dir)).exists():
        errors.append(f"tts.model_dir does not exist: {model_dir}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rk3576-moss-ort-stream.yaml"))
    parser.add_argument("--require-model-dir", action="store_true")
    parser.add_argument("--validate-artifacts", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = load_config(str(args.config))
    errors = _check_config(config, require_model_dir=args.require_model_dir)
    artifact_summary: dict[str, Any] | None = None
    if args.validate_artifacts and not errors:
        tts = config["tts"]
        try:
            manifest = validate_moss_ort_artifacts(tts["model_dir"], tts["manifest"])
            artifact_summary = {
                "target_platform": manifest.get("target_platform"),
                "sample_rate": manifest.get("sample_rate"),
                "channels": manifest.get("channels"),
                "required_artifacts": len([a for a in manifest.get("artifacts", []) if a.get("required", True)]),
            }
        except MossORTArtifactError as exc:
            errors.append(f"artifact validation failed: {exc}")

    report = {
        "config": str(args.config),
        "passed": not errors,
        "errors": errors,
        "expected": EXPECTED,
        "forbidden_experimental_keys": sorted(FORBIDDEN_EXPERIMENTAL_KEYS),
        "artifact_manifest": artifact_summary,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())

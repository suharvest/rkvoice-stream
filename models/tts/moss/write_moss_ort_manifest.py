#!/usr/bin/env python3
"""Write a MOSS-TTS-Nano ONNX Runtime production manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def default_required(require_streaming_codec: bool = True) -> list[str]:
    artifacts = [
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "codec_browser_onnx_meta.json",
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_shared.data",
    ]
    if require_streaming_codec:
        artifacts.append("moss_audio_tokenizer_decode_step.onnx")
    return artifacts


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
    parser.add_argument("--out", default="moss-ort-manifest.json")
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588", "generic"])
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--channels", type=int, default=2)
    parser.add_argument("--voice", default="Junhao")
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--require-streaming-codec", type=int, default=1)
    args = parser.parse_args()

    require_streaming = bool(args.require_streaming_codec)
    manifest = {
        "model_id": "moss-tts-nano-onnx",
        "target_platform": args.target,
        "sample_rate": args.sample_rate,
        "channels": args.channels,
        "streaming_required": require_streaming,
        "production_profile": {
            "backend": "moss_ort",
            "voice": args.voice,
            "seed": args.seed,
            "codec_streaming": int(require_streaming),
        },
        "artifacts": [artifact_entry(args.model_dir, rel) for rel in default_required(require_streaming)],
        "production_gates": {
            "max_tts_first_payload_ms": 1500,
            "max_dialogue_first_payload_ms": 1500,
            "max_tts_wall_ms": 2000,
            "max_dialogue_wall_ms": 2000,
            "max_avg_cer": 0.5,
            "max_cer": 1.0,
            "min_rms": 0.02,
        },
    }
    out_path = args.model_dir / args.out
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export lightweight MOSS embedding assets for RKLLM INPUT_EMBED runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


def export_assets(source_scaffold_dir: Path, folded_scaffold_dir: Path, output: Path) -> dict[str, Any]:
    source_scaffold_dir = source_scaffold_dir.resolve()
    folded_scaffold_dir = folded_scaffold_dir.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, np.ndarray] = {}
    state_path = source_scaffold_dir / "model.safetensors"
    with safe_open(state_path, framework="pt", device="cpu") as handle:
        tensors["embed_tokens"] = handle.get_tensor("model.embed_tokens.weight").numpy().astype(np.float32)
        audio = []
        idx = 0
        while f"model.audio_embeddings.{idx}.weight" in handle.keys():
            audio.append(handle.get_tensor(f"model.audio_embeddings.{idx}.weight").numpy().astype(np.float32))
            idx += 1
    tensors["audio_embeddings"] = np.stack(audio, axis=0).astype(np.float32)
    final_bias_path = folded_scaffold_dir / "moss_final_norm_bias.npy"
    if final_bias_path.exists():
        tensors["final_norm_bias"] = np.load(final_bias_path).astype(np.float32)
    else:
        with safe_open(state_path, framework="pt", device="cpu") as handle:
            tensors["final_norm_bias"] = handle.get_tensor("model.norm.bias").numpy().astype(np.float32)
    np.savez_compressed(output, **tensors)

    report: dict[str, Any] = {
        "source_scaffold_dir": str(source_scaffold_dir),
        "folded_scaffold_dir": str(folded_scaffold_dir),
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "embed_tokens_shape": list(tensors["embed_tokens"].shape),
        "audio_embeddings_shape": list(tensors["audio_embeddings"].shape),
        "final_norm_bias_shape": list(tensors["final_norm_bias"].shape),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scaffold-dir", required=True, type=Path)
    parser.add_argument("--folded-scaffold-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = export_assets(args.source_scaffold_dir, args.folded_scaffold_dir, args.output)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

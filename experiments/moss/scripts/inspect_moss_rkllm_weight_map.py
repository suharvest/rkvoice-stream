#!/usr/bin/env python3
"""Inventory MOSS ONNX weights for an official RKLLM custom-model bridge."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


def _load_initializers(path: Path) -> list[dict[str, Any]]:
    model = onnx.load(str(path), load_external_data=False)
    return [
        {
            "name": init.name,
            "shape": list(init.dims),
            "data_type": int(init.data_type),
            "external": any(entry.key == "location" for entry in init.external_data),
        }
        for init in model.graph.initializer
    ]


def _classify(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered or "wte" in lowered:
        return "embedding"
    if "ln_" in lowered or "layernorm" in lowered or "norm" in lowered:
        return "norm"
    if "c_attn" in lowered or "q_proj" in lowered or "k_proj" in lowered or "v_proj" in lowered:
        return "attention_qkv"
    if "c_proj" in lowered or "o_proj" in lowered:
        return "attention_out"
    if "fc_in" in lowered or "gate" in lowered or "up_proj" in lowered:
        return "mlp_in"
    if "fc_out" in lowered or "down_proj" in lowered:
        return "mlp_out"
    if "lm_head" in lowered:
        return "lm_head"
    return "other"


def inspect(model_dir: Path) -> dict[str, Any]:
    tts_prefill = model_dir / "moss_tts_prefill.onnx"
    tts_decode = model_dir / "moss_tts_decode_step.onnx"
    sampler = model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    models = {
        "prefill": tts_prefill,
        "decode_step": tts_decode,
        "sampler": sampler,
    }
    inventories: dict[str, Any] = {}
    for key, path in models.items():
        if not path.exists():
            inventories[key] = {"path": str(path), "exists": False}
            continue
        inits = _load_initializers(path)
        classes = Counter(_classify(item["name"]) for item in inits)
        shape_counts = Counter(tuple(item["shape"]) for item in inits)
        inventories[key] = {
            "path": str(path),
            "exists": True,
            "initializer_count": len(inits),
            "class_counts": dict(classes.most_common()),
            "common_shapes": [
                {"shape": list(shape), "count": count}
                for shape, count in shape_counts.most_common(20)
            ],
            "sample_initializers": inits[:80],
        }

    prefill_classes = inventories.get("prefill", {}).get("class_counts", {})
    has_transformer_weights = any(
        prefill_classes.get(name, 0) > 0
        for name in ("embedding", "attention_qkv", "attention_out", "mlp_in", "mlp_out", "norm")
    )
    return {
        "model_dir": str(model_dir.resolve()),
        "rkllm_bridge_feasibility": {
            "can_reconstruct_from_onnx_initializers": has_transformer_weights,
            "preferred_runtime_output": "RKLLM_INFER_GET_LAST_HIDDEN_LAYER",
            "sampler_should_remain_external_initially": True,
            "reason": (
                "MOSS sampler has multi-head text/audio codebook logic; preserve it outside RKLLM "
                "until global hidden parity is proven."
            ),
        },
        "inventories": inventories,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = inspect(args.model_dir)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

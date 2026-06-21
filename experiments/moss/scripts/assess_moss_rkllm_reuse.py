#!/usr/bin/env python3
"""Assess whether a MOSS ONNX bundle can be promoted through official RKLLM.

The RKLLM toolkit exports from HuggingFace/GGUF style model directories, not
from ONNX graphs. This verifier makes that boundary explicit so MOSS work can
prioritize official RKLLM before falling back to RKNN islands or rkmatmul.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUPPORTED_RKLLM_FAMILIES = (
    "llama",
    "tinyllama",
    "qwen2",
    "qwen2.5",
    "qwen3",
    "phi2",
    "phi3",
    "chatglm3",
    "gemma2",
    "gemma3",
    "internlm2",
    "minicpm3",
    "minicpm4",
    "telechat2",
    "qwen2-vl",
    "qwen3-vl",
    "minicpm-v-2_6",
    "deepseek-r1-distill",
    "janus-pro",
    "internvl2",
    "internvl3",
    "smolvlm",
    "rwkv7",
    "deepseekocr",
)

MOSS_REQUIRED_ONNX = (
    "moss_tts_prefill.onnx",
    "moss_tts_decode_step.onnx",
    "moss_tts_local_fixed_sampled_frame.onnx",
    "moss_tts_global_shared.data",
    "moss_tts_local_shared.data",
    "tts_browser_onnx_meta.json",
    "tokenizer.model",
)

HF_WEIGHT_PATTERNS = (
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "model.safetensors",
    "model-*.safetensors",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _glob_any(root: Path, patterns: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path.relative_to(root)) for path in sorted(root.glob(pattern)))
    return found


def _detect_hf_family(config: dict[str, Any] | None) -> str | None:
    if not config:
        return None
    haystack = " ".join(
        str(item).lower()
        for item in [
            config.get("model_type", ""),
            config.get("architectures", ""),
            config.get("_name_or_path", ""),
            config.get("auto_map", ""),
        ]
    )
    for family in SUPPORTED_RKLLM_FAMILIES:
        if family in haystack:
            return family
    return None


def _moss_shape_summary(meta: dict[str, Any] | None) -> dict[str, Any]:
    cfg = (meta or {}).get("model_config") or {}
    return {
        "row_width": cfg.get("row_width"),
        "hidden_size": cfg.get("hidden_size"),
        "global_layers": cfg.get("global_layers"),
        "global_heads": cfg.get("global_heads"),
        "local_layers": cfg.get("local_layers"),
        "text_vocab_size": cfg.get("vocab_size"),
        "audio_codebooks": len(cfg.get("audio_codebook_sizes") or []),
        "audio_codebook_sizes": cfg.get("audio_codebook_sizes"),
    }


def assess(model_dir: Path, rkllm_repo: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    missing = [name for name in MOSS_REQUIRED_ONNX if not (model_dir / name).exists()]
    meta_path = model_dir / "tts_browser_onnx_meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else None

    config_path = model_dir / "config.json"
    hf_config = _read_json(config_path) if config_path.exists() else None
    hf_weights = _glob_any(model_dir, HF_WEIGHT_PATTERNS)
    hf_family = _detect_hf_family(hf_config)

    rkllm_custom_demo = None
    if rkllm_repo is not None:
        custom_dir = rkllm_repo / "rkllm-toolkit" / "examples" / "custom_demo"
        rkllm_custom_demo = {
            "path": str(custom_dir),
            "available": (custom_dir / "configuration_custom.py").exists()
            and (custom_dir / "modeling_custom.py").exists()
            and (custom_dir / "config.json").exists(),
        }

    shape = _moss_shape_summary(meta)
    looks_like_moss = (
        shape.get("row_width") == 17
        and shape.get("hidden_size") == 768
        and shape.get("global_layers") == 12
        and shape.get("audio_codebooks") == 16
    )

    direct_builtin_ready = bool(hf_family and hf_weights)
    custom_bridge_candidate = bool(looks_like_moss and not missing)
    needs_weight_reconstruction = custom_bridge_candidate and not hf_weights

    blockers: list[str] = []
    if missing:
        blockers.append("missing required MOSS ONNX bundle files")
    if not hf_config:
        blockers.append("no HuggingFace config.json for RKLLM load_huggingface")
    if not hf_weights:
        blockers.append("no HuggingFace PyTorch/safetensors weights for RKLLM load_huggingface")
    if not hf_family:
        blockers.append("not an RKLLM built-in supported family as-is")
    if shape.get("row_width") == 17:
        blockers.append("MOSS uses 1 text column + 16 audio codebooks, not a single-token CausalLM vocabulary")

    verdict = "blocked"
    if direct_builtin_ready:
        verdict = "direct_builtin_export_ready"
    elif custom_bridge_candidate:
        verdict = "custom_hf_bridge_candidate"

    return {
        "model_dir": str(model_dir),
        "verdict": verdict,
        "official_rkllm_first": True,
        "direct_builtin_rkllm": {
            "ready": direct_builtin_ready,
            "detected_family": hf_family,
            "supported_families": list(SUPPORTED_RKLLM_FAMILIES),
            "hf_config": str(config_path) if hf_config else None,
            "hf_weights": hf_weights,
        },
        "custom_hf_bridge": {
            "candidate": custom_bridge_candidate,
            "reuse": "rkllm-toolkit/examples/custom_demo",
            "rkllm_custom_demo": rkllm_custom_demo,
            "needs_onnx_weight_reconstruction": needs_weight_reconstruction,
            "runtime_api_to_reuse": [
                "RKLLM_INPUT_EMBED",
                "RKLLM_INFER_GET_LAST_HIDDEN_LAYER",
                "RKLLM_INFER_GET_LOGITS",
                "base_domain_id=1",
                "embed_flash=1",
            ],
        },
        "moss_shape": shape,
        "required_files": {
            "missing": missing,
            "present": [name for name in MOSS_REQUIRED_ONNX if (model_dir / name).exists()],
        },
        "blockers": blockers,
        "recommended_path": [
            "build a MOSS HuggingFace custom model directory from official RKLLM custom_demo",
            "map ONNX initializers/external data into that custom model state_dict if original checkpoint is unavailable",
            "export the global transformer to RKLLM and call it with RKLLM_INPUT_EMBED or token input",
            "use GET_LAST_HIDDEN_LAYER to preserve the existing MOSS fixed sampler first",
            "only after hidden parity passes, consider folding sampler logits into RKLLM GET_LOGITS or RKNN",
            "keep codec streaming as a separate RKNN/ORT stage until codec RKNN MatMul surgery passes real-device probes",
        ],
        "fallback_order": [
            "official RKLLM custom bridge",
            "RKNN islands for sampler/codec",
            "rkmatmul only for remaining MatMul-heavy projections if RKLLM export is blocked",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rkllm-repo", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = assess(args.model_dir, args.rkllm_repo)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

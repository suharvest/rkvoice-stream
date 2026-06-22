#!/usr/bin/env python3
"""Create a HuggingFace custom-model scaffold for exporting MOSS through RKLLM."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.tts.moss.assess_moss_rkllm_reuse import assess


TEMPLATE_FILES = (
    "configuration_custom.py",
    "modeling_custom.py",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_intermediate_size(model_dir: Path, hidden_size: int) -> int:
    try:
        import onnx
    except ImportError:
        return hidden_size * 4
    prefill = model_dir / "moss_tts_prefill.onnx"
    if not prefill.exists():
        return hidden_size * 4
    try:
        model = onnx.load(str(prefill), load_external_data=False)
    except Exception:
        return hidden_size * 4
    for init in model.graph.initializer:
        if ".mlp.fc_in.bias" in init.name and len(init.dims) == 1:
            return int(init.dims[0])
    return hidden_size * 4


def _matmul_weight_sources(model_dir: Path, hidden_size: int, intermediate_size: int, num_layers: int) -> list[dict[str, Any]]:
    try:
        import onnx
    except ImportError:
        return []
    prefill = model_dir / "moss_tts_prefill.onnx"
    if not prefill.exists():
        return []
    try:
        model = onnx.load(str(prefill), load_external_data=False)
    except Exception:
        return []
    expected_cycle = [
        ("attn.c_attn.weight", [hidden_size, hidden_size * 3]),
        ("attn.c_proj.weight", [hidden_size, hidden_size]),
        ("mlp.fc_in.weight", [hidden_size, intermediate_size]),
        ("mlp.fc_out.weight", [intermediate_size, hidden_size]),
    ]
    candidates = [
        {"name": init.name, "shape": list(init.dims)}
        for init in model.graph.initializer
        if init.name.startswith("onnx::MatMul_")
    ]
    mapped: list[dict[str, Any]] = []
    idx = 0
    for layer in range(num_layers):
        for logical, expected_shape in expected_cycle:
            source = candidates[idx] if idx < len(candidates) else None
            mapped.append(
                {
                    "layer": layer,
                    "logical": logical,
                    "source": source["name"] if source else None,
                    "source_shape": source["shape"] if source else None,
                    "expected_shape": expected_shape,
                    "shape_matches": bool(source and source["shape"] == expected_shape),
                }
            )
            idx += 1
    return mapped


def _build_config(model_dir: Path) -> dict[str, Any]:
    meta = _read_json(model_dir / "tts_browser_onnx_meta.json")
    cfg = meta["model_config"]
    hidden_size = int(cfg["hidden_size"])
    intermediate_size = _infer_intermediate_size(model_dir, hidden_size)
    audio_sizes = [int(v) for v in cfg.get("audio_codebook_sizes") or []]
    return {
        "architectures": ["CustomForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_custom.CustomConfig",
            "AutoModel": "modeling_custom.CustomModel",
            "AutoModelForCausalLM": "modeling_custom.CustomForCausalLM",
        },
        "model_type": "moss_rkllm_custom",
        "torch_dtype": "float16",
        "transformers_version": "4.36.0",
        "use_cache": True,
        "hidden_act": "gelu_new",
        "hidden_act_param": 0.0,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": int(cfg["global_heads"]),
        "num_key_value_heads": int(cfg["global_heads"]),
        "num_hidden_layers": int(cfg["global_layers"]),
        "max_position_embeddings": 32768,
        "position_embedding_type": "rope",
        "rope_base": 10000.0,
        "rms_norm_eps": 1e-5,
        "vocab_size": int(cfg["vocab_size"]),
        "bos_token_id": int(cfg["im_start_token_id"]),
        "eos_token_id": int(cfg["im_end_token_id"]),
        "pad_token_id": int(cfg["pad_token_id"]),
        "tie_word_embeddings": False,
        "scale_emb": 1,
        "dim_model_base": hidden_size,
        "scale_depth": 1,
        "moss_rkllm": {
            "source": "MOSS ONNX bundle",
            "row_width": int(cfg["row_width"]),
            "text_column": 0,
            "audio_columns": list(range(1, int(cfg["row_width"]))),
            "audio_codebook_sizes": audio_sizes,
            "audio_pad_token_id": int(cfg["audio_pad_token_id"]),
            "audio_assistant_slot_token_id": int(cfg["audio_assistant_slot_token_id"]),
            "preferred_runtime_input": "RKLLM_INPUT_EMBED",
            "preferred_runtime_output": "RKLLM_INFER_GET_LAST_HIDDEN_LAYER",
            "sampler_external_until_hidden_parity": True,
        },
    }


def _source_for(
    matmul_sources: dict[tuple[int, str], dict[str, Any]],
    layer: int,
    logical: str,
    conceptual: str,
) -> str:
    item = matmul_sources.get((layer, logical))
    if item and item.get("source"):
        return str(item["source"])
    return conceptual


def _layer_weight_map(num_layers: int, matmul_source_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    matmul_sources = {
        (int(row["layer"]), str(row["logical"])): row
        for row in (matmul_source_rows or [])
    }
    rows: list[dict[str, Any]] = [
        {
            "source": "core.model.transformer.wte.weight",
            "target": "model.embed_tokens.weight",
            "transform": "copy",
        }
    ]
    for idx in range(16):
        rows.append(
            {
                "source": f"core.model.audio_embeddings.{idx}.weight",
                "target": f"model.audio_embeddings.{idx}.weight",
                "transform": "copy; used by external MOSS row-to-embedding adapter",
            }
        )
    for layer in range(num_layers):
        prefix = f"core.model.transformer.h.{layer}"
        target = f"model.layers.{layer}"
        rows.extend(
            [
                {
                    "source": f"{prefix}.ln_1.weight",
                    "target": f"{target}.input_layernorm.weight",
                    "transform": "copy if present; otherwise verify tied/default behavior",
                },
                {
                    "source": f"{prefix}.ln_1.bias",
                    "target": f"{target}.input_layernorm.bias",
                    "transform": "copy; requires MOSS-specific LayerNorm/RMSNorm bridge",
                },
                {
                    "source": _source_for(
                        matmul_sources,
                        layer,
                        "attn.c_attn.weight",
                        f"{prefix}.attn.c_attn.weight",
                    ),
                    "conceptual_source": f"{prefix}.attn.c_attn.weight",
                    "target": [
                        f"{target}.self_attn.q_proj.weight",
                        f"{target}.self_attn.k_proj.weight",
                        f"{target}.self_attn.v_proj.weight",
                    ],
                    "transform": "split fused [hidden, 3*hidden] MatMul weight on output dimension",
                },
                {
                    "source": f"{prefix}.attn.c_attn.bias",
                    "target": [
                        f"{target}.self_attn.q_proj.bias",
                        f"{target}.self_attn.k_proj.bias",
                        f"{target}.self_attn.v_proj.bias",
                    ],
                    "transform": "split fused [3*hidden] bias on output dimension",
                },
                {
                    "source": _source_for(
                        matmul_sources,
                        layer,
                        "attn.c_proj.weight",
                        f"{prefix}.attn.c_proj.weight",
                    ),
                    "conceptual_source": f"{prefix}.attn.c_proj.weight",
                    "target": f"{target}.self_attn.o_proj.weight",
                    "transform": "copy or transpose according to reconstructed module convention",
                },
                {
                    "source": f"{prefix}.attn.c_proj.bias",
                    "target": f"{target}.self_attn.o_proj.bias",
                    "transform": "copy; requires bias-capable attention projection",
                },
                {
                    "source": f"{prefix}.ln_2.bias",
                    "target": f"{target}.post_attention_layernorm.bias",
                    "transform": "copy; requires MOSS-specific LayerNorm/RMSNorm bridge",
                },
                {
                    "source": _source_for(
                        matmul_sources,
                        layer,
                        "mlp.fc_in.weight",
                        f"{prefix}.mlp.fc_in.weight",
                    ),
                    "conceptual_source": f"{prefix}.mlp.fc_in.weight",
                    "target": f"{target}.mlp.up_proj.weight",
                    "transform": "copy or transpose according to reconstructed module convention",
                },
                {
                    "source": f"{prefix}.mlp.fc_in.bias",
                    "target": f"{target}.mlp.up_proj.bias",
                    "transform": "copy; requires bias-capable MLP",
                },
                {
                    "source": _source_for(
                        matmul_sources,
                        layer,
                        "mlp.fc_out.weight",
                        f"{prefix}.mlp.fc_out.weight",
                    ),
                    "conceptual_source": f"{prefix}.mlp.fc_out.weight",
                    "target": f"{target}.mlp.down_proj.weight",
                    "transform": "copy or transpose according to reconstructed module convention",
                },
                {
                    "source": f"{prefix}.mlp.fc_out.bias",
                    "target": f"{target}.mlp.down_proj.bias",
                    "transform": "copy; requires bias-capable MLP",
                },
            ]
        )
    rows.append(
        {
            "source": "core.model.transformer.ln_f.bias",
            "target": "model.norm.bias",
            "transform": "copy; requires MOSS-specific final norm bridge",
        }
    )
    return rows


def prepare(model_dir: Path, rkllm_custom_demo: Path, out_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    rkllm_custom_demo = rkllm_custom_demo.resolve()
    out_dir = out_dir.resolve()
    assessment = assess(model_dir, rkllm_custom_demo.parents[2] if rkllm_custom_demo.name == "custom_demo" else None)
    if assessment["verdict"] not in {"custom_hf_bridge_candidate", "direct_builtin_export_ready"}:
        raise RuntimeError(f"MOSS RKLLM custom bridge is not ready: {assessment['blockers']}")
    config = _build_config(model_dir)
    num_layers = int(config["num_hidden_layers"])
    matmul_sources = _matmul_weight_sources(
        model_dir,
        int(config["hidden_size"]),
        int(config["intermediate_size"]),
        num_layers,
    )
    mapping = {
        "format_version": 1,
        "source_model_dir": str(model_dir),
        "target_model_dir": str(out_dir),
        "status": "scaffold_only",
        "ready_for_rkllm_export": False,
        "why_not_ready": [
            "model.safetensors/state_dict conversion has not been written yet",
            "modeling_custom.py must be replaced or patched for MOSS bias, GELU, fused c_attn, and 17-column row embeddings",
            "global hidden parity against ONNX prefill/decode has not been proven",
        ],
        "anonymous_onnx_weight_sources": matmul_sources,
        "anonymous_onnx_weight_sources_complete": len(matmul_sources) == num_layers * 4
        and all(row.get("shape_matches") for row in matmul_sources),
        "weight_map": _layer_weight_map(num_layers, matmul_sources),
    }
    report = {
        "out_dir": str(out_dir),
        "dry_run": dry_run,
        "config": config,
        "mapping": mapping,
        "template_files": {
            name: {
                "source": str(rkllm_custom_demo / name),
                "exists": (rkllm_custom_demo / name).exists(),
                "target": str(out_dir / name),
            }
            for name in TEMPLATE_FILES
        },
        "next_commands": {
            "rkllm_export_after_state_dict": (
                "python export_rkllm.py --path "
                f"{out_dir} --target-platform rk3576 --num_npu_core 2 --quantized_dtype w4a16"
            )
        },
    }
    if dry_run:
        return report

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_FILES:
        source = rkllm_custom_demo / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    tokenizer = model_dir / "tokenizer.model"
    if tokenizer.exists():
        shutil.copy2(tokenizer, out_dir / "tokenizer.model")
    (out_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "moss_rkllm_weight_map.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# MOSS RKLLM Custom Scaffold\n\n"
        "This directory is a scaffold for the official RKLLM custom-model path. "
        "It is not export-ready until `model.safetensors` is generated and hidden parity passes.\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rkllm-custom-demo", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = prepare(args.model_dir, args.rkllm_custom_demo, args.out_dir, dry_run=args.dry_run)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

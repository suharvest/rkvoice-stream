#!/usr/bin/env python3
"""Convert MOSS ONNX initializers into a HuggingFace state_dict scaffold."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from prepare_moss_rkllm_custom_model import prepare


def _load_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    try:
        model = onnx.load(str(onnx_path), load_external_data=True)
    except Exception as exc:
        if "hard links" not in str(exc):
            raise
        model = _load_with_external_data_copy(onnx_path)
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def _external_locations(onnx_path: Path) -> set[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: set[str] = set()
    for init in model.graph.initializer:
        for entry in init.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    return locations


def _load_with_external_data_copy(onnx_path: Path) -> onnx.ModelProto:
    with tempfile.TemporaryDirectory(prefix="moss_onnx_external_") as tmp:
        tmp_dir = Path(tmp)
        copied_onnx = tmp_dir / onnx_path.name
        shutil.copyfile(onnx_path, copied_onnx)
        for location in _external_locations(onnx_path):
            source = onnx_path.parent / location
            target = tmp_dir / location
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return onnx.load(str(copied_onnx), load_external_data=True)


def _tensor(arr: np.ndarray, *, transpose_linear: bool = False):
    import torch

    value = np.asarray(arr)
    if transpose_linear:
        value = value.T
    if value.dtype == np.float64:
        value = value.astype(np.float32)
    return torch.from_numpy(np.array(value, copy=True, order="C"))


def _split_qkv(arr: np.ndarray, hidden_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if arr.shape[-1] != hidden_size * 3:
        raise ValueError(f"expected fused qkv last dim {hidden_size * 3}, got {arr.shape}")
    return tuple(np.split(arr, 3, axis=-1))  # type: ignore[return-value]


def convert_state(model_dir: Path, scaffold_dir: Path, *, write: bool = True) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    scaffold_dir = scaffold_dir.resolve()
    config_path = scaffold_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing scaffold config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])

    mapping_path = scaffold_dir / "moss_rkllm_weight_map.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"missing weight map: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_rows = {
        (int(row["layer"]), str(row["logical"])): row
        for row in mapping.get("anonymous_onnx_weight_sources", [])
    }
    if not mapping.get("anonymous_onnx_weight_sources_complete"):
        raise RuntimeError("anonymous ONNX MatMul sources are incomplete; regenerate scaffold first")

    initializers = _load_initializers(model_dir / "moss_tts_prefill.onnx")
    state: dict[str, Any] = {}
    report: dict[str, Any] = {
        "model_dir": str(model_dir),
        "scaffold_dir": str(scaffold_dir),
        "state_path": str(scaffold_dir / "model.safetensors"),
        "write": write,
        "converted": [],
        "synthetic": [],
        "missing": [],
        "ready_for_modeling_custom_patch": True,
        "ready_for_rkllm_export": False,
        "why_not_ready": [
            "modeling_custom.py still needs the MOSS 17-column embedding adapter, LayerNorm+bias, GELU MLP, and fused-QKV-compatible shapes",
            "global hidden parity against ONNX prefill/decode has not been proven",
        ],
    }

    def add(source: str, target: str, *, transpose_linear: bool = False) -> None:
        arr = initializers.get(source)
        if arr is None:
            report["missing"].append({"source": source, "target": target})
            return
        state[target] = _tensor(arr, transpose_linear=transpose_linear)
        report["converted"].append(
            {
                "source": source,
                "target": target,
                "source_shape": list(arr.shape),
                "target_shape": list(state[target].shape),
                "transpose_linear": transpose_linear,
            }
        )

    def add_synthetic_ones(target: str, shape: list[int], reason: str) -> None:
        import torch

        state[target] = torch.ones(*shape, dtype=torch.float32)
        report["synthetic"].append({"target": target, "shape": shape, "reason": reason})

    add("core.model.transformer.wte.weight", "model.embed_tokens.weight")
    for idx in range(16):
        add(f"core.model.audio_embeddings.{idx}.weight", f"model.audio_embeddings.{idx}.weight")

    for layer in range(layers):
        prefix = f"core.model.transformer.h.{layer}"
        target = f"model.layers.{layer}"
        ln1_weight = f"{prefix}.ln_1.weight"
        if ln1_weight in initializers:
            add(ln1_weight, f"{target}.input_layernorm.weight")
        else:
            add_synthetic_ones(
                f"{target}.input_layernorm.weight",
                [hidden],
                "ONNX omits identity LayerNorm gamma; one is mathematically equivalent",
            )
        add(f"{prefix}.ln_1.bias", f"{target}.input_layernorm.bias")

        qkv_w = initializers[source_rows[(layer, "attn.c_attn.weight")]["source"]]
        q_w, k_w, v_w = _split_qkv(qkv_w, hidden)
        for name, arr in (("q_proj", q_w), ("k_proj", k_w), ("v_proj", v_w)):
            state[f"{target}.self_attn.{name}.weight"] = _tensor(arr, transpose_linear=True)
            report["converted"].append(
                {
                    "source": source_rows[(layer, "attn.c_attn.weight")]["source"],
                    "target": f"{target}.self_attn.{name}.weight",
                    "source_shape": list(arr.shape),
                    "target_shape": list(state[f"{target}.self_attn.{name}.weight"].shape),
                    "split": name,
                    "transpose_linear": True,
                }
            )
        qkv_b = initializers.get(f"{prefix}.attn.c_attn.bias")
        if qkv_b is None:
            report["missing"].append({"source": f"{prefix}.attn.c_attn.bias", "target": f"{target}.self_attn.*.bias"})
        else:
            for name, arr in zip(("q_proj", "k_proj", "v_proj"), _split_qkv(qkv_b, hidden), strict=True):
                state[f"{target}.self_attn.{name}.bias"] = _tensor(arr)
                report["converted"].append(
                    {
                        "source": f"{prefix}.attn.c_attn.bias",
                        "target": f"{target}.self_attn.{name}.bias",
                        "source_shape": list(arr.shape),
                        "target_shape": list(state[f"{target}.self_attn.{name}.bias"].shape),
                        "split": name,
                    }
                )

        add(source_rows[(layer, "attn.c_proj.weight")]["source"], f"{target}.self_attn.o_proj.weight", transpose_linear=True)
        add(f"{prefix}.attn.c_proj.bias", f"{target}.self_attn.o_proj.bias")
        ln2_weight = f"{prefix}.ln_2.weight"
        if ln2_weight in initializers:
            add(ln2_weight, f"{target}.post_attention_layernorm.weight")
        else:
            add_synthetic_ones(
                f"{target}.post_attention_layernorm.weight",
                [hidden],
                "ONNX omits identity LayerNorm gamma; one is mathematically equivalent",
            )
        add(f"{prefix}.ln_2.bias", f"{target}.post_attention_layernorm.bias")
        add(source_rows[(layer, "mlp.fc_in.weight")]["source"], f"{target}.mlp.fc_in.weight", transpose_linear=True)
        add(f"{prefix}.mlp.fc_in.bias", f"{target}.mlp.fc_in.bias")
        add(source_rows[(layer, "mlp.fc_out.weight")]["source"], f"{target}.mlp.fc_out.weight", transpose_linear=True)
        add(f"{prefix}.mlp.fc_out.bias", f"{target}.mlp.fc_out.bias")

    ln_f_weight = "core.model.transformer.ln_f.weight"
    if ln_f_weight in initializers:
        add(ln_f_weight, "model.norm.weight")
    else:
        add_synthetic_ones("model.norm.weight", [hidden], "ONNX omits identity final LayerNorm gamma")
    add("core.model.transformer.ln_f.bias", "model.norm.bias")

    report["tensor_count"] = len(state)
    report["missing_count"] = len(report["missing"])
    report["synthetic_count"] = len(report["synthetic"])
    report["converted_count"] = len(report["converted"])
    if report["missing"]:
        report["ready_for_modeling_custom_patch"] = False
    if write:
        from safetensors.torch import save_file

        save_file(state, str(scaffold_dir / "model.safetensors"))
        (scaffold_dir / "moss_rkllm_state_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument("--rkllm-custom-demo", type=Path)
    parser.add_argument("--prepare-scaffold", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.prepare_scaffold:
        if args.rkllm_custom_demo is None:
            raise SystemExit("--prepare-scaffold requires --rkllm-custom-demo")
        prepare(args.model_dir, args.rkllm_custom_demo, args.scaffold_dir)
    report = convert_state(args.model_dir, args.scaffold_dir, write=not args.no_write)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create an RKLLM embed-input scaffold by stripping unsupported tensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _patch_modeling(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)",
        "nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True, bias=False)",
    )
    text = text.replace(
        '''\
        audio_sizes = config.moss_rkllm.get("audio_codebook_sizes") or [1024] * 16
        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(int(size), config.hidden_size) for size in audio_sizes]
        )
''',
        "",
    )
    start = text.index("    def _embed_rows(self, input_ids: torch.Tensor) -> torch.Tensor:\n")
    end = text.index("\n    def forward(\n", start)
    replacement = '''\
    def _embed_rows(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(f"RKLLM embed-only scaffold expects rank-2 token ids, got {tuple(input_ids.shape)}")
        return self.embed_tokens(input_ids)
'''
    text = text[:start] + replacement + text[end:]
    path.write_text(text, encoding="utf-8")


def _fold_layernorm_biases(tensors: dict[str, torch.Tensor], layers: int) -> dict[str, Any]:
    folded: list[str] = []
    for layer in range(layers):
        ln1_bias_key = f"model.layers.{layer}.input_layernorm.bias"
        if ln1_bias_key in tensors:
            beta = tensors[ln1_bias_key].to(dtype=torch.float32)
            for proj in ("q_proj", "k_proj", "v_proj"):
                weight_key = f"model.layers.{layer}.self_attn.{proj}.weight"
                bias_key = f"model.layers.{layer}.self_attn.{proj}.bias"
                tensors[bias_key] = tensors[bias_key].to(dtype=torch.float32) + torch.mv(
                    tensors[weight_key].to(dtype=torch.float32),
                    beta,
                )
                folded.append(f"{ln1_bias_key}->{bias_key}")

        ln2_bias_key = f"model.layers.{layer}.post_attention_layernorm.bias"
        if ln2_bias_key in tensors:
            beta = tensors[ln2_bias_key].to(dtype=torch.float32)
            weight_key = f"model.layers.{layer}.mlp.fc_in.weight"
            bias_key = f"model.layers.{layer}.mlp.fc_in.bias"
            tensors[bias_key] = tensors[bias_key].to(dtype=torch.float32) + torch.mv(
                tensors[weight_key].to(dtype=torch.float32),
                beta,
            )
            folded.append(f"{ln2_bias_key}->{bias_key}")
    return {"folded_count": len(folded), "folded": folded}


def _interleaved_to_half_indices(head_dim: int) -> torch.Tensor:
    return torch.tensor([*range(0, head_dim, 2), *range(1, head_dim, 2)], dtype=torch.long)


def _permute_qk_for_half_rope(tensors: dict[str, torch.Tensor], layers: int, heads: int, head_dim: int) -> dict[str, Any]:
    permuted: list[str] = []
    idx = _interleaved_to_half_indices(head_dim)
    for layer in range(layers):
        for proj in ("q_proj", "k_proj"):
            weight_key = f"model.layers.{layer}.self_attn.{proj}.weight"
            bias_key = f"model.layers.{layer}.self_attn.{proj}.bias"
            weight = tensors[weight_key].reshape(heads, head_dim, -1)
            tensors[weight_key] = weight[:, idx, :].reshape_as(tensors[weight_key]).contiguous()
            bias = tensors[bias_key].reshape(heads, head_dim)
            tensors[bias_key] = bias[:, idx].reshape_as(tensors[bias_key]).contiguous()
            permuted.extend([weight_key, bias_key])
    return {"permuted_count": len(permuted), "permuted": permuted}


def prepare(
    source_dir: Path,
    out_dir: Path,
    *,
    fold_norm_bias: bool = True,
    rkllm_half_rope: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(source_dir, out_dir, ignore=shutil.ignore_patterns("__pycache__"))

    config_path = out_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.setdefault("moss_rkllm", {})["rkllm_input_mode"] = "embed_only"
    config["moss_rkllm"]["norm_bias_mode"] = "folded_and_external_final" if fold_norm_bias else "dropped"
    config["moss_rkllm"]["rkllm_rope_layout"] = "half" if rkllm_half_rope else "interleaved"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _patch_modeling(out_dir / "modeling_custom.py")

    source_state = source_dir / "model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    dropped_norm_bias: list[str] = []
    final_norm_bias: torch.Tensor | None = None
    with safe_open(source_state, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("model.audio_embeddings."):
                dropped.append(key)
                continue
            tensor = handle.get_tensor(key)
            if key == "model.norm.bias":
                final_norm_bias = tensor.to(dtype=torch.float32)
                dropped_norm_bias.append(key)
                continue
            if key == "model.norm.bias" or key.endswith(".input_layernorm.bias") or key.endswith(".post_attention_layernorm.bias"):
                dropped_norm_bias.append(key)
                tensors[key] = tensor.to(dtype=torch.float32)
                continue
            tensors[key] = tensor
    fold_report = _fold_layernorm_biases(tensors, int(config["num_hidden_layers"])) if fold_norm_bias else {
        "folded_count": 0,
        "folded": [],
    }
    rope_report = _permute_qk_for_half_rope(
        tensors,
        int(config["num_hidden_layers"]),
        int(config["num_attention_heads"]),
        int(config["hidden_size"]) // int(config["num_attention_heads"]),
    ) if rkllm_half_rope else {"permuted_count": 0, "permuted": []}
    for key in list(tensors):
        if key.endswith(".input_layernorm.bias") or key.endswith(".post_attention_layernorm.bias"):
            del tensors[key]
    if "lm_head.weight" not in tensors:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].clone()
    save_file(tensors, out_dir / "model.safetensors")
    if final_norm_bias is not None:
        npy_path = out_dir / "moss_final_norm_bias.npy"
        import numpy as np

        np.save(npy_path, final_norm_bias.detach().cpu().numpy().astype(np.float32))

    report = {
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "tensor_count": len(tensors),
        "dropped_audio_embedding_count": len(dropped),
        "dropped_norm_bias_count": len(dropped_norm_bias),
        "fold_norm_bias": fold_norm_bias,
        "fold_report": fold_report,
        "rkllm_half_rope": rkllm_half_rope,
        "rope_report": rope_report,
        "external_final_norm_bias": str(out_dir / "moss_final_norm_bias.npy") if final_norm_bias is not None else None,
        "added_lm_head_weight": "lm_head.weight" in tensors,
        "rkllm_input_mode": "embed_only",
        "norm_bias_mode": "folded_and_external_final" if fold_norm_bias else "dropped_for_rkllm_runtime_tensor_table",
    }
    (out_dir / "moss_rkllm_embed_only_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--no-fold-norm-bias", action="store_true")
    parser.add_argument("--rkllm-half-rope", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = prepare(
        args.source_dir,
        args.out_dir,
        fold_norm_bias=not args.no_fold_norm_bias,
        rkllm_half_rope=args.rkllm_half_rope,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

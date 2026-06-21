#!/usr/bin/env python3
"""Compare MOSS HF semantic variants against dumped RKLLM hidden tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM


class _RMSNorm(nn.Module):
    def __init__(self, source: nn.LayerNorm):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.variance_epsilon = float(source.eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(hidden_states.device, hidden_states.dtype) * hidden_states.to(input_dtype)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = actual.astype(np.float32, copy=False)
    expected = expected.astype(np.float32, copy=False)
    diff = actual - expected
    denom = float(np.linalg.norm(expected.reshape(-1)) + 1e-12)
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": float(
            np.dot(actual.reshape(-1), expected.reshape(-1))
            / ((np.linalg.norm(actual.reshape(-1)) + 1e-12) * denom)
        ),
        "finite": bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
    }


def _replace_layernorms_with_rms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, _RMSNorm(child))
        else:
            _replace_layernorms_with_rms(child)


def _zero_layernorm_biases(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.LayerNorm) and child.bias is not None:
            child.bias.data.zero_()


def _rows_to_embeddings(input_ids: np.ndarray, assets_path: Path) -> np.ndarray:
    assets = np.load(assets_path)
    text_key = "text_embeddings" if "text_embeddings" in assets.files else "embed_tokens"
    text_embeddings = assets[text_key].astype(np.float32)
    audio_embeddings = assets["audio_embeddings"].astype(np.float32)
    audio_pad = 1024
    hidden = text_embeddings[input_ids[..., 0]]
    for index in range(audio_embeddings.shape[0]):
        codes = input_ids[..., index + 1]
        mask = codes != audio_pad
        safe_codes = np.clip(codes, 0, audio_embeddings.shape[1] - 1)
        hidden = hidden + audio_embeddings[index, safe_codes] * mask[..., None].astype(np.float32)
    return hidden.astype(np.float32, copy=False)


def _run_prefill(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    assets_path: Path | None,
    final_norm_bias: np.ndarray | None,
) -> np.ndarray:
    kwargs: dict[str, Any]
    if input_ids.ndim == 3:
        if assets_path is None:
            raise ValueError("rank-3 MOSS rows require --assets for embed-only scaffold comparison")
        kwargs = {"inputs_embeds": torch.from_numpy(_rows_to_embeddings(input_ids, assets_path)).float()}
    else:
        kwargs = {"input_ids": torch.from_numpy(input_ids).long()}
    with torch.no_grad():
        outputs = model.model(
            attention_mask=torch.from_numpy(attention_mask).long(),
            use_cache=True,
            return_dict=True,
            **kwargs,
        )
    hidden = outputs.last_hidden_state.detach().cpu().numpy().astype(np.float32)
    if final_norm_bias is not None:
        hidden = hidden + final_norm_bias.reshape(1, 1, -1).astype(np.float32)
    return hidden


def _load_model(model_dir: Path, *, rmsnorm: bool = False, zero_ln_bias: bool = False) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()
    if zero_ln_bias:
        _zero_layernorm_biases(model)
    if rmsnorm:
        _replace_layernorms_with_rms(model)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    print(f"loading dump: {args.dump}", file=sys.stderr, flush=True)
    dump = np.load(args.dump)
    input_ids = dump["input_ids"].astype(np.int64)
    attention_mask = dump["attention_mask"].astype(np.int64)
    valid_len = int(dump["valid_len"].reshape(-1)[0])
    rk_hidden = dump["rk_hidden"][:, :valid_len, :]
    onnx_hidden = dump["onnx_hidden"][:, :valid_len, :]
    final_norm_bias = None
    if args.assets is not None:
        final_norm_bias = np.load(args.assets)["final_norm_bias"].astype(np.float32)

    variants: dict[str, dict[str, Any]] = {}
    for name, kwargs in {
        "hf_original": {},
        "hf_zero_layernorm_bias": {"zero_ln_bias": True},
        "hf_rmsnorm": {"rmsnorm": True, "zero_ln_bias": True},
    }.items():
        print(f"running variant: {name}", file=sys.stderr, flush=True)
        model = _load_model(args.model_dir, **kwargs)
        hidden = _run_prefill(model, input_ids, attention_mask, args.assets, final_norm_bias)[:, :valid_len, :]
        variants[name] = {
            "vs_rkllm": _metrics(hidden, rk_hidden),
            "vs_onnx": _metrics(hidden, onnx_hidden),
        }

    report = {
        "model_dir": str(args.model_dir),
        "dump": str(args.dump),
        "valid_len": valid_len,
        "variants": variants,
        "best_vs_rkllm": min(variants.items(), key=lambda item: item[1]["vs_rkllm"]["rel_l2"])[0],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

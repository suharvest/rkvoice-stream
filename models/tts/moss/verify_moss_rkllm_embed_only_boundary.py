#!/usr/bin/env python3
"""Measure embed-only RKLLM scaffold drift against original MOSS ONNX graphs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_moss_rkllm_hidden_parity import _build_probe_rows, _metrics, _onnx_session, _run_onnx_decode


def _to_numpy_cache(past_key_values: Any) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for layer, (key, value) in enumerate(past_key_values):
        cache[f"present_key_{layer}"] = key.detach().cpu().numpy().astype(np.float32)
        cache[f"present_value_{layer}"] = value.detach().cpu().numpy().astype(np.float32)
    return cache


def _torch_cache_metrics(actual: dict[str, np.ndarray], expected: dict[str, np.ndarray], layers: int) -> list[dict[str, Any]]:
    metrics = []
    for layer in range(layers):
        metrics.append(
            {
                "layer": layer,
                "key": _metrics(actual[f"present_key_{layer}"], expected[f"present_key_{layer}"]),
                "value": _metrics(actual[f"present_value_{layer}"], expected[f"present_value_{layer}"]),
            }
        )
    return metrics


def _torch_past_from_cache(cache: dict[str, np.ndarray], layers: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            torch.from_numpy(cache[f"present_key_{layer}"]).float(),
            torch.from_numpy(cache[f"present_value_{layer}"]).float(),
        )
        for layer in range(layers)
    )


def verify(
    model_dir: Path,
    original_scaffold_dir: Path,
    target_scaffold_dir: Path,
    seq_len: int,
    threads: int,
    external_final_norm_bias: Path | None = None,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    original_scaffold_dir = original_scaffold_dir.resolve()
    target_scaffold_dir = target_scaffold_dir.resolve()
    config = AutoConfig.from_pretrained(original_scaffold_dir, trust_remote_code=True)
    layers = int(config.num_hidden_layers)
    rows, attention_mask = _build_probe_rows(config, seq_len)

    prefill = _onnx_session(model_dir / "moss_tts_prefill.onnx", threads)
    prefill_names = ["global_hidden"]
    for layer in range(layers):
        prefill_names.extend([f"present_key_{layer}", f"present_value_{layer}"])
    onnx_outputs = prefill.run(prefill_names, {"input_ids": rows, "attention_mask": attention_mask})
    onnx_hidden = np.asarray(onnx_outputs[0], dtype=np.float32)
    onnx_cache = {name: np.asarray(value) for name, value in zip(prefill_names[1:], onnx_outputs[1:])}

    original_model = AutoModelForCausalLM.from_pretrained(
        original_scaffold_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        target_scaffold_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    original_model.eval()
    target_model.eval()
    final_bias = None
    if external_final_norm_bias is not None:
        final_bias = np.load(external_final_norm_bias).astype(np.float32).reshape(1, 1, -1)
    with torch.no_grad():
        input_embeds = original_model.model._embed_rows(torch.from_numpy(rows).long())
        target_out = target_model(
            inputs_embeds=input_embeds,
            attention_mask=torch.from_numpy(attention_mask).long(),
            use_cache=True,
            output_hidden_states=True,
        )
    target_hidden = target_out.hidden_states[-1].detach().cpu().numpy().astype(np.float32)
    if final_bias is not None:
        target_hidden = target_hidden + final_bias
    target_cache = _to_numpy_cache(target_out.past_key_values)

    next_row, _ = _build_probe_rows(config, 1)
    next_row[0, 0, 0] = 13
    next_row[0, 0, 1:] = (np.arange(16, dtype=np.int32) * 11) % 1024
    onnx_decode_hidden, onnx_decode_cache = _run_onnx_decode(
        model_dir,
        onnx_cache,
        next_row,
        seq_len,
        threads,
        layers,
    )
    with torch.no_grad():
        next_embed = original_model.model._embed_rows(torch.from_numpy(next_row).long())
        target_decode = target_model(
            inputs_embeds=next_embed,
            past_key_values=_torch_past_from_cache(target_cache, layers),
            past_valid_lengths=torch.tensor([seq_len], dtype=torch.long),
            use_cache=True,
            output_hidden_states=True,
        )
    target_decode_hidden = target_decode.hidden_states[-1].detach().cpu().numpy().astype(np.float32)
    if final_bias is not None:
        target_decode_hidden = target_decode_hidden + final_bias
    target_decode_cache = _to_numpy_cache(target_decode.past_key_values)

    valid = attention_mask.astype(bool)
    report = {
        "model_dir": str(model_dir),
        "original_scaffold_dir": str(original_scaffold_dir),
        "target_scaffold_dir": str(target_scaffold_dir),
        "external_final_norm_bias": str(external_final_norm_bias.resolve()) if external_final_norm_bias else None,
        "seq_len": seq_len,
        "thresholds": {
            "max_prefill_rel_l2": 0.01,
            "max_decode_rel_l2": 0.01,
            "min_cosine": 0.999,
        },
        "prefill_metrics_all": _metrics(target_hidden, onnx_hidden),
        "prefill_metrics_valid_tokens": _metrics(target_hidden, onnx_hidden, valid),
        "prefill_cache_metrics": _torch_cache_metrics(target_cache, onnx_cache, layers),
        "decode_metrics": _metrics(target_decode_hidden, onnx_decode_hidden),
        "decode_cache_metrics": _torch_cache_metrics(target_decode_cache, onnx_decode_cache, layers),
        "target_hidden_shape": list(target_hidden.shape),
        "onnx_hidden_shape": list(onnx_hidden.shape),
        "target_decode_hidden_shape": list(target_decode_hidden.shape),
        "onnx_decode_hidden_shape": list(onnx_decode_hidden.shape),
    }
    valid_metrics = report["prefill_metrics_valid_tokens"]
    decode_metrics = report["decode_metrics"]
    report["passed"] = bool(
        valid_metrics["finite"]
        and decode_metrics["finite"]
        and valid_metrics["rel_l2"] <= report["thresholds"]["max_prefill_rel_l2"]
        and decode_metrics["rel_l2"] <= report["thresholds"]["max_decode_rel_l2"]
        and valid_metrics["cosine"] >= report["thresholds"]["min_cosine"]
        and decode_metrics["cosine"] >= report["thresholds"]["min_cosine"]
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--original-scaffold-dir", required=True, type=Path)
    parser.add_argument("--target-scaffold-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--external-final-norm-bias", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = verify(
        args.model_dir,
        args.original_scaffold_dir,
        args.target_scaffold_dir,
        args.seq_len,
        args.threads,
        args.external_final_norm_bias,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

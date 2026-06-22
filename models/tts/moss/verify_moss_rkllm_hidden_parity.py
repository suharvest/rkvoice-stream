#!/usr/bin/env python3
"""Compare MOSS RKLLM custom HF hidden states against ONNX prefill/decode graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _metrics(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        while m.ndim < a.ndim:
            m = np.expand_dims(m, -1)
        a = a[m.repeat(a.shape[-1], axis=-1)].reshape(-1)
        e = e[m.repeat(e.shape[-1], axis=-1)].reshape(-1)
    diff = a - e
    denom = float(np.linalg.norm(e.reshape(-1)) + 1e-12)
    cosine = float(np.dot(a.reshape(-1), e.reshape(-1)) / ((np.linalg.norm(a.reshape(-1)) + 1e-12) * denom))
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
        "finite": bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
    }


def _build_probe_rows(config: Any, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    moss = config.moss_rkllm
    audio_pad = int(moss["audio_pad_token_id"])
    rows = np.full((1, seq_len, 17), audio_pad, dtype=np.int32)
    text_ids = [int(config.bos_token_id or 4), 10, 11, 12, int(config.eos_token_id or 5)]
    for idx in range(seq_len):
        rows[0, idx, 0] = text_ids[idx % len(text_ids)]
    if seq_len >= 3:
        rows[0, 1, 1:] = np.arange(16, dtype=np.int32)
        rows[0, 2, 1:] = (np.arange(16, dtype=np.int32) * 7) % 1024
    attention_mask = np.ones((1, seq_len), dtype=np.int32)
    return rows, attention_mask


def _onnx_session(path: Path, threads: int) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=sess_options, providers=["CPUExecutionProvider"])


def _to_torch_past(kv_cache: dict[str, np.ndarray], layers: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple(
        (
            torch.from_numpy(kv_cache[f"present_key_{layer}"]).float(),
            torch.from_numpy(kv_cache[f"present_value_{layer}"]).float(),
        )
        for layer in range(layers)
    )


def _run_onnx_decode(
    model_dir: Path,
    prefill_cache: dict[str, np.ndarray],
    next_row: np.ndarray,
    past_len: int,
    threads: int,
    layers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    decode = _onnx_session(model_dir / "moss_tts_decode_step.onnx", threads)
    inputs: dict[str, np.ndarray] = {
        "input_ids": next_row.astype(np.int32, copy=False),
        "past_valid_lengths": np.array([past_len], dtype=np.int32),
    }
    for layer in range(layers):
        inputs[f"past_key_{layer}"] = prefill_cache[f"present_key_{layer}"].astype(np.float32, copy=False)
        inputs[f"past_value_{layer}"] = prefill_cache[f"present_value_{layer}"].astype(np.float32, copy=False)
    output_names = ["global_hidden"]
    for layer in range(layers):
        output_names.extend([f"present_key_{layer}", f"present_value_{layer}"])
    outputs = decode.run(output_names, inputs)
    return np.asarray(outputs[0], dtype=np.float32), {
        name: np.asarray(value)
        for name, value in zip(output_names[1:], outputs[1:])
    }


def verify(model_dir: Path, scaffold_dir: Path, seq_len: int = 8, threads: int = 2) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    scaffold_dir = scaffold_dir.resolve()
    config = AutoConfig.from_pretrained(scaffold_dir, trust_remote_code=True)
    rows, attention_mask = _build_probe_rows(config, seq_len)

    prefill = _onnx_session(model_dir / "moss_tts_prefill.onnx", threads)
    prefill_output_names = ["global_hidden"]
    for layer in range(int(config.num_hidden_layers)):
        prefill_output_names.extend([f"present_key_{layer}", f"present_value_{layer}"])
    onnx_outputs = prefill.run(prefill_output_names, {"input_ids": rows, "attention_mask": attention_mask})
    onnx_hidden = np.asarray(onnx_outputs[0], dtype=np.float32)
    onnx_cache = {
        name: np.asarray(value)
        for name, value in zip(prefill_output_names[1:], onnx_outputs[1:])
    }

    torch_model = AutoModelForCausalLM.from_pretrained(scaffold_dir, trust_remote_code=True, local_files_only=True)
    torch_model.eval()
    with torch.no_grad():
        out = torch_model(
            input_ids=torch.from_numpy(rows).long(),
            attention_mask=torch.from_numpy(attention_mask).long(),
            use_cache=True,
            output_hidden_states=True,
        )
    torch_hidden = out.hidden_states[-1].detach().cpu().numpy().astype(np.float32)
    next_row, _next_mask = _build_probe_rows(config, 1)
    next_row[0, 0, 0] = 13
    next_row[0, 0, 1:] = (np.arange(16, dtype=np.int32) * 11) % 1024
    onnx_decode_hidden, onnx_decode_cache = _run_onnx_decode(
        model_dir,
        onnx_cache,
        next_row,
        seq_len,
        threads,
        int(config.num_hidden_layers),
    )
    with torch.no_grad():
        decode_out = torch_model(
            input_ids=torch.from_numpy(next_row).long(),
            past_key_values=_to_torch_past(onnx_cache, int(config.num_hidden_layers)),
            past_valid_lengths=torch.tensor([seq_len], dtype=torch.long),
            use_cache=True,
            output_hidden_states=True,
        )
    torch_decode_hidden = decode_out.hidden_states[-1].detach().cpu().numpy().astype(np.float32)
    torch_decode_cache = decode_out.past_key_values
    cache_metrics = []
    if torch_decode_cache is not None:
        for layer, (key, value) in enumerate(torch_decode_cache):
            cache_metrics.append(
                {
                    "layer": layer,
                    "key": _metrics(key.detach().cpu().numpy(), onnx_decode_cache[f"present_key_{layer}"]),
                    "value": _metrics(value.detach().cpu().numpy(), onnx_decode_cache[f"present_value_{layer}"]),
                }
            )
    valid = attention_mask.astype(bool)
    return {
        "model_dir": str(model_dir),
        "scaffold_dir": str(scaffold_dir),
        "seq_len": seq_len,
        "passed": False,
        "thresholds": {"max_rel_l2": 1e-3, "min_cosine": 0.999},
        "metrics_all": _metrics(torch_hidden, onnx_hidden),
        "metrics_valid_tokens": _metrics(torch_hidden, onnx_hidden, valid),
        "decode_metrics": _metrics(torch_decode_hidden, onnx_decode_hidden),
        "decode_cache_metrics": cache_metrics,
        "torch_hidden_shape": list(torch_hidden.shape),
        "onnx_hidden_shape": list(onnx_hidden.shape),
        "torch_decode_hidden_shape": list(torch_decode_hidden.shape),
        "onnx_decode_hidden_shape": list(onnx_decode_hidden.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = verify(args.model_dir, args.scaffold_dir, args.seq_len, args.threads)
    valid = report["metrics_valid_tokens"]
    report["passed"] = bool(
        valid["finite"]
        and valid["rel_l2"] <= report["thresholds"]["max_rel_l2"]
        and valid["cosine"] >= report["thresholds"]["min_cosine"]
        and report["decode_metrics"]["finite"]
        and report["decode_metrics"]["rel_l2"] <= report["thresholds"]["max_rel_l2"]
        and report["decode_metrics"]["cosine"] >= report["thresholds"]["min_cosine"]
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare RKLLM toolkit logits against HuggingFace for the MOSS scaffold.

This is a host-side RKLLM diagnostic.  It does not prove RK3576 runtime
accuracy, but it separates toolkit conversion/config issues from board runtime
issues before spending time on black-box NPU probes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = actual.astype(np.float32, copy=False)
    expected = expected.astype(np.float32, copy=False)
    diff = actual - expected
    denom = float(np.linalg.norm(expected.reshape(-1)) + 1e-12)
    cosine = float(
        np.dot(actual.reshape(-1), expected.reshape(-1))
        / ((np.linalg.norm(actual.reshape(-1)) + 1e-12) * denom)
    )
    return {
        "shape": list(actual.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
        "finite": bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _to_numpy(value[0])
        return np.asarray(value)
    if isinstance(value, dict):
        for key in ("logits", "output", "outputs"):
            if key in value:
                return _to_numpy(value[key])
    return np.asarray(value)


def _describe_value(value: Any) -> dict[str, Any]:
    desc = {"type": type(value).__name__, "repr": repr(value)[:500]}
    if isinstance(value, dict):
        desc["keys"] = sorted(str(key) for key in value.keys())
    if isinstance(value, (list, tuple)):
        desc["len"] = len(value)
        desc["item_types"] = [type(item).__name__ for item in value[:4]]
    if isinstance(value, np.ndarray):
        desc["shape"] = list(value.shape)
        desc["dtype"] = str(value.dtype)
    if torch.is_tensor(value):
        desc["shape"] = list(value.shape)
        desc["dtype"] = str(value.dtype)
    return desc


def _make_input(seq_len: int, vocab_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Keep IDs away from special high rows so this probe isolates the text-token
    # path accepted by RKLLM.get_logits().
    return rng.integers(1, min(vocab_size, 2048), size=(1, seq_len), dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--custom-config", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--target-platform", default="rk3576")
    parser.add_argument("--num-npu-core", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--max-rel-l2", type=float, default=0.02)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    from rkllm.api import RKLLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()
    input_ids = _make_input(args.seq_len, int(hf_model.config.vocab_size), args.seed)

    report: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "custom_config": str(args.custom_config),
        "seq_len": args.seq_len,
        "seed": args.seed,
        "thresholds": {"max_rel_l2": args.max_rel_l2, "min_cosine": args.min_cosine},
        "input_ids": input_ids.tolist(),
    }

    with torch.no_grad():
        t0 = time.perf_counter()
        hf_logits = hf_model(input_ids=torch.from_numpy(input_ids)).logits.detach().cpu().numpy()
        report["hf_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    llm = RKLLM()
    t0 = time.perf_counter()
    ret = llm.load_huggingface(
        model=str(args.model_dir),
        model_lora=None,
        device=args.device,
        dtype=args.dtype,
        custom_config=str(args.custom_config),
        load_weight=True,
    )
    report["load_huggingface_ret"] = ret
    report["load_huggingface_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    if ret != 0:
        report["passed"] = False
    else:
        if args.build:
            t0 = time.perf_counter()
            build_ret = llm.build(
                do_quantization=False,
                optimization_level=1,
                target_platform=args.target_platform,
                num_npu_core=args.num_npu_core,
                max_context=args.max_context,
            )
            report["build_ret"] = build_ret
            report["build_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        t0 = time.perf_counter()
        raw_logits = llm.get_logits({"input_ids": input_ids})
        report["rkllm_get_logits_raw"] = _describe_value(raw_logits)
        rk_logits = _to_numpy(raw_logits)
        report["rkllm_get_logits_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        if rk_logits.shape != hf_logits.shape:
            report["rkllm_logits_shape"] = list(rk_logits.shape)
            report["hf_logits_shape"] = list(hf_logits.shape)
            report["passed"] = False
        else:
            report["logit_metrics"] = _metrics(rk_logits, hf_logits)
            report["passed"] = bool(
                report["logit_metrics"]["finite"]
                and report["logit_metrics"]["rel_l2"] <= args.max_rel_l2
                and report["logit_metrics"]["cosine"] >= args.min_cosine
            )

    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())

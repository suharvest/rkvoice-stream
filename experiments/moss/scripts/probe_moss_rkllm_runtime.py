#!/usr/bin/env python3
"""Probe an exported MOSS global RKLLM model on RK3576 with embedding input."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from rkvoice_stream.runtime.rkllm_wrapper import RKLLM_INFER_GET_LAST_HIDDEN_LAYER, RKLLMTalker


def _summary(name: str, value: np.ndarray | None) -> dict[str, Any]:
    if value is None:
        return {"name": name, "present": False}
    arr = np.asarray(value)
    return {
        "name": name,
        "present": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(arr).all()),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _apply_final_bias(result: dict[str, np.ndarray], final_bias: np.ndarray | None) -> dict[str, np.ndarray]:
    if final_bias is not None and result.get("hidden") is not None:
        result = dict(result)
        result["hidden"] = np.asarray(result["hidden"], dtype=np.float32) + final_bias.reshape(1, -1)
    return result


def probe(model_path: Path, seq_len: int, hidden_size: int, seed: int, final_norm_bias: Path | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "seed": seed,
        "final_norm_bias": str(final_norm_bias.resolve()) if final_norm_bias else None,
        "loaded": False,
        "prefill_ok": False,
        "decode_ok": False,
        "exception": None,
    }
    rng = np.random.default_rng(seed)
    final_bias = np.load(final_norm_bias).astype(np.float32) if final_norm_bias else None
    model: RKLLMTalker | None = None
    try:
        t0 = time.perf_counter()
        model = RKLLMTalker(str(model_path), max_context_len=512, max_new_tokens=1)
        report["load_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        report["loaded"] = True

        prefill_embeds = rng.normal(0.0, 0.2, size=(seq_len, hidden_size)).astype(np.float32)
        t0 = time.perf_counter()
        prefill = _apply_final_bias(
            model.run_embed(prefill_embeds, mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER, keep_history=1),
            final_bias,
        )
        report["prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        report["prefill_hidden"] = _summary("prefill_hidden", prefill.get("hidden"))
        report["prefill_ok"] = bool(report["prefill_hidden"].get("finite"))

        decode_embed = rng.normal(0.0, 0.2, size=(1, hidden_size)).astype(np.float32)
        t0 = time.perf_counter()
        decode = _apply_final_bias(
            model.run_embed(decode_embed, mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER, keep_history=1),
            final_bias,
        )
        report["decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        report["decode_hidden"] = _summary("decode_hidden", decode.get("hidden"))
        report["decode_ok"] = bool(report["decode_hidden"].get("finite"))
        model.clear_kv_cache()
    except Exception as exc:  # noqa: BLE001 - probe must report runtime failures.
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }
    finally:
        if model is not None:
            model.destroy()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--final-norm-bias", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = probe(args.model, args.seq_len, args.hidden_size, args.seed, args.final_norm_bias)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["loaded"] and report["prefill_ok"] and report["decode_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

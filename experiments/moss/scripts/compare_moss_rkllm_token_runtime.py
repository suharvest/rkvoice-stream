#!/usr/bin/env python3
"""Compare RKLLM token-input hidden output against MOSS ONNX text-only rows."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from rkvoice_stream.backends.tts.moss_ort import MossORTBackend
from rkvoice_stream.runtime.rkllm_wrapper import RKLLM_INFER_GET_LAST_HIDDEN_LAYER, RKLLMTalker
from smoke_moss_rkllm_stream import MossRkllmEmbedder


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    diff = a - e
    denom = float(np.linalg.norm(e.reshape(-1)) + 1e-12)
    return {
        "shape": list(a.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": float(np.dot(a.reshape(-1), e.reshape(-1)) / ((np.linalg.norm(a.reshape(-1)) + 1e-12) * denom)),
        "finite": bool(np.isfinite(a).all() and np.isfinite(e).all()),
    }


def _make_ort_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rkllm-model", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--token-ids", default="945,1889,1388,1459,1147,1540,1442,1167")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("MOSS_ORT_MODEL_DIR", str(args.model_dir))
    backend = MossORTBackend()
    rkllm: RKLLMTalker | None = None
    report: dict[str, Any] = {
        "model_dir": str(args.model_dir),
        "rkllm_model": str(args.rkllm_model),
        "token_ids": [int(x) for x in args.token_ids.split(",") if x.strip()],
        "thresholds": {"max_rel_l2": 0.02, "min_cosine": 0.999},
        "passed": False,
    }
    try:
        backend.preload()
        token_ids = np.asarray(report["token_ids"], dtype=np.int32).reshape(1, -1)
        rows = np.full((1, token_ids.shape[1], 17), backend._audio_pad_token_id(), dtype=np.int32)
        rows[..., 0] = token_ids
        attention_mask = np.ones((1, token_ids.shape[1]), dtype=np.int32)
        prefill_names = backend._tts_meta["onnx"]["prefill_output_names"]
        prefill = _make_ort_session(args.model_dir / "moss_tts_prefill.onnx", args.threads)
        t0 = time.perf_counter()
        onnx_hidden = prefill.run(prefill_names, {"input_ids": rows, "attention_mask": attention_mask})[0].astype(np.float32)
        report["onnx_prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        embedder = MossRkllmEmbedder(args.assets, audio_pad_token_id=backend._audio_pad_token_id())
        rkllm = RKLLMTalker(str(args.rkllm_model), max_context_len=512, max_new_tokens=1)
        t0 = time.perf_counter()
        rk_hidden = rkllm.run_tokens(
            token_ids.reshape(-1),
            mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
            keep_history=0,
        )["hidden"]
        report["rkllm_prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rk_hidden = embedder.apply_final_bias(rk_hidden)[None, :, :]
        report["prefill_metrics"] = _metrics(rk_hidden, onnx_hidden)
        report["passed"] = bool(
            report["prefill_metrics"]["finite"]
            and report["prefill_metrics"]["rel_l2"] <= report["thresholds"]["max_rel_l2"]
            and report["prefill_metrics"]["cosine"] >= report["thresholds"]["min_cosine"]
        )
    finally:
        if rkllm is not None:
            rkllm.destroy()
        backend.cleanup()

    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

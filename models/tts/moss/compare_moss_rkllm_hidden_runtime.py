#!/usr/bin/env python3
"""Compare RK3576 RKLLM hidden output against MOSS ONNX Runtime golden."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
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
    cosine = float(np.dot(a.reshape(-1), e.reshape(-1)) / ((np.linalg.norm(a.reshape(-1)) + 1e-12) * denom))
    return {
        "shape": list(a.shape),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
        "finite": bool(np.isfinite(a).all() and np.isfinite(e).all()),
    }


def _make_ort_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _session_input_dtype(session: ort.InferenceSession, name: str) -> np.dtype:
    for item in session.get_inputs():
        if item.name == name:
            return np.float16 if item.type == "tensor(float16)" else np.float32
    return np.float32


def compare(
    model_dir: Path,
    rkllm_model: Path,
    assets: Path,
    text: str,
    threads: int,
    npz_out: Path | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("MOSS_ORT_MODEL_DIR", str(model_dir))
    backend = MossORTBackend()
    rkllm: RKLLMTalker | None = None
    report: dict[str, Any] = {
        "model_dir": str(model_dir.resolve()),
        "rkllm_model": str(rkllm_model.resolve()),
        "assets": str(assets.resolve()),
        "text": text,
        "threads": threads,
        "passed": False,
        "thresholds": {"max_prefill_rel_l2": 0.02, "max_decode_rel_l2": 0.03, "min_cosine": 0.999},
    }
    try:
        backend.preload()
        embedder = MossRkllmEmbedder(assets, audio_pad_token_id=backend._audio_pad_token_id())
        input_ids, mode = backend._build_prefill_rows(text)
        attention_mask = (input_ids[:, :, 0] != backend._pad_token_id()).astype(np.int32)
        if not attention_mask.any():
            attention_mask[:] = 1
        prefill_names = backend._tts_meta["onnx"]["prefill_output_names"]
        prefill = _make_ort_session(model_dir / "moss_tts_prefill.onnx", threads)
        t0 = time.perf_counter()
        onnx_outputs = prefill.run(prefill_names, {"input_ids": input_ids, "attention_mask": attention_mask})
        report["onnx_prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        onnx_hidden = np.asarray(onnx_outputs[0], dtype=np.float32)
        onnx_cache = {prefill_names[i]: onnx_outputs[i] for i in range(1, len(prefill_names))}

        rkllm = RKLLMTalker(str(rkllm_model), max_context_len=512, max_new_tokens=1)
        t0 = time.perf_counter()
        rk_prefill = rkllm.run_embed(
            embedder.rows_to_embeddings(input_ids),
            mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
            keep_history=1,
        )
        report["rkllm_prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rk_hidden = embedder.apply_final_bias(rk_prefill["hidden"])[None, :, :]
        valid_len = int(attention_mask.sum())
        report["mode"] = mode
        report["seq_len"] = int(input_ids.shape[1])
        report["valid_len"] = valid_len
        report["prefill_metrics"] = _metrics(rk_hidden[:, :valid_len, :], onnx_hidden[:, :valid_len, :])

        next_row = backend._make_audio_row(np.arange(16, dtype=np.int32) % 1024).reshape(1, 1, 17)
        decode_inputs: dict[str, np.ndarray] = {
            "input_ids": next_row.astype(np.int32, copy=False),
            "past_valid_lengths": np.asarray([valid_len], dtype=np.int32),
        }
        for layer in range(12):
            decode_inputs[f"past_key_{layer}"] = onnx_cache[f"present_key_{layer}"].astype(np.float32, copy=False)
            decode_inputs[f"past_value_{layer}"] = onnx_cache[f"present_value_{layer}"].astype(np.float32, copy=False)
        decode_names = backend._tts_meta["onnx"]["decode_output_names"]
        decode = _make_ort_session(model_dir / "moss_tts_decode_step.onnx", threads)
        kv_dtype = _session_input_dtype(decode, "past_key_0")
        for layer in range(12):
            decode_inputs[f"past_key_{layer}"] = decode_inputs[f"past_key_{layer}"].astype(kv_dtype, copy=False)
            decode_inputs[f"past_value_{layer}"] = decode_inputs[f"past_value_{layer}"].astype(kv_dtype, copy=False)
        t0 = time.perf_counter()
        onnx_decode_hidden = decode.run(decode_names, decode_inputs)[0].astype(np.float32)
        report["onnx_decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        t0 = time.perf_counter()
        rk_decode = rkllm.run_embed(
            embedder.rows_to_embeddings(next_row),
            mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
            keep_history=1,
        )
        report["rkllm_decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        rk_decode_hidden = embedder.apply_final_bias(rk_decode["hidden"])[None, :, :]
        report["decode_metrics"] = _metrics(rk_decode_hidden, onnx_decode_hidden)
        if npz_out is not None:
            npz_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_out,
                input_ids=input_ids,
                attention_mask=attention_mask,
                next_row=next_row,
                onnx_hidden=onnx_hidden,
                rk_hidden=rk_hidden,
                onnx_decode_hidden=onnx_decode_hidden,
                rk_decode_hidden=rk_decode_hidden,
                valid_len=np.asarray([valid_len], dtype=np.int32),
            )
            report["npz_out"] = str(npz_out)
        report["passed"] = bool(
            report["prefill_metrics"]["finite"]
            and report["decode_metrics"]["finite"]
            and report["prefill_metrics"]["rel_l2"] <= report["thresholds"]["max_prefill_rel_l2"]
            and report["decode_metrics"]["rel_l2"] <= report["thresholds"]["max_decode_rel_l2"]
            and report["prefill_metrics"]["cosine"] >= report["thresholds"]["min_cosine"]
            and report["decode_metrics"]["cosine"] >= report["thresholds"]["min_cosine"]
        )
    except Exception as exc:  # noqa: BLE001 - verifier must report runtime failures.
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }
    finally:
        if rkllm is not None:
            rkllm.destroy()
        backend.cleanup()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rkllm-model", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--text", default="你好，欢迎使用本地语音助手。")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--npz-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = compare(args.model_dir, args.rkllm_model, args.assets, args.text, args.threads, args.npz_out)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

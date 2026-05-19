#!/usr/bin/env python3
"""Run a pre-extracted Kokoro tail RKNN island benchmark on device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

TAIL_INPUT = "/decoder/decode.3/Mul_output_0"
STYLE_INPUT = "/Slice_2_output_0"
ISLAND_INPUT = "/decoder/generator/LeakyRelu_1_output_0"
ISLAND_OUTPUT = "/decoder/generator/ups.1/ConvTranspose_output_0"


def _make_session(path: Path, intra: int) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = intra
    opt.inter_op_num_threads = 1
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-onnx", required=True, type=Path)
    parser.add_argument("--pre-onnx", required=True, type=Path)
    parser.add_argument("--post-onnx", required=True, type=Path)
    parser.add_argument("--island-rknn", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--intra-op", type=int, default=4)
    args = parser.parse_args()

    full_sess = _make_session(args.tail_onnx, args.intra_op)
    pre_sess = _make_session(args.pre_onnx, args.intra_op)
    post_sess = _make_session(args.post_onnx, args.intra_op)
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.island_rknn))
    if ret != 0:
        raise RuntimeError(f"load_rknn returned {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime returned {ret}")

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((1, 512, 218)).astype(np.float32)
    style = rng.standard_normal((1, 128)).astype(np.float32)
    full_feed = {TAIL_INPUT: hidden, STYLE_INPUT: style}
    pre_feed = {TAIL_INPUT: hidden, STYLE_INPUT: style}

    full_times = []
    split_times = []
    parts = []
    full_out = None
    split_out = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        full_out = full_sess.run(None, full_feed)[0]
        full_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        island_in = pre_sess.run(None, pre_feed)[0]
        pre_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        island_outputs = rknn.inference(inputs=[island_in])
        if not island_outputs:
            raise RuntimeError("RKNN island inference returned None")
        island_out = np.asarray(island_outputs[0], dtype=np.float32)
        rknn_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        split_out = post_sess.run(None, {ISLAND_OUTPUT: island_out, STYLE_INPUT: style})[0]
        post_s = time.perf_counter() - t2
        split_times.append(pre_s + rknn_s + post_s)
        parts.append({"pre_ms": pre_s * 1000, "rknn_ms": rknn_s * 1000, "post_ms": post_s * 1000})

    rknn.release()
    diff = split_out.astype(np.float32) - full_out.astype(np.float32)
    result = {
        "runs": args.runs,
        "full_ms": [round(v * 1000, 3) for v in full_times],
        "split_ms": [round(v * 1000, 3) for v in split_times],
        "full_avg_ms": round(sum(full_times) / len(full_times) * 1000, 3),
        "split_avg_ms": round(sum(split_times) / len(split_times) * 1000, 3),
        "parts_avg_ms": {
            key: round(sum(item[key] for item in parts) / len(parts), 3)
            for key in ("pre_ms", "rknn_ms", "post_ms")
        },
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / max(np.linalg.norm(full_out.reshape(-1)), 1e-12)),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

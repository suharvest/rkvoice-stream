#!/usr/bin/env python3
"""Benchmark one ONNX Runtime model with one chosen config."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _dims(shape: list[int | str | None]) -> list[int]:
    dims = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            raise ValueError(f"dynamic input shape is unsupported: {shape}")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--intra", type=int, default=4)
    parser.add_argument("--graph-opt", default="all", choices=["disable", "basic", "extended", "all"])
    parser.add_argument("--mem-pattern", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    opt = ort.SessionOptions()
    opt.intra_op_num_threads = args.intra
    opt.inter_op_num_threads = 1
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opt.graph_optimization_level = levels[args.graph_opt]
    opt.enable_mem_pattern = args.mem_pattern
    sess = ort.InferenceSession(str(args.model), opt, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    feed = {item.name: rng.standard_normal(_dims(item.shape)).astype(np.float32) for item in sess.get_inputs()}
    for _ in range(args.warmup):
        outputs = sess.run(None, feed)
    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = sess.run(None, feed)
        times.append(time.perf_counter() - start)
    samples = int(np.asarray(outputs[0]).size)
    avg = sum(times) / len(times)
    print(json.dumps({
        "model": str(args.model),
        "avg_ms": round(avg * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "rtf": round(avg / (samples / 24000.0), 4),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

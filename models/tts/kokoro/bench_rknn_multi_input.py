#!/usr/bin/env python3
"""Benchmark an RKNN model with one or more .npy inputs on device."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rknnlite.api import RKNNLite


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, action="append")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    inputs = [np.load(path).astype(np.float32, copy=False) for path in args.input]
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.model))
    if ret != 0:
        raise RuntimeError(f"load_rknn returned {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime returned {ret}")

    outputs = None
    for _ in range(args.warmup):
        outputs = rknn.inference(inputs=inputs)
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")

    times = []
    for _ in range(args.runs):
        start = time.perf_counter()
        outputs = rknn.inference(inputs=inputs)
        times.append(time.perf_counter() - start)
        if not outputs:
            raise RuntimeError("RKNN inference returned no outputs")

    rknn.release()
    out_stats = []
    for output in outputs or []:
        arr = np.asarray(output)
        out_stats.append(
            {
                "shape": list(arr.shape),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    result = {
        "model": str(args.model),
        "runs": args.runs,
        "times_ms": [round(item * 1000, 3) for item in times],
        "avg_ms": round(sum(times) / len(times) * 1000, 3),
        "outputs": out_stats,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

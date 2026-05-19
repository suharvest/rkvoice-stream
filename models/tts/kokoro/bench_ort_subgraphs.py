#!/usr/bin/env python3
"""Benchmark Kokoro ORT subgraphs with random static-shape inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _session(path: Path, intra_op: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _shape(item) -> list[int]:
    out = []
    for dim in item.shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{item.name} has non-static shape: {item.shape}")
        out.append(dim)
    return out


def _bench(sess: ort.InferenceSession, runs: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    feeds = {
        item.name: rng.standard_normal(_shape(item)).astype(np.float32)
        for item in sess.get_inputs()
    }
    times = []
    outputs = None
    for _ in range(runs):
        t0 = time.perf_counter()
        outputs = sess.run(None, feeds)
        times.append((time.perf_counter() - t0) * 1000)
    assert outputs is not None
    return {
        "inputs": [{"name": item.name, "shape": _shape(item)} for item in sess.get_inputs()],
        "outputs": [
            {
                "name": item.name,
                "shape": list(arr.shape),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }
            for item, arr in zip(sess.get_outputs(), outputs)
        ],
        "times_ms": [round(v, 3) for v in times],
        "avg_ms": round(sum(times) / len(times), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--intra-op", type=int, default=4)
    args = parser.parse_args()

    result = {}
    for path in args.model:
        result[str(path)] = _bench(_session(path, args.intra_op), args.runs, seed=len(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

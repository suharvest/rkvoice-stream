#!/usr/bin/env python3
"""Benchmark ONNX Runtime thread options for Kokoro CPU tail."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _dims(shape: list[int | str | None]) -> list[int]:
    result = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        else:
            raise ValueError(f"dynamic input shape is unsupported: {shape}")
    return result


def _make_session(model: Path, intra: int, inter: int, graph_opt: str) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = intra
    opt.inter_op_num_threads = inter
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opt.graph_optimization_level = levels[graph_opt]
    return ort.InferenceSession(str(model), opt, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    configs = [
        (1, 1, "all"),
        (2, 1, "all"),
        (3, 1, "all"),
        (4, 1, "all"),
        (6, 1, "all"),
        (4, 2, "all"),
        (4, 1, "extended"),
    ]
    rng = np.random.default_rng(0)
    for intra, inter, graph_opt in configs:
        sess = _make_session(args.model, intra, inter, graph_opt)
        feed = {
            item.name: rng.standard_normal(_dims(item.shape)).astype(np.float32)
            for item in sess.get_inputs()
        }
        for _ in range(args.warmup):
            sess.run(None, feed)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            outputs = sess.run(None, feed)
            times.append(time.perf_counter() - t0)
        audio_samples = int(np.asarray(outputs[0]).size)
        duration_s = audio_samples / 24000.0
        avg_s = sum(times) / len(times)
        print(
            json.dumps(
                {
                    "intra": intra,
                    "inter": inter,
                    "graph_opt": graph_opt,
                    "avg_ms": round(avg_s * 1000, 3),
                    "min_ms": round(min(times) * 1000, 3),
                    "rtf": round(avg_s / duration_s, 4) if duration_s else 0.0,
                    "shape": list(np.asarray(outputs[0]).shape),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

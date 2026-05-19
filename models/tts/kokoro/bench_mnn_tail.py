#!/usr/bin/env python3
"""Benchmark a converted MNN Kokoro tail model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import MNN
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--hidden-shape", default="1,512,420")
    parser.add_argument("--style-shape", default="1,128")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    hidden_shape = tuple(int(x) for x in args.hidden_shape.split(","))
    style_shape = tuple(int(x) for x in args.style_shape.split(","))
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal(hidden_shape).astype(np.float32)
    style = rng.standard_normal(style_shape).astype(np.float32)

    interpreter = MNN.Interpreter(str(args.model))
    session = interpreter.createSession({"numThread": args.threads, "backend": "CPU"})
    hidden_input = interpreter.getSessionInput(session, "/decoder/decode.3/Mul_output_0")
    style_input = interpreter.getSessionInput(session, "/Slice_2_output_0")
    hidden_tensor = MNN.Tensor(hidden_shape, MNN.Halide_Type_Float, hidden, MNN.Tensor_DimensionType_Caffe)
    style_tensor = MNN.Tensor(style_shape, MNN.Halide_Type_Float, style, MNN.Tensor_DimensionType_Caffe)

    output = None
    for _ in range(args.warmup):
        hidden_input.copyFrom(hidden_tensor)
        style_input.copyFrom(style_tensor)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session, "audio")

    times = []
    for _ in range(args.runs):
        hidden_input.copyFrom(hidden_tensor)
        style_input.copyFrom(style_tensor)
        start = time.perf_counter()
        interpreter.runSession(session)
        times.append(time.perf_counter() - start)
        output = interpreter.getSessionOutput(session, "audio")

    host = MNN.Tensor(output.getShape(), MNN.Halide_Type_Float, np.zeros(output.getShape(), dtype=np.float32), MNN.Tensor_DimensionType_Caffe)
    output.copyToHostTensor(host)
    arr = np.array(host.getData(), dtype=np.float32)
    avg = sum(times) / len(times)
    print(json.dumps({
        "model": str(args.model),
        "avg_ms": round(avg * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "output_shape": list(output.getShape()),
        "output_mean": float(np.mean(arr)),
        "output_std": float(np.std(arr)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

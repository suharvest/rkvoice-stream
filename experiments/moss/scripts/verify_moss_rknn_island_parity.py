#!/usr/bin/env python3
"""Compare a small MOSS RKNN island against its extracted ONNX Runtime graph."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite


def _make_input(shape: list[int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.25, size=shape).astype(np.float32)


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, float | bool | list[int] | str]:
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    diff = got - ref
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    denom = float(np.linalg.norm(ref_flat)) + 1e-12
    cosine = float(np.dot(ref_flat, got_flat) / ((np.linalg.norm(ref_flat) * np.linalg.norm(got_flat)) + 1e-12))
    return {
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "finite": bool(np.isfinite(got).all()),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
        "ref_min": float(np.min(ref)),
        "ref_max": float(np.max(ref)),
        "got_min": float(np.min(got)),
        "got_max": float(np.max(got)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--rknn", required=True, type=Path)
    parser.add_argument("--shape", default="1,32,768")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-rel-l2", type=float, default=0.08)
    parser.add_argument("--min-cosine", type=float, default=0.995)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    shape = [int(x) for x in args.shape.split(",") if x.strip()]

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in sess.get_inputs()]
    inputs = [_make_input(shape, args.seed + index) for index, _name in enumerate(input_names)]
    ort_inputs = dict(zip(input_names, inputs, strict=True))
    t0 = time.perf_counter()
    ref_outputs = sess.run(None, ort_inputs)
    ort_first_ms = (time.perf_counter() - t0) * 1000.0

    rknn = RKNNLite(verbose=False)
    try:
        ret = rknn.load_rknn(str(args.rknn))
        if ret != 0:
            raise RuntimeError(f"load_rknn returned {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime returned {ret}")
        t0 = time.perf_counter()
        got_outputs = rknn.inference(inputs=inputs)
        rknn_first_ms = (time.perf_counter() - t0) * 1000.0
        if got_outputs is None:
            raise RuntimeError("rknn.inference returned None")
        ort_times = []
        rknn_times = []
        for _ in range(max(0, args.repeat)):
            t0 = time.perf_counter()
            sess.run(None, ort_inputs)
            ort_times.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            repeat_outputs = rknn.inference(inputs=inputs)
            rknn_times.append((time.perf_counter() - t0) * 1000.0)
            if repeat_outputs is None:
                raise RuntimeError("repeat rknn.inference returned None")
    finally:
        try:
            rknn.release()
        except Exception:
            pass

    output_metrics = [_metrics(ref, got) for ref, got in zip(ref_outputs, got_outputs, strict=True)]
    passed = all(
        bool(item["finite"])
        and float(item["rel_l2"]) <= args.max_rel_l2
        and float(item["cosine"]) >= args.min_cosine
        for item in output_metrics
    )
    report = {
        "onnx": str(args.onnx),
        "rknn": str(args.rknn),
        "input_names": input_names,
        "input_shape": shape,
        "outputs": output_metrics,
        "latency_ms": {
            "ort_first": round(ort_first_ms, 3),
            "rknn_first": round(rknn_first_ms, 3),
            "ort_avg": round(float(np.mean(ort_times)), 3) if ort_times else None,
            "rknn_avg": round(float(np.mean(rknn_times)), 3) if rknn_times else None,
            "repeat": args.repeat,
        },
        "gates": {
            "max_rel_l2": args.max_rel_l2,
            "min_cosine": args.min_cosine,
            "passed": passed,
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

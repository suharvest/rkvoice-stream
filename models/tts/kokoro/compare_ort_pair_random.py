#!/usr/bin/env python3
"""Compare two ONNX Runtime models using identical random static-shape inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _dims(shape: list[int | str | None]) -> list[int]:
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            raise ValueError(f"dynamic input shape is unsupported: {shape}")
    return dims


def _make_session(path: Path, intra: int, graph_opt: str) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = intra
    opt.inter_op_num_threads = 1
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opt.graph_optimization_level = levels[graph_opt]
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    diff = af - bf
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / (np.linalg.norm(af) + 1e-12)),
        "cosine": float(np.dot(af, bf) / ((np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-12)),
        "ref_mean": float(np.mean(af)),
        "cand_mean": float(np.mean(bf)),
        "ref_std": float(np.std(af)),
        "cand_std": float(np.std(bf)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--intra", type=int, default=4)
    parser.add_argument("--graph-opt", default="all")
    args = parser.parse_args()

    ref = _make_session(args.reference, args.intra, args.graph_opt)
    cand = _make_session(args.candidate, args.intra, args.graph_opt)
    rng = np.random.default_rng(0)
    names = [item.name for item in ref.get_inputs()]
    shapes = [_dims(item.shape) for item in ref.get_inputs()]

    totals = []
    for sample in range(args.samples):
        feed = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in zip(names, shapes, strict=True)
        }
        ref_out = ref.run(None, feed)
        cand_out = cand.run(None, feed)
        sample_metrics = {
            f"output_{idx}": _metrics(np.asarray(a), np.asarray(b))
            for idx, (a, b) in enumerate(zip(ref_out, cand_out, strict=True))
        }
        totals.append(sample_metrics)
        print(json.dumps({"sample": sample, **sample_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

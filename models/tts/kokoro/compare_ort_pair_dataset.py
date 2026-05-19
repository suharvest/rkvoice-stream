#!/usr/bin/env python3
"""Compare two ONNX Runtime models using sample*_input*.npy datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _session(path: Path, intra: int, graph_opt: str) -> ort.InferenceSession:
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
        "ref_std": float(np.std(af)),
        "cand_std": float(np.std(bf)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--intra", type=int, default=4)
    parser.add_argument("--graph-opt", default="all")
    args = parser.parse_args()

    ref = _session(args.reference, args.intra, args.graph_opt)
    cand = _session(args.candidate, args.intra, args.graph_opt)
    names = [item.name for item in ref.get_inputs()]
    first_inputs = sorted(args.dataset_dir.glob("sample*_input0.npy"))
    for first in first_inputs:
        sample_id = first.name.split("_input", 1)[0]
        feed = {}
        for input_idx, name in enumerate(names):
            feed[name] = np.load(args.dataset_dir / f"{sample_id}_input{input_idx}.npy").astype(np.float32, copy=False)
        ref_out = ref.run(None, feed)
        cand_out = cand.run(None, feed)
        sample_metrics = {
            f"output_{idx}": _metrics(np.asarray(a), np.asarray(b))
            for idx, (a, b) in enumerate(zip(ref_out, cand_out, strict=True))
        }
        print(json.dumps({"sample": sample_id, **sample_metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare two ONNX models with the same inputs and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _load_inputs(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        return [data[key] for key in data.files]
    return [data]


def _run(model_path: Path, inputs: list[np.ndarray]) -> list[np.ndarray]:
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for meta, arr in zip(sess.get_inputs(), inputs, strict=True):
        feeds[meta.name] = arr.astype(np.float32, copy=False)
    return sess.run(None, feeds)


def _metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a.astype(np.float64) - b.astype(np.float64)
    denom = float(np.linalg.norm(a.astype(np.float64))) + 1e-12
    flat_a = a.reshape(-1).astype(np.float64)
    flat_b = b.reshape(-1).astype(np.float64)
    cosine = float(np.dot(flat_a, flat_b) / ((np.linalg.norm(flat_a) * np.linalg.norm(flat_b)) + 1e-12))
    return {
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff) / denom),
        "cosine": cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()

    inputs = _load_inputs(args.inputs)
    ref_outputs = _run(args.reference, inputs)
    cand_outputs = _run(args.candidate, inputs)
    for idx, (ref, cand) in enumerate(zip(ref_outputs, cand_outputs, strict=True)):
        values = _metrics(ref, cand)
        print(
            f"output[{idx}] shape={ref.shape} "
            f"mae={values['mae']:.9g} "
            f"max_abs={values['max_abs']:.9g} "
            f"rel_l2={values['rel_l2']:.9g} "
            f"cosine={values['cosine']:.9g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

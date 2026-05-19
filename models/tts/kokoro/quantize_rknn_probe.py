#!/usr/bin/env python3
"""Quantize a static ONNX probe graph to RKNN int8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx


def _dims(value_info) -> list[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        value = dim.dim_value
        if value <= 0:
            raise ValueError(f"dynamic input shape is unsupported: {value_info.name}")
        dims.append(int(value))
    return dims


def _np_dtype(value_info) -> np.dtype:
    elem_type = value_info.type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.INT64:
        return np.int64
    if elem_type == onnx.TensorProto.INT32:
        return np.int32
    return np.float32


def _make_dataset(onnx_path: Path, out_dir: Path, samples: int) -> Path:
    model = onnx.load(str(onnx_path))
    dataset_dir = out_dir / "quant_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    lines = []
    manifest = []
    for sample_idx in range(samples):
        paths = []
        for input_idx, value_info in enumerate(model.graph.input):
            shape = _dims(value_info)
            dtype = _np_dtype(value_info)
            if np.issubdtype(dtype, np.integer):
                data = rng.integers(0, 10, size=shape, dtype=dtype)
            else:
                data = rng.standard_normal(shape).astype(dtype)
            path = dataset_dir / f"sample{sample_idx:03d}_input{input_idx}.npy"
            np.save(path, data)
            paths.append(str(path))
            if sample_idx == 0:
                manifest.append({"name": value_info.name, "shape": shape, "dtype": str(dtype)})
        lines.append(" ".join(paths))
    dataset = out_dir / "dataset.txt"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset


def quantize(onnx_path: Path, output_rknn: Path, target: str, dataset: Path) -> None:
    from rknn.api import RKNN

    output_rknn.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=target, optimization_level=0)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")
        ret = rknn.build(do_quantization=True, dataset=str(dataset))
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")
        ret = rknn.export_rknn(str(output_rknn))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3588", choices=["rk3576", "rk3588"])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output-name", default="model.int8.rknn")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = _make_dataset(args.onnx, args.out_dir, args.samples)
    output_rknn = args.out_dir / args.target / args.output_name
    quantize(args.onnx, output_rknn, args.target, dataset)
    result = {"onnx": str(args.onnx), "dataset": str(dataset), "rknn": str(output_rknn)}
    (args.out_dir / "quant-manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

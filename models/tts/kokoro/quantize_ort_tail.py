#!/usr/bin/env python3
"""Create ORT CPU quantized variants for Kokoro tail ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)


def _dims(value_info) -> list[int]:
    dims: list[int] = []
    for dim in value_info.type.tensor_type.shape.dim:
        value = dim.dim_value
        if value <= 0:
            raise ValueError(f"dynamic input shape is unsupported: {value_info.name}")
        dims.append(int(value))
    return dims


class RandomReader(CalibrationDataReader):
    def __init__(self, model_path: Path, samples: int) -> None:
        model = onnx.load(str(model_path))
        rng = np.random.default_rng(0)
        self._items = []
        for _ in range(samples):
            feed = {}
            for item in model.graph.input:
                feed[item.name] = rng.standard_normal(_dims(item)).astype(np.float32)
            self._items.append(feed)
        self._iter = iter(self._items)

    def get_next(self):
        return next(self._iter, None)


class NpyDatasetReader(CalibrationDataReader):
    def __init__(self, model_path: Path, dataset_dir: Path) -> None:
        model = onnx.load(str(model_path))
        input_names = [item.name for item in model.graph.input]
        samples = sorted(dataset_dir.glob("sample*_input0.npy"))
        self._items = []
        for first in samples:
            sample_id = first.name.split("_input", 1)[0]
            feed = {}
            for input_idx, name in enumerate(input_names):
                path = dataset_dir / f"{sample_id}_input{input_idx}.npy"
                if not path.exists():
                    raise FileNotFoundError(path)
                feed[name] = np.load(path).astype(np.float32, copy=False)
            self._items.append(feed)
        if not self._items:
            raise ValueError(f"No sample*_input0.npy files found in {dataset_dir}")
        self._iter = iter(self._items)

    def get_next(self):
        return next(self._iter, None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--calib-dir", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    op_types = ["Conv", "Gemm", "MatMul"]

    dyn = args.out_dir / "tail.dynamic.u8s8.onnx"
    quantize_dynamic(
        model_input=str(args.input),
        model_output=str(dyn),
        op_types_to_quantize=op_types,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    print(f"dynamic={dyn}")

    reader = NpyDatasetReader(args.input, args.calib_dir) if args.calib_dir else RandomReader(args.input, args.samples)

    qdq = args.out_dir / "tail.static.qdq.u8s8.onnx"
    quantize_static(
        model_input=str(args.input),
        model_output=str(qdq),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=True,
    )
    print(f"static_qdq={qdq}")

    reader = NpyDatasetReader(args.input, args.calib_dir) if args.calib_dir else RandomReader(args.input, args.samples)
    qop = args.out_dir / "tail.static.qop.u8s8.onnx"
    quantize_static(
        model_input=str(args.input),
        model_output=str(qop),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types,
        per_channel=True,
    )
    print(f"static_qop={qop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

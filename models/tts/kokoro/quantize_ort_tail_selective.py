#!/usr/bin/env python3
"""Create selective ORT static int8 variants for Kokoro tail ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static


def _group(name: str) -> str:
    if "m_source" in name:
        return "source"
    if "noise_convs" in name or "noise_res" in name:
        return "noise"
    if "ups." in name:
        return "ups"
    if "resblocks.0" in name or "resblocks.1" in name or "resblocks.2" in name:
        return "resblocks_0_2"
    if "resblocks.3" in name or "resblocks.4" in name or "resblocks.5" in name:
        return "resblocks_3_5"
    if "conv_post" in name:
        return "post"
    if "adain" in name and "/fc/" in name:
        return "adain_fc"
    return "other"


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


def _nodes_for_groups(model_path: Path, groups: set[str], ops: set[str]) -> list[str]:
    model = onnx.load(str(model_path))
    nodes = []
    for node in model.graph.node:
        if node.op_type in ops and _group(node.name) in groups:
            nodes.append(node.name)
    return nodes


def _quantize(model: Path, output: Path, calib_dir: Path, nodes: list[str], fmt: QuantFormat) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(model),
        model_output=str(output),
        calibration_data_reader=NpyDatasetReader(model, calib_dir),
        quant_format=fmt,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["Conv", "ConvTranspose", "Gemm", "MatMul"],
        nodes_to_quantize=nodes,
        per_channel=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--calib-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--groups", nargs="+", required=True)
    parser.add_argument("--name-contains", nargs="*", default=[])
    parser.add_argument("--ops", nargs="+", default=["Conv", "ConvTranspose", "Gemm", "MatMul"])
    parser.add_argument("--format", choices=["qdq", "qop"], default="qdq")
    args = parser.parse_args()

    ops = set(args.ops)
    groups = set(args.groups)
    nodes = _nodes_for_groups(args.input, groups, ops)
    if args.name_contains:
        needles = tuple(args.name_contains)
        nodes = [name for name in nodes if any(needle in name for needle in needles)]
    if not nodes:
        raise ValueError(f"No quantizable nodes found for groups={sorted(groups)}")
    fmt = QuantFormat.QDQ if args.format == "qdq" else QuantFormat.QOperator
    name = "+".join(args.groups).replace("/", "_")
    output = args.out_dir / f"tail.selective.{name}.{args.format}.onnx"
    _quantize(args.input, output, args.calib_dir, nodes, fmt)
    print(json.dumps({"output": str(output), "groups": args.groups, "format": args.format, "nodes": nodes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

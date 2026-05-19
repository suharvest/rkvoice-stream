#!/usr/bin/env python3
"""Probe internal RKNN split points inside Kokoro generator tail.

This is an exploration tool for the CPU-tail bottleneck:

  /decoder/decode.3/Mul_output_0, /Slice_2_output_0 -> audio

It lists candidate tensors and can extract/convert a tail-prefix subgraph:

  /decoder/decode.3/Mul_output_0, /Slice_2_output_0 -> <candidate>

Any exported RKNN still needs real-device validation.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import onnx
from onnx import utils

FRONT_OUTPUT = "/decoder/decode.3/Mul_output_0"
STYLE_SLICE = "/Slice_2_output_0"


def _dims(value_info) -> list[int | str]:
    shape = value_info.type.tensor_type.shape
    dims: list[int | str] = []
    for dim in shape.dim:
        dims.append(dim.dim_value or dim.dim_param or "?")
    return dims


def _numel(dims: list[int | str]) -> int | None:
    total = 1
    for dim in dims:
        if not isinstance(dim, int):
            return None
        total *= dim
    return total


def _shape_map(model: onnx.ModelProto) -> dict[str, list[int | str]]:
    values = {}
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in coll:
            values[item.name] = _dims(item)
    return values


def _ensure_shapes(path: Path, out_path: Path) -> Path:
    model = onnx.load(str(path))
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def list_candidates(path: Path, pattern: str, max_rows: int) -> None:
    prepared = _ensure_shapes(path, path.with_suffix(".shaped.onnx"))
    model = onnx.load(str(prepared))
    shapes = _shape_map(model)
    rx = re.compile(pattern)
    rows = []
    for idx, node in enumerate(model.graph.node):
        if not rx.search(node.name):
            continue
        for out in node.output:
            dims = shapes.get(out, ["?"])
            rows.append(
                {
                    "idx": idx,
                    "op": node.op_type,
                    "node": node.name,
                    "out": out,
                    "shape": dims,
                    "numel": _numel(dims),
                }
            )
    rows.sort(key=lambda item: item["idx"])
    for item in rows[:max_rows]:
        print(json.dumps(item, ensure_ascii=False))
    if len(rows) > max_rows:
        print(json.dumps({"more": len(rows) - max_rows}, ensure_ascii=False))


def extract_and_verify(path: Path, input_names: list[str], output_name: str, out_dir: Path) -> Path:
    prepared = _ensure_shapes(path, out_dir / "tail.shaped.onnx")
    extracted = out_dir / "tail-prefix.onnx"
    utils.extract_model(
        str(prepared),
        str(extracted),
        input_names,
        [output_name],
        check_model=True,
    )
    verify_ort(extracted)
    return extracted


def sanitize_io(path: Path, out_path: Path) -> Path:
    model = onnx.load(str(path))
    rename = {}
    for idx, item in enumerate(model.graph.input):
        rename[item.name] = f"input_{idx}"
        item.name = rename[item.name]
    for idx, item in enumerate(model.graph.output):
        rename[item.name] = f"output_{idx}"
        item.name = rename[item.name]
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name in rename:
                node.input[idx] = rename[name]
        for idx, name in enumerate(node.output):
            if name in rename:
                node.output[idx] = rename[name]
    for item in model.graph.value_info:
        if item.name in rename:
            item.name = rename[item.name]
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def verify_ort(path: Path) -> dict:
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {}
    rng = np.random.default_rng(0)
    for item in sess.get_inputs():
        dims = [int(dim) for dim in item.shape]
        feed[item.name] = rng.standard_normal(dims).astype(np.float32)
    t0 = time.perf_counter()
    outputs = sess.run(None, feed)
    elapsed = time.perf_counter() - t0
    result = {
        "elapsed_s": elapsed,
        "outputs": [
            {
                "shape": list(out.shape),
                "mean": float(np.mean(out)),
                "std": float(np.std(out)),
            }
            for out in outputs
        ],
    }
    print(json.dumps({"ort": result}, indent=2))
    return result


def convert(path: Path, output_rknn: Path, target: str) -> None:
    from rknn.api import RKNN

    output_rknn.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=target, optimization_level=0)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(model=str(path))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")
        ret = rknn.export_rknn(str(output_rknn))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-onnx", required=True, type=Path)
    parser.add_argument("--pattern", default=r"/decoder/generator")
    parser.add_argument("--max-rows", type=int, default=120)
    parser.add_argument("--output-name", default=None)
    parser.add_argument(
        "--input-name",
        action="append",
        default=None,
        help="Input tensor for extracted subgraph. Repeatable. Defaults to decoder-front output and style slice.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/kokoro-tail-probe"))
    parser.add_argument("--target", default="rk3588", choices=["rk3576", "rk3588"])
    parser.add_argument("--convert-rknn", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_name is None:
        list_candidates(args.tail_onnx, args.pattern, args.max_rows)
        return 0

    input_names = args.input_name or [FRONT_OUTPUT, STYLE_SLICE]
    extracted = extract_and_verify(args.tail_onnx, input_names, args.output_name, args.out_dir)
    result = {"tail_prefix_onnx": str(extracted), "input_names": input_names, "output_name": args.output_name}
    if args.convert_rknn:
        rknn_path = args.out_dir / args.target / "tail-prefix.rknn"
        rknn_onnx = sanitize_io(extracted, args.out_dir / "tail-prefix.rknn-io.onnx")
        convert(rknn_onnx, rknn_path, args.target)
        result["rknn"] = str(rknn_path)
    (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

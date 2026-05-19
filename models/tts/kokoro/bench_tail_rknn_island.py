#!/usr/bin/env python3
"""Benchmark a CPU/RKNN/CPU island split inside Kokoro generator tail."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, utils

TAIL_INPUT = "/decoder/decode.3/Mul_output_0"
STYLE_INPUT = "/Slice_2_output_0"
ISLAND_INPUT = "/decoder/generator/LeakyRelu_1_output_0"
ISLAND_OUTPUT = "/decoder/generator/ups.1/ConvTranspose_output_0"
TAIL_OUTPUT = "audio"


def _shape_from_model(model: onnx.ModelProto, name: str) -> list[int]:
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in coll:
            if item.name != name:
                continue
            dims = []
            for dim in item.type.tensor_type.shape.dim:
                value = dim.dim_value
                if value <= 0:
                    raise ValueError(f"missing static shape for {name}")
                dims.append(int(value))
            return dims
    raise KeyError(name)


def _prepare_tail(tail_onnx: Path, work_dir: Path) -> tuple[Path, dict[str, list[int]]]:
    model = onnx.load(str(tail_onnx))
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {
        TAIL_INPUT: _shape_from_model(model, TAIL_INPUT),
        STYLE_INPUT: _shape_from_model(model, STYLE_INPUT),
        ISLAND_INPUT: _shape_from_model(model, ISLAND_INPUT),
        ISLAND_OUTPUT: _shape_from_model(model, ISLAND_OUTPUT),
        TAIL_OUTPUT: _shape_from_model(model, TAIL_OUTPUT),
    }
    existing = {v.name for coll in (model.graph.input, model.graph.output, model.graph.value_info) for v in coll}
    for name in (ISLAND_INPUT, ISLAND_OUTPUT):
        if name not in existing:
            model.graph.value_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name]))
    prepared = work_dir / "tail.with-island-value-info.onnx"
    onnx.checker.check_model(model)
    onnx.save(model, str(prepared))
    return prepared, shapes


def _extract_models(tail_onnx: Path, work_dir: Path) -> tuple[Path, Path, dict[str, list[int]]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared, shapes = _prepare_tail(tail_onnx, work_dir)
    pre = work_dir / "tail-pre.onnx"
    post = work_dir / "tail-post.onnx"
    utils.extract_model(
        str(prepared),
        str(pre),
        [TAIL_INPUT, STYLE_INPUT],
        [ISLAND_INPUT],
        check_model=True,
    )
    utils.extract_model(
        str(prepared),
        str(post),
        [ISLAND_OUTPUT, STYLE_INPUT],
        [TAIL_OUTPUT],
        check_model=True,
    )
    return pre, post, shapes


def _make_session(path: Path, intra: int) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = intra
    opt.inter_op_num_threads = 1
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def _rknn_run(rknn, x: np.ndarray) -> np.ndarray:
    outputs = rknn.inference(inputs=[x])
    if not outputs:
        raise RuntimeError("RKNN island inference returned None")
    return np.asarray(outputs[0], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-onnx", required=True, type=Path)
    parser.add_argument("--island-rknn", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--intra-op", type=int, default=4)
    args = parser.parse_args()

    from rknnlite.api import RKNNLite

    pre_path, post_path, shapes = _extract_models(args.tail_onnx, args.work_dir)
    full_sess = _make_session(args.tail_onnx, args.intra_op)
    pre_sess = _make_session(pre_path, args.intra_op)
    post_sess = _make_session(post_path, args.intra_op)
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(args.island_rknn))
    if ret != 0:
        raise RuntimeError(f"load_rknn returned {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime returned {ret}")

    rng = np.random.default_rng(0)
    hidden = rng.standard_normal(shapes[TAIL_INPUT]).astype(np.float32)
    style = rng.standard_normal(shapes[STYLE_INPUT]).astype(np.float32)
    full_feed = {TAIL_INPUT: hidden, STYLE_INPUT: style}
    pre_feed = {TAIL_INPUT: hidden, STYLE_INPUT: style}

    full_times = []
    split_times = []
    parts = []
    full_out = None
    split_out = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        full_out = full_sess.run(None, full_feed)[0]
        full_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        island_in = pre_sess.run(None, pre_feed)[0]
        pre_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        island_out = _rknn_run(rknn, island_in)
        rknn_s = time.perf_counter() - t1
        t2 = time.perf_counter()
        split_out = post_sess.run(None, {ISLAND_OUTPUT: island_out, STYLE_INPUT: style})[0]
        post_s = time.perf_counter() - t2
        split_times.append(pre_s + rknn_s + post_s)
        parts.append({"pre_ms": pre_s * 1000, "rknn_ms": rknn_s * 1000, "post_ms": post_s * 1000})

    rknn.release()
    assert full_out is not None and split_out is not None
    diff = split_out.astype(np.float32) - full_out.astype(np.float32)
    result = {
        "shapes": shapes,
        "runs": args.runs,
        "full_ms": [round(v * 1000, 3) for v in full_times],
        "split_ms": [round(v * 1000, 3) for v in split_times],
        "full_avg_ms": round(sum(full_times) / len(full_times) * 1000, 3),
        "split_avg_ms": round(sum(split_times) / len(split_times) * 1000, 3),
        "parts_avg_ms": {
            key: round(sum(item[key] for item in parts) / len(parts), 3)
            for key in ("pre_ms", "rknn_ms", "post_ms")
        },
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / max(np.linalg.norm(full_out.reshape(-1)), 1e-12)),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify MOSS codec layer pipeline with RKNN front/suffix and ORT middle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


def _parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return layers


def _shape_dtype_map(model: onnx.ModelProto) -> dict[str, tuple[list[int], int]]:
    rows: dict[str, tuple[list[int], int]] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                break
            dims.append(int(dim.dim_value))
        else:
            rows[value.name] = (dims, int(tensor_type.elem_type))
    return rows


def _dtype(elem_type: int) -> np.dtype[Any]:
    if elem_type == onnx.TensorProto.INT64:
        return np.dtype("int64")
    if elem_type == onnx.TensorProto.INT32:
        return np.dtype("int32")
    return np.dtype("float32")


def _make_ort_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _make_feeds(session: ort.InferenceSession, model: onnx.ModelProto, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = _shape_dtype_map(model)
    feeds: dict[str, np.ndarray] = {}
    for index, item in enumerate(session.get_inputs()):
        shape, elem_type = shapes[item.name]
        dtype = _dtype(elem_type)
        if np.issubdtype(dtype, np.integer):
            value = np.zeros(shape, dtype=dtype)
            if "positions" in item.name:
                value.fill(-1)
            feeds[item.name] = value
        else:
            feeds[item.name] = rng.normal(0.0, 0.2, size=shape).astype(np.float32) + np.float32(index * 0.01)
    return feeds


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    diff = got - ref
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    ref_norm = float(np.linalg.norm(ref_flat))
    got_norm = float(np.linalg.norm(got_flat))
    denom = ref_norm + 1e-12
    return {
        "shape": list(got.shape),
        "dtype": str(got.dtype),
        "finite": bool(np.isfinite(got).all()),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom) if diff.size else 0.0,
        "cosine": float(np.dot(ref_flat, got_flat) / ((ref_norm * got_norm) + 1e-12)) if diff.size else 1.0,
        "ref_min": float(np.min(ref)) if ref.size else 0.0,
        "ref_max": float(np.max(ref)) if ref.size else 0.0,
        "got_min": float(np.min(got)) if got.size else 0.0,
        "got_max": float(np.max(got)) if got.size else 0.0,
    }


class _RknnSession:
    def __init__(self, path: Path) -> None:
        from rknnlite.api import RKNNLite

        self.path = path
        self._rknn = RKNNLite(verbose=False)
        started = time.perf_counter()
        self.load_ret = self._rknn.load_rknn(str(path))
        self.load_ms = (time.perf_counter() - started) * 1000.0
        if self.load_ret != 0:
            raise RuntimeError(f"load_rknn({path}) returned {self.load_ret}")
        started = time.perf_counter()
        self.init_ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
        self.init_ms = (time.perf_counter() - started) * 1000.0
        if self.init_ret != 0:
            raise RuntimeError(f"init_runtime({path}) returned {self.init_ret}")

    def run(self, inputs: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
        started = time.perf_counter()
        outputs = self._rknn.inference(inputs=[np.asarray(item, dtype=np.float32) for item in inputs])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if outputs is None:
            raise RuntimeError(f"rknn.inference({self.path}) returned None")
        return [np.asarray(output) for output in outputs], elapsed_ms

    def release(self) -> None:
        try:
            self._rknn.release()
        except Exception:
            pass


def _run_ort(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    outputs = session.run(None, feeds)
    return [np.asarray(output) for output in outputs], (time.perf_counter() - started) * 1000.0


def verify_layer(
    layer: int,
    full_layer_onnx: Path,
    middle_onnx: Path,
    front_rknn: Path,
    suffix_rknn: Path,
    seed: int,
    threads: int,
    max_hidden_rel_l2: float,
    max_hidden_abs: float,
    min_hidden_cosine: float,
) -> dict[str, Any]:
    full_model = onnx.load(str(full_layer_onnx), load_external_data=True)
    full_sess = _make_ort_session(full_layer_onnx, threads)
    middle_sess = _make_ort_session(middle_onnx, threads)
    full_feeds = _make_feeds(full_sess, full_model, seed + layer)
    full_outputs, full_ms = _run_ort(full_sess, full_feeds)

    front = _RknnSession(front_rknn)
    suffix = None
    try:
        front_outputs, front_ms = front.run([full_feeds[full_sess.get_inputs()[0].name]])
        middle_feeds = {middle_sess.get_inputs()[0].name: front_outputs[0]}
        for item in middle_sess.get_inputs()[1:]:
            middle_feeds[item.name] = full_feeds[item.name]
        middle_outputs, middle_ms = _run_ort(middle_sess, middle_feeds)
        suffix = _RknnSession(suffix_rknn)
        suffix_outputs, suffix_ms = suffix.run([middle_outputs[0], full_feeds[full_sess.get_inputs()[0].name]])
    finally:
        front.release()
        if suffix is not None:
            suffix.release()

    split_outputs = [suffix_outputs[0], *middle_outputs[1:]]
    output_names = [output.name for output in full_sess.get_outputs()]
    metrics = {name: _metrics(ref, got) for name, ref, got in zip(output_names, full_outputs, split_outputs, strict=True)}
    hidden_name = output_names[0]
    hidden_metrics = metrics[hidden_name]
    cache_passed = all(bool(metrics[name]["finite"]) and float(metrics[name]["max_abs"]) == 0.0 for name in output_names[1:])
    hidden_passed = (
        bool(hidden_metrics["finite"])
        and float(hidden_metrics["rel_l2"]) <= max_hidden_rel_l2
        and float(hidden_metrics["max_abs"]) <= max_hidden_abs
        and float(hidden_metrics["cosine"]) >= min_hidden_cosine
    )
    return {
        "layer": layer,
        "artifacts": {
            "full_layer_onnx": str(full_layer_onnx),
            "middle_onnx": str(middle_onnx),
            "front_rknn": str(front_rknn),
            "suffix_rknn": str(suffix_rknn),
        },
        "latency_ms": {
            "full_ort": round(full_ms, 3),
            "front_rknn_load": round(front.load_ms, 3),
            "front_rknn_init": round(front.init_ms, 3),
            "front_rknn_infer": round(front_ms, 3),
            "middle_ort": round(middle_ms, 3),
            "suffix_rknn_load": round(suffix.load_ms if suffix is not None else 0.0, 3),
            "suffix_rknn_init": round(suffix.init_ms if suffix is not None else 0.0, 3),
            "suffix_rknn_infer": round(suffix_ms, 3),
            "hybrid_infer_total": round(front_ms + middle_ms + suffix_ms, 3),
        },
        "metrics": metrics,
        "gates": {
            "max_hidden_rel_l2": max_hidden_rel_l2,
            "max_hidden_abs": max_hidden_abs,
            "min_hidden_cosine": min_hidden_cosine,
            "cache_exact": True,
        },
        "passed": bool(hidden_passed and cache_passed),
    }


def verify_layers(
    full_layer_dir: Path,
    middle_dir: Path,
    front_rknn_dir: Path,
    suffix_rknn_dir: Path,
    layers: list[int],
    seed: int,
    threads: int,
    max_hidden_rel_l2: float,
    max_hidden_abs: float,
    min_hidden_cosine: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for layer in layers:
        item: dict[str, Any] = {"layer": layer}
        try:
            item.update(
                verify_layer(
                    layer=layer,
                    full_layer_onnx=full_layer_dir / f"codec_layer{layer}_full.onnx",
                    middle_onnx=middle_dir / f"codec_middle_layer{layer}_attention.onnx",
                    front_rknn=front_rknn_dir / f"codec_front_layer{layer}_qkv.fp16.rk3576.rknn",
                    suffix_rknn=suffix_rknn_dir / f"codec_suffix_layer{layer}_outproj_ffn.fp16.rk3576.rknn",
                    seed=seed,
                    threads=threads,
                    max_hidden_rel_l2=max_hidden_rel_l2,
                    max_hidden_abs=max_hidden_abs,
                    min_hidden_cosine=min_hidden_cosine,
                )
            )
        except Exception as exc:
            item["passed"] = False
            item["error"] = str(exc)
        results.append(item)
    return {
        "full_layer_dir": str(full_layer_dir),
        "middle_dir": str(middle_dir),
        "front_rknn_dir": str(front_rknn_dir),
        "suffix_rknn_dir": str(suffix_rknn_dir),
        "layers": layers,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "passed": all(bool(item.get("passed")) for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-layer-dir", required=True, type=Path)
    parser.add_argument("--middle-dir", required=True, type=Path)
    parser.add_argument("--front-rknn-dir", required=True, type=Path)
    parser.add_argument("--suffix-rknn-dir", required=True, type=Path)
    parser.add_argument("--layers", default="0-11")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-hidden-rel-l2", type=float, default=0.08)
    parser.add_argument("--max-hidden-abs", type=float, default=0.35)
    parser.add_argument("--min-hidden-cosine", type=float, default=0.995)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_layers(
        full_layer_dir=args.full_layer_dir,
        middle_dir=args.middle_dir,
        front_rknn_dir=args.front_rknn_dir,
        suffix_rknn_dir=args.suffix_rknn_dir,
        layers=_parse_layers(args.layers),
        seed=args.seed,
        threads=args.threads,
        max_hidden_rel_l2=args.max_hidden_rel_l2,
        max_hidden_abs=args.max_hidden_abs,
        min_hidden_cosine=args.min_hidden_cosine,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

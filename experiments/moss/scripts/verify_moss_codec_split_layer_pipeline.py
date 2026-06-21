#!/usr/bin/env python3
"""Verify MOSS codec front/middle/suffix layer pipeline parity against ONNX."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

from build_moss_codec_front_islands import CodecFrontSpec, discover_codec_front_specs
from build_moss_codec_middle_bridges import CodecMiddleSpec, discover_codec_middle_specs
from build_moss_codec_suffix_islands import CodecSuffixSpec, discover_codec_suffix_specs
from models.tts.moss.convert_moss_rknn import sha256_file


@dataclass(frozen=True)
class CodecLayerPipelineSpec:
    layer: int
    front: CodecFrontSpec
    middle: CodecMiddleSpec
    suffix: CodecSuffixSpec

    @property
    def full_inputs(self) -> list[str]:
        return [self.front.input, *self.middle.inputs[1:]]

    @property
    def full_outputs(self) -> list[str]:
        return [self.suffix.output, *self.middle.outputs[1:]]


def discover_pipeline_specs(model: onnx.ModelProto) -> list[CodecLayerPipelineSpec]:
    front = {spec.layer: spec for spec in discover_codec_front_specs(model)}
    middle = {spec.layer: spec for spec in discover_codec_middle_specs(model)}
    suffix = {spec.layer: spec for spec in discover_codec_suffix_specs(model)}
    layers = sorted(set(front) & set(middle) & set(suffix))
    return [CodecLayerPipelineSpec(layer=layer, front=front[layer], middle=middle[layer], suffix=suffix[layer]) for layer in layers]


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
            values = rng.normal(0.0, 0.2, size=shape).astype(np.float32)
            feeds[item.name] = values + np.float32(index * 0.01)
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


def _extract_full_layer(source: Path, out_path: Path, spec: CodecLayerPipelineSpec) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extract_model(
        str(source),
        str(out_path),
        input_names=spec.full_inputs,
        output_names=spec.full_outputs,
        check_model=True,
    )
    return out_path


def _session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _run_session(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> tuple[list[np.ndarray], float]:
    started = time.perf_counter()
    outputs = session.run(None, feeds)
    return outputs, (time.perf_counter() - started) * 1000.0


def verify_layer(
    spec: CodecLayerPipelineSpec,
    full_layer_onnx: Path,
    front_onnx: Path,
    middle_onnx: Path,
    suffix_onnx: Path,
    seed: int,
    threads: int,
    max_rel_l2: float,
    max_abs: float,
    min_cosine: float,
) -> dict[str, Any]:
    full_model = onnx.load(str(full_layer_onnx), load_external_data=True)
    full_sess = _session(full_layer_onnx, threads)
    front_sess = _session(front_onnx, threads)
    middle_sess = _session(middle_onnx, threads)
    suffix_sess = _session(suffix_onnx, threads)

    full_feeds = _make_feeds(full_sess, full_model, seed + spec.layer)
    full_outputs, full_ms = _run_session(full_sess, full_feeds)

    front_outputs, front_ms = _run_session(front_sess, {front_sess.get_inputs()[0].name: full_feeds[spec.front.input]})
    middle_inputs = {middle_sess.get_inputs()[0].name: front_outputs[0]}
    for item in middle_sess.get_inputs()[1:]:
        middle_inputs[item.name] = full_feeds[item.name]
    middle_outputs, middle_ms = _run_session(middle_sess, middle_inputs)
    suffix_inputs = {
        suffix_sess.get_inputs()[0].name: middle_outputs[0],
        suffix_sess.get_inputs()[1].name: full_feeds[spec.suffix.inputs[1]],
    }
    suffix_outputs, suffix_ms = _run_session(suffix_sess, suffix_inputs)

    split_outputs = [suffix_outputs[0], *middle_outputs[1:]]
    names = [output.name for output in full_sess.get_outputs()]
    metrics = {name: _metrics(ref, got) for name, ref, got in zip(names, full_outputs, split_outputs, strict=True)}
    passed = all(
        bool(item["finite"])
        and float(item["rel_l2"]) <= max_rel_l2
        and float(item["max_abs"]) <= max_abs
        and float(item["cosine"]) >= min_cosine
        for item in metrics.values()
    )
    return {
        "layer": spec.layer,
        "spec": {
            "full_inputs": spec.full_inputs,
            "full_outputs": spec.full_outputs,
            "front": asdict(spec.front),
            "middle": asdict(spec.middle),
            "suffix": asdict(spec.suffix),
        },
        "artifacts": {
            "full_layer_onnx": str(full_layer_onnx),
            "front_onnx": str(front_onnx),
            "middle_onnx": str(middle_onnx),
            "suffix_onnx": str(suffix_onnx),
        },
        "latency_ms": {
            "full": round(full_ms, 3),
            "front": round(front_ms, 3),
            "middle": round(middle_ms, 3),
            "suffix": round(suffix_ms, 3),
            "split_total": round(front_ms + middle_ms + suffix_ms, 3),
        },
        "metrics": metrics,
        "passed": passed,
    }


def verify_pipeline_layers(
    source_onnx: Path,
    front_dir: Path,
    middle_dir: Path,
    suffix_dir: Path,
    out_dir: Path,
    layers: list[int],
    seed: int,
    threads: int,
    max_rel_l2: float,
    max_abs: float,
    min_cosine: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    shaped_path = out_dir / "_fixed_onnx" / "codec_split_pipeline_source.shape_inferred.onnx"
    shaped_path.parent.mkdir(parents=True, exist_ok=True)
    shaped = onnx.shape_inference.infer_shapes(onnx.load(str(source_onnx), load_external_data=True))
    onnx.save_model(
        shaped,
        str(shaped_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=shaped_path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )

    specs = discover_pipeline_specs(shaped)
    selected = [spec for spec in specs if spec.layer in set(layers)]
    results: list[dict[str, Any]] = []
    for spec in selected:
        full_layer_onnx = out_dir / "_fixed_onnx" / f"codec_layer{spec.layer}_full.onnx"
        front_onnx = front_dir / "_fixed_onnx" / f"codec_front_layer{spec.layer}_qkv.onnx"
        middle_onnx = middle_dir / f"codec_middle_layer{spec.layer}_attention.onnx"
        suffix_onnx = suffix_dir / "_fixed_onnx" / f"codec_suffix_layer{spec.layer}_outproj_ffn.onnx"
        item: dict[str, Any] = {"layer": spec.layer}
        try:
            for path in (front_onnx, middle_onnx, suffix_onnx):
                if not path.exists():
                    raise FileNotFoundError(path)
            _extract_full_layer(shaped_path, full_layer_onnx, spec)
            item.update(
                verify_layer(
                    spec=spec,
                    full_layer_onnx=full_layer_onnx,
                    front_onnx=front_onnx,
                    middle_onnx=middle_onnx,
                    suffix_onnx=suffix_onnx,
                    seed=seed,
                    threads=threads,
                    max_rel_l2=max_rel_l2,
                    max_abs=max_abs,
                    min_cosine=min_cosine,
                )
            )
            item["full_layer_sha256"] = sha256_file(full_layer_onnx)
        except Exception as exc:
            item["passed"] = False
            item["error"] = str(exc)
        results.append(item)

    return {
        "source_onnx": str(source_onnx),
        "front_dir": str(front_dir),
        "middle_dir": str(middle_dir),
        "suffix_dir": str(suffix_dir),
        "out_dir": str(out_dir),
        "layers": layers,
        "available_layers": [spec.layer for spec in specs],
        "gates": {
            "max_rel_l2": max_rel_l2,
            "max_abs": max_abs,
            "min_cosine": min_cosine,
        },
        "elapsed_s": round(time.perf_counter() - started, 3),
        "passed": all(bool(item.get("passed")) for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-onnx", required=True, type=Path)
    parser.add_argument("--front-dir", required=True, type=Path)
    parser.add_argument("--middle-dir", required=True, type=Path)
    parser.add_argument("--suffix-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default="0-11")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-rel-l2", type=float, default=1e-5)
    parser.add_argument("--max-abs", type=float, default=1e-4)
    parser.add_argument("--min-cosine", type=float, default=0.999999)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_pipeline_layers(
        source_onnx=args.source_onnx,
        front_dir=args.front_dir,
        middle_dir=args.middle_dir,
        suffix_dir=args.suffix_dir,
        out_dir=args.out_dir,
        layers=_parse_layers(args.layers),
        seed=args.seed,
        threads=args.threads,
        max_rel_l2=args.max_rel_l2,
        max_abs=args.max_abs,
        min_cosine=args.min_cosine,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

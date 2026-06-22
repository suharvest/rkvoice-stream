#!/usr/bin/env python3
"""Extract MOSS codec RoPE/cache/attention CPU bridge ONNX subgraphs."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx.utils import extract_model

from models.tts.moss.convert_moss_rknn import sha256_file


@dataclass(frozen=True)
class CodecMiddleSpec:
    layer: int
    front_output: str
    inputs: list[str]
    outputs: list[str]


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            producers[output] = node
    return producers


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _single_consumer(consumers: dict[str, list[onnx.NodeProto]], tensor: str, *, op_type: str | None = None) -> onnx.NodeProto:
    matches = consumers.get(tensor, [])
    if op_type is not None:
        matches = [node for node in matches if node.op_type == op_type]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one consumer for {tensor!r}, found {len(matches)}")
    return matches[0]


def _in_proj_layer_index(name: str) -> int | None:
    if name == "/in_proj/MatMul":
        return 0
    match = re.fullmatch(r"/in_proj_(\d+)/MatMul", name)
    if not match:
        return None
    return int(match.group(1))


def _discover_front_outputs(model: onnx.ModelProto) -> dict[int, str]:
    consumers = _consumer_map(model)
    rows: dict[int, str] = {}
    for node in model.graph.node:
        layer = _in_proj_layer_index(node.name)
        if layer is None or node.op_type != "MatMul":
            continue
        reshape = _single_consumer(consumers, node.output[0], op_type="Reshape")
        transpose = _single_consumer(consumers, reshape.output[0], op_type="Transpose")
        rows[layer] = transpose.output[0]
    return rows


def _discover_attention_outputs(model: onnx.ModelProto) -> dict[int, str]:
    outputs: dict[int, str] = {}
    for node in model.graph.node:
        match = re.fullmatch(r"/out_proj_(\d+)/MatMul", node.name)
        if match and node.op_type == "MatMul":
            outputs[len(outputs)] = node.input[0]
    return outputs


def discover_codec_middle_specs(model: onnx.ModelProto) -> list[CodecMiddleSpec]:
    front_outputs = _discover_front_outputs(model)
    attention_outputs = _discover_attention_outputs(model)
    specs: list[CodecMiddleSpec] = []
    for layer in sorted(front_outputs):
        if layer not in attention_outputs:
            raise RuntimeError(f"Cannot find attention output for codec layer {layer}")
        inputs = [
            front_outputs[layer],
            f"attn_offset_{layer}",
            f"attn_cached_keys_{layer}",
            f"attn_cached_values_{layer}",
            f"attn_cached_positions_{layer}",
        ]
        outputs = [
            attention_outputs[layer],
            f"attn_offset_out_{layer}",
            f"attn_cached_keys_out_{layer}",
            f"attn_cached_values_out_{layer}",
            f"attn_cached_positions_out_{layer}",
        ]
        specs.append(CodecMiddleSpec(layer=layer, front_output=front_outputs[layer], inputs=inputs, outputs=outputs))
    return specs


def _extract_middle_onnx(source: Path, output_path: Path, spec: CodecMiddleSpec) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_model(
        str(source),
        str(output_path),
        input_names=spec.inputs,
        output_names=spec.outputs,
        check_model=True,
    )
    return output_path


def _shape_map(model: onnx.ModelProto) -> dict[str, tuple[list[int], int]]:
    shapes: dict[str, tuple[list[int], int]] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                ok = False
                break
        if ok:
            shapes[value.name] = (dims, int(tensor_type.elem_type))
    return shapes


def _dtype(elem_type: int) -> np.dtype[Any]:
    if elem_type == onnx.TensorProto.INT64:
        return np.dtype("int64")
    if elem_type == onnx.TensorProto.INT32:
        return np.dtype("int32")
    return np.dtype("float32")


def _verify_ort(path: Path) -> dict[str, Any]:
    import onnxruntime as ort

    model = onnx.load(str(path), load_external_data=True)
    shapes = _shape_map(model)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds: dict[str, np.ndarray] = {}
    for index, item in enumerate(sess.get_inputs()):
        shape, elem_type = shapes[item.name]
        dtype = _dtype(elem_type)
        if np.issubdtype(dtype, np.integer):
            feeds[item.name] = np.zeros(shape, dtype=dtype)
            if "positions" in item.name:
                feeds[item.name].fill(-1)
        else:
            values = np.linspace(-0.25, 0.25, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
            feeds[item.name] = values + np.float32(index * 0.01)
    started = time.perf_counter()
    outputs = sess.run(None, feeds)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "status": "OK",
        "elapsed_ms": round(elapsed_ms, 3),
        "inputs": [{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)} for name, value in feeds.items()],
        "outputs": [
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "finite": bool(np.isfinite(output).all()) if np.issubdtype(output.dtype, np.floating) else True,
            }
            for output in outputs
        ],
    }


def build_codec_middle_bridges(
    onnx_path: Path,
    out_dir: Path,
    layers: list[int],
    verify_ort: bool,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    shaped_path = out_dir / "_fixed_onnx" / "codec_middle_source.shape_inferred.onnx"
    shaped_path.parent.mkdir(parents=True, exist_ok=True)
    shaped = onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path), load_external_data=True))
    onnx.save_model(
        shaped,
        str(shaped_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=shaped_path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )

    specs = discover_codec_middle_specs(shaped)
    selected = [spec for spec in specs if spec.layer in set(layers)]
    results: list[dict[str, Any]] = []
    for spec in selected:
        onnx_out = out_dir / f"codec_middle_layer{spec.layer}_attention.onnx"
        item: dict[str, Any] = {"spec": asdict(spec), "onnx": str(onnx_out)}
        started = time.perf_counter()
        try:
            _extract_middle_onnx(shaped_path, onnx_out, spec)
            item["onnx_size_bytes"] = onnx_out.stat().st_size
            item["onnx_sha256"] = sha256_file(onnx_out)
            if verify_ort:
                item["ort_verify"] = _verify_ort(onnx_out)
            item["status"] = "OK"
        except Exception as exc:
            item["status"] = "FAIL"
            item["error"] = str(exc)
        item["elapsed_s"] = round(time.perf_counter() - started, 3)
        results.append(item)

    return {
        "onnx": str(onnx_path),
        "out_dir": str(out_dir),
        "layers": layers,
        "available_layers": [spec.layer for spec in specs],
        "verify_ort": verify_ort,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "passed": all(item["status"] == "OK" for item in results),
        "results": results,
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default="0-11")
    parser.add_argument("--verify-ort", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = build_codec_middle_bridges(
        onnx_path=args.onnx,
        out_dir=args.out_dir,
        layers=_parse_layers(args.layers),
        verify_ort=args.verify_ort,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

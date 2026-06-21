#!/usr/bin/env python3
"""Extract and build MOSS codec norm1 + QKV projection RKNN front islands."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import onnx
from onnx.utils import extract_model

from models.tts.moss.convert_moss_rknn import convert_onnx, sha256_file


@dataclass(frozen=True)
class CodecFrontSpec:
    layer: int
    in_proj_node: str
    input: str
    output: str


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


def discover_codec_front_specs(model: onnx.ModelProto) -> list[CodecFrontSpec]:
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    rows: list[tuple[int, onnx.NodeProto]] = []
    for node in model.graph.node:
        layer = _in_proj_layer_index(node.name)
        if layer is not None and node.op_type == "MatMul":
            rows.append((layer, node))
    rows.sort(key=lambda item: item[0])

    specs: list[CodecFrontSpec] = []
    for layer, in_proj in rows:
        if len(in_proj.input) < 1 or not in_proj.output:
            raise RuntimeError(f"{in_proj.name} has incomplete inputs/outputs")
        norm = producers.get(in_proj.input[0])
        if norm is None or norm.op_type != "LayerNormalization" or len(norm.input) < 1:
            raise RuntimeError(f"Cannot identify norm1 input for {in_proj.name}")
        reshape = _single_consumer(consumers, in_proj.output[0], op_type="Reshape")
        transpose = _single_consumer(consumers, reshape.output[0], op_type="Transpose")
        specs.append(
            CodecFrontSpec(
                layer=layer,
                in_proj_node=in_proj.name,
                input=norm.input[0],
                output=transpose.output[0],
            )
        )
    return specs


def _extract_front_onnx(source: Path, output_path: Path, spec: CodecFrontSpec) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_model(
        str(source),
        str(output_path),
        input_names=[spec.input],
        output_names=[spec.output],
        check_model=True,
    )
    return output_path


def build_codec_front_islands(
    onnx_path: Path,
    out_dir: Path,
    layers: list[int],
    target: str,
    precision: str,
    force: bool,
    optimization_level: int,
    disable_rules: list[str],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    shaped_path = out_dir / "_fixed_onnx" / "codec_front_source.shape_inferred.onnx"
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

    specs = discover_codec_front_specs(shaped)
    selected = [spec for spec in specs if spec.layer in set(layers)]
    results: list[dict[str, Any]] = []
    for spec in selected:
        onnx_out = out_dir / "_fixed_onnx" / f"codec_front_layer{spec.layer}_qkv.onnx"
        rknn_out = out_dir / f"codec_front_layer{spec.layer}_qkv.{precision}.{target}.rknn"
        item: dict[str, Any] = {"spec": asdict(spec), "onnx": str(onnx_out), "rknn": str(rknn_out)}
        started = time.perf_counter()
        try:
            _extract_front_onnx(shaped_path, onnx_out, spec)
            item["onnx_size_bytes"] = onnx_out.stat().st_size
            item["onnx_sha256"] = sha256_file(onnx_out)
            item["rknn_build"] = convert_onnx(
                onnx_path=onnx_out,
                rknn_path=rknn_out,
                target=target,
                precision=precision,
                overrides={},
                outputs=None,
                optimization_level=optimization_level,
                disable_rules=disable_rules,
                dataset=None,
                force=force,
                verbose=False,
            )
            item["status"] = item["rknn_build"]["status"]
        except Exception as exc:
            item["status"] = "FAIL"
            item["error"] = str(exc)
        item["elapsed_s"] = round(time.perf_counter() - started, 3)
        results.append(item)

    return {
        "onnx": str(onnx_path),
        "out_dir": str(out_dir),
        "target": target,
        "precision": precision,
        "layers": layers,
        "available_layers": [spec.layer for spec in specs],
        "disable_rules": disable_rules,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "passed": all(item["status"] in ("OK", "SKIP") for item in results),
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
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--disable-rules", default="merge_conv_channel_inner_perm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = build_codec_front_islands(
        onnx_path=args.onnx,
        out_dir=args.out_dir,
        layers=_parse_layers(args.layers),
        target=args.target,
        precision=args.precision,
        force=args.force,
        optimization_level=args.optimization_level,
        disable_rules=[part.strip() for part in args.disable_rules.split(",") if part.strip()],
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

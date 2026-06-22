#!/usr/bin/env python3
"""Extract and build MOSS codec out-projection + FFN RKNN suffix islands."""

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
class CodecSuffixSpec:
    layer: int
    out_proj_node: str
    inputs: list[str]
    output: str


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    return consumers


def _single_consumer(consumers: dict[str, list[onnx.NodeProto]], tensor: str, *, op_type: str | None = None, name_contains: str | None = None) -> onnx.NodeProto:
    matches = consumers.get(tensor, [])
    if op_type is not None:
        matches = [node for node in matches if node.op_type == op_type]
    if name_contains is not None:
        matches = [node for node in matches if name_contains in node.name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one consumer for {tensor!r}, found {len(matches)}")
    return matches[0]


def _find_downstream_layer_scale_2(consumers: dict[str, list[onnx.NodeProto]], start_tensor: str) -> onnx.NodeProto:
    queue = [start_tensor]
    seen: set[str] = set()
    while queue:
        tensor = queue.pop(0)
        if tensor in seen:
            continue
        seen.add(tensor)
        for node in consumers.get(tensor, []):
            if node.op_type == "Mul" and "layer_scale_2" in node.name:
                return node
            queue.extend(node.output)
    raise RuntimeError(f"Cannot find downstream layer_scale_2 from {start_tensor!r}")


def discover_codec_suffix_specs(model: onnx.ModelProto) -> list[CodecSuffixSpec]:
    consumers = _consumer_map(model)
    specs: list[CodecSuffixSpec] = []
    out_proj_nodes = []
    for node in model.graph.node:
        match = re.fullmatch(r"/out_proj_(\d+)/MatMul", node.name)
        if match and node.op_type == "MatMul":
            out_proj_nodes.append((int(match.group(1)), node))
    out_proj_nodes.sort(key=lambda item: item[0])

    for layer, (_index, out_proj) in enumerate(out_proj_nodes):
        if not out_proj.output:
            raise RuntimeError(f"{out_proj.name} has no output")
        layer_scale_1 = _single_consumer(consumers, out_proj.output[0], op_type="Mul", name_contains="layer_scale_1")
        residual_add = _single_consumer(consumers, layer_scale_1.output[0], op_type="Add")
        residual_inputs = [name for name in residual_add.input if name != layer_scale_1.output[0]]
        if len(residual_inputs) != 1:
            raise RuntimeError(f"Cannot identify residual input for {residual_add.name}")
        layer_scale_2 = _find_downstream_layer_scale_2(consumers, residual_add.output[0])
        final_add = _single_consumer(consumers, layer_scale_2.output[0], op_type="Add")
        specs.append(
            CodecSuffixSpec(
                layer=layer,
                out_proj_node=out_proj.name,
                inputs=[out_proj.input[0], residual_inputs[0]],
                output=final_add.output[0],
            )
        )
    return specs


def _extract_suffix_onnx(source: Path, output_path: Path, spec: CodecSuffixSpec) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract_model(
        str(source),
        str(output_path),
        input_names=spec.inputs,
        output_names=[spec.output],
        check_model=True,
    )
    return output_path


def build_codec_suffix_islands(
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
    shaped_path = out_dir / "_fixed_onnx" / "codec_suffix_source.shape_inferred.onnx"
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

    specs = discover_codec_suffix_specs(shaped)
    selected = [spec for spec in specs if spec.layer in set(layers)]
    results: list[dict[str, Any]] = []
    for spec in selected:
        onnx_out = out_dir / "_fixed_onnx" / f"codec_suffix_layer{spec.layer}_outproj_ffn.onnx"
        rknn_out = out_dir / f"codec_suffix_layer{spec.layer}_outproj_ffn.{precision}.{target}.rknn"
        item: dict[str, Any] = {"spec": asdict(spec), "onnx": str(onnx_out), "rknn": str(rknn_out)}
        started = time.perf_counter()
        try:
            _extract_suffix_onnx(shaped_path, onnx_out, spec)
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

    report = build_codec_suffix_islands(
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

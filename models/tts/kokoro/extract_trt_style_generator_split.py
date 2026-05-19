#!/usr/bin/env python3
"""Extract the TensorRT-style Kokoro generator split from a fixed tail ONNX.

The Jetson TensorRT path uses:
  source:    decoder source -> generator source tensor
  generator: hidden + source + style -> magnitude/phase tensors
  post:      magnitude/phase -> audio

For fixed RKNN buckets the source subgraph often constant-folds, so this script
also writes a source .npy that can be cached at runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import utils

TAIL_INPUT = "/decoder/decode.3/Mul_output_0"
STYLE_INPUT = "/Slice_2_output_0"
SOURCE_OUTPUT = "/decoder/generator/Concat_1_output_0"
MAG_OUTPUT = "/decoder/generator/Slice_1_output_0"
PHASE_OUTPUT = "/decoder/generator/Slice_2_output_0"
TAIL_OUTPUT = "audio"


def _sanitize_io(path: Path, out_path: Path) -> Path:
    model = onnx.load(str(path))
    rename: dict[str, str] = {}
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


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _shape(item) -> list[int]:
    dims = []
    for dim in item.shape:
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"{item.name} has non-static shape: {item.shape}")
        dims.append(dim)
    return dims


def _random_feed(sess: ort.InferenceSession, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        item.name: rng.standard_normal(_shape(item)).astype(np.float32)
        for item in sess.get_inputs()
    }


def _convert_rknn(path: Path, output: Path, target: str) -> None:
    from rknn.api import RKNN

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
        output.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(output))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--convert-rknn", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shaped = args.out_dir / "tail.shaped.onnx"
    model = onnx.load(str(args.tail_onnx))
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(shaped))

    source = args.out_dir / "source.onnx"
    generator = args.out_dir / "generator-rest-preexp.onnx"
    post = args.out_dir / "postspec-istft.onnx"
    utils.extract_model(str(shaped), str(source), [TAIL_INPUT, STYLE_INPUT], [SOURCE_OUTPUT], check_model=True)
    utils.extract_model(
        str(shaped),
        str(generator),
        [TAIL_INPUT, SOURCE_OUTPUT, STYLE_INPUT],
        [MAG_OUTPUT, PHASE_OUTPUT],
        check_model=True,
    )
    utils.extract_model(str(shaped), str(post), [MAG_OUTPUT, PHASE_OUTPUT], [TAIL_OUTPUT], check_model=True)

    source_sess = _session(source)
    source_out = source_sess.run(None, _random_feed(source_sess, seed=1))[0]
    source_npy = args.out_dir / "source.npy"
    np.save(source_npy, source_out.astype(np.float32, copy=False))

    manifest = {
        "source": str(source),
        "source_npy": str(source_npy),
        "generator": str(generator),
        "post": str(post),
        "source_output": SOURCE_OUTPUT,
        "generator_outputs": [MAG_OUTPUT, PHASE_OUTPUT],
        "post_output": TAIL_OUTPUT,
    }
    if args.convert_rknn:
        gen_rknn_onnx = _sanitize_io(generator, args.out_dir / "generator-rest-preexp.rknn-io.onnx")
        gen_rknn = args.out_dir / args.target / "generator-rest-preexp.rknn"
        _convert_rknn(gen_rknn_onnx, gen_rknn, args.target)
        manifest["generator_rknn_onnx"] = str(gen_rknn_onnx)
        manifest["generator_rknn"] = str(gen_rknn)

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

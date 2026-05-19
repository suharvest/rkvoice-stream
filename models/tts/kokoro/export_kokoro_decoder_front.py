#!/usr/bin/env python3
"""Export Kokoro decoder front (pre-generator) to RKNN and CPU tail ONNX.

Inputs:
  /MatMul_1_output_0: [1, seq_len, decoder_width]
  /Slice_2_output_0:  [1, 128]

Output:
  /decoder/decode.3/Mul_output_0: [1, 512, 5232]

This avoids the generator/vocoder tail where RKNN hits REGTASK width limits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import onnx
from onnx import TensorProto, helper, utils

DECODER_INPUT = "/MatMul_1_output_0"
STYLE_SLICE = "/Slice_2_output_0"
FRONT_OUTPUT = "/decoder/decode.3/Mul_output_0"


def _shape_from_value_info(model: onnx.ModelProto, name: str) -> list[int] | None:
    for coll in (model.graph.input, model.graph.output, model.graph.value_info):
        for item in coll:
            if item.name != name:
                continue
            dims = []
            for dim in item.type.tensor_type.shape.dim:
                if dim.dim_value <= 0:
                    return None
                dims.append(dim.dim_value)
            return dims
    return None


def _ensure_value_info(model_path: Path, out_path: Path, seq_len: int, decoder_width: int) -> Path:
    model = onnx.load(str(model_path))
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model
    known = {v.name for v in model.graph.input}
    known.update(v.name for v in model.graph.output)
    known.update(v.name for v in model.graph.value_info)
    additions = {
        DECODER_INPUT: _shape_from_value_info(inferred, DECODER_INPUT) or [1, seq_len, decoder_width],
        STYLE_SLICE: _shape_from_value_info(inferred, STYLE_SLICE) or [1, 128],
        FRONT_OUTPUT: _shape_from_value_info(inferred, FRONT_OUTPUT) or [1, 512, 5232],
    }
    for name, dims in additions.items():
        if name not in known:
            model.graph.value_info.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def _cut_shapes(model_path: Path) -> dict[str, list[int]]:
    model = onnx.load(str(model_path))
    shapes = {}
    for name in (DECODER_INPUT, STYLE_SLICE, FRONT_OUTPUT):
        shape = _shape_from_value_info(model, name)
        if shape is None:
            raise ValueError(f"Missing static shape for {name} in {model_path}")
        shapes[name] = shape
    return shapes


def extract_front(fixed_onnx: Path, out_dir: Path, seq_len: int, decoder_width: int) -> Path:
    prepared = _ensure_value_info(
        fixed_onnx,
        out_dir / "kokoro.with-decoder-front-cut.onnx",
        seq_len,
        decoder_width,
    )
    front = out_dir / "kokoro-decoder-front.onnx"
    utils.extract_model(
        str(prepared),
        str(front),
        [DECODER_INPUT, STYLE_SLICE],
        [FRONT_OUTPUT],
        check_model=True,
    )
    return front


def extract_prefix(fixed_onnx: Path, out_dir: Path, seq_len: int, decoder_width: int) -> Path:
    prepared = _ensure_value_info(
        fixed_onnx,
        out_dir / "kokoro.with-decoder-front-cut.onnx",
        seq_len,
        decoder_width,
    )
    prefix = out_dir / "kokoro-prefix-cpu.onnx"
    utils.extract_model(
        str(prepared),
        str(prefix),
        ["tokens", "style", "speed"],
        [DECODER_INPUT, STYLE_SLICE],
        check_model=True,
    )
    return prefix


def extract_tail(fixed_onnx: Path, out_dir: Path, seq_len: int, decoder_width: int) -> Path:
    prepared = _ensure_value_info(
        fixed_onnx,
        out_dir / "kokoro.with-decoder-front-cut.onnx",
        seq_len,
        decoder_width,
    )
    tail = out_dir / "kokoro-generator-tail-cpu.onnx"
    utils.extract_model(
        str(prepared),
        str(tail),
        [FRONT_OUTPUT, STYLE_SLICE],
        ["audio"],
        check_model=True,
    )
    return tail


def verify_ort(front: Path, shapes: dict[str, list[int]]) -> dict:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal(shapes[DECODER_INPUT]).astype(np.float32)
    s = rng.standard_normal(shapes[STYLE_SLICE]).astype(np.float32)
    sess = ort.InferenceSession(str(front), providers=["CPUExecutionProvider"])
    y = sess.run(None, {DECODER_INPUT: x, STYLE_SLICE: s})[0]
    return {
        "output_shape": list(y.shape),
        "output_mean": float(np.mean(y)),
        "output_std": float(np.std(y)),
    }


def verify_tail_ort(front: Path, tail: Path, shapes: dict[str, list[int]]) -> dict:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal(shapes[DECODER_INPUT]).astype(np.float32)
    s = rng.standard_normal(shapes[STYLE_SLICE]).astype(np.float32)
    front_sess = ort.InferenceSession(str(front), providers=["CPUExecutionProvider"])
    tail_sess = ort.InferenceSession(str(tail), providers=["CPUExecutionProvider"])
    h = front_sess.run(None, {DECODER_INPUT: x, STYLE_SLICE: s})[0]
    audio = tail_sess.run(None, {FRONT_OUTPUT: h, STYLE_SLICE: s})[0]
    return {
        "tail_audio_shape": list(audio.shape),
        "tail_audio_mean": float(np.mean(audio)),
        "tail_audio_std": float(np.std(audio)),
    }


def convert(front: Path, rknn_path: Path, target: str, shapes: dict[str, list[int]]) -> None:
    from rknn.api import RKNN

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=target, optimization_level=0)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(
            model=str(front),
            inputs=[DECODER_INPUT, STYLE_SLICE],
            input_size_list=[shapes[DECODER_INPUT], shapes[STYLE_SLICE]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3588", choices=["rk3576", "rk3588"])
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--decoder-width", type=int, default=2616)
    parser.add_argument("--skip-rknn", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prefix = extract_prefix(args.fixed_onnx, args.out_dir, args.seq_len, args.decoder_width)
    front = extract_front(args.fixed_onnx, args.out_dir, args.seq_len, args.decoder_width)
    tail = extract_tail(args.fixed_onnx, args.out_dir, args.seq_len, args.decoder_width)
    shapes = _cut_shapes(args.out_dir / "kokoro.with-decoder-front-cut.onnx")
    result = {
        "target": args.target,
        "seq_len": args.seq_len,
        "decoder_width": args.decoder_width,
        "cut_shapes": shapes,
        "prefix_onnx": str(prefix),
        "front_onnx": str(front),
        "tail_onnx": str(tail),
        "ort": verify_ort(front, shapes),
        "tail_ort": verify_tail_ort(front, tail, shapes),
    }
    if not args.skip_rknn:
        rknn_path = args.out_dir / args.target / "kokoro-decoder-front.rknn"
        convert(front, rknn_path, args.target, shapes)
        result["rknn"] = str(rknn_path)
    result["elapsed_s"] = round(time.time() - t0, 3)
    manifest = args.out_dir / f"manifest-{args.target}.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

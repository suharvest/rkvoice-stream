#!/usr/bin/env python3
"""Split Kokoro fixed ONNX into CPU prefix + RKNN decoder suffix.

The full graph currently fails RKNN constant folding in text_encoder.  This
script extracts:

  prefix: tokens/style/speed -> decoder_input, style_slice
  suffix: decoder_input/style_slice -> audio

Run on WSL2/x86 with onnx + rknn-toolkit2.  Validate any exported RKNN on real
RK3588/RK3576 hardware.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import onnx
from onnx import TensorProto, helper, utils

DECODER_INPUT = "/MatMul_1_output_0"
STYLE_SLICE = "/Slice_2_output_0"
OUTPUT = "audio"


def _ensure_value_info(model_path: Path, tensor_name: str, dims: list[int], out_path: Path) -> Path:
    model = onnx.load(str(model_path))
    known = {v.name for v in model.graph.input}
    known.update(v.name for v in model.graph.output)
    known.update(v.name for v in model.graph.value_info)
    if tensor_name not in known:
        model.graph.value_info.append(
            helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, dims)
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def split(fixed_onnx: Path, out_dir: Path, seq_len: int, audio_len: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = _ensure_value_info(
        fixed_onnx,
        DECODER_INPUT,
        [1, seq_len, audio_len],
        out_dir / "kokoro.with-decoder-cut.onnx",
    )
    prepared = _ensure_value_info(
        prepared,
        STYLE_SLICE,
        [1, 128],
        out_dir / "kokoro.with-decoder-cut.onnx",
    )

    prefix = out_dir / "kokoro-prefix-cpu.onnx"
    suffix = out_dir / "kokoro-decoder-suffix.onnx"

    utils.extract_model(
        str(prepared),
        str(prefix),
        ["tokens", "style", "speed"],
        [DECODER_INPUT, STYLE_SLICE],
        check_model=True,
    )
    utils.extract_model(
        str(prepared),
        str(suffix),
        [DECODER_INPUT, STYLE_SLICE],
        [OUTPUT],
        check_model=True,
    )
    return {"prepared": str(prepared), "prefix": str(prefix), "suffix": str(suffix)}


def verify_ort(prefix: Path, suffix: Path, seq_len: int) -> dict:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    tokens = np.zeros((1, seq_len), dtype=np.int64)
    tokens[0, :5] = [1, 2, 3, 4, 5]
    style = rng.standard_normal((1, 256)).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)
    prefix_sess = ort.InferenceSession(str(prefix), providers=["CPUExecutionProvider"])
    suffix_sess = ort.InferenceSession(str(suffix), providers=["CPUExecutionProvider"])
    dec_in, style_slice = prefix_sess.run(None, {"tokens": tokens, "style": style, "speed": speed})
    audio = suffix_sess.run(None, {DECODER_INPUT: dec_in, STYLE_SLICE: style_slice})[0]
    return {
        "decoder_input_shape": list(dec_in.shape),
        "style_slice_shape": list(style_slice.shape),
        "audio_shape": list(audio.shape),
        "audio_rms": float(np.sqrt(np.mean(audio.astype(np.float32) ** 2))),
    }


def convert_suffix(suffix: Path, rknn_path: Path, target: str, seq_len: int, audio_len: int) -> None:
    from rknn.api import RKNN

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=target, optimization_level=0)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(
            model=str(suffix),
            inputs=[DECODER_INPUT, STYLE_SLICE],
            input_size_list=[[1, seq_len, audio_len], [1, 128]],
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

    t0 = time.time()
    result = {
        "target": args.target,
        "seq_len": args.seq_len,
        "decoder_width": args.decoder_width,
        **split(args.fixed_onnx, args.out_dir, args.seq_len, args.decoder_width),
    }
    result["ort"] = verify_ort(Path(result["prefix"]), Path(result["suffix"]), args.seq_len)

    if not args.skip_rknn:
        rknn_path = args.out_dir / args.target / "kokoro-decoder-suffix.rknn"
        convert_suffix(Path(result["suffix"]), rknn_path, args.target, args.seq_len, args.decoder_width)
        result["rknn"] = str(rknn_path)

    result["elapsed_s"] = round(time.time() - t0, 3)
    manifest = args.out_dir / f"manifest-{args.target}.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

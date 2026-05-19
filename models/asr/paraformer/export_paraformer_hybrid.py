#!/usr/bin/env python3
"""Export Paraformer hybrid RKNN/CPU artifacts.

The validated split keeps the numerically sensitive encoder tail on CPU:

  encoder.onnx -> encoder-rknn.onnx
  encoder-rknn.onnx -> encoder_prefix_to_block30.onnx
  encoder-rknn.onnx -> encoder_suffix_from_block30.onnx
  encoder_prefix_to_block30.onnx -> encoder_prefix_to_block30.<frames>.<precision>.rknn

Run RKNN conversion on x86/WSL2 with rknn-toolkit2 installed.  Always validate
the resulting RKNN on real RK3588/RK3576 hardware with
verify_paraformer_rknn_hybrid.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, utils

from convert_paraformer_rknn import convert_onnx, prepare_decoder_onnx, prepare_encoder_onnx


DEFAULT_CUT = "/encoder/encoders/encoders.30/Add_1_output_0"


def _ensure_value_info(model_path: Path, tensor_name: str, output_path: Path) -> Path:
    """Add value_info for the cut tensor when ONNX shape inference omitted it."""
    model = onnx.load(str(model_path))
    known = {v.name for v in model.graph.value_info}
    known.update(v.name for v in model.graph.output)
    known.update(v.name for v in model.graph.input)
    if tensor_name in known:
        if output_path != model_path:
            shutil.copy2(model_path, output_path)
        return output_path

    producer = next((n for n in model.graph.node if tensor_name in n.output), None)
    if producer is None:
        raise RuntimeError(f"Cut tensor not found in graph outputs: {tensor_name}")
    model.graph.value_info.append(
        helper.make_tensor_value_info(tensor_name, TensorProto.FLOAT, ["batch_size", "enc_length", 512])
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def split_encoder(encoder_onnx: Path, out_dir: Path, cut_tensor: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = _ensure_value_info(encoder_onnx, cut_tensor, out_dir / "encoder-rknn-with-cut.onnx")
    prefix = out_dir / "encoder_prefix_to_block30.onnx"
    suffix = out_dir / "encoder_suffix_from_block30.onnx"

    utils.extract_model(
        str(prepared),
        str(prefix),
        ["speech", "speech_lengths", "encoder_pad_mask", "cif_pad_mask"],
        [cut_tensor],
        check_model=True,
    )
    utils.extract_model(
        str(prepared),
        str(suffix),
        [cut_tensor, "speech_lengths", "encoder_pad_mask", "cif_pad_mask"],
        ["enc", "enc_len", "alphas"],
        check_model=True,
    )
    return prefix, suffix


def verify_ort_split(full_onnx: Path, prefix_onnx: Path, suffix_onnx: Path, cut_tensor: str, frames: int) -> dict:
    import onnxruntime as ort

    speech = np.random.default_rng(0).normal(0, 1, (1, frames, 560)).astype(np.float32)
    speech_len = np.array([frames], dtype=np.int32)
    mask = np.ones((1, frames), dtype=np.float32)
    feeds = {
        "speech": speech,
        "speech_lengths": speech_len,
        "encoder_pad_mask": mask,
        "cif_pad_mask": mask,
    }
    full = ort.InferenceSession(str(full_onnx), providers=["CPUExecutionProvider"])
    prefix = ort.InferenceSession(str(prefix_onnx), providers=["CPUExecutionProvider"])
    suffix = ort.InferenceSession(str(suffix_onnx), providers=["CPUExecutionProvider"])
    full_out = full.run(["enc", "enc_len", "alphas"], feeds)
    cut = prefix.run([cut_tensor], feeds)[0]
    split_out = suffix.run(
        ["enc", "enc_len", "alphas"],
        {
            cut_tensor: cut,
            "speech_lengths": speech_len,
            "encoder_pad_mask": mask,
            "cif_pad_mask": mask,
        },
    )
    return {
        "enc_max_abs": float(np.max(np.abs(split_out[0] - full_out[0]))),
        "enc_len_equal": bool(np.array_equal(split_out[1], full_out[1])),
        "alphas_max_abs": float(np.max(np.abs(split_out[2] - full_out[2]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory with encoder.onnx, decoder.onnx, tokens.txt")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", required=True, choices=["rk3576", "rk3588"])
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    parser.add_argument("--cut-tensor", default=DEFAULT_CUT)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--skip-rknn", action="store_true")
    parser.add_argument("--skip-ort-verify", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder_onnx = prepare_encoder_onnx(args.model_dir / "encoder.onnx", args.out_dir / "encoder-rknn.onnx")
    decoder_onnx = prepare_decoder_onnx(args.model_dir / "decoder.onnx", args.out_dir / "decoder-rknn.onnx")
    prefix_onnx, suffix_onnx = split_encoder(encoder_onnx, args.out_dir, args.cut_tensor)

    if (args.model_dir / "tokens.txt").exists():
        shutil.copy2(args.model_dir / "tokens.txt", args.out_dir / "tokens.txt")

    result = {
        "target": args.target,
        "frames": args.frames,
        "precision": args.precision,
        "cut_tensor": args.cut_tensor,
        "encoder_onnx": str(encoder_onnx),
        "decoder_onnx": str(decoder_onnx),
        "prefix_onnx": str(prefix_onnx),
        "suffix_onnx": str(suffix_onnx),
    }

    if not args.skip_ort_verify:
        result["ort_split"] = verify_ort_split(encoder_onnx, prefix_onnx, suffix_onnx, args.cut_tensor, args.frames)

    if not args.skip_rknn:
        rknn_path = args.out_dir / "rknn" / args.target / f"encoder_prefix_to_block30.{args.frames}.{args.precision}.rknn"
        convert_onnx(
            prefix_onnx,
            rknn_path,
            args.target,
            ["speech", "encoder_pad_mask"],
            [[1, args.frames, 560], [1, args.frames]],
            args.precision,
            args.dataset,
            args.optimization_level,
        )
        result["prefix_rknn"] = str(rknn_path)

    manifest = args.out_dir / f"manifest-{args.target}-{args.frames}-{args.precision}.json"
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

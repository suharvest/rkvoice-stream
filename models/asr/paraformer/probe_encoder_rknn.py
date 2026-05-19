#!/usr/bin/env python3
"""Build and run Paraformer encoder RKNN checkpoint probes.

The probe keeps the original encoder graph but asks RKNN to expose selected
intermediate tensors.  It is used to locate the first block that produces
non-finite values on real RK3588/RK3576 hardware.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from convert_paraformer_rknn import FLOAT_DTYPES, prepare_encoder_onnx


def _checkpoint_outputs(onnx_path: Path, stride: int) -> list[tuple[str, str]]:
    model = onnx.load(str(onnx_path))
    checkpoints: list[tuple[str, str]] = []
    for node in model.graph.node:
        if node.name == "/encoder/encoders0/encoders0.0/Add":
            checkpoints.append(("pre", node.output[0]))
            break

    block_outputs: list[tuple[int, str]] = []
    for node in model.graph.node:
        match = re.match(r"/encoder/encoders/encoders\.(\d+)/Add_1$", node.name)
        if match:
            block_outputs.append((int(match.group(1)), node.output[0]))

    if not block_outputs:
        raise RuntimeError("No /encoder/encoders/encoders.N/Add_1 checkpoints found")
    last_block = max(i for i, _ in block_outputs)
    selected = {0, last_block}
    selected.update(range(stride - 1, last_block + 1, stride))
    for idx, output in sorted(block_outputs):
        if idx in selected:
            checkpoints.append((f"block_{idx}", output))
    checkpoints.append(("enc", "enc"))
    checkpoints.append(("alphas", "alphas"))
    return checkpoints


def _focus_block_outputs(onnx_path: Path, block: int) -> list[tuple[str, str]]:
    model = onnx.load(str(onnx_path))
    wanted_suffixes = [
        "norm1/Add_1",
        "self_attn/linear_q_k_v/Add",
        "self_attn/MatMul",
        "self_attn/Softmax",
        "self_attn/MatMul_1",
        "self_attn/linear_out/Add",
        "self_attn/Add_2",
        "Add",
        "norm2/Add_1",
        "feed_forward/w_1/Add",
        "feed_forward/activation/Relu",
        "feed_forward/w_2/Add",
        "Add_1",
    ]
    prefix = f"/encoder/encoders/encoders.{block}/"
    outputs: list[tuple[str, str]] = []
    for suffix in wanted_suffixes:
        full_name = prefix + suffix
        node = next((n for n in model.graph.node if n.name == full_name), None)
        if node is not None and node.output:
            label = suffix.replace("/", "_")
            outputs.append((f"block_{block}_{label}", node.output[0]))
    if not outputs:
        raise RuntimeError(f"No focus checkpoints found for block {block}")
    outputs.append(("enc", "enc"))
    return outputs


def _encoder_shapes(frames: int) -> tuple[list[str], list[list[int]]]:
    return (
        ["speech", "speech_lengths", "encoder_pad_mask", "cif_pad_mask"],
        [[1, frames, 560], [1], [1, frames], [1, frames]],
    )


def build_probe(args: argparse.Namespace) -> None:
    from rknn.api import RKNN

    model_dir = Path(args.model_dir)
    encoder_onnx = Path(args.encoder_onnx) if args.encoder_onnx else model_dir / "encoder-rknn.onnx"
    if not encoder_onnx.exists():
        encoder_onnx = prepare_encoder_onnx(model_dir / "encoder.onnx", encoder_onnx)

    if args.focus_block >= 0:
        checkpoints = _focus_block_outputs(encoder_onnx, args.focus_block)
    else:
        checkpoints = _checkpoint_outputs(encoder_onnx, args.stride)
    output_names = [name for _, name in checkpoints]
    input_names, input_shapes = _encoder_shapes(args.frames)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rknn_path = out_dir / f"encoder_probe.{args.frames}.{args.precision}.rknn"
    meta_path = out_dir / f"encoder_probe.{args.frames}.{args.precision}.json"
    if rknn_path.exists():
        rknn_path.unlink()

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            target_platform=args.target,
            optimization_level=args.optimization_level,
            float_dtype=FLOAT_DTYPES.get(args.precision, "float16"),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config ret={ret}")
        ret = rknn.load_onnx(
            model=str(encoder_onnx),
            inputs=input_names,
            input_size_list=input_shapes,
            outputs=output_names,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx ret={ret}")
        ret = rknn.build(do_quantization=args.precision == "int8", dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build ret={ret}")
        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn ret={ret}")
    finally:
        rknn.release()

    meta = {
        "target": args.target,
        "frames": args.frames,
        "precision": args.precision,
        "outputs": [{"label": label, "name": name} for label, name in checkpoints],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"rknn": str(rknn_path), "meta": str(meta_path), "outputs": len(checkpoints)}, indent=2))


def _read_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    from rkvoice_stream.backends.asr.paraformer_rknn import SAMPLE_RATE, add_preroll_silence

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        new_len = int(round(duration * SAMPLE_RATE))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return add_preroll_silence(audio.astype(np.float32))


def _features(wav: Path, frames: int) -> tuple[np.ndarray, int]:
    from rkvoice_stream.backends.asr.paraformer_rknn import compute_fbank, stack_frames

    feats = stack_frames(compute_fbank(_read_audio(wav)))
    orig_frames = min(feats.shape[0], frames)
    if feats.shape[0] < frames:
        feats = np.pad(feats, ((0, frames - feats.shape[0]), (0, 0)), mode="edge")
    else:
        feats = feats[:frames]
    return np.ascontiguousarray(feats[np.newaxis, :].astype(np.float32)), orig_frames


def _stats(label: str, arr: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    return {
        "label": label,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(finite.all()),
        "finite_ratio": float(finite.mean()) if arr.size else 1.0,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }


def run_probe(args: argparse.Namespace) -> None:
    from rknnlite.api import RKNNLite

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    frames = int(meta["frames"])
    speech, orig_frames = _features(Path(args.wav), frames)
    speech_len = np.array([orig_frames], dtype=np.int32)
    mask = np.zeros((1, frames), dtype=np.float32)
    mask[:, :orig_frames] = 1.0

    rknn = RKNNLite(verbose=False)
    try:
        ret = rknn.load_rknn(args.rknn)
        if ret != 0:
            raise RuntimeError(f"load_rknn ret={ret}")
        core = getattr(RKNNLite, args.core_mask, RKNNLite.NPU_CORE_AUTO)
        ret = rknn.init_runtime(core_mask=core)
        if ret != 0:
            raise RuntimeError(f"init_runtime ret={ret}")
        outputs = rknn.inference(inputs=[speech, speech_len, mask, mask])
        if outputs is None:
            # RKNN may prune speech_lengths and/or CIF-only masks from probe
            # graphs.  Retry likely remaining static encoder inputs in ONNX
            # order.
            outputs = rknn.inference(inputs=[speech, mask, mask])
        if outputs is None:
            outputs = rknn.inference(inputs=[speech, mask])
        if outputs is None:
            raise RuntimeError("RKNN probe inference failed")
    finally:
        rknn.release()

    labels = [item["label"] for item in meta["outputs"]]
    result = {
        "rknn": args.rknn,
        "wav": args.wav,
        "frames": frames,
        "orig_frames": orig_frames,
        "outputs": [_stats(label, np.asarray(out)) for label, out in zip(labels, outputs)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    build = sub.add_parser("build")
    build.add_argument("--model-dir", required=True)
    build.add_argument("--encoder-onnx", default="")
    build.add_argument("--out-dir", required=True)
    build.add_argument("--target", choices=["rk3576", "rk3588"], required=True)
    build.add_argument("--frames", type=int, default=400)
    build.add_argument("--stride", type=int, default=8)
    build.add_argument("--focus-block", type=int, default=-1)
    build.add_argument("--precision", choices=["fp16", "bf16", "tf32", "int8"], default="fp16")
    build.add_argument("--dataset", default=None)
    build.add_argument("--optimization-level", type=int, default=3)
    build.set_defaults(func=build_probe)

    run = sub.add_parser("run")
    run.add_argument("--rknn", required=True)
    run.add_argument("--meta", required=True)
    run.add_argument("--wav", required=True)
    run.add_argument("--core-mask", default="NPU_CORE_1")
    run.set_defaults(func=run_probe)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Convert streaming Paraformer ONNX models to fixed-shape RKNN artifacts.

This follows the Jetson Paraformer TRT layout but targets Rockchip RKNN:

  encoder.onnx -> encoder.{40,80,160,400}.{fp16|bf16|tf32|int8}.rknn
  decoder.onnx -> decoder.400x40.{fp16|bf16|tf32|int8}.rknn

RKNN has no dynamic TensorRT profile equivalent, so runtime uses fixed buckets.
FP16 is the smallest non-quantized artifact, but Paraformer encoder is sensitive
to overflow and must be checked on real RK3588/RK3576 hardware.  The converter
also exposes BF16/TF32 because RKNN Toolkit accepts those config names, but
support is model/operator dependent.  INT8 is supported only when a
representative dataset is supplied and must be validated on both RK3588 and
RK3576 hardware.

Example on WSL2:

  python3 models/asr/paraformer/convert_paraformer_rknn.py \\
    --model-dir /mnt/d/models/paraformer-streaming \\
    --out-dir /mnt/d/models/paraformer-streaming/rknn \\
    --target rk3588 \\
    --precision fp16
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ENCODER_BUCKETS = (40, 80, 160, 400)
DECODER_ENC_FRAMES = 400
DECODER_TOKENS = 40
CACHE_COUNT = 16
FLOAT_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "tf32": "tfloat32",
}


def _shape_from_dims(dims: list[int]) -> list[int]:
    return [int(d) for d in dims]


def _input_shapes_for_encoder(frames: int) -> tuple[list[str], list[list[int]]]:
    return (
        ["speech", "speech_lengths", "encoder_pad_mask", "cif_pad_mask"],
        [[1, frames, 560], [1], [1, frames], [1, frames]],
    )


def _input_shapes_for_decoder(enc_frames: int, tokens: int) -> tuple[list[str], list[list[int]]]:
    names = ["enc", "acoustic_embeds"]
    shapes = [[1, enc_frames, 512], [1, tokens, 512]]
    for i in range(CACHE_COUNT):
        names.append(f"in_cache_{i}")
        shapes.append([1, 512, 10])
    names.extend(["pad_mask", "enc_pad_mask"])
    shapes.extend([[1, tokens], [1, enc_frames]])
    return names, shapes


def _find_mask_subgraphs(model: onnx.ModelProto):
    mask_nodes = []
    mask1_nodes = []
    for node in model.graph.node:
        if node.name.startswith("/decoder/make_pad_mask/"):
            mask_nodes.append(node)
        elif node.name.startswith("/decoder/make_pad_mask_1/"):
            mask1_nodes.append(node)

    cast3 = next((n for n in mask_nodes if n.name == "/decoder/make_pad_mask/Cast_3"), None)
    cast3_1 = next((n for n in mask1_nodes if n.name == "/decoder/make_pad_mask_1/Cast_3"), None)
    if cast3 is None or cast3_1 is None:
        raise RuntimeError("Could not find Paraformer decoder make_pad_mask Cast_3 nodes")
    return mask_nodes, mask1_nodes, cast3, cast3_1


def _find_consumer(model: onnx.ModelProto, tensor_name: str):
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp == tensor_name:
                return node, idx
    return None, -1


def _remove_graph_input(model: onnx.ModelProto, name: str) -> None:
    for inp in list(model.graph.input):
        if inp.name == name:
            model.graph.input.remove(inp)
            return


def _add_int32_initializer(model: onnx.ModelProto, name: str, value: int) -> None:
    model.graph.initializer.append(
        helper.make_tensor(name, TensorProto.INT32, [1], [int(value)])
    )


def prepare_decoder_onnx(
    input_path: Path,
    output_path: Path,
    enc_frames: int = DECODER_ENC_FRAMES,
    tokens: int = DECODER_TOKENS,
) -> Path:
    """Externalize decoder pad masks, matching the Jetson TRT surgery."""
    if output_path.exists() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return output_path

    model = onnx.load(str(input_path))
    mask_nodes, mask1_nodes, cast3, cast3_1 = _find_mask_subgraphs(model)
    consumer, consumer_idx = _find_consumer(model, cast3.output[0])
    consumer1, consumer1_idx = _find_consumer(model, cast3_1.output[0])
    if consumer is None or consumer1 is None:
        raise RuntimeError("Could not find decoder mask consumers")

    consumer.input[consumer_idx] = "pad_mask"
    consumer1.input[consumer1_idx] = "enc_pad_mask"

    model.graph.input.append(
        helper.make_tensor_value_info("pad_mask", TensorProto.FLOAT, ["batch_size", "token_length"])
    )
    model.graph.input.append(
        helper.make_tensor_value_info("enc_pad_mask", TensorProto.FLOAT, ["batch_size", "enc_length"])
    )

    _remove_graph_input(model, "enc_len")
    _remove_graph_input(model, "acoustic_embeds_len")
    _add_int32_initializer(model, "enc_len", enc_frames)
    _add_int32_initializer(model, "acoustic_embeds_len", tokens)

    for node in mask_nodes + mask1_nodes:
        model.graph.node.remove(node)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def prepare_encoder_onnx(input_path: Path, output_path: Path) -> Path:
    """Externalize encoder pad masks so RKNN sees a static graph."""
    if output_path.exists() and output_path.stat().st_mtime >= input_path.stat().st_mtime:
        return output_path

    model = onnx.load(str(input_path))
    encoder_mask_nodes = [n for n in model.graph.node if n.name.startswith("/encoder/make_pad_mask/")]
    cif_mask_nodes = [n for n in model.graph.node if n.name.startswith("/make_pad_mask/")]

    encoder_cast = next((n for n in encoder_mask_nodes if n.name == "/encoder/make_pad_mask/Cast_3"), None)
    cif_cast = next((n for n in cif_mask_nodes if n.name == "/make_pad_mask/Cast_1"), None)
    if encoder_cast is None or cif_cast is None:
        raise RuntimeError("Could not find Paraformer encoder make_pad_mask output nodes")

    encoder_out = encoder_cast.output[0]
    cif_out = cif_cast.output[0]
    encoder_consumers = []
    cif_consumers = []
    for node in model.graph.node:
        for idx, inp in enumerate(node.input):
            if inp == encoder_out:
                encoder_consumers.append((node, idx))
            elif inp == cif_out:
                cif_consumers.append((node, idx))
    if not encoder_consumers or not cif_consumers:
        raise RuntimeError("Could not find Paraformer encoder mask consumers")

    for node, idx in encoder_consumers:
        node.input[idx] = "encoder_pad_mask"
    for node, idx in cif_consumers:
        node.input[idx] = "cif_pad_mask"

    model.graph.input.append(
        helper.make_tensor_value_info("encoder_pad_mask", TensorProto.FLOAT, ["batch_size", "enc_length"])
    )
    model.graph.input.append(
        helper.make_tensor_value_info("cif_pad_mask", TensorProto.FLOAT, ["batch_size", "enc_length"])
    )

    for node in encoder_mask_nodes + cif_mask_nodes:
        if node in model.graph.node:
            model.graph.node.remove(node)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return output_path


def convert_onnx(
    onnx_path: Path,
    rknn_path: Path,
    target: str,
    input_names: list[str],
    input_shapes: list[list[int]],
    precision: str,
    dataset: str | None,
    optimization_level: int,
) -> None:
    from rknn.api import RKNN

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    if rknn_path.exists():
        rknn_path.unlink()

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(
            target_platform=target,
            optimization_level=optimization_level,
            float_dtype=FLOAT_DTYPES.get(precision, "float16"),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config ret={ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=input_names,
            input_size_list=[_shape_from_dims(s) for s in input_shapes],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx ret={ret}")

        do_quantization = precision == "int8"
        if do_quantization and not dataset:
            raise RuntimeError("INT8 conversion requires --dataset")

        ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build ret={ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn ret={ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert streaming Paraformer ONNX to RKNN")
    parser.add_argument("--model-dir", required=True, help="Directory containing encoder.onnx, decoder.onnx, tokens.txt")
    parser.add_argument("--out-dir", default=None, help="Output RKNN directory; default: <model-dir>/rknn")
    parser.add_argument("--target", default="rk3588", choices=["rk3576", "rk3588"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8", "all"])
    parser.add_argument("--dataset", default=None, help="Representative dataset file for INT8 quantization")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--decoder-onnx", default=None, help="Optional pre-surgeried decoder ONNX path")
    parser.add_argument("--skip-encoder", action="store_true")
    parser.add_argument("--skip-decoder", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir) if args.out_dir else model_dir / "rknn"
    raw_encoder_onnx = model_dir / "encoder.onnx"
    encoder_onnx = model_dir / "encoder-rknn.onnx"
    decoder_onnx = Path(args.decoder_onnx) if args.decoder_onnx else model_dir / "decoder-rknn.onnx"
    raw_decoder_onnx = model_dir / "decoder.onnx"

    if not raw_encoder_onnx.exists() and not encoder_onnx.exists():
        raise FileNotFoundError(raw_encoder_onnx)
    if not raw_decoder_onnx.exists() and not args.skip_decoder and not decoder_onnx.exists():
        raise FileNotFoundError(raw_decoder_onnx)

    precisions = ["fp16", "bf16", "tf32", "int8"] if args.precision == "all" else [args.precision]

    if (model_dir / "tokens.txt").exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        token_dst = out_dir.parent / "tokens.txt"
        if (model_dir / "tokens.txt").resolve() != token_dst.resolve():
            shutil.copy2(model_dir / "tokens.txt", token_dst)

    print("=== Paraformer RKNN conversion ===")
    print(f"Model dir:  {model_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Target:     {args.target}")
    print(f"Precision:  {', '.join(precisions)}")

    for precision in precisions:
        if not args.skip_encoder:
            if not encoder_onnx.exists():
                encoder_onnx = prepare_encoder_onnx(raw_encoder_onnx, encoder_onnx)
            for frames in ENCODER_BUCKETS:
                names, shapes = _input_shapes_for_encoder(frames)
                out_path = out_dir / f"encoder.{frames}.{precision}.rknn"
                print(f"\n[encoder] frames={frames} precision={precision} -> {out_path.name}")
                convert_onnx(
                    encoder_onnx,
                    out_path,
                    args.target,
                    names,
                    shapes,
                    precision,
                    args.dataset,
                    args.optimization_level,
                )

        if not args.skip_decoder:
            if not decoder_onnx.exists():
                decoder_onnx = prepare_decoder_onnx(
                    raw_decoder_onnx,
                    decoder_onnx,
                    DECODER_ENC_FRAMES,
                    DECODER_TOKENS,
                )
            names, shapes = _input_shapes_for_decoder(DECODER_ENC_FRAMES, DECODER_TOKENS)
            out_path = out_dir / f"decoder.{DECODER_ENC_FRAMES}x{DECODER_TOKENS}.{precision}.rknn"
            print(f"\n[decoder] enc={DECODER_ENC_FRAMES} tokens={DECODER_TOKENS} precision={precision} -> {out_path.name}")
            convert_onnx(
                decoder_onnx,
                out_path,
                args.target,
                names,
                shapes,
                precision,
                args.dataset,
                args.optimization_level,
            )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Convert a fixed-shape SenseVoice encoder ONNX to an RK3576/RK3588 RKNN.

SenseVoice-small (lovemefan/SenseVoice-onnx) is an encoder + CTC model. Unlike
Paraformer it has NO separate decoder and NO CIF — a single forward pass yields
[1, T, 25055] CTC logits. The 4 SenseVoice prompt embeddings (language, event,
speech, textnorm) are prepended to the LFR features as the first 4 frames before
the encoder, so the ONNX has a single float32 `speech` input.

RKNN has no dynamic dims, so the ONNX must already be frozen to a fixed sequence
length T_FIXED (see sv_fix_shape.py; speech_lengths folded to a constant). This
converter just runs:

    rknn.config(target_platform=...) -> load_onnx -> build(no quant) -> export

fp16 is the smallest non-quantized artifact and is clean on real RK3576 silicon.
If a future SoC overflows, fall back to int8 (`--precision int8 --dataset ...`).

Self-written (not copied from the AGPL community converter); follows the
documented RKNN-Toolkit2 call sequence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert fixed-shape SenseVoice encoder ONNX to RKNN")
    ap.add_argument("--onnx", required=True, help="fixed-shape encoder ONNX (single 'speech' input)")
    ap.add_argument("--out", required=True, help="output .rknn path")
    ap.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    ap.add_argument("--t-fixed", type=int, default=344)
    ap.add_argument("--dataset", default=None, help="representative dataset txt (int8 only)")
    ap.add_argument("--opt-level", type=int, default=3)
    args = ap.parse_args()

    from rknn.api import RKNN

    float_dtype = {"fp16": "float16", "bf16": "bfloat16", "tf32": "tfloat32"}.get(args.precision, "float16")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform=args.target, optimization_level=args.opt_level,
                          float_dtype=float_dtype)
        if ret != 0:
            raise RuntimeError(f"config ret={ret}")

        ret = rknn.load_onnx(model=args.onnx, inputs=["speech"],
                             input_size_list=[[1, args.t_fixed, 560]])
        if ret != 0:
            raise RuntimeError(f"load_onnx ret={ret}")

        do_quant = args.precision == "int8"
        if do_quant and not args.dataset:
            raise RuntimeError("int8 requires --dataset")
        ret = rknn.build(do_quantization=do_quant, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"build ret={ret}")

        ret = rknn.export_rknn(str(out))
        if ret != 0:
            raise RuntimeError(f"export_rknn ret={ret}")
    finally:
        rknn.release()

    print(f"EXPORTED {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

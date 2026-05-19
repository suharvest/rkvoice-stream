#!/usr/bin/env python3
"""Convert a static ONNX model to RKNN."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--optimization-level", type=int, default=0)
    args = parser.parse_args()

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=args.target, optimization_level=args.optimization_level)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(model=str(args.input))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")
        ret = rknn.build(do_quantization=args.quantize, dataset=args.dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

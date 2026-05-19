#!/usr/bin/env python3
"""Export Kokoro ONNX to RKNN after graph surgery.

Run this on WSL2/x86 where rknn-toolkit2 is installed.  The generated RKNN is
validated on RK3588/RK3576 with rknnlite, not with the toolkit simulator.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _run_fix(input_onnx: Path, fixed_onnx: Path, seq_len: int) -> None:
    script = Path(__file__).with_name("fix_kokoro_rknn.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_onnx),
            "--output",
            str(fixed_onnx),
            "--seq-len",
            str(seq_len),
        ],
        check=True,
    )


def _convert(
    fixed_onnx: Path,
    output_rknn: Path,
    target: str,
    seq_len: int,
    optimization_level: int,
) -> None:
    from rknn.api import RKNN

    output_rknn.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=False)
    try:
        ret = rknn.config(target_platform=target, optimization_level=optimization_level)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")
        ret = rknn.load_onnx(
            model=str(fixed_onnx),
            inputs=["tokens", "style", "speed"],
            input_size_list=[[1, seq_len], [1, 256], [1]],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")
        ret = rknn.export_rknn(str(output_rknn))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Kokoro fixed-shape ONNX to RKNN")
    parser.add_argument("--input", required=True, type=Path, help="Original Kokoro model.onnx")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3588", choices=["rk3576", "rk3588"])
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--optimization-level", type=int, default=0)
    parser.add_argument("--skip-fix", action="store_true", help="Reuse an existing fixed ONNX in out-dir")
    parser.add_argument("--tokens", type=Path, default=None, help="Optional tokens.txt/tokens.json to copy")
    parser.add_argument("--style", type=Path, default=None, help="Optional style .npy to copy")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fixed_onnx = args.out_dir / f"kokoro.seq{args.seq_len}.rknn-ready.onnx"
    output_rknn = args.out_dir / args.target / "kokoro.rknn"

    t0 = time.time()
    if not args.skip_fix:
        _run_fix(args.input.expanduser(), fixed_onnx, args.seq_len)
    if not fixed_onnx.exists():
        raise FileNotFoundError(f"Fixed ONNX not found: {fixed_onnx}")

    _convert(fixed_onnx, output_rknn, args.target, args.seq_len, args.optimization_level)

    copied = {}
    if args.tokens:
        dst = output_rknn.parent / args.tokens.name
        shutil.copy2(args.tokens.expanduser(), dst)
        copied["tokens"] = str(dst)
    if args.style:
        dst = output_rknn.parent / "style.npy"
        shutil.copy2(args.style.expanduser(), dst)
        copied["style"] = str(dst)

    manifest = {
        "target": args.target,
        "seq_len": args.seq_len,
        "optimization_level": args.optimization_level,
        "fixed_onnx": str(fixed_onnx),
        "rknn": str(output_rknn),
        "elapsed_s": round(time.time() - t0, 3),
        **copied,
    }
    manifest_path = output_rknn.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export the MOSS global transformer scaffold with official RKLLM toolkit."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def export_rkllm(
    scaffold_dir: Path,
    custom_config: Path,
    output_path: Path,
    target_platform: str,
    num_npu_core: int,
    max_context: int,
    quantized_dtype: str,
    quantized_algorithm: str,
    dataset: Path | None,
    do_quantization: bool,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "scaffold_dir": str(scaffold_dir.resolve()),
        "custom_config": str(custom_config.resolve()),
        "output_path": str(output_path.resolve()),
        "target_platform": target_platform,
        "num_npu_core": num_npu_core,
        "max_context": max_context,
        "do_quantization": do_quantization,
        "quantized_dtype": quantized_dtype,
        "quantized_algorithm": quantized_algorithm,
        "dataset": str(dataset.resolve()) if dataset else None,
        "device": device,
        "dtype": dtype,
        "loaded": False,
        "built": False,
        "exported": False,
        "ret": {},
        "exception": None,
    }
    try:
        from rkllm.api import RKLLM

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        llm = RKLLM()
        ret = llm.load_huggingface(
            model=str(scaffold_dir.resolve()),
            model_lora=None,
            device=device,
            dtype=dtype,
            custom_config=str(custom_config.resolve()),
            load_weight=True,
        )
        report["ret"]["load_huggingface"] = int(ret) if isinstance(ret, int) else ret
        report["loaded"] = ret == 0
        if ret != 0:
            return report

        build_kwargs: dict[str, Any] = {
            "do_quantization": do_quantization,
            "optimization_level": 1,
            "target_platform": target_platform,
            "num_npu_core": num_npu_core,
            "hybrid_rate": 0,
            "max_context": max_context,
        }
        if do_quantization:
            build_kwargs.update(
                {
                    "quantized_dtype": quantized_dtype,
                    "quantized_algorithm": quantized_algorithm,
                    "extra_qparams": None,
                    "dataset": str(dataset.resolve()) if dataset else None,
                }
            )
        ret = llm.build(**build_kwargs)
        report["ret"]["build"] = int(ret) if isinstance(ret, int) else ret
        report["built"] = ret == 0
        if ret != 0:
            return report

        ret = llm.export_rkllm(str(output_path))
        report["ret"]["export_rkllm"] = int(ret) if isinstance(ret, int) else ret
        report["exported"] = ret == 0 and output_path.exists()
        if output_path.exists():
            report["output_size_bytes"] = output_path.stat().st_size
    except Exception as exc:  # noqa: BLE001 - exporter must report toolkit failures.
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument("--custom-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-platform", default="rk3576")
    parser.add_argument("--num-npu-core", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--quantized-dtype", default="w8a8")
    parser.add_argument("--quantized-algorithm", default="normal")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--do-quantization", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = export_rkllm(
        args.scaffold_dir,
        args.custom_config,
        args.output,
        args.target_platform,
        args.num_npu_core,
        args.max_context,
        args.quantized_dtype,
        args.quantized_algorithm,
        args.dataset,
        args.do_quantization,
        args.device,
        args.dtype,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["exported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

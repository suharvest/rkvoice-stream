#!/usr/bin/env python3
"""Probe whether official RKLLM toolkit can load the MOSS custom HF scaffold."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def probe(scaffold_dir: Path, device: str = "cpu", dtype: str = "float32", custom_config: Path | None = None) -> dict[str, Any]:
    scaffold_dir = scaffold_dir.resolve()
    custom_config = custom_config.resolve() if custom_config else None
    report: dict[str, Any] = {
        "scaffold_dir": str(scaffold_dir),
        "custom_config": str(custom_config) if custom_config else None,
        "device": device,
        "dtype": dtype,
        "loaded": False,
        "ret": None,
        "exception": None,
        "versions": {},
    }
    try:
        import torch
        import transformers
        import rkllm
        from rkllm.api import RKLLM

        report["versions"] = {
            "torch": getattr(torch, "__version__", None),
            "transformers": getattr(transformers, "__version__", None),
            "rkllm": getattr(rkllm, "__version__", None),
        }
        llm = RKLLM()
        ret = llm.load_huggingface(
            model=str(scaffold_dir),
            model_lora=None,
            device=device,
            dtype=dtype,
            custom_config=str(custom_config) if custom_config else None,
            load_weight=True,
        )
        report["ret"] = int(ret) if isinstance(ret, int) else ret
        report["loaded"] = ret == 0
    except Exception as exc:  # noqa: BLE001 - probe must report toolkit failures.
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-12:],
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--custom-config", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = probe(args.scaffold_dir, args.device, args.dtype, args.custom_config)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["loaded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

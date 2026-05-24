#!/usr/bin/env python3
"""Build a batch of MOSS RKNN island artifacts with the single-island extractor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout


def build_batch(
    onnx_path: Path,
    out_dir: Path,
    preset: str,
    layers: list[int],
    seq_len: int,
    target: str,
    precision: str,
    force: bool,
    optimization_level: int,
) -> dict[str, Any]:
    script = Path(__file__).with_name("extract_moss_rknn_island.py")
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for layer in layers:
        cmd = [
            sys.executable,
            str(script),
            "--onnx",
            str(onnx_path),
            "--out-dir",
            str(out_dir),
            "--preset",
            preset,
            "--layer",
            str(layer),
            "--seq-len",
            str(seq_len),
            "--convert-rknn",
            "--target",
            target,
            "--precision",
            precision,
            "--optimization-level",
            str(optimization_level),
        ]
        if force:
            cmd.append("--force")
        started = time.perf_counter()
        ret, output = _run(cmd)
        report_path = out_dir / f"moss_block{layer}_{preset}.s{seq_len}.json"
        item: dict[str, Any] = {
            "layer": layer,
            "returncode": ret,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "report": str(report_path),
            "output_tail": output.splitlines()[-20:],
        }
        if report_path.exists():
            try:
                item["report_json"] = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                item["report_parse_error"] = str(exc)
        results.append(item)
    passed = all(item["returncode"] == 0 and (item.get("report_json") or {}).get("rknn_build", {}).get("status") == "OK" for item in results)
    return {
        "preset": preset,
        "layers": layers,
        "seq_len": seq_len,
        "target": target,
        "precision": precision,
        "onnx": str(onnx_path),
        "out_dir": str(out_dir),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "passed": passed,
        "results": results,
    }


def _parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = [int(x) for x in part.split("-", 1)]
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--layers", default="0-11")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = build_batch(
        onnx_path=args.onnx,
        out_dir=args.out_dir,
        preset=args.preset,
        layers=_parse_layers(args.layers),
        seq_len=args.seq_len,
        target=args.target,
        precision=args.precision,
        force=args.force,
        optimization_level=args.optimization_level,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

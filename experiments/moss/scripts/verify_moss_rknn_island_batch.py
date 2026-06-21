#!/usr/bin/env python3
"""Run parity verification for a batch of MOSS RKNN island artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


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


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout


def _artifact_paths(artifact_dir: Path, preset: str, layer: int, seq_len: int, precision: str, target: str) -> tuple[Path, Path]:
    base = f"moss_block{layer}_{preset}.s{seq_len}"
    return artifact_dir / f"{base}.onnx", artifact_dir / f"{base}.{precision}.{target}.rknn"


def verify_batch(
    artifact_dir: Path,
    preset: str,
    layers: list[int],
    seq_len: int,
    precision: str,
    target: str,
    repeat: int,
    max_rel_l2: float,
    min_cosine: float,
    out_dir: Path,
) -> dict[str, Any]:
    script = Path(__file__).with_name("verify_moss_rknn_island_parity.py")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for layer in layers:
        onnx_path, rknn_path = _artifact_paths(artifact_dir, preset, layer, seq_len, precision, target)
        json_out = out_dir / f"parity_block{layer}_{preset}_s{seq_len}.json"
        cmd = [
            sys.executable,
            str(script),
            "--onnx",
            str(onnx_path),
            "--rknn",
            str(rknn_path),
            "--shape",
            f"1,{seq_len},768",
            "--repeat",
            str(repeat),
            "--max-rel-l2",
            str(max_rel_l2),
            "--min-cosine",
            str(min_cosine),
            "--json-out",
            str(json_out),
        ]
        started = time.perf_counter()
        ret, output = _run(cmd)
        item: dict[str, Any] = {
            "layer": layer,
            "returncode": ret,
            "onnx": str(onnx_path),
            "rknn": str(rknn_path),
            "json": str(json_out),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "output_tail": output.splitlines()[-20:],
        }
        if json_out.exists():
            try:
                item["report"] = json.loads(json_out.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                item["report_parse_error"] = str(exc)
        results.append(item)

    reports = [item["report"] for item in results if isinstance(item.get("report"), dict)]
    output_metrics = [report["outputs"][0] for report in reports if report.get("outputs")]
    latencies = [report.get("latency_ms", {}) for report in reports]
    total_ort = sum(float(lat["ort_avg"]) for lat in latencies if lat.get("ort_avg") is not None)
    total_rknn = sum(float(lat["rknn_avg"]) for lat in latencies if lat.get("rknn_avg") is not None)
    passed = (
        len(reports) == len(layers)
        and all(item["returncode"] == 0 for item in results)
        and all((report.get("gates") or {}).get("passed") is True for report in reports)
    )
    return {
        "preset": preset,
        "layers": layers,
        "seq_len": seq_len,
        "precision": precision,
        "target": target,
        "artifact_dir": str(artifact_dir),
        "out_dir": str(out_dir),
        "repeat": repeat,
        "thresholds": {"max_rel_l2": max_rel_l2, "min_cosine": min_cosine},
        "passed": passed,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "summary": {
            "layers_total": len(layers),
            "reports": len(reports),
            "max_rel_l2": max((float(item["rel_l2"]) for item in output_metrics), default=None),
            "min_cosine": min((float(item["cosine"]) for item in output_metrics), default=None),
            "sum_ort_avg_ms": round(total_ort, 3),
            "sum_rknn_avg_ms": round(total_rknn, 3),
            "speedup": round(total_ort / total_rknn, 3) if total_rknn > 0 else None,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--layers", default="0-11")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--target", default="rk3576")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-rel-l2", type=float, default=0.01)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = verify_batch(
        artifact_dir=args.artifact_dir,
        preset=args.preset,
        layers=_parse_layers(args.layers),
        seq_len=args.seq_len,
        precision=args.precision,
        target=args.target,
        repeat=args.repeat,
        max_rel_l2=args.max_rel_l2,
        min_cosine=args.min_cosine,
        out_dir=args.out_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

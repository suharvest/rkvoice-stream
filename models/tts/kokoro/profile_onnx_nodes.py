#!/usr/bin/env python3
"""Profile ONNX Runtime node time for Kokoro probe graphs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _shape(shape: list[int | str | None]) -> list[int]:
    dims = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            raise ValueError(f"dynamic or unknown input shape is not supported: {shape}")
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    options = ort.SessionOptions()
    options.enable_profiling = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(args.model), options, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    feed = {
        item.name: rng.standard_normal(_shape(item.shape)).astype(np.float32)
        for item in sess.get_inputs()
    }
    for _ in range(args.runs):
        sess.run(None, feed)

    profile_path = Path(sess.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    totals: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {"dur_us": 0.0, "count": 0, "op": ""}
    )
    for event in events:
        if event.get("cat") != "Node":
            continue
        name = event.get("name", "")
        args_data = event.get("args", {})
        op = args_data.get("op_name", "")
        key = name.replace("_kernel_time", "")
        totals[key]["dur_us"] = float(totals[key]["dur_us"]) + float(event.get("dur", 0))
        totals[key]["count"] = int(totals[key]["count"]) + 1
        totals[key]["op"] = op

    rows = sorted(totals.items(), key=lambda item: float(item[1]["dur_us"]), reverse=True)
    total_us = sum(float(item["dur_us"]) for item in totals.values())
    print(json.dumps({"model": str(args.model), "runs": args.runs, "total_ms": total_us / 1000.0}, indent=2))
    for name, item in rows[: args.top]:
        print(
            json.dumps(
                {
                    "node": name,
                    "op": item["op"],
                    "count": item["count"],
                    "total_ms": float(item["dur_us"]) / 1000.0,
                    "avg_ms": float(item["dur_us"]) / 1000.0 / max(int(item["count"]), 1),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize MOSS attention slice verifier JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def _layer_from_report(path: Path, data: dict[str, Any]) -> int:
    if "layer" in data:
        return int(data["layer"])
    text = path.name.split("block", 1)[1].split("_", 1)[0]
    return int(text)


def summarize(pattern: str) -> dict[str, Any]:
    files = [Path(p) for p in glob.glob(pattern)]
    rows = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        output_metrics = list((data.get("outputs") or {}).values())
        if not output_metrics:
            raise RuntimeError(f"No outputs in verifier report: {path}")
        rows.append(
            {
                "layer": _layer_from_report(path, data),
                "path": str(path),
                "passed": bool((data.get("gates") or {}).get("passed")),
                "slice_ms": float((data.get("latency_ms") or {}).get("slice", 0.0)),
                "max_rel_l2": float(max(item["rel_l2"] for item in output_metrics)),
                "min_cosine": float(min(item["cosine"] for item in output_metrics)),
            }
        )
    rows.sort(key=lambda item: item["layer"])
    if [item["layer"] for item in rows] != list(range(12)):
        raise RuntimeError(f"Expected layers 0..11, got {[item['layer'] for item in rows]}")
    return {
        "files": len(rows),
        "all_passed": all(item["passed"] for item in rows),
        "sum_slice_ms": round(sum(item["slice_ms"] for item in rows), 3),
        "avg_slice_ms": round(sum(item["slice_ms"] for item in rows) / max(len(rows), 1), 3),
        "max_rel_l2": max(item["max_rel_l2"] for item in rows),
        "min_cosine": min(item["min_cosine"] for item in rows),
        "layers": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = summarize(args.pattern)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

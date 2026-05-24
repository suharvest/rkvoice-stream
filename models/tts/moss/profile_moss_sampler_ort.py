#!/usr/bin/env python3
"""Profile the MOSS fixed-frame sampler ONNX graph on CPU ORT.

This isolates the per-frame sampler hot path after prefill has produced a
single `global_hidden` row. The output is intentionally small and JSON-only so
it can be run on RK3576 and checked into production evidence when a split point
is being considered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _make_session(path: Path, threads: int, profile_prefix: Path | None = None):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _summarize_times(times: list[float]) -> dict[str, float]:
    return {
        "count": len(times),
        "mean_ms": round(statistics.fmean(times), 3) if times else 0.0,
        "p50_ms": round(_percentile(times, 50), 3),
        "p95_ms": round(_percentile(times, 95), 3),
        "max_ms": round(max(times), 3) if times else 0.0,
    }


def _load_profile(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _node_name(event: dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict):
        name = args.get("node_name") or event.get("name") or args.get("op_name")
    else:
        name = event.get("name")
    return str(name or "unknown")


def _node_op(event: dict[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict):
        provider = str(args.get("provider") or "")
        op_name = str(args.get("op_name") or "")
        if op_name:
            return op_name
        if provider:
            return provider
    name = str(event.get("name") or "")
    if "_kernel_time" in name:
        return name.split("_kernel_time", 1)[0].rsplit("/", 1)[-1]
    return name or "unknown"


def _summarize_profile(events: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    op_totals: Counter[str] = Counter()
    node_totals: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()
    for event in events:
        if event.get("cat") != "Node":
            continue
        dur_us = float(event.get("dur") or 0.0)
        if dur_us <= 0:
            continue
        op_totals[_node_op(event)] += dur_us
        node = _node_name(event)
        node_totals[node] += dur_us
        node_counts[node] += 1

    total_us = float(sum(op_totals.values()))
    def row(name: str, dur_us: float, count: int | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "name": name,
            "total_ms": round(dur_us / 1000.0, 3),
            "pct": round((dur_us / total_us) * 100.0, 2) if total_us else 0.0,
        }
        if count is not None:
            item["count"] = count
            item["mean_us"] = round(dur_us / max(1, count), 3)
        return item

    return {
        "node_event_count": int(sum(node_counts.values())),
        "node_total_ms": round(total_us / 1000.0, 3),
        "top_ops": [row(name, dur) for name, dur in op_totals.most_common(top_k)],
        "top_nodes": [row(name, dur, node_counts[name]) for name, dur in node_totals.most_common(top_k)],
    }


def _initializer_summary(path: Path, top_k: int) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    initializers = []
    for init in model.graph.initializer:
        elem_count = 1
        for dim in init.dims:
            elem_count *= int(dim)
        initializers.append(
            {
                "name": init.name,
                "shape": list(init.dims),
                "elem_type": int(init.data_type),
                "elements": elem_count,
            }
        )
    initializers.sort(key=lambda item: item["elements"], reverse=True)
    return {
        "node_count": len(model.graph.node),
        "op_counts": dict(op_counts.most_common(top_k)),
        "top_initializers": initializers[:top_k],
    }


def _random_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = rng.normal(0.0, 0.75, size=(1, 768)).astype(np.float32)
    seen = np.zeros((1, 16, 1024), dtype=np.int32)
    # Mark a few deterministic ids so repetition masking paths execute.
    for ch in range(16):
        seen[0, ch, (ch * 31 + seed) % 1024] = 1
        seen[0, ch, (ch * 47 + seed + 7) % 1024] = 1
    return {
        "global_hidden": hidden,
        "repetition_seen_mask": seen,
        "assistant_random_u": rng.random((1,), dtype=np.float32),
        "audio_random_u": rng.random((1, 16), dtype=np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sampler_path = args.model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    if not sampler_path.exists():
        raise FileNotFoundError(sampler_path)

    with tempfile.TemporaryDirectory(prefix="moss_sampler_profile_") as tmp:
        profile_prefix = Path(tmp) / "sampler"
        load_start = time.perf_counter()
        session = _make_session(sampler_path, args.threads, profile_prefix)
        load_ms = (time.perf_counter() - load_start) * 1000.0

        inputs = _random_inputs(args.seed)
        for _ in range(args.warmup):
            session.run(None, inputs)

        run_times: list[float] = []
        last_outputs: list[np.ndarray] = []
        for i in range(args.runs):
            inputs = _random_inputs(args.seed + i + 1)
            start = time.perf_counter()
            outputs = session.run(None, inputs)
            run_times.append((time.perf_counter() - start) * 1000.0)
            last_outputs = [np.asarray(output) for output in outputs]

        profile_path = Path(session.end_profiling())
        events = _load_profile(profile_path)

    result = {
        "model_dir": str(args.model_dir),
        "sampler_path": str(sampler_path),
        "threads": args.threads,
        "warmup": args.warmup,
        "runs": args.runs,
        "load_ms": round(load_ms, 3),
        "run_ms": _summarize_times(run_times),
        "outputs": [
            {
                "index": index,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "finite": bool(np.isfinite(output).all()) if np.issubdtype(output.dtype, np.number) else None,
            }
            for index, output in enumerate(last_outputs)
        ],
        "graph": _initializer_summary(sampler_path, args.top_k),
        "profile": _summarize_profile(events, args.top_k),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

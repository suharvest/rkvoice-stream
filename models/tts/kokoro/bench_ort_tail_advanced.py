#!/usr/bin/env python3
"""Benchmark advanced ONNX Runtime options for Kokoro CPU tail."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def _dims(shape: list[int | str | None]) -> list[int]:
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            dims.append(dim)
        else:
            raise ValueError(f"dynamic input shape is unsupported: {shape}")
    return dims


def _set_entries(options: ort.SessionOptions, entries: dict[str, str]) -> None:
    for key, value in entries.items():
        options.add_session_config_entry(key, value)


def _make_session(model: Path, cfg: dict, optimized_model: Path | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(cfg["intra"])
    options.inter_op_num_threads = int(cfg["inter"])
    options.execution_mode = (
        ort.ExecutionMode.ORT_PARALLEL if cfg.get("parallel") else ort.ExecutionMode.ORT_SEQUENTIAL
    )
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = levels[str(cfg.get("graph_opt", "all"))]
    options.enable_cpu_mem_arena = bool(cfg.get("arena", True))
    options.enable_mem_pattern = bool(cfg.get("mem_pattern", True))
    options.enable_mem_reuse = bool(cfg.get("mem_reuse", True))
    _set_entries(options, cfg.get("entries", {}))
    if optimized_model is not None:
        options.optimized_model_filepath = str(optimized_model)
    return ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])


def _feed_for(sess: ort.InferenceSession, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        item.name: rng.standard_normal(_dims(item.shape)).astype(np.float32)
        for item in sess.get_inputs()
    }


def _bench(model: Path, cfg: dict, runs: int, warmup: int, seed: int) -> dict:
    sess = _make_session(model, cfg)
    feed = _feed_for(sess, seed)
    for _ in range(warmup):
        outputs = sess.run(None, feed)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        outputs = sess.run(None, feed)
        times.append(time.perf_counter() - start)
    samples = int(np.asarray(outputs[0]).size)
    duration_s = samples / 24000.0
    return {
        "name": cfg["name"],
        "avg_ms": round(sum(times) / len(times) * 1000, 3),
        "min_ms": round(min(times) * 1000, 3),
        "max_ms": round(max(times) * 1000, 3),
        "rtf": round((sum(times) / len(times)) / duration_s, 4) if duration_s else 0.0,
        "shape": list(np.asarray(outputs[0]).shape),
        "cfg": cfg,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-optimized", type=Path)
    args = parser.parse_args()

    print("ort_version", ort.__version__)
    print("providers", ort.get_available_providers())
    print("cpu_count", os.cpu_count())

    configs = []
    for intra in (1, 2, 3, 4, 5, 6, 8):
        configs.append({"name": f"seq_intra{intra}", "intra": intra, "inter": 1, "graph_opt": "all"})
    for intra in (2, 3, 4):
        configs.append({"name": f"parallel_{intra}x2", "intra": intra, "inter": 2, "graph_opt": "all", "parallel": True})
    for intra in (3, 4):
        configs.append({"name": f"no_arena_{intra}", "intra": intra, "inter": 1, "graph_opt": "all", "arena": False})
        configs.append({"name": f"no_pattern_{intra}", "intra": intra, "inter": 1, "graph_opt": "all", "mem_pattern": False})
    for intra in (3, 4):
        configs.append(
            {
                "name": f"spin_off_{intra}",
                "intra": intra,
                "inter": 1,
                "graph_opt": "all",
                "entries": {
                    "session.intra_op.allow_spinning": "0",
                    "session.inter_op.allow_spinning": "0",
                },
            }
        )
    for level in ("extended", "basic", "disable"):
        configs.append({"name": f"graph_{level}", "intra": 4, "inter": 1, "graph_opt": level})

    best = None
    for cfg in configs:
        try:
            result = _bench(args.model, cfg, args.runs, args.warmup, args.seed)
            print(json.dumps(result, sort_keys=True))
            if best is None or result["avg_ms"] < best["avg_ms"]:
                best = result
        except Exception as exc:
            print(json.dumps({"name": cfg["name"], "error": str(exc)}, sort_keys=True))

    if best and args.save_optimized:
        cfg = dict(best["cfg"])
        cfg["name"] = "save_optimized"
        _make_session(args.model, cfg, optimized_model=args.save_optimized)
        print(json.dumps({"optimized_model": str(args.save_optimized), "from_best": best["name"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

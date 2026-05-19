#!/usr/bin/env python3
"""Benchmark selected ONNX Runtime options for Kokoro CPU tail."""

from __future__ import annotations

import argparse
import json
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


def _session(model: Path, cfg: dict) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = int(cfg.get("intra", 4))
    opt.inter_op_num_threads = int(cfg.get("inter", 1))
    opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    opt.graph_optimization_level = levels[str(cfg.get("graph_opt", "all"))]
    opt.enable_cpu_mem_arena = bool(cfg.get("arena", True))
    opt.enable_mem_pattern = bool(cfg.get("mem_pattern", True))
    opt.enable_mem_reuse = bool(cfg.get("mem_reuse", True))
    return ort.InferenceSession(str(model), opt, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configs = [
        {"name": "default_3_all", "intra": 3, "inter": 1, "graph_opt": "all"},
        {"name": "default_4_all", "intra": 4, "inter": 1, "graph_opt": "all"},
        {"name": "no_pattern_3_all", "intra": 3, "inter": 1, "graph_opt": "all", "mem_pattern": False},
        {"name": "no_pattern_4_all", "intra": 4, "inter": 1, "graph_opt": "all", "mem_pattern": False},
        {"name": "no_arena_3_all", "intra": 3, "inter": 1, "graph_opt": "all", "arena": False},
        {"name": "no_arena_4_all", "intra": 4, "inter": 1, "graph_opt": "all", "arena": False},
        {"name": "basic_4", "intra": 4, "inter": 1, "graph_opt": "basic"},
        {"name": "extended_4", "intra": 4, "inter": 1, "graph_opt": "extended"},
    ]

    for cfg in configs:
        sess = _session(args.model, cfg)
        rng = np.random.default_rng(args.seed)
        feed = {
            item.name: rng.standard_normal(_dims(item.shape)).astype(np.float32)
            for item in sess.get_inputs()
        }
        for _ in range(args.warmup):
            outputs = sess.run(None, feed)
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            outputs = sess.run(None, feed)
            times.append(time.perf_counter() - start)
        samples = int(np.asarray(outputs[0]).size)
        avg = sum(times) / len(times)
        print(
            json.dumps(
                {
                    "name": cfg["name"],
                    "avg_ms": round(avg * 1000, 3),
                    "min_ms": round(min(times) * 1000, 3),
                    "max_ms": round(max(times) * 1000, 3),
                    "rtf": round(avg / (samples / 24000.0), 4),
                    "cfg": cfg,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

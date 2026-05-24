#!/usr/bin/env python3
"""Profile a pruned MOSS sampler suffix ONNX graph on CPU ORT."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from models.tts.moss.profile_moss_sampler_ort import (
    _load_profile,
    _summarize_profile,
    _summarize_times,
)


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


def _shape_from_ort(shape: list[Any]) -> list[int]:
    result = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            result.append(dim)
        elif dim == "batch":
            result.append(1)
        else:
            result.append(1)
    return result


def _inputs_for_session(session: Any, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = _shape_from_ort(list(item.shape))
        if item.name == "repetition_seen_mask":
            seen = np.zeros(shape, dtype=np.int32)
            if len(shape) == 3 and shape[1] == 16:
                for ch in range(16):
                    seen[0, ch, (seed + ch * 37) % shape[2]] = 1
            inputs[item.name] = seen
        elif item.type in {"tensor(int32)", "tensor(int64)"}:
            dtype = np.int64 if item.type == "tensor(int64)" else np.int32
            inputs[item.name] = np.zeros(shape, dtype=dtype)
        elif item.name == "/text_lm_head/MatMul_output_0":
            inputs[item.name] = rng.normal(0.0, 0.35, size=shape).astype(np.float32)
        elif item.name == "assistant_random_u":
            inputs[item.name] = rng.random(shape, dtype=np.float32)
        elif item.name == "audio_random_u":
            inputs[item.name] = rng.random(shape, dtype=np.float32)
        else:
            inputs[item.name] = rng.normal(0.0, 0.75, size=shape).astype(np.float32)
    return inputs


def _graph_summary(path: Path, top_k: int) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return {
        "node_count": len(model.graph.node),
        "op_counts": dict(Counter(node.op_type for node in model.graph.node).most_common(top_k)),
        "inputs": [inp.name for inp in model.graph.input],
        "outputs": [out.name for out in model.graph.output],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not args.onnx.exists():
        raise FileNotFoundError(args.onnx)

    with tempfile.TemporaryDirectory(prefix="moss_sampler_suffix_profile_") as tmp:
        profile_prefix = Path(tmp) / "suffix"
        t0 = time.perf_counter()
        session = _make_session(args.onnx, args.threads, profile_prefix)
        load_ms = (time.perf_counter() - t0) * 1000.0

        for i in range(args.warmup):
            session.run(None, _inputs_for_session(session, args.seed + i))

        run_times: list[float] = []
        outputs: list[np.ndarray] = []
        for i in range(args.runs):
            inputs = _inputs_for_session(session, args.seed + 100 + i)
            t0 = time.perf_counter()
            outputs = [np.asarray(output) for output in session.run(None, inputs)]
            run_times.append((time.perf_counter() - t0) * 1000.0)

        profile_path = Path(session.end_profiling())
        events = _load_profile(profile_path)

    result = {
        "onnx": str(args.onnx),
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
            for index, output in enumerate(outputs)
        ],
        "graph": _graph_summary(args.onnx, args.top_k),
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

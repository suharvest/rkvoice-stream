#!/usr/bin/env python3
"""Verify a MOSS sampler split that replaces text_lm_head with RKNN.

The verifier rewrites the sampler ONNX graph without copying external data:

  prefix ORT: original sampler inputs -> /Gather_11_output_0
  RKNN:       /Gather_11_output_0 -> /text_lm_head/MatMul_output_0
  suffix ORT: original sampler inputs + text logits -> sampler outputs

It checks final token parity against the full ORT sampler. This is the minimum
gate before considering a production sampler split.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, utils


SAMPLER_INPUT_SHAPES = {
    "global_hidden": [1, 768],
    "repetition_seen_mask": [1, 16, 1024],
    "assistant_random_u": [1],
    "audio_random_u": [1, 16],
    "/Gather_11_output_0": [1, 768],
    "/text_lm_head/MatMul_output_0": [1, 16384],
}
SAMPLER_INPUTS = ["global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u"]
SAMPLER_OUTPUTS = ["should_continue", "frame_token_ids"]


def _make_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _set_input_shapes(model: onnx.ModelProto) -> None:
    for inp in model.graph.input:
        shape = SAMPLER_INPUT_SHAPES.get(inp.name)
        if not shape:
            continue
        dims = inp.type.tensor_type.shape.dim
        for dim, value in zip(dims, shape, strict=True):
            dim.dim_param = ""
            dim.dim_value = int(value)


def _save_external_ref_model(model: onnx.ModelProto, path: Path) -> None:
    _set_input_shapes(model)
    onnx.save_model(model, str(path), save_as_external_data=False)


def _fix_extracted_model(path: Path) -> Path:
    model = onnx.load(str(path), load_external_data=False)
    _save_external_ref_model(model, path)
    return path


def _make_prefix_model_naive(source: Path, out_path: Path) -> Path:
    model = onnx.load(str(source), load_external_data=False)
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info("/Gather_11_output_0", TensorProto.FLOAT, [1, 768])
    )
    _save_external_ref_model(model, out_path)
    return out_path


def _make_suffix_source(source: Path, out_path: Path) -> Path:
    model = onnx.load(str(source), load_external_data=False)
    kept_nodes = [node for node in model.graph.node if node.name != "/text_lm_head/MatMul"]
    if len(kept_nodes) == len(model.graph.node):
        raise RuntimeError("Could not find /text_lm_head/MatMul in sampler graph")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    if not any(inp.name == "/text_lm_head/MatMul_output_0" for inp in model.graph.input):
        model.graph.input.append(
            helper.make_tensor_value_info("/text_lm_head/MatMul_output_0", TensorProto.FLOAT, [1, 16384])
        )
    _save_external_ref_model(model, out_path)
    return out_path


def _make_suffix_model_naive(source: Path, out_path: Path) -> Path:
    return _make_suffix_source(source, out_path)


def _make_prefix_model_pruned(source: Path, out_path: Path) -> Path:
    utils.extract_model(
        str(source),
        str(out_path),
        input_names=SAMPLER_INPUTS,
        output_names=["/Gather_11_output_0"],
        check_model=False,
    )
    return _fix_extracted_model(out_path)


def _make_suffix_model_pruned(source: Path, source_path: Path, out_path: Path) -> Path:
    _make_suffix_source(source, source_path)
    utils.extract_model(
        str(source_path),
        str(out_path),
        input_names=[*SAMPLER_INPUTS, "/text_lm_head/MatMul_output_0"],
        output_names=SAMPLER_OUTPUTS,
        check_model=False,
    )
    return _fix_extracted_model(out_path)


def _make_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    seen = np.zeros((1, 16, 1024), dtype=np.int32)
    for ch in range(16):
        seen[0, ch, (seed + ch * 29) % 1024] = 1
        seen[0, ch, (seed + ch * 43 + 11) % 1024] = 1
    return {
        "global_hidden": rng.normal(0.0, 0.75, size=(1, 768)).astype(np.float32),
        "repetition_seen_mask": seen,
        "assistant_random_u": rng.random((1,), dtype=np.float32),
        "audio_random_u": rng.random((1, 16), dtype=np.float32),
    }


def _load_rknn(path: Path):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(str(path))
    if ret != 0:
        raise RuntimeError(f"load_rknn returned {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime returned {ret}")
    return rknn


def _metrics(ref: np.ndarray, got: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    diff = got - ref
    ref_flat = ref.reshape(-1)
    got_flat = got.reshape(-1)
    denom = float(np.linalg.norm(ref_flat)) + 1e-12
    cosine = float(np.dot(ref_flat, got_flat) / ((np.linalg.norm(ref_flat) * np.linalg.norm(got_flat)) + 1e-12))
    return {
        "shape": list(got.shape),
        "finite": bool(np.isfinite(got).all()),
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
    }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 3) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rknn", type=Path)
    parser.add_argument("--head-onnx", type=Path, help="Optional extracted text_lm_head ONNX for logit parity")
    parser.add_argument("--prefix-onnx", type=Path, help="Use a prebuilt prefix ONNX")
    parser.add_argument("--suffix-onnx", type=Path, help="Use a prebuilt suffix ONNX")
    parser.add_argument("--prepare-only", action="store_true", help="Only generate split ONNX files and exit")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split-mode", choices=["pruned", "naive"], default="pruned")
    parser.add_argument("--max-logit-rel-l2", type=float, default=0.01)
    parser.add_argument("--require-token-parity", action="store_true", default=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sampler_path = args.model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    if not sampler_path.exists():
        raise FileNotFoundError(sampler_path)
    if not args.prepare_only and (args.rknn is None or not args.rknn.exists()):
        raise FileNotFoundError(args.rknn)

    prefix_path = args.prefix_onnx or args.model_dir / f".sampler_text_head_prefix.{args.split_mode}.onnx"
    suffix_path = args.suffix_onnx or args.model_dir / f".sampler_text_head_suffix.{args.split_mode}.onnx"
    suffix_source_path = args.model_dir / f".sampler_text_head_suffix_source.{time.time_ns()}.onnx"
    cleanup_paths: list[Path] = []
    try:
        if args.prefix_onnx is None or args.suffix_onnx is None:
            if args.split_mode == "pruned":
                _make_prefix_model_pruned(sampler_path, prefix_path)
                _make_suffix_model_pruned(sampler_path, suffix_source_path, suffix_path)
            else:
                _make_prefix_model_naive(sampler_path, prefix_path)
                _make_suffix_model_naive(sampler_path, suffix_path)
            cleanup_paths.append(suffix_source_path)
            if not args.prepare_only:
                cleanup_paths.extend([prefix_path, suffix_path])

        if args.prepare_only:
            result = {
                "model_dir": str(args.model_dir),
                "split_mode": args.split_mode,
                "prefix_onnx": str(prefix_path),
                "suffix_onnx": str(suffix_path),
                "prefix_size_bytes": prefix_path.stat().st_size,
                "suffix_size_bytes": suffix_path.stat().st_size,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0

        full = _make_session(sampler_path, args.threads)
        prefix = _make_session(prefix_path, args.threads)
        suffix = _make_session(suffix_path, args.threads)
        assert args.rknn is not None
        rknn = _load_rknn(args.rknn)

        try:
            for i in range(args.warmup):
                inputs = _make_inputs(args.seed + i)
                head_in = prefix.run(None, inputs)[0]
                head_out = rknn.inference(inputs=[head_in])
                if head_out is None:
                    raise RuntimeError("warmup RKNN inference returned None")
                suffix_inputs = dict(inputs)
                suffix_inputs["/text_lm_head/MatMul_output_0"] = np.asarray(head_out[0], dtype=np.float32)
                suffix.run(None, suffix_inputs)

            rows: list[dict[str, Any]] = []
            full_times: list[float] = []
            prefix_times: list[float] = []
            rknn_times: list[float] = []
            suffix_times: list[float] = []
            token_equal_count = 0
            continue_equal_count = 0
            logit_metrics: list[dict[str, Any]] = []

            # ORT-only head session from the extracted one-output graph gives a
            # direct logit parity check for the RKNN island.
            head_onnx = args.head_onnx
            if head_onnx is None:
                candidate = args.rknn.parent / "moss_sampler_text_lm_head.onnx"
                head_onnx = candidate if candidate.exists() else None
            head_ort = _make_session(head_onnx, args.threads) if head_onnx is not None and head_onnx.exists() else None

            for i in range(args.runs):
                inputs = _make_inputs(args.seed + 100 + i)
                t0 = time.perf_counter()
                full_outputs = full.run(None, inputs)
                full_times.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                head_in = prefix.run(None, inputs)[0]
                prefix_times.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                head_outputs = rknn.inference(inputs=[head_in])
                rknn_times.append((time.perf_counter() - t0) * 1000.0)
                if head_outputs is None:
                    raise RuntimeError("RKNN inference returned None")
                head_logits = np.asarray(head_outputs[0], dtype=np.float32)

                if head_ort is not None:
                    ref_logits = head_ort.run(None, {head_ort.get_inputs()[0].name: head_in})[0]
                    logit_metrics.append(_metrics(ref_logits, head_logits))

                suffix_inputs = dict(inputs)
                suffix_inputs["/text_lm_head/MatMul_output_0"] = head_logits
                t0 = time.perf_counter()
                split_outputs = suffix.run(None, suffix_inputs)
                suffix_times.append((time.perf_counter() - t0) * 1000.0)

                continue_equal = bool(np.array_equal(full_outputs[0], split_outputs[0]))
                token_equal = bool(np.array_equal(full_outputs[1], split_outputs[1]))
                continue_equal_count += int(continue_equal)
                token_equal_count += int(token_equal)
                rows.append(
                    {
                        "run": i,
                        "continue_equal": continue_equal,
                        "token_equal": token_equal,
                        "token_mismatches": int(np.count_nonzero(full_outputs[1] != split_outputs[1])),
                        "full_continue": np.asarray(full_outputs[0]).reshape(-1).astype(int).tolist(),
                        "split_continue": np.asarray(split_outputs[0]).reshape(-1).astype(int).tolist(),
                    }
                )
        finally:
            try:
                rknn.release()
            except Exception:
                pass
    finally:
        for path in cleanup_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    max_logit_rel_l2 = max((float(item["rel_l2"]) for item in logit_metrics), default=0.0)
    result = {
        "model_dir": str(args.model_dir),
        "rknn": str(args.rknn),
        "split_mode": args.split_mode,
        "runs": args.runs,
        "token_equal": token_equal_count,
        "continue_equal": continue_equal_count,
        "latency_ms": {
            "full_ort_avg": _mean(full_times),
            "prefix_ort_avg": _mean(prefix_times),
            "rknn_head_avg": _mean(rknn_times),
            "suffix_ort_avg": _mean(suffix_times),
            "split_total_avg": _mean([a + b + c for a, b, c in zip(prefix_times, rknn_times, suffix_times, strict=True)]),
        },
        "logits": {
            "checked": bool(logit_metrics),
            "max_rel_l2": max_logit_rel_l2,
            "min_cosine": min((float(item["cosine"]) for item in logit_metrics), default=None),
        },
        "rows": rows,
        "gates": {
            "max_logit_rel_l2": args.max_logit_rel_l2,
            "token_parity": token_equal_count == args.runs,
            "continue_parity": continue_equal_count == args.runs,
            "logit_parity": max_logit_rel_l2 <= args.max_logit_rel_l2 if logit_metrics else True,
        },
    }
    result["gates"]["passed"] = bool(
        result["gates"]["continue_parity"]
        and (result["gates"]["token_parity"] or not args.require_token_parity)
        and result["gates"]["logit_parity"]
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

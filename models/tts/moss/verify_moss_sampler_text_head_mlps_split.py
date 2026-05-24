#!/usr/bin/env python3
"""Verify replacing MOSS sampler text head and local MLPs with RKNN.

Pipeline under test:

  full ORT sampler
  vs
  prefix ORT -> text_lm_head RKNN + 17 local MLP RKNN outputs -> suffix ORT

This is a verifier only. Passing it proves the split boundary is numerically
safe and measures whether the split can beat full ORT before backend wiring.
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


SAMPLER_INPUTS = ["global_hidden", "repetition_seen_mask", "assistant_random_u", "audio_random_u"]
SAMPLER_OUTPUTS = ["should_continue", "frame_token_ids"]
TEXT_HEAD_IN = "/Gather_11_output_0"
TEXT_HEAD_OUT = "/text_lm_head/MatMul_output_0"
LN2_OUTPUTS = [
    "/ln_2/LayerNormalization_output_0",
    *[f"/ln_2_{i}/LayerNormalization_output_0" for i in range(1, 17)],
]
MLP_OUTPUTS = [
    "/mlp/fc_out/Add_output_0",
    *[f"/mlp/fc_out_{i}/Add_output_0" for i in range(1, 17)],
]
SAMPLER_INPUT_SHAPES = {
    "global_hidden": [1, 768],
    "repetition_seen_mask": [1, 16, 1024],
    "assistant_random_u": [1],
    "audio_random_u": [1, 16],
    TEXT_HEAD_IN: [1, 768],
    TEXT_HEAD_OUT: [1, 16384],
    **{name: [1, 1, 768] for name in LN2_OUTPUTS},
    **{name: [1, 1, 768] for name in MLP_OUTPUTS},
}


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


def _save_model(model: onnx.ModelProto, path: Path) -> Path:
    _set_input_shapes(model)
    onnx.save_model(model, str(path), save_as_external_data=False)
    return path


def _fix_extracted_model(path: Path) -> Path:
    model = onnx.load(str(path), load_external_data=False)
    return _save_model(model, path)


def _make_prefix_model(source: Path, out_path: Path) -> Path:
    utils.extract_model(
        str(source),
        str(out_path),
        input_names=SAMPLER_INPUTS,
        output_names=[TEXT_HEAD_IN, *LN2_OUTPUTS],
        check_model=False,
    )
    return _fix_extracted_model(out_path)


def _mlp_node_prefixes() -> list[str]:
    prefixes: list[str] = []
    for i in range(17):
        suffix = "" if i == 0 else f"_{i}"
        prefixes.extend([f"/mlp/fc_in{suffix}/", f"/mlp/act{suffix}/", f"/mlp/fc_out{suffix}/"])
    return prefixes


def _make_suffix_source(source: Path, out_path: Path) -> Path:
    model = onnx.load(str(source), load_external_data=False)
    mlp_prefixes = tuple(_mlp_node_prefixes())
    kept_nodes = []
    removed = {"text_head": 0, "mlp": 0}
    for node in model.graph.node:
        if node.name == "/text_lm_head/MatMul":
            removed["text_head"] += 1
            continue
        if node.name.startswith(mlp_prefixes):
            removed["mlp"] += 1
            continue
        kept_nodes.append(node)
    if removed["text_head"] != 1:
        raise RuntimeError(f"Expected one text head MatMul, removed {removed['text_head']}")
    if removed["mlp"] == 0:
        raise RuntimeError("No MLP nodes were removed")
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    existing_inputs = {inp.name for inp in model.graph.input}
    if TEXT_HEAD_OUT not in existing_inputs:
        model.graph.input.append(helper.make_tensor_value_info(TEXT_HEAD_OUT, TensorProto.FLOAT, [1, 16384]))
    for name in MLP_OUTPUTS:
        if name not in existing_inputs:
            model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 768]))
    return _save_model(model, out_path)


def _make_suffix_model(source: Path, source_path: Path, out_path: Path) -> Path:
    _make_suffix_source(source, source_path)
    utils.extract_model(
        str(source_path),
        str(out_path),
        input_names=[*SAMPLER_INPUTS, TEXT_HEAD_OUT, *MLP_OUTPUTS],
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


def _max(values: list[float]) -> float:
    return max(values) if values else 0.0


def _run_reference(session, input_names: list[str], inputs: list[np.ndarray]):
    return session.run(None, {name: value for name, value in zip(input_names, inputs, strict=True)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-rknn", type=Path, required=True)
    parser.add_argument("--mlps-rknn", type=Path, required=True)
    parser.add_argument("--text-head-onnx", type=Path)
    parser.add_argument("--mlps-onnx", type=Path)
    parser.add_argument("--prefix-onnx", type=Path)
    parser.add_argument("--suffix-onnx", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-logit-rel-l2", type=float, default=0.01)
    parser.add_argument("--max-mlp-rel-l2", type=float, default=0.01)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sampler_path = args.model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    if not sampler_path.exists():
        raise FileNotFoundError(sampler_path)

    prefix_path = args.prefix_onnx or args.model_dir / ".sampler_text_head_mlps_prefix.pruned.onnx"
    suffix_path = args.suffix_onnx or args.model_dir / ".sampler_text_head_mlps_suffix.pruned.onnx"
    suffix_source_path = args.model_dir / f".sampler_text_head_mlps_suffix_source.{time.time_ns()}.onnx"
    cleanup_paths = [suffix_source_path]
    try:
        if args.prefix_onnx is None:
            _make_prefix_model(sampler_path, prefix_path)
            if not args.prepare_only:
                cleanup_paths.append(prefix_path)
        if args.suffix_onnx is None:
            _make_suffix_model(sampler_path, suffix_source_path, suffix_path)
            if not args.prepare_only:
                cleanup_paths.append(suffix_path)

        if args.prepare_only:
            result = {
                "model_dir": str(args.model_dir),
                "prefix_onnx": str(prefix_path),
                "suffix_onnx": str(suffix_path),
                "prefix_size_bytes": prefix_path.stat().st_size,
                "suffix_size_bytes": suffix_path.stat().st_size,
            }
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0

        for required in (args.text_rknn, args.mlps_rknn):
            if not required.exists():
                raise FileNotFoundError(required)

        full = _make_session(sampler_path, args.threads)
        prefix = _make_session(prefix_path, args.threads)
        suffix = _make_session(suffix_path, args.threads)
        text_ref = _make_session(args.text_head_onnx, args.threads) if args.text_head_onnx and args.text_head_onnx.exists() else None
        mlps_ref = _make_session(args.mlps_onnx, args.threads) if args.mlps_onnx and args.mlps_onnx.exists() else None
        text_rknn = _load_rknn(args.text_rknn)
        mlps_rknn = _load_rknn(args.mlps_rknn)

        try:
            for i in range(args.warmup):
                inputs = _make_inputs(args.seed + i)
                prefix_outputs = prefix.run(None, inputs)
                head_in = prefix_outputs[0]
                ln2_rows = [np.asarray(item, dtype=np.float32) for item in prefix_outputs[1:]]
                head_out = text_rknn.inference(inputs=[head_in])
                mlp_out = mlps_rknn.inference(inputs=ln2_rows)
                if head_out is None or mlp_out is None:
                    raise RuntimeError("warmup RKNN inference returned None")
                suffix_inputs = dict(inputs)
                suffix_inputs[TEXT_HEAD_OUT] = np.asarray(head_out[0], dtype=np.float32)
                suffix_inputs.update({name: np.asarray(value, dtype=np.float32) for name, value in zip(MLP_OUTPUTS, mlp_out, strict=True)})
                suffix.run(None, suffix_inputs)

            rows: list[dict[str, Any]] = []
            full_times: list[float] = []
            prefix_times: list[float] = []
            text_times: list[float] = []
            mlps_times: list[float] = []
            suffix_times: list[float] = []
            token_equal_count = 0
            continue_equal_count = 0
            logit_metrics: list[dict[str, Any]] = []
            mlp_metrics: list[dict[str, Any]] = []

            for i in range(args.runs):
                inputs = _make_inputs(args.seed + 100 + i)
                t0 = time.perf_counter()
                full_outputs = full.run(None, inputs)
                full_times.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                prefix_outputs = prefix.run(None, inputs)
                prefix_times.append((time.perf_counter() - t0) * 1000.0)
                head_in = prefix_outputs[0]
                ln2_rows = [np.asarray(item, dtype=np.float32) for item in prefix_outputs[1:]]

                t0 = time.perf_counter()
                text_outputs = text_rknn.inference(inputs=[head_in])
                text_times.append((time.perf_counter() - t0) * 1000.0)
                if text_outputs is None:
                    raise RuntimeError("text RKNN inference returned None")
                text_logits = np.asarray(text_outputs[0], dtype=np.float32)

                t0 = time.perf_counter()
                mlp_outputs = mlps_rknn.inference(inputs=ln2_rows)
                mlps_times.append((time.perf_counter() - t0) * 1000.0)
                if mlp_outputs is None:
                    raise RuntimeError("MLP RKNN inference returned None")
                mlp_outputs = [np.asarray(item, dtype=np.float32) for item in mlp_outputs]

                if text_ref is not None:
                    ref_logits = text_ref.run(None, {text_ref.get_inputs()[0].name: head_in})[0]
                    logit_metrics.append(_metrics(ref_logits, text_logits))
                if mlps_ref is not None:
                    ref_mlps = _run_reference(mlps_ref, LN2_OUTPUTS, ln2_rows)
                    for ref, got in zip(ref_mlps, mlp_outputs, strict=True):
                        mlp_metrics.append(_metrics(ref, got))

                suffix_inputs = dict(inputs)
                suffix_inputs[TEXT_HEAD_OUT] = text_logits
                suffix_inputs.update({name: value for name, value in zip(MLP_OUTPUTS, mlp_outputs, strict=True)})
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
            for rknn in (text_rknn, mlps_rknn):
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

    max_logit_rel_l2 = _max([float(item["rel_l2"]) for item in logit_metrics])
    max_mlp_rel_l2 = _max([float(item["rel_l2"]) for item in mlp_metrics])
    split_totals = [a + b + c + d for a, b, c, d in zip(prefix_times, text_times, mlps_times, suffix_times, strict=True)]
    result = {
        "model_dir": str(args.model_dir),
        "text_rknn": str(args.text_rknn),
        "mlps_rknn": str(args.mlps_rknn),
        "prefix_onnx": str(prefix_path),
        "suffix_onnx": str(suffix_path),
        "runs": args.runs,
        "token_equal": token_equal_count,
        "continue_equal": continue_equal_count,
        "latency_ms": {
            "full_ort_avg": _mean(full_times),
            "prefix_ort_avg": _mean(prefix_times),
            "rknn_text_avg": _mean(text_times),
            "rknn_mlps_avg": _mean(mlps_times),
            "suffix_ort_avg": _mean(suffix_times),
            "split_total_avg": _mean(split_totals),
        },
        "logits": {
            "checked": bool(logit_metrics),
            "max_rel_l2": max_logit_rel_l2,
            "min_cosine": min((float(item["cosine"]) for item in logit_metrics), default=None),
        },
        "mlps": {
            "checked": bool(mlp_metrics),
            "max_rel_l2": max_mlp_rel_l2,
            "min_cosine": min((float(item["cosine"]) for item in mlp_metrics), default=None),
        },
        "rows": rows,
        "gates": {
            "max_logit_rel_l2": args.max_logit_rel_l2,
            "max_mlp_rel_l2": args.max_mlp_rel_l2,
            "token_parity": token_equal_count == args.runs,
            "continue_parity": continue_equal_count == args.runs,
            "logit_parity": max_logit_rel_l2 <= args.max_logit_rel_l2 if logit_metrics else True,
            "mlp_parity": max_mlp_rel_l2 <= args.max_mlp_rel_l2 if mlp_metrics else True,
        },
    }
    result["gates"]["passed"] = bool(
        result["gates"]["token_parity"]
        and result["gates"]["continue_parity"]
        and result["gates"]["logit_parity"]
        and result["gates"]["mlp_parity"]
    )
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

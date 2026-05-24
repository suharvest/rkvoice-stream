#!/usr/bin/env python3
"""Verify a dependency-preserving sampler split with RKNN local MLPs.

This verifier executes the local transformer in dependency order:

  ORT stage0 -> ln2_0
  RKNN MLP_0 -> ORT text-head input stage -> RKNN text head
  ORT stage1 -> ln2_1
  RKNN MLP_1 -> ORT stage2 -> ln2_2
  ...
  RKNN MLP_16 -> ORT final suffix -> sampler outputs

It is intentionally separate from the service backend. Passing this gate is the
minimum evidence needed before considering a production sampler split.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper

from models.tts.moss.compare_moss_hybrid_sampler import _make_sampler_debug_session, _sampler_margin_report
from models.tts.moss.verify_moss_sampler_text_head_mlps_split import (
    LN2_OUTPUTS,
    MLP_OUTPUTS,
    SAMPLER_INPUTS,
    SAMPLER_INPUT_SHAPES,
    SAMPLER_OUTPUTS,
    TEXT_HEAD_IN,
    TEXT_HEAD_OUT,
    _make_inputs,
    _mean,
    _metrics,
    _mlp_node_prefixes,
    _set_input_shapes,
)


def _make_session(path: Path, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _save_model(model: onnx.ModelProto, path: Path) -> Path:
    _set_input_shapes(model)
    onnx.save_model(model, str(path), save_as_external_data=False)
    return path


def _fix_extracted_model(path: Path) -> Path:
    model = onnx.load(str(path), load_external_data=False)
    return _save_model(model, path)


def _suffix(index: int) -> str:
    return "" if index == 0 else f"_{index}"


def _mlp_prefixes_for(index: int) -> tuple[str, ...]:
    suffix = _suffix(index)
    return (f"/mlp/fc_in{suffix}/", f"/mlp/act{suffix}/", f"/mlp/fc_out{suffix}/")


def _make_source_with_replacements(
    source: Path,
    out_path: Path,
    replaced_mlps: int,
    replace_text_head: bool,
    output_names: list[str],
) -> Path:
    """Remove replaced MLP/text nodes and expose their outputs as inputs."""

    model = onnx.load(str(source), load_external_data=False)
    removed_mlps = 0
    removed_text = 0
    prefixes = tuple(prefix for i in range(replaced_mlps) for prefix in _mlp_prefixes_for(i))
    kept_nodes = []
    for node in model.graph.node:
        if replace_text_head and node.name == "/text_lm_head/MatMul":
            removed_text += 1
            continue
        if prefixes and node.name.startswith(prefixes):
            removed_mlps += 1
            continue
        kept_nodes.append(node)

    if replaced_mlps and removed_mlps == 0:
        raise RuntimeError("No MLP nodes were removed")
    if replace_text_head and removed_text != 1:
        raise RuntimeError(f"Expected one text head node, removed {removed_text}")

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    existing_inputs = {inp.name for inp in model.graph.input}
    for name in MLP_OUTPUTS[:replaced_mlps]:
        if name not in existing_inputs:
            model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 768]))
    if replace_text_head and TEXT_HEAD_OUT not in existing_inputs:
        model.graph.input.append(helper.make_tensor_value_info(TEXT_HEAD_OUT, TensorProto.FLOAT, [1, 16384]))
    del model.graph.output[:]
    for name in output_names:
        shape = SAMPLER_INPUT_SHAPES.get(name)
        if name in SAMPLER_OUTPUTS:
            elem_type = TensorProto.INT32
            shape = [1, 1] if name == "should_continue" else [1, 16]
        else:
            elem_type = TensorProto.FLOAT
        if shape is None:
            raise RuntimeError(f"Missing shape for graph output {name}")
        model.graph.output.append(helper.make_tensor_value_info(name, elem_type, shape))
    return _save_model(model, out_path)


def _node_suffix(name: str, prefix: str) -> int:
    if name == prefix:
        return 0
    return int(name.removeprefix(prefix + "_")) + 1


def _add_graph_output(model: onnx.ModelProto, name: str, elem_type: int) -> None:
    if any(output.name == name for output in model.graph.output):
        return
    model.graph.output.append(helper.make_tensor_value_info(name, elem_type, None))


def _sampler_debug_specs(model: onnx.ModelProto) -> list[dict[str, str]]:
    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    specs: list[dict[str, str]] = []
    topk_nodes = sorted(
        [node for node in model.graph.node if node.op_type == "TopK" and node.name.startswith("/TopK")],
        key=lambda node: _node_suffix(node.name, "/TopK"),
    )
    for channel, node in enumerate(topk_nodes):
        if len(specs) >= 16:
            break
        values_name, indices_name = node.output[0], node.output[1]
        where = next(
            (
                candidate
                for candidate in consumers.get(values_name, [])
                if candidate.op_type == "Where" and values_name in candidate.input
            ),
            None,
        )
        final_softmax = None
        final_cumsum = None
        if where is not None:
            final_softmax = next(
                (
                    candidate
                    for candidate in consumers.get(where.output[0], [])
                    if candidate.op_type == "Softmax"
                ),
                None,
            )
        if final_softmax is not None:
            final_cumsum = next(
                (
                    candidate
                    for candidate in consumers.get(final_softmax.output[0], [])
                    if candidate.op_type == "CumSum"
                ),
                None,
            )
        selected = next(
            (
                candidate
                for candidate in consumers.get(indices_name, [])
                if candidate.op_type == "GatherElements"
            ),
            None,
        )
        if final_softmax is None or final_cumsum is None or selected is None:
            continue
        specs.append(
            {
                "channel": channel,
                "topk_values": values_name,
                "topk_indices": indices_name,
                "final_probs": final_softmax.output[0],
                "final_cdf": final_cumsum.output[0],
                "selected_token": selected.output[0],
            }
        )
    if len(specs) != 16:
        raise RuntimeError(f"Expected 16 sampler TopK debug specs, found {len(specs)}")
    return specs


def _extract_final_suffix_debug(source: Path, work_dir: Path) -> tuple[Path, list[dict[str, str]]]:
    out_path = work_dir / "final_suffix_debug.onnx"
    model = onnx.load(str(source), load_external_data=False)
    mlp_prefixes = tuple(prefix for i in range(17) for prefix in _mlp_prefixes_for(i))
    kept_nodes = []
    removed_text = 0
    removed_mlp = 0
    for node in model.graph.node:
        if node.name == "/text_lm_head/MatMul":
            removed_text += 1
            continue
        if node.name.startswith(mlp_prefixes):
            removed_mlp += 1
            continue
        kept_nodes.append(node)
    if removed_text != 1:
        raise RuntimeError(f"Expected one text head node, removed {removed_text}")
    if removed_mlp == 0:
        raise RuntimeError("No MLP nodes were removed")

    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    existing_inputs = {inp.name for inp in model.graph.input}
    for name in MLP_OUTPUTS:
        if name not in existing_inputs:
            model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 1, 768]))
    if TEXT_HEAD_OUT not in existing_inputs:
        model.graph.input.append(helper.make_tensor_value_info(TEXT_HEAD_OUT, TensorProto.FLOAT, [1, 16384]))

    specs = _sampler_debug_specs(model)
    del model.graph.output[:]
    _add_graph_output(model, "should_continue", TensorProto.INT32)
    _add_graph_output(model, "frame_token_ids", TensorProto.INT32)
    for spec in specs:
        _add_graph_output(model, spec["topk_values"], TensorProto.FLOAT)
        _add_graph_output(model, spec["topk_indices"], TensorProto.INT64)
        _add_graph_output(model, spec["final_probs"], TensorProto.FLOAT)
        _add_graph_output(model, spec["final_cdf"], TensorProto.FLOAT)
        _add_graph_output(model, spec["selected_token"], TensorProto.INT64)
    return _save_model(model, out_path), specs


def _extract_stage(
    source: Path,
    work_dir: Path,
    name: str,
    replaced_mlps: int,
    replace_text_head: bool,
    output_name: str,
) -> Path:
    out_path = work_dir / f"{name}.onnx"
    _make_source_with_replacements(source, out_path, replaced_mlps, replace_text_head, [output_name])
    return _fix_extracted_model(out_path)


def _link_external_data(source: Path, work_dir: Path) -> None:
    for data_file in source.parent.glob("moss_tts_local_shared.data"):
        target = work_dir / data_file.name
        if target.exists():
            continue
        shutil.copy2(data_file, target)


def _extract_final_suffix(source: Path, work_dir: Path) -> Path:
    out_path = work_dir / "final_suffix.onnx"
    _make_source_with_replacements(source, out_path, replaced_mlps=17, replace_text_head=True, output_names=SAMPLER_OUTPUTS)
    return _fix_extracted_model(out_path)


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


def _resolve_mlp_rknn_paths(mlps_rknn: Path | None, mlps_rknn_dir: Path | None) -> list[Path] | None:
    if mlps_rknn is not None:
        return None
    if mlps_rknn_dir is None:
        raise ValueError("Either --mlps-rknn or --mlps-rknn-dir is required")
    paths: list[Path] = []
    for index in range(17):
        candidates = sorted(mlps_rknn_dir.glob(f"moss_sampler_mlp{index}.*.rknn"))
        if not candidates:
            candidates = sorted(mlps_rknn_dir.glob(f"*sampler_mlp{index}*.rknn"))
        if not candidates:
            raise FileNotFoundError(f"Missing per-block sampler MLP RKNN for block {index} in {mlps_rknn_dir}")
        if len(candidates) > 1:
            raise RuntimeError(f"Multiple per-block sampler MLP RKNN candidates for block {index}: {candidates}")
        paths.append(candidates[0])
    return paths


def _parse_layer_spec(spec: str) -> set[int]:
    if spec.strip().lower() in {"", "all"}:
        return set(range(17))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"Invalid descending layer range: {part}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer > 16)
    if invalid:
        raise ValueError(f"Sampler MLP layers must be in [0,16], got {invalid}")
    return layers


def _zero_ln2() -> np.ndarray:
    return np.zeros((1, 1, 768), dtype=np.float32)


def _run_combined_mlp(mlps_rknn, index: int, ln2: np.ndarray) -> np.ndarray:
    inputs = [_zero_ln2() for _ in range(17)]
    inputs[index] = np.asarray(ln2, dtype=np.float32)
    outputs = mlps_rknn.inference(inputs=inputs)
    if outputs is None:
        raise RuntimeError(f"MLP RKNN inference returned None for block {index}")
    return np.asarray(outputs[index], dtype=np.float32)


def _run_block_mlp(mlps_rknns: list[Any], index: int, ln2: np.ndarray) -> np.ndarray:
    outputs = mlps_rknns[index].inference(inputs=[np.asarray(ln2, dtype=np.float32)])
    if outputs is None:
        raise RuntimeError(f"per-block MLP RKNN inference returned None for block {index}")
    return np.asarray(outputs[0], dtype=np.float32)


def _run_combined_mlp_ref(session, index: int, ln2: np.ndarray) -> np.ndarray:
    inputs = [_zero_ln2() for _ in range(17)]
    inputs[index] = np.asarray(ln2, dtype=np.float32)
    outputs = session.run(None, {name: value for name, value in zip(LN2_OUTPUTS, inputs, strict=True)})
    return np.asarray(outputs[index], dtype=np.float32)


def _make_stage_inputs(base_inputs: dict[str, np.ndarray], mlp_outputs: list[np.ndarray], text_logits: np.ndarray | None) -> dict[str, np.ndarray]:
    merged = dict(base_inputs)
    merged.update({name: value for name, value in zip(MLP_OUTPUTS, mlp_outputs, strict=False)})
    if text_logits is not None:
        merged[TEXT_HEAD_OUT] = text_logits
    return merged


def _run_split_once(
    inputs: dict[str, np.ndarray],
    stages,
    text_rknn,
    mlps_rknn,
    mlps_rknns: list[Any] | None,
    mlps_ref,
    control_ort_mlps: bool,
    rknn_mlp_layers: set[int],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    mlp_outputs: list[np.ndarray] = []
    text_logits: np.ndarray | None = None
    mlp_metrics: list[dict[str, Any]] = []
    timings = {
        "stage_ort_ms": 0.0,
        "rknn_text_ms": 0.0,
        "rknn_mlps_ms": 0.0,
        "ort_mlps_ms": 0.0,
        "suffix_ort_ms": 0.0,
    }

    t0 = time.perf_counter()
    ln2 = stages["ln2"][0].run(None, inputs)[0]
    timings["stage_ort_ms"] += (time.perf_counter() - t0) * 1000.0

    for index in range(17):
        use_rknn_mlp = index in rknn_mlp_layers and not control_ort_mlps
        if not use_rknn_mlp:
            if mlps_ref is None:
                raise RuntimeError("ORT MLP fallback/control requires --mlps-onnx")
            t0 = time.perf_counter()
            mlp_out = _run_combined_mlp_ref(mlps_ref, index, ln2)
            timings["ort_mlps_ms"] += (time.perf_counter() - t0) * 1000.0
        elif mlps_rknns is not None:
            t0 = time.perf_counter()
            mlp_out = _run_block_mlp(mlps_rknns, index, ln2)
            timings["rknn_mlps_ms"] += (time.perf_counter() - t0) * 1000.0
        else:
            if mlps_rknn is None:
                raise RuntimeError("MLP RKNN runner is not loaded")
            t0 = time.perf_counter()
            mlp_out = _run_combined_mlp(mlps_rknn, index, ln2)
            timings["rknn_mlps_ms"] += (time.perf_counter() - t0) * 1000.0
        if mlps_ref is not None:
            ref = _run_combined_mlp_ref(mlps_ref, index, ln2)
            mlp_metrics.append(_metrics(ref, mlp_out))
        mlp_outputs.append(mlp_out)

        if index == 0:
            stage_inputs = _make_stage_inputs(inputs, mlp_outputs, None)
            t0 = time.perf_counter()
            head_in = stages["text_in"].run(None, stage_inputs)[0]
            timings["stage_ort_ms"] += (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            text_outputs = text_rknn.inference(inputs=[head_in])
            timings["rknn_text_ms"] += (time.perf_counter() - t0) * 1000.0
            if text_outputs is None:
                raise RuntimeError("text RKNN inference returned None")
            text_logits = np.asarray(text_outputs[0], dtype=np.float32)

        if index < 16:
            stage_inputs = _make_stage_inputs(inputs, mlp_outputs, text_logits)
            t0 = time.perf_counter()
            ln2 = stages["ln2"][index + 1].run(None, stage_inputs)[0]
            timings["stage_ort_ms"] += (time.perf_counter() - t0) * 1000.0

    stage_inputs = _make_stage_inputs(inputs, mlp_outputs, text_logits)
    t0 = time.perf_counter()
    split_outputs = stages["final"].run(None, stage_inputs)
    timings["suffix_ort_ms"] += (time.perf_counter() - t0) * 1000.0
    return split_outputs, {
        "timings": timings,
        "mlp_metrics": mlp_metrics,
        "mlp_outputs": mlp_outputs,
        "text_logits": text_logits,
    }


def _debug_output_names(specs: list[dict[str, str]]) -> list[str]:
    output_names = ["should_continue", "frame_token_ids"]
    for spec in specs:
        output_names.extend(
            [
                spec["topk_values"],
                spec["topk_indices"],
                spec["final_probs"],
                spec["final_cdf"],
                spec["selected_token"],
            ]
        )
    return output_names


def _run_debug_session(session, specs: list[dict[str, str]], inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    outputs = session.run(_debug_output_names(specs), inputs)
    debug: dict[str, np.ndarray] = {}
    cursor = 2
    for spec in specs:
        for key in ("topk_values", "topk_indices", "final_probs", "final_cdf", "selected_token"):
            debug[f"ch{spec['channel']}.{key}"] = np.asarray(outputs[cursor])
            cursor += 1
    return np.asarray(outputs[0]), np.asarray(outputs[1], dtype=np.int32).reshape(16), debug


def _promotion_decision(result: dict[str, Any], *, min_speedup: float) -> dict[str, Any]:
    gates = result.get("gates", {})
    latency = result.get("latency_ms", {})
    full_avg = latency.get("full_ort_avg")
    split_avg = latency.get("split_total_avg")
    speedup = None
    errors: list[str] = []
    if not gates.get("token_parity", False):
        errors.append(f"token parity failed: {result.get('token_equal')}/{result.get('runs')}")
    if not gates.get("continue_parity", False):
        errors.append(f"continue parity failed: {result.get('continue_equal')}/{result.get('runs')}")
    if not gates.get("mlp_parity", False):
        errors.append(f"MLP parity failed: max_rel_l2={result.get('mlps', {}).get('max_rel_l2')}")
    if full_avg is None or split_avg is None:
        errors.append("latency comparison missing")
    elif float(split_avg) <= 0:
        errors.append(f"invalid split_total_avg={split_avg}")
    else:
        speedup = float(full_avg) / float(split_avg)
        if speedup < min_speedup:
            errors.append(f"split speedup {speedup:.3f}x below required {min_speedup:.3f}x")
    return {
        "allow_service_integration": not errors,
        "min_speedup": min_speedup,
        "speedup": speedup,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-rknn", type=Path, required=True)
    parser.add_argument("--mlps-rknn", type=Path)
    parser.add_argument("--mlps-rknn-dir", type=Path, help="Directory containing 17 per-block moss_sampler_mlp<N>*.rknn files")
    parser.add_argument("--mlps-onnx", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-mlp-rel-l2", type=float, default=0.01)
    parser.add_argument(
        "--min-promotion-speedup",
        type=float,
        default=1.05,
        help="minimum full_ort_avg/split_total_avg required before a passing split may be considered for service integration",
    )
    parser.add_argument("--control-ort-mlps", action="store_true", help="Use ORT MLP outputs in the sequential split control path")
    parser.add_argument("--sampler-debug", action="store_true", help="Expose TopK/CDF margin diagnostics for full vs split sampler")
    parser.add_argument(
        "--rknn-mlp-layers",
        default="all",
        help="Comma/range layer spec for per-block RKNN MLPs, e.g. all, 0, 0,2,5, 0-3. Other layers use --mlps-onnx.",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    sampler_path = args.model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    if not sampler_path.exists():
        raise FileNotFoundError(sampler_path)
    rknn_mlp_layers = set() if args.control_ort_mlps else _parse_layer_spec(args.rknn_mlp_layers)
    per_block_mlp_paths = _resolve_mlp_rknn_paths(args.mlps_rknn, args.mlps_rknn_dir)
    for required in [args.text_rknn, *([] if args.mlps_rknn is None else [args.mlps_rknn]), *(per_block_mlp_paths or [])]:
        if not required.exists():
            raise FileNotFoundError(required)

    with tempfile.TemporaryDirectory(prefix=".moss_seq_mlps_", dir=args.model_dir) as td:
        work_dir = Path(td)
        _link_external_data(sampler_path, work_dir)
        full = _make_session(sampler_path, args.threads)
        stage_paths = {
            "ln2": [
                _extract_stage(
                    sampler_path,
                    work_dir,
                    f"stage_ln2_{i}",
                    replaced_mlps=i,
                    replace_text_head=i > 0,
                    output_name=LN2_OUTPUTS[i],
                )
                for i in range(17)
            ],
            "text_in": _extract_stage(
                sampler_path,
                work_dir,
                "stage_text_in",
                replaced_mlps=1,
                replace_text_head=False,
                output_name=TEXT_HEAD_IN,
            ),
            "final": _extract_final_suffix(sampler_path, work_dir),
        }
        final_debug_specs = None
        if args.sampler_debug:
            final_debug_path, final_debug_specs = _extract_final_suffix_debug(sampler_path, work_dir)
            stage_paths["final_debug"] = final_debug_path
        stages = {
            "ln2": [_make_session(path, args.threads) for path in stage_paths["ln2"]],
            "text_in": _make_session(stage_paths["text_in"], args.threads),
            "final": _make_session(stage_paths["final"], args.threads),
        }
        if args.sampler_debug:
            stages["final_debug"] = _make_session(stage_paths["final_debug"], args.threads)
            full_debug, full_debug_specs, full_debug_path = _make_sampler_debug_session(args.model_dir, args.threads)
        else:
            full_debug = None
            full_debug_specs = None
            full_debug_path = None
        text_rknn = _load_rknn(args.text_rknn)
        mlps_rknn = _load_rknn(args.mlps_rknn) if args.mlps_rknn is not None else None
        mlps_rknns = [_load_rknn(path) for path in per_block_mlp_paths] if per_block_mlp_paths is not None else None
        mlps_ref = _make_session(args.mlps_onnx, args.threads) if args.mlps_onnx and args.mlps_onnx.exists() else None

        try:
            for i in range(args.warmup):
                _run_split_once(
                    _make_inputs(args.seed + i),
                    stages,
                    text_rknn,
                    mlps_rknn,
                    mlps_rknns,
                    mlps_ref,
                    args.control_ort_mlps,
                    rknn_mlp_layers=rknn_mlp_layers,
                )

            rows: list[dict[str, Any]] = []
            full_times: list[float] = []
            stage_times: list[float] = []
            text_times: list[float] = []
            mlps_times: list[float] = []
            ort_mlps_times: list[float] = []
            suffix_times: list[float] = []
            token_equal_count = 0
            continue_equal_count = 0
            mlp_metrics: list[dict[str, Any]] = []

            for i in range(args.runs):
                inputs = _make_inputs(args.seed + 100 + i)
                t0 = time.perf_counter()
                full_outputs = full.run(None, inputs)
                full_times.append((time.perf_counter() - t0) * 1000.0)

                split_outputs, info = _run_split_once(
                    inputs,
                    stages,
                    text_rknn,
                    mlps_rknn,
                    mlps_rknns,
                    mlps_ref,
                    args.control_ort_mlps,
                    rknn_mlp_layers,
                )
                timing = info["timings"]
                stage_times.append(float(timing["stage_ort_ms"]))
                text_times.append(float(timing["rknn_text_ms"]))
                mlps_times.append(float(timing["rknn_mlps_ms"]))
                ort_mlps_times.append(float(timing.get("ort_mlps_ms", 0.0)))
                suffix_times.append(float(timing["suffix_ort_ms"]))
                mlp_metrics.extend(info["mlp_metrics"])

                continue_equal = bool(np.array_equal(full_outputs[0], split_outputs[0]))
                token_equal = bool(np.array_equal(full_outputs[1], split_outputs[1]))
                continue_equal_count += int(continue_equal)
                token_equal_count += int(token_equal)
                row = {
                    "run": i,
                    "continue_equal": continue_equal,
                    "token_equal": token_equal,
                    "token_mismatches": int(np.count_nonzero(full_outputs[1] != split_outputs[1])),
                    "mismatch_indices": np.flatnonzero(full_outputs[1].reshape(-1) != split_outputs[1].reshape(-1)).astype(int).tolist(),
                    "full_continue": np.asarray(full_outputs[0]).reshape(-1).astype(int).tolist(),
                    "split_continue": np.asarray(split_outputs[0]).reshape(-1).astype(int).tolist(),
                    "full_tokens": np.asarray(full_outputs[1]).reshape(-1).astype(int).tolist(),
                    "split_tokens": np.asarray(split_outputs[1]).reshape(-1).astype(int).tolist(),
                }
                if args.sampler_debug:
                    stage_inputs = _make_stage_inputs(inputs, info["mlp_outputs"], info["text_logits"])
                    full_continue_debug, full_frame_debug, full_debug_values = _run_debug_session(full_debug, full_debug_specs, inputs)
                    split_continue_debug, split_frame_debug, split_debug_values = _run_debug_session(
                        stages["final_debug"],
                        final_debug_specs,
                        stage_inputs,
                    )
                    margin_report = _sampler_margin_report(
                        full_debug_values,
                        split_debug_values,
                        full_frame_debug,
                        split_frame_debug,
                        inputs["audio_random_u"].reshape(16),
                    )
                    row["sampler_debug"] = {
                        "full_continue": np.asarray(full_continue_debug).reshape(-1).astype(int).tolist(),
                        "split_continue": np.asarray(split_continue_debug).reshape(-1).astype(int).tolist(),
                        "summary": margin_report["summary"],
                        "channels": margin_report["channels"],
                    }
                rows.append(row)
        finally:
            runners = [text_rknn]
            if mlps_rknn is not None:
                runners.append(mlps_rknn)
            if mlps_rknns is not None:
                runners.extend(mlps_rknns)
            for rknn in runners:
                try:
                    rknn.release()
                except Exception:
                    pass

    max_mlp_rel_l2 = max((float(item["rel_l2"]) for item in mlp_metrics), default=0.0)
    split_totals = [
        stage + text + rknn_mlp + ort_mlp + suffix
        for stage, text, rknn_mlp, ort_mlp, suffix in zip(
            stage_times, text_times, mlps_times, ort_mlps_times, suffix_times, strict=True
        )
    ]
    result = {
        "model_dir": str(args.model_dir),
        "text_rknn": str(args.text_rknn),
        "mlps_rknn": str(args.mlps_rknn) if args.mlps_rknn is not None else None,
        "mlps_rknn_dir": str(args.mlps_rknn_dir) if args.mlps_rknn_dir is not None else None,
        "mlps_mode": "per_block" if per_block_mlp_paths is not None else "combined",
        "rknn_mlp_layers": sorted(rknn_mlp_layers),
        "per_block_mlp_paths": [str(path) for path in per_block_mlp_paths or []],
        "control_ort_mlps": bool(args.control_ort_mlps),
        "sampler_debug": {
            "enabled": bool(args.sampler_debug),
            "full_debug_model": str(full_debug_path) if full_debug_path is not None else None,
        },
        "runs": args.runs,
        "token_equal": token_equal_count,
        "continue_equal": continue_equal_count,
        "latency_ms": {
            "full_ort_avg": _mean(full_times),
            "stage_ort_avg": _mean(stage_times),
            "rknn_text_avg": _mean(text_times),
            "rknn_mlps_avg": _mean(mlps_times),
            "ort_mlps_avg": _mean(ort_mlps_times),
            "suffix_ort_avg": _mean(suffix_times),
            "split_total_avg": _mean(split_totals),
        },
        "mlps": {
            "checked": bool(mlp_metrics),
            "max_rel_l2": max_mlp_rel_l2,
            "min_cosine": min((float(item["cosine"]) for item in mlp_metrics), default=None),
        },
        "rows": rows,
        "gates": {
            "max_mlp_rel_l2": args.max_mlp_rel_l2,
            "token_parity": token_equal_count == args.runs,
            "continue_parity": continue_equal_count == args.runs,
            "mlp_parity": max_mlp_rel_l2 <= args.max_mlp_rel_l2 if mlp_metrics else True,
        },
    }
    result["gates"]["passed"] = bool(
        result["gates"]["token_parity"] and result["gates"]["continue_parity"] and result["gates"]["mlp_parity"]
    )
    result["promotion"] = _promotion_decision(result, min_speedup=args.min_promotion_speedup)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if full_debug_path is not None:
        try:
            full_debug_path.unlink()
        except OSError:
            pass
    return 0 if result["gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

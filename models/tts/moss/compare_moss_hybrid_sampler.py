#!/usr/bin/env python3
"""Compare full ORT and hybrid RKNN MOSS sampler decisions.

This verifier isolates the quality-sensitive boundary after prefill. It runs
the same text prompt through full ORT prefill and hybrid ORT+RKNN prefill, then
feeds both paths the same sampler random values for a fixed number of frames.
If frame tokens diverge here, ASR roundtrip quality failures are explained
before codec/ASR enters the picture.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


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
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": float(np.linalg.norm(diff.reshape(-1)) / denom),
        "cosine": cosine,
    }


def _set_env(args: argparse.Namespace) -> None:
    os.environ["MOSS_ORT_MODEL_DIR"] = str(args.model_dir)
    os.environ["MOSS_ORT_THREADS"] = str(args.threads)
    os.environ["MOSS_ORT_VOICE"] = args.voice
    os.environ["MOSS_ORT_PREFILL_SEQ"] = str(args.prefill_seq)
    os.environ["MOSS_ORT_CODEC_STREAMING"] = "1"
    os.environ["MOSS_ORT_CACHE_VOICE_PREFIX"] = "0"
    os.environ["MOSS_ORT_WARMUP_TEXT"] = ""
    os.environ["MOSS_ORT_HYBRID_RKNN"] = "0"
    os.environ["MOSS_ORT_HYBRID_STRICT"] = "0"
    os.environ["MOSS_ORT_HYBRID_DIR"] = str(args.artifact_dir)
    os.environ["MOSS_ORT_HYBRID_SEQ_LEN"] = str(args.seq_len)


def _attention_input_name(layer: int) -> str:
    if layer == 0:
        return "/Add_15_output_0"
    return f"/Mul_{22 + (layer - 1) * 6}_output_0"


def _trim_kv_to_length(value: np.ndarray, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim >= 3 and arr.shape[2] >= length:
        return arr[:, :, :length, ...].astype(np.float32, copy=False)
    if arr.ndim >= 2 and arr.shape[1] >= length:
        return arr[:, :length, ...].astype(np.float32, copy=False)
    return arr


def _ort_session(path: Path, threads: int) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def _node_suffix(name: str, prefix: str) -> int:
    if name == prefix:
        return 0
    return int(name.removeprefix(prefix + "_")) + 1


def _add_graph_output(model: Any, name: str, elem_type: int) -> None:
    import onnx

    if any(output.name == name for output in model.graph.output):
        return
    model.graph.output.append(onnx.helper.make_tensor_value_info(name, elem_type, None))


def _make_sampler_debug_session(model_dir: Path, threads: int) -> tuple[Any, list[dict[str, str]], Path]:
    """Create a sampler session that also returns per-channel sampling margins.

    The MOSS sampler chooses every audio code from a TopK distribution. RKNN
    prefill drift can be numerically small but still flip sampled tokens when
    the random draw lies close to a CDF boundary. This debug session exposes
    the TopK indices, final probabilities, CDF, and selected token for each
    of the 16 audio code channels.
    """

    import onnx
    from onnx import TensorProto

    sampler_path = model_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    model = onnx.load(str(sampler_path), load_external_data=False)
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

        spec = {
            "channel": channel,
            "topk_values": values_name,
            "topk_indices": indices_name,
            "final_probs": final_softmax.output[0],
            "final_cdf": final_cumsum.output[0],
            "selected_token": selected.output[0],
        }
        specs.append(spec)
        _add_graph_output(model, values_name, TensorProto.FLOAT)
        _add_graph_output(model, indices_name, TensorProto.INT64)
        _add_graph_output(model, final_softmax.output[0], TensorProto.FLOAT)
        _add_graph_output(model, final_cumsum.output[0], TensorProto.FLOAT)
        _add_graph_output(model, selected.output[0], TensorProto.INT64)

    if len(specs) != 16:
        raise RuntimeError(f"Expected 16 sampler TopK debug specs, found {len(specs)}")

    debug_path = model_dir / f".moss_tts_local_fixed_sampled_frame.debug.{os.getpid()}.onnx"
    onnx.save(model, str(debug_path))
    return _ort_session(debug_path, threads), specs, debug_path


class _RknnSession:
    def __init__(self, path: Path) -> None:
        from rknnlite.api import RKNNLite

        self.path = path
        self._rknn = RKNNLite(verbose=False)
        ret = self._rknn.load_rknn(str(path))
        if ret != 0:
            raise RuntimeError(f"load_rknn returned {ret}: {path}")
        ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime returned {ret}: {path}")

    def run(self, hidden: np.ndarray) -> np.ndarray:
        outputs = self._rknn.inference(inputs=[hidden.astype(np.float32, copy=False)])
        if outputs is None:
            raise RuntimeError(f"RKNN inference returned None: {self.path}")
        return np.asarray(outputs[0], dtype=np.float32)

    def release(self) -> None:
        self._rknn.release()


def _parse_rknn_layers(raw: str) -> set[int]:
    raw = raw.strip().lower()
    if raw in {"", "none"}:
        return set()
    if raw == "all":
        return set(range(12))
    layers: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer > 11)
    if invalid:
        raise ValueError(f"invalid layer ids: {invalid}")
    return layers


def _compose_prefill(
    artifact_dir: Path,
    rknn_dir: Path,
    seq_len: int,
    threads: int,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    rknn_layers: set[int],
    rknn_precision: str,
    mlp_split: str,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    actual_len = int(input_ids.shape[1])
    if actual_len > seq_len:
        raise ValueError(f"input length {actual_len} exceeds seq_len {seq_len}")
    padded_ids = np.full((1, seq_len, 17), 1024, dtype=np.int32)
    padded_ids[:, :actual_len, :] = input_ids.astype(np.int32, copy=False)
    padded_mask = np.zeros((1, seq_len), dtype=np.int32)
    padded_mask[:, :actual_len] = attention_mask.astype(np.int32, copy=False)
    mask3 = padded_mask[:, :, None].astype(np.float32)

    embedding = _ort_session(artifact_dir / f"moss_embedding_prefix.s{seq_len}.onnx", threads)
    final_norm = _ort_session(artifact_dir / f"moss_final_norm.s{seq_len}.onnx", threads)
    attention = [_ort_session(artifact_dir / f"moss_block{layer}_attn_residual.s{seq_len}.onnx", threads) for layer in range(12)]
    mlp_ort = {
        layer: _ort_session(artifact_dir / f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx", threads)
        for layer in range(12)
        if layer not in rknn_layers
    }
    ln2_ort = {
        layer: _ort_session(rknn_dir / f"moss_block{layer}_ln2.s{seq_len}.onnx", threads)
        for layer in sorted(rknn_layers)
        if mlp_split in {"mlp_only", "fc_in_act_only", "fc_out_only"}
    }
    fc_in_act_ort = {
        layer: _ort_session(rknn_dir / f"moss_block{layer}_fc_in_act.s{seq_len}.onnx", threads)
        for layer in sorted(rknn_layers)
        if mlp_split == "fc_out_only"
    }
    fc_out_ort = {
        layer: _ort_session(rknn_dir / f"moss_block{layer}_fc_out.s{seq_len}.onnx", threads)
        for layer in sorted(rknn_layers)
        if mlp_split == "fc_in_act_only"
    }
    mlp_rknn = {
        layer: _RknnSession(
            rknn_dir
            / (
                f"moss_block{layer}_mlp.s{seq_len}.{rknn_precision}.rk3576.rknn"
                if mlp_split == "mlp_only"
                else f"moss_block{layer}_fc_in_act.s{seq_len}.{rknn_precision}.rk3576.rknn"
                if mlp_split == "fc_in_act_only"
                else f"moss_block{layer}_fc_out.s{seq_len}.{rknn_precision}.rk3576.rknn"
                if mlp_split == "fc_out_only"
                else f"moss_block{layer}_ln2_mlp.s{seq_len}.{rknn_precision}.rk3576.rknn"
            )
        )
        for layer in sorted(rknn_layers)
    }
    timings: dict[str, Any] = {"layers": []}
    try:
        t0 = time.perf_counter()
        hidden = embedding.run(None, {"input_ids": padded_ids})[0].astype(np.float32, copy=False)
        timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        kv_cache: dict[str, np.ndarray] = {}
        for layer in range(12):
            layer_start = time.perf_counter()
            attn_start = time.perf_counter()
            attn_residual, key, value = attention[layer].run(
                None,
                {_attention_input_name(layer): hidden, "attention_mask": padded_mask},
            )
            attn_ms = (time.perf_counter() - attn_start) * 1000.0
            mlp_start = time.perf_counter()
            if layer in rknn_layers:
                if mlp_split == "mlp_only":
                    ln2_input_name = ln2_ort[layer].get_inputs()[0].name
                    normalized = ln2_ort[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    mlp_out = mlp_rknn[layer].run(np.asarray(normalized, dtype=np.float32))
                    mlp_kind = "rknn_mlp_only"
                elif mlp_split == "fc_in_act_only":
                    ln2_input_name = ln2_ort[layer].get_inputs()[0].name
                    normalized = ln2_ort[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    activation = mlp_rknn[layer].run(np.asarray(normalized, dtype=np.float32))
                    fc_out_input_name = fc_out_ort[layer].get_inputs()[0].name
                    mlp_out = fc_out_ort[layer].run(None, {fc_out_input_name: np.asarray(activation, dtype=np.float32)})[0]
                    mlp_kind = "rknn_fc_in_act_only"
                elif mlp_split == "fc_out_only":
                    ln2_input_name = ln2_ort[layer].get_inputs()[0].name
                    normalized = ln2_ort[layer].run(None, {ln2_input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                    fc_in_input_name = fc_in_act_ort[layer].get_inputs()[0].name
                    activation = fc_in_act_ort[layer].run(None, {fc_in_input_name: np.asarray(normalized, dtype=np.float32)})[0]
                    mlp_out = mlp_rknn[layer].run(np.asarray(activation, dtype=np.float32))
                    mlp_kind = "rknn_fc_out_only"
                else:
                    mlp_out = mlp_rknn[layer].run(np.asarray(attn_residual, dtype=np.float32))
                    mlp_kind = "rknn_ln2_mlp"
            else:
                input_name = mlp_ort[layer].get_inputs()[0].name
                mlp_out = mlp_ort[layer].run(None, {input_name: np.asarray(attn_residual, dtype=np.float32)})[0]
                mlp_kind = "ort"
            mlp_ms = (time.perf_counter() - mlp_start) * 1000.0
            hidden = (np.asarray(attn_residual, dtype=np.float32) + np.asarray(mlp_out, dtype=np.float32)) * mask3
            kv_cache[f"present_key_{layer}"] = _trim_kv_to_length(key, actual_len)
            kv_cache[f"present_value_{layer}"] = _trim_kv_to_length(value, actual_len)
            timings["layers"].append(
                {
                    "layer": layer,
                    "attention_ms": round(attn_ms, 3),
                    "mlp_kind": mlp_kind,
                    "mlp_ms": round(mlp_ms, 3),
                    "layer_ms": round((time.perf_counter() - layer_start) * 1000.0, 3),
                }
            )
        final_start = time.perf_counter()
        ln_f = final_norm.run(None, {"/Mul_88_output_0": hidden})[0]
        global_hidden = np.asarray(ln_f, dtype=np.float32) * mask3
        timings["final_norm_ms"] = round((time.perf_counter() - final_start) * 1000.0, 3)
        timings["composed_prefill_ms"] = round(
            timings["embedding_ms"] + sum(float(item["layer_ms"]) for item in timings["layers"]) + timings["final_norm_ms"],
            3,
        )
        return global_hidden, kv_cache, timings
    finally:
        for session in mlp_rknn.values():
            session.release()


def _run_sampler(
    backend: Any,
    hidden: np.ndarray,
    repetition_seen_mask: np.ndarray,
    assistant_u: float,
    audio_u: np.ndarray,
    debug_session: Any | None = None,
    debug_specs: list[dict[str, str]] | None = None,
) -> tuple[int, np.ndarray, float, dict[str, np.ndarray]]:
    t0 = time.perf_counter()
    session = debug_session or backend._sampler
    output_names = ["should_continue", "frame_token_ids"]
    if debug_specs:
        for spec in debug_specs:
            output_names.extend(
                [
                    spec["topk_values"],
                    spec["topk_indices"],
                    spec["final_probs"],
                    spec["final_cdf"],
                    spec["selected_token"],
                ]
            )
    outputs = session.run(
        output_names,
        {
            "global_hidden": hidden.astype(np.float32, copy=False),
            "repetition_seen_mask": repetition_seen_mask,
            "assistant_random_u": np.asarray([min(0.99999994, max(0.0, float(assistant_u)))], dtype=np.float32),
            "audio_random_u": np.asarray(audio_u, dtype=np.float32).reshape(1, 16),
        },
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    should_continue, frame_token_ids = outputs[0], outputs[1]
    debug: dict[str, np.ndarray] = {}
    if debug_specs:
        values = outputs[2:]
        cursor = 0
        for spec in debug_specs:
            for key in ("topk_values", "topk_indices", "final_probs", "final_cdf", "selected_token"):
                debug[f"ch{spec['channel']}.{key}"] = np.asarray(values[cursor])
                cursor += 1
    return int(np.asarray(should_continue).reshape(-1)[0]), np.asarray(frame_token_ids, dtype=np.int32).reshape(16), elapsed_ms, debug


def _rank_of(indices: np.ndarray, token_id: int) -> int | None:
    hits = np.flatnonzero(np.asarray(indices).reshape(-1).astype(np.int64) == int(token_id))
    return int(hits[0]) if hits.size else None


def _cdf_margin(cdf: np.ndarray, rank: int | None, random_u: float) -> dict[str, Any]:
    values = np.asarray(cdf, dtype=np.float32).reshape(-1)
    if rank is None or rank < 0 or rank >= values.size:
        return {"cdf_low": None, "cdf_high": None, "random_margin": None}
    high = float(values[rank])
    low = float(values[rank - 1]) if rank > 0 else 0.0
    return {
        "cdf_low": low,
        "cdf_high": high,
        "random_margin": min(abs(float(random_u) - low), abs(high - float(random_u))),
    }


def _sampler_margin_report(
    full_debug: dict[str, np.ndarray],
    hybrid_debug: dict[str, np.ndarray],
    full_frame: np.ndarray,
    hybrid_frame: np.ndarray,
    audio_u: np.ndarray,
) -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    for channel in range(16):
        full_indices = np.asarray(full_debug[f"ch{channel}.topk_indices"]).reshape(-1)
        hybrid_indices = np.asarray(hybrid_debug[f"ch{channel}.topk_indices"]).reshape(-1)
        full_values = np.asarray(full_debug[f"ch{channel}.topk_values"], dtype=np.float32).reshape(-1)
        hybrid_values = np.asarray(hybrid_debug[f"ch{channel}.topk_values"], dtype=np.float32).reshape(-1)
        full_probs = np.asarray(full_debug[f"ch{channel}.final_probs"], dtype=np.float32).reshape(-1)
        hybrid_probs = np.asarray(hybrid_debug[f"ch{channel}.final_probs"], dtype=np.float32).reshape(-1)
        full_token = int(full_frame[channel])
        hybrid_token = int(hybrid_frame[channel])
        full_rank = _rank_of(full_indices, full_token)
        hybrid_rank = _rank_of(hybrid_indices, hybrid_token)
        full_rank_of_hybrid = _rank_of(full_indices, hybrid_token)
        hybrid_rank_of_full = _rank_of(hybrid_indices, full_token)
        full_cdf = _cdf_margin(full_debug[f"ch{channel}.final_cdf"], full_rank, float(audio_u[channel]))
        hybrid_cdf = _cdf_margin(hybrid_debug[f"ch{channel}.final_cdf"], hybrid_rank, float(audio_u[channel]))
        channels.append(
            {
                "channel": channel,
                "token_equal": full_token == hybrid_token,
                "random_u": float(audio_u[channel]),
                "full_token": full_token,
                "hybrid_token": hybrid_token,
                "full_rank": full_rank,
                "hybrid_rank": hybrid_rank,
                "full_rank_of_hybrid_token": full_rank_of_hybrid,
                "hybrid_rank_of_full_token": hybrid_rank_of_full,
                "full_selected_prob": float(full_probs[full_rank]) if full_rank is not None and full_rank < full_probs.size else None,
                "hybrid_selected_prob": float(hybrid_probs[hybrid_rank]) if hybrid_rank is not None and hybrid_rank < hybrid_probs.size else None,
                "full_random_margin": full_cdf["random_margin"],
                "hybrid_random_margin": hybrid_cdf["random_margin"],
                "full_cdf_low": full_cdf["cdf_low"],
                "full_cdf_high": full_cdf["cdf_high"],
                "hybrid_cdf_low": hybrid_cdf["cdf_low"],
                "hybrid_cdf_high": hybrid_cdf["cdf_high"],
                "full_top1_token": int(full_indices[0]) if full_indices.size else None,
                "hybrid_top1_token": int(hybrid_indices[0]) if hybrid_indices.size else None,
                "full_top1_prob": float(full_probs[0]) if full_probs.size else None,
                "hybrid_top1_prob": float(hybrid_probs[0]) if hybrid_probs.size else None,
                "full_top1_top2_prob_margin": float(full_probs[0] - full_probs[1]) if full_probs.size > 1 else None,
                "hybrid_top1_top2_prob_margin": float(hybrid_probs[0] - hybrid_probs[1]) if hybrid_probs.size > 1 else None,
                "full_top1_top2_logit_margin": float(full_values[0] - full_values[1]) if full_values.size > 1 else None,
                "hybrid_top1_top2_logit_margin": float(hybrid_values[0] - hybrid_values[1]) if hybrid_values.size > 1 else None,
            }
        )

    margins = [
        item["full_random_margin"]
        for item in channels
        if item["full_random_margin"] is not None
    ]
    mismatches = [item for item in channels if not item["token_equal"]]
    return {
        "channels": channels,
        "summary": {
            "mismatched_channels": len(mismatches),
            "full_random_margin_min": min(margins) if margins else None,
            "mismatched_full_random_margin_min": min(
                [item["full_random_margin"] for item in mismatches if item["full_random_margin"] is not None],
                default=None,
            ),
            "mismatched_full_token_missing_from_hybrid_topk": sum(
                1 for item in mismatches if item["hybrid_rank_of_full_token"] is None
            ),
            "mismatched_hybrid_token_missing_from_full_topk": sum(
                1 for item in mismatches if item["full_rank_of_hybrid_token"] is None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--rknn-dir", type=Path, help="Directory for RKNN MLP island artifacts; defaults to --artifact-dir")
    parser.add_argument("--text", default="你好")
    parser.add_argument("--voice", default="Lingyu")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--prefill-seq", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rknn-layers", default="all", help="Layer set for RKNN MLP: all, none, 0-3, or 0,4,8")
    parser.add_argument("--rknn-precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    parser.add_argument("--mlp-split", default="ln2_mlp", choices=["ln2_mlp", "mlp_only", "fc_in_act_only", "fc_out_only"])
    parser.add_argument("--sampler-debug", action="store_true", help="Expose sampler TopK/CDF margins for each audio channel")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    _set_env(args)
    from rkvoice_stream.backends.tts.moss_ort import MossORTBackend

    backend = MossORTBackend()
    t0 = time.perf_counter()
    backend.preload()
    preload_ms = (time.perf_counter() - t0) * 1000.0
    try:
        debug_session = None
        debug_specs = None
        debug_model_path = None
        if args.sampler_debug:
            debug_session, debug_specs, debug_model_path = _make_sampler_debug_session(args.model_dir, args.threads)

        input_ids, mode = backend._build_prefill_rows(args.text)
        attention_mask = (input_ids[:, :, 0] != backend._pad_token_id()).astype(np.int32)
        if not attention_mask.any():
            attention_mask[:] = 1
        past_len = int(attention_mask.sum())
        rknn_layers = _parse_rknn_layers(args.rknn_layers)

        names = backend._tts_meta["onnx"]["prefill_output_names"]
        full_t0 = time.perf_counter()
        full_outputs = backend._prefill.run(names, {"input_ids": input_ids, "attention_mask": attention_mask})
        full_prefill_ms = (time.perf_counter() - full_t0) * 1000.0
        full_hidden_all = full_outputs[0]
        full_kv = {names[i]: full_outputs[i] for i in range(1, len(names))}

        hybrid_t0 = time.perf_counter()
        hybrid_hidden_all, hybrid_kv, hybrid_timings = _compose_prefill(
            args.artifact_dir,
            args.rknn_dir or args.artifact_dir,
            args.seq_len,
            args.threads,
            input_ids,
            attention_mask,
            rknn_layers,
            args.rknn_precision,
            args.mlp_split,
        )
        hybrid_prefill_ms = (time.perf_counter() - hybrid_t0) * 1000.0

        full_hidden = full_hidden_all[:, past_len - 1, :].astype(np.float32, copy=False)
        hybrid_hidden = hybrid_hidden_all[:, past_len - 1, :].astype(np.float32, copy=False)
        hidden_metrics = _metrics(full_hidden, hybrid_hidden)

        rng = np.random.default_rng(args.seed)
        full_seen = np.zeros((1, 16, 1024), dtype=np.int32)
        hybrid_seen = np.zeros((1, 16, 1024), dtype=np.int32)
        frame_reports: list[dict[str, Any]] = []
        full_current = full_hidden
        hybrid_current = hybrid_hidden
        full_past_len = past_len
        hybrid_past_len = past_len
        full_last: np.ndarray | None = None
        hybrid_last: np.ndarray | None = None

        for frame_index in range(args.frames):
            full_decode_ms = None
            hybrid_decode_ms = None
            if frame_index > 0:
                assert full_last is not None and hybrid_last is not None
                full_row = backend._make_audio_row(full_last).reshape(1, 1, 17)
                hybrid_row = backend._make_audio_row(hybrid_last).reshape(1, 1, 17)
                t_decode = time.perf_counter()
                full_current, full_kv = backend._decode_step(full_row, full_kv, full_past_len)
                full_decode_ms = (time.perf_counter() - t_decode) * 1000.0
                t_decode = time.perf_counter()
                hybrid_current, hybrid_kv = backend._decode_step(hybrid_row, hybrid_kv, hybrid_past_len)
                hybrid_decode_ms = (time.perf_counter() - t_decode) * 1000.0
                full_past_len += 1
                hybrid_past_len += 1

            assistant_u = float(rng.random())
            audio_u = np.asarray([float(rng.random()) for _ in range(16)], dtype=np.float32)
            full_continue, full_frame, full_sampler_ms, full_debug = _run_sampler(
                backend,
                full_current,
                full_seen,
                assistant_u,
                audio_u,
                debug_session=debug_session,
                debug_specs=debug_specs,
            )
            hybrid_continue, hybrid_frame, hybrid_sampler_ms, hybrid_debug = _run_sampler(
                backend,
                hybrid_current,
                hybrid_seen,
                assistant_u,
                audio_u,
                debug_session=debug_session,
                debug_specs=debug_specs,
            )

            for i, token_id in enumerate(full_frame):
                if 0 <= int(token_id) < full_seen.shape[2]:
                    full_seen[0, i, int(token_id)] = 1
            for i, token_id in enumerate(hybrid_frame):
                if 0 <= int(token_id) < hybrid_seen.shape[2]:
                    hybrid_seen[0, i, int(token_id)] = 1

            full_last = full_frame
            hybrid_last = hybrid_frame
            equal = bool(np.array_equal(full_frame, hybrid_frame) and full_continue == hybrid_continue)
            frame_report = {
                "frame_index": frame_index,
                "equal": equal,
                "token_mismatches": int(np.count_nonzero(full_frame != hybrid_frame)),
                "assistant_random_u": assistant_u,
                "audio_random_u": audio_u.tolist(),
                "full_should_continue": full_continue,
                "hybrid_should_continue": hybrid_continue,
                "full_tokens": full_frame.tolist(),
                "hybrid_tokens": hybrid_frame.tolist(),
                "current_hidden": _metrics(full_current, hybrid_current),
                "full_decode_ms": round(full_decode_ms, 3) if full_decode_ms is not None else None,
                "hybrid_decode_ms": round(hybrid_decode_ms, 3) if hybrid_decode_ms is not None else None,
                "full_sampler_ms": round(full_sampler_ms, 3),
                "hybrid_sampler_ms": round(hybrid_sampler_ms, 3),
            }
            if debug_specs:
                frame_report["sampler_margins"] = _sampler_margin_report(
                    full_debug,
                    hybrid_debug,
                    full_frame,
                    hybrid_frame,
                    audio_u,
                )
            frame_reports.append(frame_report)
            if full_continue == 0 or hybrid_continue == 0:
                break

        passed = bool(all(item["equal"] for item in frame_reports))
        report = {
            "model_dir": str(args.model_dir),
            "artifact_dir": str(args.artifact_dir),
            "rknn_dir": str(args.rknn_dir or args.artifact_dir),
            "text": args.text,
            "voice": args.voice,
            "mode": mode,
            "actual_len": past_len,
            "seq_len": args.seq_len,
            "seed": args.seed,
            "rknn_layers": sorted(rknn_layers),
            "rknn_precision": args.rknn_precision,
            "mlp_split": args.mlp_split,
            "timings_ms": {
                "preload": round(preload_ms, 3),
                "full_prefill": round(full_prefill_ms, 3),
                "composed_prefill_wall": round(hybrid_prefill_ms, 3),
                "composed_prefill_reported": hybrid_timings,
            },
            "sampler_debug": {
                "enabled": bool(args.sampler_debug),
                "debug_model": str(debug_model_path) if debug_model_path else None,
            },
            "first_hidden": hidden_metrics,
            "frames": frame_reports,
            "gates": {"tokens_match": passed, "passed": passed},
        }
    finally:
        backend.cleanup()
        if "debug_model_path" in locals() and debug_model_path is not None:
            try:
                Path(debug_model_path).unlink(missing_ok=True)
            except OSError:
                pass

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

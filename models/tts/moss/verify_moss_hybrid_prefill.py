#!/usr/bin/env python3
"""Verify composed MOSS hybrid prefill against full ONNX prefill on RK3576."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from rknnlite.api import RKNNLite


def _attention_input_name(layer: int) -> str:
    if layer == 0:
        return "/Add_15_output_0"
    return f"/Mul_{22 + (layer - 1) * 6}_output_0"


def _suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


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


def _write_augmented_prefill(model_dir: Path, out_path: Path) -> Path:
    model = onnx.load(str(model_dir / "moss_tts_prefill.onnx"), load_external_data=False)
    existing = {out.name for out in model.graph.output}
    names = ["global_hidden"]
    for layer in range(12):
        names.extend([f"present_key_{layer}", f"present_value_{layer}"])
    for name in names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
            existing.add(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_path))
    data = model_dir / "moss_tts_global_shared.data"
    if data.exists():
        target = out_path.parent / data.name
        if not target.exists():
            target.write_bytes(data.read_bytes())
    return out_path


def _build_prefill_inputs(model_dir: Path, text: str, seq_len: int, voice: str) -> tuple[np.ndarray, np.ndarray, int]:
    os.environ["MOSS_ORT_MODEL_DIR"] = str(model_dir)
    os.environ["MOSS_ORT_VOICE"] = voice
    from rkvoice_stream.backends.tts.moss_ort import MossORTBackend

    backend = MossORTBackend()
    backend._tts_meta = json.loads((model_dir / "tts_browser_onnx_meta.json").read_text(encoding="utf-8"))
    backend._config = dict(backend._tts_meta.get("model_config") or {})
    manifest_path = model_dir / "browser_poc_manifest.json"
    if manifest_path.exists():
        backend._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend._config.update(backend._manifest.get("tts_config") or {})
    backend._load_tokenizer()
    rows, _mode = backend._build_prefill_rows(text)
    actual_len = int(rows.shape[1])
    if actual_len > seq_len:
        raise RuntimeError(f"prefill rows {actual_len} exceed seq_len {seq_len}")
    padded = np.full((1, seq_len, 17), backend._audio_pad_token_id(), dtype=np.int32)
    padded[:, :actual_len, :] = rows
    attention_mask = np.zeros((1, seq_len), dtype=np.int32)
    attention_mask[:, :actual_len] = 1
    return padded, attention_mask, actual_len


def _ort_session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


class _RknnSession:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(str(path))
        if ret != 0:
            raise RuntimeError(f"load_rknn returned {ret}: {path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime returned {ret}: {path}")

    def run(self, x: np.ndarray) -> np.ndarray:
        out = self.rknn.inference(inputs=[x.astype(np.float32, copy=False)])
        if out is None:
            raise RuntimeError(f"RKNN inference returned None: {self.path}")
        return np.asarray(out[0], dtype=np.float32)

    def release(self) -> None:
        self.rknn.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--cattn-dir",
        type=Path,
        help="optional directory containing ln1 ONNX, cattn RKNN, and attn_after_cattn ONNX artifacts",
    )
    parser.add_argument(
        "--ln1-cattn-dir",
        type=Path,
        help="optional directory containing fused ln1+cattn RKNN artifacts; uses --cattn-dir for attention suffix when provided",
    )
    parser.add_argument("--text", default="你好")
    parser.add_argument("--voice", default="Lingyu")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/moss-hybrid-prefill"))
    parser.add_argument("--max-global-rel-l2", type=float, default=0.02)
    parser.add_argument("--min-global-cosine", type=float, default=0.999)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    input_ids, attention_mask, actual_len = _build_prefill_inputs(args.model_dir, args.text, args.seq_len, args.voice)
    mask3 = attention_mask[:, :, None].astype(np.float32)

    augmented = args.work_dir / f"moss_tts_prefill.hybrid_targets.s{args.seq_len}.onnx"
    _write_augmented_prefill(args.model_dir, augmented)
    target_names = ["global_hidden"]
    for layer in range(12):
        target_names.extend([f"present_key_{layer}", f"present_value_{layer}"])
    full = _ort_session(augmented, args.threads)
    t0 = time.perf_counter()
    target_outputs = full.run(target_names, {"input_ids": input_ids, "attention_mask": attention_mask})
    full_ms = (time.perf_counter() - t0) * 1000.0
    targets = dict(zip(target_names, target_outputs, strict=True))
    del full
    gc.collect()

    timings: dict[str, Any] = {"full_prefill_target_ms": round(full_ms, 3), "layers": []}
    preload_start = time.perf_counter()
    embedding = _ort_session(args.artifact_dir / f"moss_embedding_prefix.s{args.seq_len}.onnx", args.threads)
    final_norm = _ort_session(args.artifact_dir / f"moss_final_norm.s{args.seq_len}.onnx", args.threads)
    use_cattn = args.cattn_dir is not None or args.ln1_cattn_dir is not None
    use_ln1_cattn = args.ln1_cattn_dir is not None
    if use_cattn and not use_ln1_cattn:
        cattn_dir = args.cattn_dir
        ln1 = [
            _ort_session(cattn_dir / f"moss_block{layer}_ln1.s{args.seq_len}.onnx", args.threads)
            for layer in range(12)
        ]
        cattn = [
            _RknnSession(cattn_dir / f"moss_block{layer}_cattn.s{args.seq_len}.fp16.rk3576.rknn")
            for layer in range(12)
        ]
        attention_suffix = [
            _ort_session(cattn_dir / f"moss_block{layer}_attn_after_cattn.s{args.seq_len}.onnx", args.threads)
            for layer in range(12)
        ]
        ln1_cattn = []
    elif use_ln1_cattn:
        ln1_cattn_dir = args.ln1_cattn_dir
        suffix_dir = args.cattn_dir or args.ln1_cattn_dir
        ln1 = []
        cattn = []
        ln1_cattn = [
            _RknnSession(ln1_cattn_dir / f"moss_block{layer}_ln1_cattn.s{args.seq_len}.fp16.rk3576.rknn")
            for layer in range(12)
        ]
        attention_suffix = [
            _ort_session(suffix_dir / f"moss_block{layer}_attn_after_cattn.s{args.seq_len}.onnx", args.threads)
            for layer in range(12)
        ]
    else:
        ln1 = []
        cattn = []
        ln1_cattn = []
        attention_suffix = []
    attention = [] if use_cattn else [
        _ort_session(args.artifact_dir / f"moss_block{layer}_attn_residual.s{args.seq_len}.onnx", args.threads)
        for layer in range(12)
    ]
    mlp = [
        _RknnSession(args.artifact_dir / f"moss_block{layer}_ln2_mlp.s{args.seq_len}.fp16.rk3576.rknn")
        for layer in range(12)
    ]
    timings["hybrid_preload_ms"] = round((time.perf_counter() - preload_start) * 1000.0, 3)

    try:
        t0 = time.perf_counter()
        hidden = embedding.run(None, {"input_ids": input_ids})[0].astype(np.float32, copy=False)
        timings["embedding_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        kv_outputs: dict[str, np.ndarray] = {}
        for layer in range(12):
            layer_t0 = time.perf_counter()
            attn_t0 = time.perf_counter()
            if use_cattn:
                if use_ln1_cattn:
                    ln1_ms = None
                    cattn_t0 = time.perf_counter()
                    qkv = ln1_cattn[layer].run(np.asarray(hidden * mask3, dtype=np.float32))
                    cattn_ms = (time.perf_counter() - cattn_t0) * 1000.0
                else:
                    ln1_t0 = time.perf_counter()
                    ln1_name = ln1[layer].get_inputs()[0].name
                    normalized_attn = ln1[layer].run(None, {ln1_name: hidden * mask3})[0]
                    ln1_ms = (time.perf_counter() - ln1_t0) * 1000.0
                    cattn_t0 = time.perf_counter()
                    qkv = cattn[layer].run(np.asarray(normalized_attn, dtype=np.float32))
                    cattn_ms = (time.perf_counter() - cattn_t0) * 1000.0
                suffix_t0 = time.perf_counter()
                suffix_inputs = {
                    f"/c_attn{_suffix(layer)}/Add_output_0": qkv,
                    _attention_input_name(layer): hidden,
                    "attention_mask": attention_mask,
                }
                attn_residual, key, value = attention_suffix[layer].run(None, suffix_inputs)
                suffix_ms = (time.perf_counter() - suffix_t0) * 1000.0
                attn_ms = (ln1_ms or 0.0) + cattn_ms + suffix_ms
            else:
                ln1_ms = None
                cattn_ms = None
                suffix_ms = None
                attn_residual, key, value = attention[layer].run(
                    None,
                    {_attention_input_name(layer): hidden, "attention_mask": attention_mask},
                )
                attn_ms = (time.perf_counter() - attn_t0) * 1000.0
            mlp_t0 = time.perf_counter()
            mlp_out = mlp[layer].run(np.asarray(attn_residual, dtype=np.float32))
            mlp_ms = (time.perf_counter() - mlp_t0) * 1000.0
            hidden = (np.asarray(attn_residual, dtype=np.float32) + mlp_out) * mask3
            kv_outputs[f"present_key_{layer}"] = np.asarray(key, dtype=np.float32)
            kv_outputs[f"present_value_{layer}"] = np.asarray(value, dtype=np.float32)
            timings["layers"].append(
                {
                    "layer": layer,
                    "ln1_ms": round(ln1_ms, 3) if ln1_ms is not None else None,
                    "cattn_rknn_ms": round(cattn_ms, 3) if cattn_ms is not None and not use_ln1_cattn else None,
                    "ln1_cattn_rknn_ms": round(cattn_ms, 3) if cattn_ms is not None and use_ln1_cattn else None,
                    "attention_suffix_ms": round(suffix_ms, 3) if suffix_ms is not None else None,
                    "attention_ms": round(attn_ms, 3),
                    "mlp_rknn_ms": round(mlp_ms, 3),
                    "layer_ms": round((time.perf_counter() - layer_t0) * 1000.0, 3),
                }
            )
        final_t0 = time.perf_counter()
        ln_f = final_norm.run(None, {"/Mul_88_output_0": hidden})[0]
        hybrid_global = np.asarray(ln_f, dtype=np.float32) * mask3
        timings["final_norm_ms"] = round((time.perf_counter() - final_t0) * 1000.0, 3)
        timings["hybrid_prefill_ms"] = round(
            timings["embedding_ms"]
            + sum(item["layer_ms"] for item in timings["layers"])
            + timings["final_norm_ms"],
            3,
        )
    finally:
        for item in mlp:
            item.release()
        for item in cattn:
            item.release()
        for item in ln1_cattn:
            item.release()

    outputs = {"global_hidden": _metrics(targets["global_hidden"], hybrid_global)}
    kv_metrics = {}
    for layer in range(12):
        for kind in ("key", "value"):
            name = f"present_{kind}_{layer}"
            kv_metrics[name] = _metrics(targets[name], kv_outputs[name])
    outputs["kv_max_rel_l2"] = max(item["rel_l2"] for item in kv_metrics.values())
    outputs["kv_min_cosine"] = min(item["cosine"] for item in kv_metrics.values())

    global_ok = (
        outputs["global_hidden"]["finite"]
        and outputs["global_hidden"]["rel_l2"] <= args.max_global_rel_l2
        and outputs["global_hidden"]["cosine"] >= args.min_global_cosine
    )
    passed = bool(global_ok and all(item["finite"] for item in kv_metrics.values()))
    report = {
        "model_dir": str(args.model_dir),
        "artifact_dir": str(args.artifact_dir),
        "cattn_dir": str(args.cattn_dir) if args.cattn_dir is not None else None,
        "ln1_cattn_dir": str(args.ln1_cattn_dir) if args.ln1_cattn_dir is not None else None,
        "use_cattn": use_cattn,
        "use_ln1_cattn": use_ln1_cattn,
        "text": args.text,
        "voice": args.voice,
        "seq_len": args.seq_len,
        "actual_len": actual_len,
        "timings_ms": timings,
        "outputs": outputs,
        "kv_metrics": kv_metrics,
        "gates": {
            "max_global_rel_l2": args.max_global_rel_l2,
            "min_global_cosine": args.min_global_cosine,
            "passed": passed,
        },
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

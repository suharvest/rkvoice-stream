#!/usr/bin/env python3
"""Verify MOSS MLP RKNN islands on real prefill hidden tensors."""

from __future__ import annotations

import argparse
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


def _layer_suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


def _ln2_input_name(layer: int) -> str:
    return f"/Add_{19 + layer * 5}_output_0"


def _mlp_output_name(layer: int) -> str:
    return f"/mlp/fc_out{_layer_suffix(layer)}/Add_output_0"


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
        "ref_min": float(np.min(ref)),
        "ref_max": float(np.max(ref)),
        "got_min": float(np.min(got)),
        "got_max": float(np.max(got)),
    }


def _write_augmented_prefill(model_dir: Path, out_path: Path) -> Path:
    source = model_dir / "moss_tts_prefill.onnx"
    model = onnx.load(str(source), load_external_data=False)
    existing = {out.name for out in model.graph.output}
    for layer in range(12):
        for name in (_ln2_input_name(layer), _mlp_output_name(layer)):
            if name not in existing:
                model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
                existing.add(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(out_path))
    # Keep external data next to the augmented graph.
    for data_name in ("moss_tts_global_shared.data",):
        src = model_dir / data_name
        dst = out_path.parent / data_name
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
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


def _run_ort_outputs(augmented: Path, feeds: dict[str, np.ndarray], threads: int) -> dict[str, np.ndarray]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(augmented), sess_options=opts, providers=["CPUExecutionProvider"])
    names = []
    for layer in range(12):
        names.extend([_ln2_input_name(layer), _mlp_output_name(layer)])
    outputs = sess.run(names, feeds)
    return dict(zip(names, outputs, strict=True))


def _run_island_ort(onnx_path: Path, x: np.ndarray, repeat: int, threads: int) -> tuple[np.ndarray, float]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    y = sess.run(None, {input_name: x})[0]
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        times.append((time.perf_counter() - t0) * 1000.0)
    return y, float(np.mean(times)) if times else 0.0


def _run_island_rknn(rknn_path: Path, x: np.ndarray, repeat: int) -> tuple[np.ndarray, float]:
    r = RKNNLite(verbose=False)
    try:
        ret = r.load_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn returned {ret}: {rknn_path}")
        ret = r.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime returned {ret}: {rknn_path}")
        y = r.inference(inputs=[x])
        if y is None:
            raise RuntimeError(f"rknn.inference returned None: {rknn_path}")
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = r.inference(inputs=[x])
            if out is None:
                raise RuntimeError(f"repeat rknn.inference returned None: {rknn_path}")
            times.append((time.perf_counter() - t0) * 1000.0)
        return np.asarray(y[0]), float(np.mean(times)) if times else 0.0
    finally:
        try:
            r.release()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rknn-dir", required=True, type=Path)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--voice", default="Lingyu")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--max-rel-l2", type=float, default=0.01)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--work-dir", type=Path, help="Directory for temporary augmented ONNX/external data.")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    work_dir = args.work_dir or args.rknn_dir
    augmented = work_dir / f"moss_tts_prefill.real_mlp_outputs.s{args.seq_len}.onnx"
    _write_augmented_prefill(args.model_dir, augmented)
    input_ids, attention_mask, actual_len = _build_prefill_inputs(args.model_dir, args.text, args.seq_len, args.voice)
    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}

    t0 = time.perf_counter()
    tensors = _run_ort_outputs(augmented, feeds, args.threads)
    full_prefill_probe_ms = (time.perf_counter() - t0) * 1000.0

    layers = []
    for layer in range(12):
        x = np.asarray(tensors[_ln2_input_name(layer)], dtype=np.float32)
        ref_full = np.asarray(tensors[_mlp_output_name(layer)], dtype=np.float32)
        onnx_path = args.rknn_dir / f"moss_block{layer}_ln2_mlp.s{args.seq_len}.onnx"
        rknn_path = args.rknn_dir / f"moss_block{layer}_ln2_mlp.s{args.seq_len}.fp16.rk3576.rknn"
        ref_island, ort_ms = _run_island_ort(onnx_path, x, args.repeat, args.threads)
        got, rknn_ms = _run_island_rknn(rknn_path, x, args.repeat)
        layers.append(
            {
                "layer": layer,
                "input_shape": list(x.shape),
                "input_min": float(np.min(x)),
                "input_max": float(np.max(x)),
                "full_vs_island_ort": _metrics(ref_full, ref_island),
                "full_vs_rknn": _metrics(ref_full, got),
                "island_ort_vs_rknn": _metrics(ref_island, got),
                "latency_ms": {
                    "ort_avg": round(ort_ms, 3),
                    "rknn_avg": round(rknn_ms, 3),
                },
            }
        )

    passed = all(
        layer["full_vs_rknn"]["finite"]
        and float(layer["full_vs_rknn"]["rel_l2"]) <= args.max_rel_l2
        and float(layer["full_vs_rknn"]["cosine"]) >= args.min_cosine
        for layer in layers
    )
    sum_ort = float(sum(layer["latency_ms"]["ort_avg"] for layer in layers))
    sum_rknn = float(sum(layer["latency_ms"]["rknn_avg"] for layer in layers))
    report = {
        "model_dir": str(args.model_dir),
        "rknn_dir": str(args.rknn_dir),
        "text": args.text,
        "voice": args.voice,
        "seq_len": args.seq_len,
        "actual_len": actual_len,
        "full_prefill_probe_ms": round(full_prefill_probe_ms, 3),
        "summary": {
            "sum_ort_avg_ms": round(sum_ort, 3),
            "sum_rknn_avg_ms": round(sum_rknn, 3),
            "estimated_mlp_saving_ms": round(sum_ort - sum_rknn, 3),
            "mean_rel_l2": round(float(np.mean([layer["full_vs_rknn"]["rel_l2"] for layer in layers])), 6),
            "min_cosine": float(min(layer["full_vs_rknn"]["cosine"] for layer in layers)),
        },
        "gates": {
            "max_rel_l2": args.max_rel_l2,
            "min_cosine": args.min_cosine,
            "passed": passed,
        },
        "layers": layers,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

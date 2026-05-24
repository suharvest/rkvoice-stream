#!/usr/bin/env python3
"""Verify a MOSS prefill attention/residual ONNX slice against full prefill."""

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


def _attention_input_name(layer: int) -> str:
    if layer == 0:
        return "/Add_15_output_0"
    return f"/Mul_{22 + (layer - 1) * 6}_output_0"


def _attn_residual_name(layer: int) -> str:
    return f"/Add_{19 + layer * 5}_output_0"


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


def _write_augmented_prefill(model_dir: Path, out_path: Path, layer: int) -> Path:
    model = onnx.load(str(model_dir / "moss_tts_prefill.onnx"), load_external_data=False)
    existing = {out.name for out in model.graph.output}
    names = [
        _attention_input_name(layer),
        _attn_residual_name(layer),
        f"present_key_{layer}",
        f"present_value_{layer}",
    ]
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


def _session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--slice-onnx", required=True, type=Path)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--voice", default="Lingyu")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/moss-attn-slice"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    augmented = args.work_dir / f"moss_tts_prefill.attn_slice_l{args.layer}.s{args.seq_len}.onnx"
    _write_augmented_prefill(args.model_dir, augmented, args.layer)
    input_ids, attention_mask, actual_len = _build_prefill_inputs(args.model_dir, args.text, args.seq_len, args.voice)
    attn_input = _attention_input_name(args.layer)
    attn_output = _attn_residual_name(args.layer)
    key_name = f"present_key_{args.layer}"
    value_name = f"present_value_{args.layer}"

    full = _session(augmented, args.threads)
    t0 = time.perf_counter()
    full_outputs = full.run(
        [attn_input, attn_output, key_name, value_name],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )
    full_ms = (time.perf_counter() - t0) * 1000.0

    sliced = _session(args.slice_onnx, args.threads)
    t0 = time.perf_counter()
    slice_outputs = sliced.run(None, {attn_input: full_outputs[0], "attention_mask": attention_mask})
    slice_ms = (time.perf_counter() - t0) * 1000.0

    outputs = {
        attn_output: _metrics(full_outputs[1], slice_outputs[0]),
        key_name: _metrics(full_outputs[2], slice_outputs[1]),
        value_name: _metrics(full_outputs[3], slice_outputs[2]),
    }
    passed = all(item["finite"] and item["rel_l2"] <= 1e-5 and item["cosine"] >= 0.99999 for item in outputs.values())
    report = {
        "model_dir": str(args.model_dir),
        "slice_onnx": str(args.slice_onnx),
        "layer": args.layer,
        "text": args.text,
        "voice": args.voice,
        "seq_len": args.seq_len,
        "actual_len": actual_len,
        "latency_ms": {"full_prefill_augmented": round(full_ms, 3), "slice": round(slice_ms, 3)},
        "outputs": outputs,
        "gates": {"max_rel_l2": 1e-5, "min_cosine": 0.99999, "passed": passed},
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

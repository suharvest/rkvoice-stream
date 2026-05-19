#!/usr/bin/env python3
"""Dump real Kokoro hybrid tail inputs from prefix ORT + front RKNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite

from rkvoice_stream.backends.tts.kokoro_rknn import (
    DECODER_INPUT,
    FRONT_OUTPUT,
    STYLE_SLICE,
    _KokoroTokenizer,
)


def _session(path: Path) -> ort.InferenceSession:
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opt, providers=["CPUExecutionProvider"])


def _select(sess: ort.InferenceSession, outputs: list[np.ndarray], name: str, fallback: int) -> np.ndarray:
    names = [item.name for item in sess.get_outputs()]
    if name in names:
        return np.asarray(outputs[names.index(name)], dtype=np.float32)
    return np.asarray(outputs[fallback], dtype=np.float32)


def _style(model_dir: Path, style_dim: int) -> np.ndarray:
    for name in ("default.npy", "style.npy", "voices.npy"):
        path = model_dir / name
        if not path.exists():
            continue
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] == style_dim:
            return arr[:1].copy()
    return np.zeros((1, style_dim), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="kokoro-prefix-cpu.onnx")
    parser.add_argument("--front", default="rk3588/kokoro-decoder-front.int8.rknn")
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--style-dim", type=int, default=256)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()

    tokenizer = _KokoroTokenizer(args.model_dir)
    tokenizer.load()
    style = _style(args.model_dir, args.style_dim)
    prefix = _session(args.model_dir / args.prefix)
    front = RKNNLite()
    ret = front.load_rknn(str(args.model_dir / args.front))
    if ret != 0:
        raise RuntimeError(f"load_rknn returned {ret}")
    ret = front.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime returned {ret}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, text in enumerate(args.texts):
        tokens, n_tokens = tokenizer.encode(text, args.seq_len)
        speed = np.asarray([1.0], dtype=np.float32)
        feed = {}
        values = {"tokens": tokens, "style": style, "speed": speed}
        for item in prefix.get_inputs():
            feed[item.name] = values[item.name]
        prefix_outputs = prefix.run(None, feed)
        decoder_input = _select(prefix, prefix_outputs, DECODER_INPUT, 0)
        style_slice = _select(prefix, prefix_outputs, STYLE_SLICE, 1)
        front_outputs = front.inference(inputs=[decoder_input, style_slice])
        if not front_outputs:
            raise RuntimeError("front RKNN returned no outputs")
        hidden = np.asarray(front_outputs[0], dtype=np.float32)
        paths = []
        for input_idx, arr in enumerate((hidden, style_slice)):
            path = args.out_dir / f"sample{idx:03d}_input{input_idx}.npy"
            np.save(path, arr)
            paths.append(str(path))
        manifest.append({"text": text, "num_tokens": n_tokens, "inputs": paths})
    front.release()
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

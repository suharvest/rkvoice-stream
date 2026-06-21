#!/usr/bin/env python3
"""Smoke MOSS streaming synthesis through RKLLM global hidden + ORT sampler/codec."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from rkvoice_stream.backends.tts.moss_ort import MossORTBackend
from rkvoice_stream.runtime.rkllm_wrapper import RKLLM_INFER_GET_LAST_HIDDEN_LAYER, RKLLMTalker


class MossRkllmEmbedder:
    def __init__(self, assets_path: Path, audio_pad_token_id: int = 1024) -> None:
        assets = np.load(assets_path)
        self.embed_tokens = assets["embed_tokens"].astype(np.float32)
        self.audio_embeddings = assets["audio_embeddings"].astype(np.float32)
        self.final_norm_bias = assets["final_norm_bias"].astype(np.float32).reshape(1, -1)
        self.audio_pad_token_id = int(audio_pad_token_id)

    def rows_to_embeddings(self, rows: np.ndarray) -> np.ndarray:
        arr = np.asarray(rows, dtype=np.int32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != 17:
            raise ValueError(f"expected rows [T,17] or [1,T,17], got {arr.shape}")
        hidden = self.embed_tokens[arr[:, 0]]
        for idx in range(16):
            codes = arr[:, idx + 1]
            mask = codes != self.audio_pad_token_id
            if np.any(mask):
                safe = np.clip(codes, 0, self.audio_embeddings.shape[1] - 1)
                hidden = hidden + self.audio_embeddings[idx, safe] * mask[:, None].astype(np.float32)
        return np.ascontiguousarray(hidden, dtype=np.float32)

    def apply_final_bias(self, hidden: np.ndarray) -> np.ndarray:
        return np.asarray(hidden, dtype=np.float32) + self.final_norm_bias


def smoke(
    model_dir: Path,
    rkllm_model: Path,
    assets: Path,
    text: str,
    max_new_frames: int,
    seed: int,
) -> dict[str, Any]:
    os.environ.setdefault("MOSS_ORT_MODEL_DIR", str(model_dir))
    os.environ.setdefault("MOSS_ORT_MAX_NEW_FRAMES", str(max_new_frames))
    backend = MossORTBackend()
    rkllm: RKLLMTalker | None = None
    report: dict[str, Any] = {
        "model_dir": str(model_dir.resolve()),
        "rkllm_model": str(rkllm_model.resolve()),
        "assets": str(assets.resolve()),
        "text": text,
        "max_new_frames": max_new_frames,
        "seed": seed,
        "passed": False,
    }
    try:
        t0 = time.perf_counter()
        backend.preload()
        report["ort_aux_preload_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        embedder = MossRkllmEmbedder(assets, audio_pad_token_id=backend._audio_pad_token_id())
        t0 = time.perf_counter()
        rkllm = RKLLMTalker(str(rkllm_model), max_context_len=512, max_new_tokens=1)
        report["rkllm_load_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        input_ids, mode = backend._build_prefill_rows(text)
        prefill_embeddings = embedder.rows_to_embeddings(input_ids)
        t0 = time.perf_counter()
        prefill = rkllm.run_embed(prefill_embeddings, mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER, keep_history=1)
        prefill_hidden = embedder.apply_final_bias(prefill["hidden"])
        report["prefill_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        current_hidden = prefill_hidden[-1:].astype(np.float32, copy=False)

        rng = np.random.default_rng(seed)
        repetition_seen_mask = np.zeros((1, 16, 1024), dtype=np.int32)
        frames: list[list[int]] = []
        token_ms: list[dict[str, Any]] = []
        if backend._codec_stream_session is not None:
            backend._codec_stream_session.reset()
        audio_chunks: list[np.ndarray] = []
        for frame_index in range(max_new_frames):
            if frame_index > 0:
                row = backend._make_audio_row(np.asarray(frames[-1], dtype=np.int32))
                t0 = time.perf_counter()
                decoded = rkllm.run_embed(
                    embedder.rows_to_embeddings(row.reshape(1, 1, 17)),
                    mode=RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
                    keep_history=1,
                )
                current_hidden = embedder.apply_final_bias(decoded["hidden"])
                decode_ms = (time.perf_counter() - t0) * 1000.0
            else:
                decode_ms = None

            t0 = time.perf_counter()
            should_continue, frame_token_ids = backend._sampler.run(
                None,
                {
                    "global_hidden": current_hidden.astype(np.float32, copy=False),
                    "repetition_seen_mask": repetition_seen_mask,
                    "assistant_random_u": np.asarray([min(0.99999994, max(0.0, float(rng.random())))], dtype=np.float32),
                    "audio_random_u": np.asarray(
                        [[min(0.99999994, max(0.0, float(rng.random()))) for _ in range(16)]],
                        dtype=np.float32,
                    ),
                },
            )
            sampler_ms = (time.perf_counter() - t0) * 1000.0
            frame = np.asarray(frame_token_ids, dtype=np.int32).reshape(16)
            for i, token_id in enumerate(frame):
                if 0 <= int(token_id) < repetition_seen_mask.shape[2]:
                    repetition_seen_mask[0, i, int(token_id)] = 1
            frames.append([int(x) for x in frame.tolist()])
            token_ms.append(
                {
                    "frame": frame_index,
                    "decode_ms": round(decode_ms, 3) if decode_ms is not None else None,
                    "sampler_ms": round(sampler_ms, 3),
                    "continue": int(np.asarray(should_continue).reshape(-1)[0]),
                }
            )
            if backend._codec_stream_session is not None:
                t0 = time.perf_counter()
                audio, audio_len = backend._codec_stream_session.run_frames([frames[-1]])
                codec_ms = (time.perf_counter() - t0) * 1000.0
                chunk = backend._normalize_audio(audio, np.asarray([audio_len], dtype=np.int32))
                audio_chunks.append(chunk)
                token_ms[-1]["codec_ms"] = round(codec_ms, 3)
                token_ms[-1]["samples"] = int(chunk.shape[0])
            if token_ms[-1]["continue"] == 0:
                break

        audio = np.concatenate(audio_chunks, axis=0) if audio_chunks else np.zeros((0, backend._channels), dtype=np.float32)
        report.update(
            {
                "mode": mode,
                "frames": len(frames),
                "tokens": frames,
                "token_ms": token_ms,
                "audio_shape": list(audio.shape),
                "audio_finite": bool(np.isfinite(audio).all()),
                "audio_rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
                "sample_rate": backend._sample_rate,
            }
        )
        report["passed"] = bool(frames and audio.size and report["audio_finite"] and report["audio_rms"] > 1e-5)
    except Exception as exc:  # noqa: BLE001 - smoke must report runtime failures.
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-16:],
        }
    finally:
        if rkllm is not None:
            rkllm.destroy()
        backend.cleanup()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--rkllm-model", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--text", default="你好，欢迎使用本地语音助手。")
    parser.add_argument("--max-new-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    report = smoke(args.model_dir, args.rkllm_model, args.assets, args.text, args.max_new_frames, args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

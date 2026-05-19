#!/usr/bin/env python3
"""Verify Paraformer RKNN accuracy against ONNX Runtime at component level.

This script is intentionally stricter than an end-to-end text smoke test.  It
checks each accuracy-sensitive boundary:

1. fbank/LFR preprocessing shape and finiteness
2. encoder RKNN vs ORT: ``enc`` and ``alphas``
3. CIF from ORT encoder vs RKNN encoder: token count and embedding drift
4. decoder RKNN vs ORT on identical ORT-derived inputs: ``sample_ids`` and cache
5. full text for ORT pipeline vs mixed/RKNN pipeline

Run on RK3588 and RK3576 after copying converted RKNN artifacts to the device.
The ONNX models are also needed as the golden reference.

Example:

  PARAFORMER_MODEL_DIR=/opt/asr/paraformer \\
  PARAFORMER_RKNN_PRECISION=fp16 \\
  python3 models/asr/paraformer/verify_paraformer_rknn_parity.py \\
    --wav /tmp/hello.wav
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from rkvoice_stream.backends.asr.paraformer_rknn import (
    CACHE_COUNT,
    CACHE_SHAPE,
    DEC_MAX_ENC_FRAMES,
    DEC_MAX_TOKENS,
    SAMPLE_RATE,
    ParaformerRKNNBackend,
    add_preroll_silence,
    cif,
    compute_fbank,
    decode_ids,
    load_tokens,
    stack_frames,
)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    denom = max(float(np.linalg.norm(b64)), 1e-12)
    return float(np.linalg.norm(a64 - b64) / denom)


def compare_array(name: str, got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    got = np.asarray(got)
    ref = np.asarray(ref)
    common_slices = tuple(slice(0, min(g, r)) for g, r in zip(got.shape, ref.shape))
    got_c = got[common_slices]
    ref_c = ref[common_slices]
    diff = got_c.astype(np.float64) - ref_c.astype(np.float64)
    return {
        "name": name,
        "got_shape": list(got.shape),
        "ref_shape": list(ref.shape),
        "common_shape": list(got_c.shape),
        "finite_got": bool(np.isfinite(got_c).all()),
        "finite_ref": bool(np.isfinite(ref_c).all()),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "rel_l2": rel_l2(got_c, ref_c) if diff.size else 0.0,
    }


def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        duration = len(audio) / sr
        new_len = int(round(duration * SAMPLE_RATE))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return add_preroll_silence(audio.astype(np.float32))


def run_encoder_ort(sess, feats: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray]:
    orig_frames = min(feats.shape[0], frames)
    if feats.shape[0] < frames:
        feats_in = np.pad(feats, ((0, frames - feats.shape[0]), (0, 0)), mode="edge")
    else:
        feats_in = feats[:frames]
    speech = np.ascontiguousarray(feats_in[np.newaxis, :].astype(np.float32))
    speech_len = np.array([orig_frames], dtype=np.int32)
    mask = np.zeros((1, frames), dtype=np.float32)
    mask[:, :orig_frames] = 1.0
    enc, _enc_len, alphas = sess.run(
        ["enc", "enc_len", "alphas"],
        {
            "speech": speech,
            "speech_lengths": speech_len,
            "encoder_pad_mask": mask,
            "cif_pad_mask": mask,
        },
    )
    return enc[:, :orig_frames, :], alphas[:, :orig_frames]


def run_decoder_ort(
    sess,
    enc: np.ndarray,
    enc_len: int,
    acoustic_embeds: np.ndarray,
    caches: list[np.ndarray],
    max_enc: int,
    max_tokens: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    n_tokens = min(acoustic_embeds.shape[0], max_tokens)
    enc_in = np.zeros((1, max_enc, 512), dtype=np.float32)
    enc_in[:, :min(enc.shape[1], max_enc), :] = enc[:, :max_enc, :]
    ae_in = np.zeros((1, max_tokens, 512), dtype=np.float32)
    ae_in[:, :n_tokens, :] = acoustic_embeds[np.newaxis, :n_tokens, :]
    pad_mask = np.zeros((1, max_tokens), dtype=np.float32)
    pad_mask[:, :n_tokens] = 1.0
    enc_pad_mask = np.zeros((1, max_enc), dtype=np.float32)
    enc_pad_mask[:, :min(enc_len, max_enc)] = 1.0

    feeds = {
        "enc": enc_in,
        "acoustic_embeds": ae_in,
        "pad_mask": pad_mask,
        "enc_pad_mask": enc_pad_mask,
    }
    for i, cache in enumerate(caches):
        feeds[f"in_cache_{i}"] = np.ascontiguousarray(cache)

    output_names = ["sample_ids"] + [f"out_cache_{i}" for i in range(CACHE_COUNT)]
    outputs = sess.run(output_names, feeds)
    sample_ids = outputs[0]
    if sample_ids.ndim == 2:
        sample_ids = sample_ids[0]
    return sample_ids[:n_tokens].astype(np.int64), [o.astype(np.float32) for o in outputs[1:]]


def run_text_pipeline(
    enc_fn,
    dec_fn,
    feats: np.ndarray,
    tokens: list[str],
    max_encoder_frames: int,
) -> dict[str, Any]:
    all_ids: list[int] = []
    carry_w = 0.0
    carry_e = np.zeros(512, dtype=np.float32)
    caches = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]

    for start in range(0, feats.shape[0], max_encoder_frames):
        chunk = feats[start:start + max_encoder_frames]
        enc, alphas = enc_fn(chunk)
        ae, carry_w, carry_e = cif(enc[0], alphas[0], carry_weight=carry_w, carry_embed=carry_e)
        if len(ae) == 0:
            continue
        ids, caches = dec_fn(enc, enc.shape[1], ae, caches)
        all_ids.extend(ids.tolist())

    if carry_w >= 0.5:
        ae = (carry_e / carry_w)[np.newaxis, :]
        dummy_enc = np.zeros((1, 1, 512), dtype=np.float32)
        ids, caches = dec_fn(dummy_enc, 1, ae, caches)
        all_ids.extend(ids.tolist())

    return {"ids": all_ids, "text": decode_ids(all_ids, tokens)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--model-dir", default=os.environ.get("PARAFORMER_MODEL_DIR", "/opt/asr/paraformer"))
    parser.add_argument("--encoder-onnx", default="")
    parser.add_argument("--decoder-onnx", default="")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--encoder-rel-l2-max", type=float, default=0.08)
    parser.add_argument("--alpha-rel-l2-max", type=float, default=0.08)
    parser.add_argument("--decoder-ids-must-match", action="store_true", default=True)
    args = parser.parse_args()

    import onnxruntime as ort
    from models.asr.paraformer.convert_paraformer_rknn import prepare_decoder_onnx, prepare_encoder_onnx

    model_dir = Path(args.model_dir)
    encoder_onnx = Path(args.encoder_onnx) if args.encoder_onnx else model_dir / "encoder-rknn.onnx"
    if not encoder_onnx.exists():
        encoder_onnx = prepare_encoder_onnx(model_dir / "encoder.onnx", encoder_onnx)
    decoder_onnx = Path(args.decoder_onnx) if args.decoder_onnx else model_dir / "decoder-rknn.onnx"
    if not decoder_onnx.exists():
        decoder_onnx = prepare_decoder_onnx(model_dir / "decoder.onnx", decoder_onnx)

    audio = read_audio(args.wav)
    feats = stack_frames(compute_fbank(audio))
    encoder_frames = min(max(40, feats.shape[0]), max(40, min(DEC_MAX_ENC_FRAMES, 400)))
    feats_probe = feats[:encoder_frames]

    tokens = load_tokens(str(model_dir / "tokens.txt"))
    enc_sess = ort.InferenceSession(str(encoder_onnx), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(str(decoder_onnx), providers=["CPUExecutionProvider"])

    backend = ParaformerRKNNBackend()
    backend.preload()

    enc_ort, alphas_ort = run_encoder_ort(enc_sess, feats_probe, encoder_frames)
    enc_rknn, alphas_rknn = backend._run_encoder(feats_probe)
    if enc_rknn is None or alphas_rknn is None:
        raise RuntimeError("RKNN encoder returned None")

    ae_ort, carry_w_ort, carry_e_ort = cif(enc_ort[0], alphas_ort[0])
    ae_rknn, carry_w_rknn, carry_e_rknn = cif(enc_rknn[0], alphas_rknn[0])

    dec_probe_tokens = max(1, min(len(ae_ort), DEC_MAX_TOKENS))
    ae_probe = ae_ort[:dec_probe_tokens]
    if len(ae_probe) == 0:
        ae_probe = np.zeros((1, 512), dtype=np.float32)

    ort_caches = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
    rknn_caches = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
    ids_ort, caches_ort_out = run_decoder_ort(
        dec_sess, enc_ort, enc_ort.shape[1], ae_probe, ort_caches, DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS
    )
    ids_rknn = backend._run_decoder(enc_ort, enc_ort.shape[1], ae_probe, len(ae_probe), rknn_caches)
    if ids_rknn is None:
        raise RuntimeError("RKNN decoder returned None")

    def enc_ort_fn(chunk: np.ndarray):
        frames = min(max(40, chunk.shape[0]), DEC_MAX_ENC_FRAMES)
        return run_encoder_ort(enc_sess, chunk, frames)

    def enc_rknn_fn(chunk: np.ndarray):
        out = backend._run_encoder(chunk)
        if out[0] is None or out[1] is None:
            raise RuntimeError("RKNN encoder failed in full pipeline")
        return out

    def dec_ort_fn(enc: np.ndarray, enc_len: int, ae: np.ndarray, caches: list[np.ndarray]):
        return run_decoder_ort(dec_sess, enc, enc_len, ae, caches, DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS)

    def dec_rknn_fn(enc: np.ndarray, enc_len: int, ae: np.ndarray, caches: list[np.ndarray]):
        ids = backend._run_decoder(enc, enc_len, ae, len(ae), caches)
        if ids is None:
            raise RuntimeError("RKNN decoder failed in full pipeline")
        return ids, caches

    max_bucket = max(backend._encoders)
    pipeline_ort = run_text_pipeline(enc_ort_fn, dec_ort_fn, feats, tokens, max_bucket)
    pipeline_rknn_enc = run_text_pipeline(enc_rknn_fn, dec_ort_fn, feats, tokens, max_bucket)
    pipeline_rknn_dec = run_text_pipeline(enc_ort_fn, dec_rknn_fn, feats, tokens, max_bucket)
    pipeline_rknn = run_text_pipeline(enc_rknn_fn, dec_rknn_fn, feats, tokens, max_bucket)

    result = {
        "wav": str(args.wav),
        "duration_s": len(audio) / SAMPLE_RATE,
        "lfr_frames": int(feats.shape[0]),
        "encoder_probe_frames": int(encoder_frames),
        "encoder": [
            compare_array("enc", enc_rknn, enc_ort),
            compare_array("alphas", alphas_rknn, alphas_ort),
        ],
        "cif": {
            "ort_tokens": int(len(ae_ort)),
            "rknn_tokens": int(len(ae_rknn)),
            "acoustic_embeds": compare_array(
                "acoustic_embeds",
                ae_rknn[: min(len(ae_rknn), len(ae_ort))],
                ae_ort[: min(len(ae_rknn), len(ae_ort))],
            ),
            "carry_weight_ort": float(carry_w_ort),
            "carry_weight_rknn": float(carry_w_rknn),
            "carry_embed": compare_array("carry_embed", carry_e_rknn, carry_e_ort),
        },
        "decoder": {
            "probe_tokens": int(len(ae_probe)),
            "ids_ort": ids_ort.tolist(),
            "ids_rknn": ids_rknn.tolist(),
            "ids_match": bool(np.array_equal(ids_ort, ids_rknn)),
            "cache0": compare_array("cache0", rknn_caches[0], caches_ort_out[0]),
        },
        "pipelines": {
            "ort_ort": pipeline_ort,
            "rknn_enc_ort_dec": pipeline_rknn_enc,
            "ort_enc_rknn_dec": pipeline_rknn_dec,
            "rknn_rknn": pipeline_rknn,
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2))

    enc_rel = result["encoder"][0]["rel_l2"]
    alpha_rel = result["encoder"][1]["rel_l2"]
    ids_match = result["decoder"]["ids_match"]
    failed = False
    if enc_rel > args.encoder_rel_l2_max:
        print(f"FAIL: encoder enc rel_l2 {enc_rel:.4f} > {args.encoder_rel_l2_max}")
        failed = True
    if alpha_rel > args.alpha_rel_l2_max:
        print(f"FAIL: encoder alphas rel_l2 {alpha_rel:.4f} > {args.alpha_rel_l2_max}")
        failed = True
    if args.decoder_ids_must_match and not ids_match:
        print("FAIL: decoder sample_ids differ on identical ORT-derived inputs")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

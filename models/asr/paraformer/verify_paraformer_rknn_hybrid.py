#!/usr/bin/env python3
"""Verify Paraformer hybrid RKNN/CPU accuracy.

This checks the split that keeps the overflow-prone tail of the encoder on CPU:

  fbank/LFR -> encoder prefix RKNN -> encoder suffix ONNX Runtime -> CIF -> decoder ONNX Runtime

The script intentionally compares every split boundary against ONNX Runtime so
we can tell whether a failure comes from the RKNN prefix, CPU suffix, CIF, or
decoder inputs.
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
    add_preroll_silence,
    cif,
    compute_fbank,
    decode_ids,
    load_tokens,
    stack_frames,
)
from models.asr.paraformer.verify_paraformer_rknn_parity import (
    compare_array,
    run_decoder_ort,
    run_text_pipeline,
)


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


def make_encoder_inputs(feats: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    orig_frames = min(feats.shape[0], frames)
    if feats.shape[0] < frames:
        feats_in = np.pad(feats, ((0, frames - feats.shape[0]), (0, 0)), mode="edge")
    else:
        feats_in = feats[:frames]
    speech = np.ascontiguousarray(feats_in[np.newaxis, :].astype(np.float32))
    speech_len = np.array([orig_frames], dtype=np.int32)
    mask = np.zeros((1, frames), dtype=np.float32)
    mask[:, :orig_frames] = 1.0
    return speech, speech_len, mask, mask.copy(), orig_frames


class PrefixRKNN:
    def __init__(self, path: Path, core_mask: str):
        from rknnlite.api import RKNNLite

        self._api = RKNNLite(verbose=False)
        ret = self._api.load_rknn(str(path))
        if ret != 0:
            raise RuntimeError(f"load_rknn({path}) failed: ret={ret}")
        core = getattr(RKNNLite, core_mask, RKNNLite.NPU_CORE_AUTO)
        ret = self._api.init_runtime(core_mask=core)
        if ret != 0:
            raise RuntimeError(f"init_runtime({path}, {core_mask}) failed: ret={ret}")

    def run(self, speech: np.ndarray, encoder_pad_mask: np.ndarray) -> np.ndarray:
        outputs = self._api.inference(inputs=[speech, encoder_pad_mask])
        if outputs is None:
            raise RuntimeError("prefix RKNN inference failed")
        if len(outputs) != 1:
            raise RuntimeError(f"prefix RKNN expected 1 output, got {len(outputs)}")
        return np.asarray(outputs[0], dtype=np.float32)

    def release(self) -> None:
        try:
            self._api.release()
        except Exception:
            pass


class HybridEncoder:
    def __init__(self, prefix_rknn: Path, prefix_onnx: Path, suffix_onnx: Path, full_onnx: Path, core_mask: str):
        import onnxruntime as ort

        self.prefix = PrefixRKNN(prefix_rknn, core_mask)
        self.prefix_sess = ort.InferenceSession(str(prefix_onnx), providers=["CPUExecutionProvider"])
        self.suffix_sess = ort.InferenceSession(str(suffix_onnx), providers=["CPUExecutionProvider"])
        self.full_sess = ort.InferenceSession(str(full_onnx), providers=["CPUExecutionProvider"])
        self.cut_input_name = self.suffix_sess.get_inputs()[0].name

    def close(self) -> None:
        self.prefix.release()

    def run_prefix_ort(self, speech: np.ndarray, speech_len: np.ndarray, enc_mask: np.ndarray, cif_mask: np.ndarray) -> np.ndarray:
        feeds = {
            "speech": speech,
            "encoder_pad_mask": enc_mask,
        }
        names = {i.name for i in self.prefix_sess.get_inputs()}
        if "speech_lengths" in names:
            feeds["speech_lengths"] = speech_len
        if "cif_pad_mask" in names:
            feeds["cif_pad_mask"] = cif_mask
        return np.asarray(self.prefix_sess.run(None, feeds)[0], dtype=np.float32)

    def run_suffix(
        self,
        cut: np.ndarray,
        speech_len: np.ndarray,
        enc_mask: np.ndarray,
        cif_mask: np.ndarray,
        orig_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enc, enc_len, alphas = self.suffix_sess.run(
            ["enc", "enc_len", "alphas"],
            {
                self.cut_input_name: np.ascontiguousarray(cut.astype(np.float32)),
                "speech_lengths": speech_len,
                "encoder_pad_mask": enc_mask,
                "cif_pad_mask": cif_mask,
            },
        )
        return enc[:, :orig_frames, :], enc_len, alphas[:, :orig_frames]

    def run_full_ort(
        self,
        speech: np.ndarray,
        speech_len: np.ndarray,
        enc_mask: np.ndarray,
        cif_mask: np.ndarray,
        orig_frames: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        enc, enc_len, alphas = self.full_sess.run(
            ["enc", "enc_len", "alphas"],
            {
                "speech": speech,
                "speech_lengths": speech_len,
                "encoder_pad_mask": enc_mask,
                "cif_pad_mask": cif_mask,
            },
        )
        return enc[:, :orig_frames, :], enc_len, alphas[:, :orig_frames]

    def run_hybrid(self, feats: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        speech, speech_len, enc_mask, cif_mask, orig_frames = make_encoder_inputs(feats, frames)
        prefix_rknn = self.prefix.run(speech, enc_mask)
        enc, enc_len, alphas = self.run_suffix(prefix_rknn, speech_len, enc_mask, cif_mask, orig_frames)
        return enc, enc_len, alphas, prefix_rknn


def finite_stats(name: str, arr: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(arr)
    finite_values = arr[finite]
    return {
        "name": name,
        "shape": list(arr.shape),
        "finite": bool(finite.all()),
        "finite_ratio": float(finite.mean()) if arr.size else 1.0,
        "min": float(finite_values.min()) if finite_values.size else None,
        "max": float(finite_values.max()) if finite_values.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--model-dir", default=os.environ.get("PARAFORMER_MODEL_DIR", "/opt/asr/paraformer"))
    parser.add_argument("--prefix-rknn", required=True, type=Path)
    parser.add_argument("--prefix-onnx", required=True, type=Path)
    parser.add_argument("--suffix-onnx", required=True, type=Path)
    parser.add_argument("--full-encoder-onnx", required=True, type=Path)
    parser.add_argument("--decoder-onnx", default="")
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--core-mask", default=os.environ.get("PARAFORMER_RKNN_ENC_CORE", "NPU_CORE_1"))
    parser.add_argument("--json-out", default="")
    parser.add_argument("--prefix-rel-l2-max", type=float, default=0.12)
    parser.add_argument("--encoder-rel-l2-max", type=float, default=0.12)
    parser.add_argument("--alpha-rel-l2-max", type=float, default=0.12)
    parser.add_argument("--decoder-ids-must-match", action="store_true", default=True)
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip full chunked text pipeline; useful on slow CPUs.")
    args = parser.parse_args()

    import onnxruntime as ort

    model_dir = Path(args.model_dir)
    decoder_onnx = Path(args.decoder_onnx) if args.decoder_onnx else model_dir / "decoder-rknn.onnx"
    audio = read_audio(args.wav)
    feats = stack_frames(compute_fbank(audio))
    frames = min(max(40, args.frames), max(40, feats.shape[0], args.frames))
    speech, speech_len, enc_mask, cif_mask, orig_frames = make_encoder_inputs(feats, frames)

    tokens = load_tokens(str(model_dir / "tokens.txt"))
    decoder = ort.InferenceSession(str(decoder_onnx), providers=["CPUExecutionProvider"])
    encoder = HybridEncoder(args.prefix_rknn, args.prefix_onnx, args.suffix_onnx, args.full_encoder_onnx, args.core_mask)

    try:
        print("stage: prefix_ort", flush=True)
        prefix_ort = encoder.run_prefix_ort(speech, speech_len, enc_mask, cif_mask)
        print("stage: prefix_rknn", flush=True)
        prefix_rknn = encoder.prefix.run(speech, enc_mask)
        print("stage: suffix_hybrid", flush=True)
        enc_hybrid, enc_len_hybrid, alphas_hybrid = encoder.run_suffix(
            prefix_rknn, speech_len, enc_mask, cif_mask, orig_frames
        )
        print("stage: full_ort", flush=True)
        enc_ort, enc_len_ort, alphas_ort = encoder.run_full_ort(speech, speech_len, enc_mask, cif_mask, orig_frames)

        print("stage: cif_decoder", flush=True)
        ae_ort, carry_w_ort, carry_e_ort = cif(enc_ort[0], alphas_ort[0])
        ae_hybrid, carry_w_hybrid, carry_e_hybrid = cif(enc_hybrid[0], alphas_hybrid[0])
        probe_tokens = max(1, min(len(ae_ort), DEC_MAX_TOKENS))
        ae_probe = ae_ort[:probe_tokens] if len(ae_ort) else np.zeros((1, 512), dtype=np.float32)
        caches = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
        ids_ort, _ = run_decoder_ort(decoder, enc_ort, enc_ort.shape[1], ae_probe, caches, DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS)
        caches = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
        ids_hybrid, _ = run_decoder_ort(
            decoder,
            enc_hybrid,
            enc_hybrid.shape[1],
            ae_probe,
            caches,
            DEC_MAX_ENC_FRAMES,
            DEC_MAX_TOKENS,
        )

        if args.skip_pipeline:
            pipeline_ort = {"ids": ids_ort.tolist(), "text": decode_ids(ids_ort.tolist(), tokens)}
            pipeline_hybrid = {"ids": ids_hybrid.tolist(), "text": decode_ids(ids_hybrid.tolist(), tokens)}
        else:
            print("stage: pipeline", flush=True)

            def enc_ort_fn(chunk: np.ndarray):
                speech_i, len_i, mask_i, cif_i, n_i = make_encoder_inputs(chunk, args.frames)
                enc_i, _len_i, alpha_i = encoder.run_full_ort(speech_i, len_i, mask_i, cif_i, n_i)
                return enc_i, alpha_i

            def enc_hybrid_fn(chunk: np.ndarray):
                enc_i, _len_i, alpha_i, _prefix_i = encoder.run_hybrid(chunk, args.frames)
                return enc_i, alpha_i

            def dec_cpu_fn(enc: np.ndarray, enc_len: int, ae: np.ndarray, caches_in: list[np.ndarray]):
                return run_decoder_ort(decoder, enc, enc_len, ae, caches_in, DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS)

            pipeline_ort = run_text_pipeline(enc_ort_fn, dec_cpu_fn, feats, tokens, args.frames)
            pipeline_hybrid = run_text_pipeline(enc_hybrid_fn, dec_cpu_fn, feats, tokens, args.frames)

        result = {
            "wav": str(args.wav),
            "duration_s": len(audio) / SAMPLE_RATE,
            "lfr_frames": int(feats.shape[0]),
            "encoder_frames": int(frames),
            "prefix": {
                "rknn": finite_stats("prefix_rknn", prefix_rknn),
                "ort": finite_stats("prefix_ort", prefix_ort),
                "compare": compare_array("prefix_cut", prefix_rknn, prefix_ort),
            },
            "encoder": {
                "enc": compare_array("enc", enc_hybrid, enc_ort),
                "alphas": compare_array("alphas", alphas_hybrid, alphas_ort),
                "enc_len_hybrid": np.asarray(enc_len_hybrid).tolist(),
                "enc_len_ort": np.asarray(enc_len_ort).tolist(),
            },
            "cif": {
                "ort_tokens": int(len(ae_ort)),
                "hybrid_tokens": int(len(ae_hybrid)),
                "acoustic_embeds": compare_array(
                    "acoustic_embeds",
                    ae_hybrid[: min(len(ae_hybrid), len(ae_ort))],
                    ae_ort[: min(len(ae_hybrid), len(ae_ort))],
                ),
                "carry_weight_ort": float(carry_w_ort),
                "carry_weight_hybrid": float(carry_w_hybrid),
                "carry_embed": compare_array("carry_embed", carry_e_hybrid, carry_e_ort),
            },
            "decoder_cpu": {
                "probe_tokens": int(len(ae_probe)),
                "ids_ort": ids_ort.tolist(),
                "ids_hybrid": ids_hybrid.tolist(),
                "ids_match": bool(np.array_equal(ids_ort, ids_hybrid)),
            },
            "pipelines": {
                "ort_cpu_decoder": pipeline_ort,
                "hybrid_cpu_decoder": pipeline_hybrid,
                "text_match": pipeline_ort["text"] == pipeline_hybrid["text"],
            },
        }
    finally:
        encoder.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    failed = False
    prefix_rel = result["prefix"]["compare"]["rel_l2"]
    enc_rel = result["encoder"]["enc"]["rel_l2"]
    alpha_rel = result["encoder"]["alphas"]["rel_l2"]
    if prefix_rel > args.prefix_rel_l2_max:
        print(f"FAIL: prefix rel_l2 {prefix_rel:.4f} > {args.prefix_rel_l2_max}")
        failed = True
    if enc_rel > args.encoder_rel_l2_max:
        print(f"FAIL: encoder enc rel_l2 {enc_rel:.4f} > {args.encoder_rel_l2_max}")
        failed = True
    if alpha_rel > args.alpha_rel_l2_max:
        print(f"FAIL: encoder alphas rel_l2 {alpha_rel:.4f} > {args.alpha_rel_l2_max}")
        failed = True
    if args.decoder_ids_must_match and not result["decoder_cpu"]["ids_match"]:
        print("FAIL: CPU decoder sample_ids differ after hybrid encoder")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

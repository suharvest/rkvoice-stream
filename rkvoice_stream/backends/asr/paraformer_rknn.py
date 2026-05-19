"""Paraformer streaming ASR using Rockchip RKNN models.

This backend mirrors the Jetson ``paraformer_trt`` pipeline:

  audio -> numpy fbank/LFR -> RKNN encoder -> CIF -> RKNN decoder -> tokens

RKNN does not support TensorRT-style dynamic profiles, so the runtime expects
fixed-shape RKNN artifacts and pads inputs to those shapes.  The default layout
is::

    $PARAFORMER_MODEL_DIR/
      tokens.txt
      rknn/
        encoder.40.tf32.rknn
        encoder.80.tf32.rknn
        encoder.160.tf32.rknn
        encoder.400.tf32.rknn
        decoder.400x40.bf16.rknn

Jetson's best-known Paraformer profile is encoder FP32 + decoder BF16; attempts
at more aggressive precision were not reliable enough.  RKNN cannot express
true FP32 on NPU.  This backend fails closed during warmup if the selected RKNN
artifacts produce non-finite encoder output.  Set
``PARAFORMER_RKNN_ENC_PRECISION`` and ``PARAFORMER_RKNN_DEC_PRECISION`` to
``fp16|bf16|tf32|int8|auto``.  INT8 is only for explicit experiments after the
non-quantized path has passed parity.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using %d", name, os.environ.get(name), default)
        return default


PARAFORMER_MODEL_DIR = os.environ.get("PARAFORMER_MODEL_DIR", "/opt/asr/paraformer")
PARAFORMER_RKNN_DIR = os.environ.get(
    "PARAFORMER_RKNN_DIR",
    os.path.join(PARAFORMER_MODEL_DIR, "rknn"),
)
TOKENS_PATH = os.environ.get("PARAFORMER_TOKENS", os.path.join(PARAFORMER_MODEL_DIR, "tokens.txt"))
PRECISION_ENV = os.environ.get("PARAFORMER_RKNN_PRECISION", "fp16").lower()
ENC_PRECISION_ENV = os.environ.get("PARAFORMER_RKNN_ENC_PRECISION", PRECISION_ENV).lower()
DEC_PRECISION_ENV = os.environ.get("PARAFORMER_RKNN_DEC_PRECISION", PRECISION_ENV).lower()
ENCODER_MODE_ENV = os.environ.get("PARAFORMER_RKNN_ENCODER_MODE", "auto").lower()
DECODER_BACKEND_ENV = os.environ.get("PARAFORMER_RKNN_DECODER", "cpu").lower()
ENCODER_SUFFIX_ONNX = os.environ.get(
    "PARAFORMER_ENCODER_SUFFIX_ONNX",
    os.path.join(PARAFORMER_MODEL_DIR, "encoder_suffix_from_block30.onnx"),
)
DECODER_ONNX = os.environ.get("PARAFORMER_DECODER_ONNX", os.path.join(PARAFORMER_MODEL_DIR, "decoder-rknn.onnx"))
ENC_CORE_MASK = os.environ.get("PARAFORMER_RKNN_ENC_CORE", "NPU_CORE_1")
DEC_CORE_MASK = os.environ.get("PARAFORMER_RKNN_DEC_CORE", "NPU_CORE_1")
DEC_MAX_ENC_FRAMES = _env_int("PARAFORMER_RKNN_DEC_ENC_FRAMES", 400)
DEC_MAX_TOKENS = _env_int("PARAFORMER_RKNN_DEC_TOKENS", 40)
PREROLL_MS = max(0, _env_int("PARAFORMER_PREROLL_MS", 100))

SAMPLE_RATE = 16000
FFT_SIZE = 512
WINDOW_SIZE = 400
HOP_SIZE = 160
NUM_MEL_BINS = 80
NUM_STACKED = 7
NUM_STRIDE = 6
PRE_EMPH = 0.97
LOW_FREQ = 20
HIGH_FREQ = 8000

CHUNK_SIZE_SEC = 0.67
LEFT_CONTEXT_SEC = 2.68
RIGHT_LOOKAHEAD_LFR = 15
CIF_THRESHOLD = 1.0
CIF_TAIL_THRESHOLD = 0.5

BLANK_ID = 0
SOS_ID = 1
EOS_ID = 2
VOCAB_SIZE = 8404
CACHE_COUNT = 16
CACHE_SHAPE = (1, 512, 10)

_MEL_FILTERBANK: Optional[np.ndarray] = None


def _get_mel_filterbank() -> np.ndarray:
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is not None:
        return _MEL_FILTERBANK

    low_mel = 2595.0 * np.log10(1.0 + LOW_FREQ / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + HIGH_FREQ / 700.0)
    mel_points = np.linspace(low_mel, high_mel, NUM_MEL_BINS + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bin_indices = np.floor(hz_points * (FFT_SIZE // 2 + 1) / (SAMPLE_RATE / 2.0)).astype(np.int32)

    fbank = np.zeros((NUM_MEL_BINS, FFT_SIZE // 2 + 1), dtype=np.float32)
    for i in range(NUM_MEL_BINS):
        left, center, right = bin_indices[i], bin_indices[i + 1], bin_indices[i + 2]
        for j in range(left, center):
            fbank[i, j] = (j - left) / (center - left) if center != left else 1.0
        for j in range(center, right):
            fbank[i, j] = (right - j) / (right - center) if right != center else 1.0

    _MEL_FILTERBANK = fbank
    return fbank


def compute_fbank(audio: np.ndarray) -> np.ndarray:
    if len(audio) < WINDOW_SIZE:
        audio = np.pad(audio, (0, WINDOW_SIZE - len(audio)))

    audio = np.concatenate([[audio[0]], audio[1:] - PRE_EMPH * audio[:-1]])
    num_frames = (len(audio) - WINDOW_SIZE) // HOP_SIZE + 1
    frames = np.zeros((num_frames, WINDOW_SIZE), dtype=np.float32)
    for i in range(num_frames):
        start = i * HOP_SIZE
        frames[i] = audio[start:start + WINDOW_SIZE]

    frames *= np.hamming(WINDOW_SIZE).astype(np.float32)
    spectrum = np.fft.rfft(frames, n=FFT_SIZE)
    power = (spectrum.real ** 2 + spectrum.imag ** 2) / FFT_SIZE
    mel_feats = power @ _get_mel_filterbank().T
    mel_feats = np.log(np.maximum(mel_feats, 1e-10))
    mean = mel_feats.mean(axis=0, keepdims=True)
    std = np.maximum(mel_feats.std(axis=0, keepdims=True), 1e-10)
    return ((mel_feats - mean) / std).astype(np.float32)


def stack_frames(feats: np.ndarray) -> np.ndarray:
    n, d = feats.shape
    out_n = (n + NUM_STRIDE - 1) // NUM_STRIDE
    needed = (out_n - 1) * NUM_STRIDE + NUM_STACKED
    if needed > n:
        feats = np.concatenate([feats, np.repeat(feats[-1:], needed - n, axis=0)], axis=0)
    stacked = np.zeros((out_n, d * NUM_STACKED), dtype=np.float32)
    for i in range(out_n):
        stacked[i] = feats[i * NUM_STRIDE:i * NUM_STRIDE + NUM_STACKED].ravel()
    return stacked


def cif(
    enc: np.ndarray,
    alphas: np.ndarray,
    threshold: float = CIF_THRESHOLD,
    tail_threshold: float = CIF_TAIL_THRESHOLD,
    carry_weight: float = 0.0,
    carry_embed: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if carry_embed is None:
        carry_embed = np.zeros(512, dtype=np.float32)

    acoustic_embeds = []
    accum_weight = carry_weight
    accum_embed = carry_embed.copy()

    for t in range(len(enc)):
        alpha = float(alphas[t])
        if alpha <= 0:
            continue
        accum_weight += alpha
        accum_embed += alpha * enc[t]
        while accum_weight >= threshold:
            excess = accum_weight - threshold
            token_embed = (accum_embed - excess * enc[t]) / threshold
            acoustic_embeds.append(token_embed)
            accum_weight = excess
            accum_embed = excess * enc[t]

    embeds = np.stack(acoustic_embeds) if acoustic_embeds else np.empty((0, 512), dtype=np.float32)
    return embeds.astype(np.float32), accum_weight, accum_embed.astype(np.float32)


def load_tokens(path: str) -> list[str]:
    tokens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.rstrip("\n")
            parts = token.rsplit(None, 1)
            if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                token = parts[0]
            else:
                token = token.strip()
            tokens.append(token)
    return tokens


def decode_ids(token_ids: list[int], tokens: list[str]) -> str:
    pieces = []
    for tid in token_ids:
        if tid in (BLANK_ID, SOS_ID, EOS_ID):
            continue
        if 0 <= tid < len(tokens):
            token = tokens[tid]
            if token.startswith("<") and token.endswith(">"):
                continue
            if token.endswith("@@"):
                token = token[:-2]
            pieces.append(token)
    return "".join(pieces)


def add_preroll_silence(audio: np.ndarray) -> np.ndarray:
    if PREROLL_MS <= 0 or len(audio) == 0:
        return audio.astype(np.float32, copy=False)
    pad = np.zeros(int(SAMPLE_RATE * PREROLL_MS / 1000), dtype=np.float32)
    return np.concatenate([pad, audio.astype(np.float32, copy=False)])


def initial_preroll_audio() -> np.ndarray:
    if PREROLL_MS <= 0:
        return np.array([], dtype=np.float32)
    return np.zeros(int(SAMPLE_RATE * PREROLL_MS / 1000), dtype=np.float32)


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = _resample(audio, sr, SAMPLE_RATE)
    return audio.astype(np.float32)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class _RknnRuntime:
    def __init__(self, path: str, core_mask: str):
        from rknnlite.api import RKNNLite

        self.path = path
        self.model = RKNNLite(verbose=False)
        ret = self.model.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"load_rknn({path}) failed: ret={ret}")
        core = getattr(RKNNLite, core_mask, RKNNLite.NPU_CORE_AUTO)
        ret = self.model.init_runtime(core_mask=core)
        if ret != 0:
            raise RuntimeError(f"init_runtime({path}, {core_mask}) failed: ret={ret}")

    def inference(self, inputs: list[np.ndarray]) -> list[np.ndarray]:
        return self.model.inference(inputs=inputs)

    def release(self) -> None:
        try:
            self.model.release()
        except Exception:
            pass


def _precision_candidates(precision: str) -> list[str]:
    if precision == "auto":
        return ["tf32", "bf16", "fp16", "int8"]
    if precision in ("fp16", "bf16", "tf32", "int8"):
        return [precision]
    logger.warning("Invalid Paraformer RKNN precision %r; using fp16", precision)
    return ["fp16"]


def _find_model_file(directory: Path, patterns: list[str], precision: str) -> Optional[Path]:
    for prec in _precision_candidates(precision):
        for pat in patterns:
            candidates = sorted(directory.glob(pat.format(precision=prec)))
            if candidates:
                return candidates[0]
    return None


class ParaformerRKNNBackend(ASRBackend):
    def __init__(self):
        self._tokens: list[str] = []
        self._encoders: dict[int, _RknnRuntime] = {}
        self._decoder: Optional[_RknnRuntime] = None
        self._encoder_suffix = None
        self._encoder_suffix_cut_input = ""
        self._decoder_ort = None
        self._decoder_shape = (DEC_MAX_ENC_FRAMES, DEC_MAX_TOKENS)
        self._ready = False
        self._enc_precision = ENC_PRECISION_ENV
        self._dec_precision = DEC_PRECISION_ENV
        self._encoder_mode = ENCODER_MODE_ENV
        self._decoder_backend = DECODER_BACKEND_ENV

    @property
    def name(self) -> str:
        return "paraformer_rknn"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        rknn_dir = Path(PARAFORMER_RKNN_DIR)
        if not rknn_dir.is_dir():
            raise FileNotFoundError(f"Paraformer RKNN dir not found: {rknn_dir}")
        if not os.path.isfile(TOKENS_PATH):
            raise FileNotFoundError(f"Paraformer tokens not found: {TOKENS_PATH}")

        self._tokens = load_tokens(TOKENS_PATH)
        logger.info("Loaded %d Paraformer tokens from %s", len(self._tokens), TOKENS_PATH)

        if self._encoder_mode not in ("auto", "full", "hybrid"):
            logger.warning("Invalid PARAFORMER_RKNN_ENCODER_MODE=%r; using auto", self._encoder_mode)
            self._encoder_mode = "auto"
        if self._decoder_backend not in ("cpu", "rknn"):
            logger.warning("Invalid PARAFORMER_RKNN_DECODER=%r; using cpu", self._decoder_backend)
            self._decoder_backend = "cpu"

        if self._encoder_mode in ("auto", "hybrid"):
            hybrid_files = self._find_hybrid_encoder_files(rknn_dir)
            if hybrid_files:
                self._load_hybrid_encoder(hybrid_files)
            elif self._encoder_mode == "hybrid":
                raise FileNotFoundError(
                    f"No encoder_prefix_to_block*.rknn files found in {rknn_dir} for hybrid encoder"
                )

        if not self._encoders:
            self._load_full_encoder(rknn_dir)

        if self._decoder_backend == "cpu":
            self._load_cpu_decoder()
        else:
            self._load_rknn_decoder(rknn_dir)

        self._validate_warmup()
        self._ready = True

    def _find_hybrid_encoder_files(self, rknn_dir: Path) -> dict[int, Path]:
        encoder_files: dict[int, Path] = {}
        for path in sorted(rknn_dir.glob("encoder_prefix_to_block*.rknn")):
            match = re.search(r"encoder_prefix_to_block\d+\.(\d+)\.(fp16|bf16|tf32|int8)\.rknn$", path.name)
            if not match:
                continue
            frames = int(match.group(1))
            prec = match.group(2)
            if self._enc_precision == "auto" or prec == self._enc_precision:
                encoder_files.setdefault(frames, path)
        return encoder_files

    def _load_hybrid_encoder(self, encoder_files: dict[int, Path]) -> None:
        suffix_path = Path(ENCODER_SUFFIX_ONNX)
        if not suffix_path.exists():
            raise FileNotFoundError(f"Paraformer encoder suffix ONNX not found: {suffix_path}")
        import onnxruntime as ort

        for frames, path in sorted(encoder_files.items()):
            self._encoders[frames] = _RknnRuntime(str(path), ENC_CORE_MASK)
            logger.info("Loaded Paraformer hybrid encoder prefix bucket %d frames: %s", frames, path.name)
        self._encoder_suffix = ort.InferenceSession(str(suffix_path), providers=["CPUExecutionProvider"])
        self._encoder_suffix_cut_input = self._encoder_suffix.get_inputs()[0].name
        self._encoder_mode = "hybrid"
        logger.info("Loaded Paraformer encoder CPU suffix: %s", suffix_path)

    def _load_full_encoder(self, rknn_dir: Path) -> None:
        encoder_files: dict[int, Path] = {}
        for path in sorted(rknn_dir.glob("encoder.*.rknn")):
            match = re.search(r"encoder\.(\d+)\.(fp16|bf16|tf32|int8)\.rknn$", path.name)
            if not match:
                continue
            frames = int(match.group(1))
            prec = match.group(2)
            if self._enc_precision == "auto" or prec == self._enc_precision:
                encoder_files.setdefault(frames, path)

        if not encoder_files and self._enc_precision != "auto":
            logger.warning("No %s encoder RKNN files found; trying auto precision", self._enc_precision)
            self._enc_precision = "auto"
            return self._load_full_encoder(rknn_dir)
        if not encoder_files:
            raise FileNotFoundError(f"No encoder.*.(fp16|bf16|tf32|int8).rknn files found in {rknn_dir}")

        for frames, path in sorted(encoder_files.items()):
            self._encoders[frames] = _RknnRuntime(str(path), ENC_CORE_MASK)
            logger.info("Loaded Paraformer full encoder bucket %d frames: %s", frames, path.name)
        self._encoder_mode = "full"

    def _load_cpu_decoder(self) -> None:
        decoder_path = Path(DECODER_ONNX)
        if not decoder_path.exists():
            raise FileNotFoundError(f"Paraformer CPU decoder ONNX not found: {decoder_path}")
        import onnxruntime as ort

        self._decoder_ort = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
        logger.info("Loaded Paraformer CPU decoder ONNX: %s", decoder_path)

    def _load_rknn_decoder(self, rknn_dir: Path) -> None:
        dec_patterns = [
            f"decoder.{DEC_MAX_ENC_FRAMES}x{DEC_MAX_TOKENS}.{{precision}}.rknn",
            "decoder.*.{precision}.rknn",
        ]
        decoder_path = _find_model_file(rknn_dir, dec_patterns, self._dec_precision)
        if decoder_path is None and self._dec_precision != "auto":
            decoder_path = _find_model_file(rknn_dir, dec_patterns, "auto")
        if decoder_path is None:
            raise FileNotFoundError(f"No decoder RKNN found in {rknn_dir}")

        shape_match = re.search(r"decoder\.(\d+)x(\d+)\.(fp16|bf16|tf32|int8)\.rknn$", decoder_path.name)
        if shape_match:
            self._decoder_shape = (int(shape_match.group(1)), int(shape_match.group(2)))
        self._decoder = _RknnRuntime(str(decoder_path), DEC_CORE_MASK)
        logger.info(
            "Loaded Paraformer RKNN decoder %dx%d: %s",
            self._decoder_shape[0],
            self._decoder_shape[1],
            decoder_path.name,
        )

    def cleanup(self) -> None:
        for model in self._encoders.values():
            model.release()
        self._encoders.clear()
        if self._decoder is not None:
            self._decoder.release()
            self._decoder = None
        self._encoder_suffix = None
        self._decoder_ort = None
        self._ready = False

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        audio = add_preroll_silence(_decode_audio(audio_bytes))
        text = self._transcribe_audio(audio)
        return TranscriptionResult(text=text, language=language)

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("Paraformer RKNN backend not ready")
        return ParaformerRKNNStream(self)

    def _validate_warmup(self) -> None:
        warmup_audio = (
            np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE) * 0.3
        ).astype(np.float32)
        feats = stack_frames(compute_fbank(warmup_audio))
        enc, alphas = self._run_encoder(feats[: min(feats.shape[0], max(self._encoders))])
        if enc is None or alphas is None or not np.isfinite(enc).all() or not np.isfinite(alphas).all():
            raise RuntimeError("Paraformer RKNN encoder warmup produced invalid output")
        cache = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
        out = self._run_decoder(
            np.zeros((1, 1, 512), dtype=np.float32),
            1,
            np.zeros((1, 512), dtype=np.float32),
            1,
            cache,
        )
        if out is None:
            raise RuntimeError("Paraformer RKNN decoder warmup failed")

    def _select_encoder_bucket(self, n_frames: int) -> int:
        for frames in sorted(self._encoders):
            if n_frames <= frames:
                return frames
        return max(self._encoders)

    def _run_encoder(self, feats: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._encoders:
            raise RuntimeError("No Paraformer RKNN encoders loaded")

        orig_frames = feats.shape[0]
        bucket = self._select_encoder_bucket(orig_frames)
        if orig_frames < bucket:
            feats = np.pad(feats, ((0, bucket - orig_frames), (0, 0)), mode="edge")
        else:
            feats = feats[:bucket]
            orig_frames = min(orig_frames, bucket)

        speech = np.ascontiguousarray(feats[np.newaxis, :].astype(np.float32))
        speech_len = np.array([orig_frames], dtype=np.int32)
        encoder_pad_mask = np.zeros((1, bucket), dtype=np.float32)
        encoder_pad_mask[:, :orig_frames] = 1.0
        cif_pad_mask = encoder_pad_mask.copy()
        t0 = time.perf_counter()
        if self._encoder_mode == "hybrid":
            outputs = self._encoders[bucket].inference([speech, encoder_pad_mask])
        else:
            outputs = self._encoders[bucket].inference([
                speech,
                speech_len,
                encoder_pad_mask,
                cif_pad_mask,
            ])
        enc_ms = (time.perf_counter() - t0) * 1000

        if self._encoder_mode == "hybrid":
            if self._encoder_suffix is None:
                raise RuntimeError("Paraformer hybrid encoder suffix not loaded")
            if len(outputs) != 1:
                logger.error("Paraformer hybrid encoder prefix expected 1 output, got %d", len(outputs))
                return None, None
            t1 = time.perf_counter()
            enc, enc_len, alphas = self._encoder_suffix.run(
                ["enc", "enc_len", "alphas"],
                {
                    self._encoder_suffix_cut_input: outputs[0].astype(np.float32, copy=False),
                    "speech_lengths": speech_len,
                    "encoder_pad_mask": encoder_pad_mask,
                    "cif_pad_mask": cif_pad_mask,
                },
            )
            suffix_ms = (time.perf_counter() - t1) * 1000
            enc = enc.astype(np.float32, copy=False)
            alphas = alphas.astype(np.float32, copy=False)
            enc = enc[:, :orig_frames, :]
            alphas = alphas[:, :orig_frames]
            logger.debug(
                "Paraformer hybrid encoder bucket=%d frames=%d prefix=%.1fms suffix=%.1fms enc_len=%s",
                bucket,
                orig_frames,
                enc_ms,
                suffix_ms,
                enc_len,
            )
            return enc, alphas

        if len(outputs) < 3:
            logger.error("Paraformer encoder expected 3 outputs, got %d", len(outputs))
            return None, None
        enc = outputs[0].astype(np.float32, copy=False)
        alphas = outputs[2].astype(np.float32, copy=False)
        if enc.ndim == 2:
            enc = enc[np.newaxis, :]
        if alphas.ndim == 1:
            alphas = alphas[np.newaxis, :]
        enc = enc[:, :orig_frames, :]
        alphas = alphas[:, :orig_frames]
        logger.debug("Paraformer RKNN encoder bucket=%d frames=%d %.1fms", bucket, orig_frames, enc_ms)
        return enc, alphas

    def _run_decoder(
        self,
        enc: np.ndarray,
        enc_len: int,
        acoustic_embeds: np.ndarray,
        acoustic_embeds_len: int,
        cache: list[np.ndarray],
    ) -> Optional[np.ndarray]:
        if self._decoder_backend == "cpu":
            if self._decoder_ort is None:
                raise RuntimeError("Paraformer CPU decoder not loaded")
        elif self._decoder is None:
            raise RuntimeError("Paraformer RKNN decoder not loaded")

        n_tokens = int(acoustic_embeds.shape[0])
        if n_tokens <= 0:
            return np.array([], dtype=np.int64)

        max_enc, max_tokens = self._decoder_shape
        if enc.shape[1] > max_enc:
            enc = enc[:, -max_enc:, :]
            enc_len = max_enc
        if n_tokens > max_tokens:
            acoustic_embeds = acoustic_embeds[:max_tokens]
            n_tokens = max_tokens
            acoustic_embeds_len = max_tokens

        enc_in = np.zeros((1, max_enc, 512), dtype=np.float32)
        enc_in[:, :enc.shape[1], :] = enc[:, :max_enc, :]
        ae_in = np.zeros((1, max_tokens, 512), dtype=np.float32)
        ae_in[:, :n_tokens, :] = acoustic_embeds[np.newaxis, :n_tokens, :]

        pad_mask = np.zeros((1, max_tokens), dtype=np.float32)
        pad_mask[:, :n_tokens] = 1.0
        enc_pad_mask = np.zeros((1, max_enc), dtype=np.float32)
        enc_pad_mask[:, :enc_len] = 1.0

        inputs: list[np.ndarray] = [
            enc_in,
            ae_in,
        ]
        inputs.extend(np.ascontiguousarray(c.astype(np.float32)) for c in cache)
        inputs.extend([pad_mask, enc_pad_mask])

        t0 = time.perf_counter()
        if self._decoder_backend == "cpu":
            feeds = {
                "enc": enc_in,
                "acoustic_embeds": ae_in,
                "pad_mask": pad_mask,
                "enc_pad_mask": enc_pad_mask,
            }
            for i, cache_i in enumerate(cache):
                feeds[f"in_cache_{i}"] = np.ascontiguousarray(cache_i.astype(np.float32))
            output_names = ["sample_ids"] + [f"out_cache_{i}" for i in range(CACHE_COUNT)]
            outputs = self._decoder_ort.run(output_names, feeds)
            sample_ids = outputs[0]
            cache_offset = 1
        else:
            outputs = self._decoder.inference(inputs)
            if len(outputs) < 2:
                logger.error("Paraformer decoder expected at least 2 outputs, got %d", len(outputs))
                return None
            # RKNN decoder outputs are logits, sample_ids, out_cache_0..15.
            sample_ids = outputs[1]
            cache_offset = 2
        dec_ms = (time.perf_counter() - t0) * 1000

        if sample_ids.ndim == 2:
            sample_ids = sample_ids[0]
        sample_ids = sample_ids.astype(np.int64, copy=False)[:n_tokens]

        for i in range(min(CACHE_COUNT, max(0, len(outputs) - cache_offset))):
            cache[i][...] = outputs[i + cache_offset].astype(np.float32, copy=False)

        logger.debug("Paraformer %s decoder enc=%d tokens=%d %.1fms", self._decoder_backend, enc_len, n_tokens, dec_ms)
        return sample_ids

    def _transcribe_audio(self, audio: np.ndarray) -> str:
        if not self.is_ready():
            raise RuntimeError("Paraformer RKNN backend not ready")

        feats = stack_frames(compute_fbank(audio))
        all_token_ids: list[int] = []
        carry_w = 0.0
        carry_e = np.zeros(512, dtype=np.float32)
        cache = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
        max_bucket = max(self._encoders)

        for start in range(0, feats.shape[0], max_bucket):
            chunk = feats[start:start + max_bucket]
            enc, alphas = self._run_encoder(chunk)
            if enc is None or alphas is None:
                continue
            acoustic_embeds, carry_w, carry_e = cif(
                enc[0],
                alphas[0],
                carry_weight=carry_w,
                carry_embed=carry_e,
            )
            if len(acoustic_embeds) == 0:
                continue
            sample_ids = self._run_decoder(enc, enc.shape[1], acoustic_embeds, len(acoustic_embeds), cache)
            if sample_ids is not None:
                all_token_ids.extend(sample_ids.tolist())

        if carry_w >= CIF_TAIL_THRESHOLD:
            acoustic_embeds = (carry_e / carry_w)[np.newaxis, :]
            sample_ids = self._run_decoder(
                np.zeros((1, 1, 512), dtype=np.float32),
                1,
                acoustic_embeds,
                1,
                cache,
            )
            if sample_ids is not None:
                all_token_ids.extend(sample_ids.tolist())

        return decode_ids(all_token_ids, self._tokens)


class ParaformerRKNNStream(ASRStream):
    def __init__(self, backend: ParaformerRKNNBackend):
        self._backend = backend
        self._audio_buf = initial_preroll_audio()
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        self._cif_processed_lfr = 0
        self._all_token_ids: list[int] = []
        self._partial_text = ""
        self._is_endpoint = False
        self._carry_weight = 0.0
        self._carry_embed = np.zeros(512, dtype=np.float32)
        self._cache = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]
        self._cancelled = False
        self._final_text_cache = ""

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        if self._cancelled:
            return
        audio = samples.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            audio = _resample(audio, sample_rate, SAMPLE_RATE)
        self._audio_buf = np.concatenate([self._audio_buf, audio])
        self._process_chunks()

    def get_partial(self) -> tuple[str, bool]:
        return self._partial_text, self._is_endpoint

    def cancel_and_finalize(self) -> None:
        self._final_text_cache = self._partial_text
        self._cancelled = True
        self._audio_buf = np.array([], dtype=np.float32)

    def finalize(self) -> str:
        if self._cancelled:
            return self._final_text_cache
        if len(self._audio_buf) > 0:
            self._all_audio = np.concatenate([self._all_audio, self._audio_buf])
            self._audio_buf = np.array([], dtype=np.float32)
        self._drain_all_pending()
        self._flush_cif_tail()
        text = self._partial_text
        self._reset()
        return text

    def _reset(self) -> None:
        self._audio_buf = initial_preroll_audio()
        self._all_audio = np.array([], dtype=np.float32)
        self._prev_total_frames = 0
        self._cif_processed_lfr = 0
        self._all_token_ids = []
        self._partial_text = ""
        self._is_endpoint = False
        self._carry_weight = 0.0
        self._carry_embed = np.zeros(512, dtype=np.float32)
        self._cache = [np.zeros(CACHE_SHAPE, dtype=np.float32) for _ in range(CACHE_COUNT)]

    def _process_chunks(self) -> None:
        chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)
        while len(self._audio_buf) >= chunk_samples:
            chunk_audio = self._audio_buf[:chunk_samples]
            self._audio_buf = self._audio_buf[chunk_samples:]
            self._all_audio = np.concatenate([self._all_audio, chunk_audio])
            self._process_current_audio(final=False)

    def _drain_all_pending(self) -> None:
        if len(self._all_audio) >= WINDOW_SIZE:
            self._process_current_audio(final=True)

    def _process_current_audio(self, final: bool) -> None:
        feats = stack_frames(compute_fbank(self._all_audio))
        cur_total_lfr = feats.shape[0]
        if cur_total_lfr <= self._prev_total_frames and not final:
            return
        self._prev_total_frames = cur_total_lfr

        max_bucket = max(self._backend._encoders)
        enc_input = feats if cur_total_lfr <= max_bucket else feats[-max_bucket:]
        enc, alphas = self._backend._run_encoder(enc_input)
        if enc is None or alphas is None:
            return

        window_start_abs = cur_total_lfr - enc.shape[1]
        if final:
            cif_end_abs = cur_total_lfr
        else:
            cif_end_abs = max(window_start_abs, cur_total_lfr - RIGHT_LOOKAHEAD_LFR)

        cif_start_abs = max(self._cif_processed_lfr, window_start_abs)
        if cif_end_abs <= cif_start_abs:
            return

        start = cif_start_abs - window_start_abs
        end = cif_end_abs - window_start_abs
        self._cif_processed_lfr = cif_end_abs

        acoustic_embeds, self._carry_weight, self._carry_embed = cif(
            enc[0][start:end],
            alphas[0][start:end],
            carry_weight=self._carry_weight,
            carry_embed=self._carry_embed,
        )
        if len(acoustic_embeds) == 0:
            return

        sample_ids = self._backend._run_decoder(
            enc,
            enc.shape[1],
            acoustic_embeds,
            len(acoustic_embeds),
            self._cache,
        )
        if sample_ids is None:
            return
        self._all_token_ids.extend(sample_ids.tolist())
        self._partial_text = decode_ids(self._all_token_ids, self._backend._tokens)

    def _flush_cif_tail(self) -> None:
        if self._carry_weight < CIF_TAIL_THRESHOLD:
            return
        acoustic_embeds = (self._carry_embed / self._carry_weight)[np.newaxis, :]
        sample_ids = self._backend._run_decoder(
            np.zeros((1, 1, 512), dtype=np.float32),
            1,
            acoustic_embeds,
            1,
            self._cache,
        )
        if sample_ids is not None:
            self._all_token_ids.extend(sample_ids.tolist())
            self._partial_text = decode_ids(self._all_token_ids, self._backend._tokens)

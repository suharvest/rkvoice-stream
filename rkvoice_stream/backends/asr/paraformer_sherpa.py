"""Paraformer ASR backend using sherpa-onnx (CPU only).

No NPU usage — runs entirely on CPU via ONNX runtime.
Select via: ASR_BACKEND=paraformer_sherpa

Supports two model layouts (auto-detected):

  Offline/non-streaming (single model file):
    decoder.onnx  (or decoder.int8.onnx)   — the full model
    tokens.txt
    Uses sherpa_onnx.OfflineRecognizer.from_paraformer()

  Online/streaming (separate encoder + decoder):
    encoder.onnx  (or encoder.int8.onnx)
    decoder.onnx  (or decoder.int8.onnx)
    tokens.txt
    Uses sherpa_onnx.OnlineRecognizer.from_paraformer()

PARAFORMER_MODEL_DIR defaults to /opt/asr/paraformer.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)


def _opt_path(model_dir: str, name: str, int8_name: str) -> str:
    """Return int8 variant path if it exists, otherwise the standard path."""
    int8 = Path(model_dir) / int8_name
    if int8.exists():
        return str(int8)
    return str(Path(model_dir) / name)


def _offline_model_path(model_dir: str) -> str:
    """Find the offline Paraformer model file.

    Checks for model.int8.onnx, model.onnx, decoder.int8.onnx, decoder.onnx
    in that order of preference.
    """
    for name in ("model.int8.onnx", "model.onnx", "decoder.int8.onnx", "decoder.onnx"):
        p = Path(model_dir) / name
        if p.exists():
            return str(p)
    # Fall back to decoder.onnx (will fail at load time with a clear error)
    return str(Path(model_dir) / "model.onnx")


def _has_encoder(model_dir: str) -> bool:
    """Return True if an encoder model file exists in model_dir."""
    return (
        (Path(model_dir) / "encoder.onnx").exists()
        or (Path(model_dir) / "encoder.int8.onnx").exists()
    )


class ParaformerSherpaBackend(ASRBackend):
    """ASR backend using Paraformer via sherpa-onnx (CPU).

    Auto-detects whether to use the offline or online (streaming) API based on
    the presence of encoder.onnx in the model directory.
    """

    def __init__(self):
        self._recognizer = None
        self._online_mode = False  # True = OnlineRecognizer, False = OfflineRecognizer
        self._ready = False

    @property
    def name(self) -> str:
        return "paraformer_sherpa"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._recognizer is not None

    def preload(self) -> None:
        import sherpa_onnx  # lazy import — optional dependency

        model_dir = os.environ.get("PARAFORMER_MODEL_DIR", "/opt/asr/paraformer")
        num_threads = int(os.environ.get("PARAFORMER_NUM_THREADS", "2"))
        tokens = str(Path(model_dir) / "tokens.txt")

        if _has_encoder(model_dir):
            # Streaming (online) mode: separate encoder + decoder files
            encoder = _opt_path(model_dir, "encoder.onnx", "encoder.int8.onnx")
            decoder = _opt_path(model_dir, "decoder.onnx", "decoder.int8.onnx")
            logger.info(
                "Loading Paraformer (online) from %s (encoder=%s, decoder=%s)",
                model_dir,
                Path(encoder).name,
                Path(decoder).name,
            )
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_paraformer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                num_threads=num_threads,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                decoding_method="greedy_search",
                provider="cpu",
            )
            self._online_mode = True
        else:
            # Offline mode: single model file (model.onnx or decoder.onnx)
            model = _offline_model_path(model_dir)
            logger.info(
                "Loading Paraformer (offline) from %s (model=%s)",
                model_dir,
                Path(model).name,
            )
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=model,
                tokens=tokens,
                num_threads=num_threads,
                provider="cpu",
            )
            self._online_mode = False

        self._ready = True
        logger.info(
            "Paraformer sherpa-onnx backend ready (mode=%s).",
            "online" if self._online_mode else "offline",
        )

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        audio = _decode_audio(audio_bytes)

        if self._online_mode:
            # OnlineRecognizer: feed audio + silence tail, then drain
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            tail_paddings = np.zeros(int(0.5 * 16000), dtype=np.float32)
            stream.accept_waveform(16000, tail_paddings)
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            result = self._recognizer.get_result(stream)
            text = result.text if hasattr(result, "text") else result
        else:
            # OfflineRecognizer: decode_stream() does everything in one call
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self._recognizer.decode_stream(stream)
            text = stream.result.text

        return TranscriptionResult(text=text.strip())

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        if self._online_mode:
            sherpa_stream = self._recognizer.create_stream()
            return ParaformerSherpaStream(self._recognizer, sherpa_stream)
        else:
            # Offline mode: wrap OfflineRecognizer as a pseudo-stream
            return ParaformerOfflineStream(self._recognizer)


class ParaformerSherpaStream(ASRStream):
    """Wraps a sherpa_onnx online stream as an ASRStream."""

    def __init__(self, recognizer, sherpa_stream):
        self._recognizer = recognizer
        self._stream = sherpa_stream

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed float32 audio samples into the stream, then decode while ready."""
        audio = samples.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        self._stream.accept_waveform(16000, audio)

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

    def get_partial(self) -> tuple[str, bool]:
        """Return (partial_text, is_endpoint)."""
        result = self._recognizer.get_result(self._stream)
        text = result.text if hasattr(result, "text") else result
        is_endpoint = self._recognizer.is_endpoint(self._stream)
        return text.strip(), is_endpoint

    def finalize(self) -> str:
        """Feed tail silence, flush decoder, return final text."""
        # Pad with silence so endpoint detection triggers
        tail_paddings = np.zeros(int(0.5 * 16000), dtype=np.float32)
        self._stream.accept_waveform(16000, tail_paddings)

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        result = self._recognizer.get_result(self._stream)
        text = result.text if hasattr(result, "text") else result
        return text.strip()


class ParaformerOfflineStream(ASRStream):
    """Pseudo-streaming wrapper around sherpa_onnx OfflineRecognizer for Paraformer.

    Accumulates audio chunks and runs the offline recognizer on finalize().
    """

    def __init__(self, recognizer):
        self._recognizer = recognizer
        self._chunks: list[np.ndarray] = []

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        audio = samples.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)
        self._chunks.append(audio)

    def get_partial(self) -> tuple[str, bool]:
        return "", False

    def finalize(self) -> str:
        if not self._chunks:
            return ""
        audio = np.concatenate(self._chunks)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode WAV/FLAC/etc. bytes to 16 kHz float32 mono numpy array."""
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)
    try:
        audio, sr = sf.read(buf, dtype="float32")
    except Exception as exc:
        raise ValueError(f"Cannot decode audio: {exc}") from exc

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:
        logger.warning("Input sample rate %d != 16000, resampling.", sr)
        audio = _resample(audio, sr, 16000)

    return audio.astype(np.float32)


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D float32 audio array using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)

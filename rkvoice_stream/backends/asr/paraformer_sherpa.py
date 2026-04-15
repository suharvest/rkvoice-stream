"""Paraformer streaming ASR backend using sherpa-onnx (CPU only).

No NPU usage — runs entirely on CPU via ONNX runtime.
Select via: ASR_BACKEND=paraformer_sherpa

Model directory layout (PARAFORMER_MODEL_DIR, default /opt/asr/paraformer):
    encoder.onnx  (or encoder.int8.onnx)
    decoder.onnx  (or decoder.int8.onnx)
    tokens.txt
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


def _model_path(model_dir: str, name: str, int8_name: str) -> str:
    """Return int8 variant path if it exists, otherwise the standard path."""
    int8 = Path(model_dir) / int8_name
    if int8.exists():
        return str(int8)
    return str(Path(model_dir) / name)


class ParaformerSherpaBackend(ASRBackend):
    """ASR backend using Paraformer streaming model via sherpa-onnx (CPU)."""

    def __init__(self):
        self._recognizer = None
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

        encoder = _model_path(model_dir, "encoder.onnx", "encoder.int8.onnx")
        decoder = _model_path(model_dir, "decoder.onnx", "decoder.int8.onnx")
        tokens = str(Path(model_dir) / "tokens.txt")

        logger.info(
            "Loading Paraformer from %s (encoder=%s, decoder=%s)",
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
        self._ready = True
        logger.info("Paraformer sherpa-onnx backend ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        import sherpa_onnx  # lazy import

        audio = _decode_audio(audio_bytes)

        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)

        # Feed end-of-stream silence so the endpoint detector fires
        tail_paddings = np.zeros(int(0.5 * 16000), dtype=np.float32)
        stream.accept_waveform(16000, tail_paddings)

        while self._recognizer.is_ready(stream):
            self._recognizer.decode_stream(stream)

        text = self._recognizer.get_result(stream).text.strip()
        return TranscriptionResult(text=text)

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        sherpa_stream = self._recognizer.create_stream()
        return ParaformerSherpaStream(self._recognizer, sherpa_stream)


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
        text = self._recognizer.get_result(self._stream).text.strip()
        is_endpoint = self._recognizer.is_endpoint(self._stream)
        return text, is_endpoint

    def finalize(self) -> str:
        """Feed tail silence, flush decoder, return final text."""
        # Pad with silence so endpoint detection triggers
        tail_paddings = np.zeros(int(0.5 * 16000), dtype=np.float32)
        self._stream.accept_waveform(16000, tail_paddings)

        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        return self._recognizer.get_result(self._stream).text.strip()


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

"""SenseVoice ASR backend: uses sherpa-onnx OfflineRecognizer on CPU.

SenseVoice supports 50+ languages and runs entirely on CPU via ONNX.
Pseudo-streaming is implemented using sherpa-onnx's VoiceActivityDetector
(Silero VAD): audio chunks are fed to the VAD, and each completed speech
segment is immediately transcribed with the offline recognizer.

Environment variables
---------------------
SENSEVOICE_MODEL_DIR   Directory containing model.onnx (or model.int8.onnx),
                       tokens.txt, and optionally silero_vad.onnx.
                       Default: /opt/asr/sensevoice
ASR_MODEL_DIR          Fallback for VAD model at <ASR_MODEL_DIR>/vad/silero_vad.onnx
                       Default: /opt/asr/models
"""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import numpy as np

from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language mapping: human-readable / BCP-47 → SenseVoice language tag
# ---------------------------------------------------------------------------

_LANGUAGE_MAP: dict[str, str] = {
    # English labels
    "auto": "auto",
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "cantonese": "yue",
    "yue": "yue",
    # BCP-47 / ISO 639-1
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "ja": "ja",
    "ko": "ko",
}


def _map_language(language: str) -> str:
    """Map a user-supplied language string to a SenseVoice language tag."""
    return _LANGUAGE_MAP.get(language.lower(), "auto")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class SenseVoiceSherpaBackend(ASRBackend):
    """ASR backend using SenseVoice via sherpa-onnx on CPU."""

    def __init__(self) -> None:
        self._recognizer = None
        self._vad_model_path: Optional[str] = None
        self._ready = False

    # ------------------------------------------------------------------
    # ASRBackend properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "sensevoice_sherpa"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._recognizer is not None

    # ------------------------------------------------------------------
    # Preload
    # ------------------------------------------------------------------

    def preload(self) -> None:
        import sherpa_onnx  # lazy import — not available on dev machines

        model_dir = os.environ.get("SENSEVOICE_MODEL_DIR", "/opt/asr/sensevoice")
        logger.info("Loading SenseVoice model from %s", model_dir)

        # Prefer int8 quantized model when available
        int8_path = os.path.join(model_dir, "model.int8.onnx")
        fp32_path = os.path.join(model_dir, "model.onnx")
        if os.path.isfile(int8_path):
            model_path = int8_path
            logger.info("Using int8 quantized model: %s", int8_path)
        elif os.path.isfile(fp32_path):
            model_path = fp32_path
            logger.info("Using fp32 model: %s", fp32_path)
        else:
            raise FileNotFoundError(
                f"No SenseVoice model found in {model_dir!r}. "
                "Expected 'model.onnx' or 'model.int8.onnx'."
            )

        tokens_path = os.path.join(model_dir, "tokens.txt")
        if not os.path.isfile(tokens_path):
            raise FileNotFoundError(f"tokens.txt not found: {tokens_path!r}")

        num_threads = int(os.environ.get("SENSEVOICE_NUM_THREADS", "2"))

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=num_threads,
            use_itn=True,
            language="auto",
            provider="cpu",
        )

        # Resolve VAD model path (used by SenseVoiceSherpaStream)
        self._vad_model_path = self._resolve_vad_model(model_dir)

        self._ready = True
        logger.info("SenseVoice sherpa-onnx backend ready.")

    # ------------------------------------------------------------------
    # Transcribe (offline / batch)
    # ------------------------------------------------------------------

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready — call preload() first")

        audio = self._decode_audio(audio_bytes)
        return self._transcribe_array(audio, language)

    # ------------------------------------------------------------------
    # Streaming (VAD-based pseudo-streaming)
    # ------------------------------------------------------------------

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready — call preload() first")

        if self._vad_model_path is None:
            raise RuntimeError(
                "silero_vad.onnx not found — streaming requires a VAD model. "
                "Place silero_vad.onnx in SENSEVOICE_MODEL_DIR or "
                "ASR_MODEL_DIR/vad/."
            )

        return SenseVoiceSherpaStream(
            recognizer=self._recognizer,
            vad_model_path=self._vad_model_path,
            language=language,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transcribe_array(self, audio: np.ndarray, language: str = "auto") -> TranscriptionResult:
        """Run offline recognizer on a float32 numpy array.

        Note: sherpa-onnx OfflineRecognizer.create_stream() does not accept a
        language argument — language is fixed at recognizer creation time (set
        to "auto" in preload()).  The per-call language parameter is ignored by
        the recognizer but the detected language is read back from stream.result.
        """
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()

        detected_lang = getattr(stream.result, "lang", None)

        return TranscriptionResult(
            text=text,
            language=detected_lang,
        )

    @staticmethod
    def _resolve_vad_model(model_dir: str) -> Optional[str]:
        """Return path to silero_vad.onnx, or None if not found."""
        # Check inside the SenseVoice model directory first
        candidate1 = os.path.join(model_dir, "silero_vad.onnx")
        if os.path.isfile(candidate1):
            return candidate1

        # Fall back to ASR_MODEL_DIR/vad/silero_vad.onnx
        asr_model_dir = os.environ.get("ASR_MODEL_DIR", "/opt/asr/models")
        candidate2 = os.path.join(asr_model_dir, "vad", "silero_vad.onnx")
        if os.path.isfile(candidate2):
            return candidate2

        logger.warning(
            "silero_vad.onnx not found (checked %s and %s). "
            "Streaming will be unavailable.",
            candidate1,
            candidate2,
        )
        return None

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        """Decode audio bytes (WAV/FLAC/etc.) to 16 kHz float32 mono numpy."""
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


# ---------------------------------------------------------------------------
# Streaming session (VAD-based pseudo-streaming)
# ---------------------------------------------------------------------------


class SenseVoiceSherpaStream(ASRStream):
    """Pseudo-streaming ASR session backed by Silero VAD + SenseVoice offline.

    Audio chunks fed via accept_waveform() are accumulated in a VAD buffer.
    Whenever the VAD detects the end of a speech segment the segment is
    immediately transcribed and appended to an internal transcript list.
    get_partial() returns the accumulated transcript.  finalize() flushes any
    remaining audio in the VAD buffer and returns the full transcript.
    """

    # sherpa-onnx VAD works in 512-sample windows at 16 kHz (~32 ms each).
    _WINDOW_SIZE = 512

    def __init__(
        self,
        recognizer,
        vad_model_path: str,
        language: str = "auto",
    ) -> None:
        import sherpa_onnx

        self._recognizer = recognizer
        self._language = language
        self._sv_lang = _map_language(language)
        self._segments: list[str] = []

        # Leftover samples that don't yet fill a complete VAD window
        self._pending: np.ndarray = np.empty(0, dtype=np.float32)

        # Build VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.silero_vad.min_silence_duration = 0.25
        vad_config.silero_vad.min_speech_duration = 0.1
        vad_config.sample_rate = 16000
        self._vad = sherpa_onnx.VoiceActivityDetector(
            vad_config, buffer_size_in_seconds=30
        )

    # ------------------------------------------------------------------
    # ASRStream interface
    # ------------------------------------------------------------------

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed float32 audio samples into the VAD pipeline."""
        audio = samples.astype(np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        # Prepend leftover samples from the previous call
        if self._pending.size > 0:
            audio = np.concatenate([self._pending, audio])

        # Feed complete windows to the VAD
        n_complete = (len(audio) // self._WINDOW_SIZE) * self._WINDOW_SIZE
        if n_complete > 0:
            self._vad.accept_waveform(audio[:n_complete])
            self._pending = audio[n_complete:].copy()
        else:
            self._pending = audio.copy()

        # Drain any completed speech segments
        self._drain_vad()

    def get_partial(self) -> tuple[str, bool]:
        """Return accumulated transcript so far (not a true partial result)."""
        return " ".join(self._segments), False

    def finalize(self) -> str:
        """Flush remaining audio, transcribe final segment, return full text."""
        # Feed any leftover samples (pad to a full window)
        if self._pending.size > 0:
            pad_len = self._WINDOW_SIZE - (self._pending.size % self._WINDOW_SIZE)
            if pad_len < self._WINDOW_SIZE:
                padded = np.concatenate([self._pending, np.zeros(pad_len, dtype=np.float32)])
            else:
                padded = self._pending
            self._vad.accept_waveform(padded)
            self._pending = np.empty(0, dtype=np.float32)

        # Signal end-of-stream so VAD flushes any in-progress segment
        self._vad.flush()

        # Drain remaining segments
        self._drain_vad()

        return " ".join(self._segments)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_vad(self) -> None:
        """Pop all completed speech segments from the VAD and transcribe them."""
        while not self._vad.empty():
            segment = self._vad.front
            self._vad.pop()
            samples: np.ndarray = np.array(segment.samples, dtype=np.float32)
            if samples.size == 0:
                continue
            text = self._transcribe_samples(samples)
            if text:
                self._segments.append(text)
                logger.debug("SenseVoice segment: %r", text)

    def _transcribe_samples(self, samples: np.ndarray) -> str:
        """Run the offline recognizer on a float32 sample array."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text.strip()


# ---------------------------------------------------------------------------
# Simple resampler (mirrors qwen3_rk.py — no librosa dependency)
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 audio array using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)

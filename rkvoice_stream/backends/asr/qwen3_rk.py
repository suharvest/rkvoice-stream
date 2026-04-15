"""Qwen3-ASR RK3576 backend: wraps qwen3asr library as ASRBackend.

With decoder_type="matmul", the decoder runs on CPU and there is no NPU
contention with the TTS vocoder, so the NPU lock is not used by ASR.

With decoder_type="rkllm", the RKLLM decoder uses the NPU and must hold the
NPU lock while running to avoid contention with the Matcha/Vocos TTS RKNN
models. The lock is shared across both backends via get_npu_lock().
"""

from __future__ import annotations

import io
import logging
import os
import threading
from typing import Optional

import numpy as np

# Import from engine package
from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Shared NPU lock — imported by TTS backend too when available
_npu_lock: Optional[threading.Lock] = None


def get_npu_lock() -> threading.Lock:
    """Return the shared NPU lock, creating it on first call."""
    global _npu_lock
    if _npu_lock is None:
        _npu_lock = threading.Lock()
    return _npu_lock


class Qwen3ASRRKBackend(ASRBackend):
    """ASR backend using Qwen3-ASR RKNN/RKLLM on RK3576."""

    def __init__(self):
        self._engine = None
        self._ready = False

    @property
    def name(self) -> str:
        return "qwen3_asr_rk"

    @property
    def capabilities(self) -> set[ASRCapability]:
        return {ASRCapability.OFFLINE, ASRCapability.STREAMING, ASRCapability.MULTI_LANGUAGE}

    @property
    def sample_rate(self) -> int:
        return 16000

    def is_ready(self) -> bool:
        return self._ready and self._engine is not None

    def preload(self) -> None:
        from rkvoice_stream.backends.asr.qwen3 import Qwen3ASREngine

        model_dir = os.environ.get("ASR_MODEL_DIR", "/opt/asr/models")
        decoder_type = os.environ.get("ASR_DECODER_TYPE", "matmul")
        logger.info("Loading Qwen3-ASR engine from %s (decoder_type=%s)", model_dir, decoder_type)

        # lib_path: only needed when decoder_type="rkllm"; ignored by matmul decoder.
        # Still pass it for backward compat if the env var is set.
        lib_path = os.environ.get("RKLLM_LIB_PATH")

        engine_kwargs = dict(
            model_dir=model_dir,
            platform="rk3576",
            decoder_type=decoder_type,
            decoder_exec_mode=os.environ.get("MATMUL_EXEC_MODE", "dual_core"),
            decoder_quant="w4a16",      # decoder_hf.w4a16.rk3576.rkllm / matmul weights
            encoder_sizes=[2, 4],       # 2s for short audio (faster), 4s for longer
            enabled_cpus=2,
            max_context_len=int(os.environ.get("RKLLM_MAX_CONTEXT_LEN", "512")),
            repeat_penalty=1.15,
            compact_suffix=True,
            verbose=True,
            npu_core_mask="NPU_CORE_1",  # Reserve NPU_CORE_0 for TTS vocoder
        )
        if lib_path:
            engine_kwargs["lib_path"] = lib_path

        self._engine = Qwen3ASREngine(**engine_kwargs)
        self._use_npu_lock = (decoder_type == "rkllm")
        if self._use_npu_lock:
            logger.info("NPU lock enabled for RKLLM decoder (shared with TTS).")
        self._ready = True
        logger.info("Qwen3-ASR RK backend ready.")

    def transcribe(self, audio_bytes: bytes, language: str = "auto") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        audio = self._decode_audio(audio_bytes)
        lang_hint = None if language == "auto" else language

        # max_new_tokens: For typical ASR (2-10s audio), 50-80 tokens suffice.
        # The default 500 lets the decoder wander past EOS, producing trailing
        # garbage (e.g. "你好世界" → "你好世界，你") because EOS logit is weak.
        max_new_tokens = int(os.environ.get("ASR_MAX_NEW_TOKENS", "80"))

        if self._use_npu_lock:
            # RKLLM decoder uses NPU — serialize with TTS RKNN models.
            with get_npu_lock():
                result = self._engine.transcribe(
                    audio=audio,
                    language=lang_hint,
                    chunk_size=2.0,
                    memory_num=2,
                    rollback_tokens=2,
                    max_new_tokens=max_new_tokens,
                )
        else:
            # matmul decoder runs on CPU, no NPU contention.
            result = self._engine.transcribe(
                audio=audio,
                language=lang_hint,
                chunk_size=2.0,
                memory_num=2,
                rollback_tokens=2,
                max_new_tokens=max_new_tokens,
            )

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            rtf=result.get("stats", {}).get("rtf"),
            enc_ms=result.get("stats", {}).get("enc_ms"),
            llm_ms=result.get("stats", {}).get("llm_ms"),
        )

    def create_stream(self, language: str = "auto") -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        lang_hint = None if language == "auto" else language
        stream_session = self._engine.create_stream(
            language=lang_hint,
            chunk_size=2.0,
            memory_num=2,
            rollback_tokens=2,
        )
        return Qwen3ASRRKStream(stream_session, use_npu_lock=self._use_npu_lock)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> np.ndarray:
        """Decode audio_bytes (WAV/FLAC/etc.) to 16kHz float32 mono numpy."""
        import soundfile as sf

        buf = io.BytesIO(audio_bytes)
        try:
            audio, sr = sf.read(buf, dtype="float32")
        except Exception as exc:
            raise ValueError(f"Cannot decode audio: {exc}") from exc

        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed (simple linear interpolation)
        if sr != 16000:
            logger.warning("Input sample rate %d != 16000, resampling.", sr)
            audio = _resample(audio, sr, 16000)

        return audio.astype(np.float32)


class Qwen3ASRRKStream(ASRStream):
    """Wraps StreamSession as ASRStream interface."""

    def __init__(self, stream_session, use_npu_lock: bool = False):
        self._stream = stream_session
        self._use_npu_lock = use_npu_lock

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        """Feed float32 audio (already in [-1,1]) into stream."""
        audio = samples.astype(np.float32)

        if samples.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            audio = _resample(audio, sample_rate, 16000)

        self._stream.feed_audio(audio)

    def prepare_finalize(self) -> None:
        self._stream.prepare_finalize()

    def finalize(self) -> str:
        if self._use_npu_lock:
            with get_npu_lock():
                result = self._stream.finish()
        else:
            result = self._stream.finish()
        return result["text"]

    def get_partial(self) -> tuple[str, bool]:
        result = self._stream.get_result()
        return result["text"], False


# ------------------------------------------------------------------
# Simple resampler (no librosa dependency)
# ------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D float32 audio array using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_len = int(round(duration * target_sr))
    x_old = np.linspace(0, 1, len(audio))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, audio).astype(np.float32)

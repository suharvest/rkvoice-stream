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
            platform=os.environ.get("ASR_PLATFORM", "rk3576"),
            decoder_type=decoder_type,
            decoder_exec_mode=os.environ.get("MATMUL_EXEC_MODE", "dual_core"),
            decoder_quant=os.environ.get("ASR_DECODER_QUANT", "w8a8"),  # decoder model quantization
            encoder_sizes=[2, 4],       # 2s for short audio (faster), 4s for longer
            enabled_cpus=int(os.environ.get("ASR_ENABLED_CPUS", "2")),
            max_context_len=int(os.environ.get("RKLLM_MAX_CONTEXT_LEN", "512")),
            repeat_penalty=1.15,
            compact_suffix=True,
            verbose=True,
            npu_core_mask=os.environ.get("ASR_NPU_CORE_MASK", "NPU_CORE_1"),
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
                    chunk_size=4.0,
                    memory_num=4,
                    rollback_tokens=2,
                    max_new_tokens=max_new_tokens,
                )
        else:
            # matmul decoder runs on CPU, no NPU contention.
            result = self._engine.transcribe(
                audio=audio,
                language=lang_hint,
                chunk_size=4.0,
                memory_num=4,
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

        # ── True streaming mode (port of Jetson Qwen3StreamingASRStream) ──
        # 400ms chunks + left-context encoder + VAD endpoint → emits early
        # final on VAD silence, so the WebSocket stop→final latency is near
        # zero when the user pauses naturally.
        true_streaming = os.environ.get(
            "QWEN3_ASR_STREAM_TRUE", "0") in ("1", "true", "yes")
        if true_streaming:
            from rkvoice_stream.backends.asr.qwen3.streaming import (
                Qwen3TrueStreamingASRStream,
            )
            vad = self._build_vad()
            npu_lock = get_npu_lock() if self._use_npu_lock else None
            stream = Qwen3TrueStreamingASRStream(
                engine=self._engine,
                language=lang_hint,
                context="",
                vad=vad,
                use_npu_lock=self._use_npu_lock,
                npu_lock=npu_lock,
            )
            return Qwen3ASRRKStream(stream, use_npu_lock=self._use_npu_lock)

        final_mode = os.environ.get("QWEN3_ASR_STREAM_FINAL_MODE", "offline")
        reuse_min_audio_ms = int(os.environ.get(
            "QWEN3_ASR_STREAM_REUSE_MIN_AUDIO_MS", "500"))
        # KV streaming: encode-only during speech, single decode at finalize.
        # Optimal stop-to-final-text latency for V2V. Off by default to keep
        # backwards-compatible behaviour for existing partial-text consumers.
        kv_streaming = os.environ.get(
            "QWEN3_ASR_STREAM_KV", "0") in ("1", "true", "yes")
        stream_session = self._engine.create_stream(
            language=lang_hint,
            chunk_size=4.0,
            memory_num=4,
            rollback_tokens=2,
            final_mode=final_mode,
            reuse_min_audio_ms=reuse_min_audio_ms,
            kv_streaming=kv_streaming,
        )
        return Qwen3ASRRKStream(stream_session, use_npu_lock=self._use_npu_lock)

    def _build_vad(self):
        """Construct a SileroVAD instance for true-streaming endpoint
        detection.  Returns ``None`` if the model isn't available — the
        true-streaming class then falls back to "always-speech" mode (no
        early endpoint, finalize only on stop signal).
        """
        try:
            from rkvoice_stream.vad.silero import SileroVAD
        except Exception as exc:  # pragma: no cover
            logger.warning("SileroVAD import failed (%s); endpoint disabled", exc)
            return None

        model_path = os.environ.get(
            "VAD_MODEL_PATH",
            os.path.join(
                os.environ.get("ASR_MODEL_DIR", "/opt/asr/models"),
                "vad", "silero_vad.onnx"),
        )
        if not os.path.exists(model_path):
            logger.warning(
                "Silero VAD model not found at %s; endpoint disabled", model_path)
            return None

        min_silence_s = float(os.environ.get(
            "VAD_ENDPOINT_SILENCE_MS", "400")) / 1000.0
        try:
            return SileroVAD(
                model_path=model_path,
                threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
                # Use a short internal min_silence so is_speech flips quickly
                # — we apply our own endpoint debounce on top.
                min_silence_duration=max(0.1, min_silence_s * 0.5),
                min_speech_duration=0.1,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("SileroVAD init failed: %s", exc)
            return None

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

    def cancel_and_finalize(self) -> None:
        """Hard-cancel pending partials and skip residual tail encode.

        Only implemented by the true-streaming session; legacy sessions fall
        back to a no-op (the subsequent ``finalize()`` will do the right
        thing — they don't have a pending-encode problem).
        """
        cancel = getattr(self._stream, "cancel_and_finalize", None)
        if callable(cancel):
            cancel()

    def finalize(self) -> dict:
        # The true-streaming class does its own NPU locking inside
        # _run_decoder. Acquiring the (non-reentrant) NPU lock here would
        # deadlock on the VAD-triggered early-final path or whenever the
        # final decode is invoked from within an already-locked context.
        own_lock = getattr(self._stream, "_npu_lock", None) is not None
        if self._use_npu_lock and not own_lock:
            with get_npu_lock():
                result = self._stream.finish()
        else:
            result = self._stream.finish()

        final_mode = result.get("final_mode", "offline")
        fallback = result.get("fallback")
        finalize_ms = result.get("finalize_ms", 0)
        if fallback:
            logger.info("ASR finalize: mode=%s fallback=%s ms=%.0f",
                        final_mode, fallback, finalize_ms)
        else:
            logger.info("ASR finalize: mode=%s (no fallback) ms=%.0f",
                        final_mode, finalize_ms)
        return result

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

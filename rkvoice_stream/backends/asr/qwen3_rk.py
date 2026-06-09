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
import time
from typing import Optional

import numpy as np

# Import from engine package
from rkvoice_stream.engine.asr import ASRBackend, ASRCapability, ASRStream, TranscriptionResult

logger = logging.getLogger(__name__)

# Shared NPU lock — imported by TTS backend too when available
_npu_lock: Optional[threading.Lock] = None

_TRUE_ENV = {"1", "true", "yes", "on"}
_FALSE_ENV = {"0", "false", "no", "off"}


def _validate_enabled_cpus(platform: str, enabled_cpus: int) -> int:
    """Validate RKLLM CPU affinity before the runtime fails or silently slows down."""
    supported = {2, 3, 4, 8}
    if enabled_cpus not in supported:
        raise ValueError(
            f"Unsupported ASR_ENABLED_CPUS={enabled_cpus}; "
            f"supported values for RK Qwen3-ASR are {sorted(supported)}"
        )

    platform_l = platform.lower()
    min_enabled = 3 if "rk3588" in platform_l else 2 if "rk3576" in platform_l else 1
    if enabled_cpus < min_enabled:
        raise ValueError(
            f"ASR_ENABLED_CPUS={enabled_cpus} is invalid for {platform}; "
            f"RKLLM requires enabled CPUs >= NPU core count. Use ASR_ENABLED_CPUS=4 "
            "for the current RK3576/RK3588 profiles."
        )
    if "rk3576" in platform_l and enabled_cpus == 2:
        logger.warning(
            "ASR_ENABLED_CPUS=2 is valid on RK3576 but was slower than 4 in "
            "2026-06-09 high-performance Qwen3-ASR A/B; use 4 unless memory or "
            "co-scheduling requires otherwise."
        )
    return enabled_cpus


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_ENV:
        return True
    if value in _FALSE_ENV:
        return False
    logger.warning("Invalid boolean env %s=%r; using default=%s", name, raw, default)
    return default


def _qwen3_stream_mode() -> str:
    """Resolve the RK Qwen3-ASR streaming implementation.

    Historically ``QWEN3_ASR_CHUNK_CONFIRM`` defaulted to enabled and was
    checked before ``QWEN3_ASR_STREAM_TRUE``. That made older RK profiles with
    only ``QWEN3_ASR_STREAM_TRUE=1`` silently run chunk-confirm. Prefer the
    explicit mode env when present, otherwise let STREAM_TRUE select true
    streaming unless CHUNK_CONFIRM is explicitly enabled.
    """
    raw_mode = os.environ.get("QWEN3_ASR_STREAM_MODE", "").strip().lower()
    mode = raw_mode.replace("-", "_")
    aliases = {
        "chunk": "chunk_confirm",
        "chunk_confirm": "chunk_confirm",
        "cc": "chunk_confirm",
        "true": "true_streaming",
        "true_stream": "true_streaming",
        "true_streaming": "true_streaming",
        "legacy": "legacy",
        "stream_session": "legacy",
        "default": "legacy",
    }
    if mode:
        resolved = aliases.get(mode)
        if not resolved:
            logger.warning(
                "Unknown QWEN3_ASR_STREAM_MODE=%r; falling back to env flags",
                raw_mode,
            )
        else:
            return resolved

    true_streaming = _env_bool("QWEN3_ASR_STREAM_TRUE", False)
    chunk_confirm_raw = os.environ.get("QWEN3_ASR_CHUNK_CONFIRM")
    if chunk_confirm_raw is not None:
        chunk_confirm = _env_bool("QWEN3_ASR_CHUNK_CONFIRM", False)
        if chunk_confirm and true_streaming:
            logger.warning(
                "Both QWEN3_ASR_CHUNK_CONFIRM and QWEN3_ASR_STREAM_TRUE are enabled; "
                "using chunk_confirm for backward compatibility. Set "
                "QWEN3_ASR_STREAM_MODE=true_streaming to override explicitly."
            )
        if chunk_confirm:
            return "chunk_confirm"

    if true_streaming:
        return "true_streaming"

    # Preserve the pre-true-streaming default when no selector is set.
    if chunk_confirm_raw is None:
        return "chunk_confirm"
    return "legacy"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer env %s=%r; using default=%d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using default=%s", name, raw, default)
        return default


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

    @property
    def prefer_backend_endpoint_vad(self) -> bool:
        return _qwen3_stream_mode() == "true_streaming"

    @property
    def allow_frontend_eou_finalize(self) -> bool:
        return _qwen3_stream_mode() == "true_streaming"

    @property
    def frontend_eou_min_audio_s(self) -> float:
        if _qwen3_stream_mode() != "true_streaming":
            return 0.0
        return max(0.0, _env_float("QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S", 2.5))

    def is_ready(self) -> bool:
        return self._ready and self._engine is not None

    def preload(self) -> None:
        from rkvoice_stream.backends.asr.qwen3 import Qwen3ASREngine

        model_dir = os.environ.get("ASR_MODEL_DIR", "/opt/asr/models")
        decoder_type = os.environ.get("ASR_DECODER_TYPE", "matmul")
        logger.info("Loading Qwen3-ASR engine from %s (decoder_type=%s)", model_dir, decoder_type)

        # lib_path: only needed when decoder_type="rkllm"; ignored by matmul decoder.
        lib_path = os.environ.get("RKLLM_LIB_PATH")

        # chunk-confirm mode needs larger encoder models (up to 30s) for
        # full-buffer re-encode.  Legacy modes only need 2s + 4s to save
        # NPU memory.  encoder_sizes=None loads all available sizes.
        stream_mode = _qwen3_stream_mode()
        cc_mode = stream_mode == "chunk_confirm"
        default_sizes = None if cc_mode else [2, 4]
        encoder_sizes_env = os.environ.get("ASR_ENCODER_SIZES", "")
        if encoder_sizes_env:
            default_sizes = [int(x.strip()) for x in encoder_sizes_env.split(",")]

        platform = os.environ.get("ASR_PLATFORM", "rk3576")
        enabled_cpus = _validate_enabled_cpus(
            platform,
            _env_int("ASR_ENABLED_CPUS", 4),
        )

        engine_kwargs = dict(
            model_dir=model_dir,
            platform=platform,
            decoder_type=decoder_type,
            decoder_exec_mode=os.environ.get("MATMUL_EXEC_MODE", "dual_core"),
            decoder_quant=os.environ.get("ASR_DECODER_QUANT", "w8a8"),
            encoder_sizes=default_sizes,
            enabled_cpus=enabled_cpus,
            max_context_len=_env_int("RKLLM_MAX_CONTEXT_LEN", 512),
            max_new_tokens=_env_int("ASR_MAX_NEW_TOKENS", 64),
            top_k=_env_int("ASR_TOP_K", 1),
            top_p=_env_float("ASR_TOP_P", 1.0),
            temperature=_env_float("ASR_TEMPERATURE", 1.0),
            repeat_penalty=_env_float("ASR_REPEAT_PENALTY", 1.15),
            frequency_penalty=_env_float("ASR_FREQUENCY_PENALTY", 0.0),
            presence_penalty=_env_float("ASR_PRESENCE_PENALTY", 0.0),
            embed_flash=_env_int("RKLLM_EMBED_FLASH", 1),
            compact_suffix=_env_bool("QWEN3_ASR_COMPACT_SUFFIX", True),
            final_stop_on_punctuation=_env_bool("ASR_FINAL_STOP_ON_PUNCT", True),
            final_stop_punctuation=os.environ.get(
                "ASR_FINAL_STOP_PUNCT_CHARS", "。！？.!?"
            ),
            final_stop_min_chars=_env_int("ASR_FINAL_STOP_MIN_CHARS", 8),
            final_stop_min_chunks=_env_int("ASR_FINAL_STOP_MIN_CHUNKS", 2),
            decoder_embed_cache_reuse=_env_bool(
                "ASR_DECODER_EMBED_CACHE_REUSE", False
            ),
            decoder_async_mode=_env_bool("ASR_DECODER_ASYNC", False),
            decoder_async_timeout_s=_env_float(
                "ASR_DECODER_ASYNC_TIMEOUT_S", 30.0
            ),
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

    def warmup(self) -> None:
        """Run one tiny real ASR pass to materialize RKNN/RKLLM runtime state.

        ``preload()`` loads the model files, but RKNN/RKLLM still pay first-run
        costs on the first inference.  Warmup intentionally uses the public
        streaming path so it exercises the same encoder, decoder, locks, and
        finalize policy as production requests.
        """
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        duration_s = max(0.1, _env_float("ASR_WARMUP_AUDIO_S", 0.8))
        amplitude = max(0.0, _env_float("ASR_WARMUP_AMPLITUDE", 1e-4))
        language = os.environ.get("ASR_WARMUP_LANGUAGE", "Chinese")
        seed = _env_int("ASR_WARMUP_SEED", 7)

        samples = int(round(duration_s * self.sample_rate))
        rng = np.random.default_rng(seed)
        audio = (rng.standard_normal(samples) * amplitude).astype(np.float32)

        t0 = time.perf_counter()
        stream = self.create_stream(language=language)
        try:
            stream.accept_waveform(self.sample_rate, audio)
            prepare = getattr(stream, "prepare_finalize", None)
            if callable(prepare):
                prepare()
            text, detected_language = stream.finalize()
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "Qwen3-ASR RK warmup: mode=%s audio=%.2fs elapsed=%.0fms "
            "text=%r language=%s",
            _qwen3_stream_mode(),
            duration_s,
            elapsed_ms,
            (text or "")[:40],
            detected_language or language,
        )

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

    def create_stream(
        self,
        language: str = "auto",
        stream_options: Optional[dict] = None,
    ) -> ASRStream:
        if not self.is_ready():
            raise RuntimeError("ASR backend not ready")

        lang_hint = None if language == "auto" else language
        stream_options = dict(stream_options or {})
        vad_endpoint_silence_ms = stream_options.get("vad_endpoint_silence_ms")
        vad_min_utterance_s = stream_options.get("vad_min_utterance_s")
        vad_min_audio_s = stream_options.get("vad_min_audio_s")

        stream_mode = _qwen3_stream_mode()

        # ── Chunk-and-Confirm mode (recipe §3, P0 streaming) ─────────
        # Each hop re-encodes full accumulated audio + uses prefix prompt
        # so the decoder only generates the uncertain tail.  VAD-aligned
        # segmentation for multi-utterance sessions.
        if stream_mode == "chunk_confirm":
            from rkvoice_stream.backends.asr.qwen3.chunk_confirm import (
                ChunkConfirmASRStream,
            )
            vad = self._build_vad(vad_endpoint_silence_ms=vad_endpoint_silence_ms)
            npu_lock = get_npu_lock() if self._use_npu_lock else None
            stream = ChunkConfirmASRStream(
                engine=self._engine,
                language=lang_hint,
                context="",
                vad=vad,
                use_npu_lock=self._use_npu_lock,
                npu_lock=npu_lock,
            )
            logger.info("ASR stream mode: chunk_confirm")
            return Qwen3ASRRKStream(stream, use_npu_lock=self._use_npu_lock)

        # ── True streaming mode (port of Jetson Qwen3StreamingASRStream) ──
        if stream_mode == "true_streaming":
            from rkvoice_stream.backends.asr.qwen3.streaming import (
                Qwen3TrueStreamingASRStream,
            )
            vad = self._build_vad(vad_endpoint_silence_ms=vad_endpoint_silence_ms)
            npu_lock = get_npu_lock() if self._use_npu_lock else None
            stream = Qwen3TrueStreamingASRStream(
                engine=self._engine,
                language=lang_hint,
                context="",
                vad=vad,
                use_npu_lock=self._use_npu_lock,
                npu_lock=npu_lock,
                vad_endpoint_silence_ms=vad_endpoint_silence_ms,
                vad_min_utterance_s=vad_min_utterance_s,
                vad_min_audio_s=vad_min_audio_s,
            )
            logger.info("ASR stream mode: true_streaming")
            return Qwen3ASRRKStream(
                stream,
                use_npu_lock=self._use_npu_lock,
                immediate_client_eos_cancel_safe=True,
                prefer_backend_endpoint_vad=True,
                allow_frontend_eou_finalize=True,
                frontend_eou_min_audio_s=self.frontend_eou_min_audio_s,
            )

        final_mode = os.environ.get("QWEN3_ASR_STREAM_FINAL_MODE", "offline")
        reuse_min_audio_ms = int(os.environ.get(
            "QWEN3_ASR_STREAM_REUSE_MIN_AUDIO_MS", "500"))
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

    def _build_vad(self, vad_endpoint_silence_ms: Optional[int] = None):
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

        min_silence_ms = (
            max(0, int(vad_endpoint_silence_ms))
            if vad_endpoint_silence_ms is not None
            else float(os.environ.get("VAD_ENDPOINT_SILENCE_MS", "400"))
        )
        min_silence_s = float(min_silence_ms) / 1000.0
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

    def __init__(
        self,
        stream_session,
        use_npu_lock: bool = False,
        immediate_client_eos_cancel_safe: bool = False,
        prefer_backend_endpoint_vad: bool = False,
        allow_frontend_eou_finalize: bool = False,
        frontend_eou_min_audio_s: float = 0.0,
    ):
        self._stream = stream_session
        self._use_npu_lock = use_npu_lock
        self.immediate_client_eos_cancel_safe = immediate_client_eos_cancel_safe
        self.prefer_backend_endpoint_vad = prefer_backend_endpoint_vad
        self.allow_frontend_eou_finalize = allow_frontend_eou_finalize
        self.frontend_eou_min_audio_s = max(0.0, float(frontend_eou_min_audio_s or 0.0))

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

    def abort_partial_decode(self) -> None:
        """Abort in-flight partial decode without dropping queued audio."""
        abort = getattr(self._stream, "abort_partial_decode", None)
        if callable(abort):
            abort()

    def finalize(self) -> tuple[str, Optional[str]]:
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
        return result.get("text", "") or "", result.get("language")

    def get_partial(self) -> tuple[str, bool]:
        result = self._stream.get_result()
        return result["text"], result.get("is_final", False)


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

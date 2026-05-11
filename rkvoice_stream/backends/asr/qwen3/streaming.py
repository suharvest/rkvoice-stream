"""True streaming Qwen3-ASR session for RK3576/RK3588.

Port of the Jetson ``Qwen3StreamingASRStream`` (jetson-voice
``app/backends/qwen3_asr.py:313``) adapted to:

* RKNN encoder (fixed-shape, padded to 2s/4s) — returns ``(hidden, enc_ms,
  model_sec)`` with ``hidden`` shaped ``(T_tokens, 1024)``.
* RKLLM decoder (``decoder.run_embed(embd, n_tokens, keep_history=0)`` — single
  prefill+autoregressive call). Partial decode budget is enforced via
  ``decoder._early_stop_tokens``.
* ``engine.build_embed`` to assemble the full Qwen3 prompt (audio embeddings +
  language/instruction suffix tokens) instead of manual prompt construction.
* Silero VAD (``rkvoice_stream.vad.silero.SileroVAD``) for endpoint detection
  via trailing-silence on ``is_speech`` transitions.

Architecture (per chunk, 400 ms):
  1. Encode (left-context audio + new audio) → trim context frames.
  2. Append frames to a rolling encoder-output buffer (≤ 5 s).
  3. Run a budget-limited partial decode and emit incremental text.
  4. Track VAD speech/silence; when silence ≥ ``VAD_ENDPOINT_SILENCE_MS`` after
     ≥ ``VAD_MIN_UTTERANCE_S`` of speech → fire an early final decode and mark
     the episode as final.  ``finalize()`` then returns instantly.

The early-final-on-VAD overlap is the key V2V win: by the time the WebSocket
``stop`` arrives, the final decode has already (mostly) completed.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE
from .utils import apply_itn, parse_asr_output

logger = logging.getLogger(__name__)


# ── True-streaming parameters ─────────────────────────────────────────────
CHUNK_SIZE_SEC = float(os.environ.get("QWEN3_ASR_TRUE_CHUNK_SEC", "0.4"))
LEFT_CONTEXT_SEC = float(os.environ.get("QWEN3_ASR_TRUE_LCTX_SEC", "1.0"))
ENCODER_HOP_SAMPLES = 1280                  # mel hop 160 × conv stride 8
ROLLING_BUFFER_SEC = float(os.environ.get("QWEN3_ASR_TRUE_ROLL_SEC", "5.0"))
PARTIAL_MAX_TOKENS = int(os.environ.get("QWEN3_ASR_TRUE_PARTIAL_TOKENS", "12"))
# Minimum wall-clock interval between partial decodes (ms). On RK each
# partial costs ~600-900 ms of NPU time; running one per 400 ms chunk
# backs the executor up by ~2× and balloons stop→final latency.
PARTIAL_MIN_INTERVAL_MS = int(
    os.environ.get("QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS", "900"))
# How many chunks (each CHUNK_SIZE_SEC) to skip at the start before running
# the first partial. Earliest possible partial = (warmup+1) chunks of audio.
PARTIAL_WARMUP_CHUNKS = int(
    os.environ.get("QWEN3_ASR_TRUE_PARTIAL_WARMUP", "1"))

# ── VAD endpoint parameters ──────────────────────────────────────────────
VAD_ENDPOINT_SILENCE_MS = int(os.environ.get("VAD_ENDPOINT_SILENCE_MS", "400"))
VAD_MIN_UTTERANCE_S = float(os.environ.get("VAD_MIN_UTTERANCE_S", "0.5"))
# Silero ``is_speech`` flickers True for ~1–2 frames during dense Chinese
# inter-character pauses (commas, polysyllabic transitions).  Resetting the
# silence accumulator on instantaneous True frames defeats the endpoint.
# Require N **consecutive** speech frames before considering the speaker
# truly active again; isolated True flips count as silence and are absorbed
# by the accumulator.
VAD_SUSTAIN_FRAMES = int(os.environ.get("QWEN3_ASR_VAD_SUSTAIN_FRAMES", "3"))

# Backend selector for the VAD endpoint detector: "webrtc" (preferred for
# dense CJK content — webrtcvad reports clean silence ~immediately after
# speech ends, where Silero stays True for ~400 ms on commas/inter-syll
# pauses) or "silero" (legacy / non-CJK).  "auto" prefers webrtc when the
# ``webrtcvad`` module is importable, else falls back to silero.
VAD_BACKEND_ENV = os.environ.get("QWEN3_ASR_VAD_BACKEND", "webrtc").lower()
# webrtcvad aggressiveness 0..3 (3 = most aggressive at filtering non-speech)
VAD_WEBRTC_AGGR = int(os.environ.get("QWEN3_ASR_VAD_WEBRTC_AGGR", "2"))
# webrtcvad frame size in ms (must be 10/20/30; 20 ms = 320 samples @16k)
VAD_WEBRTC_FRAME_MS = int(os.environ.get("QWEN3_ASR_VAD_WEBRTC_FRAME_MS", "20"))


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        0x3400 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2FFFF
    )


class Qwen3TrueStreamingASRStream:
    """True streaming session: 400 ms chunks + left-context + VAD endpoint."""

    def __init__(self, engine, language: Optional[str] = None,
                 context: str = "",
                 vad=None,
                 use_npu_lock: bool = False,
                 npu_lock: Optional[threading.Lock] = None):
        self._engine = engine
        self._language = language
        self._context = context
        self._vad = vad
        self._use_npu_lock = use_npu_lock
        self._npu_lock = npu_lock

        self._chunk_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)
        self._left_context_samples = int(LEFT_CONTEXT_SEC * SAMPLE_RATE)

        # Audio buffers
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._processed_samples = 0
        self._utterance_audio_buffer: list[np.ndarray] = []

        # Rolling encoder-output buffer (frames for decoder prefill).
        # Each entry is (T_frames, 1024).
        self._encoder_frames: list[np.ndarray] = []
        self._total_encoder_frames = 0
        self._max_encoder_frames = int(ROLLING_BUFFER_SEC * 13)  # ~13 fps

        # Output state
        self._archive_text: str = ""
        self._partial_text: str = ""
        self._current_language: str = language or ""
        self._episode_final: bool = False

        # VAD state (silence accumulator)
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._last_is_speech: Optional[bool] = None
        # Sustained-speech filter: count consecutive True frames so isolated
        # flicker mid-utterance doesn't reset the silence accumulator.
        self._vad_consec_speech_frames = 0
        # Buffer of pending-silence samples that we tentatively credit while
        # waiting to see if a speech blip is sustained.  If the blip ends up
        # being sustained → we drop the buffer.  If it ends up isolated → we
        # commit the buffer to ``_vad_silence_samples``.
        self._vad_pending_silence_samples = 0

        # Stats
        self._n_chunks = 0
        self._total_audio_s = 0.0
        self._total_enc_ms = 0.0
        self._total_dec_ms = 0.0

        # Partial-decode throttling
        self._last_partial_ts = 0.0
        # Set True once finalize/prepare_finalize starts → suppress any
        # further partials so the executor isn't blocked by stale decodes.
        self._finalizing = False

        # VAD-triggered final decode runs in a background thread to avoid
        # blocking the WS executor loop.  ``finish()`` joins it.
        self._final_decode_thread: Optional[threading.Thread] = None
        self._final_decode_thread_started = False
        # True while the (sync) VAD-triggered final decode is running.
        # Used by the WS server to know not to abort the in-flight decoder
        # call when EOU arrives — that would truncate the transcript.
        self._final_decode_in_progress = False

        # webrtcvad backend (preferred). Falls back to Silero if webrtcvad
        # is not importable or backend explicitly set to "silero".
        self._webrtc_vad = None
        self._webrtc_frame_samples = int(
            VAD_WEBRTC_FRAME_MS * SAMPLE_RATE / 1000)
        self._webrtc_carry = np.zeros(0, dtype=np.float32)
        if VAD_BACKEND_ENV in ("webrtc", "auto"):
            try:
                import webrtcvad as _wv  # type: ignore
                self._webrtc_vad = _wv.Vad(VAD_WEBRTC_AGGR)
                logger.info(
                    "Qwen3 streaming VAD backend: webrtcvad "
                    "(aggr=%d frame=%dms)", VAD_WEBRTC_AGGR,
                    VAD_WEBRTC_FRAME_MS)
            except Exception as exc:
                if VAD_BACKEND_ENV == "webrtc":
                    logger.warning(
                        "webrtcvad requested but unavailable (%s); "
                        "falling back to silero", exc)
                self._webrtc_vad = None
        if self._webrtc_vad is None and self._vad is not None:
            logger.info("Qwen3 streaming VAD backend: silero (fallback)")

    # ── ASRStream API expected by Qwen3ASRRKStream wrapper ────────────

    def feed_audio(self, pcm16k: np.ndarray) -> dict:
        """Accept a chunk of 16 kHz float32 PCM and process complete chunks."""
        x = np.asarray(pcm16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)

        # New utterance after VAD-triggered final → reset on incoming speech.
        if self._episode_final:
            self._maybe_resume_new_utterance(x)

        self._audio_buf = np.concatenate([self._audio_buf, x])
        self._utterance_audio_buffer.append(x.copy())

        # Once the episode is final (VAD pre-fired) or we're in the finalize
        # window, drop residual audio without encoding — saves ~200-600 ms
        # of stop→final latency that would otherwise be spent encoding the
        # silence trailer or any post-EOU residual.
        if self._episode_final or self._finalizing:
            self._processed_samples = len(self._audio_buf)
            return {
                "language": self._current_language,
                "text": self._composed_text(),
                "is_final": self._episode_final,
                "is_speech": False,
                "chunks_processed": self._n_chunks,
            }

        # Maintain VAD speech / silence accumulators.
        self._update_vad(x)

        # Process complete 400 ms chunks.
        while len(self._audio_buf) - self._processed_samples >= self._chunk_samples:
            self._process_streaming_chunk()

        # Check VAD endpoint AFTER processing → if endpoint, fire final decode.
        # We run synchronously here (same executor thread as feed_audio).
        # A background-thread variant was tried but doesn't help because
        # ``finish()`` joins anyway, and serializes against any concurrent
        # partial decode on the single-core NPU decoder path.
        if os.environ.get("QWEN3_ASR_DEBUG_VAD", "0") == "1":
            logger.info(
                "VAD after-feed: speech=%.2fs silence=%.0fms pending=%.0fms is_speech=%s",
                self._vad_speech_samples / SAMPLE_RATE,
                self._vad_silence_samples * 1000 / SAMPLE_RATE,
                self._vad_pending_silence_samples * 1000 / SAMPLE_RATE,
                self._last_is_speech)
        if self._check_vad_endpoint() and not self._final_decode_thread_started:
            self._final_decode_thread_started = True
            try:
                self._do_final_decode()
            except Exception as exc:  # pragma: no cover
                logger.warning("VAD-triggered final decode failed: %s", exc)

        # Trim _audio_buf periodically (keep 2× left_context headroom).
        max_prefix = self._left_context_samples + self._chunk_samples
        if self._processed_samples > max_prefix * 2:
            trim = self._processed_samples - max_prefix
            self._audio_buf = self._audio_buf[trim:]
            self._processed_samples -= trim

        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._episode_final,
            "is_speech": True,
            "chunks_processed": self._n_chunks,
        }

    def get_result(self) -> dict:
        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._episode_final,
            "is_speech": True,
        }

    def prepare_finalize(self) -> None:
        """Drain any sub-chunk audio into encoder frames before finalize."""
        logger.debug(
            "VAD state at finalize: speech=%.2fs silence=%.0fms pending=%.0fms "
            "consec=%d is_speech=%s episode_final=%s",
            self._vad_speech_samples / SAMPLE_RATE,
            self._vad_silence_samples * 1000 / SAMPLE_RATE,
            self._vad_pending_silence_samples * 1000 / SAMPLE_RATE,
            self._vad_consec_speech_frames,
            self._last_is_speech,
            self._episode_final,
        )
        self._finalizing = True
        # If a partial decode is currently in flight (different executor
        # thread), abort it so this thread can grab the NPU lock quickly.
        # Skip the abort if the in-flight call is the *final* decode —
        # aborting it would truncate the transcript.
        if not self._final_decode_in_progress:
            try:
                self._engine.decoder.abort()
            except Exception:
                pass
        if self._episode_final:
            return
        # Drain whole chunks.
        while len(self._audio_buf) - self._processed_samples >= self._chunk_samples:
            self._process_streaming_chunk()

        tail_len = len(self._audio_buf) - self._processed_samples
        if tail_len > 0:
            ctx_audio, n_ctx = self._get_left_context(self._processed_samples)
            new_audio = self._audio_buf[self._processed_samples:]
            t0 = time.perf_counter()
            enc_out = self._encode_with_context(ctx_audio, new_audio, n_ctx)
            self._total_enc_ms += (time.perf_counter() - t0) * 1000
            if enc_out is not None and enc_out.shape[0] > 0:
                self._encoder_frames.append(enc_out)
                self._total_encoder_frames += enc_out.shape[0]
            self._processed_samples = len(self._audio_buf)

    def cancel_and_finalize(self) -> None:
        """Hard-cancel any pending partial decode and drop residual sub-chunk
        audio.  Caller should follow with ``finish()`` to obtain the final.

        Difference from ``prepare_finalize``:
        * Sub-chunk tail audio is *dropped* rather than encoded — avoids the
          200-600 ms encoder pass when the user signals EOU explicitly.
        * Any in-flight partial decode on another thread is aborted.
        * If VAD already pre-fired (``_episode_final``), this is a no-op.
        """
        self._finalizing = True
        # Only abort if a final decode is NOT in progress — aborting it
        # would truncate the transcript. Partial decodes are fine to abort.
        if not self._final_decode_in_progress:
            try:
                self._engine.decoder.abort()
            except Exception:
                pass
        if self._episode_final:
            return
        # Drop residual tail audio entirely — we already have ≥ N×400 ms of
        # encoder frames in the rolling buffer; trailing 0-400 ms doesn't
        # change the transcript meaningfully and skipping the encode is
        # worth ~200-600 ms of stop→final latency.
        self._processed_samples = len(self._audio_buf)

    def finish(self, apply_itn_flag: bool = True) -> dict:
        """Finalize and return dict matching StreamSession.finish() schema."""
        self._finalizing = True
        if not self._final_decode_in_progress:
            try:
                self._engine.decoder.abort()
            except Exception:
                pass
        t0 = time.perf_counter()

        if not self._episode_final:
            self.prepare_finalize()
            if self._encoder_frames:
                text = self._final_decode_text()
                self._archive_text = text
            else:
                # No audio buffered → empty result.
                self._archive_text = ""
            self._episode_final = True

        finalize_ms = (time.perf_counter() - t0) * 1000
        text = self._archive_text
        if apply_itn_flag:
            text = apply_itn(text)

        logger.info(
            "Qwen3-true-stream finalize: %d chunks, %.2fs audio, "
            "enc=%.0fms dec=%.0fms, finalize=%.0fms text=%r",
            self._n_chunks, self._total_audio_s,
            self._total_enc_ms, self._total_dec_ms, finalize_ms,
            text[:60],
        )

        rtf = ((self._total_enc_ms + self._total_dec_ms) / 1000.0
               / max(self._total_audio_s, 0.1))
        return {
            "language": self._current_language,
            "text": text,
            "is_final": True,
            "final_mode": "true_streaming",
            "fallback": None,
            "finalize_ms": finalize_ms,
            "stats": {
                "total_enc_ms": self._total_enc_ms,
                "total_llm_ms": self._total_dec_ms,
                "total_audio_s": self._total_audio_s,
                "total_chunks": self._n_chunks,
                "rtf": rtf,
            },
        }

    # ── Internal: audio / encoder ────────────────────────────────────

    def _get_left_context(self, chunk_start: int) -> tuple[np.ndarray, int]:
        ctx_start = max(0, chunk_start - self._left_context_samples)
        return self._audio_buf[ctx_start:chunk_start], chunk_start - ctx_start

    def _encode_with_context(
        self, ctx_audio: np.ndarray, new_audio: np.ndarray, n_context: int,
    ) -> Optional[np.ndarray]:
        """Encode (context + new) and return only the *new* portion of frames.

        Returns 2D ``(T_new, 1024)`` array or None.
        """
        if len(ctx_audio) > 0:
            audio = np.concatenate([ctx_audio, new_audio])
        else:
            audio = new_audio

        enc_result = self._engine.encoder.encode(audio)
        if isinstance(enc_result, tuple):
            hidden = enc_result[0]
        else:
            hidden = enc_result
        # hidden shape: (T_tokens, 1024)  (RK encoder strips batch dim)
        if hidden.ndim == 3:
            hidden = hidden[0]

        trim = int(n_context // ENCODER_HOP_SAMPLES)
        if trim >= hidden.shape[0]:
            return None
        return hidden[trim:, :]

    # ── Internal: VAD ────────────────────────────────────────────────

    def _update_vad(self, samples: np.ndarray) -> None:
        if self._webrtc_vad is not None:
            self._update_vad_webrtc(samples)
            return
        if self._vad is None:
            # No VAD: treat all audio as speech for the min-utterance gate.
            self._vad_speech_samples += len(samples)
            self._vad_silence_samples = 0
            return

        t0 = time.perf_counter()
        self._vad.feed(samples)
        # SileroVAD ``is_speech`` reflects current internal state.
        is_speech = self._vad.is_speech
        n = len(samples)
        if is_speech:
            # Tentatively count as speech, but only commit (and reset the
            # silence accumulator) once we have ≥ VAD_SUSTAIN_FRAMES in a row.
            # Until sustained, hold the would-be silence in a pending bucket;
            # if the blip turns out to be flicker, we'll commit it.
            self._vad_consec_speech_frames += 1
            self._vad_speech_samples += n
            if self._vad_consec_speech_frames >= VAD_SUSTAIN_FRAMES:
                # Sustained speech → speaker is genuinely active.  Drop the
                # tentative-silence buffer (it was the gap between sustained
                # bursts, not real end-of-utterance) and reset the silence
                # accumulator.
                self._vad_pending_silence_samples = 0
                self._vad_silence_samples = 0
            else:
                # Not yet sustained: treat this frame as pending silence
                # (matches the "ignore short flicker" policy).
                if self._vad_speech_samples > 0:
                    self._vad_pending_silence_samples += n
        else:
            self._vad_consec_speech_frames = 0
            if self._vad_speech_samples > 0:
                # Commit any buffered pending-silence (flicker that didn't
                # become sustained) plus this frame.
                self._vad_silence_samples += (
                    self._vad_pending_silence_samples + n)
                self._vad_pending_silence_samples = 0
        self._last_is_speech = is_speech
        # Drain any segmenter output (not used; we rely on is_speech).
        while self._vad.has_speech():
            self._vad.pop_speech()
        # Time consumed by VAD is accounted for in encoder timing bucket
        # (negligible — Silero ~1.5 ms / 32 ms frame).
        _ = t0

    def _update_vad_webrtc(self, samples: np.ndarray) -> None:
        """webrtcvad-based accumulator. Processes audio in 20 ms int16 frames.

        Carries over any sub-frame remainder for the next call.  Applies the
        same sustain-filter logic as the Silero path so isolated speech blips
        mid-silence don't reset the silence accumulator.
        """
        if len(samples) == 0:
            return
        buf = np.concatenate([self._webrtc_carry, samples])
        frame_len = self._webrtc_frame_samples
        n_frames = len(buf) // frame_len
        if n_frames == 0:
            self._webrtc_carry = buf
            return
        used = n_frames * frame_len
        pcm = (np.clip(buf[:used], -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        self._webrtc_carry = buf[used:]
        fb = frame_len * 2  # bytes per frame
        for i in range(n_frames):
            try:
                is_speech = self._webrtc_vad.is_speech(
                    pcm[i * fb:(i + 1) * fb], SAMPLE_RATE)
            except Exception:
                is_speech = False
            n = frame_len
            if is_speech:
                self._vad_consec_speech_frames += 1
                self._vad_speech_samples += n
                if self._vad_consec_speech_frames >= VAD_SUSTAIN_FRAMES:
                    self._vad_pending_silence_samples = 0
                    self._vad_silence_samples = 0
                else:
                    if self._vad_speech_samples > 0:
                        self._vad_pending_silence_samples += n
            else:
                self._vad_consec_speech_frames = 0
                if self._vad_speech_samples > 0:
                    self._vad_silence_samples += (
                        self._vad_pending_silence_samples + n)
                    self._vad_pending_silence_samples = 0
            self._last_is_speech = is_speech

    def _check_vad_endpoint(self) -> bool:
        if self._episode_final:
            return False
        min_speech = int(VAD_MIN_UTTERANCE_S * SAMPLE_RATE)
        min_silence = int(VAD_ENDPOINT_SILENCE_MS * SAMPLE_RATE / 1000)
        return (
            self._vad_speech_samples >= min_speech
            and self._vad_silence_samples >= min_silence
        )

    def _maybe_resume_new_utterance(self, samples: np.ndarray) -> None:
        """If a new utterance starts after an endpoint, reset state."""
        # Lightweight energy gate (avoid false-resets on pure silence between
        # utterances). VAD state from the previous utterance is irrelevant here.
        if len(samples) == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms <= 1e-3:
            return
        # Full reset for a new utterance.
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._processed_samples = 0
        self._utterance_audio_buffer = []
        self._encoder_frames = []
        self._total_encoder_frames = 0
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._vad_consec_speech_frames = 0
        self._vad_pending_silence_samples = 0
        self._webrtc_carry = np.zeros(0, dtype=np.float32)
        self._final_decode_thread = None
        self._final_decode_thread_started = False
        self._final_decode_in_progress = False
        self._finalizing = False
        self._partial_text = ""
        self._archive_text = ""
        self._episode_final = False
        if self._vad is not None:
            try:
                self._vad.reset()
            except Exception:
                pass

    # ── Internal: chunk processing ────────────────────────────────────

    def _process_streaming_chunk(self) -> None:
        new_start = self._processed_samples
        new_end = new_start + self._chunk_samples
        new_audio = self._audio_buf[new_start:new_end]
        ctx_audio, n_context = self._get_left_context(new_start)

        t0 = time.perf_counter()
        enc_out = self._encode_with_context(ctx_audio, new_audio, n_context)
        self._total_enc_ms += (time.perf_counter() - t0) * 1000

        self._processed_samples = new_end
        self._n_chunks += 1
        self._total_audio_s += self._chunk_samples / SAMPLE_RATE

        if enc_out is None or enc_out.shape[0] == 0:
            return

        # Append to rolling encoder buffer.
        self._encoder_frames.append(enc_out)
        self._total_encoder_frames += enc_out.shape[0]
        while (self._total_encoder_frames > self._max_encoder_frames
               and len(self._encoder_frames) > 1):
            dropped = self._encoder_frames.pop(0)
            self._total_encoder_frames -= dropped.shape[0]

        if self._episode_final or not self._encoder_frames or self._finalizing:
            return

        if os.environ.get("QWEN3_ASR_STREAM_PARTIAL", "1").lower() in ("0", "false", "no"):
            return

        # Throttle partials: skip warmup chunks + min interval since last.
        if self._n_chunks <= PARTIAL_WARMUP_CHUNKS:
            return
        now = time.monotonic() * 1000
        if (self._last_partial_ts > 0
                and now - self._last_partial_ts < PARTIAL_MIN_INTERVAL_MS):
            return

        # Partial decode.
        all_frames = np.concatenate(self._encoder_frames, axis=0)
        t0 = time.perf_counter()
        text = self._decode_partial(all_frames)
        self._total_dec_ms += (time.perf_counter() - t0) * 1000
        self._last_partial_ts = time.monotonic() * 1000
        if text:
            self._partial_text = text.strip()

    # ── Internal: decode ─────────────────────────────────────────────

    def _run_decoder(self, full_embd: np.ndarray, n_tokens: int,
                     early_stop: int) -> dict:
        decoder = self._engine.decoder
        decoder._early_stop_tokens = early_stop
        try:
            if self._use_npu_lock and self._npu_lock is not None:
                with self._npu_lock:
                    result = decoder.run_embed(full_embd, n_tokens, keep_history=0)
            else:
                result = decoder.run_embed(full_embd, n_tokens, keep_history=0)
        finally:
            decoder._early_stop_tokens = 0
        return result

    def _decode_partial(self, all_frames: np.ndarray) -> str:
        full_embd, n_tokens = self._engine.build_embed(
            all_frames, prefix_text="",
            language=self._language, context=self._context,
            skip_prefix=False)
        result = self._run_decoder(full_embd, n_tokens, PARTIAL_MAX_TOKENS)
        raw = result.get("text", "") or ""
        was_aborted = result.get("aborted", False)
        early_stopped = True  # partial always runs with early_stop>0
        if was_aborted and not early_stopped:
            return ""

        if self._language:
            text = raw
        else:
            lang, text = parse_asr_output(raw)
            if lang:
                self._current_language = lang
        return text or ""

    def _decode_final(self, all_frames: np.ndarray) -> str:
        full_embd, n_tokens = self._engine.build_embed(
            all_frames, prefix_text="",
            language=self._language, context=self._context,
            skip_prefix=False)
        result = self._run_decoder(full_embd, n_tokens, 0)
        raw = result.get("text", "") or ""
        was_aborted = result.get("aborted", False)
        if was_aborted:
            # Repetition abort — keep the (truncated) text from decoder.
            pass
        if self._language:
            text = raw
        else:
            lang, text = parse_asr_output(raw)
            if lang:
                self._current_language = lang
        return text or ""

    def _final_decode_text(self) -> str:
        if not self._encoder_frames:
            return self._partial_text
        all_frames = np.concatenate(self._encoder_frames, axis=0)
        return self._decode_final(all_frames)

    def _do_final_decode(self) -> None:
        """VAD-triggered early final decode."""
        if not self._encoder_frames:
            self._episode_final = True
            return
        self._final_decode_in_progress = True
        try:
            text = self._final_decode_text()
        finally:
            self._final_decode_in_progress = False
        self._archive_text = text
        self._partial_text = ""
        self._episode_final = True
        logger.info("VAD endpoint: text=%r (silence=%.0fms speech=%.2fs)",
                    text[:60],
                    self._vad_silence_samples * 1000 / SAMPLE_RATE,
                    self._vad_speech_samples / SAMPLE_RATE)

    def _do_final_decode_safe(self) -> None:
        try:
            self._do_final_decode()
        except Exception as exc:  # pragma: no cover
            logger.warning("VAD-triggered final decode (thread) failed: %s", exc)

    # ── Internal: helpers ────────────────────────────────────────────

    def _composed_text(self) -> str:
        if self._episode_final or not self._partial_text:
            return self._archive_text
        if not self._archive_text:
            return self._partial_text
        sep = "" if _is_cjk(self._archive_text[-1]) else " "
        return (self._archive_text + sep + self._partial_text).strip()

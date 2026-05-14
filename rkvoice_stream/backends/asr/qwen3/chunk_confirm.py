"""Chunk-and-Confirm streaming ASR session for RK3576/RK3588.

Implements the P0 streaming recipe from
``qwen3-edgellm-jetson/docs/qwen3-asr-streaming-conversion-recipe.md`` §3,
adapted to RKNN encoder + RKLLM/Matmul decoder.

Core algorithm (mirrors official Qwen3-ASR ``qwen3_asr.py:657-765``):

1. Accumulate audio in a per-utterance buffer.
2. Every ``chunk_size_sec`` hop: encode the **full** accumulated utterance audio,
   build the Qwen3 prompt embed with a prefix-text prompt (previous decode minus
   the last K tokens), then run the decoder.  The decoder sees the prefix as
   prefill context and only generates continuation tokens.
3. Prefix rollback + UTF-8 guard avoids boundary artefacts.
4. VAD-aligned segmentation: VAD speech/silence transitions trigger utterance
   boundaries.  Each utterance runs its own chunk-and-confirm sub-session.

Architecture::

    feed_audio(pcm)
      ├── _update_vad(pcm)           # sustain-filter VAD accumulator
      ├── speech → accumulate
      ├── silence after speech → _finalize_utterance() → save segment
      └── while buffer ≥ hop_threshold:
            ├── _audio_accum += chunk
            └── _run_hop()
                  ├── encoder.encode(_audio_accum)   # full re-encode
                  ├── _compute_prefix()              # UTF-8 guarded rollback
                  ├── _decode(embed, n_tokens)        # NPU-locked decoder
                  └── parse output, update state
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Optional

import numpy as np

from .config import SAMPLE_RATE
from .utils import apply_itn, parse_asr_output

logger = logging.getLogger(__name__)

# ── Hop timing ────────────────────────────────────────────────────────────
CHUNK_SIZE_SEC = float(os.environ.get("QWEN3_ASR_CC_CHUNK_SEC", "0.5"))

# ── Prefix rollback (official qwen3_asr.py:588-590) ──────────────────────
UNFIXED_CHUNK_NUM = int(os.environ.get("QWEN3_ASR_CC_UNFIXED_CHUNKS", "2"))
UNFIXED_TOKEN_NUM = int(os.environ.get("QWEN3_ASR_CC_UNFIXED_TOKENS", "5"))

# ── Decode budget ────────────────────────────────────────────────────────
MAX_DECODE_TOKENS = int(os.environ.get("QWEN3_ASR_CC_MAX_TOKENS", "64"))

# ── Auto-segmentation (recipe §4.2) ──────────────────────────────────────
AUTO_SEGMENT_CAP_SEC = float(os.environ.get("QWEN3_ASR_CC_SEGMENT_CAP_S", "28"))
CARRYOVER_SEC = float(os.environ.get("QWEN3_ASR_CC_CARRYOVER_S", "0.8"))

# ── VAD endpoint (from streaming.py) ─────────────────────────────────────
VAD_ENDPOINT_SILENCE_MS = int(os.environ.get("VAD_ENDPOINT_SILENCE_MS", "400"))
VAD_MIN_UTTERANCE_S = float(os.environ.get("VAD_MIN_UTTERANCE_S", "0.5"))
VAD_SUSTAIN_FRAMES = int(os.environ.get("QWEN3_ASR_VAD_SUSTAIN_FRAMES", "3"))
VAD_BACKEND_ENV = os.environ.get("QWEN3_ASR_VAD_BACKEND", "webrtc").lower()
VAD_WEBRTC_AGGR = int(os.environ.get("QWEN3_ASR_VAD_WEBRTC_AGGR", "2"))
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


class ChunkConfirmASRStream:
    """Chunk-and-Confirm streaming ASR session with VAD-aligned segmentation.

    Implements the P0 recipe: each hop re-encodes the full accumulated audio
    and uses prefix-prompt to avoid re-decoding confirmed text.  VAD detects
    utterance boundaries for multi-utterance sessions.

    Compatible with the ``ASRStream`` interface expected by
    ``Qwen3ASRRKStream`` and the WebSocket ``/asr/stream`` handler.
    """

    # Regex patterns for trailing garbage detection (same as
    # StreamSession._RE_TRAILING_GARBAGE).
    _RE_TRAILING_GARBAGE = re.compile(
        r'(?:[。？！.?!…][^。？！.?!…，,；：:、\s]{1,2}'
        r'|[，,；][^。？！.?!…，,；：:、\s])$'
    )

    def __init__(self, engine, language: Optional[str] = None,
                 context: str = "",
                 vad=None,
                 use_npu_lock: bool = False,
                 npu_lock: Optional[threading.Lock] = None):
        self._engine = engine
        self._language = language
        self._context = context
        self._silero_vad = vad
        self._use_npu_lock = use_npu_lock
        self._npu_lock = npu_lock

        # ── Hop timing ──────────────────────────────────────────────────
        self._hop_samples = int(CHUNK_SIZE_SEC * SAMPLE_RATE)

        # ── Per-utterance state ─────────────────────────────────────────
        self._audio_accum = np.zeros(0, dtype=np.float32)
        self._raw_decoded: str = ""
        self._chunk_id: int = 0
        self._audio_buf = np.zeros(0, dtype=np.float32)

        # ── Multi-utterance state ───────────────────────────────────────
        self._segments: list[str] = []
        self._archive_text: str = ""  # read by server for VAD-triggered finals
        self._current_partial: str = ""
        self._current_language: str = language or ""
        self._episode_final: bool = False

        # ── VAD state (sustain-filter accumulators) ─────────────────────
        self._vad_speech_samples: int = 0
        self._vad_silence_samples: int = 0
        self._vad_consec_speech_frames: int = 0
        self._vad_pending_silence_samples: int = 0
        self._vad_was_speech: bool = False
        self._last_is_speech: Optional[bool] = None

        # ── webrtcvad backend (preferred for CJK) ───────────────────────
        self._webrtc_vad = None
        self._webrtc_frame_samples = int(
            VAD_WEBRTC_FRAME_MS * SAMPLE_RATE / 1000)
        self._webrtc_carry = np.zeros(0, dtype=np.float32)
        if VAD_BACKEND_ENV in ("webrtc", "auto"):
            try:
                import webrtcvad as _wv
                self._webrtc_vad = _wv.Vad(VAD_WEBRTC_AGGR)
                logger.info(
                    "ChunkConfirm VAD: webrtcvad aggr=%d frame=%dms",
                    VAD_WEBRTC_AGGR, VAD_WEBRTC_FRAME_MS)
            except Exception as exc:
                if VAD_BACKEND_ENV == "webrtc":
                    logger.warning(
                        "webrtcvad unavailable (%s); falling back to silero", exc)
                self._webrtc_vad = None
        if self._webrtc_vad is None and self._silero_vad is not None:
            logger.info("ChunkConfirm VAD: silero (fallback)")

        # ── Finalize gate ───────────────────────────────────────────────
        self._finalizing: bool = False
        self._final_decode_in_progress: bool = False

        # ── Stats ───────────────────────────────────────────────────────
        self._n_hops: int = 0
        self._total_audio_s: float = 0.0   # total fed audio duration (not re-encoded)
        self._total_enc_ms: float = 0.0
        self._total_dec_ms: float = 0.0

        # Warn if encoder capacity may be insufficient (recipe §4.2).
        encoder_max = getattr(engine.encoder, 'max_seconds', 4.0)
        if AUTO_SEGMENT_CAP_SEC > encoder_max * 0.9:
            logger.warning(
                "AUTO_SEGMENT_CAP_SEC=%.1fs but encoder max=%.1fs — "
                "audio beyond %.1fs will be truncated. "
                "Load larger encoder models (e.g. 30s) for full coverage.",
                AUTO_SEGMENT_CAP_SEC, encoder_max, encoder_max)

    # ── Public API (ASRStream-compatible) ─────────────────────────────────

    def feed_audio(self, pcm16k: np.ndarray) -> dict:
        """Accept a chunk of 16 kHz float32 PCM and process complete hops."""
        x = np.asarray(pcm16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)

        self._total_audio_s += len(x) / SAMPLE_RATE

        # New utterance after VAD-triggered final → reset on incoming speech.
        if self._episode_final:
            self._maybe_resume_new_utterance(x)

        # Run VAD on incoming samples.
        self._update_vad(x)

        # During speech: accumulate to audio buffer.
        # During silence: VAD endpoint check may finalize the utterance.
        is_speech = self._last_is_speech

        if is_speech:
            self._audio_buf = np.concatenate([self._audio_buf, x])
        elif self._vad_was_speech and not is_speech:
            # Speech→silence transition: append last speech frame, then check.
            self._audio_buf = np.concatenate([self._audio_buf, x])
            if self._check_vad_endpoint() and not self._episode_final:
                self._finalize_utterance()
        else:
            # Pure silence: skip (no audio accumulation).
            pass

        # If finalizing, suppress further processing.
        if self._episode_final or self._finalizing:
            return {
                "language": self._current_language,
                "text": self._composed_text(),
                "is_final": self._episode_final,
                "is_speech": bool(is_speech),
                "chunks_processed": self._n_hops,
            }

        # Process complete hops from accumulated speech audio.
        hops = 0
        while len(self._audio_buf) >= self._hop_samples:
            chunk = self._audio_buf[:self._hop_samples]
            self._audio_buf = self._audio_buf[self._hop_samples:]
            self._audio_accum = np.concatenate([self._audio_accum, chunk])
            self._run_hop()
            hops += 1

        # Check VAD endpoint after processing hops (may have changed state).
        if (self._vad_speech_samples > 0
                and self._check_vad_endpoint()
                and not self._episode_final):
            self._finalize_utterance()

        # Trim _audio_buf periodically.
        max_prefix = self._hop_samples * 3
        if len(self._audio_buf) > max_prefix * 2:
            self._audio_buf = self._audio_buf[-max_prefix:]

        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._episode_final,
            "is_speech": bool(is_speech),
            "chunks_processed": self._n_hops,
        }

    def get_result(self) -> dict:
        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._episode_final,
            "is_speech": bool(self._last_is_speech),
        }

    def prepare_finalize(self) -> None:
        """Drain residual sub-hop audio into encoder before finalize."""
        logger.debug(
            "ChunkConfirm prepare_finalize: speech=%.2fs silence=%.0fms "
            "episode_final=%s",
            self._vad_speech_samples / SAMPLE_RATE,
            self._vad_silence_samples * 1000 / SAMPLE_RATE,
            self._episode_final,
        )
        self._finalizing = True
        if not self._final_decode_in_progress:
            try:
                self._engine.decoder.abort()
            except Exception:
                pass
        if self._episode_final:
            return
        # Flush remaining buffer as tail of current utterance.
        if len(self._audio_buf) > 0:
            tail = self._audio_buf
            self._audio_buf = np.zeros(0, dtype=np.float32)
            self._audio_accum = np.concatenate([self._audio_accum, tail])
            if len(self._audio_accum) > 0:
                self._run_hop(is_final=True)
                self._finalize_utterance()

    def cancel_and_finalize(self) -> None:
        """Hard-cancel pending partials and drop residual sub-hop audio.

        Caller should follow with ``finish()`` to obtain the final result.
        """
        self._finalizing = True
        if not self._final_decode_in_progress:
            try:
                self._engine.decoder.abort()
            except Exception:
                pass
        if self._episode_final:
            return
        # Drop residual tail — we already have hop-aligned audio in accum.
        self._audio_buf = np.zeros(0, dtype=np.float32)
        # Finalize current utterance if any audio accumulated.
        if len(self._audio_accum) > 0:
            self._run_hop(is_final=True)
            self._finalize_utterance()
        else:
            self._episode_final = True

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
            # Flush residual and finalize current utterance.
            if len(self._audio_buf) > 0:
                self._audio_accum = np.concatenate(
                    [self._audio_accum, self._audio_buf])
                self._audio_buf = np.zeros(0, dtype=np.float32)
            if len(self._audio_accum) > 0:
                self._run_hop(is_final=True)
                self._finalize_utterance()
            elif self._current_partial:
                # No audio but have partial text from earlier hops.
                self._segments.append(self._current_partial)
            self._episode_final = True

        finalize_ms = (time.perf_counter() - t0) * 1000

        # Merge segments.
        if self._segments:
            text = "".join(self._segments)
        elif self._current_partial:
            text = self._current_partial
        else:
            text = ""

        if apply_itn_flag:
            text = apply_itn(text)

        logger.info(
            "ChunkConfirm finish: %d hops, %.2fs audio, "
            "enc=%.0fms dec=%.0fms finalize=%.0fms text=%r",
            self._n_hops, self._total_audio_s,
            self._total_enc_ms, self._total_dec_ms, finalize_ms,
            text[:60],
        )

        rtf = ((self._total_enc_ms + self._total_dec_ms) / 1000.0
               / max(self._total_audio_s, 0.1))
        return {
            "language": self._current_language,
            "text": text,
            "is_final": True,
            "final_mode": "chunk_confirm",
            "fallback": None,
            "finalize_ms": finalize_ms,
            "stats": {
                "total_enc_ms": self._total_enc_ms,
                "total_llm_ms": self._total_dec_ms,
                "total_audio_s": self._total_audio_s,
                "total_chunks": self._n_hops,
                "rtf": rtf,
                "segments": len(self._segments),
            },
        }

    # ── Internal: chunk-and-confirm hop ───────────────────────────────────

    def _run_hop(self, is_final: bool = False) -> None:
        """Run one chunk-and-confirm hop on the full accumulated utterance audio.

        Mirrors official ``qwen3_asr.py:731-763``.
        """
        if len(self._audio_accum) == 0:
            return

        # 1. Encode FULL accumulated utterance audio.
        t0 = time.perf_counter()
        enc_result = self._engine.encoder.encode(self._audio_accum)
        if isinstance(enc_result, tuple):
            audio_embd = enc_result[0]
        else:
            audio_embd = enc_result
        enc_ms = (time.perf_counter() - t0) * 1000
        self._total_enc_ms += enc_ms

        # 2. Compute prefix with UTF-8 guarded rollback.
        prefix = self._compute_prefix()

        # 3. Build embed with prefix + decode.
        full_embd, n_tokens = self._engine.build_embed(
            audio_embd, prefix_text=prefix,
            language=self._language, context=self._context,
            skip_prefix=False)

        early_stop = 0 if is_final else MAX_DECODE_TOKENS
        result = self._decode(full_embd, n_tokens, early_stop)
        self._n_hops += 1

        # 4. Parse output.  Decoder generates continuation only (prefix is
        #    prefill context, not generated tokens).  Append to prefix for
        #    the full decoded text.  Mirrors qwen3_asr.py:754-761.
        raw = result.get("text", "") or ""
        was_aborted = result.get("aborted", False)

        if self._language:
            new_text = raw
        else:
            lang, new_text = parse_asr_output(raw)
            if lang:
                self._current_language = lang

        if was_aborted and not is_final:
            # Repetition abort on intermediate hop: keep prefix, discard new.
            new_text = ""

        self._raw_decoded = (prefix + new_text) if prefix else new_text

        # Strip trailing garbage from raw_decoded (same heuristic as
        # StreamSession._strip_trailing_garbage).
        self._raw_decoded = self._strip_trailing(self._raw_decoded)

        # Update partial for get_result().
        self._current_partial = self._raw_decoded
        self._chunk_id += 1

        if os.environ.get("QWEN3_ASR_DEBUG_CC", "0") == "1":
            logger.info(
                "  [hop %d] accum=%.1fs enc=%.0fms dec=%.0fms "
                "prefix=%r new=%r",
                self._chunk_id, len(self._audio_accum) / SAMPLE_RATE,
                enc_ms,
                (time.perf_counter() - t0) * 1000 - enc_ms,
                prefix[:30], new_text[:40])

    def _compute_prefix(self) -> str:
        """Compute UTF-8-safe prefix from previous decode.

        Mirrors official ``qwen3_asr.py:736-746`` exactly.
        """
        if self._chunk_id < UNFIXED_CHUNK_NUM:
            return ""
        if not self._raw_decoded:
            return ""

        cur_ids = self._engine.tokenizer.encode(self._raw_decoded).ids
        if len(cur_ids) <= UNFIXED_TOKEN_NUM:
            return ""

        k = UNFIXED_TOKEN_NUM
        while True:
            end = max(0, len(cur_ids) - k)
            prefix = (
                self._engine.tokenizer.decode(cur_ids[:end])
                if end > 0 else "")
            if '�' not in prefix:  # UTF-8 guard
                return prefix
            if end == 0:
                return ""
            k += 1

    def _decode(self, full_embd: np.ndarray, n_tokens: int,
                early_stop: int) -> dict:
        """Run decoder with NPU locking (same pattern as streaming.py:603-615)."""
        decoder = self._engine.decoder
        decoder._early_stop_tokens = early_stop
        t0 = time.perf_counter()
        try:
            if self._use_npu_lock and self._npu_lock is not None:
                with self._npu_lock:
                    result = decoder.run_embed(
                        full_embd, n_tokens, keep_history=0)
            else:
                result = decoder.run_embed(
                    full_embd, n_tokens, keep_history=0)
        finally:
            decoder._early_stop_tokens = 0
        self._total_dec_ms += (time.perf_counter() - t0) * 1000
        return result

    # ── Internal: utterance segmentation ──────────────────────────────────

    def _finalize_utterance(self) -> None:
        """Save current utterance text to segments and reset per-utterance state.

        Sets ``_archive_text`` (concatenation of all completed segments) and
        ``_episode_final = True`` so the WebSocket server can emit a
        ``session_complete=False`` frame for this utterance boundary.
        """
        text = self._current_partial or self._raw_decoded
        if text:
            self._segments.append(text)
        self._archive_text = "".join(self._segments) if self._segments else ""
        self._episode_final = True  # signal server: utterance boundary ready
        logger.debug("ChunkConfirm utterance final: %r (total segments: %d)",
                     text[:60] if text else "", len(self._segments))
        # Reset per-utterance state for next utterance.
        self._audio_accum = np.zeros(0, dtype=np.float32)
        self._raw_decoded = ""
        self._chunk_id = 0
        self._current_partial = ""
        # Reset VAD accumulators for next utterance.
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._vad_consec_speech_frames = 0
        self._vad_pending_silence_samples = 0

    def _maybe_resume_new_utterance(self, samples: np.ndarray) -> None:
        """If a new utterance starts after an endpoint, reset state."""
        if len(samples) == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms <= 1e-3:
            return
        # Full reset for new utterance.
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._audio_accum = np.zeros(0, dtype=np.float32)
        self._raw_decoded = ""
        self._chunk_id = 0
        self._current_partial = ""
        # Do NOT reset _segments or _archive_text here — they accumulate
        # across utterances.  Only reset the episode flag so the server
        # knows we've moved past the previous utterance boundary.
        self._episode_final = False
        self._finalizing = False
        self._vad_speech_samples = 0
        self._vad_silence_samples = 0
        self._vad_consec_speech_frames = 0
        self._vad_pending_silence_samples = 0
        self._webrtc_carry = np.zeros(0, dtype=np.float32)
        if self._silero_vad is not None:
            try:
                self._silero_vad.reset()
            except Exception:
                pass

    # ── Internal: VAD ─────────────────────────────────────────────────────
    # Sustain-filter logic copied from streaming.py:465-506.

    def _update_vad(self, samples: np.ndarray) -> None:
        """Run VAD on incoming samples and update speech/silence accumulators."""
        if self._webrtc_vad is not None:
            self._update_vad_webrtc(samples)
            return
        if self._silero_vad is None:
            # No VAD: treat all audio as speech.
            self._vad_speech_samples += len(samples)
            self._vad_silence_samples = 0
            self._last_is_speech = True
            return

        self._silero_vad.feed(samples)
        is_speech = self._silero_vad.is_speech
        n = len(samples)
        self._vad_was_speech = self._last_is_speech or False

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

        # Drain segmenter output.
        while self._silero_vad.has_speech():
            self._silero_vad.pop_speech()

    def _update_vad_webrtc(self, samples: np.ndarray) -> None:
        """webrtcvad-based accumulator. 20 ms int16 frames with sustain filter."""
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
        fb = frame_len * 2

        self._vad_was_speech = self._last_is_speech or False

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
        """True when sufficient speech followed by sufficient silence."""
        if self._episode_final:
            return False
        min_speech = int(VAD_MIN_UTTERANCE_S * SAMPLE_RATE)
        min_silence = int(VAD_ENDPOINT_SILENCE_MS * SAMPLE_RATE / 1000)
        return (
            self._vad_speech_samples >= min_speech
            and self._vad_silence_samples >= min_silence
        )

    # ── Internal: helpers ─────────────────────────────────────────────────

    def _composed_text(self) -> str:
        """Merge completed segments with current partial for display."""
        if self._episode_final:
            return "".join(self._segments) if self._segments else ""
        base = "".join(self._segments) if self._segments else ""
        if not self._current_partial:
            return base
        if not base:
            return self._current_partial
        sep = "" if (base and _is_cjk(base[-1])) else " "
        return (base + sep + self._current_partial).strip()

    @classmethod
    def _strip_trailing(cls, text: str) -> str:
        """Remove trailing garbage characters after sentence-ending punctuation.

        Same heuristic as ``StreamSession._strip_trailing_garbage``.
        Only applied for short utterances (≤200 chars).
        """
        if not text or len(text) < 2:
            return text
        if len(text) > 200:
            return text

        SENT_END = '。？！.?!…'

        prev = None
        while text != prev and len(text) >= 2:
            prev = text
            text = text.rstrip(SENT_END)
            if text == prev:
                m = cls._RE_TRAILING_GARBAGE.search(text)
                if m:
                    text = text[:m.start()]
        return text

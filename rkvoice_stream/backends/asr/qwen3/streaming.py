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

import ast
import hashlib
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


# ── True-streaming parameters ─────────────────────────────────────────────
ENCODER_HOP_SAMPLES = 1280                  # mel hop 160 × conv stride 8
ENCODER_FRAMES_PER_SEC = 13                 # encoder output rate (~13 fps)

# How many text units (English words/punctuation, CJK characters) of the
# already-committed text may be re-matched against the start of the next
# window when stitching window commits together.  The acoustic overlap is
# ~1 s, i.e. ≲4 English words or ≲8 CJK characters; 16 leaves headroom for a
# fast speaker.  The match is anchored to the very end of the committed text
# and the very start of the new window and the longest match wins, so a larger
# cap admits no false match that a smaller one would not.  Not an env knob: it
# is a property of the overlap window.  With no acoustic overlap nothing is
# decoded twice, so nothing is de-duplicated (see ``_window_dedup_units``).
WINDOW_COMMIT_OVERLAP_UNITS = 16

# The window after a commit starts on a re-decode of the overlap, but not always
# cleanly: the cut can clip a word, and the decoder sometimes opens with a stray
# token before it locks on (measured through the service on RK3588: a window
# came back as "Assistant: Assistant for the factory floor, ..." after one that
# ended "... a voice assistant for the factory"). Up to this many leading units
# of the new window may be discarded to find where it rejoins the committed
# text; a match found that way has to be at least two words long.
WINDOW_SEAM_MAX_SKIP_UNITS = 3

# A pause is a run of at least this many encoder frames (80 ms each) whose level
# stays under QWEN3_ASR_TRUE_ROLL_PAUSE_RATIO of the window's loud frames. Two
# frames is a breath or a comma; gaps between words in running speech are shorter.
WINDOW_PAUSE_MIN_FRAMES = 2

# A window commit that came back aborted is retried on the following chunks
# rather than committed; the buffer may run this many chunks past its cap before
# the oldest frames are dropped after all.
WINDOW_COMMIT_RETRY_CHUNKS = 2

# Characters ignored when matching text units across a window boundary, and
# the sentence terminators stripped from a mid-utterance commit.
_MATCH_STRIP_CHARS = ".,!?;:'\"。，！？；：、…—-"
_MIDSTREAM_TERMINATORS = ".。!！?？…"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float env %s=%r; using default=%s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer env %s=%r; using default=%d", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _array_fingerprint(arr: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(arr)
    return {
        "shape": tuple(int(v) for v in contiguous.shape),
        "dtype": str(contiguous.dtype),
        "mean": float(np.mean(contiguous)) if contiguous.size else 0.0,
        "std": float(np.std(contiguous)) if contiguous.size else 0.0,
        "sha1": hashlib.sha1(contiguous.view(np.uint8)).hexdigest()[:16],
    }


def _normalize_decoder_text(value) -> tuple[str, Optional[str]]:
    """Return ``(text, language)`` from decoder text-like payloads."""
    if isinstance(value, (tuple, list)):
        text = value[0] if value else ""
        lang = value[1] if len(value) > 1 else None
        return str(text or ""), str(lang) if lang else None
    if isinstance(value, str) and value[:1] in ("(", "["):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if (
            isinstance(parsed, (tuple, list))
            and 1 <= len(parsed) <= 2
            and isinstance(parsed[0], str)
        ):
            lang = parsed[1] if len(parsed) > 1 else None
            return parsed[0], str(lang) if isinstance(lang, str) and lang else None
    return str(value or ""), None


def _is_cjk(ch: str) -> bool:
    if not ch:
        return False
    o = ord(ch)
    return (
        0x3400 <= o <= 0x9FFF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2FFFF
    )


_RE_PROMPT_LEAK_SUFFIX = re.compile(
    r"(?:\s*(?:转录|轉錄|請轉錄|请转录|transcribe|transcript|transkribieren)\s*[:：。.!?]*)+$",
    re.IGNORECASE,
)
_RE_PROMPT_LEAK_PREFIX = re.compile(
    r"^\s*(?:you are a helpful assistant\s*[\.\n]+)+",
    re.IGNORECASE,
)


def _strip_midstream_terminator(text: str) -> str:
    """Drop one trailing sentence terminator from a mid-utterance commit.

    A window commit cuts the utterance at an arbitrary acoustic point, and the
    decoder (``final_stop_on_punctuation``) happily ends its output with a full
    stop there.  Keeping it would (a) read as a sentence boundary that the
    speaker never made — "…the product. Such as…" — and (b) break overlap
    dedup, because the next window decodes the same audio *without* that
    invented terminator, so the unit sequences no longer line up.  The cost of
    stripping is one piece of punctuation in the rare case the cut really did
    land on a sentence end; the cost of keeping it is a duplicated word plus a
    false boundary, so we strip.
    """
    stripped = (text or "").rstrip()
    if stripped and stripped[-1] in _MIDSTREAM_TERMINATORS:
        return stripped[:-1].rstrip()
    return stripped


def _strip_prompt_leaks(text: str) -> str:
    """Remove short prompt fragments that RKLLM may emit after weak EOS."""
    if not text:
        return text
    cleaned = _RE_PROMPT_LEAK_PREFIX.sub("", str(text).strip())
    cleaned = _RE_PROMPT_LEAK_SUFFIX.sub("", cleaned).strip()
    return cleaned


class Qwen3TrueStreamingASRStream:
    """True streaming session: 400 ms chunks + left-context + VAD endpoint."""

    def __init__(self, engine, language: Optional[str] = None,
                 context: str = "",
                 vad=None,
                 use_npu_lock: bool = False,
                 npu_lock: Optional[threading.Lock] = None,
                 vad_endpoint_silence_ms: Optional[int] = None,
                 vad_min_utterance_s: Optional[float] = None,
                 vad_min_audio_s: Optional[float] = None):
        self._engine = engine
        self._language = language
        self._context = context
        self._vad = vad
        self._use_npu_lock = use_npu_lock
        self._npu_lock = npu_lock

        self._chunk_size_sec = _env_float("QWEN3_ASR_TRUE_CHUNK_SEC", 0.4)
        self._left_context_sec = _env_float("QWEN3_ASR_TRUE_LCTX_SEC", 1.0)
        self._rolling_buffer_sec = _env_float("QWEN3_ASR_TRUE_ROLL_SEC", 5.0)
        # Acoustic overlap carried into the next window after a window commit.
        # Without it the cut can fall inside a word and that word is lost from
        # both sides.  0 disables overlap (commit exactly at the boundary).
        self._roll_overlap_sec = max(
            0.0, _env_float("QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC", 1.0)
        )
        # Where to retire a full window. Carrying an acoustic overlap into the
        # next window means that second of audio is decoded twice and the two
        # texts have to be reconciled afterwards, and a decoder that starts
        # mid-phrase does not always start cleanly (through the service on
        # RK3588 a window opened with a stray "Assistant:"; the 2026-06 segment
        # audio-carry experiment raised English long WER 9.64% -> 15.89% and
        # leaked a prompt the same way). So the window is cut at a pause inside
        # its last CUT_SEARCH_SEC instead, the next one starts right after it,
        # and nothing is decoded twice. Only when no pause is found does the
        # overlap-and-reconcile path run. 0 turns the pause search off.
        self._cut_search_sec = max(
            0.0, _env_float("QWEN3_ASR_TRUE_ROLL_CUT_SEARCH_SEC", 2.0))
        self._pause_ratio = max(
            0.0, _env_float("QWEN3_ASR_TRUE_ROLL_PAUSE_RATIO", 0.15))
        self._partial_max_tokens = _env_int("QWEN3_ASR_TRUE_PARTIAL_TOKENS", 12)
        # Minimum wall-clock interval between partial decodes (ms). On RK each
        # partial costs ~600-900 ms of NPU time; running one per 400 ms chunk
        # backs the executor up by ~2x and balloons stop->final latency.
        self._partial_min_interval_ms = _env_int(
            "QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS", 900)
        # How many chunks to skip at the start before running the first partial.
        # Earliest possible partial = (warmup+1) chunks of audio.
        self._partial_warmup_chunks = _env_int(
            "QWEN3_ASR_TRUE_PARTIAL_WARMUP", 1)
        # When backend VAD detects an endpoint, expose the endpoint immediately
        # and run the final RKLLM decode in a short-lived background thread.
        # ``finish()`` joins that thread, so final text semantics stay the same,
        # while dialogue code can react to endpoint before final text is ready.
        self._vad_final_async = os.environ.get(
            "QWEN3_ASR_VAD_FINAL_ASYNC", "0").lower() not in (
                "0", "false", "no", "off"
            )
        self._debug_final_input = _env_bool("QWEN3_ASR_DEBUG_FINAL_INPUT", False)
        self._allow_auto_resume_after_endpoint = _env_bool(
            "QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT", False
        )
        # Dictation/long-transcription mode: when a VAD endpoint fires inside a
        # single stream, keep the finalized segment and append later segments
        # after auto-resume.  V2V keeps this disabled because one stream maps to
        # one dialogue turn.
        self._accumulate_segments = _env_bool(
            "QWEN3_ASR_ACCUMULATE_SEGMENTS", False
        )
        self._segment_context_prefix = _env_bool(
            "QWEN3_ASR_SEGMENT_CONTEXT_PREFIX", False
        )
        self._segment_audio_carry_sec = max(
            0.0, _env_float("QWEN3_ASR_SEGMENT_AUDIO_CARRY_SEC", 0.0)
        )
        self._segment_text_overlap_tokens = max(
            0, _env_int("QWEN3_ASR_SEGMENT_TEXT_OVERLAP_TOKENS", 0)
        )

        self._vad_endpoint_silence_ms = (
            max(0, int(vad_endpoint_silence_ms))
            if vad_endpoint_silence_ms is not None
            else _env_int("VAD_ENDPOINT_SILENCE_MS", 400)
        )
        self._vad_min_utterance_s = (
            max(0.0, float(vad_min_utterance_s))
            if vad_min_utterance_s is not None
            else _env_float("VAD_MIN_UTTERANCE_S", 0.5)
        )
        self._vad_min_audio_s = (
            max(0.0, float(vad_min_audio_s))
            if vad_min_audio_s is not None
            else _env_float("QWEN3_ASR_VAD_MIN_AUDIO_S", 0.0)
        )
        # Silero ``is_speech`` flickers True for ~1-2 frames during dense Chinese
        # inter-character pauses.  Require N consecutive speech frames before
        # considering the speaker active again; isolated True flips count as
        # silence and are absorbed by the accumulator.
        self._vad_sustain_frames = _env_int("QWEN3_ASR_VAD_SUSTAIN_FRAMES", 3)
        # Energy gate for accepting a new utterance after an endpoint fired.
        # The previous hardcoded 1e-3 assumed a close, hot mic; browser/AGC
        # feeds with low input gain can sit below it on soft onsets, which
        # presents as "the first syllables never arrive". Lower via env for
        # such clients; raise on noisy far-field setups.
        self._vad_resume_rms = _env_float("QWEN3_ASR_VAD_RESUME_RMS", 1e-3)
        # Backend selector for endpointing. "auto" prefers webrtcvad if importable.
        self._vad_backend_env = os.environ.get("QWEN3_ASR_VAD_BACKEND", "webrtc").lower()
        self._vad_webrtc_aggr = _env_int("QWEN3_ASR_VAD_WEBRTC_AGGR", 2)
        self._vad_webrtc_frame_ms = _env_int("QWEN3_ASR_VAD_WEBRTC_FRAME_MS", 20)

        self._chunk_samples = int(self._chunk_size_sec * SAMPLE_RATE)
        self._left_context_samples = int(self._left_context_sec * SAMPLE_RATE)

        # Audio buffers
        self._audio_buf = np.zeros(0, dtype=np.float32)
        self._processed_samples = 0
        self._utterance_audio_buffer: list[np.ndarray] = []

        # Rolling encoder-output buffer (frames for decoder prefill).
        # Each entry is (T_frames, 1024).
        self._encoder_frames: list[np.ndarray] = []
        # Per-frame RMS of the audio behind each encoder frame, kept in step
        # with ``_encoder_frames`` entry by entry; the pause search reads it.
        self._frame_rms: list[np.ndarray] = []
        self._total_encoder_frames = 0
        self._max_encoder_frames = int(
            self._rolling_buffer_sec * ENCODER_FRAMES_PER_SEC)
        self._roll_overlap_frames = min(
            int(self._roll_overlap_sec * ENCODER_FRAMES_PER_SEC),
            max(0, self._max_encoder_frames - 1),
        )

        # Output state
        self._archive_text: str = ""
        self._partial_text: str = ""
        # Text of windows already decoded and retired from the rolling buffer
        # within the *current* utterance.  Distinct from ``_archive_text``,
        # which means "finalized segments" and is cleared/overwritten by
        # ``_commit_final_text`` whenever ``_accumulate_segments`` is off —
        # window commits must survive in that (default) configuration too.
        self._window_committed_text: str = ""
        self._window_commits: int = 0
        self._cut_search_frames = int(self._cut_search_sec * ENCODER_FRAMES_PER_SEC)
        # Whether the seam between the committed text and the current window
        # carries an acoustic overlap (and so needs its text reconciled).
        self._seam_overlap: bool = False
        self._pause_cuts: int = 0
        self._window_commit_in_progress: bool = False
        self._window_commit_retries: int = 0
        # Nothing is decoded twice without an acoustic overlap, so an equal
        # word on both sides of the seam is the speaker repeating it.
        self._window_dedup_units: int = (
            WINDOW_COMMIT_OVERLAP_UNITS if self._roll_overlap_frames > 0 else 0)
        self._last_final_abort_reason: str = ""
        self._current_language: str = language or ""
        self._episode_final: bool = False
        self._vad_endpoint_detected: bool = False

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

        # VAD-triggered final decode can run in a background thread to avoid
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
            self._vad_webrtc_frame_ms * SAMPLE_RATE / 1000)
        self._webrtc_carry = np.zeros(0, dtype=np.float32)
        if self._vad_backend_env in ("webrtc", "auto"):
            try:
                import webrtcvad as _wv  # type: ignore
                self._webrtc_vad = _wv.Vad(self._vad_webrtc_aggr)
                logger.info(
                    "Qwen3 streaming VAD backend: webrtcvad "
                    "(aggr=%d frame=%dms)", self._vad_webrtc_aggr,
                    self._vad_webrtc_frame_ms)
            except Exception as exc:
                if self._vad_backend_env == "webrtc":
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

        # New utterance after VAD-triggered final is opt-in. V2V creates a
        # fresh ASRStream per turn; auto-resetting inside the old stream lets
        # trailing non-silence become a spurious second final.
        if self._episode_final:
            if self._allow_auto_resume_after_endpoint:
                self._maybe_resume_new_utterance(x)
            if self._episode_final:
                return {
                    "language": self._current_language,
                    "text": self._composed_text(),
                    "is_final": True,
                    "is_speech": False,
                    "chunks_processed": self._n_chunks,
                }

        self._audio_buf = np.concatenate([self._audio_buf, x])
        self._utterance_audio_buffer.append(x.copy())

        # Once the episode is final (VAD pre-fired) or we're in the finalize
        # window, drop residual audio without encoding — saves ~200-600 ms
        # of stop→final latency that would otherwise be spent encoding the
        # silence trailer or any post-EOU residual.
        if self._episode_final or self._vad_endpoint_detected or self._finalizing:
            self._processed_samples = len(self._audio_buf)
            return {
                "language": self._current_language,
                "text": self._composed_text(),
                "is_final": self._vad_endpoint_detected or self._episode_final,
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
            self._start_vad_final_decode()

        # Trim _audio_buf periodically (keep 2× left_context headroom).
        max_prefix = self._left_context_samples + self._chunk_samples
        if self._processed_samples > max_prefix * 2:
            trim = self._processed_samples - max_prefix
            self._audio_buf = self._audio_buf[trim:]
            self._processed_samples -= trim

        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._vad_endpoint_detected or self._episode_final,
            "is_speech": True,
            "chunks_processed": self._n_chunks,
        }

    def get_result(self) -> dict:
        return {
            "language": self._current_language,
            "text": self._composed_text(),
            "is_final": self._vad_endpoint_detected or self._episode_final,
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
        self._join_final_decode_thread()
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
                self._append_frames(enc_out, new_audio)
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
        self._join_final_decode_thread()
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

    def abort_partial_decode(self) -> None:
        """Abort the current partial decode without entering finalizing mode.

        The V2V server uses this when client EOU arrives while the receive loop
        still has queued audio to feed.  ``cancel_and_finalize()`` would set
        ``_finalizing`` and make later ``feed_audio()`` calls drop that queued
        audio; this hook only interrupts stale partial work so the ingest queue
        can drain before the real finalize marker is processed.
        """
        if self._final_decode_in_progress or self._window_commit_in_progress:
            return
        try:
            self._engine.decoder.abort()
        except Exception:
            pass

    def finish(self, apply_itn_flag: bool = True) -> dict:
        """Finalize and return dict matching StreamSession.finish() schema."""
        self._finalizing = True
        self._join_final_decode_thread()
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
                self._commit_final_text(text)
            elif self._window_committed_text:
                # Buffer already retired into window commits — keep that text.
                self._commit_final_text("")
            else:
                # No audio buffered → empty result.
                if not self._accumulate_segments:
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
            if self._vad_consec_speech_frames >= self._vad_sustain_frames:
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
                if self._vad_consec_speech_frames >= self._vad_sustain_frames:
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
        min_speech = int(self._vad_min_utterance_s * SAMPLE_RATE)
        min_audio = int(self._vad_min_audio_s * SAMPLE_RATE)
        min_silence = int(self._vad_endpoint_silence_ms * SAMPLE_RATE / 1000)
        return (
            len(self._audio_buf) >= min_audio
            and self._vad_speech_samples >= min_speech
            and self._vad_silence_samples >= min_silence
        )

    def _maybe_resume_new_utterance(self, samples: np.ndarray) -> None:
        """If a new utterance starts after an endpoint, reset state."""
        # Lightweight energy gate (avoid false-resets on pure silence between
        # utterances). VAD state from the previous utterance is irrelevant here.
        if len(samples) == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        if rms <= self._vad_resume_rms:
            return
        # Full reset for a new utterance.  In dictation mode we can carry a
        # short audio tail from the previous segment so the next final decode
        # has acoustic context around the VAD boundary.
        carry = np.zeros(0, dtype=np.float32)
        if self._accumulate_segments and self._segment_audio_carry_sec > 0:
            carry_n = int(self._segment_audio_carry_sec * SAMPLE_RATE)
            if carry_n > 0 and len(self._audio_buf) > 0:
                carry = self._audio_buf[-carry_n:].copy()
        self._audio_buf = carry
        self._processed_samples = len(carry)
        self._utterance_audio_buffer = []
        self._encoder_frames = []
        self._frame_rms = []
        self._total_encoder_frames = 0
        self._seam_overlap = False
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
        self._window_commit_retries = 0
        # Window commits belong to the utterance that just ended; they were
        # already folded into ``_archive_text`` by ``_commit_final_text``.
        # Carrying them over would prepend the old sentence to the new one.
        self._window_committed_text = ""
        if not self._accumulate_segments:
            self._archive_text = ""
        self._episode_final = False
        self._vad_endpoint_detected = False
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
        self._append_frames(enc_out, new_audio)
        if self._total_encoder_frames > self._max_encoder_frames:
            committed = self._commit_window_overflow()
            if committed:
                self._window_commit_retries = 0
            elif (self._window_commit_allowed()
                    and self._window_commit_retries < WINDOW_COMMIT_RETRY_CHUNKS):
                # The decode ran but came back aborted (external abort, repeat
                # collapse, timeout): its text is not the window's text, so it
                # was not committed and the frames were kept.  Try again with
                # the next chunk before giving anything up.
                self._window_commit_retries += 1
            else:
                # A final decode owns the decoder / the episode is closing (it
                # will decode these frames itself), or the retries ran out.
                # Drop the oldest frames so the buffer keeps a bound -- loudly,
                # because their text is gone.
                dropped_frames = 0
                while (self._total_encoder_frames > self._max_encoder_frames
                       and len(self._encoder_frames) > 1):
                    dropped = self._encoder_frames.pop(0)
                    self._frame_rms.pop(0)
                    self._total_encoder_frames -= dropped.shape[0]
                    dropped_frames += dropped.shape[0]
                if dropped_frames:
                    logger.warning(
                        "Qwen3-true-stream dropped %d encoder frames without "
                        "committing their text (retries=%d finalizing=%s "
                        "episode_final=%s)",
                        dropped_frames, self._window_commit_retries,
                        self._finalizing, self._episode_final)
                self._window_commit_retries = 0

        if self._episode_final or not self._encoder_frames or self._finalizing:
            return

        if os.environ.get("QWEN3_ASR_STREAM_PARTIAL", "1").lower() in ("0", "false", "no"):
            return

        # Throttle partials: skip warmup chunks + min interval since last.
        if self._n_chunks <= self._partial_warmup_chunks:
            return
        now = time.monotonic() * 1000
        if (self._last_partial_ts > 0
                and now - self._last_partial_ts < self._partial_min_interval_ms):
            return

        # Partial decode.
        all_frames = np.concatenate(self._encoder_frames, axis=0)
        t0 = time.perf_counter()
        text = self._decode_partial(all_frames)
        self._total_dec_ms += (time.perf_counter() - t0) * 1000
        self._last_partial_ts = time.monotonic() * 1000
        if text:
            self._partial_text = text.strip()

    # ── Internal: window commit ───────────────────────────────────────

    def _window_commit_allowed(self) -> bool:
        """Whether it is safe to run a decode from the ``feed_audio`` thread.

        The RKLLM decoder is a single, non-reentrant resource (see
        ``_run_decoder``): a second ``run_embed`` while a final decode is in
        flight would corrupt both.  The VAD-triggered final decode may run on
        its own thread (``QWEN3_ASR_VAD_FINAL_ASYNC=1``), so check for it
        explicitly.  Once the episode is closing there is also nothing to gain
        — ``finish()`` decodes the remaining window anyway.
        """
        if self._episode_final or self._finalizing:
            return False
        if self._final_decode_in_progress:
            return False
        thread = self._final_decode_thread
        if thread is not None and thread.is_alive():
            return False
        return True

    def _append_frames(self, enc_out: np.ndarray, audio: np.ndarray) -> None:
        """Add encoder frames and the level of the audio each one came from."""
        n = enc_out.shape[0]
        parts = np.array_split(np.asarray(audio, dtype=np.float32), n)
        rms = np.array([float(np.sqrt(np.mean(np.square(part)))) if len(part) else 0.0
                        for part in parts], dtype=np.float32)
        self._encoder_frames.append(enc_out)
        self._frame_rms.append(rms)
        self._total_encoder_frames += n

    def _keep_tail_frames(self, keep_frames: int) -> None:
        """Retain only the last ``keep_frames`` encoder frames."""
        if keep_frames <= 0:
            self._encoder_frames = []
            self._frame_rms = []
            self._total_encoder_frames = 0
            return
        kept: list[np.ndarray] = []
        kept_rms: list[np.ndarray] = []
        total = 0
        for arr, rms in zip(reversed(self._encoder_frames), reversed(self._frame_rms)):
            if total >= keep_frames:
                break
            need = keep_frames - total
            if arr.shape[0] > need:
                arr, rms = arr[-need:, :], rms[-need:]
            kept.append(arr)
            kept_rms.append(rms)
            total += arr.shape[0]
        kept.reverse()
        kept_rms.reverse()
        self._encoder_frames = kept
        self._frame_rms = kept_rms
        self._total_encoder_frames = total

    def _find_pause_cut(self) -> Optional[int]:
        """How many leading frames to retire so the cut lands in a pause, or None.

        Looks only at the last ``_cut_search_frames`` of the window and never
        gives back less than half of it, so a commit always makes room. The
        level that counts as quiet is relative to this window's own loud frames
        (90th percentile), so it follows the microphone gain and the room.
        """
        if self._cut_search_frames <= 0 or not self._frame_rms:
            return None
        rms = np.concatenate(self._frame_rms)
        n = rms.shape[0]
        if n != self._total_encoder_frames or n < 2 * WINDOW_PAUSE_MIN_FRAMES:
            return None
        loud = float(np.percentile(rms, 90))
        if loud <= 0.0:
            return None
        lo = max(n - self._cut_search_frames, n // 2)
        quiet = rms[lo:] < self._pause_ratio * loud
        best_start, best_len, run_start = -1, 0, None
        for i, q in enumerate(list(quiet) + [False]):
            if q and run_start is None:
                run_start = i
            elif not q and run_start is not None:
                if i - run_start > best_len:       # longest pause; earliest on ties
                    best_start, best_len = run_start, i - run_start
                run_start = None
        if best_len < WINDOW_PAUSE_MIN_FRAMES:
            return None
        # Middle of the pause: both sides keep a little silence around the words.
        return lo + best_start + (best_len + 1) // 2

    def _commit_window_overflow(self) -> bool:
        """Decode the full window before retiring it, instead of dropping it.

        The rolling buffer only holds ``QWEN3_ASR_TRUE_ROLL_SEC`` of encoder
        output.  Dropping the oldest frames on overflow lost the *text* too:
        both partial and final decode only ever see the frames still in the
        buffer, so a 9 s sentence came back as its last ~5 s.  Here we run one
        full decode (no early stop — this text is permanent, unlike a partial)
        and keep the result, then restart the window from the trailing overlap
        frames.

        Returns True when the window was committed.
        """
        if not self._window_commit_allowed() or not self._encoder_frames:
            return False

        all_frames = np.concatenate(self._encoder_frames, axis=0)
        cut = self._find_pause_cut()
        if cut is not None:
            all_frames = all_frames[:cut, :]
        t0 = time.perf_counter()
        # ``abort_partial_decode`` must not reach this decode: it exists to
        # interrupt throwaway partial work, and this text is permanent.
        self._window_commit_in_progress = True
        try:
            # Decode the WHOLE window.  The final-decode punctuation stop ends
            # generation at the first sentence terminator, which is right for a
            # one-sentence turn and wrong here: a window holding "…today. And
            # tomorrow we…" would commit the first sentence and then retire the
            # frames of the second.
            text = self._decode_final(all_frames, stop_on_punctuation=False)
        except Exception as exc:  # pragma: no cover - decoder-specific
            logger.warning("Window commit decode failed: %s", exc)
            return False
        finally:
            self._window_commit_in_progress = False
        self._total_dec_ms += (time.perf_counter() - t0) * 1000

        if self._last_final_abort_reason:
            # Aborted output is a prefix of the window (external abort,
            # timeout) or garbage (repeat collapse).  Committing it and
            # trimming the frames would lose the rest for good; keep the
            # frames and let the caller retry.
            logger.warning(
                "Window commit decode aborted (%s); keeping the window",
                self._last_final_abort_reason)
            return False

        text = _strip_prompt_leaks(text or "")
        if cut is None:
            # Arbitrary cut: a terminator here is the decoder closing a
            # sentence the speaker has not finished. At a pause it is kept --
            # that is where sentences do end.
            text = _strip_midstream_terminator(text)
        self._window_committed_text = self._join_window(
            self._window_committed_text, text)
        self._window_commits += 1

        if cut is not None:
            self._keep_tail_frames(self._total_encoder_frames - cut)
            self._seam_overlap = False
            self._pause_cuts += 1
        else:
            self._keep_tail_frames(self._roll_overlap_frames)
            self._seam_overlap = self._roll_overlap_frames > 0
        # The partial belonged to the window we just retired; its content is
        # now inside ``_window_committed_text``.  Reset the throttle as well so
        # the very next chunk doesn't immediately spend another decode on the
        # (near-empty) new window right after this ~750 ms commit.
        self._partial_text = ""
        self._last_partial_ts = time.monotonic() * 1000

        logger.info(
            "Qwen3-true-stream window commit #%d: %s, kept %d frames, text=%r",
            self._window_commits,
            "cut at a pause" if cut is not None else "no pause, overlap kept",
            self._total_encoder_frames, text[:60],
        )
        return True

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
        result = self._run_decoder(full_embd, n_tokens, self._partial_max_tokens)
        raw, decoded_language = _normalize_decoder_text(result.get("text", ""))
        was_aborted = result.get("aborted", False)
        early_stopped = True  # partial always runs with early_stop>0
        if was_aborted and not early_stopped:
            return ""

        if self._language:
            text = raw
            if decoded_language:
                self._current_language = decoded_language
        else:
            lang, text = parse_asr_output(raw)
            self._current_language = lang or decoded_language or self._current_language
        return text or ""

    def _decode_final(self, all_frames: np.ndarray,
                      stop_on_punctuation: bool = True) -> str:
        prefix_text = (
            self._archive_text
            if self._accumulate_segments and self._segment_context_prefix
            else ""
        )
        full_embd, n_tokens = self._engine.build_embed(
            all_frames, prefix_text=prefix_text,
            language=self._language, context=self._context,
            skip_prefix=False)
        if self._debug_final_input:
            source = "vad" if self._vad_endpoint_detected else "client"
            frame_fp = _array_fingerprint(all_frames)
            embed_fp = _array_fingerprint(full_embd)
            logger.info(
                "Qwen3-true-stream final input debug: source=%s chunks=%d "
                "audio=%.2fs frames=%s/%s mean=%.6g std=%.6g sha1=%s "
                "embed=%s/%s mean=%.6g std=%.6g sha1=%s input_tokens=%d "
                "vad_speech=%.2fs vad_silence=%.0fms processed=%.2fs",
                source,
                self._n_chunks,
                self._total_audio_s,
                frame_fp["shape"],
                frame_fp["dtype"],
                frame_fp["mean"],
                frame_fp["std"],
                frame_fp["sha1"],
                embed_fp["shape"],
                embed_fp["dtype"],
                embed_fp["mean"],
                embed_fp["std"],
                embed_fp["sha1"],
                n_tokens,
                self._vad_speech_samples / SAMPLE_RATE,
                self._vad_silence_samples * 1000 / SAMPLE_RATE,
                self._processed_samples / SAMPLE_RATE,
            )
        decoder = self._engine.decoder
        saved_stop = getattr(decoder, "_final_stop_on_punctuation", None)
        if not stop_on_punctuation and saved_stop is not None:
            decoder._final_stop_on_punctuation = False
        try:
            result = self._run_decoder(full_embd, n_tokens, 0)
        finally:
            if not stop_on_punctuation and saved_stop is not None:
                decoder._final_stop_on_punctuation = saved_stop
        raw, decoded_language = _normalize_decoder_text(result.get("text", ""))
        was_aborted = result.get("aborted", False)
        # The RKLLM decoder only raises ``aborted`` for the stops it decides on
        # itself (repeat, early stop, punctuation).  ``decoder.abort()`` and the
        # async timeout leave ``aborted`` False and record just the reason, so
        # either field means the generation did not run to its own end.
        abort_reason = result.get("abort_reason") or ""
        self._last_final_abort_reason = (
            abort_reason or ("aborted" if was_aborted else ""))
        perf = result.get("perf") or {}
        logger.info(
            "Qwen3-true-stream final decode perf: input_tokens=%d "
            "generated=%s aborted=%s abort_reason=%s "
            "prefill_ms=%s generate_ms=%s",
            n_tokens,
            result.get("n_tokens_generated"),
            was_aborted,
            result.get("abort_reason", ""),
            perf.get("prefill_time_ms"),
            perf.get("generate_time_ms"),
        )
        if was_aborted:
            # Repetition abort — keep the (truncated) text from decoder.
            pass
        if self._language:
            text = raw
            if decoded_language:
                self._current_language = decoded_language
        else:
            lang, text = parse_asr_output(raw)
            self._current_language = lang or decoded_language or self._current_language
        return text or ""

    def _final_decode_text(self) -> str:
        if not self._encoder_frames:
            return self._partial_text
        all_frames = np.concatenate(self._encoder_frames, axis=0)
        return self._decode_final(all_frames)

    def _do_final_decode(self) -> None:
        """VAD-triggered early final decode."""
        self._vad_endpoint_detected = True
        if not self._encoder_frames:
            if self._window_committed_text:
                # Everything decoded so far lives in the window commits.
                self._commit_final_text("")
            self._episode_final = True
            return
        self._final_decode_in_progress = True
        try:
            text = self._final_decode_text()
        finally:
            self._final_decode_in_progress = False
        self._commit_final_text(text)
        self._partial_text = ""
        self._episode_final = True
        logger.info("VAD endpoint: text=%r (silence=%.0fms speech=%.2fs)",
                    text[:60],
                    self._vad_silence_samples * 1000 / SAMPLE_RATE,
                    self._vad_speech_samples / SAMPLE_RATE)

    def _start_vad_final_decode(self) -> None:
        self._vad_endpoint_detected = True
        self._final_decode_thread_started = True
        # Drop subsequent silence/residual audio. The endpoint frame has already
        # been accepted into the encoder buffer; additional VAD trailer audio
        # only delays dialogue turn handoff.
        self._finalizing = True
        if not self._vad_final_async:
            try:
                self._do_final_decode()
            except Exception as exc:  # pragma: no cover
                logger.warning("VAD-triggered final decode failed: %s", exc)
            return
        self._final_decode_thread = threading.Thread(
            target=self._do_final_decode_safe,
            name="qwen3-vad-final-decode",
            daemon=True,
        )
        self._final_decode_thread.start()

    def _join_final_decode_thread(self) -> None:
        thread = self._final_decode_thread
        if thread is not None and thread.is_alive():
            thread.join()

    def _do_final_decode_safe(self) -> None:
        try:
            self._do_final_decode()
        except Exception as exc:  # pragma: no cover
            logger.warning("VAD-triggered final decode (thread) failed: %s", exc)

    # ── Internal: helpers ────────────────────────────────────────────

    def _text_units(self, text: str) -> list[str]:
        if any(_is_cjk(ch) for ch in text):
            return [ch for ch in re.sub(r"\s+", "", text) if ch]
        return re.findall(r"[A-Za-z0-9']+|[^\w\s]", text.lower())

    @staticmethod
    def _match_units(units: list[str]) -> list[str]:
        """Normalize units for cross-boundary comparison only.

        Case and edge punctuation differ between two decodes of the same audio
        ("product." vs "Product"), and the real final output must keep them, so
        normalization happens here and never on the text we write back.
        """
        return [u.casefold().strip(_MATCH_STRIP_CHARS) for u in units]

    def _drop_overlapping_prefix(self, left: str, right: str,
                                 max_units: Optional[int] = None,
                                 max_skip: int = 0) -> str:
        if max_units is None:
            max_units = self._segment_text_overlap_tokens
        if max_units <= 0 or not left or not right:
            return right
        # Compare words only. Two decodes of the same audio punctuate it
        # differently ("factory floor, and" / "factory floor and"), so a
        # punctuation unit in the middle of the overlap must not break the
        # match; ``right_at`` remembers where each word sits among ALL of
        # right's units, which is what the cut below is counted in.
        left_words = [u for u in self._match_units(self._text_units(left)) if u]
        right_all = self._match_units(self._text_units(right))
        right_at = [i for i, u in enumerate(right_all) if u]
        right_words = [right_all[i] for i in right_at]

        cut = 0  # number of right's units (words and punctuation) to drop
        for k in range(min(max_units, len(left_words), len(right_words)), 0, -1):
            # Longest match wins; among equals, the one that skips least.
            for skip in range(0, max_skip + 1):
                if skip and k < 2:
                    break  # one word after junk is a coincidence, not a seam
                if skip + k > len(right_words):
                    break
                if left_words[-k:] == right_words[skip:skip + k]:
                    cut = right_at[skip + k - 1] + 1
                    break
            if cut:
                break
        if not cut:
            return right

        if any(_is_cjk(ch) for ch in right):
            # ``_text_units`` counts whitespace-free characters; walk the raw
            # string to the same count so the remainder keeps its own spacing
            # ("你好 New York" must not come back as "NewYork").
            seen = 0
            for idx, ch in enumerate(right):
                if seen >= cut:
                    return right[idx:].lstrip()
                if not ch.isspace():
                    seen += 1
            return ""

        matches = list(re.finditer(r"[A-Za-z0-9']+|[^\w\s]", right))
        if cut >= len(matches):
            return ""
        return right[matches[cut].start():].lstrip()

    def _join_text(self, left: str, right: str,
                   max_units: Optional[int] = None,
                   max_skip: int = 0) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        right = self._drop_overlapping_prefix(left, right, max_units, max_skip)
        if not right:
            return left
        # What is left of the new window can open on the punctuation that
        # followed the overlap ("..., and we need"): it belongs to the word
        # before it, no space.
        sep = "" if (_is_cjk(left[-1]) or _is_cjk(right[0])
                     or right[0] in _MATCH_STRIP_CHARS) else " "
        return (left + sep + right).strip()

    def _join_window(self, left: str, right: str) -> str:
        """Join text across a window seam: de-dup the overlap, tolerate junk."""
        if not self._seam_overlap:
            # Cut at a pause: nothing was decoded twice, so an equal word on
            # both sides is the speaker repeating it.
            return self._join_text(left, right, max_units=0)
        return self._join_text(
            left, right, max_units=self._window_dedup_units,
            max_skip=WINDOW_SEAM_MAX_SKIP_UNITS if self._window_dedup_units else 0)

    def _commit_final_text(self, text: str) -> None:
        text = _strip_prompt_leaks(text or "")
        # Fold the windows retired during this utterance in front of the last
        # window's text, then clear them: from here on ``_archive_text`` is the
        # single source of truth for the finished utterance (``finish()`` and
        # ``_composed_text`` both read it), and leaving the prefix behind would
        # duplicate it.
        if self._window_committed_text:
            text = self._join_window(self._window_committed_text, text)
            self._window_committed_text = ""
        if self._accumulate_segments:
            self._archive_text = self._join_text(self._archive_text, text)
        else:
            self._archive_text = text

    def _composed_text(self) -> str:
        # No window has been retired yet → identical to the pre-window-commit
        # behaviour, byte for byte.
        head = self._archive_text
        if self._window_committed_text:
            head = self._join_text(
                head, self._window_committed_text,
                max_units=self._window_dedup_units)
        if self._episode_final or not self._partial_text:
            return head
        if not head:
            return self._partial_text
        # The partial re-decodes the overlap frames, so dedup against the
        # committed tail when there is one.
        if self._window_committed_text:
            return self._join_window(head, self._partial_text)
        return self._join_text(head, self._partial_text)

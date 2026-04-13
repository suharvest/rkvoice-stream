"""
Streaming ASR session with optional VAD pre-filtering.

Separated from engine.py for modularity.  StreamSession is created by
Qwen3ASREngine.create_stream() and should not be instantiated directly.
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Callable

import re

from .config import SAMPLE_RATE
from .utils import apply_itn, parse_asr_output


class StreamSession:
    """
    Streaming ASR session with optional VAD pre-filtering.

    Two modes:
    - Without VAD: Fixed chunking.  Audio is always processed.
    - With VAD (scheme B): VAD runs continuously.  Only speech triggers
      encoder/decoder.  Silence is skipped.  When speech ends, remaining
      audio is flushed immediately.

    Usage::

        stream = engine.create_stream(language="Chinese")
        stream.feed_audio(pcm_chunk)
        result = stream.get_result()
        final = stream.finish()
    """

    # -------------------------------------------------------------- #
    # Init                                                            #
    # -------------------------------------------------------------- #

    def __init__(self, engine: "Qwen3ASREngine",
                 language: Optional[str] = "Chinese",
                 context: str = "",
                 chunk_size: float = 5.0,
                 memory_num: int = 2,
                 unfixed_chunks: int = 2,
                 rollback_tokens: int = 5,
                 max_new_tokens: int = 128,
                 on_text: Callable = None,
                 vad=None):
        """
        Args:
            engine:           Parent Qwen3ASREngine instance
            language:         Language hint (None = auto-detect)
            context:          Context description
            chunk_size:       Seconds per audio chunk (capped by encoder)
            memory_num:       Sliding-window width (≥ 2)
            unfixed_chunks:   First N chunks don't commit text
            rollback_tokens:  Remove last N tokens from prefix for stability
            max_new_tokens:   Max tokens to generate per chunk
            on_text:          Callback(text: str) on each new text
            vad:              Optional SileroVAD for speech gating
        """
        self.engine = engine
        self.language = language
        self.context = context
        self.chunk_size = min(chunk_size, engine.max_chunk_seconds)
        self.memory_num = max(2, memory_num)
        self.unfixed_chunks = unfixed_chunks
        self.rollback_tokens = rollback_tokens
        self.max_new_tokens = max_new_tokens
        self.on_text = on_text
        self.vad = vad

        self.chunk_samples = int(self.chunk_size * SAMPLE_RATE)
        self.buffer = np.zeros(0, dtype=np.float32)

        # Sliding window of (embedding, committed_text) pairs
        self._segments = deque(maxlen=self.memory_num)
        self._archive_text = ""
        self._chunk_id = 0
        self._current_text = ""
        self._current_language = language or ""

        # VAD state
        self._vad_speech_active = False
        self._speech_buf = np.zeros(0, dtype=np.float32)
        self._utterance_count = 0
        self._silence_since = 0.0
        self._pre_buf = np.zeros(0, dtype=np.float32)
        self._pre_buf_max = int(0.5 * SAMPLE_RATE)

        # Speculative encoding cache
        self._spec_embd = None
        self._spec_audio_len = 0

        # Stats
        self._total_enc_ms = 0.0
        self._total_llm_ms = 0.0
        self._total_audio_s = 0.0
        self._total_vad_ms = 0.0
        self._total_chunks = 0
        self._utterance_latencies = []

    # -------------------------------------------------------------- #
    # Public API                                                      #
    # -------------------------------------------------------------- #

    def feed_audio(self, pcm16k: np.ndarray) -> dict:
        """
        Feed audio data and process any complete chunks.

        Returns:
            dict with keys: language, text, is_final, is_speech,
            chunks_processed, [utterances, utterance_latency_ms]
        """
        x = np.asarray(pcm16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / 32768.0

        if self.vad is not None:
            return self._feed_with_vad(x)
        return self._feed_raw(x)

    def get_result(self) -> dict:
        """Get current recognition result without processing new audio."""
        return {
            "language": self._current_language,
            "text": self._current_text,
            "is_final": False,
            "is_speech": self._vad_speech_active if self.vad else True,
        }

    def finish(self, apply_itn_flag: bool = True) -> dict:
        """
        Finish streaming: process remaining buffer and return final result.

        Returns:
            dict with keys: language, text, is_final, stats
        """
        # Flush VAD speech buffer
        if self.vad and len(self._speech_buf) > 0:
            if len(self._speech_buf) >= int(0.5 * SAMPLE_RATE):
                self._process_chunk(self._speech_buf)
            self._speech_buf = np.zeros(0, dtype=np.float32)
            self._vad_speech_active = False

        # Process remaining raw buffer (non-VAD)
        if len(self.buffer) > 0:
            buf = self.buffer
            if len(buf) < int(0.5 * SAMPLE_RATE):
                buf = np.pad(buf, (0, int(0.5 * SAMPLE_RATE) - len(buf)))
            self._process_chunk(buf)
            self.buffer = np.zeros(0, dtype=np.float32)

        text = self._current_text
        if apply_itn_flag:
            text = apply_itn(text)

        return {
            "language": self._current_language,
            "text": text,
            "is_final": True,
            "stats": {
                "total_enc_ms": self._total_enc_ms,
                "total_llm_ms": self._total_llm_ms,
                "total_audio_s": self._total_audio_s,
                "total_vad_ms": self._total_vad_ms,
                "total_chunks": self._total_chunks,
                "utterances": self._utterance_count,
                "avg_utterance_latency_ms": (
                    sum(self._utterance_latencies)
                    / len(self._utterance_latencies)
                    if self._utterance_latencies else 0
                ),
                "rtf": ((self._total_enc_ms + self._total_llm_ms) / 1000.0
                        / max(self._total_audio_s, 0.1)),
            },
        }

    # -------------------------------------------------------------- #
    # Feed modes                                                      #
    # -------------------------------------------------------------- #

    def _feed_raw(self, x: np.ndarray) -> dict:
        """Fixed-chunking mode (no VAD)."""
        self.buffer = np.concatenate([self.buffer, x])
        chunks_processed = 0
        while len(self.buffer) >= self.chunk_samples:
            chunk = self.buffer[:self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples:]
            self._process_chunk(chunk)
            chunks_processed += 1

        return {
            "language": self._current_language,
            "text": self._current_text,
            "is_final": False,
            "is_speech": True,
            "chunks_processed": chunks_processed,
        }

    def _feed_with_vad(self, x: np.ndarray) -> dict:
        """
        VAD-gated processing (scheme B).

        During speech → accumulate & chunk through ASR.
        At speech-end → flush tail (with speculative encoding if available).
        During silence → no ASR, only maintain pre-buffer.
        """
        # 1. Feed to VAD
        t0 = time.perf_counter()
        self.vad.feed(x)
        self._total_vad_ms += (time.perf_counter() - t0) * 1000

        was_speech = self._vad_speech_active
        is_speech = self.vad.is_speech
        chunks_processed = 0

        if is_speech:
            if not was_speech:
                # Speech onset: prepend pre-buffer
                self._speech_buf = np.concatenate([self._pre_buf, x])
                self._pre_buf = np.zeros(0, dtype=np.float32)
                self._spec_embd = None
            else:
                self._speech_buf = np.concatenate([self._speech_buf, x])
            self._vad_speech_active = True

            # Process complete chunks
            while len(self._speech_buf) >= self.chunk_samples:
                chunk = self._speech_buf[:self.chunk_samples]
                self._speech_buf = self._speech_buf[self.chunk_samples:]
                self._process_chunk(chunk)
                chunks_processed += 1
                self._spec_embd = None

            # Speculative encoding: pre-encode if buffer ≥ 50% of chunk
            min_spec = self.chunk_samples // 2
            if (len(self._speech_buf) >= min_spec
                    and len(self._speech_buf) != self._spec_audio_len):
                spec_audio = self._speech_buf
                min_samples = int(0.5 * SAMPLE_RATE)
                if len(spec_audio) < min_samples:
                    spec_audio = np.pad(
                        spec_audio, (0, min_samples - len(spec_audio)))
                t0s = time.perf_counter()
                embd = self.engine.encoder.encode(spec_audio)
                if isinstance(embd, tuple):
                    embd = embd[0]
                self._total_enc_ms += (time.perf_counter() - t0s) * 1000
                self._spec_embd = embd
                self._spec_audio_len = len(self._speech_buf)

        elif was_speech and not is_speech:
            # Speech just ended: flush remaining audio
            self._speech_buf = np.concatenate([self._speech_buf, x])
            t_speech_end = time.perf_counter()

            if len(self._speech_buf) > 0:
                while len(self._speech_buf) >= self.chunk_samples:
                    chunk = self._speech_buf[:self.chunk_samples]
                    self._speech_buf = self._speech_buf[self.chunk_samples:]
                    self._process_chunk(chunk)
                    chunks_processed += 1
                    self._spec_embd = None

                # Tail audio: reuse speculative embedding if possible
                if len(self._speech_buf) > 0:
                    if (self._spec_embd is not None
                            and self._spec_audio_len == len(self._speech_buf)):
                        self._decode_with_window(
                            self._spec_embd,
                            len(self._speech_buf) / SAMPLE_RATE,
                            enc_ms=0, spec_tag=True)
                        self._total_audio_s += (
                            len(self._speech_buf) / SAMPLE_RATE)
                    else:
                        tail = self._speech_buf
                        min_samples = int(0.5 * SAMPLE_RATE)
                        if len(tail) < min_samples:
                            tail = np.pad(
                                tail, (0, min_samples - len(tail)))
                        self._process_chunk(tail)
                    chunks_processed += 1
                    self._speech_buf = np.zeros(0, dtype=np.float32)
                    self._spec_embd = None

            lat_ms = (time.perf_counter() - t_speech_end) * 1000
            self._utterance_latencies.append(lat_ms)
            self._vad_speech_active = False
            self._utterance_count += 1

            # Sentence boundary reset
            self._segments.clear()
            self._archive_text = self._current_text
            self._chunk_id = 0

        else:
            # Pure silence: ring-buffer pre-buffer
            self._pre_buf = np.concatenate([self._pre_buf, x])
            if len(self._pre_buf) > self._pre_buf_max:
                self._pre_buf = self._pre_buf[-self._pre_buf_max:]

        # Drain VAD queue
        while self.vad.has_speech():
            self.vad.pop_speech()

        return {
            "language": self._current_language,
            "text": self._current_text,
            "is_final": False,
            "is_speech": is_speech,
            "chunks_processed": chunks_processed,
            "utterances": self._utterance_count,
            "utterance_latency_ms": (
                self._utterance_latencies[-1]
                if self._utterance_latencies and not is_speech and was_speech
                else None
            ),
        }

    # -------------------------------------------------------------- #
    # Core processing pipeline                                        #
    # -------------------------------------------------------------- #

    def _process_chunk(self, audio_chunk: np.ndarray):
        """Encode audio chunk, then decode with sliding window."""
        chunk_sec = len(audio_chunk) / SAMPLE_RATE
        self._total_audio_s += chunk_sec

        t0 = time.perf_counter()
        result_enc = self.engine.encoder.encode(audio_chunk)
        if isinstance(result_enc, tuple):
            audio_embd = result_enc[0]
        else:
            audio_embd = result_enc
        enc_ms = (time.perf_counter() - t0) * 1000
        self._total_enc_ms += enc_ms

        self._decode_with_window(audio_embd, chunk_sec, enc_ms)

    def _decode_with_window(self, audio_embd: np.ndarray,
                            chunk_sec: float, enc_ms: float,
                            spec_tag: bool = False):
        """
        Shared decode core: sliding window → rollback → decode → commit.

        Called by ``_process_chunk`` (after encoding) and directly by the
        speculative-encoding path (with enc_ms=0, spec_tag=True).
        """
        # 1. Update sliding window
        if len(self._segments) >= self.memory_num:
            oldest_embd, oldest_text = self._segments.popleft()
            self._archive_text += oldest_text
        self._segments.append((audio_embd, ""))

        # 2. Concatenate audio embeddings from window
        all_audio = np.concatenate([s[0] for s in self._segments], axis=0)

        # 3. Prefix text from in-window completed segments only
        raw_prefix = "".join(
            self._segments[i][1] for i in range(len(self._segments) - 1))

        # 4. Token rollback
        prefix_str = self._apply_rollback(raw_prefix)

        # 5. Build full embedding & decode
        use_kv = self.engine._prefix_kv_cached
        full_embd, n_tokens = self.engine.build_embed(
            all_audio, prefix_str, self.language, self.context,
            skip_prefix=use_kv)

        t1 = time.perf_counter()
        result = self.engine.decoder.run_embed(
            full_embd, n_tokens, keep_prefix=use_kv)
        llm_ms = (time.perf_counter() - t1) * 1000
        self._total_llm_ms += llm_ms

        # 6. Parse output
        raw_text = result["text"]
        was_aborted = result.get("aborted", False)

        if self.language:
            new_text, lang = raw_text, self.language
        else:
            lang, new_text = parse_asr_output(raw_text)

        # Strip trailing garbage tokens (weak EOS → extra 1-3 chars after punct)
        new_text = self._strip_trailing_garbage(new_text)

        if was_aborted:
            new_text = ""
        if self._chunk_id < self.unfixed_chunks:
            new_text = ""

        # 7. Rollback alignment: trim preceding segment's text
        if self.rollback_tokens > 0 and len(self._segments) > 1:
            prev_texts = [self._segments[i][1]
                          for i in range(len(self._segments) - 1)]
            earlier = "".join(prev_texts[:-1])
            trimmed = prefix_str[len(earlier):]
            idx = len(self._segments) - 2
            self._segments[idx] = (self._segments[idx][0], trimmed)

        # 8. Commit new text
        last_embd, _ = self._segments[-1]
        self._segments[-1] = (last_embd, new_text)

        self._current_text = (self._archive_text
                              + "".join(s[1] for s in self._segments))
        self._current_language = lang or self._current_language
        self._chunk_id += 1
        self._total_chunks += 1

        if self.on_text:
            self.on_text(self._current_text)

        # 9. Performance log
        total_ms = enc_ms + llm_ms
        rtf = total_ms / 1000.0 / max(chunk_sec, 0.01)
        perf = result.get("perf", {})
        rb = f" rb={self.rollback_tokens}" if self.rollback_tokens else ""
        tag = " [SPEC]" if spec_tag else ""
        abort = " [ABORTED]" if was_aborted else ""
        enc_s = "0ms(spec)" if spec_tag else f"{enc_ms:.0f}ms"
        print(f"  [chunk {self._chunk_id}] enc={enc_s} "
              f"llm={llm_ms:.0f}ms rtf={rtf:.2f} "
              f"prefill={perf.get('prefill_time_ms', 0):.0f}ms "
              f"gen_tok={perf.get('generate_tokens', 0)}{rb}"
              f"{tag}{abort} | {new_text[:60]}...", flush=True)

    # Regex patterns for trailing garbage detection.
    # Pattern 1: sentence-end punct (。？！.?!) + 1-2 garbage chars
    # Pattern 2: comma/semicolon (，,；) + exactly 1 garbage char
    # Commas with 2+ chars after them are usually valid (e.g. "你好，世界").
    _RE_TRAILING_GARBAGE = re.compile(
        r'(?:[。？！.?!…][^。？！.?!…，,；：:、\s]{1,2}'
        r'|[，,；][^。？！.?!…，,；：:、\s])$'
    )

    @staticmethod
    def _strip_trailing_garbage(text: str) -> str:
        """
        Remove trailing garbage tokens after the last meaningful content.

        The RKLLM decoder sometimes generates 1-2 extra tokens past the
        natural ending because EOS logit is weak.  Observed pattern:
        sentence-ending punctuation + 1 random character.  Examples:
          "你好世界，你"       → "你好世界"
          "今天天气怎么样？铁"  → "今天天气怎么样"
          "今天天气怎么样？铁。" → "今天天气怎么样"

        Strategy: iteratively strip trailing sentence-end punctuation and
        [punct + 1 garbage char] patterns.

        Only applied when total text is ≤200 chars (ASR of short utterances).
        """
        if not text or len(text) < 2:
            return text
        if len(text) > 200:
            return text

        SENT_END = '。？！.?!…'

        prev = None
        while text != prev and len(text) >= 2:
            prev = text
            # Step 1: Strip trailing sentence-end punctuation
            text = text.rstrip(SENT_END)
            if text == prev:
                # Step 2: Strip [punct + 1 garbage char] at end
                m = StreamSession._RE_TRAILING_GARBAGE.search(text)
                if m:
                    text = text[:m.start()]

        return text

    def _apply_rollback(self, text: str) -> str:
        """
        Strip the last ``rollback_tokens`` tokens from *text*.

        Boundary tokens are often unreliable — the model may correct them
        with more audio context.  Rolling back lets the next chunk
        regenerate the boundary, fixing artefacts like
        "九百六。十七期" → "九百六十七期".
        """
        if self.rollback_tokens <= 0 or not text:
            return text
        ids = self.engine.tokenizer.encode(text).ids
        if len(ids) <= self.rollback_tokens:
            return ""
        return self.engine.tokenizer.decode(ids[:-self.rollback_tokens])

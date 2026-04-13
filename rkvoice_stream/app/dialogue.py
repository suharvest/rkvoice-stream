"""Streaming pipeline dialogue orchestrator for RK3576.

Instead of the serial flow (ASR → full LLM → TTS → play), this orchestrator
pipelines LLM token streaming with TTS synthesis:

  ASR → LLM streams tokens → sentence chunking → TTS per chunk → audio out

Each sentence is synthesized as soon as the LLM finishes it, saving ~100-200ms
of voice-to-voice latency compared to waiting for the full response.

Usage:
    orchestrator = DialogueOrchestrator(tts_backend=backend, llm_client=client)
    async for audio_bytes in orchestrator.process_turn("你好"):
        play(audio_bytes)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Sentence boundary punctuation
_SENTENCE_ENDINGS_ZH = set("。！？；")
_SENTENCE_ENDINGS_EN = set(".!?;")
_SENTENCE_ENDINGS = _SENTENCE_ENDINGS_ZH | _SENTENCE_ENDINGS_EN

# Force-flush buffer after this many chars without punctuation
_MAX_BUFFER_CHARS = 30

# Minimum chunk length to avoid synthesizing single-char fragments
_MIN_CHUNK_CHARS = 2


class LLMClient(Protocol):
    """Protocol for LLM streaming clients."""

    async def stream_chat(self, user_text: str) -> AsyncGenerator[str, None]:
        """Yield token strings as they arrive from the LLM."""
        ...


class EchoLLM:
    """Dummy LLM that echoes user text back. For testing the pipeline."""

    async def stream_chat(self, user_text: str) -> AsyncGenerator[str, None]:
        # Simulate token-by-token streaming
        for char in user_text:
            yield char
            await asyncio.sleep(0.01)


class DialogueOrchestrator:
    """Orchestrates streaming dialogue: text → LLM → sentence chunks → TTS → audio.

    NPU note: RKLLM (ASR/LLM decode) uses NPU domain 1, RKNN (TTS models)
    uses NPU domain 0. Sequential usage within this orchestrator is safe.
    True concurrent NPU access is untested, so we process one sentence at a time.
    """

    def __init__(
        self,
        tts_backend=None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.tts = tts_backend
        self.llm = llm_client or EchoLLM()

    async def process_turn(
        self, user_text: str
    ) -> AsyncGenerator[bytes, None]:
        """Full dialogue turn: user text → streamed TTS audio chunks.

        Yields WAV bytes for each sentence as soon as TTS completes it.
        The caller should play each chunk immediately for lowest latency.
        """
        t0 = time.monotonic()
        chunk_idx = 0

        async for sentence in self._chunk_sentences(self.llm.stream_chat(user_text)):
            if not sentence:
                continue

            logger.info(
                "dialogue chunk[%d]: %r (%d chars)",
                chunk_idx, sentence[:40], len(sentence),
            )

            # Run TTS in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            wav_bytes, meta = await loop.run_in_executor(
                None,
                lambda text=sentence: self.tts.synthesize(text=text),
            )

            logger.info(
                "dialogue chunk[%d]: TTS done, %.0fms, RTF=%.2f",
                chunk_idx,
                meta.get("inference_time", 0) * 1000,
                meta.get("rtf", 0),
            )

            yield wav_bytes
            chunk_idx += 1

        elapsed = time.monotonic() - t0
        logger.info("dialogue turn complete: %d chunks in %.1fs", chunk_idx, elapsed)

    async def process_turn_pcm(
        self, user_text: str
    ) -> AsyncGenerator[bytes, None]:
        """Like process_turn but yields raw int16 PCM (no WAV header).

        Prefixes the stream with 4 bytes: sample_rate as uint32 LE.
        Suitable for WebSocket streaming.
        """
        import struct

        sr = self.tts.get_sample_rate() if self.tts else 24000
        yield struct.pack("<I", sr)

        async for sentence in self._chunk_sentences(self.llm.stream_chat(user_text)):
            if not sentence:
                continue

            loop = asyncio.get_event_loop()
            wav_bytes, meta = await loop.run_in_executor(
                None,
                lambda text=sentence: self.tts.synthesize(text=text),
            )

            # Convert WAV to raw PCM int16
            import io
            import soundfile as sf

            buf = io.BytesIO(wav_bytes)
            audio, _ = sf.read(buf, dtype="float32")
            pcm = (np.clip(audio * 32767, -32768, 32767)).astype(np.int16)
            yield pcm.tobytes()

    @staticmethod
    async def _chunk_sentences(
        token_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        """Buffer LLM tokens and yield complete sentences.

        Sentence boundaries:
          - Chinese: 。！？；
          - English: . ! ? ;
          - Forced flush after _MAX_BUFFER_CHARS without punctuation
        """
        buffer = ""

        async for token_text in token_stream:
            buffer += token_text

            # Check if buffer ends with sentence-ending punctuation
            stripped = buffer.rstrip()
            if stripped and stripped[-1] in _SENTENCE_ENDINGS:
                if len(stripped) >= _MIN_CHUNK_CHARS:
                    yield stripped
                    buffer = ""
                continue

            # Force flush long segments (e.g. no punctuation in LLM output)
            if len(buffer) >= _MAX_BUFFER_CHARS:
                # Try to break at the last space/comma for English
                break_pos = -1
                for delim in (",", "，", " "):
                    pos = buffer.rfind(delim, _MIN_CHUNK_CHARS)
                    if pos > break_pos:
                        break_pos = pos

                if break_pos > 0:
                    yield buffer[: break_pos + 1].strip()
                    buffer = buffer[break_pos + 1 :]
                else:
                    yield buffer.strip()
                    buffer = ""

        # Flush remaining buffer
        if buffer.strip():
            yield buffer.strip()

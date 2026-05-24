from __future__ import annotations

import asyncio
import struct
import time

import numpy as np

from rkvoice_stream.app.dialogue import DialogueOrchestrator


class _OneSentenceLLM:
    async def stream_chat(self, user_text: str):
        yield user_text
        yield "。"


class _StreamingTTS:
    supports_streaming = True

    def __init__(self) -> None:
        self.full_synthesize_calls = 0

    def get_sample_rate(self) -> int:
        return 48000

    def synthesize(self, *args, **kwargs):
        self.full_synthesize_calls += 1
        raise AssertionError("dialogue PCM path must not call full synthesize()")

    def synthesize_stream(self, *args, **kwargs):
        yield np.full((4, 2), 0.25, dtype=np.float32), {"chunk_index": 0}
        time.sleep(0.01)
        yield np.full((2, 2), -0.25, dtype=np.float32), {"chunk_index": 1}


class _NonStreamingTTS:
    supports_streaming = False

    def get_sample_rate(self) -> int:
        return 48000


class _FailingStreamingTTS(_StreamingTTS):
    def synthesize_stream(self, *args, **kwargs):
        yield np.full((4, 2), 0.25, dtype=np.float32), {"chunk_index": 0}
        raise RuntimeError("codec failed")


def test_dialogue_pcm_uses_backend_streaming_chunks():
    tts = _StreamingTTS()
    orchestrator = DialogueOrchestrator(tts_backend=tts, llm_client=_OneSentenceLLM())

    async def collect() -> list[bytes]:
        out = []
        async for chunk in orchestrator.process_turn_pcm("你好"):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())

    assert struct.unpack("<I", chunks[0])[0] == 48000
    assert len(chunks) == 3
    assert len(chunks[1]) == 4 * 2 * 2
    assert len(chunks[2]) == 2 * 2 * 2
    assert tts.full_synthesize_calls == 0


def test_dialogue_pcm_rejects_non_streaming_tts():
    orchestrator = DialogueOrchestrator(tts_backend=_NonStreamingTTS(), llm_client=_OneSentenceLLM())

    async def collect() -> list[bytes]:
        out = []
        async for chunk in orchestrator.process_turn_pcm("你好"):
            out.append(chunk)
        return out

    try:
        asyncio.run(collect())
    except RuntimeError as exc:
        assert "requires a streaming TTS backend" in str(exc)
    else:
        raise AssertionError("non-streaming TTS backend must be rejected")


def test_dialogue_pcm_propagates_streaming_tts_errors():
    orchestrator = DialogueOrchestrator(tts_backend=_FailingStreamingTTS(), llm_client=_OneSentenceLLM())

    async def collect() -> list[bytes]:
        out = []
        async for chunk in orchestrator.process_turn_pcm("你好"):
            out.append(chunk)
        return out

    try:
        asyncio.run(collect())
    except RuntimeError as exc:
        assert "dialogue streaming TTS failed" in str(exc)
    else:
        raise AssertionError("streaming TTS errors must propagate to dialogue clients")

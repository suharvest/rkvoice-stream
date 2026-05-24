"""Health endpoint runtime-info tests."""

from __future__ import annotations

import asyncio
import signal
import struct

import numpy as np
import pytest
from fastapi.responses import JSONResponse, StreamingResponse

from rkvoice_stream.app import server


class _Backend:
    supports_streaming = True
    name = "moss_ort"

    def is_ready(self):
        return True

    def runtime_info(self):
        return {"profile": {"voice": "Junhao"}, "manifest": {"validated": True}}


class _StreamingBackend(_Backend):
    def __init__(self) -> None:
        self.full_synthesize_calls = 0

    def get_sample_rate(self):
        return 48000

    def synthesize(self, *args, **kwargs):
        self.full_synthesize_calls += 1
        raise AssertionError("/tts/stream must not call full synthesize()")

    def synthesize_stream(self, *args, **kwargs):
        yield np.full((4, 2), 0.5, dtype=np.float32), {"chunk_index": 0}
        yield np.full((2, 2), -0.5, dtype=np.float32), {"chunk_index": 1}


class _NonStreamingBackend(_Backend):
    supports_streaming = False


class _FailingStreamingBackend(_StreamingBackend):
    def synthesize_stream(self, *args, **kwargs):
        yield np.full((4, 2), 0.5, dtype=np.float32), {"chunk_index": 0}
        raise RuntimeError("codec failed")


class _BrokenPreloadBackend(_Backend):
    name = "broken"

    def preload(self):
        raise RuntimeError("preload failed")


class _Dialogue:
    async def process_turn_pcm(self, user_text: str):
        yield struct.pack("<I", 48000)
        yield np.full((2, 2), 0.25, dtype=np.float32).astype(np.int16).tobytes()


class _FailingDialogue:
    async def process_turn_pcm(self, user_text: str):
        yield struct.pack("<I", 48000)
        raise RuntimeError("dialogue failed")


class _WebSocket:
    def __init__(self, messages=None) -> None:
        self.messages = list(messages or [])
        self.accepted = False
        self.closed = False
        self.json_sent = []
        self.bytes_sent = []

    async def accept(self):
        self.accepted = True

    async def receive_json(self):
        if self.messages:
            return self.messages.pop(0)
        raise RuntimeError("client closed")

    async def send_json(self, payload):
        self.json_sent.append(payload)

    async def send_bytes(self, payload):
        self.bytes_sent.append(payload)

    async def close(self):
        self.closed = True


def test_health_includes_tts_runtime_info(monkeypatch):
    monkeypatch.setattr(server, "_backend", _Backend())
    monkeypatch.setattr(server, "_asr_backend", None)
    monkeypatch.setattr(server, "_speech_mode", "custom")

    result = asyncio.run(server.health())

    assert result["tts"] is True
    assert result["tts_backend"] == "moss_ort"
    assert result["streaming_tts"] is True
    assert result["tts_info"]["profile"]["voice"] == "Junhao"
    assert result["tts_info"]["manifest"]["validated"] is True


def test_server_import_does_not_override_uvicorn_signal_handlers():
    assert not hasattr(server, "_signal_handler")
    assert signal.getsignal(signal.SIGTERM) is not getattr(server, "_signal_handler", None)
    assert signal.getsignal(signal.SIGINT) is not getattr(server, "_signal_handler", None)


def test_startup_fails_when_required_tts_backend_cannot_preload(monkeypatch):
    import rkvoice_stream.engine.tts as tts_engine

    monkeypatch.delenv("CONFIG", raising=False)
    monkeypatch.delenv("SPEECH_MODE", raising=False)
    monkeypatch.setenv("TTS_BACKEND", "moss_ort")
    monkeypatch.setenv("REQUIRE_TTS_BACKEND", "1")
    monkeypatch.setenv("ASR_BACKEND", "disabled")
    monkeypatch.setattr(tts_engine, "create_backend", lambda _name: _BrokenPreloadBackend())
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setattr(server, "_asr_backend", None)
    monkeypatch.setattr(server, "_dialogue", None)

    with pytest.raises(RuntimeError, match="Required TTS backend 'moss_ort' failed to load"):
        asyncio.run(server.startup())


def test_startup_fails_when_required_tts_backend_is_disabled(monkeypatch):
    monkeypatch.delenv("CONFIG", raising=False)
    monkeypatch.delenv("SPEECH_MODE", raising=False)
    monkeypatch.setenv("TTS_BACKEND", "disabled")
    monkeypatch.setenv("REQUIRE_TTS_BACKEND", "1")
    monkeypatch.setenv("ASR_BACKEND", "disabled")

    with pytest.raises(RuntimeError, match="REQUIRE_TTS_BACKEND=1"):
        asyncio.run(server.startup())


def test_tts_stream_uses_backend_streaming_chunks(monkeypatch):
    backend = _StreamingBackend()
    monkeypatch.setattr(server, "_backend", backend)

    async def collect():
        response = await server.tts_stream(server.TTSRequest(text="你好"))
        assert isinstance(response, StreamingResponse)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())

    assert struct.unpack("<I", chunks[0])[0] == 48000
    assert len(chunks) == 3
    assert len(chunks[1]) == 4 * 2 * 2
    assert len(chunks[2]) == 2 * 2 * 2
    assert backend.full_synthesize_calls == 0


def test_tts_stream_rejects_non_streaming_backend(monkeypatch):
    monkeypatch.setattr(server, "_backend", _NonStreamingBackend())

    response = asyncio.run(server.tts_stream(server.TTSRequest(text="你好")))

    assert isinstance(response, JSONResponse)
    assert response.status_code == 501


def test_tts_stream_propagates_backend_stream_errors(monkeypatch):
    monkeypatch.setattr(server, "_backend", _FailingStreamingBackend())

    async def collect():
        response = await server.tts_stream(server.TTSRequest(text="你好"))
        assert isinstance(response, StreamingResponse)
        chunks = []
        try:
            async for chunk in response.body_iterator:
                chunks.append(chunk)
        except RuntimeError as exc:
            assert "TTS stream generation failed" in str(exc)
            return chunks
        raise AssertionError("streaming backend errors must propagate")

    chunks = asyncio.run(collect())

    assert struct.unpack("<I", chunks[0])[0] == 48000
    assert len(chunks) == 2


def test_dialogue_ws_rejects_non_streaming_backend(monkeypatch):
    ws = _WebSocket()
    monkeypatch.setattr(server, "_backend", _NonStreamingBackend())
    monkeypatch.setattr(server, "_dialogue", _Dialogue())

    asyncio.run(server.dialogue_ws(ws))

    assert ws.accepted is True
    assert ws.closed is True
    assert ws.json_sent == [{"error": "Dialogue requires streaming TTS backend"}]
    assert ws.bytes_sent == []


def test_dialogue_ws_streams_pcm_chunks(monkeypatch):
    ws = _WebSocket(messages=[{"text": "你好"}])
    monkeypatch.setattr(server, "_backend", _StreamingBackend())
    monkeypatch.setattr(server, "_dialogue", _Dialogue())

    asyncio.run(server.dialogue_ws(ws))

    assert ws.accepted is True
    assert ws.closed is True
    assert struct.unpack("<I", ws.bytes_sent[0])[0] == 48000
    assert len(ws.bytes_sent) == 2
    assert ws.json_sent == [{"done": True, "chunks": 2}]


def test_dialogue_ws_reports_streaming_failures(monkeypatch):
    ws = _WebSocket(messages=[{"text": "你好"}])
    monkeypatch.setattr(server, "_backend", _StreamingBackend())
    monkeypatch.setattr(server, "_dialogue", _FailingDialogue())

    asyncio.run(server.dialogue_ws(ws))

    assert ws.accepted is True
    assert ws.closed is True
    assert struct.unpack("<I", ws.bytes_sent[0])[0] == 48000
    assert ws.json_sent == [{"error": "dialogue streaming failed", "chunks": 1}]

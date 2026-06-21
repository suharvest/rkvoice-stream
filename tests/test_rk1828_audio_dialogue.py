"""Phase 2c app-integration tests (off-device, mock worker + in-process WS).

Covers:
  - server.py third backend slot ``_audio_llm_backend``: env-driven load,
    REQUIRE_AUDIO_LLM_BACKEND fail-fast, /health reporting.
  - /audio_dialogue WS endpoint: audio in -> AudioLLM (audio->text) -> TTS ->
    PCM out, sample-rate prefix, control (prompt) frame, error reporting,
    AudioLLM-not-loaded rejection.
  - DialogueOrchestrator pluggable understanding stage: AudioLLM path
    (process_audio_turn_pcm) vs text path (process_turn_pcm) — both reach TTS.
  - _apply_audio_llm_env config wiring.
  - End-to-end through the real mock gemma4 worker subprocess.

Dual-mode: these are pure in-process ASGI-handler tests (no SERVICE_URL needed),
so they run identically in HTTP and direct mode.
"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import wave

import numpy as np
import pytest

from rkvoice_stream.app import server
from rkvoice_stream.app.dialogue import DialogueOrchestrator

MOCK_WORKER = os.path.join(
    os.path.dirname(__file__), "fixtures", "mock_rk1828_gemma4_worker.py"
)
EXPECTED_TEXT = "你好，这是一段测试转写。"


# ── fakes ─────────────────────────────────────────────────────────────────

class _FakeTTS:
    """Minimal streaming TTS backend: each char -> one float32 PCM chunk."""

    supports_streaming = True
    name = "fake_tts"

    def is_ready(self):
        return True

    def get_sample_rate(self):
        return 24000

    def synthesize_stream(self, text, **kwargs):
        # one PCM chunk per synthesized sentence (deterministic chunk count)
        yield np.full(4, 0.1, dtype=np.float32), {"chunk_index": 0}


class _FakeAudioLLM:
    """AudioLLM that yields a fixed sentence-bearing token stream."""

    name = "fake_audio_llm"

    def __init__(self, ready=True, tokens=None):
        self._ready = ready
        self._tokens = tokens if tokens is not None else ["你好，", "世界。"]

    def is_ready(self):
        return self._ready

    def preload(self):
        self._ready = True

    def generate_stream(self, audio, sample_rate, prompt=None, **kwargs):
        for t in self._tokens:
            yield t


class _FailingAudioLLM(_FakeAudioLLM):
    def generate_stream(self, audio, sample_rate, prompt=None, **kwargs):
        yield "部分"
        raise RuntimeError("audio llm crashed")


class _WebSocket:
    """Mock WS feeding a fixed message sequence; records sends."""

    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.accepted = False
        self.closed = False
        self.json_sent = []
        self.bytes_sent = []

    async def accept(self):
        self.accepted = True

    async def receive(self):
        if self.messages:
            return self.messages.pop(0)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload):
        self.json_sent.append(payload)

    async def send_bytes(self, payload):
        self.bytes_sent.append(payload)

    async def close(self):
        self.closed = True


def _wav_bytes(seconds=0.5, sr=16000) -> bytes:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    pcm = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _audio_msg(audio_bytes: bytes) -> dict:
    return {"type": "websocket.receive", "bytes": audio_bytes}


def _ctl_msg(payload: str) -> dict:
    return {"type": "websocket.receive", "text": payload}


# ── orchestrator: pluggable understanding stage ────────────────────────────

def test_orchestrator_has_audio_llm_flag():
    orch = DialogueOrchestrator(tts_backend=_FakeTTS(), audio_llm_backend=_FakeAudioLLM())
    assert orch.has_audio_llm() is True
    orch2 = DialogueOrchestrator(tts_backend=_FakeTTS())
    assert orch2.has_audio_llm() is False
    orch3 = DialogueOrchestrator(tts_backend=_FakeTTS(), audio_llm_backend=_FakeAudioLLM(ready=False))
    assert orch3.has_audio_llm() is False


def test_process_audio_turn_pcm_streams_audio():
    orch = DialogueOrchestrator(tts_backend=_FakeTTS(), audio_llm_backend=_FakeAudioLLM())

    async def collect():
        out = []
        async for chunk in orch.process_audio_turn_pcm(np.zeros(8000, np.float32), 16000):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    # first chunk = sample-rate prefix
    assert struct.unpack("<I", chunks[0])[0] == 24000
    assert len(chunks) > 1  # got PCM after the prefix


def test_process_audio_turn_pcm_requires_audio_llm():
    orch = DialogueOrchestrator(tts_backend=_FakeTTS())  # no audio llm

    async def run():
        async for _ in orch.process_audio_turn_pcm(np.zeros(8, np.float32), 16000):
            pass

    with pytest.raises(RuntimeError, match="AudioLLM"):
        asyncio.run(run())


def test_text_path_still_works():
    """The text understanding stage (process_turn_pcm) is unchanged."""
    orch = DialogueOrchestrator(tts_backend=_FakeTTS())  # echo LLM

    async def collect():
        out = []
        async for chunk in orch.process_turn_pcm("hi."):
            out.append(chunk)
        return out

    chunks = asyncio.run(collect())
    assert struct.unpack("<I", chunks[0])[0] == 24000
    assert len(chunks) > 1


def test_process_audio_turn_pcm_propagates_audio_llm_error():
    orch = DialogueOrchestrator(tts_backend=_FakeTTS(), audio_llm_backend=_FailingAudioLLM())

    async def run():
        async for _ in orch.process_audio_turn_pcm(np.zeros(8, np.float32), 16000):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(run())


# ── /audio_dialogue endpoint ───────────────────────────────────────────────

def _wire(monkeypatch, audio_llm=None, tts=None):
    tts = tts or _FakeTTS()
    audio_llm = audio_llm if audio_llm is not None else _FakeAudioLLM()
    orch = DialogueOrchestrator(tts_backend=tts, audio_llm_backend=audio_llm)
    monkeypatch.setattr(server, "_backend", tts)
    monkeypatch.setattr(server, "_audio_llm_backend", audio_llm)
    monkeypatch.setattr(server, "_dialogue", orch)
    return orch


def test_audio_dialogue_streams_pcm(monkeypatch):
    _wire(monkeypatch)
    ws = _WebSocket(messages=[_audio_msg(_wav_bytes())])
    asyncio.run(server.audio_dialogue_ws(ws))

    assert ws.accepted and ws.closed
    assert struct.unpack("<I", ws.bytes_sent[0])[0] == 24000
    assert len(ws.bytes_sent) > 1
    assert ws.json_sent[-1]["done"] is True
    # chunk_count counts every binary frame incl. the sample-rate prefix
    # (same convention as /dialogue).
    assert ws.json_sent[-1]["chunks"] == len(ws.bytes_sent)


def test_audio_dialogue_accepts_prompt_control_frame(monkeypatch):
    seen = {}

    class _PromptCapturingLLM(_FakeAudioLLM):
        def generate_stream(self, audio, sample_rate, prompt=None, **kwargs):
            seen["prompt"] = prompt
            seen["sr"] = sample_rate
            yield from ["好。"]

    _wire(monkeypatch, audio_llm=_PromptCapturingLLM())
    ws = _WebSocket(messages=[
        _ctl_msg('{"prompt": "请转写", "sample_rate": 16000}'),
        _audio_msg(_wav_bytes(sr=16000)),
    ])
    asyncio.run(server.audio_dialogue_ws(ws))
    assert seen["prompt"] == "请转写"
    # WAV header carries 16000; decoded sr passed through
    assert seen["sr"] == 16000
    assert ws.json_sent[-1]["done"] is True


def test_audio_dialogue_rejected_without_audio_llm(monkeypatch):
    monkeypatch.setattr(server, "_backend", _FakeTTS())
    monkeypatch.setattr(server, "_audio_llm_backend", None)
    monkeypatch.setattr(server, "_dialogue", DialogueOrchestrator(tts_backend=_FakeTTS()))
    ws = _WebSocket(messages=[_audio_msg(_wav_bytes())])
    asyncio.run(server.audio_dialogue_ws(ws))
    assert ws.json_sent == [{"error": "Audio dialogue not available (AudioLLM not loaded)"}]
    assert ws.bytes_sent == []


def test_audio_dialogue_reports_streaming_failure(monkeypatch):
    _wire(monkeypatch, audio_llm=_FailingAudioLLM())
    ws = _WebSocket(messages=[_audio_msg(_wav_bytes())])
    asyncio.run(server.audio_dialogue_ws(ws))
    # error JSON emitted (chunks may include the sample-rate prefix bytes sent
    # before the failure surfaced)
    assert any("audio dialogue streaming failed" == j.get("error") for j in ws.json_sent)


def test_decode_audio_bytes_wav_and_raw():
    audio, sr = server._decode_audio_bytes(_wav_bytes(sr=16000))
    assert sr == 16000
    assert audio.dtype == np.float32
    # raw int16 PCM fallback
    raw = (np.ones(100, np.int16) * 1000).tobytes()
    audio2, sr2 = server._decode_audio_bytes(raw, fallback_sr=16000)
    assert sr2 == 16000
    assert audio2.shape[0] == 100


# ── server third-slot loading ───────────────────────────────────────────────

def _clear_startup_env(monkeypatch):
    for k in ("CONFIG", "SPEECH_MODE", "TTS_BACKEND", "ASR_BACKEND",
              "REQUIRE_TTS_BACKEND", "REQUIRE_ASR_BACKEND"):
        monkeypatch.delenv(k, raising=False)


def test_startup_loads_audio_llm_slot(monkeypatch):
    _clear_startup_env(monkeypatch)
    monkeypatch.setenv("TTS_BACKEND", "disabled")
    monkeypatch.setenv("ASR_BACKEND", "disabled")
    monkeypatch.setenv("AUDIO_LLM_BACKEND", "fake")

    import rkvoice_stream.engine.audio_llm as ae
    monkeypatch.setattr(ae, "create_audio_llm", lambda _n: _FakeAudioLLM())
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setattr(server, "_asr_backend", None)
    monkeypatch.setattr(server, "_audio_llm_backend", None)

    asyncio.run(server.startup())
    assert server._audio_llm_backend is not None
    assert server._audio_llm_backend.is_ready() is True
    # tear down the module global so other tests are unaffected
    server._audio_llm_backend = None


def test_startup_fails_when_required_audio_llm_cannot_load(monkeypatch):
    _clear_startup_env(monkeypatch)
    monkeypatch.setenv("TTS_BACKEND", "disabled")
    monkeypatch.setenv("ASR_BACKEND", "disabled")
    monkeypatch.setenv("AUDIO_LLM_BACKEND", "fake")
    monkeypatch.setenv("REQUIRE_AUDIO_LLM_BACKEND", "1")

    def _boom(_n):
        raise RuntimeError("no device")

    import rkvoice_stream.engine.audio_llm as ae
    monkeypatch.setattr(ae, "create_audio_llm", _boom)
    monkeypatch.setattr(server, "_audio_llm_backend", None)

    with pytest.raises(RuntimeError, match="Required AudioLLM backend"):
        asyncio.run(server.startup())


def test_startup_fails_when_required_audio_llm_disabled(monkeypatch):
    _clear_startup_env(monkeypatch)
    monkeypatch.setenv("TTS_BACKEND", "disabled")
    monkeypatch.setenv("ASR_BACKEND", "disabled")
    monkeypatch.setenv("AUDIO_LLM_BACKEND", "disabled")
    monkeypatch.setenv("REQUIRE_AUDIO_LLM_BACKEND", "1")

    with pytest.raises(RuntimeError, match="REQUIRE_AUDIO_LLM_BACKEND=1"):
        asyncio.run(server.startup())


def test_health_reports_audio_llm(monkeypatch):
    monkeypatch.setattr(server, "_backend", None)
    monkeypatch.setattr(server, "_asr_backend", None)
    monkeypatch.setattr(server, "_audio_llm_backend", _FakeAudioLLM())
    monkeypatch.setattr(server, "_speech_mode", "custom")
    result = asyncio.run(server.health())
    assert result["audio_llm"] is True
    assert result["audio_llm_backend"] == "fake_audio_llm"


# ── config env wiring ───────────────────────────────────────────────────────

def test_apply_audio_llm_env(monkeypatch):
    from rkvoice_stream import _apply_audio_llm_env
    for k in ("AUDIO_LLM_BACKEND", "RK1828_GEMMA4_BINARY", "RK1828_GEMMA4_MODEL_DIR",
              "RK1828_DEVICE_ID", "GEMMA4_LLM_CORE_MASK", "GEMMA4_AUDIO_CORE_MASK",
              "REQUIRE_AUDIO_LLM_BACKEND"):
        monkeypatch.delenv(k, raising=False)

    _apply_audio_llm_env({
        "backend": "gemma4_rk1828",
        "binary_path": "/opt/rk1828/rknn_gemma4_demo",
        "model_dir": "/opt/rk1828/models/gemma4",
        "device_id": "0001:11:00.0",
        "require_backend": True,
        "env": {"GEMMA4_LLM_CORE_MASK": "0xff", "GEMMA4_AUDIO_CORE_MASK": "0xf"},
    })
    assert os.environ["AUDIO_LLM_BACKEND"] == "gemma4_rk1828"
    assert os.environ["RK1828_GEMMA4_BINARY"] == "/opt/rk1828/rknn_gemma4_demo"
    assert os.environ["RK1828_GEMMA4_MODEL_DIR"] == "/opt/rk1828/models/gemma4"
    assert os.environ["RK1828_DEVICE_ID"] == "0001:11:00.0"
    assert os.environ["REQUIRE_AUDIO_LLM_BACKEND"] == "True"
    assert os.environ["GEMMA4_LLM_CORE_MASK"] == "0xff"
    assert os.environ["GEMMA4_AUDIO_CORE_MASK"] == "0xf"


# ── end-to-end through the real mock gemma4 worker ──────────────────────────

def test_audio_dialogue_e2e_mock_worker(monkeypatch):
    """Full path: real gemma4 mock worker subprocess -> TTS -> PCM out."""
    from rkvoice_stream.backends.audio_llm.gemma4_rk1828 import Gemma4RK1828Service
    from rkvoice_stream.engine.audio_llm import AudioLLMBackend, DEFAULT_MAX_NEW_TOKENS

    svc = Gemma4RK1828Service(binary_path=MOCK_WORKER, model_dir="/tmp/fake", device_id="0001:11:00.0")
    svc.load()

    class _SvcBackend(AudioLLMBackend):
        name = "gemma4_rk1828"
        def is_ready(self): return svc.is_ready()
        def preload(self): pass
        def get_capabilities(self): return {"audio"}
        def generate(self, audio, sample_rate, prompt=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, **kw):
            return svc.generate(audio, sample_rate, prompt, max_new_tokens)
        def generate_stream(self, audio, sample_rate, prompt=None, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, **kw):
            yield from svc.generate_stream(audio, sample_rate, prompt, max_new_tokens)

    try:
        orch = DialogueOrchestrator(tts_backend=_FakeTTS(), audio_llm_backend=_SvcBackend())
        monkeypatch.setattr(server, "_backend", _FakeTTS())
        monkeypatch.setattr(server, "_audio_llm_backend", _SvcBackend())
        monkeypatch.setattr(server, "_dialogue", orch)

        ws = _WebSocket(messages=[_audio_msg(_wav_bytes())])
        asyncio.run(server.audio_dialogue_ws(ws))

        assert struct.unpack("<I", ws.bytes_sent[0])[0] == 24000
        # mock emits "你好，这是一段测试转写。"; "，" is not a sentence ending,
        # so the chunker yields ONE sentence -> _FakeTTS = 1 PCM chunk.
        n_sentences = 1
        assert len(ws.bytes_sent) - 1 == n_sentences  # minus SR prefix
        # chunk_count includes the SR prefix frame (same as /dialogue).
        assert ws.json_sent[-1] == {"done": True, "chunks": n_sentences + 1}
    finally:
        svc.cleanup()

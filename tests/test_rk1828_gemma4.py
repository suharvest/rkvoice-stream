"""RK1828 Gemma-4 AudioLLM Phase 2a tests (off-device, mock worker + unit).

Covers:
  - AudioLLMWorker protocol parsing: text token frames, EOS sentinel
    (0xFFFFFFFE, distinct from TTS 0xFFFFFFFF), truncated frame (crash),
    stdout-EOF crash detection, empty-frame skip.
  - READY handshake + protocol version negotiation (match + mismatch).
  - factory selects gemma4_rk1828; get_capabilities == {"audio"}.
  - end-to-end generate (concat) / generate_stream (per-token) against the mock
    worker produce a non-empty text stream.
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np
import pytest

MOCK_WORKER = os.path.join(
    os.path.dirname(__file__), "fixtures", "mock_rk1828_gemma4_worker.py"
)

# Must match the mock worker's fixed token sequence.
EXPECTED_TOKENS = ["你", "好", "，", "这是", "一段", "测试", "转写", "。"]


def _fake_audio(seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


# ── factory / capabilities (no worker spawn) ──────────────────────

def test_factory_selects_gemma4_rk1828():
    from rkvoice_stream.engine.audio_llm import create_audio_llm
    from rkvoice_stream.backends.audio_llm.gemma4_rk1828 import Gemma4RK1828Backend

    backend = create_audio_llm("gemma4_rk1828")
    assert isinstance(backend, Gemma4RK1828Backend)
    assert backend.name == "gemma4_rk1828"
    assert backend.get_capabilities() == {"audio"}
    assert backend.is_ready() is False  # not preloaded


def test_factory_unknown_backend():
    from rkvoice_stream.engine.audio_llm import create_audio_llm

    with pytest.raises(ValueError):
        create_audio_llm("nope_llm")


def test_factory_env_default(monkeypatch):
    from rkvoice_stream.engine.audio_llm import create_audio_llm
    from rkvoice_stream.backends.audio_llm.gemma4_rk1828 import Gemma4RK1828Backend

    monkeypatch.setenv("AUDIO_LLM_BACKEND", "gemma4_rk1828")
    backend = create_audio_llm()
    assert isinstance(backend, Gemma4RK1828Backend)


# ── protocol parsing (synthetic streams via a fake stdout) ────────

class _FakeProc:
    """Minimal subprocess.Popen stand-in feeding a fixed stdout byte stream."""

    def __init__(self, stdout_bytes: bytes, alive: bool = True):
        self.stdout = io.BytesIO(stdout_bytes)
        self.stdin = io.BytesIO()
        self._alive = alive
        self.returncode = None if alive else 1

    def poll(self):
        return None if self._alive else 1


def _make_worker_with_stream(stdout_bytes: bytes):
    from rkvoice_stream.runtime.rknn3_worker import AudioLLMWorker

    w = AudioLLMWorker(binary_path="x", model_dir="x")
    w._proc = _FakeProc(stdout_bytes)
    w._ready = True
    return w


def _tframe(tok: str) -> bytes:
    data = tok.encode("utf-8")
    return struct.pack("<I", len(data)) + data


def _eos() -> bytes:
    return struct.pack("<I", 0xFFFFFFFE)


def test_eos_distinct_from_tts_sentinel():
    from rkvoice_stream.runtime.rknn3_worker import END_OF_STREAM, END_OF_UTTERANCE

    assert END_OF_STREAM == 0xFFFFFFFE
    assert END_OF_UTTERANCE == 0xFFFFFFFF
    assert END_OF_STREAM != END_OF_UTTERANCE


def test_protocol_normal_token_stream():
    stream = _tframe("你好") + _tframe("世界") + _eos()
    w = _make_worker_with_stream(stream)
    toks = list(w.generate_stream("/tmp/a.wav"))
    assert toks == ["你好", "世界"]


def test_protocol_eos_immediate():
    w = _make_worker_with_stream(_eos())
    assert list(w.generate_stream("/tmp/a.wav")) == []


def test_protocol_empty_frame_skipped():
    stream = struct.pack("<I", 0) + _tframe("hi") + _eos()
    w = _make_worker_with_stream(stream)
    assert list(w.generate_stream("/tmp/a.wav")) == ["hi"]


def test_protocol_truncated_frame_crashes():
    from rkvoice_stream.runtime.rknn3_worker import WorkerCrashError

    stream = struct.pack("<I", 8) + b"\x01\x02\x03\x04"  # declares 8, EOF after 4
    w = _make_worker_with_stream(stream)
    with pytest.raises(WorkerCrashError):
        list(w.generate_stream("/tmp/a.wav"))


def test_protocol_stdout_eof_crashes():
    from rkvoice_stream.runtime.rknn3_worker import WorkerCrashError

    w = _make_worker_with_stream(b"")
    with pytest.raises(WorkerCrashError):
        list(w.generate_stream("/tmp/a.wav"))


def test_protocol_desync_implausible_length():
    from rkvoice_stream.runtime.rknn3_worker import (
        ProtocolDesyncError,
        WorkerCrashError,
        MAX_FRAME_BYTES,
    )

    leaked = b"DIAG some diagnostic leaked to stdout\n"
    assert struct.unpack("<I", leaked[:4])[0] > MAX_FRAME_BYTES
    w = _make_worker_with_stream(leaked)
    with pytest.raises(ProtocolDesyncError) as ei:
        list(w.generate_stream("/tmp/a.wav"))
    assert isinstance(ei.value, WorkerCrashError)
    assert w._ready is False


def test_generate_full_concat():
    stream = _tframe("foo") + _tframe("bar") + _eos()
    w = _make_worker_with_stream(stream)
    assert w.generate("/tmp/a.wav") == "foobar"


def test_ready_version_parse():
    from rkvoice_stream.runtime.rknn3_worker import AudioLLMWorker

    w = AudioLLMWorker(binary_path="x", model_dir="x")
    w._parse_ready("READY 1")
    assert w._worker_protocol_version == 1
    w._parse_ready("Init done READY v2 extra")
    assert w._worker_protocol_version == 2


# ── end-to-end against the real mock worker subprocess ────────────

@pytest.fixture
def gemma4_service():
    from rkvoice_stream.backends.audio_llm.gemma4_rk1828 import Gemma4RK1828Service

    svc = Gemma4RK1828Service(
        binary_path=MOCK_WORKER,
        model_dir="/tmp/fake-model",
        device_id="0001:11:00.0",
    )
    svc.load()
    yield svc
    svc.cleanup()


def test_e2e_service_ready(gemma4_service):
    assert gemma4_service.is_ready() is True


def test_e2e_generate_concat(gemma4_service):
    text = gemma4_service.generate(_fake_audio(), 16000, prompt="转写")
    assert text == "".join(EXPECTED_TOKENS)
    assert len(text) > 0


def test_e2e_generate_stream(gemma4_service):
    toks = list(gemma4_service.generate_stream(_fake_audio(), 16000, prompt="转写"))
    assert toks == EXPECTED_TOKENS
    assert all(isinstance(t, str) for t in toks)


def test_e2e_generate_resamples_non_16k(gemma4_service):
    """Service materialises audio at any input rate (resample path)."""
    text = gemma4_service.generate(_fake_audio(sr=24000), 24000)
    assert text == "".join(EXPECTED_TOKENS)


def test_e2e_max_new_tokens_truncates(gemma4_service):
    toks = list(
        gemma4_service.generate_stream(_fake_audio(), 16000, max_new_tokens=3)
    )
    assert toks == EXPECTED_TOKENS[:3]


def test_e2e_backend_via_factory():
    """Full backend path: factory -> preload (env-driven) -> generate."""
    from rkvoice_stream.engine.audio_llm import create_audio_llm

    os.environ["RK1828_GEMMA4_BINARY"] = MOCK_WORKER
    os.environ["RK1828_GEMMA4_MODEL_DIR"] = "/tmp/fake-model"
    os.environ["RK1828_DEVICE_ID"] = "0001:11:00.0"
    backend = None
    try:
        backend = create_audio_llm("gemma4_rk1828")
        backend.preload()
        assert backend.is_ready() is True
        info = backend.runtime_info()
        assert info["device_id"] == "0001:11:00.0"
        assert info["capabilities"] == ["audio"]

        text = backend.generate(_fake_audio(), 16000, prompt="转写")
        assert text == "".join(EXPECTED_TOKENS)

        toks = list(backend.generate_stream(_fake_audio(), 16000))
        assert toks == EXPECTED_TOKENS
    finally:
        if backend is not None:
            backend.cleanup()
        for k in ("RK1828_GEMMA4_BINARY", "RK1828_GEMMA4_MODEL_DIR", "RK1828_DEVICE_ID"):
            os.environ.pop(k, None)


def test_protocol_version_mismatch():
    """Worker advertising a different protocol version must fail start()."""
    from rkvoice_stream.runtime.rknn3_worker import AudioLLMWorker, ProtocolMismatchError

    w = AudioLLMWorker(
        binary_path=MOCK_WORKER,
        model_dir="/tmp/fake-model",
        extra_args=["--proto-version", "99"],
    )
    with pytest.raises(ProtocolMismatchError):
        w.start()
    w.stop()

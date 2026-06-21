"""RK1828 Qwen3-TTS Phase 1 tests (off-device, mock worker + unit).

Covers:
  - rknn3_worker protocol parsing: normal frames, end-of-utterance sentinel,
    truncated frame (crash), stdout-EOF crash detection.
  - Service int16 PCM -> float32 conversion.
  - factory selects qwen3_tts_rk1828.
  - platform registry resolves rk1828 (device_id / is_coprocessor).
  - end-to-end synthesize / synthesize_stream against the mock worker produce
    non-empty audio with the contract types.
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np
import pytest

MOCK_WORKER = os.path.join(os.path.dirname(__file__), "fixtures", "mock_rk1828_tts_worker.py")

SAMPLES_PER_CHUNK = 1920
N_CHUNKS = 3


# ── factory / platform (no worker spawn) ──────────────────────────

def test_factory_selects_rk1828():
    from rkvoice_stream.engine.tts import create_tts
    from rkvoice_stream.backends.tts.qwen3_tts_rk1828 import Qwen3TTSRK1828Backend

    backend = create_tts("qwen3_tts_rk1828")
    assert isinstance(backend, Qwen3TTSRK1828Backend)
    assert backend.name == "qwen3_tts_rk1828"
    assert backend.get_sample_rate() == 24000
    assert backend.is_ready() is False  # not preloaded


def test_platform_registry_rk1828():
    from rkvoice_stream.platform import get_platform, PLATFORMS

    assert "rk1828" in PLATFORMS
    p = get_platform("rk1828")
    assert p.name == "rk1828"
    assert p.is_coprocessor is True
    assert p.device_id == "0001:11:00.0"
    assert p.npu_memory_limit_mb == 5120


def test_existing_platforms_unaffected():
    """Defaulted fields must not break rk3576/rk3588 construction."""
    from rkvoice_stream.platform import get_platform

    for name in ("rk3576", "rk3588"):
        p = get_platform(name)
        assert p.is_coprocessor is False
        assert p.device_id is None


# ── int16 -> float32 conversion ───────────────────────────────────

def test_pcm16_to_float32():
    from rkvoice_stream.backends.tts.qwen3_tts_rk1828 import _pcm16_to_float32

    assert _pcm16_to_float32(b"").shape == (0,)

    pcm = np.array([0, 32767, -32768, 16384], dtype="<i2").tobytes()
    out = _pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out.shape == (4,)
    assert abs(out[0]) < 1e-6
    assert out[1] == pytest.approx(32767 / 32768.0, abs=1e-6)
    assert out[2] == pytest.approx(-1.0, abs=1e-6)
    assert np.all(np.abs(out) <= 1.0)


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
    from rkvoice_stream.runtime.rknn3_worker import RKNN3Worker

    w = RKNN3Worker(binary_path="x", model_dir="x")
    w._proc = _FakeProc(stdout_bytes)
    w._ready = True
    return w


def _frame(pcm: bytes) -> bytes:
    return struct.pack("<I", len(pcm)) + pcm


def _eou() -> bytes:
    return struct.pack("<I", 0xFFFFFFFF)


def test_protocol_normal_stream():
    a = b"\x01\x00\x02\x00"
    b = b"\x03\x00\x04\x00"
    stream = _frame(a) + _frame(b) + _eou()
    w = _make_worker_with_stream(stream)
    chunks = list(w.synthesize_stream("hi"))
    assert chunks == [a, b]


def test_protocol_end_of_utterance_immediate():
    w = _make_worker_with_stream(_eou())
    assert list(w.synthesize_stream("hi")) == []


def test_protocol_empty_frame_skipped():
    # len==0 non-sentinel frame must be skipped, then real data + eou.
    payload = b"\xaa\xbb"
    stream = struct.pack("<I", 0) + _frame(payload) + _eou()
    w = _make_worker_with_stream(stream)
    assert list(w.synthesize_stream("hi")) == [payload]


def test_protocol_truncated_frame_crashes():
    from rkvoice_stream.runtime.rknn3_worker import WorkerCrashError

    # Declares 8 bytes but only 4 follow then EOF.
    stream = struct.pack("<I", 8) + b"\x01\x02\x03\x04"
    w = _make_worker_with_stream(stream)
    with pytest.raises(WorkerCrashError):
        list(w.synthesize_stream("hi"))


def test_protocol_stdout_eof_crashes():
    from rkvoice_stream.runtime.rknn3_worker import WorkerCrashError

    # EOF before any length prefix.
    w = _make_worker_with_stream(b"")
    with pytest.raises(WorkerCrashError):
        list(w.synthesize_stream("hi"))


def test_synthesize_full_concat():
    a = b"\x01\x00"
    b = b"\x02\x00"
    w = _make_worker_with_stream(_frame(a) + _frame(b) + _eou())
    assert w.synthesize("hi") == a + b


# ── end-to-end against the real mock worker subprocess ────────────

@pytest.fixture
def rk1828_service():
    from rkvoice_stream.backends.tts.qwen3_tts_rk1828 import Qwen3TTSRK1828Service

    svc = Qwen3TTSRK1828Service(
        binary_path=MOCK_WORKER,
        model_dir="/tmp/fake-model",
        ref_speaker="girl_base",
        device_id="0001:11:00.0",
    )
    svc.load()
    yield svc
    svc.cleanup()


def test_e2e_service_ready(rk1828_service):
    assert rk1828_service.is_ready() is True
    assert rk1828_service.get_sample_rate() == 24000


def test_e2e_synthesize_wav(rk1828_service):
    import soundfile as sf

    wav_bytes, meta = rk1828_service.synthesize("你好世界")
    assert wav_bytes[:4] == b"RIFF"
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    assert sr == 24000
    assert len(audio) == SAMPLES_PER_CHUNK * N_CHUNKS
    assert np.abs(audio).max() > 0.01  # non-silent
    assert meta["samples"] == SAMPLES_PER_CHUNK * N_CHUNKS


def test_e2e_synthesize_stream(rk1828_service):
    chunks = list(rk1828_service.synthesize_stream("你好世界"))
    assert len(chunks) == N_CHUNKS
    for i, (audio, meta) in enumerate(chunks):
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == SAMPLES_PER_CHUNK
        assert np.abs(audio).max() > 0.01
        assert meta["chunk_index"] == i


def test_e2e_backend_via_factory():
    """Full backend path: factory -> preload (env-driven) -> synthesize."""
    from rkvoice_stream.engine.tts import create_tts

    os.environ["RK1828_TTS_BINARY"] = MOCK_WORKER
    os.environ["RK1828_TTS_MODEL_DIR"] = "/tmp/fake-model"
    os.environ["RK1828_DEVICE_ID"] = "0001:11:00.0"
    try:
        backend = create_tts("qwen3_tts_rk1828")
        backend.preload()
        assert backend.is_ready() is True
        info = backend.runtime_info()
        assert info["device_id"] == "0001:11:00.0"

        wav_bytes, meta = backend.synthesize("hello")
        assert wav_bytes[:4] == b"RIFF"
        assert meta["samples"] == SAMPLES_PER_CHUNK * N_CHUNKS

        stream = list(backend.synthesize_stream("hello"))
        assert len(stream) == N_CHUNKS
    finally:
        backend.cleanup()
        for k in ("RK1828_TTS_BINARY", "RK1828_TTS_MODEL_DIR", "RK1828_DEVICE_ID"):
            os.environ.pop(k, None)

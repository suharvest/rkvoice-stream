"""Pytest fixtures for rk3576 speech service tests.

Dual-mode: HTTP (SERVICE_URL env) or direct backend loading.
All session-scoped; skips gracefully when service/models unavailable.
"""

from __future__ import annotations

import io
import os
import sys
import time

import pytest

# ---------------------------------------------------------------------------
# Test sentences
# ---------------------------------------------------------------------------

TEST_SENTENCES_ZH = [
    "你好世界",
    "今天天气真不错",
    "语音识别测试，一二三四五",
    "深度学习改变了语音技术",
    "欢迎使用语音服务",
]

TEST_SENTENCES_EN = [
    "Hello world",
    "The quick brown fox jumps over the lazy dog",
    "Speech recognition is amazing",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_service_url() -> str | None:
    """Return SERVICE_URL if set, else None (direct mode)."""
    url = os.environ.get("SERVICE_URL", "").strip()
    return url if url else None


def _is_http_mode() -> bool:
    return _get_service_url() is not None


def _ensure_app_on_path():
    """Add rk3576/app to sys.path so backend imports work."""
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    app_dir = os.path.abspath(app_dir)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


# ---------------------------------------------------------------------------
# HTTP-mode fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def http_session():
    """requests.Session pointed at the running service. Skips if unreachable."""
    import requests

    url = _get_service_url()
    if not url:
        pytest.skip("SERVICE_URL not set -- HTTP mode unavailable")

    session = requests.Session()
    try:
        resp = session.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except Exception as exc:
        pytest.skip(f"Service at {url} not reachable: {exc}")

    session._base_url = url  # stash for convenience
    return session


# ---------------------------------------------------------------------------
# Direct-mode fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tts_backend():
    """Directly-loaded TTS backend. Skips if models not available."""
    if _is_http_mode():
        pytest.skip("HTTP mode -- skipping direct backend load")

    _ensure_app_on_path()

    try:
        from tts_backend import create_backend
        backend = create_backend()
        backend.preload()
    except Exception as exc:
        pytest.skip(f"TTS backend not available: {exc}")

    return backend


@pytest.fixture(scope="session")
def asr_engine():
    """Directly-loaded ASR backend. Skips if models not available."""
    if _is_http_mode():
        pytest.skip("HTTP mode -- skipping direct backend load")

    _ensure_app_on_path()

    try:
        from asr_backend import create_asr_backend
        backend = create_asr_backend()
        backend.preload()
    except Exception as exc:
        pytest.skip(f"ASR backend not available: {exc}")

    return backend


# ---------------------------------------------------------------------------
# Unified callable fixtures (mode-aware, no cross-mode dependencies)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tts_fn(request):
    """Unified TTS callable: tts_fn(text) -> (wav_bytes, meta_dict).

    Uses HTTP mode if SERVICE_URL is set, otherwise direct backend.
    """
    if _is_http_mode():
        session = request.getfixturevalue("http_session")
        base_url = session._base_url

        def _tts_http(text: str) -> tuple[bytes, dict]:
            t0 = time.monotonic()
            resp = session.post(
                f"{base_url}/tts", json={"text": text}, timeout=120
            )
            elapsed = time.monotonic() - t0
            resp.raise_for_status()
            rtf_header = resp.headers.get("X-RTF")
            meta = {
                "inference_time": elapsed,
                "rtf": float(rtf_header) if rtf_header else None,
                "duration": resp.headers.get("X-Audio-Duration"),
            }
            return resp.content, meta

        return _tts_http
    else:
        backend = request.getfixturevalue("tts_backend")

        def _tts_direct(text: str) -> tuple[bytes, dict]:
            return backend.synthesize(text=text)

        return _tts_direct


@pytest.fixture(scope="session")
def asr_fn(request):
    """Unified ASR callable: asr_fn(wav_bytes, language="auto") -> str.

    Uses HTTP mode if SERVICE_URL is set, otherwise direct backend.
    """
    if _is_http_mode():
        session = request.getfixturevalue("http_session")
        base_url = session._base_url

        def _asr_http(wav_bytes: bytes, language: str = "auto") -> str:
            files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
            resp = session.post(
                f"{base_url}/asr",
                files=files,
                params={"language": language},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return (data.get("text") or data.get("transcript") or "").strip()

        return _asr_http
    else:
        backend = request.getfixturevalue("asr_engine")

        def _asr_direct(wav_bytes: bytes, language: str = "auto") -> str:
            result = backend.transcribe(wav_bytes, language=language)
            return result.text.strip()

        return _asr_direct

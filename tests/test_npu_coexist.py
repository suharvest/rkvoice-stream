"""NPU coexistence tests: RKLLM + RKNN alternating inference without crashes.

Direct-mode only (needs actual NPU hardware). Skips in HTTP mode.
"""

from __future__ import annotations

import os
import sys
import time

import pytest


def _require_direct_mode():
    """Skip if running in HTTP mode."""
    url = os.environ.get("SERVICE_URL", "").strip()
    if url:
        pytest.skip("NPU coexist tests require direct mode (no SERVICE_URL)")


def _load_backends():
    """Load TTS and ASR backends directly. Skip if unavailable."""
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    app_dir = os.path.abspath(app_dir)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    try:
        from tts_backend import create_backend
        tts = create_backend()
        tts.preload()
    except Exception as exc:
        pytest.skip(f"TTS backend not available: {exc}")
        return None, None

    try:
        from asr_backend import create_asr_backend
        asr = create_asr_backend()
        asr.preload()
    except Exception as exc:
        pytest.skip(f"ASR backend not available: {exc}")
        return None, None

    return tts, asr


def test_sequential_tts_asr():
    """TTS -> ASR -> TTS -> ASR in sequence without crash."""
    _require_direct_mode()
    tts, asr = _load_backends()

    texts = ["你好世界", "Hello world", "深度学习", "Speech test"]
    for text in texts:
        # TTS
        wav_bytes, meta = tts.synthesize(text=text)
        assert len(wav_bytes) > 44, f"TTS failed for: {text!r}"

        # ASR
        result = asr.transcribe(wav_bytes, language="auto")
        assert isinstance(result.text, str), f"ASR failed for: {text!r}"
        print(f"  {text!r} -> TTS({len(wav_bytes)}B) -> ASR({result.text!r})")


def test_repeated_cycles():
    """5 rounds of TTS+ASR without crash or resource leak."""
    _require_direct_mode()
    tts, asr = _load_backends()

    sentence = "语音识别测试"
    for cycle in range(5):
        t0 = time.monotonic()

        wav_bytes, meta = tts.synthesize(text=sentence)
        assert len(wav_bytes) > 44

        result = asr.transcribe(wav_bytes, language="Chinese")
        assert isinstance(result.text, str)

        elapsed = time.monotonic() - t0
        print(f"  Cycle {cycle + 1}/5: {elapsed:.3f}s  hyp={result.text!r}")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    """Run NPU coexistence tests directly (no pytest needed)."""
    import sys

    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    app_dir = os.path.abspath(app_dir)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    from tts_backend import create_backend
    from asr_backend import create_asr_backend

    print("Loading TTS backend...")
    tts = create_backend()
    tts.preload()
    print(f"  TTS ready: {tts.name}")

    print("Loading ASR backend...")
    asr = create_asr_backend()
    asr.preload()
    print(f"  ASR ready: {asr.name}")

    print("\n--- Sequential test ---")
    texts = ["你好世界", "Hello world", "深度学习", "Speech test"]
    for text in texts:
        wav_bytes, meta = tts.synthesize(text=text)
        result = asr.transcribe(wav_bytes, language="auto")
        print(f"  {text!r} -> {result.text!r}")

    print("\n--- Repeated cycles ---")
    for cycle in range(5):
        t0 = time.monotonic()
        wav_bytes, _meta = tts.synthesize(text="语音识别测试")
        result = asr.transcribe(wav_bytes, language="Chinese")
        elapsed = time.monotonic() - t0
        print(f"  Cycle {cycle + 1}/5: {elapsed:.3f}s  hyp={result.text!r}")

    print("\nAll NPU coexistence tests passed.")


if __name__ == "__main__":
    _main()

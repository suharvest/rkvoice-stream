"""TTS synthesis tests: valid WAV output, duration, RTF."""

from __future__ import annotations

import time

import pytest

from .metrics import compute_rtf, wav_duration_seconds


TEST_TEXT = "你好世界，今天天气不错"


def test_produces_valid_wav(tts_fn):
    """TTS must return non-empty, parseable WAV bytes."""
    wav_bytes, meta = tts_fn(TEST_TEXT)
    assert len(wav_bytes) > 44, "WAV too short (header is 44 bytes)"
    # Must start with RIFF header
    assert wav_bytes[:4] == b"RIFF", "Missing RIFF header"
    # Must be parseable
    dur = wav_duration_seconds(wav_bytes)
    assert dur > 0, "WAV has zero duration"


def test_audio_duration_reasonable(tts_fn):
    """Synthesized audio duration should be 0.5s - 30s for a short sentence."""
    wav_bytes, _meta = tts_fn(TEST_TEXT)
    dur = wav_duration_seconds(wav_bytes)
    assert 0.5 < dur < 30.0, f"Unreasonable duration: {dur:.2f}s"


def test_rtf_under_threshold(tts_fn):
    """RTF (Real-Time Factor) should be < 1.0 for acceptable performance."""
    t0 = time.monotonic()
    wav_bytes, meta = tts_fn(TEST_TEXT)
    elapsed = time.monotonic() - t0

    dur = wav_duration_seconds(wav_bytes)
    rtf = compute_rtf(elapsed, dur)
    print(f"\n  TTS: {elapsed:.3f}s inference, {dur:.3f}s audio, RTF={rtf:.3f}")

    # Use header RTF if available, else computed
    header_rtf = meta.get("rtf")
    check_rtf = float(header_rtf) if header_rtf is not None else rtf
    assert check_rtf < 1.0, f"RTF {check_rtf:.3f} >= 1.0"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="TTS smoke test")
    parser.add_argument("--host", default="localhost:8621")
    args = parser.parse_args()

    import requests

    # Allow running without package context
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from metrics import compute_rtf as _rtf, wav_duration_seconds as _dur

    base_url = f"http://{args.host}"
    session = requests.Session()

    texts = [TEST_TEXT, "Hello world", "深度学习改变了语音技术"]
    for text in texts:
        t0 = time.monotonic()
        resp = session.post(f"{base_url}/tts", json={"text": text}, timeout=120)
        elapsed = time.monotonic() - t0
        resp.raise_for_status()

        wav_bytes = resp.content
        dur = _dur(wav_bytes)
        rtf = _rtf(elapsed, dur)
        rtf_header = resp.headers.get("X-RTF", "n/a")
        print(f"  text={text!r}  wav={len(wav_bytes)}B  dur={dur:.2f}s  "
              f"elapsed={elapsed:.3f}s  RTF={rtf:.3f}  X-RTF={rtf_header}")


if __name__ == "__main__":
    _main()

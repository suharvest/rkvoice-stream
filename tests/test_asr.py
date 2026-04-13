"""ASR accuracy tests: TTS-generated audio, silence handling, EOS quality."""

from __future__ import annotations

import io
import struct
import wave

import pytest

from .metrics import cer


def _make_silence_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file (all zeros)."""
    n_frames = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


def test_asr_from_tts_audio(tts_fn, asr_fn):
    """TTS audio fed to ASR should produce CER < 0.3 for a known sentence."""
    sentence = "语音识别测试，一二三四五"
    wav_bytes, _meta = tts_fn(sentence)
    hypothesis = asr_fn(wav_bytes, language="Chinese")
    cer_val = cer(sentence, hypothesis)
    print(f"\n  ref={sentence!r}  hyp={hypothesis!r}  CER={cer_val:.1%}")
    assert cer_val < 0.3, f"CER {cer_val:.1%} >= 30%"


def test_asr_silence(asr_fn):
    """Silence input should produce empty or very short output."""
    silence = _make_silence_wav(duration_s=1.0)
    result = asr_fn(silence, language="auto")
    assert len(result) < 10, f"Expected near-empty output for silence, got: {result!r}"


def test_eos_no_garbage(tts_fn, asr_fn):
    """ASR output should not have trailing garbage after sentence punctuation."""
    sentence = "今天天气真不错。"
    wav_bytes, _meta = tts_fn(sentence)
    hypothesis = asr_fn(wav_bytes, language="Chinese")

    # After normalizing, the hypothesis should not be much longer than reference
    from .metrics import normalize_text
    ref_norm = normalize_text(sentence)
    hyp_norm = normalize_text(hypothesis)

    # Allow up to 50% extra characters as "garbage"
    max_len = int(len(ref_norm) * 1.5) + 2
    assert len(hyp_norm) <= max_len, (
        f"Possible trailing garbage: hyp has {len(hyp_norm)} chars vs "
        f"ref {len(ref_norm)} chars | hyp={hypothesis!r}"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ASR accuracy test")
    parser.add_argument("--host", default="localhost:8621")
    args = parser.parse_args()

    import requests

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from metrics import cer as cer_fn

    base_url = f"http://{args.host}"
    session = requests.Session()

    # Test 1: TTS -> ASR round-trip
    sentence = "语音识别测试，一二三四五"
    resp = session.post(f"{base_url}/tts", json={"text": sentence}, timeout=120)
    resp.raise_for_status()
    wav_bytes = resp.content

    files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    resp = session.post(f"{base_url}/asr", files=files, params={"language": "Chinese"}, timeout=120)
    resp.raise_for_status()
    hyp = (resp.json().get("text") or "").strip()
    c = cer_fn(sentence, hyp)
    print(f"  Round-trip CER={c:.1%}  ref={sentence!r}  hyp={hyp!r}")

    # Test 2: Silence
    silence = _make_silence_wav()
    files = {"file": ("silence.wav", io.BytesIO(silence), "audio/wav")}
    resp = session.post(f"{base_url}/asr", files=files, timeout=120)
    resp.raise_for_status()
    hyp = (resp.json().get("text") or "").strip()
    print(f"  Silence -> {hyp!r} (len={len(hyp)})")


if __name__ == "__main__":
    _main()

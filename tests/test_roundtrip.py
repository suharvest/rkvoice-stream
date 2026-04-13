"""Round-trip tests: text -> TTS -> ASR -> text, with CER quality gates."""

from __future__ import annotations

import pytest

from .metrics import cer

# Import test sentences from conftest
from .conftest import TEST_SENTENCES_ZH, TEST_SENTENCES_EN


class TestRoundtripZH:
    """Chinese round-trip: each sentence CER < 0.5."""

    @pytest.mark.parametrize("sentence", TEST_SENTENCES_ZH)
    def test_roundtrip_zh(self, tts_fn, asr_fn, sentence):
        wav_bytes, _meta = tts_fn(sentence)
        hypothesis = asr_fn(wav_bytes, language="Chinese")
        cer_val = cer(sentence, hypothesis)
        assert cer_val < 0.5, (
            f"CER {cer_val:.1%} >= 50% | ref={sentence!r} hyp={hypothesis!r}"
        )


class TestRoundtripEN:
    """English round-trip: each sentence CER < 0.5."""

    @pytest.mark.parametrize("sentence", TEST_SENTENCES_EN)
    def test_roundtrip_en(self, tts_fn, asr_fn, sentence):
        wav_bytes, _meta = tts_fn(sentence)
        hypothesis = asr_fn(wav_bytes, language="English")
        cer_val = cer(sentence, hypothesis)
        assert cer_val < 0.5, (
            f"CER {cer_val:.1%} >= 50% | ref={sentence!r} hyp={hypothesis!r}"
        )


def test_avg_cer_gate(tts_fn, asr_fn):
    """Average CER across all test sentences must be <= 0.5."""
    all_sentences = TEST_SENTENCES_ZH + TEST_SENTENCES_EN
    cer_values = []
    for sentence in all_sentences:
        wav_bytes, _meta = tts_fn(sentence)
        lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"
        hypothesis = asr_fn(wav_bytes, language=lang)
        cer_values.append(cer(sentence, hypothesis))

    avg_cer = sum(cer_values) / len(cer_values)
    print(f"\nAverage CER: {avg_cer:.1%} ({len(cer_values)} sentences)")
    assert avg_cer <= 0.5, f"Average CER {avg_cer:.1%} > 50%"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys
    import time

    # Allow running without package context
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from metrics import cer as cer_fn

    parser = argparse.ArgumentParser(description="Round-trip TTS->ASR test")
    parser.add_argument("--host", default="localhost:8621")
    args = parser.parse_args()

    import requests

    base_url = f"http://{args.host}"
    session = requests.Session()

    sentences = TEST_SENTENCES_ZH + TEST_SENTENCES_EN
    cer_values = []

    for i, sentence in enumerate(sentences, 1):
        # TTS
        resp = session.post(f"{base_url}/tts", json={"text": sentence}, timeout=120)
        resp.raise_for_status()
        wav_bytes = resp.content

        # ASR
        import io
        files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
        lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"
        resp = session.post(f"{base_url}/asr", files=files, params={"language": lang}, timeout=120)
        resp.raise_for_status()
        hyp = (resp.json().get("text") or "").strip()

        c = cer_fn(sentence, hyp)
        cer_values.append(c)
        status = "PASS" if c < 0.5 else "FAIL"
        print(f"  [{status}] {i:>2}. CER={c:.1%}  ref={sentence!r}  hyp={hyp!r}")

    avg = sum(cer_values) / len(cer_values)
    print(f"\nAverage CER: {avg:.1%}")
    if avg > 0.5:
        print("QUALITY GATE FAILED")
        sys.exit(1)
    else:
        print("Quality gate passed")


if __name__ == "__main__":
    _main()

"""Latency benchmarks: TTS, ASR, and end-to-end V2V timing."""

from __future__ import annotations

import time

import pytest

from .conftest import TEST_SENTENCES_ZH, TEST_SENTENCES_EN
from .metrics import compute_rtf, wav_duration_seconds


def test_tts_latency(tts_fn):
    """Measure and print TTS latency per sentence."""
    sentences = TEST_SENTENCES_ZH[:3]
    print("\n  TTS Latency:")
    for sentence in sentences:
        t0 = time.monotonic()
        wav_bytes, meta = tts_fn(sentence)
        elapsed = time.monotonic() - t0
        dur = wav_duration_seconds(wav_bytes)
        rtf = compute_rtf(elapsed, dur)
        print(f"    {sentence[:30]:<30}  {elapsed:.3f}s  audio={dur:.2f}s  RTF={rtf:.3f}")


def test_asr_latency(tts_fn, asr_fn):
    """Measure and print ASR latency on TTS-generated audio."""
    sentences = TEST_SENTENCES_ZH[:3]
    print("\n  ASR Latency:")
    for sentence in sentences:
        wav_bytes, _meta = tts_fn(sentence)
        dur = wav_duration_seconds(wav_bytes)

        t0 = time.monotonic()
        _hyp = asr_fn(wav_bytes, language="Chinese")
        elapsed = time.monotonic() - t0

        rtf = compute_rtf(elapsed, dur)
        print(f"    {sentence[:30]:<30}  {elapsed:.3f}s  audio={dur:.2f}s  RTF={rtf:.3f}")


def test_v2v_latency(tts_fn, asr_fn):
    """End-to-end V2V latency with markdown table output."""
    all_sentences = TEST_SENTENCES_ZH + TEST_SENTENCES_EN
    rows = []

    for sentence in all_sentences:
        # TTS
        t0 = time.monotonic()
        wav_bytes, meta = tts_fn(sentence)
        tts_elapsed = time.monotonic() - t0
        dur = wav_duration_seconds(wav_bytes)
        tts_rtf = meta.get("rtf") or compute_rtf(tts_elapsed, dur)

        # ASR
        lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"
        t0 = time.monotonic()
        hyp = asr_fn(wav_bytes, language=lang)
        asr_elapsed = time.monotonic() - t0
        asr_rtf = compute_rtf(asr_elapsed, dur)

        v2v = tts_elapsed + asr_elapsed
        rows.append((sentence, tts_elapsed, asr_elapsed, v2v, tts_rtf, asr_rtf))

    # Print markdown table
    print("\n| Sentence | TTS (s) | ASR (s) | V2V (s) | TTS RTF | ASR RTF |")
    print("|----------|---------|---------|---------|---------|---------|")
    for sentence, tts_t, asr_t, v2v_t, t_rtf, a_rtf in rows:
        t_rtf_s = f"{float(t_rtf):.3f}" if t_rtf is not None else "n/a"
        print(f"| {sentence[:25]:<25} | {tts_t:.3f} | {asr_t:.3f} | {v2v_t:.3f} "
              f"| {t_rtf_s} | {a_rtf:.3f} |")

    avg_v2v = sum(r[3] for r in rows) / len(rows)
    print(f"\n  Average V2V: {avg_v2v:.3f}s")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import io
    import sys

    parser = argparse.ArgumentParser(description="V2V latency benchmark")
    parser.add_argument("--host", default="localhost:8621")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    import requests

    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from metrics import compute_rtf as _rtf, wav_duration_seconds as _dur

    base_url = f"http://{args.host}"
    session = requests.Session()
    sentences = TEST_SENTENCES_ZH + TEST_SENTENCES_EN

    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {run + 1}/{args.repeat} ---")

        print("\n| Sentence | TTS (s) | ASR (s) | V2V (s) |")
        print("|----------|---------|---------|---------|")

        for sentence in sentences:
            t0 = time.monotonic()
            resp = session.post(f"{base_url}/tts", json={"text": sentence}, timeout=120)
            tts_t = time.monotonic() - t0
            resp.raise_for_status()
            wav_bytes = resp.content

            files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
            lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"
            t0 = time.monotonic()
            resp = session.post(f"{base_url}/asr", files=files,
                                params={"language": lang}, timeout=120)
            asr_t = time.monotonic() - t0
            resp.raise_for_status()

            v2v = tts_t + asr_t
            print(f"| {sentence[:25]:<25} | {tts_t:.3f} | {asr_t:.3f} | {v2v:.3f} |")


if __name__ == "__main__":
    _main()

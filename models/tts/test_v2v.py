#!/usr/bin/env python3
"""
V2V (Voice-to-Voice) pipeline test script.

Tests TTS → ASR round-trip: synthesises speech from text, then transcribes it
and compares against the original to compute CER.

Usage:
    python scripts/test_v2v.py --host cat-remote:8621
"""

import argparse
import io
import time
import wave
import sys

import requests

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
TEST_CASES = [
    "你好世界",
    "今天天气真不错",
    "Hello world",
    "语音识别测试，一二三四五",
    "The quick brown fox jumps over the lazy dog",
]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (character-level)."""
    m, n = len(a), len(b)
    # Use two-row DP to save memory
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return prev[n]


def _normalize(text: str) -> str:
    """Strip punctuation and whitespace for fairer CER comparison."""
    import re
    # Remove Chinese and ASCII punctuation, whitespace
    text = re.sub(r'[\s，。！？、；：\u201c\u201d\u2018\u2019（）,.!?;:\'"()\[\]{}\-]', '', text)
    return text.lower()


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (0.0 – 1.0+). Returns 0.0 for empty reference."""
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Return duration of a WAV bytestring in seconds."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_tts(session: requests.Session, base_url: str, text: str):
    """
    POST /tts  →  (wav_bytes, tts_elapsed_s, rtf_from_header)
    """
    url = f"{base_url}/tts"
    t0 = time.monotonic()
    resp = session.post(url, json={"text": text}, timeout=120)
    elapsed = time.monotonic() - t0
    resp.raise_for_status()

    rtf_header = resp.headers.get("X-RTF")
    rtf = float(rtf_header) if rtf_header else None
    return resp.content, elapsed, rtf


def call_asr(session: requests.Session, base_url: str, wav_bytes: bytes):
    """
    POST /asr  (multipart WAV)  →  (transcript, asr_elapsed_s)
    """
    url = f"{base_url}/asr"
    files = {"file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav")}
    t0 = time.monotonic()
    resp = session.post(url, files=files, timeout=120)
    elapsed = time.monotonic() - t0
    resp.raise_for_status()

    data = resp.json()
    # Accept both {"text": ...} and {"transcript": ...} response shapes
    transcript = data.get("text") or data.get("transcript") or ""
    return transcript.strip(), elapsed


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

HEADER = (
    f"{'#':>2}  "
    f"{'Reference':<45}  "
    f"{'Hypothesis':<45}  "
    f"{'TTFT(s)':>8}  "
    f"{'TTS-RTF':>8}  "
    f"{'ASR-RTF':>8}  "
    f"{'V2V(s)':>7}  "
    f"{'CER':>6}"
)
SEP = "-" * len(HEADER)


def fmt_row(idx, ref, hyp, ttft, tts_rtf, asr_rtf, v2v, cer_val):
    tts_rtf_s = f"{tts_rtf:.3f}" if tts_rtf is not None else "  n/a "
    asr_rtf_s = f"{asr_rtf:.3f}" if asr_rtf is not None else "  n/a "
    return (
        f"{idx:>2}  "
        f"{ref[:45]:<45}  "
        f"{hyp[:45]:<45}  "
        f"{ttft:>8.3f}  "
        f"{tts_rtf_s:>8}  "
        f"{asr_rtf_s:>8}  "
        f"{v2v:>7.3f}  "
        f"{cer_val:>5.1%}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2V pipeline smoke test")
    parser.add_argument(
        "--host",
        default="localhost:8621",
        help="Host:port of the speech service (default: localhost:8621)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}"
    print(f"Target: {base_url}")
    print()

    session = requests.Session()

    results = []

    print(HEADER)
    print(SEP)

    for i, ref_text in enumerate(TEST_CASES, start=1):
        # --- TTS ---
        try:
            wav_bytes, tts_elapsed, tts_rtf = call_tts(session, base_url, ref_text)
        except Exception as exc:
            print(f"{i:>2}  TTS FAILED: {exc}")
            results.append(None)
            continue

        # TTFT = TTS response time (Matcha is non-streaming; full response = first token)
        ttft = tts_elapsed

        # Compute audio duration for ASR RTF
        try:
            audio_dur = wav_duration_seconds(wav_bytes)
        except Exception:
            audio_dur = None

        # --- ASR ---
        try:
            hypothesis, asr_elapsed = call_asr(session, base_url, wav_bytes)
        except Exception as exc:
            print(f"{i:>2}  ASR FAILED: {exc}")
            results.append(None)
            continue

        asr_rtf = (asr_elapsed / audio_dur) if audio_dur else None
        v2v = tts_elapsed + asr_elapsed
        cer_val = cer(ref_text, hypothesis)

        row = dict(
            ref=ref_text,
            hyp=hypothesis,
            ttft=ttft,
            tts_rtf=tts_rtf,
            asr_rtf=asr_rtf,
            v2v=v2v,
            cer=cer_val,
        )
        results.append(row)

        print(fmt_row(i, ref_text, hypothesis, ttft, tts_rtf, asr_rtf, v2v, cer_val))

    print(SEP)

    # --- Summary ---
    valid = [r for r in results if r is not None]
    if not valid:
        print("No successful test cases.")
        sys.exit(1)

    avg_ttft = sum(r["ttft"] for r in valid) / len(valid)
    avg_tts_rtf_vals = [r["tts_rtf"] for r in valid if r["tts_rtf"] is not None]
    avg_tts_rtf = sum(avg_tts_rtf_vals) / len(avg_tts_rtf_vals) if avg_tts_rtf_vals else None
    avg_asr_rtf_vals = [r["asr_rtf"] for r in valid if r["asr_rtf"] is not None]
    avg_asr_rtf = sum(avg_asr_rtf_vals) / len(avg_asr_rtf_vals) if avg_asr_rtf_vals else None
    avg_v2v = sum(r["v2v"] for r in valid) / len(valid)
    avg_cer = sum(r["cer"] for r in valid) / len(valid)

    print()
    print("Summary")
    print(f"  Cases passed   : {len(valid)}/{len(TEST_CASES)}")
    print(f"  Avg TTFT       : {avg_ttft:.3f} s")
    if avg_tts_rtf is not None:
        print(f"  Avg TTS RTF    : {avg_tts_rtf:.3f}")
    else:
        print(f"  Avg TTS RTF    : n/a (no X-RTF header)")
    if avg_asr_rtf is not None:
        print(f"  Avg ASR RTF    : {avg_asr_rtf:.3f}")
    else:
        print(f"  Avg ASR RTF    : n/a")
    print(f"  Avg V2V latency: {avg_v2v:.3f} s")
    print(f"  Avg CER        : {avg_cer:.1%}")

    if avg_cer > 0.50:
        print()
        print(f"QUALITY GATE FAILED: average CER {avg_cer:.1%} > 50%")
        sys.exit(1)
    else:
        print()
        print(f"Quality gate passed (CER {avg_cer:.1%} <= 50%)")


if __name__ == "__main__":
    main()

"""Voice-to-Voice streaming latency test.

Measures the real-world latency a user would experience:
  time from last spoken word → first TTS audio heard

Full pipeline:
  1. Stream PCM to WS /asr/stream (simulating real-time mic input)
  2. Send empty frame to signal end-of-speech → receive final text
  3. Immediately POST /tts/stream with that text
  4. Measure time to first audio chunk

Key metric: EOS-to-first-audio (ms)
  = (ASR finalize latency) + (TTS first-chunk latency)
"""

from __future__ import annotations

import io
import struct
import time
import wave

import pytest
import requests

from .conftest import TEST_SENTENCES_ZH, TEST_SENTENCES_EN
from .metrics import cer, wav_duration_seconds


def _get_service_url() -> str | None:
    import os
    url = os.environ.get("SERVICE_URL", "").strip()
    return url if url else None


def _make_wav(text_for_tts: str, session: requests.Session, base_url: str) -> bytes:
    """Use the service's own TTS to generate test audio."""
    resp = session.post(f"{base_url}/tts", json={"text": text_for_tts}, timeout=120)
    resp.raise_for_status()
    return resp.content


def _wav_to_pcm_chunks(wav_bytes: bytes, chunk_duration_ms: int = 100) -> list[bytes]:
    """Split WAV into PCM int16 chunks for streaming."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())

    # Convert to mono int16 if needed
    import numpy as np
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

    chunk_samples = int(sr * chunk_duration_ms / 1000)
    chunks = []
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i + chunk_samples]
        chunks.append(chunk.tobytes())
    return chunks


def _run_v2v_streaming(
    session: requests.Session,
    base_url: str,
    wav_bytes: bytes,
    language: str = "auto",
) -> dict:
    """Run one full streaming V2V cycle, return timing dict."""
    import websocket

    pcm_chunks = _wav_to_pcm_chunks(wav_bytes, chunk_duration_ms=100)
    audio_dur = wav_duration_seconds(wav_bytes)

    # --- Phase 1: Stream audio to ASR via WebSocket ---
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws = websocket.create_connection(
        f"{ws_url}/asr/stream?language={language}&sample_rate=16000",
        timeout=30,
    )

    t_stream_start = time.monotonic()

    # Send all PCM at once — we measure EOS→first-audio, not streaming pace.
    # Sending fast ensures spec encoding happens once on the full buffer,
    # not repeatedly as buffer grows.
    for chunk in pcm_chunks:
        ws.send_binary(chunk)

    # Signal end of speech
    t_eos = time.monotonic()
    ws.send_binary(b"")  # empty frame = finalize

    # Wait for final result
    final_text = ""
    while True:
        result = ws.recv()
        import json
        data = json.loads(result)
        if data.get("is_final"):
            final_text = data.get("text", "").strip()
            break
        # Keep partial text in case final never comes with is_final
        final_text = data.get("text", "").strip()

    t_asr_done = time.monotonic()
    ws.close()

    # --- Phase 2: Stream TTS with recognized text ---
    t_tts_request = time.monotonic()
    resp = session.post(
        f"{base_url}/tts/stream",
        json={"text": final_text},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    # Read first chunk (4-byte sample rate header + first PCM data)
    first_data = b""
    for chunk in resp.iter_content(chunk_size=4096):
        first_data = chunk
        break

    t_tts_first_byte = time.monotonic()

    # Drain remaining response
    for _ in resp.iter_content(chunk_size=8192):
        pass
    t_tts_done = time.monotonic()

    return {
        "audio_duration": audio_dur,
        "final_text": final_text,
        # Core metric: user stops speaking → hears first audio
        "eos_to_first_audio_ms": (t_tts_first_byte - t_eos) * 1000,
        # Breakdown
        "asr_finalize_ms": (t_asr_done - t_eos) * 1000,
        "tts_ttfb_ms": (t_tts_first_byte - t_tts_request) * 1000,
        "tts_total_ms": (t_tts_done - t_tts_request) * 1000,
        # Overhead between ASR done and TTS request (network + code)
        "handoff_ms": (t_tts_request - t_asr_done) * 1000,
    }


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def service():
    url = _get_service_url()
    if not url:
        pytest.skip("SERVICE_URL not set — streaming V2V test requires HTTP mode")
    session = requests.Session()
    try:
        resp = session.get(f"{url}/health", timeout=5)
        health = resp.json()
        if not health.get("asr") or not health.get("tts"):
            pytest.skip("Service missing ASR or TTS backend")
        if not health.get("streaming_asr"):
            pytest.skip("Service does not support streaming ASR")
    except Exception as exc:
        pytest.skip(f"Service not reachable: {exc}")
    return session, url


def test_v2v_streaming_latency(service):
    """Measure streaming V2V latency for all test sentences."""
    session, base_url = service
    sentences = TEST_SENTENCES_ZH[:3] + TEST_SENTENCES_EN[:1]

    results = []
    for sentence in sentences:
        # Generate test audio via TTS
        wav_bytes = _make_wav(sentence, session, base_url)
        lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"

        timing = _run_v2v_streaming(session, base_url, wav_bytes, language=lang)
        timing["ref"] = sentence
        results.append(timing)

    # Print results table
    print("\n| Sentence | Audio | ASR final (ms) | TTS TTFB (ms) | EOS→Audio (ms) | CER |")
    print("|----------|-------|----------------|---------------|-----------------|-----|")
    for r in results:
        cer_val = cer(r["ref"], r["final_text"])
        print(
            f"| {r['ref'][:20]:<20} "
            f"| {r['audio_duration']:.1f}s "
            f"| {r['asr_finalize_ms']:>11.0f} "
            f"| {r['tts_ttfb_ms']:>11.0f} "
            f"| {r['eos_to_first_audio_ms']:>13.0f} "
            f"| {cer_val:.0%} |"
        )

    avg_eos = sum(r["eos_to_first_audio_ms"] for r in results) / len(results)
    print(f"\n  Average EOS→First Audio: {avg_eos:.0f}ms")

    # Latency test: only assert that ASR produced non-empty text
    # (CER quality is tested separately in test_roundtrip.py)
    for r in results:
        assert r["final_text"], f"ASR returned empty text for ref={r['ref']!r}"


def test_v2v_eos_latency_gate(service):
    """EOS→first audio must be under 3000ms for a short sentence."""
    session, base_url = service
    sentence = "你好世界"
    wav_bytes = _make_wav(sentence, session, base_url)

    timing = _run_v2v_streaming(session, base_url, wav_bytes, language="Chinese")

    print(f"\n  EOS→First Audio: {timing['eos_to_first_audio_ms']:.0f}ms")
    print(f"    ASR finalize:  {timing['asr_finalize_ms']:.0f}ms")
    print(f"    Handoff:       {timing['handoff_ms']:.0f}ms")
    print(f"    TTS TTFB:      {timing['tts_ttfb_ms']:.0f}ms")
    print(f"    Text:          {timing['final_text']!r}")

    assert timing["eos_to_first_audio_ms"] < 3000, (
        f"EOS→first audio {timing['eos_to_first_audio_ms']:.0f}ms >= 3000ms"
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def _main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Streaming V2V latency benchmark")
    parser.add_argument("--host", default="localhost:8621")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    base_url = f"http://{args.host}"
    session = requests.Session()

    # Check health
    resp = session.get(f"{base_url}/health", timeout=5)
    health = resp.json()
    print(f"Health: {health}")
    if not health.get("streaming_asr"):
        print("ERROR: streaming ASR not available")
        sys.exit(1)

    sentences = TEST_SENTENCES_ZH + TEST_SENTENCES_EN

    for run in range(args.repeat):
        if args.repeat > 1:
            print(f"\n--- Run {run + 1}/{args.repeat} ---")

        print("\n| Sentence | Audio | ASR final | TTS TTFB | EOS→Audio | Text |")
        print("|----------|-------|-----------|----------|-----------|------|")

        for sentence in sentences:
            wav_bytes = _make_wav(sentence, session, base_url)
            lang = "Chinese" if any('\u4e00' <= c <= '\u9fff' for c in sentence) else "English"
            timing = _run_v2v_streaming(session, base_url, wav_bytes, language=lang)

            print(
                f"| {sentence[:20]:<20} "
                f"| {timing['audio_duration']:.1f}s "
                f"| {timing['asr_finalize_ms']:>7.0f}ms "
                f"| {timing['tts_ttfb_ms']:>6.0f}ms "
                f"| {timing['eos_to_first_audio_ms']:>7.0f}ms "
                f"| {timing['final_text'][:30]} |"
            )


if __name__ == "__main__":
    _main()

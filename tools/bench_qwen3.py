#!/usr/bin/env python3
"""TTS Baseline Benchmark Tool for rkvoice-stream.

Measures TTS performance (latency, RTF, memory) and verifies ASR round-trip
accuracy. Outputs one JSONL record per trial.

Usage:
  uv run python tools/bench_qwen3.py \
    --service-url http://localhost:8621 \
    --text-id short_zh --language zh_CN \
    --repeat 3 \
    --output-jsonl baseline/results.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import subprocess
import threading
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests  # type: ignore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MEM_SAMPLER_INTERVAL_S = 0.1  # 100ms sampling interval
CONTINUOUS_AUDIO_THRESHOLD_BYTES = 16000  # 0.5s @ 16kHz int16 (0.5 * 16000 * 2)


# ---------------------------------------------------------------------------
# Memory Sampler (background thread)
# ---------------------------------------------------------------------------
class MemorySampler:
    """Samples /proc/meminfo MemAvailable every interval_s seconds."""

    def __init__(self, interval_s: float = MEM_SAMPLER_INTERVAL_S):
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            self.samples.append(int(line.split()[1]))
                            break
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int | None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self.samples:
            return min(self.samples)
        return None

    @property
    def min_kb(self) -> int | None:
        return min(self.samples) if self.samples else None

    @property
    def count(self) -> int:
        return len(self.samples)


# ---------------------------------------------------------------------------
# Audio duration via ffprobe
# ---------------------------------------------------------------------------
def audio_duration_ffprobe(filepath: str | Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(filepath)],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except Exception:
        return 0.0


def pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 PCM in a WAV container (1 channel, 16-bit)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TTS request (non-streaming)
# ---------------------------------------------------------------------------
def tts_call(
    service_url: str,
    text: str,
    language: str = "zh_CN",
    sid: int = 0,
    speed: float | None = None,
) -> tuple[bytes, float, float, dict[str, str]]:
    """Call POST /tts, return (wav_bytes, total_seconds, server_inference_s, headers_dict)."""
    url = f"{service_url.rstrip('/')}/tts"
    payload: dict[str, Any] = {"text": text, "language": language, "sid": sid}
    if speed is not None:
        payload["speed"] = speed

    t0 = time.perf_counter()
    timeout_s = int(os.environ.get("BENCH_TTS_TIMEOUT", "300"))
    resp = requests.post(url, json=payload, timeout=timeout_s, proxies={"http": None, "https": None})
    t1 = time.perf_counter()

    resp.raise_for_status()
    total_s = t1 - t0
    server_inf = float(resp.headers.get("x-inference-time", 0))
    return resp.content, total_s, server_inf, dict(resp.headers)


def tts_stream_call(
    service_url: str,
    text: str,
    language: str = "zh_CN",
    sid: int = 0,
    speed: float | None = None,
) -> dict[str, Any]:
    """Call POST /tts/stream, return dict with byte-level timing, full PCM, and sample_rate.

    Response format: first 4 bytes = sample_rate (uint32 LE), then raw int16 PCM chunks.
    Returns a dict with keys:
      - pcm_bytes: full PCM payload (bytes without the 4-byte header)
      - sample_rate: extracted from the header
      - total_s: wall-clock duration from POST to last byte
      - time_to_header_s: seconds from POST to first header byte
      - time_to_first_pcm_s: seconds from POST to first PCM byte
      - time_to_continuous_s: seconds from POST to accumulating 0.5s of audio
      - headers: response headers dict
    """
    url = f"{service_url.rstrip('/')}/tts/stream"
    payload: dict[str, Any] = {"text": text, "language": language, "sid": sid}
    if speed is not None:
        payload["speed"] = speed

    t0 = time.perf_counter()
    resp = requests.post(url, json=payload, stream=True, timeout=120, proxies={"http": None, "https": None})
    resp.raise_for_status()

    header_bytes = b""
    pcm_data = bytearray()
    t_header: float | None = None
    t_first_pcm: float | None = None
    t_continuous: float | None = None
    sample_rate: int = 0

    # Read response in 4KB chunks; the first chunk(s) contain the 4-byte header
    # followed by int16 PCM data.
    for chunk in resp.iter_content(chunk_size=4096):
        if not chunk:
            continue

        if t_header is None:
            t_header = time.perf_counter()

        # If we haven't finished reading the header (first 4 bytes)
        if len(header_bytes) < 4:
            needed = 4 - len(header_bytes)
            header_bytes += chunk[:needed]
            if len(header_bytes) == 4:
                sample_rate = struct.unpack("<I", bytes(header_bytes))[0]
            # The rest of this chunk (if any) is PCM
            pcm_part = chunk[needed:]
            if pcm_part:
                if t_first_pcm is None:
                    t_first_pcm = time.perf_counter()
                pcm_data.extend(pcm_part)
                if t_continuous is None and len(pcm_data) >= CONTINUOUS_AUDIO_THRESHOLD_BYTES:
                    t_continuous = time.perf_counter()
            continue

        # Header already consumed; this entire chunk is PCM
        if t_first_pcm is None:
            t_first_pcm = time.perf_counter()

        pcm_data.extend(chunk)
        if t_continuous is None and len(pcm_data) >= CONTINUOUS_AUDIO_THRESHOLD_BYTES:
            t_continuous = time.perf_counter()

    t_end = time.perf_counter()

    return {
        "pcm_bytes": bytes(pcm_data),
        "sample_rate": sample_rate,
        "total_s": t_end - t0,
        "time_to_header_s": (t_header - t0) if t_header else None,
        "time_to_first_pcm_s": (t_first_pcm - t0) if t_first_pcm else None,
        "time_to_continuous_s": (t_continuous - t0) if t_continuous else None,
        "headers": dict(resp.headers),
    }


# ---------------------------------------------------------------------------
# ASR round-trip
# ---------------------------------------------------------------------------
def asr_call(
    service_url: str,
    wav_bytes: bytes,
    language: str = "auto",
    filename: str = "audio.wav",
) -> dict[str, Any]:
    """Call POST /asr with multipart file upload, return parsed JSON."""
    url = f"{service_url.rstrip('/')}/asr"
    resp = requests.post(
        url,
        params={"language": language},
        files={"file": (filename, wav_bytes, "audio/wav")},
        timeout=60,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Run a single trial
# ---------------------------------------------------------------------------
def run_trial(
    service_url: str,
    text: str,
    language: str,
    text_id: str,
    cp_groups: str,
    vocoder_profile: str,
    first_chunk_frames: int,
    chunk_frames: int,
    sampler: MemorySampler | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    """Execute one TTS+ASR trial, return measurement record."""
    timestamp = datetime.now(timezone.utc).isoformat()
    mode = "streaming" if streaming else "http"
    record: dict[str, Any] = {
        "mode": mode,
        "text_id": text_id,
        "language": language,
        "input_chars": len(text),
        "input_tokens": len(text),  # rough estimate: chars ~= tokens for CJK, ~= chars*0.25 for English
        "active_cp_groups": cp_groups,
        "vocoder_profile": vocoder_profile,
        "first_chunk_frames": first_chunk_frames,
        "chunk_frames": chunk_frames,
        "timestamp": timestamp,
    }

    # TTS call
    try:
        if streaming:
            # ── streaming /tts/stream ──
            result = tts_stream_call(service_url, text, language)

            record["total_inference_ms"] = round(result["total_s"] * 1000, 2)
            record["time_to_header_ms"] = (
                round(result["time_to_header_s"] * 1000, 2)
                if result["time_to_header_s"] is not None
                else None
            )
            record["time_to_first_pcm_ms"] = (
                round(result["time_to_first_pcm_s"] * 1000, 2)
                if result["time_to_first_pcm_s"] is not None
                else None
            )
            record["time_to_continuous_audio_ms"] = (
                round(result["time_to_continuous_s"] * 1000, 2)
                if result["time_to_continuous_s"] is not None
                else None
            )
            record["sample_rate"] = result["sample_rate"]
            # Server-side metrics are not available in streaming mode
            record["server_inference_ms"] = None
            record["server_rtf"] = None

            # Convert raw PCM to WAV for duration measurement and ASR
            wav_bytes = pcm_to_wav_bytes(result["pcm_bytes"], result["sample_rate"])
            headers = result["headers"]
        else:
            # ── non-streaming /tts ──
            wav_bytes, total_s, server_inf_s, headers = tts_call(
                service_url, text, language
            )
            record["total_inference_ms"] = round(total_s * 1000, 2)
            record["server_inference_ms"] = round(server_inf_s * 1000, 2)
            record["server_rtf"] = float(headers.get("x-rtf", 0))
            # Non-streaming: no header timing
            record["time_to_header_ms"] = None
            record["sample_rate"] = None
            # Approximate time_to_first_pcm from server inference (non-streaming)
            record["time_to_first_pcm_ms"] = record["server_inference_ms"]
            record["time_to_continuous_audio_ms"] = record["total_inference_ms"]

        # Save audio to temp and measure duration (common path)
        tmp_path = f"/tmp/bench_{uuid.uuid4().hex[:8]}.wav"
        with open(tmp_path, "wb") as f:
            f.write(wav_bytes)
        audio_dur = audio_duration_ffprobe(tmp_path)
        record["audio_duration_s"] = round(audio_dur, 3)
        if audio_dur > 0:
            record["rtf"] = round(record["total_inference_ms"] / (audio_dur * 1000), 6)
        else:
            record["rtf"] = 0.0

        # ASR round-trip (common path)
        try:
            asr_result = asr_call(service_url, wav_bytes)
            record["asr_text"] = asr_result.get("text", "")
            record["asr_exact"] = record["asr_text"] == text
            record["asr_backend"] = asr_result.get("backend", "")
        except Exception as e:
            record["asr_text"] = f"ASR_ERROR: {e}"
            record["asr_exact"] = False
            record["asr_backend"] = ""

        # Clean up temp
        try:
            Path(tmp_path).unlink()
        except Exception:
            pass

    except Exception as e:
        record["error"] = str(e)
        record["total_inference_ms"] = 0
        record["audio_duration_s"] = 0
        record["rtf"] = 0
        record["time_to_header_ms"] = None
        record["time_to_first_pcm_ms"] = 0
        record["time_to_continuous_audio_ms"] = 0
        record["asr_text"] = ""
        record["asr_exact"] = False

    return record


# ---------------------------------------------------------------------------
# Predefined test sentences
# ---------------------------------------------------------------------------
TEST_SENTENCES: dict[str, tuple[str, str]] = {
    "short_zh": ("请关闭卧室的空调。", "zh_CN"),
    "long_zh": ("今天我们继续验证低延迟流式生成的效果，请保持语义完整并避免重复。", "zh_CN"),
    "en": ("Please turn off the bedroom air conditioner.", "en_US"),
    "ja": ("寝室のエアコンを消してください。", "ja_JP"),
    "ko": ("침실 에어컨을 꺼주세요.", "ko_KR"),
}


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="rkvoice-stream TTS Benchmark Tool"
    )
    parser.add_argument("--service-url", default="http://localhost:8621",
                        help="TTS service base URL")
    parser.add_argument("--text-id", default="short_zh",
                        help="Predefined text ID or custom text")
    parser.add_argument("--language", default="zh_CN",
                        help="Language code (zh_CN, en_US, ja_JP, ko_KR)")
    parser.add_argument("--cp-groups", default="auto",
                        help="NPU core group count (ignored for HTTP bench)")
    parser.add_argument("--vocoder-profile", default="default",
                        help="Vocoder profile (ignored for HTTP bench)")
    parser.add_argument("--first-chunk-frames", type=int, default=0,
                        help="First chunk frames (streaming)")
    parser.add_argument("--chunk-frames", type=int, default=0,
                        help="Subsequent chunk frames (streaming)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Number of repetitions per test")
    parser.add_argument("--output-jsonl", default="baseline/results.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--streaming", action="store_true",
                        help="Use /tts/stream endpoint for byte-level timing")
    parser.add_argument("--no-memory-sampler", action="store_true",
                        help="Disable memory sampling")
    parser.add_argument("--device", default="unknown",
                        help="Device identifier")
    args = parser.parse_args()

    # Resolve text
    if args.text_id in TEST_SENTENCES:
        text, lang = TEST_SENTENCES[args.text_id]
        if not args.language or args.language == parser.get_default("language"):
            language = lang
        else:
            language = args.language
    else:
        text = args.text_id  # treat as literal text
        language = args.language

    # Ensure output directory
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate a unique run_id for this batch
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Start memory sampler
    sampler: MemorySampler | None = None
    if not args.no_memory_sampler:
        sampler = MemorySampler()
        sampler.start()

    results_written = 0
    for i in range(args.repeat):
        trial_id = f"{args.text_id}_r{i + 1}"
        record = run_trial(
            service_url=args.service_url,
            text=text,
            language=language,
            text_id=args.text_id,
            cp_groups=args.cp_groups,
            vocoder_profile=args.vocoder_profile,
            first_chunk_frames=args.first_chunk_frames,
            chunk_frames=args.chunk_frames,
            sampler=sampler,
            streaming=args.streaming,
        )

        # Enrich with run metadata
        record["device"] = args.device
        record["run_id"] = run_id
        record["trial_id"] = trial_id

        # Memory stats (from sampler running during this batch)
        if sampler and sampler.samples:
            record["mem_available_min_mb"] = round(sampler.min_kb / 1024, 1) if sampler.min_kb else None

        # Chunks/frames estimates (non-streaming = single chunk)
        record["chunks"] = 1
        record["frames"] = int(record.get("audio_duration_s", 0) * 16000 / 256)  # 16kHz, 256 hop

        # Write JSONL line
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        results_written += 1

    # Stop memory sampler
    if sampler:
        sampler.stop()
        if sampler.samples:
            print(f"Memory sampler: {sampler.count} samples, "
                  f"min={sampler.min_kb} KB "
                  f"({sampler.min_kb / 1024:.1f} MB)" if sampler.min_kb else "")

    print(f"Done. Wrote {results_written} records to {output_path}")


if __name__ == "__main__":
    main()

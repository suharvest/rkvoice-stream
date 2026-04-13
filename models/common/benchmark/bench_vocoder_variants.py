#!/usr/bin/env python3
"""Benchmark vocoder RKNN variants on RK3576.

Usage (on cat-remote):
    /home/cat/rknn-venv/bin/python bench_vocoder_variants.py
"""

import os
import time
import numpy as np
from rknnlite.api import RKNNLite

MODEL_DIR = "/home/cat/qwen3-tts-rknn"

# (display_name, rknn_file, input_shape, context_frames, chunk_frames)
VARIANTS = [
    ("ctx50_fp16 (baseline)", "tokenizer12hz_decode_stream_nosin.rknn", (1, 512, 75), 50, 25),
    ("ctx25_fp16", "decoder_ctx25_fp16.rknn", (1, 512, 50), 25, 25),
    ("ctx25_int8", "decoder_ctx25_int8.rknn", (1, 512, 50), 25, 25),
    ("ctx0_fp16", "decoder_ctx0_fp16.rknn", (1, 512, 25), 0, 25),
    ("ctx0_int8", "decoder_ctx0_int8.rknn", (1, 512, 25), 0, 25),
]

WARMUP = 3
ITERS = 10
SAMPLES_PER_FRAME = 1920  # upsample rate
SAMPLE_RATE = 24000


def benchmark_variant(name, model_file, input_shape, ctx_frames, chunk_frames):
    model_path = os.path.join(MODEL_DIR, model_file)
    if not os.path.exists(model_path):
        return None

    rknn_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print("  FAIL: load_rknn returned %d" % ret)
        return None

    ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1)
    if ret != 0:
        print("  FAIL: init_runtime returned %d" % ret)
        rknn.release()
        return None

    inputs = [np.random.randn(*input_shape).astype(np.float32)]

    # Warmup
    for _ in range(WARMUP):
        rknn.inference(inputs=inputs)

    # Benchmark
    times = []
    for _ in range(ITERS):
        t0 = time.time()
        rknn.inference(inputs=inputs)
        times.append((time.time() - t0) * 1000)

    rknn.release()

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    min_ms = np.min(times)

    # Audio duration = chunk_frames * samples_per_frame / sample_rate
    audio_sec = chunk_frames * SAMPLES_PER_FRAME / SAMPLE_RATE
    rtf = (avg_ms / 1000.0) / audio_sec if audio_sec > 0 else 999.0

    return {
        "name": name,
        "rknn_mb": rknn_size_mb,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "min_ms": min_ms,
        "audio_sec": audio_sec,
        "rtf": rtf,
        "input_shape": input_shape,
        "ctx": ctx_frames,
        "chunk": chunk_frames,
    }


def main():
    print("=" * 80)
    print("Vocoder RKNN Benchmark (RK3576 NPU dual-core)")
    print("Warmup: %d, Iterations: %d" % (WARMUP, ITERS))
    print("=" * 80)

    results = []
    for name, model_file, shape, ctx, chunk in VARIANTS:
        print("\nBenchmarking: %s" % name)
        r = benchmark_variant(name, model_file, shape, ctx, chunk)
        if r:
            results.append(r)
            print("  %7.0f ms (std %.0f, min %.0f) | audio=%.1fs | RTF=%.3f | size=%.0f MB" % (
                r["avg_ms"], r["std_ms"], r["min_ms"], r["audio_sec"], r["rtf"], r["rknn_mb"]
            ))
        else:
            print("  SKIPPED (model not found or load failed)")

    # Summary table
    print("\n")
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print("%-25s %8s %8s %7s %6s %5s %8s" % (
        "Variant", "Avg(ms)", "Min(ms)", "Audio", "RTF", "Ctx", "Size(MB)"
    ))
    print("-" * 80)

    baseline_ms = None
    for r in results:
        if "baseline" in r["name"]:
            baseline_ms = r["avg_ms"]

    for r in results:
        speedup = ""
        if baseline_ms and r["avg_ms"] != baseline_ms:
            speedup = " (%.1fx)" % (baseline_ms / r["avg_ms"])
        print("%-25s %7.0fms %7.0fms %6.1fs %5.2f %5d %7.0f%s" % (
            r["name"], r["avg_ms"], r["min_ms"], r["audio_sec"],
            r["rtf"], r["ctx"], r["rknn_mb"], speedup
        ))

    # For 6s audio streaming: how many chunks needed and total latency
    print("\n")
    print("=" * 80)
    print("STREAMING LATENCY FOR 6s AUDIO (e.g. TTS sentence)")
    print("=" * 80)
    target_audio = 6.0  # seconds
    target_frames = int(target_audio * SAMPLE_RATE / SAMPLES_PER_FRAME)  # 75 frames
    print("Target: %.1fs audio = %d frames (chunk=25)" % (target_audio, target_frames))
    print("")

    for r in results:
        n_chunks = target_frames // r["chunk"]
        total_ms = n_chunks * r["avg_ms"]
        print("%-25s: %d chunks x %5.0fms = %7.0fms total (RTF=%.2f)" % (
            r["name"], n_chunks, r["avg_ms"], total_ms, r["rtf"]
        ))


if __name__ == "__main__":
    main()

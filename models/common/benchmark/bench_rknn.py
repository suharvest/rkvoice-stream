#!/usr/bin/env python3
"""Benchmark Qwen3-TTS RKNN models on RK3576 NPU.

Usage:
    source ~/rknn-venv/bin/activate
    python3 bench_rknn.py [--model-dir ~/qwen3-tts-rknn] [--runs 10] [--warmup 3]

Measures per-model inference time and memory usage.
Key metric: talker_decode must be < 80ms for real-time TTS at 12.5 Hz.
"""

import argparse
import os
import sys
import time
import traceback

import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite not found. Install rknn-toolkit-lite2.")
    sys.exit(1)


def get_memory_mb():
    """Get current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        return 0


def benchmark_model(rknn_path, inputs, warmup=3, runs=10, core_mask=None):
    """Load and benchmark a single RKNN model."""
    if core_mask is None:
        core_mask = RKNNLite.NPU_CORE_AUTO

    model_name = os.path.basename(rknn_path).replace(".rknn", "")
    file_mb = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({file_mb:.1f} MB)")
    print(f"  Inputs: {len(inputs)}")
    for i, inp in enumerate(inputs[:5]):
        print(f"    [{i}] shape={inp.shape} dtype={inp.dtype}")
    if len(inputs) > 5:
        print(f"    ... and {len(inputs) - 5} more")

    mem_before = get_memory_mb()

    rknn = RKNNLite(verbose=False)

    # Load model
    t0 = time.time()
    ret = rknn.load_rknn(rknn_path)
    load_time = time.time() - t0
    if ret != 0:
        print(f"  FAIL: load_rknn returned {ret}")
        rknn.release()
        return None

    # Init runtime
    t0 = time.time()
    ret = rknn.init_runtime(core_mask=core_mask)
    init_time = time.time() - t0
    if ret != 0:
        print(f"  FAIL: init_runtime returned {ret}")
        rknn.release()
        return None

    mem_after_load = get_memory_mb()
    print(f"  Load: {load_time*1000:.0f} ms, Init: {init_time*1000:.0f} ms")
    print(f"  Memory: {mem_before:.0f} -> {mem_after_load:.0f} MB (+{mem_after_load-mem_before:.0f} MB)")

    # Warmup
    print(f"  Warming up ({warmup} runs)...", end="", flush=True)
    for i in range(warmup):
        try:
            outputs = rknn.inference(inputs=inputs)
        except Exception as e:
            print(f"\n  FAIL during warmup run {i}: {e}")
            rknn.release()
            return None
    print(" done")

    # Show output shapes
    if outputs:
        print(f"  Outputs: {len(outputs)}")
        for i, out in enumerate(outputs[:3]):
            out_arr = np.array(out) if not isinstance(out, np.ndarray) else out
            print(f"    [{i}] shape={out_arr.shape} dtype={out_arr.dtype}")
        if len(outputs) > 3:
            print(f"    ... and {len(outputs) - 3} more")

    # Benchmark
    print(f"  Benchmarking ({runs} runs)...", end="", flush=True)
    times = []
    for _ in range(runs):
        t0 = time.time()
        rknn.inference(inputs=inputs)
        times.append(time.time() - t0)
    print(" done")

    times_ms = np.array(times) * 1000
    avg = np.mean(times_ms)
    std = np.std(times_ms)
    mn = np.min(times_ms)
    mx = np.max(times_ms)

    print(f"  Result: {avg:.1f} +/- {std:.1f} ms (min={mn:.1f}, max={mx:.1f})")

    rknn.release()

    return {
        "model": model_name,
        "file_mb": file_mb,
        "load_ms": load_time * 1000,
        "init_ms": init_time * 1000,
        "mem_delta_mb": mem_after_load - mem_before,
        "avg_ms": avg,
        "std_ms": std,
        "min_ms": mn,
        "max_ms": mx,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark RKNN models")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/qwen3-tts-rknn"),
                        help="Directory containing .rknn files")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to benchmark (default: all)")
    args = parser.parse_args()

    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Qwen3-TTS RKNN Benchmark on RK3576")
    print("=" * 60)
    print(f"Model dir: {model_dir}")
    print(f"Warmup: {args.warmup}, Runs: {args.runs}")
    print(f"Initial RSS: {get_memory_mb():.0f} MB")

    # Define inputs for each model (must match fixed shapes from RKNN conversion)
    # dtype 7 = int64, dtype 1 = float32
    model_inputs = {}

    # codec_embed: [1, 1] int64
    model_inputs["codec_embed"] = [
        np.zeros((1, 1), dtype=np.int64),
    ]

    # code_predictor_embed: [1, 1] int64 + [1] int64 (RKNN needs 1D, not scalar)
    model_inputs["code_predictor_embed"] = [
        np.zeros((1, 1), dtype=np.int64),
        np.array([0], dtype=np.int64),
    ]

    # text_project: [1, 128] int64
    model_inputs["text_project"] = [
        np.ones((1, 128), dtype=np.int64),
    ]

    # speaker_encoder: [1, 300, 128] float32
    model_inputs["speaker_encoder"] = [
        np.random.randn(1, 300, 128).astype(np.float32),
    ]

    # code_predictor: [1, 2, 1024] float32
    model_inputs["code_predictor"] = [
        np.random.randn(1, 2, 1024).astype(np.float32),
    ]

    # talker_prefill: [1, 32, 1024] float32 + [1, 32] int64
    model_inputs["talker_prefill"] = [
        np.random.randn(1, 32, 1024).astype(np.float32),
        np.ones((1, 32), dtype=np.int64),
    ]

    # talker_decode: [1, 1, 1024] float32 + [1, 513] int64 + 56 KV tensors [1, 8, 512, 128] float32
    kv_tensors = [np.random.randn(1, 8, 512, 128).astype(np.float16).astype(np.float32)
                  for _ in range(56)]
    model_inputs["talker_decode"] = [
        np.random.randn(1, 1, 1024).astype(np.float32),
        np.ones((1, 513), dtype=np.int64),
    ] + kv_tensors

    # tokenizer12hz_encode: [1, 72000] float32
    model_inputs["tokenizer12hz_encode"] = [
        np.random.randn(1, 72000).astype(np.float32),
    ]

    # tokenizer12hz_decode_stream: [1, 75, 16] int64
    model_inputs["tokenizer12hz_decode_stream"] = [
        np.random.randint(0, 1024, (1, 75, 16), dtype=np.int64),
    ]

    # Determine which models to run
    available = [f.replace(".rknn", "") for f in sorted(os.listdir(model_dir))
                 if f.endswith(".rknn")]
    print(f"Available models: {available}")

    if args.models:
        to_run = [m for m in args.models if m in available]
    else:
        # Run in order: small models first, talker_decode last (biggest memory)
        order = [
            "codec_embed",
            "code_predictor_embed",
            "text_project",
            "speaker_encoder",
            "code_predictor",
            "tokenizer12hz_encode",
            "tokenizer12hz_decode_stream",
            "talker_prefill",
            "talker_decode",
        ]
        to_run = [m for m in order if m in available]
        # Add any we missed
        for m in available:
            if m not in to_run:
                to_run.append(m)

    results = []
    for model_name in to_run:
        rknn_path = os.path.join(model_dir, f"{model_name}.rknn")
        inputs = model_inputs.get(model_name)
        if inputs is None:
            print(f"\nSKIP: {model_name} (no input definition)")
            continue

        try:
            r = benchmark_model(rknn_path, inputs,
                                warmup=args.warmup, runs=args.runs)
            if r:
                results.append(r)
        except Exception as e:
            print(f"\nERROR benchmarking {model_name}: {e}")
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35s} {'File MB':>8s} {'Avg ms':>8s} {'Std ms':>8s} {'Mem MB':>8s}")
    print("-" * 70)
    total_mem = 0
    for r in results:
        print(f"{r['model']:<35s} {r['file_mb']:8.1f} {r['avg_ms']:8.1f} {r['std_ms']:8.1f} {r['mem_delta_mb']:8.0f}")
        total_mem += r['mem_delta_mb']

    print("-" * 70)
    print(f"{'TOTAL memory delta':<35s} {'':>8s} {'':>8s} {'':>8s} {total_mem:8.0f}")

    # TTS RTF analysis
    decode_result = next((r for r in results if r["model"] == "talker_decode"), None)
    if decode_result:
        step_ms = decode_result["avg_ms"]
        frame_rate = 12.5  # Hz (12 Hz tokenizer)
        frame_ms = 1000.0 / frame_rate
        rtf = step_ms / frame_ms
        print(f"\n--- TTS Real-Time Factor Analysis ---")
        print(f"talker_decode: {step_ms:.1f} ms/step")
        print(f"Frame rate: {frame_rate} Hz ({frame_ms:.1f} ms/frame)")
        print(f"RTF (decode only): {rtf:.2f}x")
        if rtf < 1.0:
            print(f"PASS: Real-time capable ({rtf:.2f}x < 1.0)")
        else:
            print(f"FAIL: NOT real-time ({rtf:.2f}x > 1.0)")
            print(f"  Need < {frame_ms:.0f} ms/step, got {step_ms:.0f} ms")

    print(f"\nFinal RSS: {get_memory_mb():.0f} MB")


if __name__ == "__main__":
    main()

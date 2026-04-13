#!/usr/bin/env python3
"""Convert all fixed-shape Qwen3-TTS ONNX models to RKNN for RK3576.

Reads from: ~/qwen3-tts-export/qwen3-tts-0.6b-12hz-fixed/
Writes to:  ~/qwen3-tts-export/qwen3-tts-0.6b-12hz-rknn-fixed/
"""

import os
import time
import onnx
from rknn.api import RKNN

ONNX_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-0.6b-12hz-fixed")
RKNN_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-0.6b-12hz-rknn-fixed")
TARGET = "rk3576"
os.makedirs(RKNN_DIR, exist_ok=True)

# Models to convert and their optimization levels
# opt_level=3 is default; use 0 for models that fail with constant folding
MODELS = [
    ("text_project", 3),
    ("codec_embed", 3),
    ("code_predictor_embed", 3),
    ("code_predictor", 3),
    ("speaker_encoder", 0),       # opt=0 to avoid constant folding issues
    ("talker_prefill", 3),
    ("talker_decode", 3),
    ("tokenizer12hz_encode", 0),  # may have complex ops
    ("tokenizer12hz_decode_stream", 0),  # has CumSum etc.
]


def convert_model(model_name, opt_level=3):
    onnx_path = os.path.join(ONNX_DIR, f"{model_name}.onnx")
    rknn_path = os.path.join(RKNN_DIR, f"{model_name}.rknn")

    if not os.path.exists(onnx_path):
        return ("SKIP", "ONNX not found")

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Converting: {model_name} ({onnx_size_mb:.1f} MB, opt_level={opt_level})")

    # Show inputs
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = inp.type.tensor_type.elem_type
        print(f"  Input: {inp.name} {dims} dtype={dtype}")
    for out in model.graph.output[:3]:  # first 3 outputs
        dims = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name} {dims}")
    n_out = len(model.graph.output)
    if n_out > 3:
        print(f"  ... and {n_out - 3} more outputs")
    del model

    t0 = time.time()
    rknn = RKNN(verbose=False)

    try:
        ret = rknn.config(target_platform=TARGET, optimization_level=opt_level)
        if ret != 0:
            return ("FAIL", f"config returned {ret}")

        print(f"  load_onnx...")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            return ("FAIL", f"load_onnx returned {ret}")

        print(f"  build...")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            return ("FAIL", f"build returned {ret}")

        print(f"  export_rknn...")
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            return ("FAIL", f"export returned {ret}")

        elapsed = time.time() - t0
        rknn_size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        result = f"{onnx_size_mb:.1f} -> {rknn_size_mb:.1f} MB ({elapsed:.1f}s)"
        print(f"  OK: {result}")
        return ("OK", result)

    except Exception as e:
        err = str(e).split('\n')[0][:200]
        print(f"  FAIL: {err}")
        return ("FAIL", err)
    finally:
        rknn.release()


def main():
    print("=" * 60)
    print("RKNN Conversion (Fixed-Shape ONNX -> RKNN for RK3576)")
    print(f"Input:  {ONNX_DIR}")
    print(f"Output: {RKNN_DIR}")
    print("=" * 60)

    results = {}
    for model_name, opt_level in MODELS:
        results[model_name] = convert_model(model_name, opt_level)

    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    for model_name, _ in MODELS:
        status, detail = results.get(model_name, ("?", "?"))
        print(f"  {model_name:35s}  {status:6s}  {detail}")

    # File listing
    print(f"\nRKNN files:")
    total = 0
    if os.path.exists(RKNN_DIR):
        for f in sorted(os.listdir(RKNN_DIR)):
            if f.endswith(".rknn"):
                sz = os.path.getsize(os.path.join(RKNN_DIR, f))
                total += sz
                print(f"  {f:45s}  {sz/(1024*1024):8.1f} MB")
    print(f"  {'TOTAL':45s}  {total/(1024*1024):8.1f} MB")

    # Check for failures
    failures = [n for n, (s, _) in results.items() if s == "FAIL"]
    if failures:
        print(f"\nFAILED models: {', '.join(failures)}")
        # Retry failures with opt_level=0 if they weren't already
        for name in failures:
            orig_opt = dict(MODELS).get(name, 3)
            if orig_opt != 0:
                print(f"\nRetrying {name} with opt_level=0...")
                results[name] = convert_model(name, opt_level=0)

        # Re-print summary if retries happened
        retry_failures = [n for n in failures if results[n][0] == "FAIL" and dict(MODELS).get(n, 3) != 0]
        if retry_failures:
            print(f"\nStill failing after retry: {', '.join(retry_failures)}")

    ok_count = sum(1 for s, _ in results.values() if s == "OK")
    print(f"\n{ok_count}/{len(MODELS)} models converted successfully")


if __name__ == "__main__":
    main()

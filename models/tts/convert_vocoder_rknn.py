#!/usr/bin/env python3
"""Convert vocoder ONNX variants to RKNN (FP16 and INT8).

Usage (on wsl2-local):
    ~/qwen3-tts-export/.venv/bin/python convert_vocoder_rknn.py
"""

import os
import time
import numpy as np
from rknn.api import RKNN

VARIANTS_DIR = os.path.expanduser("~/qwen3-tts-export/vocoder_variants")
TARGET = "rk3576"

MODELS = [
    # (onnx_name, rknn_name, input_shape, do_quantization)
    ("decoder_ctx25_nosin.onnx", "decoder_ctx25_fp16.rknn", (1, 512, 50), False),
    ("decoder_ctx25_nosin.onnx", "decoder_ctx25_int8.rknn", (1, 512, 50), True),
    ("decoder_ctx0_nosin.onnx", "decoder_ctx0_fp16.rknn", (1, 512, 25), False),
    ("decoder_ctx0_nosin.onnx", "decoder_ctx0_int8.rknn", (1, 512, 25), True),
]


def generate_calibration_data(input_shape, n_samples=20):
    """Generate random calibration data for INT8 quantization."""
    calib_dir = os.path.join(VARIANTS_DIR, "calibration")
    os.makedirs(calib_dir, exist_ok=True)
    calib_file = os.path.join(calib_dir, "calibration_%d.txt" % input_shape[2])

    paths = []
    for i in range(n_samples):
        fpath = os.path.join(calib_dir, "calib_T%d_%02d.npy" % (input_shape[2], i))
        data = np.random.randn(*input_shape).astype(np.float32) * 0.5
        np.save(fpath, data)
        paths.append(fpath)

    with open(calib_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print("    Calibration: %d samples -> %s" % (n_samples, calib_file))
    return calib_file


def convert_model(onnx_name, rknn_name, input_shape, do_quantization):
    onnx_path = os.path.join(VARIANTS_DIR, onnx_name)
    rknn_path = os.path.join(VARIANTS_DIR, rknn_name)

    if not os.path.exists(onnx_path):
        return "SKIP", "ONNX not found"

    quant_str = "INT8" if do_quantization else "FP16"
    onnx_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print("")
    print("  Converting: %s -> %s (%s, %.1f MB)" % (onnx_name, rknn_name, quant_str, onnx_mb))

    t0 = time.time()
    rknn = RKNN(verbose=False)

    try:
        ret = rknn.config(target_platform=TARGET, optimization_level=0)
        if ret != 0:
            return "FAIL", "config returned %d" % ret

        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            return "FAIL", "load_onnx returned %d" % ret

        if do_quantization:
            calib_file = generate_calibration_data(input_shape)
            ret = rknn.build(do_quantization=True, dataset=calib_file)
        else:
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            return "FAIL", "build returned %d" % ret

        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            return "FAIL", "export returned %d" % ret

        elapsed = time.time() - t0
        rknn_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        result = "%.1f -> %.1f MB (%s, %.1fs)" % (onnx_mb, rknn_mb, quant_str, elapsed)
        print("    OK: %s" % result)
        return "OK", result

    except Exception as e:
        err = str(e).split("\n")[0][:200]
        print("    FAIL: %s" % err)
        return "FAIL", err
    finally:
        rknn.release()


def main():
    print("=" * 60)
    print("RKNN Conversion: Vocoder Variants")
    print("=" * 60)

    results = {}
    for onnx_name, rknn_name, shape, quant in MODELS:
        results[rknn_name] = convert_model(onnx_name, rknn_name, shape, quant)

    print("")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for _, rknn_name, _, _ in MODELS:
        status, detail = results.get(rknn_name, ("?", "?"))
        print("  %-35s  %s  %s" % (rknn_name, status, detail))

    print("")
    print("RKNN files:")
    total = 0
    for f in sorted(os.listdir(VARIANTS_DIR)):
        if f.endswith(".rknn"):
            sz = os.path.getsize(os.path.join(VARIANTS_DIR, f))
            total += sz
            print("  %-45s  %8.1f MB" % (f, sz / (1024 * 1024)))
    print("  %-45s  %8.1f MB" % ("TOTAL", total / (1024 * 1024)))


if __name__ == "__main__":
    main()

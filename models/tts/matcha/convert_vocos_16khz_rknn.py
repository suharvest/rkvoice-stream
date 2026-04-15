#!/usr/bin/env python3
"""Convert vocos-16khz-univ.onnx to RKNN for RK3576/RK3588 deployment.

Vocos-16khz Vocoder:
    Input:  mels [1, 80, T] - Mel spectrogram
    Output: mag [1, 513, T] - STFT magnitude
            x   [1, 513, T] - cos component (real part)
            y   [1, 513, T] - sin component (imaginary part)

    Post-processing: ISTFT on CPU to get waveform
    - n_fft = 1024
    - hop_length = 256
    - sample_rate = 16000

RK3588 Clip bug workaround:
    The vocos ONNX has a Clip node with no lower bound (empty string input):
        mag = Clip(Exp(x), min='', max=100)
    On RK3588 NPU, a Clip node with an empty-string min input produces all-zero
    output. The fix is to patch the ONNX before conversion: replace the empty
    min input with an explicit 0.0 constant. This is mathematically equivalent
    because Exp() is always positive.
    Confirmed: rk3576 is unaffected; rk3588 requires this fix.

Usage (on wsl2-local):
    cd ~/matcha-data
    wget https://huggingface.co/csukuangfj/sherpa-onnx-vocoders/resolve/main/vocos-16khz-univ.onnx
    python convert_vocos_16khz_rknn.py
    # For rk3588:
    TARGET=rk3588 python convert_vocos_16khz_rknn.py
"""

import os
import time
import numpy as np
from rknn.api import RKNN

INPUT_ONNX = os.path.expanduser("~/matcha-data/vocos-16khz-univ.onnx")
OUTPUT_DIR = os.path.expanduser("~/matcha-data")
TARGET = os.environ.get("TARGET", "rk3576")

# Fixed time dimension for RKNN (can be changed based on use case)
# For streaming TTS, use smaller chunks (e.g., 50-100 frames)
# For batch synthesis, use larger values (e.g., 200-500 frames)
# Must match VOCOS_FRAMES in rknn_matcha_tts.py backend.
# 600 frames matches the matcha-s64 bucket (~599 mel frames, ~9.6s audio).
TIME_FRAMES = int(os.environ.get('VOCOS_TIME_FRAMES', '600'))


def fix_vocos_onnx_for_rk3588(src_path, dst_path):
    """Patch the vocos ONNX to work around the RK3588 Clip bug.

    RK3588 NPU bug: a Clip node with an empty-string min input (meaning 'no
    lower bound' per ONNX opset >= 11) produces all-zero output.  Replacing
    the empty string with an explicit 0.0 constant fixes this.  The fix is
    mathematically lossless because the Exp() upstream is always > 0.

    Returns dst_path.
    """
    import onnx
    from onnx import numpy_helper

    m = onnx.load(src_path)
    zero_init = numpy_helper.from_array(
        np.array(0.0, dtype=np.float32), name="_clip_min_zero_fix"
    )
    m.graph.initializer.append(zero_init)

    fixed = 0
    for node in m.graph.node:
        if node.op_type == "Clip" and len(node.input) >= 2 and node.input[1] == "":
            node.input[1] = "_clip_min_zero_fix"
            fixed += 1

    onnx.save(m, dst_path)
    print(f"  fix_vocos_onnx: patched {fixed} Clip node(s) -> {dst_path}")
    return dst_path


def generate_calibration_data(n_samples=20):
    """Generate mel spectrogram calibration data."""
    calib_dir = os.path.join(OUTPUT_DIR, "calibration_vocos")
    os.makedirs(calib_dir, exist_ok=True)

    paths = []
    for i in range(n_samples):
        fpath = os.path.join(calib_dir, f"calib_{i:02d}.npy")
        # Random mel spectrogram (typical range [-4, 4] after normalization)
        mel = np.random.randn(1, 80, TIME_FRAMES).astype(np.float32) * 2.0
        np.save(fpath, mel)
        paths.append(fpath)

    calib_file = os.path.join(calib_dir, "calibration_list.txt")
    with open(calib_file, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"Generated {n_samples} calibration samples")
    return calib_file


def convert_vocos(do_quantization=False, suffix="fp16"):
    """Convert vocos ONNX to RKNN."""
    rknn_name = f"vocos-16khz-univ-{suffix}.rknn"
    rknn_path = os.path.join(OUTPUT_DIR, rknn_name)

    print(f"\n{'='*60}")
    print(f"Converting: vocos-16khz-univ.onnx -> {rknn_name}")
    print(f"Target: {TARGET}")
    print(f"Quantization: {'INT8' if do_quantization else 'FP16'}")
    print(f"Input shape: [1, 80, {TIME_FRAMES}]")
    print(f"{'='*60}")

    # RK3588 requires patched ONNX (Clip node empty-min bug workaround)
    onnx_to_convert = INPUT_ONNX
    tmp_fixed_onnx = None
    if TARGET == "rk3588":
        tmp_fixed_onnx = os.path.join(OUTPUT_DIR, "vocos-16khz-univ-rk3588-fixed.onnx")
        onnx_to_convert = fix_vocos_onnx_for_rk3588(INPUT_ONNX, tmp_fixed_onnx)

    t0 = time.time()
    rknn = RKNN(verbose=False)

    # Config
    ret = rknn.config(
        target_platform=TARGET,
        optimization_level=3,  # Enable graph optimization
        single_core_mode=False,  # Use dual-core NPU
    )
    if ret != 0:
        print(f"FAIL: config returned {ret}")
        return None

    # Load ONNX with fixed input shape
    ret = rknn.load_onnx(
        model=onnx_to_convert,
        inputs=['mels'],
        input_size_list=[[1, 80, TIME_FRAMES]],
    )
    if ret != 0:
        print(f"FAIL: load_onnx returned {ret}")
        return None

    # Build
    if do_quantization:
        calib_file = generate_calibration_data()
        ret = rknn.build(do_quantization=True, dataset=calib_file)
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"FAIL: build returned {ret}")
        return None

    # Export
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"FAIL: export returned {ret}")
        return None

    elapsed = time.time() - t0
    rknn_mb = os.path.getsize(rknn_path) / (1024 * 1024)

    print(f"OK: {rknn_mb:.1f} MB ({elapsed:.1f}s)")
    return rknn_path


def verify_with_ort():
    """Verify ONNX model works correctly."""
    import onnxruntime as ort

    print("\n=== Verifying ONNX model ===")

    mel = np.random.randn(1, 80, TIME_FRAMES).astype(np.float32) * 0.5
    sess = ort.InferenceSession(INPUT_ONNX, providers=['CPUExecutionProvider'])

    outputs = sess.run(None, {'mels': mel})
    names = [o.name for o in sess.get_outputs()]

    for name, arr in zip(names, outputs):
        print(f"  {name}: {arr.shape}")

    # ISTFT reconstruction
    mag, x, y = outputs
    complex_spec = mag[0] * (x[0] + 1j * y[0])

    # Simple ISTFT
    n_fft, hop = 1024, 256
    n_frames = complex_spec.shape[1]
    output_len = (n_frames - 1) * hop + n_fft
    waveform = np.zeros(output_len)
    window = np.hanning(n_fft)

    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[:, i], n=n_fft) * window
        start = i * hop
        waveform[start:start+n_fft] += frame

    print(f"  waveform: {waveform.shape} ({waveform.shape[0]/16000:.2f}s)")

    return True


def main():
    print("="*60)
    print("Vocos-16kHz Vocoder RKNN Conversion")
    print(f"Target platform: {TARGET}  (set TARGET env var to change)")
    print("="*60)

    if not os.path.exists(INPUT_ONNX):
        print(f"ERROR: {INPUT_ONNX} not found")
        print("Download with:")
        print("  wget https://huggingface.co/csukuangfj/sherpa-onnx-vocoders/resolve/main/vocos-16khz-univ.onnx")
        return

    onnx_mb = os.path.getsize(INPUT_ONNX) / (1024 * 1024)
    print(f"ONNX model: {INPUT_ONNX} ({onnx_mb:.1f} MB)")
    if TARGET == "rk3588":
        print("NOTE: RK3588 mode — will patch Clip(Exp, empty_min, 100) -> Clip(Exp, 0.0, 100)")
        print("      before conversion to work around RK3588 NPU Clip bug.")

    # Verify ONNX first
    verify_with_ort()

    # Convert FP16
    fp16_path = convert_vocos(do_quantization=False, suffix="fp16")

    # Convert INT8
    int8_path = convert_vocos(do_quantization=True, suffix="int8")

    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")

    for name, path in [("FP16", fp16_path), ("INT8", int8_path)]:
        if path and os.path.exists(path):
            mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name}: {path} ({mb:.1f} MB)")

    print(f"\n{'='*60}")
    print("POST-PROCESSING (ISTFT) - Required on CPU")
    print(f"{'='*60}")
    print("""
# Python implementation:
import numpy as np

def vocos_istft(mag, x, y, n_fft=1024, hop_length=256):
    '''Convert STFT components to waveform.'''
    complex_spec = mag * (x + 1j * y)
    n_frames = complex_spec.shape[-1]
    output_len = (n_frames - 1) * hop_length + n_fft
    waveform = np.zeros(output_len)
    window = np.hanning(n_fft)

    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[..., i], n=n_fft) * window
        start = i * hop_length
        waveform[start:start+n_fft] += frame

    # Normalize
    window_sum = np.zeros(output_len)
    for i in range(n_frames):
        start = i * hop_length
        window_sum[start:start+n_fft] += window ** 2
    waveform = waveform / np.maximum(window_sum, 1e-8)

    return waveform

# Usage:
# mag, x, y = rknn.inference([mel])  # RKNN outputs
# waveform = vocos_istft(mag[0], x[0], y[0])
""")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Convert merged Qwen3-ASR Encoder ONNX to RKNN.

The merged model has 2 inputs:
  - input_features: [1, 128, T_mel]  (mel spectrogram)
  - attention_mask:  [1, 1, T_down, T_down]  (transformer attention mask)

Output:
  - audio_embeds: [1, T_down, 896]

Usage (requires RKNN toolkit env):
  python onnx2rknn_merged.py --onnx qwen3_asr_encoder_merged.fp16.onnx --platform rk3588 15
  python onnx2rknn_merged.py --onnx encoder.onnx --platform rk3576 2 4 15
  python onnx2rknn_merged.py --onnx encoder.onnx --dynamic 2 4 15
"""
import os
import sys
import argparse
from rknn.api import RKNN

MEL_BINS = 128

DURATIONS = {
    2:   200,
    3:   300,
    4:   400,
    5:   500,
    10:  1000,
    15:  1500,
}


def compute_token_len(n_frames, chunk_size=100, tokens_per_chunk=13):
    """Compute encoder output token count from mel frame count."""
    full_chunks = n_frames // chunk_size
    remainder   = n_frames % chunk_size
    if remainder == 0:
        return full_chunks * tokens_per_chunk
    feat = (remainder - 1) // 2 + 1
    feat = (feat - 1) // 2 + 1
    feat = (feat - 1) // 2 + 1
    return full_chunks * tokens_per_chunk + feat


def convert_merged_static(onnx_path, rknn_path, mel_frames, token_len, platform):
    """Convert merged encoder ONNX to RKNN with static shapes."""
    print(f"\n{'='*60}")
    print(f"  ONNX : {onnx_path}")
    print(f"  RKNN : {rknn_path}")
    print(f"  input_features: [1, {MEL_BINS}, {mel_frames}]")
    print(f"  attention_mask: [1, 1, {token_len}, {token_len}]")
    print(f"  output: audio_embeds [1, {token_len}, 896]")
    print(f"{'='*60}")

    rknn = RKNN(verbose=False)
    try:
        rknn.config(target_platform=platform, optimization_level=3)
        ret = rknn.load_onnx(
            model=onnx_path,
            inputs=["input_features", "attention_mask"],
            input_size_list=[
                [1, MEL_BINS, mel_frames],
                [1, 1, token_len, token_len],
            ],
        )
        if ret != 0:
            print(f"  [FAIL] load_onnx returned {ret}")
            return False

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print(f"  [FAIL] build returned {ret}")
            return False

        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            print(f"  [FAIL] export_rknn returned {ret}")
            return False

        size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        print(f"  [OK] {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        rknn.release()


def convert_merged_dynamic(onnx_path, rknn_path, durations_sec, platform):
    """Convert merged encoder to RKNN with dynamic_input (enumerated shapes)."""
    shape_sets = []
    for sec in sorted(durations_sec):
        mf = DURATIONS[sec]
        tl = compute_token_len(mf)
        shape_sets.append([
            [1, MEL_BINS, mf],
            [1, 1, tl, tl],
        ])

    print(f"\n{'='*60}")
    print(f"  ONNX : {onnx_path}")
    print(f"  RKNN : {rknn_path}")
    print(f"  Dynamic shapes: {len(shape_sets)} variants")
    for i, ss in enumerate(shape_sets):
        print(f"    {sorted(durations_sec)[i]}s: mel={ss[0]}, mask={ss[1]}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=False)
    try:
        rknn.config(
            target_platform=platform,
            optimization_level=3,
            dynamic_input=shape_sets,
        )
        ret = rknn.load_onnx(
            model=onnx_path,
            inputs=["input_features", "attention_mask"],
            input_size_list=shape_sets[0],
        )
        if ret != 0:
            print(f"  [FAIL] load_onnx returned {ret}")
            return False

        ret = rknn.build(do_quantization=False)
        if ret != 0:
            print(f"  [FAIL] build returned {ret}")
            return False

        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            print(f"  [FAIL] export_rknn returned {ret}")
            return False

        size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        print(f"  [OK] {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(
        description="Convert merged Qwen3-ASR encoder ONNX to RKNN")
    parser.add_argument("durations", nargs="*", type=int,
                       help=f"Durations in seconds. Available: {sorted(DURATIONS.keys())}")
    parser.add_argument("--dynamic", action="store_true",
                       help="Export as dynamic-input model")
    parser.add_argument("--onnx", required=True,
                       help="Path to merged encoder ONNX")
    parser.add_argument("--out-dir", default=".",
                       help="Output directory (default: .)")
    parser.add_argument("--platform", default="rk3588",
                       help="Target platform: rk3576 or rk3588")
    args = parser.parse_args()

    durations = args.durations or sorted(DURATIONS.keys())
    invalid = [d for d in durations if d not in DURATIONS]
    if invalid:
        print(f"[ERROR] Invalid durations: {invalid}. Available: {sorted(DURATIONS.keys())}")
        sys.exit(1)

    platform = args.platform
    out_dir = os.path.join(args.out_dir, platform)
    os.makedirs(out_dir, exist_ok=True)

    # Print size table
    print("Merged encoder size table:")
    print(f"  {'Dur':>5s}  {'Mel':>5s}  {'Tok':>5s}  {'mel input':>18s}  {'mask input':>22s}")
    for sec in sorted(durations):
        mf = DURATIONS[sec]
        tl = compute_token_len(mf)
        print(f"  {sec:>4d}s  {mf:>5d}  {tl:>5d}  "
              f"[1,128,{mf}]".ljust(18) + f"  [1,1,{tl},{tl}]")

    if not os.path.exists(args.onnx):
        print(f"[ERROR] ONNX not found: {args.onnx}")
        sys.exit(1)

    ok, total = 0, 0

    if args.dynamic:
        print(f"\n=== Dynamic-input export for: {durations} ===")
        total += 1
        rknn_path = os.path.join(out_dir,
            f"qwen3_asr_encoder_merged.fp16.dynamic.{platform}.rknn")
        if convert_merged_dynamic(args.onnx, rknn_path, durations, platform):
            ok += 1
    else:
        print(f"\n=== Static export for: {durations} ===")
        for sec in sorted(durations):
            mel_frames = DURATIONS[sec]
            token_len  = compute_token_len(mel_frames)
            label = f"{sec}s"

            total += 1
            rknn_name = f"qwen3_asr_encoder_merged.fp16.{label}.{platform}.rknn"
            rknn_path = os.path.join(out_dir, rknn_name)
            if convert_merged_static(args.onnx, rknn_path, mel_frames, token_len, platform):
                ok += 1

    print(f"\n{'='*60}")
    print(f"Done. {ok}/{total} succeeded")
    if ok != total:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Split Matcha TTS ONNX into encoder + estimator, then convert to RKNN.

The full matcha model has 3 ODE steps unrolled. When converted to RKNN FP16,
the ODE accumulation z = z + dt*v loses precision, causing mel energy dropout.

Solution: split into encoder (NPU FP16) + estimator (NPU FP16), run the ODE
accumulation loop on CPU in FP32.

Usage:
    # Step 1: Split ONNX (run on x86 host with onnx)
    python split_matcha_rknn.py --split --onnx /path/to/model-steps-3-rknn-ready.onnx

    # Step 2: Convert to RKNN (run on x86 host with rknn-toolkit2)
    python split_matcha_rknn.py --convert

    # Step 3: Verify numerical correctness
    python split_matcha_rknn.py --verify --onnx /path/to/model-steps-3-rknn-ready.onnx

    # All steps:
    python split_matcha_rknn.py --all --onnx /path/to/model-steps-3-rknn-ready.onnx

Output files (in --output-dir):
    matcha-encoder.onnx          - Encoder: text -> mu, mask, z0
    matcha-estimator.onnx        - Estimator: z, mu, mask, time_emb_0..5 -> velocity
    time_emb_step0.npy           - Time embeddings for ODE step 0 (t=0)
    time_emb_step1.npy           - Time embeddings for ODE step 1 (t=1/3)
    time_emb_step2.npy           - Time embeddings for ODE step 2 (t=2/3)
    matcha-encoder-fp16.rknn     - RKNN encoder
    matcha-estimator-fp16.rknn   - RKNN estimator
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# Denormalization constants (from the full model's Mul/Add after ODE)
MEL_SIGMA = 5.446792
MEL_MEAN = -2.9521978
ODE_DT = 1.0 / 3.0
N_ODE_STEPS = 3

# Fixed shapes from the rknn-ready model
BATCH = 1
MEL_BINS = 80
MAX_FRAMES = 600
MAX_TOKENS = 80  # input token length (fixed in rknn-ready)
TIME_EMB_DIM = 256
N_TIME_BLOCKS = 6  # 2 down + 2 mid + 2 up ResnetBlock1D


def split_onnx(onnx_path: str, output_dir: str):
    """Split full matcha ONNX into encoder + estimator submodels."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    from onnx.utils import Extractor

    print("=" * 60)
    print("Splitting Matcha ONNX model")
    print("=" * 60)

    model = onnx.load(onnx_path)
    e = Extractor(model)
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Extract encoder: text inputs -> mu, mask, z0
    # ----------------------------------------------------------------
    print("\n--- Extracting encoder ---")
    enc_model = e.extract_model(
        input_names=["x", "x_length", "noise_scale", "length_scale"],
        output_names=[
            "/Transpose_3_output_0",  # mu [1, 80, 600]
            "/Cast_3_output_0",       # mask [1, 1, 600]
            "/decoder/Mul_output_0",  # z0 = noise * noise_scale [1, 80, 600]
        ],
    )
    enc_path = os.path.join(output_dir, "matcha-encoder.onnx")
    onnx.save(enc_model, enc_path)
    print(f"  Encoder: {len(enc_model.graph.node)} nodes -> {enc_path}")
    for i in enc_model.graph.input:
        s = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"    In:  {i.name} {s}")
    for o in enc_model.graph.output:
        s = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"    Out: {o.name} {s}")

    # ----------------------------------------------------------------
    # 2. Extract step-0 estimator, then parameterize time embedding
    # ----------------------------------------------------------------
    print("\n--- Extracting estimator ---")
    est_raw = e.extract_model(
        input_names=[
            "/decoder/Mul_output_0",     # z
            "/Transpose_3_output_0",     # mu
            "/Cast_3_output_0",          # mask
        ],
        output_names=["/decoder/estimator/Mul_5_output_0"],  # velocity (masked)
    )

    # Identify the 6 baked time-embedding initializers (one per ResnetBlock)
    time_emb_names_step0 = [
        "/decoder/estimator/down_blocks.0.0/Unsqueeze_output_0",
        "/decoder/estimator/down_blocks.1.0/Unsqueeze_output_0",
        "/decoder/estimator/mid_blocks.0.0/Unsqueeze_output_0",
        "/decoder/estimator/mid_blocks.1.0/Unsqueeze_output_0",
        "/decoder/estimator/up_blocks.0.0/Unsqueeze_output_0",
        "/decoder/estimator/up_blocks.1.0/Unsqueeze_output_0",
    ]

    # Extract time embeddings for all 3 ODE steps
    step_te = {}  # step -> [6, 256] array

    # Step 0
    te_s0 = []
    for init in est_raw.graph.initializer:
        if init.name in time_emb_names_step0:
            te_s0.append(numpy_helper.to_array(init))
    # Maintain order
    te_s0_dict = {
        init.name: numpy_helper.to_array(init)
        for init in est_raw.graph.initializer
        if init.name in time_emb_names_step0
    }
    step_te[0] = np.stack(
        [te_s0_dict[n].reshape(TIME_EMB_DIM) for n in time_emb_names_step0]
    )

    # Step 1
    est1 = e.extract_model(
        input_names=["/decoder/Add_output_0", "/Transpose_3_output_0", "/Cast_3_output_0"],
        output_names=["/decoder/estimator_1/Mul_5_output_0"],
    )
    te_s1_dict = {}
    for init in est1.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim == 3 and arr.shape == (1, TIME_EMB_DIM, 1):
            te_s1_dict[init.name] = arr
    s1_names = [
        n.replace("/Unsqueeze_output_0", "_1/Unsqueeze_output_0")
        for n in time_emb_names_step0
    ]
    step_te[1] = np.stack([te_s1_dict[n].reshape(TIME_EMB_DIM) for n in s1_names])

    # Step 2
    est2 = e.extract_model(
        input_names=["/decoder/Add_1_output_0", "/Transpose_3_output_0", "/Cast_3_output_0"],
        output_names=["/decoder/estimator_2/Mul_5_output_0"],
    )
    te_s2_dict = {}
    for init in est2.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.ndim == 3 and arr.shape == (1, TIME_EMB_DIM, 1):
            te_s2_dict[init.name] = arr
    s2_names = [
        n.replace("/Unsqueeze_output_0", "_2/Unsqueeze_output_0")
        for n in time_emb_names_step0
    ]
    step_te[2] = np.stack([te_s2_dict[n].reshape(TIME_EMB_DIM) for n in s2_names])

    # Save time embeddings
    for step_idx in range(N_ODE_STEPS):
        np.save(
            os.path.join(output_dir, f"time_emb_step{step_idx}.npy"),
            step_te[step_idx],
        )
    print(f"  Saved time embeddings: {[te.shape for te in step_te.values()]}")

    # ----------------------------------------------------------------
    # 3. Rebuild estimator with time_emb as inputs (not baked constants)
    # ----------------------------------------------------------------
    # Remove the 6 time-emb initializers
    new_initializers = [
        init
        for init in est_raw.graph.initializer
        if init.name not in time_emb_names_step0
    ]

    # Rename tensor references in all nodes
    rename_map = {
        "/decoder/Mul_output_0": "z",
        "/Transpose_3_output_0": "mu",
        "/Cast_3_output_0": "mask",
        "/decoder/estimator/Mul_5_output_0": "velocity",
    }
    for i, name in enumerate(time_emb_names_step0):
        rename_map[name] = f"time_emb_{i}"

    for node in est_raw.graph.node:
        for idx in range(len(node.input)):
            if node.input[idx] in rename_map:
                node.input[idx] = rename_map[node.input[idx]]
        for idx in range(len(node.output)):
            if node.output[idx] in rename_map:
                node.output[idx] = rename_map[node.output[idx]]

    # Build clean inputs
    graph_inputs = [
        helper.make_tensor_value_info("z", TensorProto.FLOAT, [BATCH, MEL_BINS, MAX_FRAMES]),
        helper.make_tensor_value_info("mu", TensorProto.FLOAT, [BATCH, MEL_BINS, MAX_FRAMES]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [BATCH, 1, MAX_FRAMES]),
    ]
    for i in range(N_TIME_BLOCKS):
        graph_inputs.append(
            helper.make_tensor_value_info(
                f"time_emb_{i}", TensorProto.FLOAT, [BATCH, TIME_EMB_DIM, 1]
            )
        )

    graph_outputs = [
        helper.make_tensor_value_info(
            "velocity", TensorProto.FLOAT, [BATCH, MEL_BINS, MAX_FRAMES]
        ),
    ]

    # Filter value_info (remove renamed tensors)
    renamed_set = set(rename_map.keys()) | set(rename_map.values())
    new_vi = [
        vi for vi in est_raw.graph.value_info if vi.name not in renamed_set
    ]

    new_graph = helper.make_graph(
        nodes=list(est_raw.graph.node),
        name="matcha-estimator",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=new_initializers,
        value_info=new_vi,
    )

    new_model = helper.make_model(new_graph, opset_imports=est_raw.opset_import)
    new_model.ir_version = est_raw.ir_version

    est_path = os.path.join(output_dir, "matcha-estimator.onnx")
    onnx.save(new_model, est_path)
    print(f"\n  Estimator: {len(new_graph.node)} nodes -> {est_path}")
    for i in new_graph.input:
        s = [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
        print(f"    In:  {i.name} {s}")
    for o in new_graph.output:
        s = [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
        print(f"    Out: {o.name} {s}")

    # Report sizes
    enc_size = os.path.getsize(enc_path) / 1024 / 1024
    est_size = os.path.getsize(est_path) / 1024 / 1024
    full_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n  Full model:  {full_size:.1f} MB")
    print(f"  Encoder:     {enc_size:.1f} MB")
    print(f"  Estimator:   {est_size:.1f} MB")
    print(f"  Sum:         {enc_size + est_size:.1f} MB")


def verify_split(onnx_path: str, output_dir: str):
    """Verify split models produce identical output to full model."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("Verifying split model accuracy")
    print("=" * 60)

    # Prepare test input
    x = np.zeros((1, MAX_TOKENS), dtype=np.int64)
    x[0, :10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_length = np.array([10], dtype=np.int64)
    noise_scale = np.array([1.0], dtype=np.float32)
    length_scale = np.array([1.0], dtype=np.float32)

    # Full model reference
    print("  Running full model...")
    full_sess = ort.InferenceSession(onnx_path)
    full_mel = full_sess.run(
        None,
        {"x": x, "x_length": x_length, "noise_scale": noise_scale, "length_scale": length_scale},
    )[0]

    # Split model
    print("  Running split encoder...")
    enc_sess = ort.InferenceSession(os.path.join(output_dir, "matcha-encoder.onnx"))
    mu, mask, z0 = enc_sess.run(
        None,
        {"x": x, "x_length": x_length, "noise_scale": noise_scale, "length_scale": length_scale},
    )

    print("  Running split estimator (3 ODE steps)...")
    est_sess = ort.InferenceSession(os.path.join(output_dir, "matcha-estimator.onnx"))

    # Load time embeddings
    te_steps = [
        np.load(os.path.join(output_dir, f"time_emb_step{i}.npy"))
        for i in range(N_ODE_STEPS)
    ]

    z = z0.copy()
    for step in range(N_ODE_STEPS):
        feeds = {"z": z, "mu": mu, "mask": mask}
        for i in range(N_TIME_BLOCKS):
            feeds[f"time_emb_{i}"] = te_steps[step][i].reshape(1, TIME_EMB_DIM, 1)
        v = est_sess.run(None, feeds)[0]
        z = z + ODE_DT * v

    # Denormalize (the full model does Slice then Mul+Add)
    n_frames = full_mel.shape[2]
    split_mel = z[:, :, :n_frames] * MEL_SIGMA + MEL_MEAN

    # Compare
    max_diff = np.abs(full_mel - split_mel).max()
    mean_diff = np.abs(full_mel - split_mel).mean()
    rel_max = max_diff / (np.abs(full_mel).max() + 1e-8)

    print(f"\n  Max absolute diff:  {max_diff:.2e}")
    print(f"  Mean absolute diff: {mean_diff:.2e}")
    print(f"  Relative max diff:  {rel_max:.2e}")
    print(f"  Full mel range:     [{full_mel.min():.3f}, {full_mel.max():.3f}]")
    print(f"  Split mel range:    [{split_mel.min():.3f}, {split_mel.max():.3f}]")

    if max_diff < 1e-3:
        print("\n  PASS: Split model is numerically identical to full model.")
        return True
    else:
        print(f"\n  FAIL: Max diff {max_diff:.6f} exceeds threshold 1e-3")
        return False


def convert_to_rknn(output_dir: str, target_platform: str = "rk3576"):
    """Convert encoder and estimator ONNX to RKNN FP16."""
    from rknn.api import RKNN

    print("\n" + "=" * 60)
    print("Converting to RKNN")
    print("=" * 60)

    enc_onnx = os.path.join(output_dir, "matcha-encoder.onnx")
    est_onnx = os.path.join(output_dir, "matcha-estimator.onnx")

    # ----------------------------------------------------------------
    # Encoder
    # ----------------------------------------------------------------
    print("\n--- Converting encoder ---")
    enc_rknn_path = os.path.join(output_dir, "matcha-encoder-fp16.rknn")

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=target_platform,
        optimization_level=3,
        single_core_mode=False,
    )
    ret = rknn.load_onnx(
        model=enc_onnx,
        inputs=["x", "x_length", "noise_scale", "length_scale"],
        input_size_list=[
            [BATCH, MAX_TOKENS],
            [BATCH],
            [1],
            [1],
        ],
    )
    if ret != 0:
        print(f"  ERROR: load_onnx failed (ret={ret})")
        return
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"  ERROR: build failed (ret={ret})")
        return
    rknn.export_rknn(enc_rknn_path)
    rknn.release()
    enc_size = os.path.getsize(enc_rknn_path) / 1024 / 1024
    print(f"  Encoder RKNN: {enc_rknn_path} ({enc_size:.1f} MB)")

    # ----------------------------------------------------------------
    # Estimator
    # ----------------------------------------------------------------
    print("\n--- Converting estimator ---")
    est_rknn_path = os.path.join(output_dir, "matcha-estimator-fp16.rknn")

    input_names = ["z", "mu", "mask"] + [f"time_emb_{i}" for i in range(N_TIME_BLOCKS)]
    input_sizes = [
        [BATCH, MEL_BINS, MAX_FRAMES],
        [BATCH, MEL_BINS, MAX_FRAMES],
        [BATCH, 1, MAX_FRAMES],
    ] + [[BATCH, TIME_EMB_DIM, 1]] * N_TIME_BLOCKS

    rknn = RKNN(verbose=False)
    rknn.config(
        target_platform=target_platform,
        optimization_level=3,
        single_core_mode=False,
    )
    ret = rknn.load_onnx(
        model=est_onnx,
        inputs=input_names,
        input_size_list=input_sizes,
    )
    if ret != 0:
        print(f"  ERROR: load_onnx failed (ret={ret})")
        return
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"  ERROR: build failed (ret={ret})")
        return
    rknn.export_rknn(est_rknn_path)
    rknn.release()
    est_size = os.path.getsize(est_rknn_path) / 1024 / 1024
    print(f"  Estimator RKNN: {est_rknn_path} ({est_size:.1f} MB)")

    print(f"\n  Total RKNN size: {enc_size + est_size:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Split Matcha TTS into encoder + estimator for RKNN"
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="Path to model-steps-3-rknn-ready.onnx",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input ONNX)",
    )
    parser.add_argument("--split", action="store_true", help="Split ONNX model")
    parser.add_argument("--verify", action="store_true", help="Verify split accuracy")
    parser.add_argument("--convert", action="store_true", help="Convert to RKNN")
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument(
        "--target", type=str, default="rk3576", help="RKNN target platform"
    )

    args = parser.parse_args()

    if args.all:
        args.split = args.verify = args.convert = True

    if not any([args.split, args.verify, args.convert]):
        parser.print_help()
        return

    if args.output_dir is None and args.onnx:
        args.output_dir = os.path.dirname(args.onnx) or "."

    if args.output_dir is None:
        args.output_dir = "."

    if args.split:
        if not args.onnx:
            print("ERROR: --onnx is required for --split")
            sys.exit(1)
        split_onnx(args.onnx, args.output_dir)

    if args.verify:
        if not args.onnx:
            print("ERROR: --onnx is required for --verify")
            sys.exit(1)
        verify_split(args.onnx, args.output_dir)

    if args.convert:
        convert_to_rknn(args.output_dir, args.target)


if __name__ == "__main__":
    main()

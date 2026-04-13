#!/usr/bin/env python3
"""
Profile matmul decoder on RK3576 to understand 461ms/token breakdown.

Usage:
    PYTHONPATH=/home/cat/project/rknn-matmul-parallel python3 profile_matmul_decoder.py
"""

import ctypes
import time
import numpy as np
import os
import sys

# ─── Constants (Qwen3-ASR-0.6B) ───
HIDDEN_DIM = 1024
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
FFN_DIM = 3072
NUM_LAYERS = 28
VOCAB_SIZE = 151936
RMS_EPS = 1e-6
IOMMU_DOMAIN_ID = 2

# ─── Load librknnrt.so ───
lib = ctypes.CDLL("/usr/lib/librknnrt.so")

# ─── Type definitions ───
rknn_matmul_ctx = ctypes.c_uint64

class rknn_matmul_info(ctypes.Structure):
    _fields_ = [
        ("M", ctypes.c_int32), ("K", ctypes.c_int32), ("N", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("B_layout", ctypes.c_int16), ("B_quant_type", ctypes.c_int16),
        ("AC_layout", ctypes.c_int16), ("AC_quant_type", ctypes.c_int16),
        ("iommu_domain_id", ctypes.c_int32),
        ("group_size", ctypes.c_int16),
        ("reserved", ctypes.c_int8 * 34),
    ]

class rknn_tensor_mem(ctypes.Structure):
    _fields_ = [
        ("virt_addr", ctypes.c_void_p), ("phys_addr", ctypes.c_uint64),
        ("fd", ctypes.c_int32), ("offset", ctypes.c_int32),
        ("size", ctypes.c_uint32), ("flags", ctypes.c_uint32),
        ("priv_data", ctypes.c_void_p),
    ]

class rknn_matmul_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256), ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16), ("size", ctypes.c_uint32),
        ("type", ctypes.c_int32),
    ]

class rknn_matmul_io_attr(ctypes.Structure):
    _fields_ = [
        ("A", rknn_matmul_tensor_attr), ("B", rknn_matmul_tensor_attr),
        ("C", rknn_matmul_tensor_attr),
    ]

# Function prototypes
lib.rknn_matmul_create.restype = ctypes.c_int
lib.rknn_matmul_create.argtypes = [ctypes.POINTER(rknn_matmul_ctx), ctypes.POINTER(rknn_matmul_info), ctypes.POINTER(rknn_matmul_io_attr)]
lib.rknn_matmul_destroy.restype = ctypes.c_int
lib.rknn_matmul_destroy.argtypes = [rknn_matmul_ctx]
lib.rknn_matmul_run.restype = ctypes.c_int
lib.rknn_matmul_run.argtypes = [rknn_matmul_ctx]
lib.rknn_matmul_set_io_mem.restype = ctypes.c_int
lib.rknn_matmul_set_io_mem.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_tensor_mem), ctypes.POINTER(rknn_matmul_tensor_attr)]
lib.rknn_create_mem.restype = ctypes.POINTER(rknn_tensor_mem)
lib.rknn_create_mem.argtypes = [rknn_matmul_ctx, ctypes.c_uint32]
lib.rknn_destroy_mem.restype = ctypes.c_int
lib.rknn_destroy_mem.argtypes = [rknn_matmul_ctx, ctypes.POINTER(rknn_tensor_mem)]
lib.rknn_B_normal_layout_to_native_layout.restype = ctypes.c_int
lib.rknn_B_normal_layout_to_native_layout.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(rknn_matmul_info)]

RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32 = 1


def create_fp16_matmul(M, K, N):
    """Create FP16 matmul context with valid random weights in native layout."""
    ctx = rknn_matmul_ctx()
    info = rknn_matmul_info()
    io_attr = rknn_matmul_io_attr()
    info.M = M; info.K = K; info.N = N
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32
    info.B_layout = 0
    info.iommu_domain_id = IOMMU_DOMAIN_ID

    ret = lib.rknn_matmul_create(ctypes.byref(ctx), ctypes.byref(info), ctypes.byref(io_attr))
    if ret != 0:
        raise RuntimeError("rknn_matmul_create(%d,%d,%d) failed: %d" % (M, K, N, ret))

    mem_A = lib.rknn_create_mem(ctx, io_attr.A.size)
    mem_B = lib.rknn_create_mem(ctx, io_attr.B.size)
    mem_C = lib.rknn_create_mem(ctx, io_attr.C.size)

    lib.rknn_matmul_set_io_mem(ctx, mem_A, ctypes.byref(io_attr.A))
    lib.rknn_matmul_set_io_mem(ctx, mem_B, ctypes.byref(io_attr.B))
    lib.rknn_matmul_set_io_mem(ctx, mem_C, ctypes.byref(io_attr.C))

    # Fill A
    a_data = np.random.randn(M, K).astype(np.float16).tobytes()
    ctypes.memmove(mem_A.contents.virt_addr, a_data, len(a_data))

    # Fill B with FP16 and convert to native layout
    b_normal = np.random.randn(K * N).astype(np.float16).tobytes()
    b_buf = (ctypes.c_uint8 * len(b_normal))(*b_normal)
    lib.rknn_B_normal_layout_to_native_layout(
        ctypes.cast(b_buf, ctypes.c_void_p),
        ctypes.c_void_p(mem_B.contents.virt_addr),
        K, N, ctypes.byref(info))

    return ctx, info, io_attr, mem_A, mem_B, mem_C


def destroy_ctx(ctx, mem_A, mem_B, mem_C):
    lib.rknn_destroy_mem(ctx, mem_A)
    lib.rknn_destroy_mem(ctx, mem_B)
    lib.rknn_destroy_mem(ctx, mem_C)
    lib.rknn_matmul_destroy(ctx)


def bench_matmul_run(name, K, N, n_warmup=5, n_runs=30):
    """Benchmark raw rknn_matmul_run (FP16)."""
    ctx, info, io_attr, mem_A, mem_B, mem_C = create_fp16_matmul(1, K, N)
    for _ in range(n_warmup):
        lib.rknn_matmul_run(ctx)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lib.rknn_matmul_run(ctx)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    destroy_ctx(ctx, mem_A, mem_B, mem_C)
    return np.mean(times), np.std(times), np.min(times)


def bench_full_roundtrip(name, K, N, n_warmup=5, n_runs=30):
    """Benchmark full: FP32->FP16 input + B rebind + run + FP32 output memcpy."""
    ctx, info, io_attr, mem_A, mem_B, mem_C = create_fp16_matmul(1, K, N)
    # Create second B for rebind testing
    extra_B = lib.rknn_create_mem(ctx, io_attr.B.size)
    b_normal = np.random.randn(K * N).astype(np.float16).tobytes()
    b_buf = (ctypes.c_uint8 * len(b_normal))(*b_normal)
    lib.rknn_B_normal_layout_to_native_layout(
        ctypes.cast(b_buf, ctypes.c_void_p),
        ctypes.c_void_p(extra_B.contents.virt_addr),
        K, N, ctypes.byref(info))

    input_fp32 = np.random.randn(K).astype(np.float32)
    out_fp32 = np.empty(N, dtype=np.float32)

    for _ in range(n_warmup):
        inp = input_fp32.astype(np.float16).tobytes()
        ctypes.memmove(mem_A.contents.virt_addr, inp, len(inp))
        lib.rknn_matmul_set_io_mem(ctx, extra_B, ctypes.byref(io_attr.B))
        lib.rknn_matmul_run(ctx)
        ctypes.memmove(out_fp32.ctypes.data, mem_C.contents.virt_addr, N * 4)

    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        # 1. FP32->FP16 input conversion
        inp = input_fp32.astype(np.float16).tobytes()
        ctypes.memmove(mem_A.contents.virt_addr, inp, len(inp))
        # 2. B rebind (simulates pooled mode)
        lib.rknn_matmul_set_io_mem(ctx, extra_B if i % 2 == 0 else mem_B, ctypes.byref(io_attr.B))
        # 3. NPU matmul run
        lib.rknn_matmul_run(ctx)
        # 4. Read FP32 output
        ctypes.memmove(out_fp32.ctypes.data, mem_C.contents.virt_addr, N * 4)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    lib.rknn_destroy_mem(ctx, extra_B)
    destroy_ctx(ctx, mem_A, mem_B, mem_C)
    return np.mean(times), np.std(times), np.min(times)


def bench_rebind(n_rebinds=196):
    """Benchmark B rebind overhead only."""
    dims = [(1024, 2048), (1024, 1024), (1024, 3072), (3072, 1024)]
    contexts = []
    for K, N in dims:
        ctx, info, io_attr, mem_A, mem_B, mem_C = create_fp16_matmul(1, K, N)
        extra_B = lib.rknn_create_mem(ctx, io_attr.B.size)
        b = np.random.randn(K * N).astype(np.float16).tobytes()
        bb = (ctypes.c_uint8 * len(b))(*b)
        lib.rknn_B_normal_layout_to_native_layout(
            ctypes.cast(bb, ctypes.c_void_p),
            ctypes.c_void_p(extra_B.contents.virt_addr), K, N, ctypes.byref(info))
        contexts.append((ctx, info, io_attr, mem_A, mem_B, mem_C, extra_B))

    for c in contexts:
        for _ in range(10):
            lib.rknn_matmul_set_io_mem(c[0], c[6], ctypes.byref(c[2].B))

    trial_times = []
    for _ in range(20):
        t0 = time.perf_counter()
        for i in range(n_rebinds):
            c = contexts[i % len(contexts)]
            lib.rknn_matmul_set_io_mem(c[0], c[6] if i % 2 == 0 else c[4], ctypes.byref(c[2].B))
        t1 = time.perf_counter()
        trial_times.append((t1 - t0) * 1000)

    for c in contexts:
        lib.rknn_destroy_mem(c[0], c[6])
        destroy_ctx(c[0], c[3], c[4], c[5])

    return np.mean(trial_times), np.min(trial_times)


def bench_cpu_ops(n_runs=500):
    """Benchmark CPU operations with numpy on ARM A55."""
    results = {}

    # RMSNorm (1024 dim)
    x = np.random.randn(HIDDEN_DIM).astype(np.float32)
    w = np.random.randn(HIDDEN_DIM).astype(np.float32)
    for _ in range(50):
        _ = x * (1.0 / np.sqrt(np.mean(x * x) + RMS_EPS)) * w
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = x * (1.0 / np.sqrt(np.mean(x * x) + RMS_EPS)) * w
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["rmsnorm_1024"] = np.median(times)

    # QK norm: 16+8=24 per-head RMSNorm on 128-dim vectors
    q = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
    k = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
    qw = np.random.randn(HEAD_DIM).astype(np.float32)
    kw = np.random.randn(HEAD_DIM).astype(np.float32)
    for _ in range(50):
        for h in range(NUM_Q_HEADS):
            q[h] = q[h] * (1.0 / np.sqrt(np.mean(q[h] * q[h]) + RMS_EPS)) * qw
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for h in range(NUM_Q_HEADS):
            q[h] = q[h] * (1.0 / np.sqrt(np.mean(q[h] * q[h]) + RMS_EPS)) * qw
        for h in range(NUM_KV_HEADS):
            k[h] = k[h] * (1.0 / np.sqrt(np.mean(k[h] * k[h]) + RMS_EPS)) * kw
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["qk_norm_24heads"] = np.median(times)

    # RoPE
    cos = np.random.randn(HEAD_DIM // 2).astype(np.float32)
    sin = np.random.randn(HEAD_DIM // 2).astype(np.float32)
    times = []
    for _ in range(n_runs):
        q_v = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
        k_v = np.random.randn(NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
        t0 = time.perf_counter()
        qe, qo = q_v[:, 0::2].copy(), q_v[:, 1::2].copy()
        q_v[:, 0::2] = qe * cos - qo * sin
        q_v[:, 1::2] = qe * sin + qo * cos
        ke, ko = k_v[:, 0::2].copy(), k_v[:, 1::2].copy()
        k_v[:, 0::2] = ke * cos - ko * sin
        k_v[:, 1::2] = ke * sin + ko * cos
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["rope_24heads"] = np.median(times)

    # Attention GQA: 16Q/8KV heads at various seq lengths
    for seq_len in [1, 10, 20, 50]:
        scale = 1.0 / np.sqrt(HEAD_DIM)
        q_att = np.random.randn(NUM_Q_HEADS, HEAD_DIM).astype(np.float32)
        k_cache = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
        v_cache = np.random.randn(seq_len, NUM_KV_HEADS, HEAD_DIM).astype(np.float32)
        for _ in range(20):
            out = np.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=np.float32)
            for h in range(NUM_Q_HEADS):
                kv_h = h // 2
                scores = k_cache[:, kv_h] @ q_att[h] * scale
                scores = np.exp(scores - np.max(scores))
                scores /= scores.sum()
                out[h] = scores @ v_cache[:, kv_h]
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = np.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=np.float32)
            for h in range(NUM_Q_HEADS):
                kv_h = h // 2
                scores = k_cache[:, kv_h] @ q_att[h] * scale
                scores = np.exp(scores - np.max(scores))
                scores /= scores.sum()
                out[h] = scores @ v_cache[:, kv_h]
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)
        results["attn_seq%d" % seq_len] = np.median(times)

    # SiLU * up (3072)
    gate = np.random.randn(FFN_DIM).astype(np.float32)
    up = np.random.randn(FFN_DIM).astype(np.float32)
    for _ in range(50):
        _ = gate / (1.0 + np.exp(-np.clip(gate, -20, 20))) * up
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = gate / (1.0 + np.exp(-np.clip(gate, -20, 20))) * up
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["silu_mul_3072"] = np.median(times)

    # vec_add (1024)
    a = np.random.randn(HIDDEN_DIM).astype(np.float32)
    b = np.random.randn(HIDDEN_DIM).astype(np.float32)
    for _ in range(50):
        _ = a + b
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = a + b
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["vec_add_1024"] = np.median(times)

    # memcpy (1024 floats = 4KB)
    src = np.random.randn(HIDDEN_DIM).astype(np.float32)
    dst = np.empty(HIDDEN_DIM, dtype=np.float32)
    for _ in range(50):
        np.copyto(dst, src)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["memcpy_4KB"] = np.median(times)

    # FP32<->FP16 conversions
    d1024 = np.random.randn(1024).astype(np.float32)
    d3072 = np.random.randn(3072).astype(np.float16)
    for _ in range(50):
        _ = d1024.astype(np.float16)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = d1024.astype(np.float16)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["fp32_to_fp16_1024"] = np.median(times)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = d3072.astype(np.float32)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    results["fp16_to_fp32_3072"] = np.median(times)

    return results


def bench_lm_head():
    """Benchmark lm_head: (1024) dot (151936, 1024)^T on CPU."""
    hidden = np.random.randn(HIDDEN_DIM).astype(np.float32)

    # Check available memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    avail_mb = int(line.split()[1]) / 1024
                    break
    except Exception:
        avail_mb = 2000

    print("  Available: %.0f MB, need ~590 MB" % avail_mb, flush=True)

    if avail_mb > 800:
        lm = np.random.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32)
        for _ in range(2):
            _ = lm @ hidden
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            logits = lm @ hidden
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        argmax_t = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = np.argmax(logits)
            t1 = time.perf_counter()
            argmax_t.append((t1 - t0) * 1000)
        del lm
        return np.mean(times), np.min(times), np.median(argmax_t)
    else:
        # Chunked measurement
        chunk = 8192
        n_chunks = (VOCAB_SIZE + chunk - 1) // chunk
        lm_chunk = np.random.randn(chunk, HIDDEN_DIM).astype(np.float32)
        for _ in range(5):
            _ = lm_chunk @ hidden
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = lm_chunk @ hidden
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        chunk_ms = np.median(times)
        total = chunk_ms * n_chunks
        del lm_chunk
        print("  Chunked: %.2f ms/chunk x %d = %.1f ms" % (chunk_ms, n_chunks, total), flush=True)
        return total, total * 0.9, 0.1


def bench_c_lib():
    """Benchmark actual C library step function if available."""
    try:
        sys.path.insert(0, "/home/cat/project/rknn-matmul-parallel")
        import matmul_decoder as md
    except ImportError:
        print("  C extension not available", flush=True)
        return {}

    results = {}
    for qt, wdir in [
        ("fp16", "/home/cat/qwen3-asr-models/decoder/matmul"),
        ("int4", "/home/cat/qwen3-asr-models/decoder/matmul_w4a16"),
    ]:
        if not os.path.exists(wdir + "/config.json"):
            print("  %s: weights not found at %s" % (qt, wdir), flush=True)
            continue

        for mode in ["single_core", "dual_core"]:
            label = "%s_%s" % (qt, mode)
            try:
                print("  Loading %s (%s)..." % (qt, mode), flush=True)
                dec = md.MatmulDecoder(
                    model_dir=wdir,
                    max_seq_len=256,
                    quant_type=qt,
                    exec_mode=mode,
                )

                # Warmup: prefill + 3 decode steps
                emb = np.random.randn(HIDDEN_DIM).astype(np.float32)
                dec.clear_kv_cache()
                _ = dec.step_get_token(token_id=-1, embedding=emb)
                for _ in range(3):
                    _ = dec.step_get_token(token_id=0)

                # Benchmark decode steps
                times = []
                for _ in range(15):
                    t0 = time.perf_counter()
                    _ = dec.step_get_token(token_id=0)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000)

                avg = np.mean(times)
                mn = np.min(times)
                med = np.median(times)
                results[label] = {"avg": avg, "min": mn, "median": med}
                print("  %s: avg=%.1f ms, median=%.1f ms, min=%.1f ms" % (label, avg, med, mn), flush=True)

                del dec
            except Exception as e:
                print("  %s: ERROR %s" % (label, e), flush=True)

    return results


def main():
    print("=" * 70, flush=True)
    print("Matmul Decoder Profile (RK3576, Qwen3-ASR-0.6B)", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    projections = [
        ("q_proj",    1024, 2048),
        ("k_proj",    1024, 1024),
        ("v_proj",    1024, 1024),
        ("o_proj",    2048, 1024),
        ("gate_proj", 1024, 3072),
        ("up_proj",   1024, 3072),
        ("down_proj", 3072, 1024),
    ]

    # ─── 1. Raw matmul_run times (FP16) ───
    print("--- 1. Raw rknn_matmul_run (FP16, no conversion overhead) ---", flush=True)
    print(flush=True)
    fp16_raw = {}
    for name, K, N in projections:
        avg, std, mn = bench_matmul_run(name, K, N)
        fp16_raw[name] = (avg, mn)
        print("  %s  (1,%d,%d): avg=%5.2f ms  min=%5.2f ms" % (name.ljust(12), K, N, avg, mn), flush=True)
    print(flush=True)

    # ─── 2. Full round-trip (FP32->FP16 + rebind + run + FP32 output) ───
    print("--- 2. Full round-trip per projection (conversion + rebind + run + read) ---", flush=True)
    print(flush=True)
    fp16_full = {}
    for name, K, N in projections:
        avg, std, mn = bench_full_roundtrip(name, K, N)
        fp16_full[name] = (avg, mn)
        print("  %s: avg=%5.2f ms  min=%5.2f ms  overhead=+%.2f ms" % (
            name.ljust(12), avg, mn, avg - fp16_raw[name][0]), flush=True)
    print(flush=True)

    # ─── 3. B rebind overhead ───
    print("--- 3. B rebind overhead (196 = 28 layers x 7 projections) ---", flush=True)
    rebind_avg, rebind_min = bench_rebind()
    print("  196 calls: avg=%.2f ms, min=%.2f ms" % (rebind_avg, rebind_min), flush=True)
    print("  Per call: ~%.1f us" % (rebind_avg / 196 * 1000), flush=True)
    print(flush=True)

    # ─── 4. CPU ops ───
    print("--- 4. CPU ops (numpy proxy, ARM A55 @2.2GHz) ---", flush=True)
    print("  Note: NEON C code is ~2-5x faster than numpy for these ops", flush=True)
    cpu = bench_cpu_ops()
    for name, us in cpu.items():
        print("  %s: %8.1f us" % (name.ljust(25), us), flush=True)
    print(flush=True)

    # ─── 5. lm_head ───
    print("--- 5. lm_head CPU matmul: (1024) x (151936, 1024)^T ---", flush=True)
    lm_avg, lm_min, argmax_ms = bench_lm_head()
    print("  lm_head: avg=%.1f ms, min=%.1f ms" % (lm_avg, lm_min), flush=True)
    print("  argmax(151936): %.2f ms" % argmax_ms, flush=True)
    print(flush=True)

    # ─── 6. C library end-to-end ───
    print("--- 6. C library step (actual implementation) ---", flush=True)
    c_results = bench_c_lib()
    print(flush=True)

    # ═══ SUMMARY ═══
    print("=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    raw_layer = sum(v[0] for v in fp16_raw.values())
    full_layer = sum(v[0] for v in fp16_full.values())

    print("Per-layer NPU matmul (7 projections, FP16):", flush=True)
    print("  Raw run only:    %5.2f ms  x28 = %6.1f ms" % (raw_layer, raw_layer * 28), flush=True)
    print("  Full round-trip: %5.2f ms  x28 = %6.1f ms" % (full_layer, full_layer * 28), flush=True)
    print(flush=True)

    # CPU ops per layer
    cpu_layer_us = (
        2 * cpu["rmsnorm_1024"] +
        cpu["qk_norm_24heads"] +
        cpu["rope_24heads"] +
        cpu["attn_seq20"] +
        cpu["silu_mul_3072"] +
        2 * cpu["vec_add_1024"] +
        3 * cpu["memcpy_4KB"]
    )
    cpu_layer_ms = cpu_layer_us / 1000

    # Conversion overhead per layer (7 FP32->FP16 inputs + varies outputs)
    conv_layer_us = 7 * cpu["fp32_to_fp16_1024"] + 3 * cpu["fp16_to_fp32_3072"]
    conv_layer_ms = conv_layer_us / 1000

    print("Per-layer CPU ops (numpy): %.2f ms  x28 = %.1f ms" % (cpu_layer_ms, cpu_layer_ms * 28), flush=True)
    neon_factor = 3.0  # NEON typically 3x faster
    neon_est = cpu_layer_ms * 28 / neon_factor
    print("  NEON estimate (~%.0fx): %.2f ms/layer  x28 = %.1f ms" % (neon_factor, cpu_layer_ms/neon_factor, neon_est), flush=True)
    print(flush=True)

    print("Per-layer conversion overhead: %.2f ms  x28 = %.1f ms" % (conv_layer_ms, conv_layer_ms * 28), flush=True)
    print(flush=True)

    print("Other overheads:", flush=True)
    print("  B rebind (196 calls):  %.2f ms" % rebind_avg, flush=True)
    print("  lm_head (CPU):         %.1f ms" % lm_avg, flush=True)
    print("  argmax:                %.2f ms" % argmax_ms, flush=True)
    print(flush=True)

    # Theoretical totals
    total_numpy = full_layer * 28 + cpu_layer_ms * 28 + lm_avg + argmax_ms
    total_neon = full_layer * 28 + neon_est + lm_avg + argmax_ms

    print("Theoretical totals (single-core, FP16):", flush=True)
    print("  %-30s %8s" % ("Component", "Time"), flush=True)
    print("  %-30s %7.1f ms" % ("NPU matmul (28 layers)", full_layer * 28), flush=True)
    print("  %-30s %7.1f ms" % ("CPU ops (numpy)", cpu_layer_ms * 28), flush=True)
    print("  %-30s %7.1f ms" % ("CPU ops (NEON est.)", neon_est), flush=True)
    print("  %-30s %7.1f ms" % ("B rebind", rebind_avg), flush=True)
    print("  %-30s %7.1f ms" % ("lm_head", lm_avg), flush=True)
    print("  %-30s %7.2f ms" % ("argmax", argmax_ms), flush=True)
    print("  " + "-" * 42, flush=True)
    print("  %-30s %7.1f ms" % ("Total (numpy CPU)", total_numpy), flush=True)
    print("  %-30s %7.1f ms" % ("Total (NEON CPU est.)", total_neon), flush=True)
    print(flush=True)

    # C lib measured
    if c_results:
        print("C library measured:", flush=True)
        for label, vals in c_results.items():
            print("  %-20s avg=%.1f ms  median=%.1f ms  min=%.1f ms" % (
                label, vals["avg"], vals["median"], vals["min"]), flush=True)
        print(flush=True)

    # Compare
    print("Reference:", flush=True)
    print("  RKLLM (W4A16, dual-core, fused):  ~16 ms/token", flush=True)
    print(flush=True)

    # Breakdown
    print("Bottleneck analysis (FP16, single-core, NEON est.):", flush=True)
    pcts = [
        ("NPU matmul", full_layer * 28),
        ("CPU ops", neon_est),
        ("lm_head", lm_avg),
        ("B rebind", rebind_avg),
    ]
    for name, ms in pcts:
        print("  %-20s %6.1f ms  (%4.1f%%)" % (name, ms, ms / total_neon * 100), flush=True)
    print(flush=True)

    print("Optimization roadmap:", flush=True)
    print("  1. Move lm_head to NPU (%.0f ms saved)" % (lm_avg - 2), flush=True)
    print("     (1,1024)x(151936,1024)^T = ~74 tiles of (1,1024,2048) = ~%.0f ms" % (
        fp16_raw["q_proj"][0] * 74), flush=True)
    print("  2. INT4 quantization: ~30%% less NPU time (based on RKLLM W4A16)", flush=True)
    print("  3. Dual-core NPU: ~1.4x speedup on matmul portion", flush=True)
    print("  4. Eliminate FP32<->FP16: keep entire pipeline in FP16", flush=True)
    print("  5. Fuse small ops: combine norm+projection, reduce kernel launches", flush=True)
    print("  6. RKLLM gap: fused attention+projection, on-chip weight caching,", flush=True)
    print("     operator fusion, and optimized memory access patterns", flush=True)


if __name__ == "__main__":
    main()

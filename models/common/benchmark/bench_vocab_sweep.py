#!/usr/bin/env python3
"""Benchmark multiple RKLLM talker models with different vocab sizes."""
import ctypes, time, sys, os, json
import numpy as np

# ==================== RKLLM Structures ====================
RKLLM_Handle_t = ctypes.c_void_p

class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType:
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2

class RKLLMInferMode:
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
    ]

RKLLM_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

token_count = 0
first_token_time = 0
token_ids = []

def callback(result, userdata, state):
    global token_count, first_token_time, token_ids
    if state == LLMCallState.RKLLM_RUN_NORMAL:
        token_count += 1
        if token_count == 1:
            first_token_time = time.perf_counter()
        if result and result.contents.token_id >= 0:
            token_ids.append(result.contents.token_id)

cb = RKLLM_CALLBACK(callback)

def bench_model(model_path, rkllm_lib):
    global token_count, first_token_time, token_ids

    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"Model: {os.path.basename(model_path)} ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")

    param = rkllm_lib.rkllm_createDefaultParam()
    param.model_path = model_path.encode()
    param.max_context_len = 1024
    param.max_new_tokens = 256
    param.top_k = 1
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.skip_special_token = False
    param.is_async = False
    param.extend_param.embed_flash = 0

    handle = RKLLM_Handle_t()
    init_start = time.perf_counter()
    ret = rkllm_lib.rkllm_init(ctypes.byref(handle), ctypes.byref(param), cb)
    init_time = time.perf_counter() - init_start

    if ret != 0:
        print(f"  Init FAILED (ret={ret})")
        return None
    print(f"  Init: {init_time:.1f}s")

    HIDDEN_SIZE = 1024
    N_PREFILL = 50
    np.random.seed(42)
    embeddings = np.random.randn(N_PREFILL, HIDDEN_SIZE).astype(np.float32) * 0.1
    embed_flat = embeddings.flatten()
    embed_array = (ctypes.c_float * len(embed_flat))(*embed_flat)

    rkllm_input = RKLLMInput()
    rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_EMBED
    rkllm_input.input_data.embed_input.embed = embed_array
    rkllm_input.input_data.embed_input.n_tokens = N_PREFILL

    infer_param = RKLLMInferParam()
    infer_param.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
    infer_param.keep_history = 0

    results = []
    NUM_RUNS = 3
    for run in range(NUM_RUNS):
        token_count = 0
        first_token_time = 0
        token_ids = []

        start = time.perf_counter()
        ret = rkllm_lib.rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)
        end = time.perf_counter()

        prefill_ms = (first_token_time - start) * 1000 if first_token_time > 0 else 0
        decode_tokens = token_count - 1 if token_count > 1 else 0
        decode_time = end - first_token_time if first_token_time > 0 else 0

        if decode_tokens > 0:
            ms_per_tok = decode_time / decode_tokens * 1000
            tok_per_sec = decode_tokens / decode_time
        else:
            ms_per_tok = 0
            tok_per_sec = 0

        print(f"  Run {run + 1}: {token_count} tokens, prefill={prefill_ms:.1f}ms, "
              f"decode={decode_tokens}tok @ {tok_per_sec:.1f}tok/s ({ms_per_tok:.1f}ms/tok)")

        results.append({
            "total_tokens": token_count,
            "prefill_ms": prefill_ms,
            "decode_tokens": decode_tokens,
            "decode_tok_per_sec": tok_per_sec,
            "decode_ms_per_tok": ms_per_tok,
        })

    rkllm_lib.rkllm_destroy(handle)

    # Average of runs (skip first as warmup)
    avg_runs = results[1:] if len(results) > 1 else results
    avg = {
        "size_mb": size_mb,
        "init_time_s": init_time,
        "prefill_ms": sum(r["prefill_ms"] for r in avg_runs) / len(avg_runs),
        "decode_tok_per_sec": sum(r["decode_tok_per_sec"] for r in avg_runs) / len(avg_runs),
        "decode_ms_per_tok": sum(r["decode_ms_per_tok"] for r in avg_runs) / len(avg_runs),
        "total_tokens": avg_runs[0]["total_tokens"],
    }
    return avg


def main():
    models = sorted([f for f in os.listdir("/home/cat") if f.startswith("talker_v") and f.endswith(".rkllm")])

    if not models:
        print("No talker_v*.rkllm models found in /home/cat/")
        sys.exit(1)

    print(f"Found {len(models)} models to benchmark")

    ctypes.CDLL("librknnrt.so", mode=ctypes.RTLD_GLOBAL)
    rkllm = ctypes.CDLL("/usr/lib/librkllmrt.so")
    rkllm.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), RKLLM_CALLBACK]
    rkllm.rkllm_init.restype = ctypes.c_int
    rkllm.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm.rkllm_run.restype = ctypes.c_int
    rkllm.rkllm_destroy.argtypes = [RKLLM_Handle_t]

    all_results = {}
    for model_file in models:
        path = f"/home/cat/{model_file}"
        vocab = model_file.split("_v")[1].split("_")[0]
        result = bench_model(path, rkllm)
        if result:
            all_results[vocab] = result

    print(f"\n\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}")
    print(f"{'Vocab':>10} {'Size(MB)':>10} {'Init(s)':>8} {'Prefill(ms)':>12} {'Decode(tok/s)':>14} {'ms/tok':>8}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 8} {'-' * 12} {'-' * 14} {'-' * 8}")
    for vocab in sorted(all_results.keys(), key=lambda x: int(x)):
        r = all_results[vocab]
        print(f"{vocab:>10} {r['size_mb']:>10.1f} {r['init_time_s']:>8.1f} {r['prefill_ms']:>12.1f} {r['decode_tok_per_sec']:>14.1f} {r['decode_ms_per_tok']:>8.1f}")

    with open("/home/cat/vocab_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to /home/cat/vocab_sweep_results.json")


if __name__ == "__main__":
    main()

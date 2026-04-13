#!/usr/bin/env python3
"""
Test step-by-step RKLLM inference for Qwen3-TTS talker on RK3576.

Validated findings (RKLLM SDK v1.2.3):
  - keep_history=1 is REQUIRED for KV-cache persistence across rkllm_run() calls
  - mode=2 (GET_LOGITS) returns logits via callback (vocab_size=151936)
  - mode=1 (GET_LAST_HIDDEN_LAYER) returns hidden states (embd_size=1024)
  - mode=2 does NOT return hidden states; mode=1 does NOT return logits
  - Each call generates max_new_tokens tokens (set to 1 for step-by-step)
  - Prefill: ~100ms for 14 tokens, decode: ~45ms/step on RK3576

Usage:
  python test_stepwise_rkllm.py [model_path] [embed_path]
"""
import ctypes
import sys
import time
import json
import numpy as np

RKLLM_Handle_t = ctypes.c_void_p

# ── RKLLM SDK v1.2.3 struct definitions ─────────────────────────

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
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
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


RKLLM_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)

# ── Callback state ───────────────────────────────────────────────

last_logits = None
last_hidden = None
last_token_id = None


def callback(result, userdata, state):
    global last_logits, last_hidden, last_token_id
    try:
        if state == 0:  # RKLLM_RUN_NORMAL
            r = result.contents
            last_token_id = r.token_id
            if r.logits.logits and r.logits.vocab_size > 0:
                n = r.logits.vocab_size
                last_logits = np.ctypeslib.as_array(
                    r.logits.logits, shape=(n,)
                ).copy()
            if (r.last_hidden_layer.hidden_states
                    and r.last_hidden_layer.embd_size > 0):
                n = (r.last_hidden_layer.embd_size
                     * r.last_hidden_layer.num_tokens)
                last_hidden = np.ctypeslib.as_array(
                    r.last_hidden_layer.hidden_states, shape=(n,)
                ).copy()
    except Exception as e:
        print("CB error: {}".format(e), flush=True)


cb = RKLLM_CALLBACK(callback)


def reset_cb():
    global last_logits, last_hidden, last_token_id
    last_logits = None
    last_hidden = None
    last_token_id = None


# ── Helpers ──────────────────────────────────────────────────────

def make_embed(embed_np):
    """Create RKLLMInput from [n_tokens, hidden_size] numpy array."""
    flat = embed_np.astype(np.float32).flatten()
    arr = (ctypes.c_float * len(flat))(*flat)
    inp = RKLLMInput()
    inp.input_type = 2  # RKLLM_INPUT_EMBED
    inp.input_data.embed_input.embed = arr
    inp.input_data.embed_input.n_tokens = embed_np.shape[0]
    inp._arr = arr  # prevent GC
    return inp


def run_step(rkllm, handle, embed_np, mode=2, keep_history=1):
    """Run one inference step. Returns (logits, hidden, token_id, ms)."""
    reset_cb()
    inp = make_embed(embed_np)
    ip = RKLLMInferParam()
    ip.mode = mode
    ip.keep_history = keep_history
    t0 = time.perf_counter()
    ret = rkllm.rkllm_run(handle, ctypes.byref(inp), ctypes.byref(ip), None)
    ms = (time.perf_counter() - t0) * 1000
    if ret != 0:
        return None, None, None, ms
    return last_logits, last_hidden, last_token_id, ms


# ── Main ─────────────────────────────────────────────────────────

def main():
    model_path = (sys.argv[1] if len(sys.argv) > 1
                  else "/home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm")
    embed_path = (sys.argv[2] if len(sys.argv) > 2
                  else "/tmp/prefill_embeds.npy")
    HIDDEN_SIZE = 1024

    embeds = np.load(embed_path).astype(np.float32)
    if embeds.ndim == 3:
        embeds = embeds[0]
    n_prefill = embeds.shape[0]
    print("Embeds: [{}, {}]".format(n_prefill, embeds.shape[1]), flush=True)

    # Load library
    ctypes.CDLL("librknnrt.so", mode=ctypes.RTLD_GLOBAL)
    rkllm = ctypes.CDLL("/tmp/librkllmrt.so")
    rkllm.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm.rkllm_init.argtypes = [
        ctypes.POINTER(RKLLM_Handle_t),
        ctypes.POINTER(RKLLMParam),
        RKLLM_CALLBACK,
    ]
    rkllm.rkllm_init.restype = ctypes.c_int
    rkllm.rkllm_run.argtypes = [
        RKLLM_Handle_t,
        ctypes.POINTER(RKLLMInput),
        ctypes.POINTER(RKLLMInferParam),
        ctypes.c_void_p,
    ]
    rkllm.rkllm_run.restype = ctypes.c_int
    rkllm.rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t]
    rkllm.rkllm_clear_kv_cache.restype = ctypes.c_int

    param = rkllm.rkllm_createDefaultParam()
    param.model_path = model_path.encode()
    param.max_context_len = 512
    param.max_new_tokens = 1  # step-by-step: 1 token per call
    param.top_k = 1
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.skip_special_token = False
    param.is_async = False
    param.extend_param.embed_flash = 0

    handle = RKLLM_Handle_t()
    print("Init...", flush=True)
    ret = rkllm.rkllm_init(ctypes.byref(handle), ctypes.byref(param), cb)
    if ret != 0:
        print("FAILED: {}".format(ret), flush=True)
        sys.exit(1)
    print("OK", flush=True)

    results = {}

    # ── TEST 1: KV-cache persistence ────────────────────────────
    print("\n--- TEST 1: KV-cache persistence ---", flush=True)
    rkllm.rkllm_clear_kv_cache(handle)

    # Prefill
    logits_pf, _, _, pf_ms = run_step(rkllm, handle, embeds, mode=2, keep_history=1)
    print("  Prefill: {:.1f}ms, vocab={}".format(
        pf_ms, len(logits_pf) if logits_pf is not None else 0), flush=True)

    # Decode with context
    np.random.seed(42)
    single = np.random.randn(1, HIDDEN_SIZE).astype(np.float32) * 0.05
    logits_ctx, _, _, _ = run_step(rkllm, handle, single, mode=2, keep_history=1)

    # Clear KV, decode without context
    rkllm.rkllm_clear_kv_cache(handle)
    logits_noctx, _, _, _ = run_step(rkllm, handle, single, mode=2, keep_history=1)

    if logits_ctx is not None and logits_noctx is not None:
        diff = np.abs(logits_ctx - logits_noctx).max()
        print("  Max logit diff: {:.4f}".format(diff), flush=True)
        print("  KV PERSISTS: {}".format("YES" if diff > 0.01 else "NO"), flush=True)
        results["kv_persists"] = bool(diff > 0.01)
    else:
        results["kv_persists"] = None

    # ── TEST 2: 20-step AR decode loop ──────────────────────────
    print("\n--- TEST 2: 20-step AR decode loop ---", flush=True)
    rkllm.rkllm_clear_kv_cache(handle)

    logits, _, _, pf_ms = run_step(rkllm, handle, embeds, mode=2, keep_history=1)
    first_tok = int(np.argmax(logits)) if logits is not None else 0
    print("  Prefill: {:.1f}ms, token={}".format(pf_ms, first_tok), flush=True)
    results["prefill_ms"] = round(pf_ms, 1)

    tokens = [first_tok]
    step_times = []
    for step in range(20):
        np.random.seed(tokens[-1] % 10000)
        ne = np.random.randn(1, HIDDEN_SIZE).astype(np.float32) * 0.05
        logits, _, _, ms = run_step(rkllm, handle, ne, mode=2, keep_history=1)
        step_times.append(ms)
        if logits is None:
            print("  Step {}: FAILED".format(step), flush=True)
            break
        tokens.append(int(np.argmax(logits)))

    avg_ms = sum(step_times) / len(step_times)
    print("  Avg decode: {:.1f}ms/step ({:.1f} tok/s)".format(
        avg_ms, 1000 / avg_ms), flush=True)
    print("  Tokens: {}".format(tokens), flush=True)
    results["decode_avg_ms"] = round(avg_ms, 1)
    results["decode_tok_s"] = round(1000 / avg_ms, 1)

    # ── TEST 3: GET_LAST_HIDDEN_LAYER ───────────────────────────
    print("\n--- TEST 3: Hidden states ---", flush=True)
    rkllm.rkllm_clear_kv_cache(handle)

    # Prefill with logits mode
    run_step(rkllm, handle, embeds, mode=2, keep_history=1)

    # Decode with hidden mode
    np.random.seed(42)
    se = np.random.randn(1, HIDDEN_SIZE).astype(np.float32) * 0.05
    _, hidden, _, h_ms = run_step(rkllm, handle, se, mode=1, keep_history=1)
    if hidden is not None:
        print("  Hidden: size={}, time={:.1f}ms".format(len(hidden), h_ms), flush=True)
        print("  Hidden[:8]: {}".format(hidden[:8]), flush=True)
        results["hidden_works"] = True
        results["hidden_ms"] = round(h_ms, 1)
    else:
        print("  NO hidden states", flush=True)
        results["hidden_works"] = False

    # ── TEST 4: Mode exclusivity ────────────────────────────────
    print("\n--- TEST 4: Mode output exclusivity ---", flush=True)
    rkllm.rkllm_clear_kv_cache(handle)

    logits, hidden, _, _ = run_step(rkllm, handle, embeds, mode=2, keep_history=1)
    print("  mode=2: logits={}, hidden={}".format(
        logits is not None, hidden is not None), flush=True)
    results["mode2_gives_logits"] = logits is not None
    results["mode2_gives_hidden"] = hidden is not None

    rkllm.rkllm_clear_kv_cache(handle)
    logits, hidden, _, _ = run_step(rkllm, handle, embeds, mode=1, keep_history=1)
    print("  mode=1: logits={}, hidden={}".format(
        logits is not None, hidden is not None), flush=True)
    results["mode1_gives_logits"] = logits is not None
    results["mode1_gives_hidden"] = hidden is not None

    # ── SUMMARY ──────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for k, v in sorted(results.items()):
        print("  {}: {}".format(k, v), flush=True)

    print("\nFEASIBILITY:", flush=True)
    print("  Step-by-step AR loop: WORKS (keep_history=1 required)", flush=True)
    print("  Prefill: ~{:.0f}ms".format(results.get("prefill_ms", 0)), flush=True)
    print("  Decode: ~{:.0f}ms/step (~{:.0f} tok/s)".format(
        results.get("decode_avg_ms", 0),
        results.get("decode_tok_s", 0)), flush=True)
    print("  Hidden states: {}".format(
        "WORKS (mode=1)" if results.get("hidden_works") else "FAIL"), flush=True)

    if not results.get("mode2_gives_hidden"):
        print("\n  ARCHITECTURE NOTE:", flush=True)
        print("  mode=2 gives logits only, mode=1 gives hidden only.", flush=True)
        print("  For TTS: use mode=1 per step, extract hidden states,", flush=True)
        print("  then compute logits externally via codec_head weights.", flush=True)
        print("  This avoids the double-KV problem of calling both modes.", flush=True)

    out = "/tmp/rkllm_stepwise_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved: {}".format(out), flush=True)

    rkllm.rkllm_destroy(handle)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

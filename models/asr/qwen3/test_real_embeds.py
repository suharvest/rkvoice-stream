#!/usr/bin/env python3
"""
Test RKLLM talker with real prefill embeddings from GPU reference.
Feeds the same prefill embeddings that GPU used, and compares codec tokens.
"""
import ctypes, time, sys, os, json
import numpy as np

RKLLM_Handle_t = ctypes.c_void_p

class LLMCallState:
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

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

def main():
    global token_count, first_token_time, token_ids

    model_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/talker_w4a16_rk3576.rkllm"
    embed_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/prefill_embeds.npy"

    print("Model: {}".format(model_path))
    print("Embeddings: {}".format(embed_path))

    # Load real prefill embeddings
    embeds = np.load(embed_path).astype(np.float32)
    if embeds.ndim == 3:
        embeds = embeds[0]  # Remove batch dim: [1, N, 1024] -> [N, 1024]
    n_tokens, hidden_size = embeds.shape
    print("Embeddings shape: {} x {}".format(n_tokens, hidden_size))
    print("Embed stats: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        embeds.mean(), embeds.std(), embeds.min(), embeds.max()))

    ctypes.CDLL("librknnrt.so", mode=ctypes.RTLD_GLOBAL)
    rkllm = ctypes.CDLL("/usr/lib/librkllmrt.so")
    rkllm.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), RKLLM_CALLBACK]
    rkllm.rkllm_init.restype = ctypes.c_int
    rkllm.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm.rkllm_run.restype = ctypes.c_int
    rkllm.rkllm_destroy.argtypes = [RKLLM_Handle_t]

    param = rkllm.rkllm_createDefaultParam()
    param.model_path = model_path.encode()
    param.max_context_len = 1024
    param.max_new_tokens = 512
    param.top_k = 1  # greedy
    param.temperature = 1.0
    param.repeat_penalty = 1.0
    param.skip_special_token = False
    param.is_async = False
    param.extend_param.embed_flash = 0

    handle = RKLLM_Handle_t()
    print("Loading model...")
    ret = rkllm.rkllm_init(ctypes.byref(handle), ctypes.byref(param), cb)
    if ret != 0:
        print("Init FAILED: {}".format(ret))
        sys.exit(1)

    embed_flat = embeds.flatten()
    embed_array = (ctypes.c_float * len(embed_flat))(*embed_flat)

    rkllm_input = RKLLMInput()
    rkllm_input.input_type = 2  # EMBED
    rkllm_input.input_data.embed_input.embed = embed_array
    rkllm_input.input_data.embed_input.n_tokens = n_tokens

    infer_param = RKLLMInferParam()
    infer_param.mode = 0
    infer_param.keep_history = 0

    print("Running inference with real embeddings...")
    token_count = 0
    first_token_time = 0
    token_ids = []

    start = time.perf_counter()
    ret = rkllm.rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)
    end = time.perf_counter()

    prefill_ms = (first_token_time - start) * 1000 if first_token_time > 0 else 0
    decode_tokens = token_count - 1 if token_count > 1 else 0
    decode_time = end - first_token_time if first_token_time > 0 else 0

    print("\nResults:")
    print("  Total tokens: {}".format(token_count))
    print("  Prefill: {:.1f}ms".format(prefill_ms))
    if decode_tokens > 0:
        print("  Decode: {} tokens in {:.1f}ms ({:.1f} tok/s)".format(
            decode_tokens, decode_time * 1000, decode_tokens / decode_time))
    print("  Token IDs ({} total): {}".format(len(token_ids), token_ids[:50]))
    print("  Token range: [{}, {}]".format(min(token_ids) if token_ids else -1, max(token_ids) if token_ids else -1))

    # Check if tokens are in valid codec range (0-3071 for the original model)
    valid_range = sum(1 for t in token_ids if 0 <= t < 3072)
    print("  Tokens in valid codec range [0, 3072): {}/{}".format(valid_range, len(token_ids)))

    rkllm.rkllm_destroy(handle)

    # Save results
    result = {
        "model": os.path.basename(model_path),
        "n_prefill_tokens": n_tokens,
        "total_generated": token_count,
        "token_ids": token_ids,
        "prefill_ms": round(prefill_ms, 1),
        "valid_codec_range": valid_range,
    }
    with open("/tmp/rkllm_real_embed_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to /tmp/rkllm_real_embed_result.json")


if __name__ == "__main__":
    main()

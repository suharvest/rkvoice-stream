"""ctypes mirror of the RKLLM C ABI (``rkllm.h``) -- the single definition.

Every Python binding of ``librkllmrt.so`` in this package imports its structs,
callback types and constants from here instead of declaring its own copy
(``backends/asr/qwen3/decoder.py``, ``runtime/rkllm_wrapper.py``).

**Requires librkllmrt v1.3.0** (tag ``release-v1.3.0`` of
https://github.com/airockchip/rknn-llm).  This is a deliberate breaking change:
the 1.2.x ABI is not supported.  A 1.2.x runtime does not fail cleanly -- the
layouts below differ from it, so it crashes (SIGSEGV) inside ``rkllm_init`` /
``rkllm_run``.  What changed in 1.3.0 relative to 1.2.x:

  * ``RKLLMParam``: ``img_start/img_end/img_content`` removed,
    ``ignore_eos_token`` added.
  * ``RKLLMMultiModalInput``: flat image fields became nested ``image`` /
    ``video`` sub-structs (so ``RKLLMInput`` is 160 bytes).
  * ``RKLLMInferParam``: ``sampling_params`` and ``max_new_tokens`` added.
  * ``rkllm_init()``: the third argument is a pointer to an ``RKLLMCallback``
    struct, not a bare ``LLMResultCallback`` function pointer.

Only the part of the header this project uses is mirrored.  Layouts assume an
LP64 target (RK3576 / RK3588, aarch64); ``tests/test_rkllm_abi.py`` pins every
size and field offset to the values a C++ compiler computes from the real
header.  If you change a struct here, re-derive those numbers from ``rkllm.h``
-- do not adjust them to match the Python.

This module must stay free of third-party imports (stdlib ``ctypes`` only) so
the layout test runs on any machine, without an NPU or ``numpy``.
"""

import ctypes

RKLLM_Handle_t = ctypes.c_void_p

# ---- Constants (RKLLMInputType / RKLLMInferMode / LLMCallState) -------------

RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS = 2

RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3


# ---- Init parameters --------------------------------------------------------

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
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


# ---- Run input --------------------------------------------------------------

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class _RKLLMImageInput(ctypes.Structure):
    _fields_ = [
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_start", ctypes.c_char_p),
        ("image_end", ctypes.c_char_p),
        ("image_content", ctypes.c_char_p),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class _RKLLMVideoInput(ctypes.Structure):
    _fields_ = [
        ("video_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_frame_tokens", ctypes.c_size_t),
        ("n_frame_per_video", ctypes.c_size_t),
        ("n_video", ctypes.c_size_t),
        ("video_start", ctypes.c_char_p),
        ("video_end", ctypes.c_char_p),
        ("video_content", ctypes.c_char_p),
        ("frame_width", ctypes.c_size_t),
        ("frame_height", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image", _RKLLMImageInput),
        ("video", _RKLLMVideoInput),
    ]


class RKLLMInputUnion(ctypes.Union):
    # The union is as large as its biggest member (the 144-byte multimodal
    # input), which is what makes ``RKLLMInput`` 160 bytes like the C side.
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


# ---- Run parameters ---------------------------------------------------------

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMSamplingParam(ctypes.Structure):
    _fields_ = [
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("sampling_params", ctypes.POINTER(RKLLMSamplingParam)),
        ("keep_history", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int32),
    ]


# ---- Results ----------------------------------------------------------------

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


# ---- Callbacks --------------------------------------------------------------

# int (*LLMResultCallback)(RKLLMResult *result, void *userdata, LLMCallState state)
# The int return value is a status the runtime acts on (0 = continue,
# 1 = pause, 2 = release the output buffer), so a Python callback MUST return
# an int -- returning None leaves an undefined value in the return register.
RKLLM_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)

# int (*LLMTokenizerCallback)(void *userdata, const char *text, int32_t text_len,
#                             int32_t *tokens, int32_t n_tokens_max)
RKLLM_TOKENIZER_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
)

# int (*LLMGetEmbedCallback)(void *userdata, int32_t *tokens, uint64_t num_tokens,
#                            void *embed, uint64_t len)
RKLLM_GET_EMBED_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
    ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64,
)


class RKLLMCallback(ctypes.Structure):
    """Passed *by pointer* as the third argument of ``rkllm_init()``."""

    _fields_ = [
        ("result_callback", RKLLM_CALLBACK),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", RKLLM_TOKENIZER_CALLBACK),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", RKLLM_GET_EMBED_CALLBACK),
        ("embed_userdata", ctypes.c_void_p),
    ]


# ---- Helpers ----------------------------------------------------------------

def make_callback_struct(result_callback) -> RKLLMCallback:
    """Build the ``RKLLMCallback`` for ``rkllm_init(&handle, &param, &cb)``.

    ``result_callback`` is an ``RKLLM_CALLBACK`` instance.  The tokenizer and
    embedding callbacks are optional (only needed for models without an
    internal tokenizer / embedding layer) and are left NULL; ctypes
    zero-initialises every field not set here.

    The caller must keep both the returned struct and ``result_callback``
    alive for as long as the RKLLM handle exists -- the runtime may keep the
    pointer instead of copying the struct.
    """
    cb = RKLLMCallback()
    cb.result_callback = result_callback
    return cb


def make_infer_param(mode: int = RKLLM_INFER_GENERATE, keep_history: int = 0,
                     prompt_cache_params: RKLLMPromptCacheParam = None
                     ) -> RKLLMInferParam:
    """Build an ``RKLLMInferParam`` that defers to the init-time settings.

    ``sampling_params`` stays NULL (use the sampling settings from
    ``RKLLMParam``) and ``max_new_tokens`` stays 0 (``<= 0`` means use the
    ``RKLLMParam`` value); ``lora_params`` stays NULL.
    """
    p = RKLLMInferParam()
    p.mode = mode
    p.keep_history = keep_history
    if prompt_cache_params is not None:
        p.prompt_cache_params = ctypes.pointer(prompt_cache_params)
    return p

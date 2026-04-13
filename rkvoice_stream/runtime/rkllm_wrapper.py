"""RKLLM ctypes wrapper for Qwen3-TTS talker model.

Provides a Python interface to the RKLLM C library for step-by-step
inference with embedding inputs (RKLLM_INPUT_EMBED) and two output modes:
  - GET_LAST_HIDDEN_LAYER (mode=1): returns hidden states for code_predictor
  - GET_LOGITS (mode=2): returns logits for primary token sampling

Thread safety: NOT thread-safe. All calls should be from a single thread.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── RKLLM Constants ──────────────────────────────────────────────

RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2

RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS = 2

RKLLM_RUN_NORMAL = 0
RKLLM_RUN_WAITING = 1
RKLLM_RUN_FINISH = 2
RKLLM_RUN_ERROR = 3


# ── ctypes Struct Definitions (RKLLM SDK v1.2.3) ────────────────

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
RKLLM_Handle_t = ctypes.c_void_p


class RKLLMTalker:
    """Wrapper for RKLLM talker model with step-by-step inference."""

    def __init__(
        self,
        model_path: str,
        rkllm_lib: str = "/usr/lib/librkllmrt.so",
        rknn_lib: str = "librknnrt.so",
        max_context_len: int = 512,
        max_new_tokens: int = 1,
    ):
        self._model_path = model_path
        self._handle = RKLLM_Handle_t()
        self._collected_hidden: Optional[np.ndarray] = None
        self._collected_logits: Optional[np.ndarray] = None
        self._callback_error: Optional[str] = None
        self._vocab_size: int = 0
        self._embd_size: int = 0

        # Load shared libraries
        ctypes.CDLL(rknn_lib, mode=ctypes.RTLD_GLOBAL)
        self._lib = ctypes.CDLL(rkllm_lib)

        # Setup function signatures
        self._lib.rkllm_createDefaultParam.restype = RKLLMParam
        self._lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam),
            RKLLM_CALLBACK,
        ]
        self._lib.rkllm_init.restype = ctypes.c_int
        self._lib.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.c_void_p,
        ]
        self._lib.rkllm_run.restype = ctypes.c_int
        self._lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_destroy.restype = ctypes.c_int
        self._lib.rkllm_clear_kv_cache.argtypes = [RKLLM_Handle_t]
        self._lib.rkllm_clear_kv_cache.restype = ctypes.c_int

        # Create callback (prevent GC)
        self._cb = RKLLM_CALLBACK(self._callback_fn)

        # Init model
        param = self._lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode()
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.top_k = 1
        param.temperature = 1.0
        param.repeat_penalty = 1.0
        param.skip_special_token = False
        param.is_async = False
        param.extend_param.embed_flash = 0

        t0 = time.perf_counter()
        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param), self._cb
        )
        elapsed = time.perf_counter() - t0
        if ret != 0:
            raise RuntimeError(f"RKLLM init failed: ret={ret}")
        logger.info("RKLLM talker loaded in %.1fs", elapsed)

    def _callback_fn(self, result_ptr, userdata, state):
        if state == RKLLM_RUN_ERROR:
            self._callback_error = "RKLLM_RUN_ERROR"
            return
        if state != RKLLM_RUN_NORMAL:
            return

        r = result_ptr.contents

        # Capture hidden states
        if r.last_hidden_layer.hidden_states and r.last_hidden_layer.embd_size > 0:
            n = r.last_hidden_layer.embd_size * r.last_hidden_layer.num_tokens
            arr = np.ctypeslib.as_array(
                r.last_hidden_layer.hidden_states, shape=(n,)
            ).copy()
            self._collected_hidden = arr.reshape(
                r.last_hidden_layer.num_tokens, r.last_hidden_layer.embd_size
            )
            self._embd_size = r.last_hidden_layer.embd_size

        # Capture logits
        if r.logits.logits and r.logits.vocab_size > 0:
            n = r.logits.vocab_size * r.logits.num_tokens
            arr = np.ctypeslib.as_array(r.logits.logits, shape=(n,)).copy()
            self._collected_logits = arr.reshape(
                r.logits.num_tokens, r.logits.vocab_size
            )
            self._vocab_size = r.logits.vocab_size

    def _reset(self):
        self._collected_hidden = None
        self._collected_logits = None
        self._callback_error = None

    def clear_kv_cache(self):
        self._lib.rkllm_clear_kv_cache(self._handle)

    def run_embed(
        self,
        embeddings: np.ndarray,
        mode: int = RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
        keep_history: int = 0,
    ) -> dict:
        """Run RKLLM with embedding input.

        Args:
            embeddings: [n_tokens, hidden_size] float32
            mode: RKLLM_INFER_GET_LAST_HIDDEN_LAYER (1) or RKLLM_INFER_GET_LOGITS (2)
            keep_history: 0 = clear after run, 1 = keep KV-cache

        Returns:
            dict with 'hidden' and/or 'logits' numpy arrays
        """
        assert embeddings.ndim == 2
        n_tokens, hidden_size = embeddings.shape
        flat = np.ascontiguousarray(embeddings, dtype=np.float32).flatten()
        c_arr = (ctypes.c_float * len(flat))(*flat)

        inp = RKLLMInput()
        inp.input_type = RKLLM_INPUT_EMBED
        inp.input_data.embed_input.embed = c_arr
        inp.input_data.embed_input.n_tokens = n_tokens

        infer_p = RKLLMInferParam()
        infer_p.mode = mode
        infer_p.lora_params = None
        infer_p.prompt_cache_params = None
        infer_p.keep_history = keep_history

        self._reset()
        ret = self._lib.rkllm_run(
            self._handle, ctypes.byref(inp), ctypes.byref(infer_p), None
        )
        if ret != 0:
            raise RuntimeError(f"rkllm_run failed: ret={ret}")
        if self._callback_error:
            raise RuntimeError(f"RKLLM callback error: {self._callback_error}")

        result = {}
        if self._collected_hidden is not None:
            result["hidden"] = self._collected_hidden
        if self._collected_logits is not None:
            result["logits"] = self._collected_logits
        return result

    def destroy(self):
        if self._handle:
            self._lib.rkllm_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.destroy()

"""RKLLM ctypes wrapper for Qwen3-TTS talker model.

Provides a Python interface to the RKLLM C library for step-by-step
inference with embedding inputs (RKLLM_INPUT_EMBED) and two output modes:
  - GET_LAST_HIDDEN_LAYER (mode=1): returns hidden states for code_predictor
  - GET_LOGITS (mode=2): returns logits for primary token sampling

Thread safety: NOT thread-safe. All calls should be from a single thread.

Requires librkllmrt.so **v1.3.0**; the 1.2.x C ABI is not supported.  The
ctypes structs live in ``rkvoice_stream.runtime.rkllm_abi``.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from typing import Optional

import numpy as np

from rkvoice_stream.runtime.rkllm_abi import (
    RKLLM_CALLBACK,
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
    RKLLM_INPUT_EMBED,
    RKLLM_INPUT_TOKEN,
    RKLLM_RUN_ERROR,
    RKLLM_RUN_NORMAL,
    RKLLM_Handle_t,
    RKLLMCallback,
    RKLLMInferParam,
    RKLLMInput,
    RKLLMParam,
    make_callback_struct,
    make_infer_param,
)

logger = logging.getLogger(__name__)


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
            ctypes.POINTER(RKLLMCallback),
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
        # 4 args per rkllm.h (handle, keep_system_prompt, start_pos*, end_pos*).
        # ASR decoder uses this signature; TTS used to use 1-arg which triggered
        # "start_pos and end_pos are only valid..." stderr warnings and left
        # garbage values in those slots.
        self._lib.rkllm_clear_kv_cache.argtypes = [
            RKLLM_Handle_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.rkllm_clear_kv_cache.restype = ctypes.c_int
        self._lib.rkllm_set_chat_template.argtypes = [
            RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
        ]
        self._lib.rkllm_set_chat_template.restype = ctypes.c_int

        # Create callback (prevent GC).  rkllm_init() takes an RKLLMCallback*;
        # keep the struct (and self._cb) alive for the lifetime of the handle.
        self._cb = RKLLM_CALLBACK(self._callback_fn)
        self._cb_struct = make_callback_struct(self._cb)

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
        # base_domain_id=1 lets RKLLM coexist with RKNN models (domain 0). Required on
        # RK3576/RK3588 — see project_rkllm_domain_coexist memory (2026-04-12).
        param.extend_param.base_domain_id = 1
        # External embedding inputs such as MOSS audio+text rows should be able
        # to bypass flash embedding lookup. Keep the historical ASR default but
        # allow MOSS parity probes to force 0.
        param.extend_param.embed_flash = int(os.environ.get("RKLLM_EMBED_FLASH", "1"))

        t0 = time.perf_counter()
        ret = self._lib.rkllm_init(
            ctypes.byref(self._handle), ctypes.byref(param),
            ctypes.byref(self._cb_struct)
        )
        elapsed = time.perf_counter() - t0
        if ret != 0:
            raise RuntimeError(f"RKLLM init failed: ret={ret}")

        if os.environ.get("RKLLM_DISABLE_CHAT_TEMPLATE", "0").strip().lower() in {"1", "true", "yes", "on"}:
            template_ret = self._lib.rkllm_set_chat_template(self._handle, b"", b"", b"")
            if template_ret != 0:
                raise RuntimeError(f"rkllm_set_chat_template failed: ret={template_ret}")
        logger.info("RKLLM talker loaded in %.1fs", elapsed)

    def _callback_fn(self, result_ptr, userdata, state):
        # Always returns 0 ("continue"): the runtime acts on this value
        # (1 = pause, 2 = release the output buffer); the data is copied below.
        if state == RKLLM_RUN_ERROR:
            self._callback_error = "RKLLM_RUN_ERROR"
            return 0
        if state != RKLLM_RUN_NORMAL:
            return 0

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
        return 0

    def _reset(self):
        self._collected_hidden = None
        self._collected_logits = None
        self._callback_error = None

    def clear_kv_cache(self):
        # 4-arg call: keep=0 (drop all), start/end=NULL.
        self._lib.rkllm_clear_kv_cache(self._handle, 0, None, None)

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
        # Empty role + thinking disabled: stops SDK from inserting a chat-style
        # role/thinking wrapper around our raw embeddings. Mirrors ASR decoder.
        inp.role = b""
        inp.enable_thinking = ctypes.c_bool(False)
        inp.input_type = RKLLM_INPUT_EMBED
        inp.input_data.embed_input.embed = c_arr
        inp.input_data.embed_input.n_tokens = n_tokens

        infer_p = make_infer_param(mode, keep_history)

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

    def run_tokens(
        self,
        token_ids: np.ndarray,
        mode: int = RKLLM_INFER_GET_LAST_HIDDEN_LAYER,
        keep_history: int = 0,
    ) -> dict:
        """Run RKLLM with token IDs using the model's internal embedding table."""
        token_ids = np.asarray(token_ids, dtype=np.int32).reshape(-1)
        c_arr = (ctypes.c_int32 * len(token_ids))(*token_ids)

        inp = RKLLMInput()
        inp.role = b""
        inp.enable_thinking = ctypes.c_bool(False)
        inp.input_type = RKLLM_INPUT_TOKEN
        inp.input_data.token_input.input_ids = c_arr
        inp.input_data.token_input.n_tokens = len(token_ids)

        infer_p = make_infer_param(mode, keep_history)

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

"""
RKLLM Decoder wrapper for Qwen3-ASR.

Uses EMBED input mode with the Qwen3 (non-VL) model.
Critical settings for correct operation:
  - rkllm_set_chat_template(handle, "", "", "")  -- disable built-in template
  - role = ""  -- empty role for EMBED input
  - embed_flash = 1
  - model must be exported as model_type="qwen3" (not qwen3_vl)
"""

import ctypes
import threading
import numpy as np
from .config import CPU_MASKS


# ==================== ctypes struct definitions ====================

RKLLM_Handle_t = ctypes.c_void_p

# Input types
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# Infer modes
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLM_INFER_GET_LOGITS = 2

# Callback states
LLM_RUN_NORMAL = 0
LLM_RUN_WAITING = 1
LLM_RUN_FINISH = 2
LLM_RUN_ERROR = 3


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


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModalInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
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


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
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
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat),
    ]


RKLLM_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int
)


# ==================== Decoder Class ====================

class RKLLMDecoder:
    """
    RKLLM decoder with EMBED input mode for ASR.
    
    Uses the Qwen3 model (model_type="qwen3", NOT qwen3_vl) with:
    - Empty chat template (disables built-in prompt wrapping)
    - Empty role for EMBED input
    - Flash embedding mode
    - Greedy decoding by default (top_k=1)
    """

    def __init__(self, model_path: str, lib_path: str,
                 max_context_len: int = 4096,
                 max_new_tokens: int = 500,
                 top_k: int = 1,
                 top_p: float = 1.0,
                 temperature: float = 1.0,
                 repeat_penalty: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 enabled_cpus: int = 2,
                 embed_flash: int = 1,
                 n_keep: int = -1,
                 callback_fn=None):
        """
        Initialize RKLLM decoder.
        
        Args:
            model_path: Path to .rkllm model file
            lib_path: Path to librkllmrt.so
            max_context_len: Maximum KV cache size
            max_new_tokens: Maximum tokens to generate per call
            top_k: Top-K sampling (1 = greedy)
            top_p: Nucleus sampling threshold
            temperature: Sampling temperature
            repeat_penalty: Repetition penalty
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            enabled_cpus: Number of CPU cores (2 or 4)
            embed_flash: Enable flash embedding (1 = yes)
            n_keep: Prefix KV cache length for keep_system_prompt
            callback_fn: Optional callback(text, is_finish) for streaming output
        """
        self.lib = ctypes.CDLL(str(lib_path))
        self.user_callback = callback_fn
        self._output_chunks = []
        self._perf = {}
        self._lock = threading.Lock()
        self._repeat_buf = []
        self._max_repeat = 6  # Abort after 6 repeated patterns
        self._aborted = False

        # Setup callback
        @RKLLM_CALLBACK
        def _cb(result, userdata, state):
            if state == LLM_RUN_NORMAL or state == LLM_RUN_WAITING:
                if result and result.contents.text:
                    text = result.contents.text.decode("utf-8", errors="replace")
                    if text not in ("<|im_end|>", "<|endoftext|>"):
                        self._output_chunks.append(text)
                        if self.user_callback:
                            self.user_callback(text, False)
                        # Pattern repetition detection:
                        # Checks for repeated 1-gram, 2-gram, or 3-gram patterns
                        self._repeat_buf.append(text)
                        buf = self._repeat_buf
                        should_abort = False
                        # Check n-gram patterns (n=1,2,3)
                        for n in (1, 2, 3):
                            need = n * self._max_repeat
                            if len(buf) >= need:
                                tail = buf[-need:]
                                pattern = tuple(tail[:n])
                                if all(tuple(tail[i:i+n]) == pattern
                                       for i in range(0, need, n)):
                                    should_abort = True
                                    break
                        if should_abort:
                            self._aborted = True
                            self.lib.rkllm_abort(self.handle)
            elif state == LLM_RUN_FINISH:
                if result:
                    self._perf = {
                        "prefill_time_ms": result.contents.perf.prefill_time_ms,
                        "prefill_tokens": result.contents.perf.prefill_tokens,
                        "generate_time_ms": result.contents.perf.generate_time_ms,
                        "generate_tokens": result.contents.perf.generate_tokens,
                        "memory_usage_mb": result.contents.perf.memory_usage_mb,
                    }
                if self.user_callback:
                    self.user_callback("", True)
            return 0

        self._cb = _cb

        # Build parameters
        param = RKLLMParam()
        param.model_path = str(model_path).encode()
        param.max_context_len = max_context_len
        param.max_new_tokens = max_new_tokens
        param.top_k = top_k
        param.n_keep = n_keep
        param.top_p = top_p
        param.temperature = temperature
        param.repeat_penalty = repeat_penalty
        param.frequency_penalty = frequency_penalty
        param.presence_penalty = presence_penalty
        param.mirostat = 0
        param.mirostat_tau = 5.0
        param.mirostat_eta = 0.1
        param.skip_special_token = True
        param.is_async = False
        param.img_start = b""
        param.img_end = b""
        param.img_content = b""

        # CPU affinity
        cpu_mask = CPU_MASKS.get(enabled_cpus, 0xC0)
        param.extend_param.base_domain_id = 1  # Domain 1 to coexist with RKNN models (domain 0)
        param.extend_param.embed_flash = embed_flash
        param.extend_param.enabled_cpus_num = enabled_cpus
        param.extend_param.enabled_cpus_mask = cpu_mask
        param.extend_param.n_batch = 1
        param.extend_param.use_cross_attn = 0

        # Store params for later reference
        self._params = {
            "max_context_len": max_context_len,
            "max_new_tokens": max_new_tokens,
            "top_k": top_k, "top_p": top_p,
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "enabled_cpus": enabled_cpus,
        }

        # Setup function signatures
        self.lib.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam), RKLLM_CALLBACK
        ]
        self.lib.rkllm_init.restype = ctypes.c_int

        self.lib.rkllm_run.argtypes = [
            RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p
        ]
        self.lib.rkllm_run.restype = ctypes.c_int

        self.lib.rkllm_abort.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_abort.restype = ctypes.c_int

        self.lib.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.lib.rkllm_destroy.restype = ctypes.c_int

        self.lib.rkllm_set_chat_template.argtypes = [
            RKLLM_Handle_t, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_char_p
        ]
        self.lib.rkllm_set_chat_template.restype = ctypes.c_int

        self.lib.rkllm_clear_kv_cache.argtypes = [
            RKLLM_Handle_t, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int

        # Initialize
        self.handle = RKLLM_Handle_t()
        ret = self.lib.rkllm_init(
            ctypes.byref(self.handle), ctypes.byref(param), self._cb
        )
        assert ret == 0, f"rkllm_init failed: {ret}"

        # CRITICAL: Set empty chat template to disable built-in prompt wrapping
        ret = self.lib.rkllm_set_chat_template(self.handle, b"", b"", b"")
        assert ret == 0, f"rkllm_set_chat_template failed: {ret}"

        # Setup prompt cache functions
        self.lib.rkllm_load_prompt_cache.argtypes = [
            RKLLM_Handle_t, ctypes.c_char_p
        ]
        self.lib.rkllm_load_prompt_cache.restype = ctypes.c_int
        self.lib.rkllm_release_prompt_cache.argtypes = [
            RKLLM_Handle_t
        ]
        self.lib.rkllm_release_prompt_cache.restype = ctypes.c_int

        self._n_keep = n_keep
        self._prefix_kv_ready = False

        print(f"[Decoder] Loaded. cpus={enabled_cpus} max_ctx={max_context_len} "
              f"max_new_tokens={max_new_tokens} top_k={top_k} n_keep={n_keep}")

    def precompute_prefix_kv(self, prefix_embed: np.ndarray) -> float:
        """
        Pre-compute KV cache for fixed prefix tokens using GET_LOGITS mode.
        
        Call once during init. Subsequent run_embed calls with
        keep_prefix=True will skip these tokens.
        
        Args:
            prefix_embed: (n_prefix, embed_dim) float32 embedding
            
        Returns:
            Computation time in ms
        """
        import time
        t0 = time.perf_counter()
        
        self.lib.rkllm_clear_kv_cache(self.handle, 0, None, None)
        
        embed = np.ascontiguousarray(prefix_embed, dtype=np.float32)
        ptr = embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        n_tokens = embed.shape[0]
        
        rkllm_input = RKLLMInput()
        rkllm_input.role = b""
        rkllm_input.enable_thinking = ctypes.c_bool(False)
        rkllm_input.input_type = RKLLM_INPUT_EMBED
        rkllm_input.input_data.embed_input = RKLLMEmbedInput(ptr, n_tokens)
        
        infer_param = RKLLMInferParam()
        infer_param.mode = RKLLM_INFER_GET_LOGITS
        infer_param.lora_params = None
        infer_param.prompt_cache_params = None
        infer_param.keep_history = 0
        
        ret = self.lib.rkllm_run(
            self.handle, ctypes.byref(rkllm_input),
            ctypes.byref(infer_param), None
        )
        
        ms = (time.perf_counter() - t0) * 1000
        if ret == 0:
            self._prefix_kv_ready = True
        else:
            print(f"[Decoder] WARNING: prefix KV pre-compute failed (ret={ret})")
        
        return ms

    def run_embed(self, embed_array: np.ndarray, n_tokens: int,
                  keep_history: int = 0, keep_prefix: bool = False) -> dict:
        """
        Run LLM generation with embedding input.
        
        Args:
            embed_array: (n_tokens, embed_dim) float32 embedding
            n_tokens: Number of tokens
            keep_history: 0 = clear KV cache, 1 = keep history
            
        Returns:
            dict with keys: text, perf, n_tokens_generated
        """
        with self._lock:
            self._output_chunks = []
            self._perf = {}
            self._repeat_buf = []
            self._aborted = False

            # Clear KV cache
            if keep_prefix and self._prefix_kv_ready:
                # Keep prefix KV (first n_keep positions), clear the rest
                self.lib.rkllm_clear_kv_cache(self.handle, 1, None, None)
                keep_history = 1  # Must use keep_history to append to cached KV
            elif not keep_history:
                self.lib.rkllm_clear_kv_cache(self.handle, 0, None, None)

            embed_array = np.ascontiguousarray(embed_array, dtype=np.float32)
            embed_ptr = embed_array.ctypes.data_as(
                ctypes.POINTER(ctypes.c_float)
            )

            # Build input - CRITICAL: role must be empty string
            rkllm_input = RKLLMInput()
            rkllm_input.role = b""
            rkllm_input.enable_thinking = ctypes.c_bool(False)
            rkllm_input.input_type = RKLLM_INPUT_EMBED
            rkllm_input.input_data.embed_input = RKLLMEmbedInput(
                embed_ptr, n_tokens
            )

            infer_param = RKLLMInferParam()
            infer_param.mode = RKLLM_INFER_GENERATE
            infer_param.lora_params = None
            infer_param.prompt_cache_params = None
            infer_param.keep_history = keep_history

            ret = self.lib.rkllm_run(
                self.handle, ctypes.byref(rkllm_input),
                ctypes.byref(infer_param), None
            )

            text = "".join(self._output_chunks)
            # Clean any leaked special tokens
            for tag in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                text = text.replace(tag, "")

            # If repetition was detected, truncate to the non-repeating part
            if self._aborted and text:
                # Find the repeating pattern and remove it
                # Try to keep text before the repetition started
                for n in (1, 2, 3, 4, 5):
                    # Check if the last part is a repeated pattern of n chars
                    if len(text) > n * 4:
                        pattern = text[-n:]
                        count = 0
                        pos = len(text)
                        while pos >= n and text[pos-n:pos] == pattern:
                            count += 1
                            pos -= n
                        if count >= 4:
                            text = text[:pos + n]  # Keep one instance
                            break

            return {
                "text": text,
                "perf": self._perf,
                "n_tokens_generated": len(self._output_chunks),
                "ret_code": ret,
                "aborted": self._aborted,
            }

    def abort(self):
        """Abort current generation."""
        self.lib.rkllm_abort(self.handle)

    def release(self):
        """Release RKLLM resources."""
        try:
            self.lib.rkllm_destroy(self.handle)
        except Exception:
            pass

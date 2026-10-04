"""Pin the RKLLM ctypes bindings to the librkllmrt v1.3.0 C ABI.

A wrong layout does not raise: the runtime reads or writes the wrong bytes and
the process dies with SIGSEGV inside ``rkllm_init`` / ``rkllm_run``.  These
tests run anywhere -- no NPU and no ``librkllmrt.so`` -- using
  * a table of C struct sizes / field offsets, and
  * a fake library that stands in for ``librkllmrt.so`` so the code paths that
    call ``rkllm_init`` / ``rkllm_run`` are exercised with the real structs.
"""

from __future__ import annotations

import ctypes
import inspect

import numpy as np
import pytest

from rkvoice_stream.runtime import rkllm_abi as abi

pytestmark = pytest.mark.skipif(
    ctypes.sizeof(ctypes.c_void_p) != 8,
    reason="layouts below are the LP64 ones (RK3576 / RK3588, aarch64)",
)

# ---------------------------------------------------------------------------
# Layout table
#
# DO NOT edit these numbers to make a test pass.  They are sizeof()/offsetof()
# as computed by g++ (LP64) from rkllm.h at tag release-v1.3.0 of
# airockchip/rknn-llm.  If a struct in rkllm_abi.py changes, re-derive them from
# the header.
#
# Notes: RKLLMInput.input_data is the *anonymous* C union (every member sits at
# the same offset); _RKLLMImageInput / _RKLLMVideoInput are the anonymous nested
# structs `image` / `video` of RKLLMMultiModalInput (offsets relative to them).
# ---------------------------------------------------------------------------

_LAYOUT = {
    "RKLLMExtendParam": (120, [
        ("base_domain_id", 0), ("embed_flash", 4), ("enabled_cpus_num", 5),
        ("enabled_cpus_mask", 8), ("n_batch", 12), ("use_cross_attn", 13),
        ("reserved", 14),
    ]),
    "RKLLMParam": (184, [
        ("model_path", 0), ("max_context_len", 8), ("max_new_tokens", 12),
        ("top_k", 16), ("n_keep", 20), ("top_p", 24), ("temperature", 28),
        ("repeat_penalty", 32), ("frequency_penalty", 36), ("presence_penalty", 40),
        ("mirostat", 44), ("mirostat_tau", 48), ("mirostat_eta", 52),
        ("skip_special_token", 56), ("ignore_eos_token", 57), ("is_async", 58),
        ("extend_param", 60),
    ]),
    "RKLLMEmbedInput": (16, [
        ("embed", 0), ("n_tokens", 8),
    ]),
    "RKLLMTokenInput": (16, [
        ("input_ids", 0), ("n_tokens", 8),
    ]),
    "_RKLLMImageInput": (64, [
        ("image_embed", 0), ("n_image_tokens", 8), ("n_image", 16),
        ("image_start", 24), ("image_end", 32), ("image_content", 40),
        ("image_width", 48), ("image_height", 56),
    ]),
    "_RKLLMVideoInput": (72, [
        ("video_embed", 0), ("n_frame_tokens", 8), ("n_frame_per_video", 16),
        ("n_video", 24), ("video_start", 32), ("video_end", 40), ("video_content", 48),
        ("frame_width", 56), ("frame_height", 64),
    ]),
    "RKLLMMultiModalInput": (144, [
        ("prompt", 0), ("image", 8), ("video", 72),
    ]),
    "RKLLMInput": (160, [
        ("role", 0), ("enable_thinking", 8), ("input_type", 12), ("input_data", 16),
    ]),
    "RKLLMLoraParam": (8, [
        ("lora_adapter_name", 0),
    ]),
    "RKLLMPromptCacheParam": (16, [
        ("save_prompt_cache", 0), ("prompt_cache_path", 8),
    ]),
    "RKLLMSamplingParam": (36, [
        ("top_k", 0), ("top_p", 4), ("temperature", 8), ("repeat_penalty", 12),
        ("frequency_penalty", 16), ("presence_penalty", 20), ("mirostat", 24),
        ("mirostat_tau", 28), ("mirostat_eta", 32),
    ]),
    "RKLLMInferParam": (40, [
        ("mode", 0), ("lora_params", 8), ("prompt_cache_params", 16),
        ("sampling_params", 24), ("keep_history", 32), ("max_new_tokens", 36),
    ]),
    "RKLLMResultLastHiddenLayer": (16, [
        ("hidden_states", 0), ("embd_size", 8), ("num_tokens", 12),
    ]),
    "RKLLMResultLogits": (16, [
        ("logits", 0), ("vocab_size", 8), ("num_tokens", 12),
    ]),
    "RKLLMPerfStat": (20, [
        ("prefill_time_ms", 0), ("prefill_tokens", 4), ("generate_time_ms", 8),
        ("generate_tokens", 12), ("memory_usage_mb", 16),
    ]),
    "RKLLMResult": (72, [
        ("text", 0), ("token_id", 8), ("last_hidden_layer", 16), ("logits", 32),
        ("perf", 48),
    ]),
    "RKLLMCallback": (48, [
        ("result_callback", 0), ("result_userdata", 8), ("tokenizer_callback", 16),
        ("tokenizer_userdata", 24), ("embed_callback", 32), ("embed_userdata", 40),
    ]),
}


@pytest.mark.parametrize("name", list(_LAYOUT))
def test_struct_layout_matches_rkllm_h(name):
    size, fields = _LAYOUT[name]
    cls = getattr(abi, name)
    assert [f[0] for f in cls._fields_] == [f for f, _ in fields], "field names/order"
    assert ctypes.sizeof(cls) == size
    for fname, offset in fields:
        assert getattr(cls, fname).offset == offset, fname


def test_every_mirrored_struct_is_pinned():
    mirrored = {
        n for n, c in inspect.getmembers(abi, inspect.isclass)
        if issubclass(c, ctypes.Structure) and c.__module__ == abi.__name__
    }
    assert mirrored == set(_LAYOUT), "add new structs to _LAYOUT (from rkllm.h)"


def test_input_union_is_as_large_as_the_multimodal_input():
    # This is what makes RKLLMInput 160 bytes, like the C side.
    assert ctypes.sizeof(abi.RKLLMInputUnion) == 144
    for member in ("prompt_input", "embed_input", "token_input", "multimodal_input"):
        assert getattr(abi.RKLLMInputUnion, member).offset == 0


def test_callback_signatures_return_int():
    # The runtime acts on the return value (0 continue / 1 pause / 2 release).
    assert abi.RKLLM_CALLBACK._restype_ is ctypes.c_int
    assert len(abi.RKLLM_CALLBACK._argtypes_) == 3
    assert abi.RKLLM_TOKENIZER_CALLBACK._restype_ is ctypes.c_int
    assert len(abi.RKLLM_TOKENIZER_CALLBACK._argtypes_) == 5
    assert abi.RKLLM_GET_EMBED_CALLBACK._restype_ is ctypes.c_int
    assert len(abi.RKLLM_GET_EMBED_CALLBACK._argtypes_) == 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_make_callback_struct_sets_only_the_result_callback():
    cb = abi.RKLLM_CALLBACK(lambda result, userdata, state: 0)
    s = abi.make_callback_struct(cb)
    assert s.result_callback(None, None, 0) == 0
    assert not s.tokenizer_callback and not s.embed_callback
    assert s.result_userdata is None
    assert s.tokenizer_userdata is None and s.embed_userdata is None


def test_make_infer_param_defers_to_init_time_settings():
    p = abi.make_infer_param(abi.RKLLM_INFER_GET_LOGITS, keep_history=1)
    assert p.mode == abi.RKLLM_INFER_GET_LOGITS
    assert p.keep_history == 1
    assert not p.lora_params and not p.prompt_cache_params
    assert not p.sampling_params   # NULL -> sampling settings of RKLLMParam
    assert p.max_new_tokens == 0   # <= 0 -> max_new_tokens of RKLLMParam


def test_make_infer_param_points_at_the_prompt_cache_param():
    cache = abi.RKLLMPromptCacheParam()
    cache.save_prompt_cache = 1
    cache.prompt_cache_path = b"/tmp/x.bin"
    p = abi.make_infer_param(prompt_cache_params=cache)
    assert p.prompt_cache_params.contents.save_prompt_cache == 1
    assert p.prompt_cache_params.contents.prompt_cache_path == b"/tmp/x.bin"


# ---------------------------------------------------------------------------
# A fake librkllmrt.so
# ---------------------------------------------------------------------------

class _Fn:
    """Stand-in for a ctypes foreign function (records calls, takes argtypes)."""

    def __init__(self, result=0):
        self.calls = []
        self._result = result

    def __call__(self, *args):
        self.calls.append(args)
        return self._result(*args) if callable(self._result) else self._result


class _FakeLib:
    """Any ``rkllm_*`` attribute exists and returns 0 unless set up via ``fn``."""

    def __init__(self):
        self._fns = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._fns.setdefault(name, _Fn())

    def fn(self, name, result):
        self._fns[name] = _Fn(result)


def _byref_target(arg):
    """The object a ``ctypes.byref(obj)`` argument points at."""
    return arg._obj


def _install_runtime(lib, emit):
    """Make ``rkllm_init`` / ``rkllm_run`` behave like the runtime does.

    ``rkllm_init`` records the RKLLMCallback the caller handed over.
    ``rkllm_run`` records its input/infer params, then calls
    ``emit(result_callback)`` the way the runtime would while generating.
    """
    lib.cb_struct = None
    lib.last_input = lib.last_infer = None

    def init(handle, param, cb):
        lib.cb_struct = _byref_target(cb)
        return 0

    def run(handle, inp, infer, userdata):
        lib.last_input = _byref_target(inp)
        lib.last_infer = _byref_target(infer)
        emit(lib.cb_struct.result_callback)
        return 0

    lib.fn("rkllm_init", init)
    lib.fn("rkllm_run", run)


def _emit_tokens(*texts):
    def emit(callback):
        res = abi.RKLLMResult()
        for t in texts:
            res.text = t.encode()
            assert callback(ctypes.pointer(res), None, abi.RKLLM_RUN_NORMAL) == 0
        res.perf.generate_tokens = len(texts)
        assert callback(ctypes.pointer(res), None, abi.RKLLM_RUN_FINISH) == 0
    return emit


# ---------------------------------------------------------------------------
# RKLLMDecoder (ASR)
# ---------------------------------------------------------------------------

def _make_decoder(monkeypatch, lib):
    from rkvoice_stream.backends.asr.qwen3 import decoder as dec
    monkeypatch.setattr(dec.ctypes, "CDLL", lambda *a, **k: lib)
    return dec.RKLLMDecoder(model_path="m.rkllm", lib_path="librkllmrt.so")


def test_decoder_passes_an_rkllmcallback_struct_to_rkllm_init(monkeypatch):
    lib = _FakeLib()
    _install_runtime(lib, _emit_tokens("he", "llo"))
    _make_decoder(monkeypatch, lib)

    assert lib.rkllm_init.argtypes[2] is ctypes.POINTER(abi.RKLLMCallback)
    cb = lib.cb_struct
    assert isinstance(cb, abi.RKLLMCallback)
    assert cb.result_callback and not cb.tokenizer_callback and not cb.embed_callback


def test_decoder_run_embed_uses_1_3_0_infer_param_and_returns_text(monkeypatch):
    lib = _FakeLib()
    _install_runtime(lib, _emit_tokens("he", "llo"))
    dec = _make_decoder(monkeypatch, lib)

    out = dec.run_embed(np.zeros((3, 4), dtype=np.float32), 3, keep_history=1)

    assert out["text"] == "hello"
    assert out["ret_code"] == 0
    assert out["perf"]["generate_tokens"] == 2
    infer = lib.last_infer
    assert isinstance(infer, abi.RKLLMInferParam)
    assert infer.mode == abi.RKLLM_INFER_GENERATE
    assert infer.keep_history == 1
    assert infer.max_new_tokens == 0 and not infer.sampling_params
    assert lib.last_input.input_type == abi.RKLLM_INPUT_EMBED
    assert lib.last_input.input_data.embed_input.n_tokens == 3


def test_decoder_precompute_prefix_kv_saves_the_prompt_cache(monkeypatch, tmp_path):
    lib = _FakeLib()
    _install_runtime(lib, _emit_tokens("x"))
    dec = _make_decoder(monkeypatch, lib)

    dec.precompute_prefix_kv(np.zeros((2, 4), dtype=np.float32),
                             cache_path=str(tmp_path / "prefix.bin"))

    infer = lib.last_infer
    assert infer.prompt_cache_params.contents.save_prompt_cache == 1
    assert infer.prompt_cache_params.contents.prompt_cache_path == str(
        tmp_path / "prefix.bin").encode()
    assert infer.max_new_tokens == 0 and not infer.sampling_params


# ---------------------------------------------------------------------------
# RKLLMTalker (Qwen3-TTS)
# ---------------------------------------------------------------------------

def _make_talker(monkeypatch, lib):
    from rkvoice_stream.runtime import rkllm_wrapper as wrapper
    lib.fn("rkllm_createDefaultParam", lambda: abi.RKLLMParam())
    monkeypatch.setattr(wrapper.ctypes, "CDLL", lambda *a, **k: lib)
    return wrapper.RKLLMTalker("m.rkllm", rkllm_lib="x", rknn_lib="y")


def _emit_hidden(n_tokens, embd_size):
    values = (ctypes.c_float * (n_tokens * embd_size))(*range(n_tokens * embd_size))

    def emit(callback):
        res = abi.RKLLMResult()
        res.last_hidden_layer.hidden_states = ctypes.cast(
            values, ctypes.POINTER(ctypes.c_float))
        res.last_hidden_layer.embd_size = embd_size
        res.last_hidden_layer.num_tokens = n_tokens
        assert callback(ctypes.pointer(res), None, abi.RKLLM_RUN_NORMAL) == 0
    return emit


def test_talker_passes_an_rkllmcallback_struct_to_rkllm_init(monkeypatch):
    lib = _FakeLib()
    _install_runtime(lib, _emit_hidden(1, 1))
    _make_talker(monkeypatch, lib)

    assert lib.rkllm_init.argtypes[2] is ctypes.POINTER(abi.RKLLMCallback)
    cb = lib.cb_struct
    assert isinstance(cb, abi.RKLLMCallback)
    assert cb.result_callback and not cb.tokenizer_callback and not cb.embed_callback


def test_talker_callback_always_returns_zero(monkeypatch):
    lib = _FakeLib()
    _install_runtime(lib, _emit_hidden(1, 1))
    talker = _make_talker(monkeypatch, lib)
    cb = lib.cb_struct.result_callback

    # None of the exit paths may return None / garbage: the runtime acts on it.
    assert cb(None, None, abi.RKLLM_RUN_WAITING) == 0
    assert cb(None, None, abi.RKLLM_RUN_FINISH) == 0
    assert cb(None, None, abi.RKLLM_RUN_ERROR) == 0
    assert talker._callback_error == "RKLLM_RUN_ERROR"


def test_talker_run_embed_and_run_tokens_use_1_3_0_infer_param(monkeypatch):
    lib = _FakeLib()
    _install_runtime(lib, _emit_hidden(2, 3))
    talker = _make_talker(monkeypatch, lib)

    out = talker.run_embed(np.zeros((2, 3), dtype=np.float32),
                           mode=abi.RKLLM_INFER_GET_LAST_HIDDEN_LAYER, keep_history=1)
    assert out["hidden"].shape == (2, 3)
    assert lib.last_input.input_type == abi.RKLLM_INPUT_EMBED
    infer = lib.last_infer
    assert infer.mode == abi.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
    assert infer.keep_history == 1
    assert infer.max_new_tokens == 0 and not infer.sampling_params

    talker.run_tokens(np.array([1, 2, 3], dtype=np.int32), keep_history=0)
    assert lib.last_input.input_type == abi.RKLLM_INPUT_TOKEN
    assert lib.last_input.input_data.token_input.n_tokens == 3
    infer = lib.last_infer
    assert infer.keep_history == 0
    assert infer.max_new_tokens == 0 and not infer.sampling_params

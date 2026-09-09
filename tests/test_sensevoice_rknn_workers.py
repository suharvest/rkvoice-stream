"""SenseVoice RKNN multi-core worker pool.

The backend used to hold one ``RKNNLite`` context on one NPU core, so two
utterances arriving at once ran back to back even on a 3-core RK3588. It now
builds one context per core and hands them out from a pool. These tests cover
the parts that do not need an NPU: how many workers are built, which cores
they bind to, that a context is never handed to two callers at once, that a
degraded start is survivable, and that unload releases each context once.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from rkvoice_stream.backends.asr import sensevoice_rknn as sv


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRKNNLite:
    """Stand-in for ``rknnlite.api.RKNNLite``.

    ``fail_init_on`` names core masks whose ``init_runtime`` returns non-zero,
    which is how the driver reports "this core is not available".
    """

    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4

    instances: list = []
    fail_init_on: set = set()
    fail_load = False
    inference_delay_s = 0.0

    def __init__(self, verbose=False):
        self.core_mask = None
        self.released = 0
        self.inferences = 0
        self.concurrent_entries = 0
        FakeRKNNLite.instances.append(self)

    def load_rknn(self, path):
        return 1 if FakeRKNNLite.fail_load else 0

    def init_runtime(self, core_mask=None):
        self.core_mask = core_mask
        return 1 if core_mask in FakeRKNNLite.fail_init_on else 0

    def inference(self, inputs=None):
        self.inferences += 1
        self.concurrent_entries += 1
        try:
            if FakeRKNNLite.inference_delay_s:
                time.sleep(FakeRKNNLite.inference_delay_s)
            return [np.zeros((1, sv.T_FIXED, 8), dtype=np.float32)]
        finally:
            self.concurrent_entries -= 1

    def release(self):
        self.released += 1


class FakeSP:
    def load(self, path):
        pass

    def get_piece_size(self):
        return 8

    def id_to_piece(self, i):
        return "x"


@pytest.fixture
def rknn_env(monkeypatch, tmp_path):
    """Install the fake runtime + a model dir the backend can 'load'."""
    FakeRKNNLite.instances = []
    FakeRKNNLite.fail_init_on = set()
    FakeRKNNLite.fail_load = False
    FakeRKNNLite.inference_delay_s = 0.0

    api = types.ModuleType("rknnlite.api")
    api.RKNNLite = FakeRKNNLite
    pkg = types.ModuleType("rknnlite")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "rknnlite", pkg)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)

    spm = types.ModuleType("sentencepiece")
    spm.SentencePieceProcessor = FakeSP
    monkeypatch.setitem(sys.modules, "sentencepiece", spm)

    (tmp_path / "sense-voice-encoder.rk3588.fp16.rknn").write_bytes(b"")
    (tmp_path / "sense-voice-encoder.rk3576.fp16.rknn").write_bytes(b"")
    (tmp_path / "chn_jpn_yue_eng_ko_spectok.bpe.model").write_bytes(b"")
    np.save(tmp_path / "embedding.npy", np.zeros((16, sv.LFR_DIM), dtype=np.float32))
    monkeypatch.setenv("SENSEVOICE_RKNN_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        sv.SenseVoiceRKNNBackend,
        "_load_cmvn",
        staticmethod(
            lambda path: (
                np.zeros(sv.LFR_DIM, dtype=np.float32),
                np.ones(sv.LFR_DIM, dtype=np.float32),
            )
        ),
    )
    # kaldi_native_fbank is an aarch64/device dependency; the pool semantics
    # under test start after the front end, so stub it out.
    monkeypatch.setattr(
        sv.SenseVoiceRKNNBackend,
        "_build_speech",
        lambda self, audio, lang="auto", textnorm="withitn": (
            np.zeros((1, sv.T_FIXED, sv.LFR_DIM), dtype=np.float32),
            8,
        ),
    )
    monkeypatch.delenv("SENSEVOICE_RKNN_WORKERS", raising=False)
    monkeypatch.delenv("SENSEVOICE_RKNN_CORE", raising=False)
    return tmp_path


def _preloaded(platform: str, monkeypatch) -> sv.SenseVoiceRKNNBackend:
    monkeypatch.setenv("RK_PLATFORM", platform)
    be = sv.SenseVoiceRKNNBackend()
    be.preload()
    return be


# ---------------------------------------------------------------------------
# Worker count resolution
# ---------------------------------------------------------------------------


def test_worker_count_defaults_per_soc(monkeypatch):
    monkeypatch.delenv("SENSEVOICE_RKNN_WORKERS", raising=False)
    assert sv._resolve_worker_count("rk3588") == 3
    assert sv._resolve_worker_count("rk3576") == 2
    assert sv._resolve_worker_count("rv1126b") == 1


@pytest.mark.parametrize(
    "raw,expected",
    [("1", 1), ("2", 2), ("3", 3), ("9", 3), ("0", 1), ("-4", 1), ("two", 3), ("", 3)],
)
def test_worker_count_env_override(monkeypatch, raw, expected):
    monkeypatch.setenv("SENSEVOICE_RKNN_WORKERS", raw)
    assert sv._resolve_worker_count("rk3588") == expected


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------


def test_rk3588_builds_three_contexts_on_distinct_cores(rknn_env, monkeypatch):
    be = _preloaded("rk3588", monkeypatch)
    assert be.npu_worker_count == 3
    assert be.npu_worker_cores == ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"]
    masks = [c.core_mask for c in be._contexts]
    # Single-core enum values only — a bitwise combination is rejected by the
    # driver, so the pool must never build one.
    assert masks == [FakeRKNNLite.NPU_CORE_0, FakeRKNNLite.NPU_CORE_1,
                     FakeRKNNLite.NPU_CORE_2]
    assert len(set(masks)) == 3
    assert be.supports_parallel is True
    assert be.max_concurrent == 3


def test_rk3576_builds_two_contexts(rknn_env, monkeypatch):
    be = _preloaded("rk3576", monkeypatch)
    assert be.npu_worker_count == 2
    assert be.npu_worker_cores == ["NPU_CORE_0", "NPU_CORE_1"]
    assert be.max_concurrent == 2


def test_single_worker_honours_the_legacy_core_env(rknn_env, monkeypatch):
    monkeypatch.setenv("SENSEVOICE_RKNN_WORKERS", "1")
    monkeypatch.setenv("SENSEVOICE_RKNN_CORE", "NPU_CORE_2")
    be = _preloaded("rk3588", monkeypatch)
    assert be.npu_worker_count == 1
    assert be._contexts[0].core_mask == FakeRKNNLite.NPU_CORE_2
    assert be.supports_parallel is False
    assert be.max_concurrent == 1


def test_capability_reports_workers_actually_built_not_configured(
    rknn_env, monkeypatch
):
    # Core2 refuses; the backend must serve at 2 and say 2, not the 3 asked for.
    FakeRKNNLite.fail_init_on = {FakeRKNNLite.NPU_CORE_2}
    be = _preloaded("rk3588", monkeypatch)
    assert be.npu_worker_count == 2
    assert be.max_concurrent == 2
    assert be.is_ready()
    # The context that failed init was released rather than leaked.
    failed = [c for c in FakeRKNNLite.instances if c not in be._contexts]
    assert [c.released for c in failed] == [1]


def test_first_core_failing_is_fatal(rknn_env, monkeypatch):
    FakeRKNNLite.fail_init_on = {FakeRKNNLite.NPU_CORE_0}
    monkeypatch.setenv("RK_PLATFORM", "rk3588")
    be = sv.SenseVoiceRKNNBackend()
    with pytest.raises(RuntimeError, match="init_runtime"):
        be.preload()
    assert be.npu_worker_count == 0


def test_load_failure_is_fatal_when_nothing_built(rknn_env, monkeypatch):
    FakeRKNNLite.fail_load = True
    monkeypatch.setenv("RK_PLATFORM", "rk3588")
    be = sv.SenseVoiceRKNNBackend()
    with pytest.raises(RuntimeError, match="load_rknn"):
        be.preload()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def test_transcribe_before_preload_raises(rknn_env):
    be = sv.SenseVoiceRKNNBackend()
    with pytest.raises(RuntimeError, match="not ready"):
        be.transcribe_array(np.zeros(16000, dtype=np.float32))


def test_context_is_returned_to_the_pool(rknn_env, monkeypatch):
    be = _preloaded("rk3576", monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)
    for _ in range(6):
        be.transcribe_array(audio)
    assert be._pool.qsize() == 2
    assert sum(c.inferences for c in be._contexts) == 6


def test_pool_never_hands_one_context_to_two_callers(rknn_env, monkeypatch):
    FakeRKNNLite.inference_delay_s = 0.02
    be = _preloaded("rk3588", monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)
    seen_overlap = []

    def run():
        for _ in range(6):
            be.transcribe_array(audio)

    for c in be._contexts:
        original = c.inference

        def wrapped(inputs=None, _c=c, _o=original):
            # >1 means this same context is executing twice at once.
            seen_overlap.append(_c.concurrent_entries)
            return _o(inputs=inputs)

        c.inference = wrapped

    threads = [threading.Thread(target=run) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max(seen_overlap) == 0
    assert sum(c.inferences for c in be._contexts) == 36
    assert be._pool.qsize() == 3


def test_all_workers_are_used_under_concurrency(rknn_env, monkeypatch):
    FakeRKNNLite.inference_delay_s = 0.05
    be = _preloaded("rk3588", monkeypatch)
    audio = np.zeros(16000, dtype=np.float32)

    def run():
        be.transcribe_array(audio)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(c.inferences == 1 for c in be._contexts)


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------


def test_unload_releases_every_context_once(rknn_env, monkeypatch):
    be = _preloaded("rk3588", monkeypatch)
    contexts = list(be._contexts)
    be.unload()
    assert [c.released for c in contexts] == [1, 1, 1]
    assert be.npu_worker_count == 0
    assert be.is_ready() is False


def test_repeated_unload_does_not_double_release(rknn_env, monkeypatch):
    be = _preloaded("rk3588", monkeypatch)
    contexts = list(be._contexts)
    be.unload()
    be.unload()
    assert [c.released for c in contexts] == [1, 1, 1]


def test_unload_waits_for_an_in_flight_inference(rknn_env, monkeypatch):
    be = _preloaded("rk3576", monkeypatch)
    entered = threading.Event()
    allow_finish = threading.Event()
    victim = be._contexts[0]
    original = victim.inference

    def slow(inputs=None):
        entered.set()
        assert allow_finish.wait(5.0)
        return original(inputs=inputs)

    victim.inference = slow
    released_at = {}

    def release():
        released_at["t"] = time.perf_counter()
        FakeRKNNLite.release(victim)

    victim.release = release

    t = threading.Thread(
        target=be.transcribe_array, args=(np.zeros(16000, dtype=np.float32),)
    )
    t.start()
    assert entered.wait(5.0)

    unloader = threading.Thread(target=be.unload)
    unloader.start()
    time.sleep(0.1)
    # unload must still be waiting on the borrowed context.
    assert "t" not in released_at
    allow_finish.set()
    t.join(5.0)
    unloader.join(5.0)
    assert victim.released == 1

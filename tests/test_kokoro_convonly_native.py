import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
NATIVE = ROOT / "native/kokoro_convonly"


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    out = tmp_path_factory.mktemp("native") / "libkokoro_convonly_native.so"
    subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-Wpedantic",
                    "-ffp-contract=off", "-fPIC", "-shared", str(NATIVE / "kokoro_convonly_native.cpp"),
                    str(NATIVE / "tests/stub_bridge.cpp"), "-I", str(NATIVE),
                    "-lpthread", "-o", str(out)], check=True)
    return out


def make_tail(library):
    from rkvoice_stream.backends.tts.kokoro_convonly_native import NativeConvOnlyTail
    return NativeConvOnlyTail("/models", "/models/merge.rknn",
                              np.ones((18, 128), dtype=np.float32) * 0.1,
                              library)


def counters(library):
    lib = ctypes.CDLL(str(library))
    for name in ("stub_reset",):
        getattr(lib, name).restype = None
    for name in ("stub_create_count", "stub_destroy_count", "stub_run_count"):
        getattr(lib, name).restype = ctypes.c_int
    lib.stub_reset()
    return lib


def test_native_variable_lengths_and_idempotent_close(library):
    lib = counters(library)
    tail = make_tail(library)
    assert lib.stub_create_count() == 19
    for n in (1, 17, 193):
        y, timing = tail.run(np.ones((1, 128, n), np.float32),
                             np.zeros((1, 18, 128), np.float32),
                             np.zeros((1, 18, 128), np.float32))
        assert y.shape == (1, 22, n)
        assert np.isfinite(y).all()
        assert timing["masks"] == [1, 2, 1]
    tail.close()
    tail.close()
    assert lib.stub_destroy_count() == 19
    with pytest.raises(RuntimeError, match="closed"):
        tail.run(np.zeros((1, 128, 1), np.float32),
                 np.zeros((1, 18, 128), np.float32),
                 np.zeros((1, 18, 128), np.float32))


def test_strict_shapes_and_finite(library):
    tail = make_tail(library)
    z = np.zeros((1, 18, 128), np.float32)
    with pytest.raises(ValueError):
        tail.run(np.zeros((128, 2), np.float32), z, z)
    bad = np.zeros((1, 128, 1), np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        tail.run(bad, z, z)
    tail.close()


def test_c_boundary_exact_counts_and_error_buffer(library):
    lib = counters(library)
    lib.stub_set_fail_at_create.argtypes = [ctypes.c_int]
    lib.stub_set_fail_at_create(4)
    from rkvoice_stream.backends.tts.kokoro_convonly_native import NativeConvOnlyTail
    with pytest.raises(RuntimeError, match="create failed"):
        NativeConvOnlyTail("/models", "/models/merge.rknn",
                           np.zeros((18, 128), np.float32), library)
    assert lib.stub_destroy_count() == 3


def test_run_failure_is_reported_and_handle_can_close(library):
    lib = counters(library)
    lib.stub_set_fail_at_run.argtypes = [ctypes.c_int]
    lib.stub_set_fail_at_run(1)
    tail = make_tail(library)
    z = np.zeros((1, 18, 128), np.float32)
    with pytest.raises(RuntimeError, match="run failure"):
        tail.run(np.zeros((1, 128, 8193), np.float32), z, z)
    tail.close()
    assert lib.stub_destroy_count() == 19


@pytest.mark.parametrize("n", [17, 8193, 12001, 38401])
def test_real_host_math_and_nhwc_merge_match_numpy(library, n):
    """Only Conv/merge RKNN calls are stubs; run the real persistent CPU path."""
    from rkvoice_stream.backends.tts.kokoro_convonly_tail import run_branch, schedule
    lib = counters(library)
    rng = np.random.default_rng(72)
    x = rng.normal(size=(1, 128, n)).astype(np.float32)
    gamma = (rng.normal(size=(1, 18, 128)) * .02).astype(np.float32)
    beta = (rng.normal(size=(1, 18, 128)) * .03).astype(np.float32)
    slopes = [np.full((1, 128, 1), .1, np.float32) for _ in range(18)]
    g = [gamma[:, i, :].reshape(1, 128, 1) for i in range(18)]
    b = [beta[:, i, :].reshape(1, 128, 1) for i in range(18)]

    class IdentityConv:
        def run(self, tile):
            return tile.copy()

    models = {(bid, unit, conv): IdentityConv() for bid in range(3)
              for unit in range(3) for conv in (1, 2)}
    expected_branches = [run_branch(bid, x, g, b, models, n, slopes)[0]
                         for bid in range(3)]
    expected = sum(expected_branches)[:, :22] / np.float32(3)
    with make_tail(library) as tail:
        got, _ = tail.run(x, gamma, beta)
        np.testing.assert_allclose(got, expected, rtol=3e-6, atol=3e-6)
        # Changing inputs on the same contexts must not reuse the previous fixture.
        got2, _ = tail.run(np.ascontiguousarray(x * 2), gamma, beta)
        assert not np.array_equal(got2, got)
    conv_calls = sum(len(schedule(n, (kernel - 1) * (dilation if conv == 1 else 1) // 2))
                     for kernel in (3, 7, 11) for dilation in (1, 3, 5) for conv in (1, 2))
    assert lib.stub_run_count() == 2 * (conv_calls + len(schedule(n, 3)))
    assert lib.stub_create_count() == lib.stub_destroy_count() == 19


def test_c_boundary_count_failure_does_not_write_output(library):
    counters(library)
    with make_tail(library) as tail:
        x = np.zeros((1, 128, 17), np.float32)
        style = np.zeros((1, 18, 128), np.float32)
        out = np.full((1, 22, 17), 42, np.float32)
        ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        rc = tail._lib.kokoro_convonly_run(
            tail._handle, 17, ptr(x), x.size - 1,
            ptr(style), style.size, ptr(style), style.size,
            ptr(out), out.size, tail._err, len(tail._err))
        assert rc != 0 and b"count" in tail._err.value
        assert np.all(out == 42)

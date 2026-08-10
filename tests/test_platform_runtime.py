"""Platform profile + init_runtime core-mask routing (off-device, pure-Python).

Covers the rv1126b single-core NPU support:
  - rv1126b is a registered platform with npu_cores == 1, not a coprocessor;
  - platform_is_single_core() classifies host SoCs correctly (rv1126b single,
    rk3576/rk3588 multi, rk1828 coprocessor → not single, unknown → not single);
  - init_runtime_for_platform() calls a plain init_runtime() on single-core
    parts and init_runtime(core_mask=...) on multi-core parts, and honours
    force_single_core.

Runs identically in HTTP and direct mode (no service, no rknn runtime): a stub
``rknnlite.api`` module is injected so the helper's lazy import resolves.
"""

from __future__ import annotations

import sys
import types

import pytest

from rkvoice_stream.platform import (
    get_platform,
    init_runtime_for_platform,
    platform_is_single_core,
)


# --------------------------------------------------------------------------- #
# Platform registration
# --------------------------------------------------------------------------- #

def test_rv1126b_registered_single_core():
    prof = get_platform("rv1126b")
    assert prof.name == "rv1126b"
    assert prof.npu_cores == 1
    assert prof.is_coprocessor is False


def test_get_platform_case_insensitive():
    assert get_platform("RV1126B").name == "rv1126b"


# --------------------------------------------------------------------------- #
# Single-core classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "platform,expected",
    [
        ("rv1126b", True),    # single NPU core, host SoC
        ("rk3576", False),    # 2 cores
        ("rk3588", False),    # 3 cores
        ("rk1828", False),    # single core but PCIe coprocessor → device_id
        ("nope", False),      # unknown → historical multi-core default
        ("", False),          # empty → default
    ],
)
def test_platform_is_single_core(platform, expected):
    assert platform_is_single_core(platform) is expected


def test_platform_is_single_core_reads_env(monkeypatch):
    monkeypatch.setenv("RK_PLATFORM", "rv1126b")
    assert platform_is_single_core() is True
    monkeypatch.setenv("RK_PLATFORM", "rk3588")
    assert platform_is_single_core() is False


# --------------------------------------------------------------------------- #
# init_runtime routing (stubbed RKNNLite)
# --------------------------------------------------------------------------- #

class _StubRKNN:
    """Records how init_runtime was called."""

    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_0_1 = 3
    NPU_CORE_AUTO = 0

    def __init__(self):
        self.calls = []

    def init_runtime(self, core_mask="__unset__"):
        self.calls.append(core_mask)
        return 0


@pytest.fixture
def stub_rknnlite(monkeypatch):
    """Inject a fake ``rknnlite.api`` exposing the NPU_CORE_* constants."""
    api = types.ModuleType("rknnlite.api")
    api.RKNNLite = _StubRKNN
    pkg = types.ModuleType("rknnlite")
    pkg.api = api
    monkeypatch.setitem(sys.modules, "rknnlite", pkg)
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    yield


def test_single_core_uses_maskless_init(stub_rknnlite):
    rk = _StubRKNN()
    ret = init_runtime_for_platform(rk, platform="rv1126b", core_mask="NPU_CORE_0")
    assert ret == 0
    # No core_mask passed → default sentinel recorded.
    assert rk.calls == ["__unset__"]


def test_multi_core_passes_core_mask(stub_rknnlite):
    rk = _StubRKNN()
    ret = init_runtime_for_platform(rk, platform="rk3588", core_mask="NPU_CORE_1")
    assert ret == 0
    # NPU_CORE_1 resolved to its int constant on the stub.
    assert rk.calls == [_StubRKNN.NPU_CORE_1]


def test_force_single_core_overrides_multi(stub_rknnlite):
    rk = _StubRKNN()
    ret = init_runtime_for_platform(
        rk, platform="rk3588", core_mask="NPU_CORE_1", force_single_core=True
    )
    assert ret == 0
    assert rk.calls == ["__unset__"]


def test_unknown_core_name_falls_back_to_auto(stub_rknnlite):
    rk = _StubRKNN()
    ret = init_runtime_for_platform(rk, platform="rk3576", core_mask="NPU_CORE_BOGUS")
    assert ret == 0
    assert rk.calls == [_StubRKNN.NPU_CORE_AUTO]

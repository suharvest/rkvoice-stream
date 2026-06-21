"""Capability device-id dimension tests (Phase 2c, off-device, pure-Python).

Resources are bucketed by ``ResourceProfile.device_id``:
  - host SoC backends (device_id=None) and an RK1828 PCIe EP (its BDF) are
    *separate* devices → never contend (own memory + core/domain namespace);
  - two models on the SAME RK1828 EP share its ~5GB budget → gemma4 + Qwen3-TTS
    are memory-exclusive there;
  - RK3576/88 single-SoC behaviour (all device_id=None) is unchanged.

Dual-mode: pure logic, runs identically in HTTP and direct mode (no service).
"""

from __future__ import annotations

from rkvoice_stream.app.capability import (
    ResourceProfile,
    check_compatibility,
    check_resources,
)

RK1828_BDF = "0001:11:00.0"

# Rough on-EP footprints (MB): gemma4 (audio+LLM) is large, Qwen3-TTS ~1.7GB.
GEMMA4_MB = 4200
QWEN3_TTS_MB = 1700


def _host_asr(mem=60, domain=0, cores=("CORE_AUTO",)):
    return ResourceProfile(
        npu_domain=domain, npu_cores=list(cores),
        npu_memory_mb=mem, label="host_asr", device_id=None,
    )


def _host_tts(mem=50, domain=0, cores=("CORE_AUTO",)):
    return ResourceProfile(
        npu_domain=domain, npu_cores=list(cores),
        npu_memory_mb=mem, label="host_tts", device_id=None,
    )


def _rk1828_gemma4():
    return ResourceProfile(
        npu_domain=1, npu_cores=["CORE_AUTO"],
        uses_rkllm=True, npu_memory_mb=GEMMA4_MB,
        label="gemma4_rk1828", device_id=RK1828_BDF,
    )


def _rk1828_tts():
    return ResourceProfile(
        npu_domain=0, npu_cores=["CORE_AUTO"],
        npu_memory_mb=QWEN3_TTS_MB,
        label="qwen3_tts_rk1828", device_id=RK1828_BDF,
    )


# ── backward compat: device_id=None preserves single-SoC behaviour ───────

def test_host_only_unchanged_parallel():
    """Two small host backends on RK3576 → parallel, no errors (as before)."""
    r = check_compatibility(_host_asr(), _host_tts(), platform_name="rk3576")
    assert r["compatible"] is True
    assert r["mode"] == "parallel"
    assert r["errors"] == []


def test_host_only_memory_overflow_exclusive():
    """Host memory overflow still trips the existing exclusive/error path."""
    big = _host_tts(mem=300)  # > rk3576 limit (180)
    r = check_compatibility(_host_asr(mem=100), big, platform_name="rk3576")
    assert r["compatible"] is False
    assert r["mode"] == "exclusive"
    assert any("memory exceeds" in e for e in r["errors"])


def test_single_profile_unchanged():
    r = check_compatibility(_host_asr(), None)
    assert r["mode"] == "single"
    assert r["compatible"] is True


# ── cross-device: RK1828 EP does NOT contend with host ───────────────────

def test_rk1828_and_host_do_not_conflict():
    """gemma4 on the RK1828 EP + a host TTS → separate devices → parallel."""
    r = check_compatibility(
        _rk1828_gemma4(), _host_tts(), platform_name="rk3588"
    )
    assert r["compatible"] is True
    assert r["mode"] == "parallel"
    assert r["errors"] == []


def test_rk1828_huge_model_does_not_charge_host():
    """Even a 4.2GB EP model + host model = parallel: budgets are separate."""
    r = check_compatibility(
        _rk1828_gemma4(), _host_tts(mem=400), platform_name="rk3588"
    )
    assert r["compatible"] is True
    assert r["mode"] == "parallel"


# ── same RK1828 EP: gemma4 + Qwen3-TTS are memory-exclusive ──────────────

def test_gemma4_and_tts_on_same_ep_exclusive():
    """gemma4 (4.2GB) + Qwen3-TTS (1.7GB) = 5.9GB > 5GB EP budget → exclusive."""
    r = check_compatibility(
        _rk1828_gemma4(), _rk1828_tts(), platform_name="rk3588"
    )
    assert r["compatible"] is False
    assert r["mode"] == "exclusive"
    assert any("memory exceeds" in e and "5120" in e for e in r["errors"])


def test_two_small_models_on_same_ep_fit():
    """Two small models on the same EP that fit the 5GB budget → sequential/ok."""
    a = ResourceProfile(npu_memory_mb=1000, label="a", device_id=RK1828_BDF, uses_rkllm=True)
    b = ResourceProfile(npu_memory_mb=1500, label="b", device_id=RK1828_BDF, uses_rkllm=True)
    r = check_compatibility(a, b, platform_name="rk3588")
    assert r["compatible"] is True
    # both rkllm in same domain → sequential
    assert r["mode"] == "sequential"


# ── multi-backend device-aware ledger (check_resources) ──────────────────

def test_check_resources_buckets_by_device():
    """3 backends: host ASR + host TTS + RK1828 gemma4 → two device buckets."""
    res = check_resources(
        [_host_asr(), _host_tts(), _rk1828_gemma4()], platform_name="rk3588"
    )
    assert set(res["devices"].keys()) == {"host", RK1828_BDF}
    host = res["devices"]["host"]
    ep = res["devices"][RK1828_BDF]
    # host bucket: two small models, fit, parallel
    assert host["total_memory_mb"] == 110
    assert host["memory_limit_mb"] == 512  # rk3588 host NPU limit
    # EP bucket: gemma4 alone, single
    assert ep["total_memory_mb"] == GEMMA4_MB
    assert ep["memory_limit_mb"] == 5120  # RK1828 EP limit (own budget)
    assert ep["mode"] == "single"
    assert res["compatible"] is True


def test_check_resources_same_ep_overflow():
    """gemma4 + Qwen3-TTS on the SAME EP overflow that EP's 5GB budget."""
    res = check_resources(
        [_rk1828_gemma4(), _rk1828_tts()], platform_name="rk3588"
    )
    assert res["compatible"] is False
    ep = res["devices"][RK1828_BDF]
    assert ep["mode"] == "exclusive"
    assert ep["total_memory_mb"] == GEMMA4_MB + QWEN3_TTS_MB
    assert any("memory exceeds" in e for e in res["errors"])


def test_check_resources_host_and_ep_independent():
    """A full host overflow does not mark the EP bucket incompatible, and v.v."""
    res = check_resources(
        [_host_asr(mem=3000), _host_tts(mem=3000), _rk1828_gemma4()],
        platform_name="rk3576",  # host limit smaller
    )
    # host overflows (6000 > rk3576 limit), EP fine
    assert res["devices"]["host"]["mode"] == "exclusive"
    assert res["devices"][RK1828_BDF]["mode"] == "single"
    assert res["compatible"] is False

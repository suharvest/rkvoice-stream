"""Unit checks for RK3576 performance-mode verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "models" / "tts" / "moss" / "verify_rk3576_performance_mode.py"
    spec = importlib.util.spec_from_file_location("verify_rk3576_performance_mode", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_freq_block(root: Path, base: str, governor: str, cur: int, freqs: str, set_name: str) -> None:
    block = root / base
    _write(block / "governor", governor)
    _write(block / "cur_freq", str(cur))
    _write(block / "available_frequencies", freqs)
    _write(block / set_name, "")


def _write_cpu_block(root: Path, policy: str, governor: str, cur: int, freqs: str) -> None:
    block = root / "devices/system/cpu/cpufreq" / policy
    _write(block / "scaling_governor", governor)
    _write(block / "scaling_cur_freq", str(cur))
    _write(block / "scaling_available_frequencies", freqs)
    _write(block / "scaling_setspeed", "")


def _write_cpuidle(root: Path, value: str) -> None:
    for cpu in range(8):
        _write(root / "devices/system/cpu" / f"cpu{cpu}" / "cpuidle/state1/disable", value)


def test_performance_mode_accepts_fixed_max_frequency(tmp_path):
    module = _load_module()
    _write_cpu_block(tmp_path, "policy0", "userspace", 2016000, "408000 2016000")
    _write_cpu_block(tmp_path, "policy4", "performance", 2208000, "408000 2208000")
    _write_freq_block(tmp_path, "class/devfreq/27700000.npu", "userspace", 950000000, "300000000 950000000", "userspace/set_freq")
    _write_freq_block(tmp_path, "class/devfreq/dmc", "userspace", 1848000000, "528000000 1848000000", "userspace/set_freq")
    _write_cpuidle(tmp_path, "1")

    report = module.verify_performance_mode(sysfs_root=tmp_path)

    assert report["passed"] is True
    assert report["errors"] == []


def test_performance_mode_rejects_ondemand_and_low_ddr(tmp_path):
    module = _load_module()
    _write_cpu_block(tmp_path, "policy0", "ondemand", 1416000, "408000 2016000")
    _write_cpu_block(tmp_path, "policy4", "ondemand", 2208000, "408000 2208000")
    _write_freq_block(tmp_path, "class/devfreq/27700000.npu", "rknpu_ondemand", 950000000, "300000000 950000000", "userspace/set_freq")
    _write_freq_block(tmp_path, "class/devfreq/dmc", "dmc_ondemand", 528000000, "528000000 1848000000", "userspace/set_freq")
    _write_cpuidle(tmp_path, "0")

    report = module.verify_performance_mode(sysfs_root=tmp_path)

    assert report["passed"] is False
    assert any("cpu.policy0.governor='ondemand'" in error for error in report["errors"])
    assert any("ddr.cur_freq=528000000 below max_available=1848000000" in error for error in report["errors"])
    assert any("cpu0.cpuidle.state1.disable='0'" in error for error in report["errors"])
    assert any("echo userspace" in command for command in report["remediation"]["commands"])

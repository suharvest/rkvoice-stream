"""Evidence lock for RK3576 monolithic MOSS RKNN runtime failures."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence" / "rk3576-moss-monolithic-prefill-decode-rknn-probe.json"


def test_rk3576_monolithic_prefill_and_decode_rknn_crash_at_inference():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["summary"] == {"total": 2, "ok": 0, "crash": 2, "timeout": 0, "missing": 0}
    by_case = {item["case"]: item for item in report["results"]}
    assert set(by_case) == {"prefill", "decode"}
    for item in by_case.values():
        assert item["load_ret"] == 0
        assert item["init_ret"] == 0
        assert item["status"] == "CRASH"
        assert item["signal"] == "SIGSEGV"
        assert item["returncode"] == -11
    assert by_case["prefill"]["input_shapes"] == [[1, 32, 17], [1, 32]]
    assert by_case["decode"]["input_shapes"][0:2] == [[1, 1, 17], [1]]

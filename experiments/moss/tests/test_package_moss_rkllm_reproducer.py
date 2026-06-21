"""Tests for the upstream MOSS RKLLM reproducer package."""

from __future__ import annotations

import json
from pathlib import Path

from package_moss_rkllm_reproducer import package_reproducer


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "evidence"


def test_package_reproducer_contains_actionable_upstream_evidence(tmp_path):
    report = package_reproducer(EVIDENCE_DIR, tmp_path / "rkllm-reproducer")

    out_dir = Path(report["package"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    issue = (out_dir / "UPSTREAM_ISSUE_DRAFT.md").read_text(encoding="utf-8")

    assert manifest["summary"]["runtime_probe"]["loaded"] is True
    assert manifest["summary"]["runtime_probe"]["prefill_ok"] is True
    assert manifest["summary"]["hidden_parity"]["passed"] is False
    assert manifest["summary"]["hidden_parity"]["prefill_metrics"]["rel_l2"] >= 1.0
    assert manifest["summary"]["token_input"]["passed"] is False
    assert manifest["summary"]["hf_variants"]["variants"]["hf_original"]["vs_onnx"]["rel_l2"] < 1.0e-4

    copied = {Path(item["target"]).name: item for item in manifest["files"]}
    for required in (
        "rk3576-moss-rkllm-folded-runtime-probe.json",
        "rk3576-moss-rkllm-hidden-vs-onnx-s8.json",
        "rk3576-moss-rkllm-hidden-vs-onnx-s8-embedflash0.json",
        "rk3576-moss-rkllm-token-hidden-s8.json",
        "wsl2-moss-hf-variants-vs-rkllm-s8.json",
    ):
        assert copied[required]["exists"] is True

    assert manifest["npz_files"][0]["exists"] is True
    assert "GPT2-style LayerNorm-with-bias" in issue
    assert "RKLLM_EMBED_FLASH=0" in issue

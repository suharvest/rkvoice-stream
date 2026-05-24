"""Evidence lock for RK3576 RKNN artifact workspace gate."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "docs" / "evidence" / "moss"


def test_rk3576_rknn_workspace_dry_run_is_persistent_and_non_executing():
    report = json.loads((EVIDENCE_DIR / "rk3576-rknn-workspace-dry-run.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["execute"] is False
    assert report["requires_confirmation"] is True
    assert report["confirmation"] == "FORMAT_RKNN_WORKSPACE"
    assert report["device"] == "/dev/mmcblk1"
    assert report["partition"] == "/dev/mmcblk1p1"
    assert report["mount_point"] == "/mnt/rknn-workspace"
    assert report["workspace"] == "/mnt/rknn-workspace/moss-rknn-workspace"
    assert report["persist_fstab"] is True
    assert report["fstab_entry"] == "LABEL=RKNN_WS /mnt/rknn-workspace ext4 defaults,nofail,noatime 0 2"
    assert ["write_fstab_entry", "/etc/fstab", report["fstab_entry"]] in report["commands"]
    assert all(result["executed"] is False for result in report["results"])


def test_rk3576_rknn_workspace_deployment_is_not_ready_yet():
    report = json.loads((EVIDENCE_DIR / "rk3576-rknn-workspace-deployment-not-ready.json").read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["fstab_entry_present"] is False
    assert report["mounted"] is None
    assert report["workspace"] == "/mnt/rknn-workspace/moss-rknn-workspace"
    assert report["workspace_report"]["same_filesystem_as_root"] is True
    assert any("missing fstab entry" in error for error in report["errors"])
    assert any("mount point is not mounted" in error for error in report["errors"])
    assert any("workspace does not exist" in error for error in report["errors"])


def test_rk3576_existing_home_rknn_probe_workspace_is_rejected():
    report = json.loads((EVIDENCE_DIR / "rk3576-rknn-workspace-home-rejected.json").read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["workspace"] == "/home/cat/moss-rknn-probe"
    assert report["existing_parent"] == "/home/cat"
    assert report["same_filesystem_as_root"] is True
    assert report["disk"]["free_mb"] >= 2048
    assert any("workspace does not exist" in error for error in report["errors"])
    assert any("under home directory" in error for error in report["errors"])
    assert any("root filesystem" in error for error in report["errors"])


def test_rk3576_target_mnt_rknn_workspace_is_not_prepared_yet():
    report = json.loads((EVIDENCE_DIR / "rk3576-rknn-workspace-mnt-not-ready.json").read_text(encoding="utf-8"))

    assert report["passed"] is False
    assert report["workspace"] == "/mnt/rknn-workspace/moss-rknn-workspace"
    assert report["existing_parent"] == "/mnt"
    assert report["same_filesystem_as_root"] is True
    assert report["disk"]["free_mb"] >= 2048
    assert any("workspace does not exist" in error for error in report["errors"])
    assert any("root filesystem" in error for error in report["errors"])


def test_rk3576_release_audit_rknn_workspace_gate_blocks_when_not_prepared():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-release-audit-rknn-workspace-required.json").read_text(encoding="utf-8")
    )

    assert report["passed"] is False
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["ort_evidence"]["passed"] is True
    assert report["checks"]["production_entrypoint"]["passed"] is True
    assert report["checks"]["rknn_candidate_config"]["passed"] is True
    assert report["checks"]["rknn_workspace"]["passed"] is False
    assert any("workspace does not exist" in error for error in report["errors"])
    assert any("root filesystem" in error for error in report["errors"])


def test_rk3576_release_audit_rknn_workspace_deployment_gate_blocks_when_not_prepared():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-release-audit-rknn-workspace-deployment-required.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["passed"] is False
    assert report["checks"]["ort_config"]["passed"] is True
    assert report["checks"]["production_entrypoint"]["passed"] is True
    assert report["checks"]["rknn_candidate_config"]["passed"] is True
    assert report["checks"]["rknn_workspace_deployment"]["passed"] is False
    assert report["checks"]["rknn_workspace_deployment"]["fstab_entry_present"] is False
    assert report["checks"]["rknn_workspace_deployment"]["mounted"] is None
    assert any("missing fstab entry" in error for error in report["errors"])
    assert any("mount point is not mounted" in error for error in report["errors"])


def test_rk3576_production_runner_rknn_workspace_gate_blocks_when_not_prepared():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-production-server-rknn-workspace-required.json").read_text(encoding="utf-8")
    )

    assert report["passed"] is False
    assert report["checks"]["rknn_workspace"]["passed"] is False
    assert report["service"]["dry_run"] is True
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}
    assert any("workspace does not exist" in error for error in report["errors"])


def test_rk3576_production_runner_rknn_workspace_deployment_gate_blocks_when_not_prepared():
    report = json.loads(
        (EVIDENCE_DIR / "rk3576-moss-production-server-rknn-workspace-deployment-required.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["passed"] is False
    assert report["checks"]["rknn_workspace_deployment"]["passed"] is False
    assert report["checks"]["rknn_workspace_deployment"]["fstab_entry_present"] is False
    assert report["checks"]["rknn_workspace_deployment"]["mounted"] is None
    assert report["service"]["dry_run"] is True
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}
    assert any("mount point is not mounted" in error for error in report["errors"])

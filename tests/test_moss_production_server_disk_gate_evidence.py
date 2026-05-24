"""Evidence lock for RK3576 MOSS production server disk preflight."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "moss" / "rk3576-moss-production-server-disk-gate.json"


def test_rk3576_production_server_disk_gate_currently_passes():
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert report["errors"] == []
    assert report["checks"]["production_entrypoint"]["passed"] is True
    assert report["checks"]["root_disk"]["passed"] is True
    assert report["checks"]["root_disk"]["free_mb"] >= 512
    assert report["service"]["dry_run"] is True
    assert report["service"]["env"] == {"CONFIG": "configs/rk3576-moss-ort-stream.yaml"}

"""Evidence lock for clean uvicorn signal shutdown on RK3576."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "docs" / "evidence" / "moss" / "rk3576-uvicorn-signal-smoke.log"


def test_rk3576_uvicorn_sigterm_shutdown_log_is_clean():
    text = LOG.read_text(encoding="utf-8")

    assert "Shutting down" in text
    assert "Shutdown complete" in text
    assert "Application shutdown complete" in text
    assert "Traceback" not in text
    assert "SystemExit" not in text
    assert "ERROR" not in text

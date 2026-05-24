"""Unit checks for the MOSS service profile verifier wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_service_profile.py"
    spec = importlib.util.spec_from_file_location("verify_moss_service_profile", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_verifier_parses_env_overrides():
    verifier = _load_verifier()

    env = verifier._parse_env(["TTS_BACKEND=moss_ort", "MOSS_ORT_SEED=314", "EMPTY="])

    assert env == {"TTS_BACKEND": "moss_ort", "MOSS_ORT_SEED": "314", "EMPTY": ""}


def test_profile_verifier_rejects_bad_env_override():
    verifier = _load_verifier()

    with pytest.raises(ValueError, match="KEY=VALUE"):
        verifier._parse_env(["TTS_BACKEND"])

    with pytest.raises(ValueError, match="key must not be empty"):
        verifier._parse_env(["=moss_ort"])


def test_profile_verifier_writes_readiness_failure_report(tmp_path):
    verifier = _load_verifier()
    json_out = tmp_path / "failure.json"
    log_file = tmp_path / "uvicorn.log"

    verifier._write_failure_report(
        str(json_out),
        base_url="http://127.0.0.1:8625",
        ws_url="ws://127.0.0.1:8625/dialogue",
        error="service readiness failed: missing manifest",
        log_file=log_file,
    )

    text = json_out.read_text(encoding="utf-8")

    assert '"passed": false' in text
    assert "service readiness failed: missing manifest" in text
    assert str(log_file) in text

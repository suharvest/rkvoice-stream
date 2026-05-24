"""Config contract checks for the MOSS RKNN production-candidate profile."""

from __future__ import annotations

from pathlib import Path

from rkvoice_stream import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_rk3576_moss_rknn_profile_requires_production_default_manifest():
    config = load_config(str(ROOT / "configs" / "rk3576-moss-rknn-stream.yaml"))
    tts = config["tts"]

    assert tts["backend"] == "moss_rknn"
    assert tts["require_backend"] == 1
    assert tts["manifest"] == "moss-rknn-manifest.json"
    assert tts["require_production_default"] == 1
    assert tts["worker_bin"] == "/opt/rkvoice-workers/moss_rknn_worker"
    assert tts["chunk_frames"] > 0

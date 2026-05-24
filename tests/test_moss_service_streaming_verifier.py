"""Unit checks for MOSS service streaming verifier gates."""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_service_streaming.py"
    spec = importlib.util.spec_from_file_location("verify_moss_service_streaming", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_health() -> dict:
    return {
        "tts": True,
        "tts_backend": "moss_ort",
        "streaming_tts": True,
        "tts_info": {
            "profile": {
                "voice": "Junhao",
                "seed": 314,
                "codec_batch_frames": 4,
                "codec_full_loaded": False,
                "codec_async": False,
                "cache_voice_prefix": False,
            },
            "manifest": {"name": "moss-ort-manifest.json", "validated": True},
            "hybrid": {"enabled": False},
            "streaming_stats": {
                "requests": 0,
                "completed": 0,
                "errors": 0,
                "active": 0,
                "chunks": 0,
                "last_error": None,
                "last_error_time": None,
            },
        },
    }


def _valid_dialogue() -> dict:
    return {
        "sample_rate": 48000,
        "binary_chunks": 7,
        "payload_bytes": 30720,
        "first_payload_ms": 1100.0,
        "max_payload_gap_ms": 600.0,
        "last_text": {"done": True},
    }


def test_service_verifier_accepts_valid_streaming_gates():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)

    errors = verifier._collect_gate_errors(
        health=_valid_health(),
        health_after={**_valid_health(), "tts_info": {**_valid_health()["tts_info"], "streaming_stats": {
            "requests": 2,
            "completed": 2,
            "errors": 0,
            "active": 0,
            "chunks": 9,
            "last_error": None,
            "last_error_time": None,
        }}},
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue=_valid_dialogue(),
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
        max_dialogue_payload_gap_ms=1500.0,
        require_manifest_validated=True,
        expected_voice="Junhao",
        expected_seed=314,
        expected_manifest="moss-ort-manifest.json",
        expected_codec_batch_frames=4,
        require_production_runtime=True,
        min_dialogue_binary_chunks=7,
    )

    assert errors == []


def test_service_verifier_rejects_runtime_profile_mismatch():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)
    health = _valid_health()
    health["tts_info"]["profile"]["voice"] = "Lingyu"
    health["tts_info"]["manifest"]["validated"] = False

    errors = verifier._collect_gate_errors(
        health=health,
        health_after=health,
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue=_valid_dialogue(),
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
        require_manifest_validated=True,
        expected_voice="Junhao",
        expected_seed=314,
        expected_manifest="moss-ort-manifest.json",
    )

    assert "health.tts_info.manifest.validated is not true" in errors
    assert "health.tts_info.profile.voice='Lingyu'" in errors


def test_service_verifier_rejects_bad_stream_chunk_cadence():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)
    health = _valid_health()
    health["tts_info"]["profile"]["codec_batch_frames"] = 8

    errors = verifier._collect_gate_errors(
        health=health,
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue={**_valid_dialogue(), "binary_chunks": 4},
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
        expected_codec_batch_frames=4,
        min_dialogue_binary_chunks=7,
    )

    assert "health.tts_info.profile.codec_batch_frames=8" in errors
    assert "dialogue binary_chunks=4 below 7" in errors


def test_service_verifier_rejects_experimental_runtime_flags():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)
    health = _valid_health()
    health["tts_info"]["profile"]["codec_async"] = True
    health["tts_info"]["profile"]["codec_full_loaded"] = True
    health["tts_info"]["hybrid"]["enabled"] = True

    errors = verifier._collect_gate_errors(
        health=health,
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue=_valid_dialogue(),
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
        require_production_runtime=True,
    )

    assert "health.tts_info.profile.codec_async=True" in errors
    assert "health.tts_info.profile.codec_full_loaded=True" in errors
    assert "health.tts_info.hybrid.enabled=True" in errors


def test_service_verifier_rejects_streaming_stats_error_increase():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)
    health_after = _valid_health()
    health_after["tts_info"]["streaming_stats"] = {
        "requests": 2,
        "completed": 1,
        "errors": 1,
        "active": 0,
        "chunks": 3,
        "last_error": "codec failed",
        "last_error_time": 1.0,
    }

    errors = verifier._collect_gate_errors(
        health=_valid_health(),
        health_after=health_after,
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue=_valid_dialogue(),
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
        require_production_runtime=True,
    )

    assert "streaming_stats.errors increased 0->1" in errors


def test_service_verifier_rejects_slow_first_payloads():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)

    errors = verifier._collect_gate_errors(
        health=_valid_health(),
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=1400.0,
        tts_first_payload_ms=1600.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue={**_valid_dialogue(), "first_payload_ms": 1700.0},
        dialogue_wall_ms=1400.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
    )

    assert any("tts stream first_payload_ms=1600.000 exceeds 1500.000" == item for item in errors)
    assert any("dialogue first_payload_ms=1700.000 exceeds 1500.000" == item for item in errors)


def test_service_verifier_rejects_slow_payload_cadence():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)

    errors = verifier._collect_gate_errors(
        health=_valid_health(),
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=2400.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=0.0,
        max_tts_first_payload_ms=1500.0,
        dialogue={**_valid_dialogue(), "max_payload_gap_ms": 1800.0},
        dialogue_wall_ms=3000.0,
        max_dialogue_wall_ms=0.0,
        max_dialogue_first_payload_ms=1500.0,
        max_dialogue_payload_gap_ms=1500.0,
    )

    assert "dialogue max_payload_gap_ms=1800.000 exceeds 1500.000" in errors


def test_service_verifier_can_disable_full_wall_time_gate():
    verifier = _load_verifier()
    body = struct.pack("<I", 48000) + (b"\0" * 30720)
    tts_stream = verifier._check_tts_stream(body, expected_sample_rate=48000, min_payload_bytes=30720)

    errors = verifier._collect_gate_errors(
        health=_valid_health(),
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=6000.0,
        tts_first_payload_ms=1100.0,
        max_tts_wall_ms=0.0,
        max_tts_first_payload_ms=1500.0,
        dialogue=_valid_dialogue(),
        dialogue_wall_ms=6000.0,
        max_dialogue_wall_ms=0.0,
        max_dialogue_first_payload_ms=1500.0,
    )

    assert errors == []


def test_service_verifier_rejects_bad_payload_contract():
    verifier = _load_verifier()
    bad_body = struct.pack("<I", 16000) + (b"\0" * 16)
    tts_stream = verifier._check_tts_stream(bad_body, expected_sample_rate=48000, min_payload_bytes=30720)

    errors = verifier._collect_gate_errors(
        health={**_valid_health(), "streaming_tts": False},
        expected_backend="moss_ort",
        expected_sample_rate=48000,
        min_payload_bytes=30720,
        tts_stream=tts_stream,
        tts_wall_ms=100.0,
        tts_first_payload_ms=100.0,
        max_tts_wall_ms=2000.0,
        max_tts_first_payload_ms=1500.0,
        dialogue={**_valid_dialogue(), "payload_bytes": 16, "last_text": {"done": False}},
        dialogue_wall_ms=100.0,
        max_dialogue_wall_ms=2000.0,
        max_dialogue_first_payload_ms=1500.0,
    )

    assert "health.streaming_tts is not true" in errors
    assert any(item.startswith("tts stream gate failed") for item in errors)
    assert "dialogue payload_bytes=16" in errors
    assert "dialogue did not finish cleanly: {'done': False}" in errors

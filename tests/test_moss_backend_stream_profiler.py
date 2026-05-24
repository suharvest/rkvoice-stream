"""Unit checks for the MOSS backend stream profiler summary gates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_profiler():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "smoke_moss_hybrid_backend.py"
    spec = importlib.util.spec_from_file_location("smoke_moss_hybrid_backend", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stream_profiler_summarizes_stage_timings():
    profiler = _load_profiler()
    chunks = [np.zeros((3840, 2), dtype=np.float32), np.zeros((3840, 2), dtype=np.float32)]
    metas = [
        {"ttfa_ms": 980, "prefill_ms": 780.5, "sampler_ms": 90, "codec_ms": 80},
        {"decode_ms": 62, "sampler_ms": 85, "codec_ms": 145},
    ]

    summary = profiler._summarize(12000.0, 2200.0, chunks, metas)

    assert summary["chunks"] == 2
    assert summary["total_samples"] == 7680
    assert summary["audio_frames"] == 2
    assert summary["ttfa_ms"] == 980.0
    assert summary["prefill_ms"] == 780.5
    assert summary["max_codec_ms"] == 145.0
    assert summary["max_sampler_ms"] == 90.0
    assert summary["max_decode_ms"] == 62.0
    assert summary["chunk_shapes"] == [[3840, 2], [3840, 2]]


def test_stream_profiler_rejects_gate_regressions():
    profiler = _load_profiler()
    summary = {
        "chunks": 4,
        "ttfa_ms": 1600.0,
        "prefill_ms": 1200.0,
        "max_codec_ms": 180.0,
    }

    errors = profiler._collect_gate_errors(
        summary,
        min_chunks=7,
        min_audio_frames=20,
        max_ttfa_ms=1500.0,
        max_prefill_ms=1000.0,
        max_codec_ms=170.0,
    )

    assert "chunks=4 below 7" in errors
    assert "audio_frames=None below 20" in errors
    assert "ttfa_ms=1600.000 exceeds 1500.000" in errors
    assert "prefill_ms=1200.000 exceeds 1000.000" in errors
    assert "max_codec_ms=180.000 exceeds 170.000" in errors

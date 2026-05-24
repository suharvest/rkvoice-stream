"""Unit checks for the MOSS sampler ORT profiler helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_profiler():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "profile_moss_sampler_ort.py"
    spec = importlib.util.spec_from_file_location("profile_moss_sampler_ort", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_summary_uses_node_name_before_op_name():
    profiler = _load_profiler()
    events = [
        {
            "cat": "Node",
            "dur": 2000,
            "name": "/text_lm_head/MatMul_kernel_time",
            "args": {"node_name": "/text_lm_head/MatMul_kernel_time", "op_name": "MatMul"},
        },
        {
            "cat": "Node",
            "dur": 1000,
            "name": "/audio_head/TopK_kernel_time",
            "args": {"node_name": "/audio_head/TopK_kernel_time", "op_name": "TopK"},
        },
        {"cat": "Session", "dur": 5000, "name": "model_run"},
    ]

    summary = profiler._summarize_profile(events, top_k=4)

    assert summary["node_event_count"] == 2
    assert summary["node_total_ms"] == 3.0
    assert summary["top_ops"][0]["name"] == "MatMul"
    assert summary["top_nodes"][0]["name"] == "/text_lm_head/MatMul_kernel_time"
    assert summary["top_nodes"][0]["count"] == 1


def test_time_summary_reports_percentiles():
    profiler = _load_profiler()

    summary = profiler._summarize_times([10.0, 20.0, 30.0, 100.0])

    assert summary["count"] == 4
    assert summary["mean_ms"] == 40.0
    assert summary["p50_ms"] == 30.0
    assert summary["p95_ms"] == 100.0

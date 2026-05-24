from __future__ import annotations

import numpy as np

from models.tts.moss.compare_moss_hybrid_sampler import _sampler_margin_report


def _debug(indices: list[int], probs: list[float]) -> dict[str, np.ndarray]:
    cdf = np.cumsum(np.asarray(probs, dtype=np.float32))
    return {
        "topk_indices": np.asarray(indices, dtype=np.int64).reshape(1, -1),
        "topk_values": np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32).reshape(1, -1),
        "final_probs": np.asarray(probs, dtype=np.float32).reshape(1, -1),
        "final_cdf": cdf.reshape(1, -1),
        "selected_token": np.asarray([indices[0]], dtype=np.int64),
    }


def test_sampler_margin_report_detects_missing_cross_topk_tokens():
    full: dict[str, np.ndarray] = {}
    hybrid: dict[str, np.ndarray] = {}
    full_frame = np.arange(16, dtype=np.int32)
    hybrid_frame = np.arange(100, 116, dtype=np.int32)
    audio_u = np.full(16, 0.55, dtype=np.float32)
    for channel in range(16):
        full.update({f"ch{channel}.{key}": value for key, value in _debug([channel, 200, 201, 202], [0.4, 0.3, 0.2, 0.1]).items()})
        hybrid.update({f"ch{channel}.{key}": value for key, value in _debug([100 + channel, 300, 301, 302], [0.4, 0.3, 0.2, 0.1]).items()})

    report = _sampler_margin_report(full, hybrid, full_frame, hybrid_frame, audio_u)

    assert report["summary"]["mismatched_channels"] == 16
    assert report["summary"]["mismatched_full_token_missing_from_hybrid_topk"] == 16
    assert report["summary"]["mismatched_hybrid_token_missing_from_full_topk"] == 16
    assert report["channels"][0]["full_random_margin"] == np.float32(0.15).item()


def test_sampler_margin_report_keeps_equal_tokens_out_of_mismatch_summary():
    full: dict[str, np.ndarray] = {}
    hybrid: dict[str, np.ndarray] = {}
    full_frame = np.arange(16, dtype=np.int32)
    hybrid_frame = np.arange(16, dtype=np.int32)
    audio_u = np.full(16, 0.45, dtype=np.float32)
    for channel in range(16):
        values = _debug([channel, 200, 201, 202], [0.4, 0.3, 0.2, 0.1])
        full.update({f"ch{channel}.{key}": value for key, value in values.items()})
        hybrid.update({f"ch{channel}.{key}": value for key, value in values.items()})

    report = _sampler_margin_report(full, hybrid, full_frame, hybrid_frame, audio_u)

    assert report["summary"]["mismatched_channels"] == 0
    assert report["summary"]["mismatched_full_random_margin_min"] is None
    assert all(item["token_equal"] for item in report["channels"])

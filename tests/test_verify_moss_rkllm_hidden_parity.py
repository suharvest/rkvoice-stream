from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_rkllm_hidden_parity.py"
    spec = importlib.util.spec_from_file_location("verify_moss_rkllm_hidden_parity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metrics_reports_cosine_and_rel_l2():
    module = _load_module()
    expected = np.ones((1, 2, 3), dtype=np.float32)
    actual = expected + 0.01
    metrics = module._metrics(actual, expected)
    assert metrics["shape"] == [1, 2, 3]
    assert metrics["finite"] is True
    assert metrics["rel_l2"] > 0
    assert metrics["cosine"] > 0.99


def test_probe_rows_use_17_column_moss_layout():
    module = _load_module()

    class Config:
        bos_token_id = 4
        eos_token_id = 5
        moss_rkllm = {"audio_pad_token_id": 1024}

    rows, mask = module._build_probe_rows(Config(), 4)
    assert rows.shape == (1, 4, 17)
    assert mask.shape == (1, 4)
    assert rows[0, 0, 0] == 4
    assert rows[0, 1, 1] == 0
    assert rows[0, 3, 1] == 1024

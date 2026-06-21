from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "verify_moss_codec_split_rknn_layer_pipeline.py"
    spec = importlib.util.spec_from_file_location("verify_moss_codec_split_rknn_layer_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_codec_split_rknn_pipeline_metrics_detect_small_error():
    module = _load_module()

    metrics = module._metrics(np.asarray([1.0, 2.0], dtype=np.float32), np.asarray([1.01, 1.99], dtype=np.float32))

    assert metrics["finite"] is True
    assert 0.0 < metrics["max_abs"] < 0.02
    assert 0.0 < metrics["rel_l2"] < 0.02
    assert metrics["cosine"] > 0.999


def test_parse_codec_split_rknn_pipeline_layers_range():
    module = _load_module()

    assert module._parse_layers("0,2-4,7") == [0, 2, 3, 4, 7]

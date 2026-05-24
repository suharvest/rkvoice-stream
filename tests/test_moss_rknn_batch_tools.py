"""Unit checks for MOSS RKNN batch helper path handling."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    path = ROOT / "models" / "tts" / "moss" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_batch_layer_range_parser():
    module = _load("build_moss_rknn_island_batch")

    assert module._parse_layers("0-2,5,7-8") == [0, 1, 2, 5, 7, 8]


def test_verify_batch_preserves_s320_in_artifact_paths():
    module = _load("verify_moss_rknn_island_batch")

    onnx, rknn = module._artifact_paths(Path("/artifacts"), "cattn", 10, 320, "fp16", "rk3576")

    assert onnx == Path("/artifacts/moss_block10_cattn.s320.onnx")
    assert rknn == Path("/artifacts/moss_block10_cattn.s320.fp16.rk3576.rknn")

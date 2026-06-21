from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_moss_rkllm_embed_only_boundary.py"
    spec = importlib.util.spec_from_file_location("verify_moss_rkllm_embed_only_boundary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_to_numpy_cache_uses_present_key_value_names():
    module = _load_module()
    cache = (
        (torch.ones(1, 2, 3, 4), torch.zeros(1, 2, 3, 4)),
        (torch.full((1, 2, 3, 4), 2.0), torch.full((1, 2, 3, 4), 3.0)),
    )
    out = module._to_numpy_cache(cache)
    assert sorted(out) == ["present_key_0", "present_key_1", "present_value_0", "present_value_1"]
    assert out["present_key_0"].dtype == np.float32
    assert out["present_value_1"][0, 0, 0, 0] == 3.0


def test_torch_past_from_cache_roundtrips_shapes():
    module = _load_module()
    cache = {
        "present_key_0": np.ones((1, 2, 3, 4), dtype=np.float32),
        "present_value_0": np.zeros((1, 2, 3, 4), dtype=np.float32),
    }
    past = module._torch_past_from_cache(cache, 1)
    assert len(past) == 1
    assert past[0][0].shape == (1, 2, 3, 4)
    assert past[0][1].dtype == torch.float32

"""Unit checks for the MOSS ORT config contract verifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rkvoice_stream import load_config


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_ort_config.py"
    spec = importlib.util.spec_from_file_location("verify_moss_ort_config", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_config() -> dict:
    return {
        "asr": {"backend": None},
        "tts": {
            "backend": "moss_ort",
            "require_backend": 1,
            "model_dir": "/opt/tts/models/moss-tts-nano-onnx",
            "manifest": "moss-ort-manifest.json",
            "sample_rate": 48000,
            "channels": 2,
            "threads": 6,
            "prefill_threads": 8,
            "decode_threads": 5,
            "codec_threads": 5,
            "prefill_seq": 0,
            "max_new_frames": 20,
            "codec_streaming": 1,
            "codec_batch_frames": 3,
            "cache_voice_prefix": 0,
            "warmup_text": "你好",
            "voice": "Junhao",
            "seed": 314,
            "allow_deterministic_fallback": 0,
        },
    }


def test_moss_ort_config_contract_accepts_production_profile():
    verifier = _load_verifier()

    assert verifier._check_config(_valid_config()) == []


def test_moss_ort_config_contract_rejects_drift():
    verifier = _load_verifier()
    config = _valid_config()
    config["tts"]["voice"] = "Lingyu"
    config["tts"]["cache_voice_prefix"] = 1
    config["tts"]["warmup_text"] = ""

    errors = verifier._check_config(config)

    assert "tts.voice='Lingyu', expected 'Junhao'" in errors
    assert "tts.cache_voice_prefix=1, expected 0" in errors
    assert "tts.warmup_text='', expected '你好'" in errors


def test_moss_ort_config_contract_rejects_experimental_hybrid_flags():
    verifier = _load_verifier()
    config = _valid_config()
    config["tts"]["hybrid_rknn"] = 1
    config["tts"]["hybrid_split"] = "fc_out_only"
    config["tts"]["codec_async"] = 1
    config["tts"]["load_full_codec"] = 1

    errors = verifier._check_config(config)

    assert "tts.hybrid_rknn is experimental and must not be enabled in the production ORT profile" in errors
    assert "tts.hybrid_split is experimental and must not be enabled in the production ORT profile" in errors
    assert "tts.codec_async is experimental and must not be enabled in the production ORT profile" in errors
    assert "tts.load_full_codec is experimental and must not be enabled in the production ORT profile" in errors


def test_moss_ort_config_contract_allows_disabled_experimental_keys():
    verifier = _load_verifier()
    config = _valid_config()
    config["tts"]["hybrid_rknn"] = 0
    config["tts"]["hybrid_split"] = ""
    config["tts"]["codec_async"] = False
    config["tts"]["load_full_codec"] = False

    assert verifier._check_config(config) == []


def test_checked_in_rk3576_ort_profile_has_no_experimental_fields():
    verifier = _load_verifier()
    config_path = Path(__file__).resolve().parents[1] / "configs" / "rk3576-moss-ort-stream.yaml"
    config = load_config(str(config_path))

    assert verifier._check_config(config) == []

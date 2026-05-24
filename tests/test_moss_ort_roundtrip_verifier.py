"""Unit checks for the MOSS ORT roundtrip verifier helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_verifier():
    path = Path(__file__).resolve().parents[1] / "models" / "tts" / "moss" / "verify_moss_ort_roundtrip.py"
    spec = importlib.util.spec_from_file_location("verify_moss_ort_roundtrip", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_roundtrip_verifier_cer_normalizes_chinese_and_punctuation():
    verifier = _load_verifier()

    assert verifier.cer("欢迎使用语音服务", "欢迎") == 0.75
    assert verifier.cer("Hello, 你好!", "hello你好") == 0.0


def test_roundtrip_verifier_gate_summary():
    verifier = _load_verifier()
    summary = verifier.summarize(
        [
            {
                "cer": 0.25,
                "meta": {"ttfa_ms": 1800, "codec_ms": 70},
                "wav_info": {"rms": 0.04},
            },
            {
                "cer": 0.50,
                "meta": {"ttfa_ms": 2000, "codec_ms": 80},
                "wav_info": {"rms": 0.03},
            },
        ]
    )

    assert summary["avg_cer"] == 0.375
    assert summary["max_cer"] == 0.5
    assert summary["min_rms"] == 0.03
    assert summary["max_ttfa_ms"] == 2000
    assert summary["max_codec_ms"] == 80


def test_roundtrip_verifier_default_ttfa_gate_matches_production_profile(monkeypatch, tmp_path):
    verifier = _load_verifier()
    monkeypatch.setattr(
        verifier.sys,
        "argv",
        [
            "verify_moss_ort_roundtrip.py",
            "--model-dir",
            str(tmp_path / "moss"),
            "--asr-model-dir",
            str(tmp_path / "asr"),
        ],
    )

    args = verifier.parse_args()

    assert args.max_ttfa_ms == 1500.0


def test_roundtrip_verifier_propagates_seed_to_stage_command(monkeypatch, tmp_path):
    verifier = _load_verifier()

    class _Args:
        model_dir = tmp_path / "moss"
        asr_model_dir = tmp_path / "asr"
        out_dir = tmp_path / "out"
        threads = 6
        max_new_frames = 20
        voice = "Lingyu"
        prefill_seq = 0
        codec_streaming = 1
        warmup_text = "你好"
        sentences = None
        seed = 2026
        manifest = "moss-ort-manifest.json"
        prefill_threads = 8
        decode_threads = None
        sampler_threads = None
        codec_threads = 5
        codec_batch_frames = 4

    captured = {}

    def _fake_run(cmd, check, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(verifier.subprocess, "run", _fake_run)

    verifier.run_stage_subprocess(_Args, "tts")

    assert captured["check"] is True
    assert "--seed" in captured["cmd"]
    idx = captured["cmd"].index("--seed")
    assert captured["cmd"][idx + 1] == "2026"
    assert "--manifest" in captured["cmd"]
    manifest_idx = captured["cmd"].index("--manifest")
    assert captured["cmd"][manifest_idx + 1] == "moss-ort-manifest.json"
    assert captured["cmd"][captured["cmd"].index("--prefill-threads") + 1] == "8"
    assert captured["cmd"][captured["cmd"].index("--codec-threads") + 1] == "5"
    assert captured["cmd"][captured["cmd"].index("--codec-batch-frames") + 1] == "4"


def test_roundtrip_verifier_seed_sets_backend_rng_env(monkeypatch, tmp_path):
    verifier = _load_verifier()

    class _Args:
        model_dir = tmp_path / "moss"
        out_dir = tmp_path / "out"
        threads = 6
        max_new_frames = 20
        prefill_seq = 0
        voice = "Lingyu"
        codec_streaming = 1
        warmup_text = ""
        seed = 2026
        sentences = "你好"
        manifest = "moss-ort-manifest.json"
        prefill_threads = 8
        decode_threads = None
        sampler_threads = None
        codec_threads = 5
        codec_batch_frames = 4

    class _Backend:
        def preload(self):
            pass

        def synthesize(self, text, **kwargs):
            assert kwargs == {"max_new_frames": 20}
            return b"RIFF", {"ttfa_ms": 1, "codec_ms": 1}

    import types
    import sys

    module = types.SimpleNamespace(MossORTBackend=_Backend)
    monkeypatch.setitem(sys.modules, "rkvoice_stream.backends.tts.moss_ort", module)

    verifier.run_tts_stage(_Args)

    assert verifier.os.environ["MOSS_ORT_SEED"] == "2026"
    assert verifier.os.environ["MOSS_ORT_MANIFEST"] == "moss-ort-manifest.json"
    assert verifier.os.environ["MOSS_ORT_PREFILL_THREADS"] == "8"
    assert verifier.os.environ["MOSS_ORT_CODEC_THREADS"] == "5"
    assert verifier.os.environ["MOSS_ORT_CODEC_BATCH_FRAMES"] == "4"
    assert verifier.os.environ["MOSS_ORT_CODEC_ASYNC"] == "0"
    assert verifier.os.environ["MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK"] == "0"
    assert verifier.os.environ["MOSS_ORT_HYBRID_RKNN"] == "0"

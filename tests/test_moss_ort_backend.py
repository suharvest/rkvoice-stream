"""MOSS ORT backend contract tests."""

from __future__ import annotations

import numpy as np
import pytest

from rkvoice_stream import _apply_tts_env
from rkvoice_stream.backends.tts.moss_ort import (
    MossORTBackend,
    _HybridPrefillSession,
    _parse_hybrid_layers,
    _trim_kv_to_length,
    default_moss_hybrid_fc_split_artifacts,
    default_moss_hybrid_ln1_cattn_artifacts,
)


def test_moss_ort_rejects_text_without_tokenizer_by_default(monkeypatch):
    monkeypatch.delenv("MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK", raising=False)
    backend = MossORTBackend()
    backend._sp = None

    with pytest.raises(RuntimeError, match="sentencepiece is required"):
        backend._build_prefill_rows("hello")


def test_moss_ort_deterministic_probe_rows_are_fixed_width(monkeypatch):
    monkeypatch.setenv("MOSS_ORT_ALLOW_DETERMINISTIC_FALLBACK", "1")
    monkeypatch.setenv("MOSS_ORT_PREFILL_SEQ", "64")
    backend = MossORTBackend()
    backend._sp = None

    rows, mode = backend._build_prefill_rows("hello")

    assert mode == "deterministic_probe"
    assert rows.shape == (1, 32, 17)
    assert rows.dtype == np.int32
    assert np.all(rows[:, :, 0] == 1)


def test_moss_ort_normalizes_codec_stereo_to_interleaved_frames():
    backend = MossORTBackend()
    audio = np.zeros((1, 2, 3840), dtype=np.float32)
    audio[0, 0, :] = 0.25
    audio[0, 1, :] = -0.25

    chunk = backend._normalize_audio(audio, np.array([3840], dtype=np.int32))

    assert chunk.shape == (3840, 2)
    assert chunk.dtype == np.float32
    assert chunk[0].tolist() == [0.25, -0.25]


def test_moss_ort_trims_hybrid_kv_padding_to_actual_length():
    cache = np.zeros((1, 12, 320, 64), dtype=np.float32)

    trimmed = _trim_kv_to_length(cache, 297)

    assert trimmed.shape == (1, 12, 297, 64)
    assert trimmed.dtype == np.float32


def test_moss_ort_rejects_unsafe_prefix_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSS_ORT_CACHE_VOICE_PREFIX", "1")
    monkeypatch.setenv("MOSS_ORT_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("MOSS_ORT_MANIFEST", raising=False)
    for name in (
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "moss_audio_tokenizer_decode_shared.data",
        "tokenizer.model",
    ):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "tts_browser_onnx_meta.json").write_text('{"model_config": {}}', encoding="utf-8")
    (tmp_path / "codec_browser_onnx_meta.json").write_text('{"codec_config": {}}', encoding="utf-8")

    class _FakeOrt:
        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0
            graph_optimization_level = None

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = object()

        class InferenceSession:
            def __init__(self, *args, **kwargs):
                pass

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _FakeOrt)
    monkeypatch.setattr(MossORTBackend, "_load_tokenizer", lambda self: None)
    backend = MossORTBackend()

    with pytest.raises(RuntimeError, match="CACHE_VOICE_PREFIX is disabled"):
        backend.preload()


def test_moss_ort_warmup_uses_fixed_sampler_values(monkeypatch):
    backend = MossORTBackend()
    backend._sp = object()
    backend._tts_meta = {"onnx": {"prefill_output_names": ["global_hidden", "present_key_0", "present_value_0"]}}
    backend._codec_stream_session = None
    backend._build_prefill_rows = lambda text: (np.ones((1, 2, 17), dtype=np.int32), "text")
    backend._pad_token_id = lambda: 3
    backend._make_audio_row = lambda frame: np.asarray([9, *frame], dtype=np.int32)
    backend._decode_step = lambda row, kv, past_len: (np.zeros((1, 768), dtype=np.float32), kv)

    class _Prefill:
        def run(self, names, feeds):
            return [
                np.zeros((1, 2, 768), dtype=np.float32),
                np.zeros((1, 2, 12, 64), dtype=np.float32),
                np.zeros((1, 2, 12, 64), dtype=np.float32),
            ]

    class _Sampler:
        def __init__(self):
            self.feeds = None

        def run(self, names, feeds):
            self.feeds = feeds
            return np.asarray([1], dtype=np.int32), np.zeros((1, 16), dtype=np.int32)

    class _Codec:
        def run(self, names, feeds):
            return np.zeros((1, 2, 3840), dtype=np.float32), np.asarray([3840], dtype=np.int32)

    sampler = _Sampler()
    backend._prefill = _Prefill()
    backend._sampler = sampler
    backend._codec = _Codec()

    backend._warmup_sessions("hello")

    assert sampler.feeds is not None
    assert sampler.feeds["assistant_random_u"].tolist() == [0.5]
    assert sampler.feeds["audio_random_u"].shape == (1, 16)
    assert np.all(sampler.feeds["audio_random_u"] == 0.5)


def test_moss_ort_skips_full_codec_session_when_streaming_codec_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSS_ORT_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MOSS_ORT_CODEC_STREAMING", "1")
    monkeypatch.delenv("MOSS_ORT_MANIFEST", raising=False)
    monkeypatch.delenv("MOSS_ORT_LOAD_FULL_CODEC", raising=False)
    for name in (
        "moss_tts_prefill.onnx",
        "moss_tts_decode_step.onnx",
        "moss_tts_local_fixed_sampled_frame.onnx",
        "moss_audio_tokenizer_decode_full.onnx",
        "moss_audio_tokenizer_decode_step.onnx",
        "moss_tts_global_shared.data",
        "moss_tts_local_shared.data",
        "moss_audio_tokenizer_decode_shared.data",
        "tokenizer.model",
    ):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "tts_browser_onnx_meta.json").write_text('{"model_config": {}}', encoding="utf-8")
    (tmp_path / "codec_browser_onnx_meta.json").write_text(
        '{"codec_config": {}, "streaming_decode": {"transformer_offsets": [], "attention_caches": []}}',
        encoding="utf-8",
    )

    loaded_paths = []

    class _FakeOrt:
        class SessionOptions:
            intra_op_num_threads = 0
            inter_op_num_threads = 0
            graph_optimization_level = None

        class GraphOptimizationLevel:
            ORT_ENABLE_ALL = object()

        class InferenceSession:
            def __init__(self, path, *args, **kwargs):
                loaded_paths.append(str(path))

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", _FakeOrt)
    monkeypatch.setattr(MossORTBackend, "_load_tokenizer", lambda self: None)

    backend = MossORTBackend()
    backend.preload()

    assert any(path.endswith("moss_audio_tokenizer_decode_step.onnx") for path in loaded_paths)
    assert not any(path.endswith("moss_audio_tokenizer_decode_full.onnx") for path in loaded_paths)
    assert backend.runtime_info()["profile"]["codec_streaming"] is True
    assert backend.runtime_info()["profile"]["codec_full_loaded"] is False


def test_moss_ort_config_maps_hybrid_env(monkeypatch):
    for name in (
        "MOSS_ORT_HYBRID_RKNN",
        "MOSS_ORT_HYBRID_STRICT",
        "MOSS_ORT_HYBRID_DIR",
        "MOSS_ORT_HYBRID_RKNN_DIR",
        "MOSS_ORT_HYBRID_SEQ_LEN",
        "MOSS_ORT_HYBRID_SPLIT",
        "MOSS_ORT_HYBRID_LAYERS",
        "MOSS_ORT_PREFILL_THREADS",
        "MOSS_ORT_DECODE_THREADS",
        "MOSS_ORT_SAMPLER_THREADS",
        "MOSS_ORT_CODEC_THREADS",
        "MOSS_ORT_CODEC_BATCH_FRAMES",
        "MOSS_ORT_LOAD_FULL_CODEC",
        "MOSS_ORT_CODEC_ASYNC",
        "MOSS_ORT_SEED",
    ):
        monkeypatch.delenv(name, raising=False)

    _apply_tts_env(
        {
            "backend": "moss_ort",
            "model_dir": "/models/moss",
            "hybrid_rknn": 1,
            "hybrid_strict": 1,
            "hybrid_dir": "/models/moss-hybrid",
            "hybrid_rknn_dir": "/models/moss-mlp-only",
            "hybrid_seq_len": 320,
            "hybrid_split": "mlp_only",
            "hybrid_layers": "0,2,5,10",
            "prefill_threads": 8,
            "decode_threads": 5,
            "sampler_threads": 6,
            "codec_threads": 4,
            "codec_batch_frames": 4,
            "load_full_codec": 1,
            "codec_async": 1,
            "seed": 314,
        }
    )

    assert __import__("os").environ["MOSS_ORT_MODEL_DIR"] == "/models/moss"
    assert __import__("os").environ["MOSS_ORT_HYBRID_RKNN"] == "1"
    assert __import__("os").environ["MOSS_ORT_HYBRID_STRICT"] == "1"
    assert __import__("os").environ["MOSS_ORT_HYBRID_DIR"] == "/models/moss-hybrid"
    assert __import__("os").environ["MOSS_ORT_HYBRID_RKNN_DIR"] == "/models/moss-mlp-only"
    assert __import__("os").environ["MOSS_ORT_HYBRID_SEQ_LEN"] == "320"
    assert __import__("os").environ["MOSS_ORT_HYBRID_SPLIT"] == "mlp_only"
    assert __import__("os").environ["MOSS_ORT_HYBRID_LAYERS"] == "0,2,5,10"
    assert __import__("os").environ["MOSS_ORT_PREFILL_THREADS"] == "8"
    assert __import__("os").environ["MOSS_ORT_DECODE_THREADS"] == "5"
    assert __import__("os").environ["MOSS_ORT_SAMPLER_THREADS"] == "6"
    assert __import__("os").environ["MOSS_ORT_CODEC_THREADS"] == "4"
    assert __import__("os").environ["MOSS_ORT_CODEC_BATCH_FRAMES"] == "4"
    assert __import__("os").environ["MOSS_ORT_LOAD_FULL_CODEC"] == "1"
    assert __import__("os").environ["MOSS_ORT_CODEC_ASYNC"] == "1"
    assert __import__("os").environ["MOSS_ORT_SEED"] == "314"


def test_parse_hybrid_layers_accepts_ranges_and_rejects_invalid():
    assert _parse_hybrid_layers("all") == set(range(12))
    assert _parse_hybrid_layers("none") == set()
    assert _parse_hybrid_layers("0,2,5-7") == {0, 2, 5, 6, 7}

    with pytest.raises(ValueError, match="invalid"):
        _parse_hybrid_layers("12")


def test_moss_hybrid_fc_split_artifact_contract():
    fc_out = default_moss_hybrid_fc_split_artifacts(seq_len=320, split="fc_out_only")
    fc_in = default_moss_hybrid_fc_split_artifacts(seq_len=320, split="fc_in_act_only")

    assert "moss_block0_ln2.s320.onnx" in fc_out
    assert "moss_block0_fc_in_act.s320.onnx" in fc_out
    assert "moss_block0_fc_out.s320.fp16.rk3576.rknn" in fc_out
    assert "moss_block0_fc_in_act.s320.fp16.rk3576.rknn" in fc_in
    assert "moss_block0_fc_out.s320.onnx" in fc_in


def test_moss_hybrid_ln1_cattn_artifact_contract():
    artifacts = default_moss_hybrid_ln1_cattn_artifacts(seq_len=320, target="rk3576")

    assert "moss_embedding_prefix.s320.onnx" in artifacts
    assert "moss_final_norm.s320.onnx" in artifacts
    assert "moss_block0_attn_after_cattn.s320.onnx" in artifacts
    assert "moss_block0_ln1_cattn.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block0_ln2_mlp.s320.fp16.rk3576.rknn" in artifacts
    assert "moss_block0_attn_residual.s320.onnx" not in artifacts


def test_moss_hybrid_prefill_accepts_fc_split(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSS_ORT_HYBRID_MANIFEST", raising=False)
    layers = {0, 1, 4}
    for name in (
        "moss_embedding_prefix.s320.onnx",
        "moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12)],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_ln2.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_in_act.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_fc_out.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        (tmp_path / name).write_bytes(b"x")

    session = _HybridPrefillSession(
        artifact_dir=tmp_path,
        seq_len=320,
        threads=1,
        ort_module=object(),
        split="fc_out_only",
        layers=layers,
        rknn_dir=tmp_path,
    )

    assert session._split == "fc_out_only"
    assert session._rknn_layers == {0, 1, 4}


def test_moss_hybrid_prefill_accepts_ln1_cattn_split(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSS_ORT_HYBRID_MANIFEST", raising=False)
    layers = {0, 1, 4}
    for name in (
        "moss_embedding_prefix.s320.onnx",
        "moss_final_norm.s320.onnx",
        *[f"moss_block{layer}_attn_residual.s320.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_ln2_mlp.s320.onnx" for layer in range(12) if layer not in layers],
        *[f"moss_block{layer}_attn_after_cattn.s320.onnx" for layer in layers],
        *[f"moss_block{layer}_ln1_cattn.s320.fp16.rk3576.rknn" for layer in layers],
        *[f"moss_block{layer}_ln2_mlp.s320.fp16.rk3576.rknn" for layer in layers],
    ):
        (tmp_path / name).write_bytes(b"x")

    session = _HybridPrefillSession(
        artifact_dir=tmp_path,
        seq_len=320,
        threads=1,
        ort_module=object(),
        split="ln1_cattn",
        layers=layers,
        rknn_dir=tmp_path,
    )

    assert session._split == "ln1_cattn"
    assert session._rknn_layers == {0, 1, 4}


def test_moss_hybrid_prefill_validates_fc_split_artifacts(monkeypatch, tmp_path):
    monkeypatch.delenv("MOSS_ORT_HYBRID_MANIFEST", raising=False)

    with pytest.raises(FileNotFoundError, match="fc_out.*rk3576"):
        _HybridPrefillSession(
            artifact_dir=tmp_path,
            seq_len=320,
            threads=1,
            ort_module=object(),
            split="fc_out_only",
            layers={0},
            rknn_dir=tmp_path,
        )


def test_moss_ort_runtime_info_reports_profile_and_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSS_ORT_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MOSS_ORT_MANIFEST", "moss-ort-manifest.json")
    monkeypatch.setenv("MOSS_ORT_VOICE", "Junhao")
    monkeypatch.setenv("MOSS_ORT_SEED", "314")
    monkeypatch.setenv("MOSS_ORT_THREADS", "4")
    monkeypatch.setenv("MOSS_ORT_PREFILL_THREADS", "8")
    monkeypatch.setenv("MOSS_ORT_DECODE_THREADS", "5")
    monkeypatch.setenv("MOSS_ORT_SAMPLER_THREADS", "6")
    monkeypatch.setenv("MOSS_ORT_CODEC_THREADS", "4")
    monkeypatch.setenv("MOSS_ORT_CODEC_BATCH_FRAMES", "4")
    monkeypatch.setenv("MOSS_ORT_HYBRID_SPLIT", "mlp_only")
    monkeypatch.setenv("MOSS_ORT_HYBRID_LAYERS", "0,2,5,10")
    backend = MossORTBackend()
    backend._ready = True
    backend._artifact_manifest = {
        "model_id": "moss-tts-nano-onnx",
        "target_platform": "rk3576",
        "streaming_required": True,
        "artifacts": [{"path": "tokenizer.model", "required": True}],
    }
    backend._artifact_manifest_sha256 = "abc123"
    backend._codec_stream_session = object()

    info = backend.runtime_info()

    assert info["profile"]["voice"] == "Junhao"
    assert info["profile"]["seed"] == 314
    assert info["profile"]["threads"] == 4
    assert info["profile"]["session_threads"] == {
        "prefill": 8,
        "decode": 5,
        "sampler": 6,
        "codec": 4,
    }
    assert info["profile"]["codec_streaming"] is True
    assert info["profile"]["codec_full_loaded"] is False
    assert info["profile"]["codec_batch_frames"] == 4
    assert info["hybrid"]["split"] == "mlp_only"
    assert info["hybrid"]["layers"] == [0, 2, 5, 10]
    assert info["manifest"] == {
        "name": "moss-ort-manifest.json",
        "validated": True,
        "sha256": "abc123",
        "model_id": "moss-tts-nano-onnx",
        "target_platform": "rk3576",
        "required_artifacts": 1,
        "streaming_required": True,
    }
    assert info["streaming_stats"] == {
        "requests": 0,
        "completed": 0,
        "errors": 0,
        "active": 0,
        "chunks": 0,
        "last_error": None,
        "last_error_time": None,
    }


def test_moss_ort_streaming_stats_count_success_and_errors(monkeypatch):
    backend = MossORTBackend()
    backend._ready = True

    def _successful_stream(text, rng, max_new_frames):
        yield np.zeros((4, 2), dtype=np.float32), {"chunk_index": 0}
        yield np.zeros((2, 2), dtype=np.float32), {"chunk_index": 1}

    monkeypatch.setattr(backend, "_synthesize_stream_locked", _successful_stream)

    chunks = list(backend.synthesize_stream("你好", seed=1))

    assert len(chunks) == 2
    assert backend.runtime_info()["streaming_stats"] == {
        "requests": 1,
        "completed": 1,
        "errors": 0,
        "active": 0,
        "chunks": 2,
        "last_error": None,
        "last_error_time": None,
    }

    def _failing_stream(text, rng, max_new_frames):
        yield np.zeros((4, 2), dtype=np.float32), {"chunk_index": 0}
        raise RuntimeError("codec failed")

    monkeypatch.setattr(backend, "_synthesize_stream_locked", _failing_stream)

    with pytest.raises(RuntimeError, match="codec failed"):
        list(backend.synthesize_stream("你好", seed=1))

    stats = backend.runtime_info()["streaming_stats"]
    assert stats["requests"] == 2
    assert stats["completed"] == 1
    assert stats["errors"] == 1
    assert stats["active"] == 0
    assert stats["chunks"] == 3
    assert stats["last_error"] == "codec failed"
    assert isinstance(stats["last_error_time"], float)

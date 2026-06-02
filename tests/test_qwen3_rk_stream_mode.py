from rkvoice_stream.backends.asr import qwen3_rk


def test_default_stream_mode_preserves_chunk_confirm(monkeypatch):
    monkeypatch.delenv("QWEN3_ASR_STREAM_MODE", raising=False)
    monkeypatch.delenv("QWEN3_ASR_STREAM_TRUE", raising=False)
    monkeypatch.delenv("QWEN3_ASR_CHUNK_CONFIRM", raising=False)

    assert qwen3_rk._qwen3_stream_mode() == "chunk_confirm"


def test_stream_true_selects_true_streaming_without_chunk_confirm(monkeypatch):
    monkeypatch.delenv("QWEN3_ASR_STREAM_MODE", raising=False)
    monkeypatch.setenv("QWEN3_ASR_STREAM_TRUE", "1")
    monkeypatch.delenv("QWEN3_ASR_CHUNK_CONFIRM", raising=False)

    assert qwen3_rk._qwen3_stream_mode() == "true_streaming"


def test_stream_true_with_chunk_confirm_disabled_selects_true_streaming(monkeypatch):
    monkeypatch.delenv("QWEN3_ASR_STREAM_MODE", raising=False)
    monkeypatch.setenv("QWEN3_ASR_STREAM_TRUE", "1")
    monkeypatch.setenv("QWEN3_ASR_CHUNK_CONFIRM", "0")

    assert qwen3_rk._qwen3_stream_mode() == "true_streaming"


def test_explicit_stream_mode_overrides_legacy_flags(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_STREAM_MODE", "true_streaming")
    monkeypatch.setenv("QWEN3_ASR_STREAM_TRUE", "1")
    monkeypatch.setenv("QWEN3_ASR_CHUNK_CONFIRM", "1")

    assert qwen3_rk._qwen3_stream_mode() == "true_streaming"


def test_explicit_chunk_confirm_mode(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_STREAM_MODE", "chunk_confirm")
    monkeypatch.setenv("QWEN3_ASR_STREAM_TRUE", "1")
    monkeypatch.setenv("QWEN3_ASR_CHUNK_CONFIRM", "0")

    assert qwen3_rk._qwen3_stream_mode() == "chunk_confirm"


def test_rk_stream_immediate_client_eos_cancel_is_opt_in_only():
    stream = qwen3_rk.Qwen3ASRRKStream(object())
    assert stream.immediate_client_eos_cancel_safe is False
    assert stream.prefer_backend_endpoint_vad is False
    assert stream.allow_frontend_eou_finalize is False
    assert stream.frontend_eou_min_audio_s == 0.0

    stream = qwen3_rk.Qwen3ASRRKStream(
        object(),
        immediate_client_eos_cancel_safe=True,
        prefer_backend_endpoint_vad=True,
        allow_frontend_eou_finalize=True,
        frontend_eou_min_audio_s=2.5,
    )
    assert stream.immediate_client_eos_cancel_safe is True
    assert stream.prefer_backend_endpoint_vad is True
    assert stream.allow_frontend_eou_finalize is True
    assert stream.frontend_eou_min_audio_s == 2.5


def test_rk_stream_finalize_returns_asr_stream_tuple_contract():
    class _FakeSession:
        def finish(self):
            return {
                "text": "你好",
                "language": "Chinese",
                "final_mode": "true_streaming",
                "fallback": None,
                "finalize_ms": 12.3,
            }

    stream = qwen3_rk.Qwen3ASRRKStream(_FakeSession())

    assert stream.finalize() == ("你好", "Chinese")


def test_qwen3_backend_endpoint_vad_is_true_streaming_only(monkeypatch):
    backend = qwen3_rk.Qwen3ASRRKBackend()

    monkeypatch.setenv("QWEN3_ASR_STREAM_MODE", "true_streaming")
    assert backend.prefer_backend_endpoint_vad is True
    assert backend.allow_frontend_eou_finalize is True
    assert backend.frontend_eou_min_audio_s == 2.5

    monkeypatch.setenv("QWEN3_ASR_FRONTEND_EOU_MIN_AUDIO_S", "3.25")
    assert backend.frontend_eou_min_audio_s == 3.25

    monkeypatch.setenv("QWEN3_ASR_STREAM_MODE", "chunk_confirm")
    assert backend.prefer_backend_endpoint_vad is False
    assert backend.allow_frontend_eou_finalize is False
    assert backend.frontend_eou_min_audio_s == 0.0


def test_true_streaming_receives_session_vad_endpoint_silence(monkeypatch):
    captured = {}

    class _FakeTrueStreaming:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    from rkvoice_stream.backends.asr.qwen3 import streaming as streaming_mod

    monkeypatch.setenv("QWEN3_ASR_STREAM_MODE", "true_streaming")
    monkeypatch.setattr(
        streaming_mod,
        "Qwen3TrueStreamingASRStream",
        _FakeTrueStreaming,
    )

    backend = qwen3_rk.Qwen3ASRRKBackend()
    backend._ready = True
    backend._engine = object()
    backend._use_npu_lock = False
    monkeypatch.setattr(backend, "_build_vad", lambda **kwargs: None)

    backend.create_stream(
        language="Chinese",
        stream_options={
            "vad_endpoint_silence_ms": 800,
            "vad_min_utterance_s": 1.2,
            "vad_min_audio_s": 2.0,
        },
    )

    assert captured["vad_endpoint_silence_ms"] == 800
    assert captured["vad_min_utterance_s"] == 1.2
    assert captured["vad_min_audio_s"] == 2.0

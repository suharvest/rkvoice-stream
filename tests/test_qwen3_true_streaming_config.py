from rkvoice_stream.backends.asr.qwen3.streaming import Qwen3TrueStreamingASRStream
import logging
import numpy as np
import threading


class _FakeEngine:
    pass


def test_true_streaming_reads_runtime_env_per_stream(monkeypatch):
    monkeypatch.delenv("QWEN3_ASR_TRUE_ROLL_SEC", raising=False)
    monkeypatch.delenv("QWEN3_ASR_TRUE_PARTIAL_TOKENS", raising=False)
    monkeypatch.delenv("QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS", raising=False)
    monkeypatch.delenv("QWEN3_ASR_TRUE_PARTIAL_WARMUP", raising=False)
    monkeypatch.delenv("VAD_ENDPOINT_SILENCE_MS", raising=False)
    monkeypatch.delenv("VAD_MIN_UTTERANCE_S", raising=False)
    monkeypatch.delenv("QWEN3_ASR_VAD_SUSTAIN_FRAMES", raising=False)
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")

    baseline = Qwen3TrueStreamingASRStream(_FakeEngine())

    monkeypatch.setenv("QWEN3_ASR_TRUE_ROLL_SEC", "15")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_TOKENS", "8")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS", "1200")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_WARMUP", "2")
    monkeypatch.setenv("VAD_ENDPOINT_SILENCE_MS", "250")
    monkeypatch.setenv("VAD_MIN_UTTERANCE_S", "0.25")
    monkeypatch.setenv("QWEN3_ASR_VAD_SUSTAIN_FRAMES", "4")

    tuned = Qwen3TrueStreamingASRStream(_FakeEngine())

    assert baseline._max_encoder_frames == 65
    assert tuned._max_encoder_frames == 195
    assert tuned._partial_max_tokens == 8
    assert tuned._partial_min_interval_ms == 1200
    assert tuned._partial_warmup_chunks == 2
    assert tuned._vad_endpoint_silence_ms == 250
    assert tuned._vad_min_utterance_s == 0.25
    assert tuned._vad_sustain_frames == 4


def test_true_streaming_accepts_session_vad_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("VAD_ENDPOINT_SILENCE_MS", "250")
    monkeypatch.setenv("VAD_MIN_UTTERANCE_S", "0.25")
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")

    stream = Qwen3TrueStreamingASRStream(
        _FakeEngine(),
        vad_endpoint_silence_ms=900,
        vad_min_utterance_s=1.5,
        vad_min_audio_s=2.0,
    )

    assert stream._vad_endpoint_silence_ms == 900
    assert stream._vad_min_utterance_s == 1.5
    assert stream._vad_min_audio_s == 2.0


def test_vad_endpoint_async_decode_exposes_endpoint_before_final(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")
    monkeypatch.setenv("QWEN3_ASR_VAD_FINAL_ASYNC", "1")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    release = threading.Event()

    def fake_final_decode():
        stream._final_decode_in_progress = True
        release.wait(timeout=2)
        stream._archive_text = "done"
        stream._partial_text = ""
        stream._episode_final = True
        stream._final_decode_in_progress = False

    stream._do_final_decode_safe = fake_final_decode
    stream._start_vad_final_decode()

    assert stream.get_result()["is_final"] is True
    assert stream._vad_endpoint_detected is True
    assert stream._episode_final is False

    release.set()
    result = stream.finish()

    assert result["text"] == "done"
    assert stream._episode_final is True


def test_vad_endpoint_sync_decode_remains_available(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")
    monkeypatch.setenv("QWEN3_ASR_VAD_FINAL_ASYNC", "0")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    called = []

    def fake_final_decode():
        called.append(True)
        stream._archive_text = "sync"
        stream._episode_final = True

    stream._do_final_decode = fake_final_decode
    stream._start_vad_final_decode()

    assert called == [True]
    assert stream.get_result()["is_final"] is True
    assert stream.finish()["text"] == "sync"


def test_auto_resume_after_endpoint_is_opt_in(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")
    monkeypatch.delenv("QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT", raising=False)
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    stream._episode_final = True
    stream._audio_buf = np.ones(10, dtype=np.float32)

    stream.feed_audio(np.ones(1600, dtype=np.float32) * 0.1)

    assert stream._episode_final is True
    assert len(stream._audio_buf) == 10

    monkeypatch.setenv("QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT", "1")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    stream._episode_final = True
    stream._audio_buf = np.ones(10, dtype=np.float32)

    stream.feed_audio(np.ones(1600, dtype=np.float32) * 0.1)

    assert stream._episode_final is False


def test_final_input_debug_is_runtime_opt_in(monkeypatch, caplog):
    class _Decoder:
        _early_stop_tokens = 0

        def run_embed(self, full_embd, n_tokens, keep_history=0):
            return {
                "text": "debug text",
                "n_tokens_generated": 2,
                "aborted": False,
                "perf": {"prefill_time_ms": 1, "generate_time_ms": 2},
            }

    class _Engine:
        decoder = _Decoder()

        def build_embed(self, all_frames, **kwargs):
            return np.ones((all_frames.shape[0] + 1, 4), dtype=np.float32), 7

    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")
    monkeypatch.delenv("QWEN3_ASR_DEBUG_FINAL_INPUT", raising=False)

    stream = Qwen3TrueStreamingASRStream(_Engine())
    frames = np.arange(8, dtype=np.float32).reshape(2, 4)
    with caplog.at_level(logging.INFO):
        assert stream._decode_final(frames) == "debug text"

    assert "final input debug" not in caplog.text

    caplog.clear()
    monkeypatch.setenv("QWEN3_ASR_DEBUG_FINAL_INPUT", "1")
    stream = Qwen3TrueStreamingASRStream(_Engine())
    stream._vad_endpoint_detected = True

    with caplog.at_level(logging.INFO):
        assert stream._decode_final(frames) == "debug text"

    assert "Qwen3-true-stream final input debug: source=vad" in caplog.text
    assert "input_tokens=7" in caplog.text

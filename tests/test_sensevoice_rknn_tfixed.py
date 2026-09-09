"""Encoder-length resolution and long-audio windowing for the SenseVoice RKNN backend.

The .rknn is frozen to a fixed encoder length, so the backend has to know which
length the file it just opened was built for, and audio longer than one pass
has to be cut into windows instead of truncated.
"""
import numpy as np
import pytest

from rkvoice_stream.backends.asr import sensevoice_rknn as sv


@pytest.mark.parametrize(
    "name,expected",
    [
        ("sense-voice-encoder.rk3588.fp16-scaled.rknn", 344),
        ("sense-voice-encoder.rk3576.fp16.rknn", 344),
        ("sense-voice-encoder.rk3588.fp16-scaled.t172.rknn", 172),
        ("sense-voice-encoder.rk3576.fp16-scaled.t172.rknn", 172),
    ],
)
def test_t_fixed_read_off_the_filename(name, expected, monkeypatch):
    monkeypatch.delenv("SENSEVOICE_RKNN_T_FIXED", raising=False)
    assert sv._resolve_t_fixed("/opt/asr/sensevoice-rknn/" + name) == expected


def test_env_overrides_the_filename(monkeypatch):
    monkeypatch.setenv("SENSEVOICE_RKNN_T_FIXED", "256")
    assert sv._resolve_t_fixed("sense-voice-encoder.rk3588.fp16-scaled.t172.rknn") == 256


@pytest.mark.parametrize("bad", ["abc", "0", "4", "-1"])
def test_unusable_env_falls_back_to_the_filename(bad, monkeypatch):
    monkeypatch.setenv("SENSEVOICE_RKNN_T_FIXED", bad)
    assert sv._resolve_t_fixed("sense-voice-encoder.rk3588.fp16-scaled.t172.rknn") == 172


def _backend(t_fixed: int) -> sv.SenseVoiceRKNNBackend:
    be = sv.SenseVoiceRKNNBackend()
    be._t_fixed = t_fixed
    return be


def _prefix() -> np.ndarray:
    return np.ones((4, sv.LFR_DIM), dtype=np.float32)


def test_short_audio_is_one_padded_window():
    be = _backend(172)
    lfr = np.arange(50 * sv.LFR_DIM, dtype=np.float32).reshape(50, sv.LFR_DIM)
    windows = be._windows(lfr, _prefix())
    assert len(windows) == 1
    speech, valid, skip = windows[0]
    assert speech.shape == (1, 172, sv.LFR_DIM)
    assert valid == 54
    assert skip == 4  # the prompt frames only
    assert np.all(speech[0, valid:] == 0)


def test_audio_longer_than_one_pass_is_windowed_not_truncated():
    be = _backend(172)
    span = 172 - 4
    stride = span - sv._WINDOW_OVERLAP_FRAMES
    lfr = np.arange((span * 2 + 10) * sv.LFR_DIM, dtype=np.float32).reshape(-1, sv.LFR_DIM)
    windows = be._windows(lfr, _prefix())
    assert len(windows) > 1
    # the decoded regions tile the input exactly once, in order and with no gap
    seen = np.vstack([sp[0, skip:v] for sp, v, skip in windows])
    assert np.array_equal(seen, lfr)
    # and every window past the first re-reads the overlap as extra left context
    assert [skip for _, _, skip in windows] == [4] + [4 + sv._WINDOW_OVERLAP_FRAMES] * (
        len(windows) - 1
    )
    assert windows[1][0][0, 4:4 + sv._WINDOW_OVERLAP_FRAMES].tolist() == (
        lfr[stride:stride + sv._WINDOW_OVERLAP_FRAMES].tolist()
    )


def test_window_count_follows_t_fixed():
    lfr = np.zeros((400, sv.LFR_DIM), dtype=np.float32)
    assert len(_backend(344)._windows(lfr, _prefix())) < len(
        _backend(172)._windows(lfr, _prefix())
    )


def test_empty_audio_still_yields_one_window():
    be = _backend(172)
    windows = be._windows(np.zeros((0, sv.LFR_DIM), dtype=np.float32), _prefix())
    assert len(windows) == 1
    assert windows[0][1] == 4
    assert windows[0][2] == 4


def test_overlap_never_exceeds_half_a_window():
    """A tiny encoder must still make forward progress rather than loop."""
    be = _backend(40)
    lfr = np.zeros((500, sv.LFR_DIM), dtype=np.float32)
    windows = be._windows(lfr, _prefix())
    seen = np.vstack([sp[0, skip:v] for sp, v, skip in windows])
    assert np.array_equal(seen, lfr)


@pytest.mark.parametrize(
    "parts,expected",
    [
        (["hello", "world"], "hello world"),
        (["今天天气", "很好"], "今天天气很好"),
        (["hello", ""], "hello"),
        (["", ""], ""),
        (["中文", "english"], "中文english"),
    ],
)
def test_window_join_spaces_only_between_ascii_words(parts, expected):
    assert sv._join_windows(parts) == expected

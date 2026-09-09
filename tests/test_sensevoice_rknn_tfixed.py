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
        ([" hello", " world"], "hello world"),
        (["今天天气", "很好"], "今天天气很好"),
        ([" hello", ""], "hello"),
        (["", ""], ""),
        # a window that begins mid-word must not gain a space it never had
        ([" won", "derful"], "wonderful"),
    ],
)
def test_window_join_never_invents_a_separator(parts, expected):
    assert sv._join_windows(parts) == expected


@pytest.mark.parametrize("name", ["m.t4.rknn", "m.t0.rknn", "m.t1.rknn"])
def test_filename_t_below_the_prompt_frames_is_rejected(name, monkeypatch):
    monkeypatch.delenv("SENSEVOICE_RKNN_T_FIXED", raising=False)
    assert sv._resolve_t_fixed(name) == sv.T_FIXED_DEFAULT


class _FakeSP:
    def id_to_piece(self, i):
        return {1: "\u2581a", 2: "\u2581b", 3: "c"}[i]

    def get_piece_size(self):
        return 8


def _decode(rows, valid, skip):
    be = sv.SenseVoiceRKNNBackend()
    be._sp = _FakeSP()
    logits = np.zeros((len(rows), 8), dtype=np.float32)
    for i, tok in enumerate(rows):
        logits[i, tok] = 1.0
    return sv.SenseVoiceRKNNBackend._ctc_decode(be, logits, valid, skip)


P = sv._N_PROMPT_FRAMES  # prompt frames every window carries before the audio


def test_repeat_held_across_the_cut_is_emitted_once():
    """The overlap must not re-emit a token the previous window already did."""
    rows = [0] * P + [1, 1, 1, 1, 2]     # 'a' held over the cut, then 'b'
    whole = _decode(rows, len(rows), P)
    first = _decode(rows[:P + 3], P + 3, P)          # window 1 emits frames 0-2
    second = _decode(rows, len(rows), P + 3)         # window 2 re-reads them
    assert sv._join_windows([first, second]) == whole.strip()


def test_word_split_across_the_cut_is_not_broken_in_two():
    rows = [0] * P + [1, 3]              # '_a' + 'c' -> one word 'ac'
    whole = _decode(rows, len(rows), P)
    assert whole.strip() == "ac"
    first = _decode(rows[:P + 1], P + 1, P)
    second = _decode(rows, len(rows), P + 1)
    assert sv._join_windows([first, second]) == "ac"


def test_prompt_frames_never_suppress_the_first_real_token():
    """A token equal to a prompt frame's argmax must still open the transcript."""
    rows = [1] * P + [1, 2]
    assert _decode(rows, len(rows), P).strip() == "a b"


def test_a_directory_holding_both_builds_picks_the_tagged_one(tmp_path, monkeypatch):
    monkeypatch.delenv("SENSEVOICE_RKNN_MODEL", raising=False)
    monkeypatch.setenv("RK_PLATFORM", "rk3588")
    for n in ("sense-voice-encoder.rk3588.fp16-scaled.rknn",
              "sense-voice-encoder.rk3588.fp16-scaled.t172.rknn"):
        (tmp_path / n).write_bytes(b"x")
    be = sv.SenseVoiceRKNNBackend()
    assert be._resolve_model_path(str(tmp_path)).endswith(".t172.rknn")

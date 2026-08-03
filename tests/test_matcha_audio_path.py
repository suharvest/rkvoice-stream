"""Regression tests for the matcha TTS audio path.

These cover the defect chain found on radxa (RK3588) in 2026-08: long
utterances came out 25-45 dB too quiet and lost up to 43% of their content.
Three independent bugs stacked up:

  1. ``_istft`` divided by ``np.maximum(window_sum, 1e-8)``.  The outermost
     N_FFT-HOP samples are covered by fewer than the full set of windows, so
     window_sum tapers to ~0 there and the division exploded whatever the
     vocoder emitted into a transient up to 300x the speech peak.
  2. ``run_vocos`` dropped every mel frame past the vocoder's compiled window.
     The shipped "vocos-16khz-600.rknn" is in fact a 256-frame build, so a
     447-frame Chinese sentence rendered only 4.14 s of its 7.15 s.
  3. ``synthesize`` peak-normalized, so the transient from (1) decided the gain
     for the whole utterance and ducked it by ~29 dB.

Plus two text-frontend bugs that made English disproportionately likely to trip
(1) and (2): the sentence splitter did not recognise ASCII '.', and the RKNN
bucket width was imposed on the dynamic-shape ORT path.

Everything here is pure numpy — no NPU, no model files.
"""

from __future__ import annotations

import numpy as np
import pytest

from rkvoice_stream.backends.tts import matcha as m

Cls = m.RKNNMatchaVocoder
N_FFT, HOP, SR = m.N_FFT, m.HOP_LENGTH, m.SAMPLE_RATE


@pytest.fixture
def inst():
    """A backend instance with nothing loaded — we only exercise pure logic."""
    return object.__new__(Cls)


def _stft(sig: np.ndarray, n_frames: int):
    window = np.hanning(N_FFT)
    frames = [
        np.fft.rfft(sig[i * HOP: i * HOP + N_FFT] * window, n=N_FFT)
        for i in range(n_frames)
    ]
    S = np.stack(frames, axis=1)
    ang = np.angle(S)
    return np.abs(S), np.cos(ang), np.sin(ang)


def _tone(n_frames: int, freq: float = 220.0):
    length = (n_frames - 1) * HOP + N_FFT
    t = np.arange(length) / SR
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), length


# --------------------------------------------------------------------- ISTFT


def test_istft_reconstructs_the_interior(inst):
    n_frames = 128
    sig, length = _tone(n_frames)
    audio = inst._istft(*_stft(sig, n_frames))

    assert len(audio) == length
    body = slice(N_FFT, -N_FFT)
    assert np.abs(audio[body] - sig[body]).max() < 1e-3


def test_istft_does_not_explode_at_the_edges(inst):
    """A vocoder *generates* spectra; it is not an exact analysis of a signal
    defined over the full support, so the edge frames carry independent
    residue.  Perturbing the magnitudes reproduces that condition — the old
    ``np.maximum(window_sum, 1e-8)`` divided that residue by ~0."""
    n_frames = 256
    sig, length = _tone(n_frames)
    mag, cos, sin = _stft(sig, n_frames)
    mag = mag * (1.0 + 0.01 * np.random.default_rng(1).standard_normal(mag.shape))

    audio = inst._istft(mag, cos, sin)

    # The peak must be real speech, not an edge artifact.
    assert np.abs(audio).max() < 1.0, "edge transient survived"
    ratio = float(np.percentile(np.abs(audio), 99.9) / np.abs(audio).max())
    assert ratio > 0.25, f"peak is an outlier (p99.9/peak={ratio:.4f})"

    # The taper regions are still divided by a small window_sum, so assert the
    # amplification there stays negligible rather than asserting where the peak
    # lands — for a steady tone every sample has the same amplitude, so argmax
    # is arbitrary.  Measured headroom at the current floor is ~2%.
    edge = N_FFT - HOP
    body_peak = float(np.abs(audio[edge:length - edge]).max())
    for name, region in (("head", audio[:edge]), ("tail", audio[length - edge:])):
        assert float(np.abs(region).max()) < body_peak * 1.1, f"{name} taper amplified"

    # Old behaviour, for contrast: the same input blows up.
    old = np.zeros(length, dtype=np.float32)
    window = np.hanning(N_FFT)
    for i in range(n_frames):
        frame = np.fft.irfft(mag[:, i] * (cos[:, i] + 1j * sin[:, i]), n=N_FFT)
        old[i * HOP: i * HOP + N_FFT] += frame * window
    ws = np.zeros(length, dtype=np.float32)
    for i in range(n_frames):
        ws[i * HOP: i * HOP + N_FFT] += window ** 2
    old = old / np.maximum(ws, 1e-8)
    assert np.abs(old).max() > 10 * np.abs(audio).max(), "regression witness is stale"


def test_istft_zeroes_only_a_sliver_of_each_edge(inst):
    """Zeroing the degenerate edge must not eat into real audio: it is a fixed
    ~7 ms per side regardless of utterance length."""
    for n_frames in (64, 256, 447):
        sig, length = _tone(n_frames)
        audio = inst._istft(*_stft(sig, n_frames))
        zeroed = int(np.sum(audio == 0.0))
        assert zeroed < 300, f"{zeroed} samples zeroed at {n_frames} frames"
        assert zeroed / length < 0.02


# ------------------------------------------------------------ vocos chunking


class _FakeVocos:
    """The deployed static-shape vocoder: always emits ``cap`` frames, and only
    ever reads the first ``cap`` frames of the buffer it is handed — the silent
    misread rknn-lite performs on an oversized input."""

    def __init__(self, cap: int = 256):
        self.cap = cap
        self.calls = 0

    def inference(self, inputs):
        self.calls += 1
        src = inputs[0][0, :, : self.cap]
        mag = np.tile(src, (7, 1))[:513, :]
        return [mag[None], np.ones_like(mag)[None], np.zeros_like(mag)[None]]


@pytest.fixture
def voc(inst, monkeypatch):
    monkeypatch.setattr(m, "_npu_lock", lambda: None)
    inst._vocos = _FakeVocos()
    inst._vocos_frames = inst._vocos.cap
    return inst


@pytest.mark.parametrize("total", [200, 256, 257, 447, 1000])
def test_vocos_renders_every_frame(voc, total):
    mel = np.random.default_rng(7).standard_normal((1, 80, total)).astype(np.float32)

    mag, cos, sin = voc._run_vocos_frames(mel, total)

    assert mag.shape[1] == total
    assert cos.shape[1] == total and sin.shape[1] == total
    # Chunk boundaries must not shift or drop content.
    expected = np.tile(mel[0], (7, 1))[:513, :]
    assert np.allclose(mag, expected, atol=1e-5)


def test_vocos_uses_a_single_call_when_it_fits(voc):
    mel = np.zeros((1, 80, 200), dtype=np.float32)
    voc._run_vocos_frames(mel, 200)
    assert voc._vocos.calls == 1


def test_vocos_chunks_when_it_does_not_fit(voc):
    mel = np.zeros((1, 80, 447), dtype=np.float32)
    voc._run_vocos_frames(mel, 447)
    assert voc._vocos.calls > 1


def test_run_vocos_passes_the_old_ceiling(voc):
    """447 frames used to render as 256 (4.14 s); it must now be 7.15 s."""
    total = 447
    mel = np.random.default_rng(3).standard_normal((1, 80, total)).astype(np.float32)

    audio = voc.run_vocos(mel, total)

    assert len(audio) == total * HOP
    old_ceiling = (voc._vocos.cap - 1) * HOP + N_FFT
    assert len(audio) > old_ceiling


def test_run_vocos_handles_empty_input(voc):
    assert len(voc.run_vocos(np.zeros((1, 80, 0), dtype=np.float32), 0)) == 0


# -------------------------------------------------------------- text splitting


def _fake_tokens(text: str):
    """Stand-in for the lexicon/espeak frontend; only counts matter here."""
    import re

    n = len(re.findall(r"[一-鿿]", text))
    n += 3 * len(re.findall(r"[A-Za-z]+", text))
    return list(range(n))


@pytest.fixture
def splitter(inst):
    inst.text_to_tokens = _fake_tokens
    inst._matcha_backend = "ort"
    return inst


def test_ort_is_not_held_to_the_rknn_bucket(splitter):
    """The ORT path takes dynamic shapes; capping it at the RKNN bucket width
    silently dropped tokens off the end of long segments."""
    assert splitter._token_budget() > m.MAX_SEQ_LEN

    splitter._matcha_backend = "rknn"
    assert splitter._token_budget() == m.MAX_SEQ_LEN


def test_english_is_split_on_ascii_period(splitter):
    text = ("Sure, I have turned on the living room light for you. Would you "
            "also like me to adjust the bedroom air conditioner?")

    segs = splitter._split_text(text)

    assert len(segs) > 1
    assert any(s.rstrip().endswith(".") for s in segs)


def test_decimals_are_not_split(splitter):
    assert splitter._split_text("Set it to 26.5 degrees now.") == [
        "Set it to 26.5 degrees now."
    ]


def test_chinese_still_splits_on_full_width_punctuation(splitter):
    segs = splitter._split_text("今天天气不错。请把客厅的灯打开。")
    assert len(segs) == 2


def test_unpunctuated_run_is_split_without_losing_words(splitter):
    text = " ".join(["please turn on the living room light and the kitchen light"] * 8)

    segs = splitter._split_text(text)
    budget = splitter._token_budget()

    assert all(len(_fake_tokens(s)) <= budget for s in segs)
    assert sorted(" ".join(segs).split()) == sorted(text.split())


# -------------------------------------------------------------- tokens.txt


def test_tokens_parser_keeps_the_space_token(tmp_path):
    """Line 1 of the real tokens.txt is ``"  1"`` -- the token IS a space.

    sherpa-onnx requires it (``token2id_.at(" ")``,
    matcha-tts-lexicon.cc:265) and inserts it at English word boundaries.  The
    old ``line.strip().split()`` parse dropped it and registered a phantom
    token ``"1"`` instead, leaving English with no word boundaries at all.
    """
    p = tmp_path / "tokens.txt"
    p.write_text("  1\n; 2\n, 4\nðə 65\n", encoding="utf-8")

    table = m.parse_tokens_file(str(p))

    assert table[" "] == 1, "word-boundary token lost"
    assert "1" not in table, "phantom token from the id column"
    assert table[";"] == 2
    assert table[","] == 4


def test_tokens_parser_trusts_the_id_column(tmp_path):
    """Ids come from the file, not from line order."""
    p = tmp_path / "tokens.txt"
    p.write_text("  1\nzuo4 2174\ng 2175\n", encoding="utf-8")

    table = m.parse_tokens_file(str(p))

    assert table["zuo4"] == 2174
    assert table["g"] == 2175


# ------------------------------------------------------------- normalization


def _normalize(audio: np.ndarray) -> np.ndarray:
    """Apply utterance_gain the way synthesize() does."""
    gain, clip = m.utterance_gain(audio)
    out = audio * gain
    return np.clip(out, -1.0, 1.0) if clip else out


def _speech(n: int = 64000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.2 * np.sin(2 * np.pi * 180 * np.arange(n) / SR)
            + 0.02 * rng.standard_normal(n)).astype(np.float32)


def test_healthy_audio_is_peak_normalized():
    assert np.abs(_normalize(_speech())).max() == pytest.approx(0.95, abs=1e-5)


def test_a_transient_does_not_duck_the_utterance():
    speech = _speech()
    spiked = speech.copy()
    spiked[-3:] = 8.0  # the ISTFT blowup

    def rms_db(x):
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)))

    healthy = rms_db(_normalize(speech))
    rescued = rms_db(_normalize(spiked)[:-3])
    old = rms_db((spiked / np.abs(spiked).max() * 0.95)[:-3])

    assert abs(rescued - healthy) < 2.0, "rescue path changed the level"
    assert healthy - old > 15, "regression witness is stale"


def test_gain_is_a_factor_so_streaming_can_reuse_it():
    """synthesize_stream() bypasses synthesize(), so the gain must be reusable.

    The v2v conversation loop is the streaming path.  If the gain were only
    applied inside synthesize(), streaming would come out quieter than /tts
    for the same text.  Returning a factor is what lets both agree -- and
    fixing one factor for the whole utterance is what keeps the level from
    pumping between sentences of a single reply.
    """
    quiet = _speech() * 0.1
    loud = _speech() * 0.8

    g_quiet, _ = m.utterance_gain(quiet)
    g_loud, _ = m.utterance_gain(loud)

    # Normalizing each segment on its own would flatten both to 0.95.
    assert g_quiet > g_loud
    before = float(np.abs(quiet).max() / np.abs(loud).max())
    after = float(np.abs(quiet * g_loud).max() / np.abs(loud * g_loud).max())
    assert after == pytest.approx(before, rel=1e-6)


def test_empty_and_silent_audio_get_a_neutral_gain():
    assert m.utterance_gain(np.zeros(0, dtype=np.float32)) == (1.0, False)
    assert m.utterance_gain(np.zeros(100, dtype=np.float32)) == (1.0, False)

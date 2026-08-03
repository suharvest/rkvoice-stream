"""Long-form TTS acceptance: the cases the short-sentence suite cannot see.

In 2026-08 a customer reported RK3588 replies coming back too quiet and
unintelligible.  Three bugs stacked (ISTFT edge transient hijacking peak
normalization, and the vocoder silently dropping every mel frame past its
compiled window), yet the whole test suite was green.  Two reasons:

  * every sentence in TEST_SENTENCES_* fits inside a single vocoder window,
    so nothing ever reached the ceiling; and
  * the round-trip gate is CER < 0.5, but dropping 43% of a reply yields a
    CER around 0.43 -- it would have passed anyway.

So these tests deliberately use utterances longer than one vocoder window and
assert on signals that truncation and ducking cannot slip past:

  * the tail of the reply must actually be spoken (direct truncation probe);
  * duration must keep scaling with text length rather than pinning at a
    ceiling;
  * the waveform peak must be speech, not a lone edge transient.

Run against a device with:  SERVICE_URL=http://<host>:8621 pytest tests/test_tts_long_form.py -v

IMPORTANT -- the round-trip tests below are only meaningful when the ASR side
is configured for transcription, not for live turn-taking.  A service running
with ASR_FINAL_STOP_ON_PUNCT=1 (plus ASR_FINAL_STOP_MIN_CHARS /
ASR_FINAL_STOP_MIN_CHUNKS) stops decoding at the first sentence-final
punctuation, so every multi-clause reference will look truncated no matter how
good the audio is.  That configuration produced two false failures the first
time this file was run.  Cross-check any round-trip failure against an
independent recognizer before blaming TTS -- the ASR-free tests in this file
(duration, edge transient, voiced share) are the ones that hold regardless.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from .conftest import (
    TEST_SENTENCES_LONG_EN,
    TEST_SENTENCES_LONG_ZH,
)
from .metrics import cer, normalize_text

# Long-form CER budget.  The legacy gate (0.5) is useless here: it sits above
# the error a badly truncated reply produces.  0.30 still tolerates ordinary
# ASR noise on synthetic speech while failing anything that lost a clause.
LONG_FORM_CER_MAX = 0.30


def _samples(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
        width = w.getsampwidth()
    assert width == 2, f"expected PCM16, got {width * 8}-bit"
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr


def _duration(wav_bytes: bytes) -> float:
    x, sr = _samples(wav_bytes)
    return len(x) / sr


def _is_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _lang_of(text: str) -> str:
    return "Chinese" if _is_cjk(text) else "English"


ALL_LONG = TEST_SENTENCES_LONG_ZH + TEST_SENTENCES_LONG_EN


# --------------------------------------------------------------- truncation


@pytest.mark.parametrize("sentence", ALL_LONG)
def test_long_form_tail_is_spoken(tts_fn, asr_fn, sentence):
    """The end of the reply must survive.

    This is the direct probe for the vocoder-truncation class of bug: content
    is dropped from the tail, so the last words simply never get spoken.  A
    whole-utterance CER can absorb that; checking the tail cannot.

    False-positive to rule out first: an ASR configured to stop at
    sentence-final punctuation (ASR_FINAL_STOP_ON_PUNCT=1) truncates the
    transcript on its own.  See the module docstring.
    """
    wav_bytes, _meta = tts_fn(sentence)
    hyp = normalize_text(asr_fn(wav_bytes, language=_lang_of(sentence)))
    ref = normalize_text(sentence)

    if _is_cjk(sentence):
        # Last few characters, ignoring punctuation that ASR may not emit.
        tail = ref[-4:]
        overlap = sum(1 for c in tail if c in hyp)
        assert overlap >= 2, (
            f"tail {tail!r} missing from transcript -- reply was cut off\n"
            f"  ref={ref!r}\n  hyp={hyp!r}"
        )
    else:
        # Take the tail words from the RAW sentence: normalize_text strips
        # whitespace too, so splitting the normalized string yields one giant
        # token and turns this into an exact-match gate rather than a tail
        # check.  `hyp` is space-free, but substring membership still holds.
        tail_words = re.findall(r"[a-z']+", sentence.lower())[-2:]
        hit = sum(1 for wd in tail_words if wd in hyp)
        assert hit >= 1, (
            f"tail {tail_words!r} missing from transcript -- reply was cut off\n"
            f"  ref={ref!r}\n  hyp={hyp!r}"
        )


@pytest.mark.parametrize(
    "clause",
    [
        "今天天气不错我想请你把客厅的灯打开",
        "please turn on the living room light and then the kitchen light",
    ],
    ids=["zh", "en"],
)
def test_duration_keeps_scaling_past_one_window(tts_fn, clause):
    """Doubling a single unpunctuated clause must roughly double the audio.

    A fixed-capacity vocoder pins long output to a constant ceiling, so the
    ratio collapses towards 1.0.  Using one clause with no sentence or comma
    punctuation keeps the text splitter from rescuing the measurement, and
    comparing against itself means the test needs no knowledge of the
    compiled window size.
    """
    single = _duration(tts_fn(clause)[0])
    double = _duration(tts_fn(clause + " " + clause)[0])

    assert single > 0, "no audio produced"
    ratio = double / single
    assert ratio > 1.6, (
        f"duration pinned at a ceiling: {single:.3f}s -> {double:.3f}s "
        f"(ratio {ratio:.2f}, expected ~2.0)"
    )


# ------------------------------------------------------------- edge artifact


@pytest.mark.parametrize("sentence", ALL_LONG)
def test_peak_is_speech_not_an_edge_transient(tts_fn, sentence):
    """Guards the ISTFT-edge bug.

    A blown-up edge sample makes the peak a lone outlier, which then drives
    peak normalization and ducks the whole utterance by tens of dB.  Two
    fingerprints: p99.9 collapses relative to the peak (measured 0.003-0.012
    when broken, 0.75-0.92 when healthy), and the argmax sits in the first or
    last frame.
    """
    x, sr = _samples(tts_fn(sentence)[0])
    assert len(x) > 0, "no audio produced"

    peak = float(np.abs(x).max())
    assert peak > 0, "silent output"
    ratio = float(np.percentile(np.abs(x), 99.9) / peak)
    assert ratio > 0.5, (
        f"peak is an outlier (p99.9/peak={ratio:.4f}) -- edge transient"
    )

    edge = 1024  # one FFT window
    argmax = int(np.argmax(np.abs(x)))
    assert edge <= argmax < len(x) - edge, (
        f"peak at sample {argmax} of {len(x)} sits in an edge"
    )


@pytest.mark.parametrize("sentence", ALL_LONG)
def test_long_form_is_not_mostly_silence(tts_fn, sentence):
    """A ducked utterance reads as near-silence with one spike.

    Voiced share collapsed to 0.01-0.02% when broken; healthy output sits
    around 30-42%.
    """
    x, _sr = _samples(tts_fn(sentence)[0])
    peak = float(np.abs(x).max())
    voiced = float(np.mean(np.abs(x) > 0.1 * peak))
    assert voiced > 0.15, f"only {voiced:.2%} of samples are voiced"


# ------------------------------------------------------------- round-trip


@pytest.mark.parametrize("sentence", ALL_LONG)
def test_long_form_roundtrip_cer(tts_fn, asr_fn, sentence):
    """Long-form intelligibility, on a budget tight enough to matter."""
    wav_bytes, _meta = tts_fn(sentence)
    hyp = asr_fn(wav_bytes, language=_lang_of(sentence))
    value = cer(sentence, hyp)
    assert value < LONG_FORM_CER_MAX, (
        f"CER {value:.1%} >= {LONG_FORM_CER_MAX:.0%}\n"
        f"  ref={sentence!r}\n  hyp={hyp!r}"
    )

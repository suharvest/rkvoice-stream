"""Piper frontend: clause terminators reach the model, segments get a pause.

espeak-ng is mocked with the line-per-clause shape its CLI really prints
(measured on the speech image, espeak-ng 1.51), so these run without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from rkvoice_stream.backends.tts import piper

# Minimal phoneme_id_map in the shape Piper voices ship.
ID_MAP = {
    "_": [0], "^": [1], "$": [2], " ": [3],
    "!": [4], ",": [8], ".": [10], ":": [11], ";": [12], "?": [13],
    "a": [14], "b": [15], "c": [16], "d": [17],
}


def _fake_espeak(monkeypatch, table: dict[str, str], calls: list[str] | None = None):
    """Each input line comes back as one output line, like the CLI."""
    def run(text: str, voice: str) -> str:
        if calls is not None:
            calls.append(text)
        return "\n".join(table[ln] for ln in text.split("\n"))
    monkeypatch.setattr(piper, "_HAS_PIPER_PHONEMIZE", False)
    monkeypatch.setattr(piper, "_run_espeak", run)


def _strip_pad(ids: list[int]) -> list[int]:
    return [i for i in ids if i != 0]


# -- clause splitting -------------------------------------------------------

def test_split_keeps_terminators():
    assert piper._split_clauses("Hello, world. OK?") == [
        ("Hello", ","), ("world", "."), ("OK", "?"),
    ]


@pytest.mark.parametrize("text", ["It costs 1,000 dollars", "Meet at 10:30 today", "Pi is 3.14 roughly"])
def test_split_leaves_numbers_whole(text):
    assert piper._split_clauses(text) == [(text, "")]


def test_split_number_before_real_terminator():
    assert piper._split_clauses("It costs 1,000, right?") == [
        ("It costs 1,000", ","), ("right", "?"),
    ]


def test_split_fullwidth_and_runs():
    assert piper._split_clauses("你好，世界。") == [("你好", ","), ("世界", ".")]
    assert piper._split_clauses("Wait... what?!") == [("Wait", "."), ("what", "?")]


def test_split_terminator_before_closing_quote():
    assert piper._split_clauses('He said "stop." Then left') == [
        ('He said "stop', "."), ('" Then left', ""),
    ]


def test_split_punctuation_only_is_empty():
    assert piper._split_clauses("...") == []


# -- phonemization ----------------------------------------------------------

def test_terminators_reattached_in_one_call(monkeypatch):
    calls: list[str] = []
    _fake_espeak(monkeypatch, {"Hello": "ab", "world": "cd"}, calls)
    assert piper.text_to_phonemes("Hello, world.", "en-us") == "ab, cd."
    assert calls == ["Hello\nworld"]


def test_falls_back_per_clause_when_lines_do_not_pair(monkeypatch):
    # espeak broke the first clause in two: 3 lines for 2 clauses.
    def run(text: str, voice: str) -> str:
        return {"Hello there\nworld": "ab\nba\ncd", "Hello there": "ab\nba", "world": "cd"}[text]
    monkeypatch.setattr(piper, "_HAS_PIPER_PHONEMIZE", False)
    monkeypatch.setattr(piper, "_run_espeak", run)
    assert piper.text_to_phonemes("Hello there, world.", "en-us") == "ab ba, cd."


def test_native_phonemizer_keeps_spaces_and_terminators(monkeypatch):
    monkeypatch.setattr(piper, "_HAS_PIPER_PHONEMIZE", True)
    monkeypatch.setattr(
        piper, "_phonemize_espeak_lib",
        lambda text, voice: [["a", "b", ",", " ", "c", "d", "."]],
        raising=False,
    )
    assert piper.text_to_phonemes("x", "en-us") == "ab, cd."


# -- id sequence ------------------------------------------------------------

def test_ids_match_reference_layout():
    # ^ a b , ␣ c d . $   -- terminator rides on its word, no ␣ before $.
    assert _strip_pad(piper.phonemes_to_ids("ab, cd.", ID_MAP)) == [
        1, 14, 15, 8, 3, 16, 17, 10, 2,
    ]


def test_ids_interleave_pad():
    ids = piper.phonemes_to_ids("ab.", ID_MAP)
    assert ids == [0, 1, 0, 14, 0, 15, 0, 10, 0, 2, 0]


def test_ids_without_punctuation_have_no_trailing_separator():
    assert _strip_pad(piper.phonemes_to_ids("ab cd", ID_MAP)) == [1, 14, 15, 3, 16, 17, 2]


# -- segment pause ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Hello.", 300.0), ("Really?", 300.0), ('He said "stop."', 300.0), ("你好。", 300.0),
    ("Hello,", 150.0), ("First;", 150.0),
    ("Hello", 0.0), ("", 0.0),
])
def test_segment_pause_defaults(monkeypatch, text, expected):
    monkeypatch.delenv("PIPER_SENTENCE_PAUSE_MS", raising=False)
    monkeypatch.delenv("PIPER_CLAUSE_PAUSE_MS", raising=False)
    assert piper._segment_pause_ms(text) == expected


def test_segment_pause_env_read_per_call(monkeypatch):
    monkeypatch.setenv("PIPER_SENTENCE_PAUSE_MS", "500")
    assert piper._segment_pause_ms("Hi.") == 500.0
    monkeypatch.setenv("PIPER_SENTENCE_PAUSE_MS", "0")
    assert piper._segment_pause_ms("Hi.") == 0.0


@pytest.mark.parametrize("bad", ["abc", "nan", "inf", "-5"])
def test_segment_pause_rejects_bad_env(monkeypatch, bad):
    monkeypatch.setenv("PIPER_SENTENCE_PAUSE_MS", bad)
    assert piper._segment_pause_ms("Hi.") == 300.0


class _FakeModel:
    espeak_voice = "en-us"
    phoneme_id_map = ID_MAP
    sample_rate = 22050
    seq_len = 256
    length_scale = 1.0
    noise_scale = 0.667
    noise_w = 0.8

    def __init__(self):
        self.seen: list[list[int]] = []

    def infer(self, token_ids, length_scale, noise_scale, noise_w):
        self.seen.append(list(token_ids))
        return np.full(22050, 0.5, dtype=np.float32)


def test_segment_gets_trailing_pause_and_punctuated_ids(monkeypatch):
    _fake_espeak(monkeypatch, {"Hello": "ab", "world": "cd"})
    monkeypatch.delenv("PIPER_SENTENCE_PAUSE_MS", raising=False)
    model = _FakeModel()
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)

    audio, _ = backend._synthesize_segment("Hello, world.", model, speed=1.0)
    speech = (22050 // piper.SILENCE_FRAME_SIZE) * piper.SILENCE_FRAME_SIZE
    assert len(audio) == speech + int(22050 * 0.3)
    assert not audio[speech:].any()
    assert _strip_pad(model.seen[0]) == [1, 14, 15, 8, 3, 16, 17, 10, 2]

    # Faster speech, shorter pause.
    audio2, _ = backend._synthesize_segment("Hello, world.", model, speed=2.0)
    assert len(audio2) == speech + int(22050 * 0.15)

    # No final mark, no pause.
    _fake_espeak(monkeypatch, {"Hello": "ab"})
    audio3, _ = backend._synthesize_segment("Hello", model, speed=1.0)
    assert len(audio3) == speech


# -- review follow-ups: one boundary rule for sentences and clauses ---------

@pytest.mark.parametrize("text", [
    "Pi is 3.14 roughly", "Visit example.com today", "Ask Dr. Smith about it",
    "The U.S. market grew", "J. K. Rowling wrote it", "Use e.g. a fan",
    "We left at 5 p.m. sharp",
])
def test_inner_periods_split_neither_sentences_nor_clauses(text):
    assert piper._split_sentences(text) == [text]
    assert piper._split_clauses(text) == [(text, "")]


def test_sentences_split_only_at_real_ends():
    assert piper._split_sentences(
        "Pi is 3.14. Dr. Smith agrees! See example.com; then call."
    ) == ["Pi is 3.14.", "Dr. Smith agrees!", "See example.com;", "then call."]


def test_sentence_keeps_its_closing_quote():
    assert piper._split_sentences('He said "stop." Then he left.') == [
        'He said "stop."', "Then he left.",
    ]
    assert piper._segment_pause_ms('He said "stop."') == 300.0


def test_sentences_fullwidth_and_newlines():
    assert piper._split_sentences("你好。世界！\n再见") == ["你好。", "世界！", "再见"]


def test_abbreviation_at_the_very_end_still_ends_the_clause_list():
    # No following sentence to protect; the mark is simply not re-attached.
    assert piper._split_clauses("Ask the Dr.") == [("Ask the Dr.", "")]


def test_truncation_keeps_eos(monkeypatch):
    _fake_espeak(monkeypatch, {"Hello": "ab" * 40})
    model = _FakeModel()
    model.seq_len = 24
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._synthesize_segment("Hello.", model, speed=1.0)
    ids = model.seen[0]
    assert len(ids) == 24
    assert ids[-2:] == [2, 0]          # EOS + pad survive the cut


@pytest.mark.parametrize("text,expected", [
    ("Plan B. Then we go.", ["Plan B.", "Then we go."]),
    ("I got an A. Great.", ["I got an A.", "Great."]),
    ("It has vitamin C. It helps.", ["It has vitamin C.", "It helps."]),
    ("We sell in the U.S. They ship today.", ["We sell in the U.S.", "They ship today."]),
    ("We left at 5 p.m. We were late.", ["We left at 5 p.m.", "We were late."]),
])
def test_single_letters_and_chains_still_end_sentences(text, expected):
    assert piper._split_sentences(text) == expected
    # Clauses agree: the same marks end a clause.
    assert [c + p for c, p in piper._split_clauses(text)] == expected


def test_fullwidth_closer_does_not_hide_the_final_mark():
    assert piper._segment_pause_ms("他说「停。」") == 300.0

"""Piper frontend: clause terminators reach the model, segments get a pause.

espeak-ng is mocked with the line-per-clause shape its CLI really prints
(measured on the speech image, espeak-ng 1.51), so these run without it.
"""

from __future__ import annotations

import io
import json
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from rkvoice_stream.backends.tts import piper

# Minimal phoneme_id_map in the shape Piper voices ship.
ID_MAP = {
    "_": [0], "^": [1], "$": [2], " ": [3],
    "!": [4], ",": [8], ".": [10], ":": [11], ";": [12], "?": [13],
    "a": [14], "b": [15], "c": [16], "d": [17],
}


def _fake_espeak(monkeypatch, table: dict[str, str], calls: list[str] | None = None):
    """Each input line comes back as one output line, like the CLI."""
    def run(text: str, voice: str, cache=None) -> str:
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
    def run(text: str, voice: str, cache=None) -> str:
        return {"Hello there\nworld": "ab\nba\ncd", "Hello there": "ab\nba", "world": "cd"}[text]
    monkeypatch.setattr(piper, "_HAS_PIPER_PHONEMIZE", False)
    monkeypatch.setattr(piper, "_run_espeak", run)
    assert piper.text_to_phonemes("Hello there, world.", "en-us") == "ab ba, cd."


def test_subprocess_cache_reuses_only_successful_nonempty_calls(monkeypatch):
    calls = []
    responses = {"same": "ipa", "empty": "", "bad": "bad"}
    returns = {"same": 0, "empty": 0, "bad": 1}

    def run(text: str, voice: str, cache=None) -> str:
        calls.append((text, voice))
        return responses[text]

    # Exercise the cache contract through the low-level helper while keeping
    # subprocess.run out of the unit test.
    monkeypatch.setattr(piper.subprocess, "run", lambda *args, **kwargs: type(
        "Result", (), {"stdout": responses[args[0][-1]], "returncode": returns[args[0][-1]], "stderr": ""}
    )())
    cache = {}
    assert piper._run_espeak("same", "en-us", cache=cache) == "ipa"
    assert piper._run_espeak("same", "en-us", cache=cache) == "ipa"
    assert piper._run_espeak("empty", "en-us", cache=cache) == ""
    assert piper._run_espeak("empty", "en-us", cache=cache) == ""
    assert piper._run_espeak("bad", "en-us", cache=cache) == "bad"
    assert piper._run_espeak("bad", "en-us", cache=cache) == "bad"
    assert list(cache) == [("en-us", "same")]


def test_request_cache_reuses_espeak_calls_across_token_probes(monkeypatch):
    calls = []

    def run(text: str, voice: str, cache=None) -> str:
        calls.append(text)
        return "ipa"

    monkeypatch.setattr(piper, "_run_espeak", run)
    monkeypatch.setattr(piper, "_HAS_PIPER_PHONEMIZE", False)
    model = types.SimpleNamespace(espeak_voice="en-us", phoneme_id_map=ID_MAP)
    cache = {}
    piper._frontend_tokens("same, clause.", model, cache)
    piper._frontend_tokens("different", model, cache)
    piper._frontend_tokens("same, clause.", model, cache)
    assert calls.count("same\nclause") == 1
    assert calls.count("same") == 1
    assert calls.count("clause") == 1


def test_request_cache_deduplicates_subprocess_across_probes_and_requests(monkeypatch):
    calls = []

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, text):
            self.stdout = "x\ny\nz" if "\n" in text else "ipa"

    def run(argv, **_kwargs):
        text = argv[-1]
        calls.append((argv[3], text))
        return Result(text)

    monkeypatch.setattr(piper.subprocess, "run", run)
    cache = {}
    piper.text_to_phonemes("one, clause.", "en-us", espeak_cache=cache)
    piper.text_to_phonemes("two, clause.", "en-us", espeak_cache=cache)
    # Each whole-text probe is unique; the shared clause is executed once.
    assert [text for _voice, text in calls] == [
        "one\nclause", "one", "clause", "two\nclause", "two"
    ]
    # A new request cache must execute the same successful CLI calls again.
    piper.text_to_phonemes("one, clause.", "en-us", espeak_cache={})
    assert [text for _voice, text in calls[-3:]] == ["one\nclause", "one", "clause"]
    # Voice is part of the key and must not cross-hit.
    piper.text_to_phonemes("one, clause.", "en-gb", espeak_cache=cache)
    assert [text for _voice, text in calls[-3:]] == ["one\nclause", "one", "clause"]


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
    # The final 34-sample partial frame is voiced and is now retained.
    speech = 22050
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


# -- silence trimming ------------------------------------------------------

def test_trim_silence_keeps_audible_partial_tail_after_full_frame():
    audio = np.concatenate([
        np.full(piper.SILENCE_FRAME_SIZE, 0.5, dtype=np.float32),
        np.full(17, 0.5, dtype=np.float32),
    ])
    trimmed = piper._trim_silence(audio)
    assert len(trimmed) == piper.SILENCE_FRAME_SIZE + 17
    np.testing.assert_array_equal(trimmed, audio)


def test_trim_silence_keeps_audible_partial_tail_after_silent_full_frame():
    audio = np.concatenate([
        np.zeros(piper.SILENCE_FRAME_SIZE, dtype=np.float32),
        np.full(23, 0.5, dtype=np.float32),
    ])
    trimmed = piper._trim_silence(audio)
    assert len(trimmed) == 23
    np.testing.assert_array_equal(trimmed, np.full(23, 0.5, dtype=np.float32))


def test_trim_silence_removes_silent_partial_tail_and_caps_frame_end():
    audio = np.concatenate([
        np.full(piper.SILENCE_FRAME_SIZE, 0.5, dtype=np.float32),
        np.zeros(31, dtype=np.float32),
    ])
    trimmed = piper._trim_silence(audio)
    assert len(trimmed) == piper.SILENCE_FRAME_SIZE
    np.testing.assert_array_equal(trimmed, np.full(piper.SILENCE_FRAME_SIZE, 0.5, dtype=np.float32))


@pytest.mark.parametrize(
    "size, voiced",
    [(17, True), (17, False), (piper.SILENCE_FRAME_SIZE, True), (piper.SILENCE_FRAME_SIZE, False)],
)
def test_trim_silence_short_and_exact_frame_boundaries(size, voiced):
    audio = np.full(size, 0.5 if voiced else 0.0, dtype=np.float32)
    trimmed = piper._trim_silence(audio)
    if voiced:
        np.testing.assert_array_equal(trimmed, audio)
    else:
        # Preserve the historical all-silence behavior, including short input.
        np.testing.assert_array_equal(trimmed, audio)


# -- review follow-ups: one boundary rule for sentences and clauses ---------

@pytest.mark.parametrize("text", [
    "Pi is 3.14 roughly", "Visit example.com today", "Ask Dr. Smith about it",
    "The U.S. market grew", "J. K. Rowling wrote it", "Use e.g. a fan",
    "We left at 5 p.m. sharp", "She is a Ph.D. student", "Made in the U.S.A. last year",
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


def test_direct_overlong_segment_fails_closed_without_truncation(monkeypatch):
    _fake_espeak(monkeypatch, {"Hello": "ab" * 40})
    model = _FakeModel()
    model.seq_len = 24
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    with pytest.raises(ValueError, match="exceeds model seq_len=24"):
        backend._synthesize_segment("Hello.", model, speed=1.0)
    assert model.seen == []


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


class _BucketModel:
    espeak_voice = "en-us"
    phoneme_id_map = ID_MAP
    seq_len = 5
    _frontend_npu = True
    sample_rate = 22050
    length_scale = noise_scale = noise_w = 1.0


def test_frontend_manifest_bucket_overrides_legacy_env_seq_len(monkeypatch, tmp_path):
    """The staged manifest bucket controls runtime seq_len, not PIPER_SEQ_LEN."""
    class _FakeRKNN:
        def __init__(self, **_kwargs):
            pass

        def load_rknn(self, _path):
            return 0

        def init_runtime(self):
            return 0

        def release(self):
            return None

    class _FakeORTSession:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setenv("PIPER_SEQ_LEN", "256")
    rknnlite = types.ModuleType("rknnlite")
    rknnlite_api = types.ModuleType("rknnlite.api")
    rknnlite_api.RKNNLite = _FakeRKNN
    ort = types.ModuleType("onnxruntime")
    ort.InferenceSession = _FakeORTSession
    monkeypatch.setitem(sys.modules, "rknnlite", rknnlite)
    monkeypatch.setitem(sys.modules, "rknnlite.api", rknnlite_api)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    manifest = {
        "bucket": {"input": [1, 128]},
        "remainder": {
            "output_semantic": ["z", "y_mask"],
            "outputs": ["z", "y_mask"],
            "output_shapes": [[1, 192, "T"], [1, 1, "T"]],
        },
    }
    model_dir = tmp_path
    rknn_path = model_dir / "text_encoder.rknn"
    remainder_path = model_dir / "remainder.onnx"
    manifest_path = model_dir / "manifest.json"
    rknn_path.write_bytes(b"fake")
    remainder_path.write_bytes(b"fake")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    model = piper._LangModel("en_US", model_dir)
    model.seq_len = int(os.environ["PIPER_SEQ_LEN"])
    monkeypatch.setattr(model, "_probe_decoder_window", lambda: 512)
    model._load_frontend_npu(rknn_path, remainder_path, manifest_path)
    assert model.seq_len == 128
    assert model._frontend_manifest["bucket"]["input"] == [1, 128]


def _fake_bucket_tokenizer(monkeypatch):
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)
    monkeypatch.setattr(
        piper, "phonemes_to_ids",
        lambda phonemes, _id_map: [1] * len(phonemes.replace(" ", "")),
    )


def test_frontend_bucket_split_exact_over_and_preserves_text(monkeypatch):
    _fake_bucket_tokenizer(monkeypatch)
    model = _BucketModel()
    assert list(piper._frontend_segments("abcde", model)) == ["abcde"]
    parts = list(piper._frontend_segments("ab, cd ef", model))
    assert "".join(parts) == "ab, cd ef"
    assert all(part.strip() for part in parts)
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_frontend_bucket_split_handles_unpunctuated_and_nonlinear_tokens(monkeypatch):
    model = _BucketModel()
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)
    monkeypatch.setattr(
        piper, "phonemes_to_ids",
        lambda phonemes, _id_map: [1] * (len(phonemes.split()) * 3 + len(phonemes) % 2),
    )
    parts = list(piper._frontend_segments("one two three four", model))
    assert "".join(parts) == "one two three four"
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_frontend_bucket_split_handles_unspaced_cjk_and_long_word(monkeypatch):
    _fake_bucket_tokenizer(monkeypatch)
    model = _BucketModel()
    for text in ("你好世界欢迎你今天天气好", "supercalifragilisticexpialidocious"):
        parts = list(piper._frontend_segments(text, model))
        assert "".join(parts) == text
        assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_nonfrontend_keeps_sentence_split_behavior():
    model = types.SimpleNamespace(_frontend_npu=False)
    assert list(piper._frontend_segments("First sentence. Second sentence.", model)) == [
        "First sentence.", "Second sentence."
    ]


def test_nonfrontend_fixed_model_bounded_split_preserves_digits_and_abbreviations(monkeypatch):
    _fake_bucket_tokenizer(monkeypatch)
    model = _BucketModel()
    model._frontend_npu = False
    text = "It costs 1,000 dollars, Dr. Smith says 3.14 is safe, " + "word " * 7 + "word"
    parts = list(piper._frontend_segments(text, model))
    assert "".join(parts) == text
    assert all(part.strip() for part in parts)
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)
    assert any("1,000" in part for part in parts)
    assert any("3.14" in part for part in parts)


def test_nonfrontend_stream_and_ordinary_use_same_bounded_segments(monkeypatch):
    _fake_bucket_tokenizer(monkeypatch)
    model = _BucketModel()
    model._frontend_npu = False
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._ready = True
    backend._models = {"en_US": model}
    monkeypatch.setattr(piper, "detect_language", lambda _text: "en_US")
    seen = []

    def synth_segment(text, *_args, **_kwargs):
        seen.append(text)
        return np.ones(4, dtype=np.float32), {"num_tokens": len(text)}

    monkeypatch.setattr(backend, "_synthesize_segment", synth_segment)
    text = "one two three four five six seven"
    backend.synthesize(text, language="en_US")
    ordinary = seen[:]
    seen.clear()
    list(backend.synthesize_stream(text, language="en_US"))
    assert ordinary == seen
    assert "".join(ordinary) == text


def test_legacy_infer_uses_model_seq_len_not_module_default():
    class _RKNN:
        def __init__(self):
            self.inputs = None

        def inference(self, inputs):
            self.inputs = inputs
            return [np.zeros(8, dtype=np.float32)]

    model = piper._LangModel.__new__(piper._LangModel)
    model.seq_len = 7
    model._rknn = _RKNN()
    out = model._infer_legacy(list(range(20)), 1.0, 0.5, 0.8)
    assert len(out) == 8
    assert model._rknn.inputs[0].shape == (1, 7)
    assert model._rknn.inputs[1].tolist() == [7]


def test_frontend_stream_split_yields_before_processing_all_tail(monkeypatch):
    model = _BucketModel()
    calls = []
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: calls.append(text) or text)
    monkeypatch.setattr(piper, "phonemes_to_ids", lambda phonemes, _id_map: [1] * len(phonemes.replace(" ", "")))
    segments = piper._frontend_segments("one two three four five six seven eight", model)
    first = next(segments)
    assert first
    assert "eight" not in calls


def test_frontend_short_sentence_reuses_cached_phonemization(monkeypatch):
    model = _BucketModel()
    calls = []
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: calls.append(text) or text)
    monkeypatch.setattr(piper, "phonemes_to_ids", lambda phonemes, _id_map: [1] * len(phonemes))
    cache = {}
    assert list(piper._frontend_segments("hello", model, cache)) == ["hello"]
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    model.infer = lambda *args: np.ones(16, dtype=np.float32)
    backend._synthesize_segment("hello", model, token_cache=cache)
    assert calls == ["hello"]


def test_frontend_bucket_split_rejects_minimal_unit_over_cap(monkeypatch):
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)
    monkeypatch.setattr(piper, "phonemes_to_ids", lambda phonemes, _id_map: [1] * 6)
    model = _BucketModel()
    with pytest.raises(ValueError, match="minimal text unit"):
        list(piper._frontend_segments("abcdef", model))


def test_frontend_bucket_minimum_probe_is_measured(monkeypatch):
    model = _BucketModel()
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)

    def ids(phonemes, _id_map):
        return [1] if len(phonemes) == 1 else [1] * 6

    monkeypatch.setattr(piper, "phonemes_to_ids", ids)
    text = "a" * 200
    parts = list(piper._frontend_segments(text, model))
    assert "".join(parts) == text
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_frontend_bucket_minimum_probe_keeps_leading_whitespace(monkeypatch):
    model = _BucketModel()
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)
    monkeypatch.setattr(
        piper, "phonemes_to_ids",
        lambda phonemes, _id_map: [1] * (1 if len(phonemes.strip()) <= 1 else 6),
    )
    text = "     a" + "b" * 14
    parts = list(piper._split_frontend_bucket(text, model))
    assert "".join(parts) == text
    assert all(part.strip() for part in parts)
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_frontend_bucket_probe_work_is_bounded_and_rejoins(monkeypatch):
    model = _BucketModel()
    calls = []
    monkeypatch.setattr(
        piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: calls.append(text) or text
    )
    monkeypatch.setattr(
        piper, "phonemes_to_ids",
        lambda phonemes, _id_map: [1] * (6 if len(phonemes) > 20 else len(phonemes)),
    )
    text = " ".join("word" for _ in range(200))
    parts = list(piper._frontend_segments(text, model))
    assert "".join(parts) == text
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)
    assert len(calls) <= len(parts) * (piper._FRONTEND_NATURAL_PROBE_LIMIT + 3)
    assert max(map(len, calls)) <= max(
        piper._FRONTEND_MIN_WINDOW_CHARS,
        model.seq_len * piper._FRONTEND_WINDOW_CHARS_PER_TOKEN,
    )


def test_frontend_window_fit_prefers_natural_boundary(monkeypatch):
    model = _BucketModel()
    monkeypatch.setattr(piper, "_FRONTEND_WINDOW_CHARS_PER_TOKEN", 4)
    monkeypatch.setattr(piper, "_FRONTEND_MIN_WINDOW_CHARS", 8)
    monkeypatch.setattr(piper, "text_to_phonemes", lambda text, voice, espeak_cache=None: text)
    monkeypatch.setattr(
        piper, "phonemes_to_ids",
        lambda phonemes, _id_map: [1] * len(phonemes.split()),
    )
    text = "alpha beta gamma delta epsilon zeta"
    parts = list(piper._frontend_segments(text, model))
    assert "".join(parts) == text
    assert parts[0].endswith(" ")
    assert not parts[0].endswith("de")
    assert all(piper._frontend_token_count(part, model) <= model.seq_len for part in parts)


def test_frontend_bucket_split_used_by_stream_and_nonstream(monkeypatch):
    _fake_bucket_tokenizer(monkeypatch)
    model = _BucketModel()
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._ready = True
    backend._models = {"en_US": model}
    calls = []

    def synth_segment(text, *_args, **_kwargs):
        calls.append(text)
        return np.ones(4, dtype=np.float32), {"num_tokens": len(text)}

    monkeypatch.setattr(backend, "_synthesize_segment", synth_segment)
    monkeypatch.setattr(piper, "detect_language", lambda text: "en_US")
    _, ordinary_meta = backend.synthesize("one two three four", language="en_US")
    assert ordinary_meta["sample_rate"] == 22050
    streamed = list(backend.synthesize_stream("one two three four", language="en_US"))
    assert streamed
    assert "".join(calls[:len(calls) // 2]) == "one two three four"
    assert "".join(calls[len(calls) // 2:]) == "one two three four"


def test_frontend_stream_segmentation_time_excludes_inference(monkeypatch):
    model = _BucketModel()
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._ready = True
    backend._models = {"en_US": model}
    monkeypatch.setattr(piper, "detect_language", lambda text: "en_US")
    monkeypatch.setattr(piper, "_frontend_segments", lambda *args: iter(["hello"]))
    clock = iter([0.0, 0.002, 0.102])
    monkeypatch.setattr(piper.time, "perf_counter", lambda: next(clock))

    def synth_segment(*_args, **_kwargs):
        return np.ones(4, dtype=np.float32), {"num_tokens": 1, "total_ms": 3.0}

    monkeypatch.setattr(backend, "_synthesize_segment", synth_segment)
    _, meta = next(backend.synthesize_stream("hello", language="en_US"))
    assert meta["segmentation_ms"] == pytest.approx(2.0)
    assert meta["total_ms"] == pytest.approx(5.0)
    assert meta["sample_rate"] == 22050


def test_a_domain_is_not_a_dotted_abbreviation():
    text = "See example.com. Contact support. Mail a.b@x.io. Done."
    assert piper._split_sentences(text) == [
        "See example.com.", "Contact support.", "Mail a.b@x.io.", "Done.",
    ]
    assert piper._split_sentences("Dr. Smith paid 3.14 at example.com. Plan B. Then go.") == [
        "Dr. Smith paid 3.14 at example.com.", "Plan B.", "Then go.",
    ]


# -- optional Chinese Matcha fallback --------------------------------------

class _FallbackModel:
    sample_rate = 22050


class _FakeMatcha:
    def __init__(self, source_rate=16000):
        self.source_rate = source_rate
        self.cleaned = False
        self.stream_closed = False

    def preload(self):
        return None

    def get_sample_rate(self):
        return self.source_rate

    def _wav(self):
        source = np.sin(np.arange(1600, dtype=np.float32) / 13.0) * 0.2
        buf = io.BytesIO()
        sf.write(buf, source, self.source_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def synthesize(self, **_kwargs):
        return self._wav(), {"backend": "matcha_rknn"}

    def synthesize_stream(self, **_kwargs):
        source = np.sin(np.arange(1600, dtype=np.float32) / 13.0) * 0.2

        def chunks():
            try:
                yield source[:1], {"chunk": 1}
                yield np.empty(0, dtype=np.float32), {"chunk": 2}
                yield source[1:257], {"chunk": 3}
                yield source[257:], {"chunk": 4}
            finally:
                self.stream_closed = True

        return chunks()

    def cleanup(self):
        self.cleaned = True


def _fallback_backend(fake=None):
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._ready = True
    backend._models = {"en_US": _FallbackModel()}
    backend._chinese_fallback_enabled = True
    backend._chinese_fallback = fake or _FakeMatcha()
    return backend


def test_chinese_fallback_routes_aliases_and_auto_without_env_mutation(monkeypatch):
    backend = _fallback_backend()
    before = os.environ.get("TTS_BACKEND")
    assert backend._language_route("你好", "zh_CN") == "zh"
    assert backend._language_route("豈", "zh_hans") == "zh"
    assert backend._language_route("hello", "en-US") == "en"
    assert backend._language_route("hello你好", "auto") == "zh"
    with pytest.raises(ValueError, match="does not support"):
        backend._language_route("hola", "es")
    assert os.environ.get("TTS_BACKEND") == before
    monkeypatch.setattr(piper, "detect_language", lambda _text: "en_US")


def test_chinese_fallback_runtime_info_and_ready_gate():
    backend = _fallback_backend()
    backend._chinese_fallback.is_ready = lambda: False
    assert not backend.is_ready()
    info = backend.runtime_info()
    assert info["chinese_fallback_enabled"] is True
    assert info["chinese_ready"] is False
    assert info["sample_rate"] == 22050


def test_invalid_chinese_fallback_setting_fails_fast(monkeypatch):
    monkeypatch.setenv("PIPER_CHINESE_FALLBACK", "typo")
    with pytest.raises(ValueError, match="matcha_rknn"):
        piper.PiperRKNNBackend()


def test_chinese_fallback_offline_resamples_to_piper_rate():
    pytest.importorskip("soxr")
    backend = _fallback_backend()
    wav, meta = backend.synthesize("你好", language="zh")
    audio, rate = sf.read(io.BytesIO(wav), dtype="float32")
    assert rate == 22050
    assert len(audio) == 2205
    assert meta["source_sample_rate"] == 16000
    assert meta["sample_rate"] == 22050
    assert meta["resample_ms"] >= 0
    assert meta["inference_time"] >= meta["resample_ms"] / 1000
    assert meta["rtf"] >= 0
    assert np.isfinite(audio).all()


def test_chinese_fallback_omits_none_noise_scale():
    class NoNoneNoise(_FakeMatcha):
        def __init__(self):
            super().__init__()
            self.kwargs_seen = []

        def synthesize(self, **kwargs):
            self.kwargs_seen.append(kwargs)
            return super().synthesize(**kwargs)

        def synthesize_stream(self, **kwargs):
            self.kwargs_seen.append(kwargs)
            return super().synthesize_stream(**kwargs)

    fake = NoNoneNoise()
    backend = _fallback_backend(fake)
    pytest.importorskip("soxr")
    backend.synthesize("你好", language="zh")
    list(backend.synthesize_stream("你好", language="zh"))
    backend.synthesize("你好", language="zh", noise_scale=0.7)
    list(backend.synthesize_stream("你好", language="zh", noise_scale=0.7))
    assert "noise_scale" not in fake.kwargs_seen[0]
    assert "noise_scale" not in fake.kwargs_seen[1]
    assert fake.kwargs_seen[2]["noise_scale"] == 0.7
    assert fake.kwargs_seen[3]["noise_scale"] == 0.7


def test_chinese_fallback_stream_flushes_and_matches_offline():
    pytest.importorskip("soxr")
    backend = _fallback_backend()
    wav, _ = backend.synthesize("你好", language="zh")
    offline, _ = sf.read(io.BytesIO(wav), dtype="float32")
    chunks = list(backend.synthesize_stream("你好", language="zh"))
    streamed = np.concatenate([audio for audio, _meta in chunks])
    assert streamed.dtype == np.float32
    assert np.isfinite(streamed).all()
    # Offline WAV is PCM16; allow only quantisation error after the same HQ
    # resampler, including the one-sample and empty source chunks above.
    assert len(streamed) == len(offline)
    assert np.max(np.abs(streamed - offline)) < 2e-4
    assert all(meta["sample_rate"] == 22050 for _audio, meta in chunks)


def test_chinese_fallback_cancel_closes_source_without_flush(monkeypatch):
    pytest.importorskip("soxr")
    backend = _fallback_backend()
    generator = backend.synthesize_stream("你好", language="zh")
    next(generator)
    generator.close()
    assert backend._chinese_fallback.stream_closed


def test_chinese_fallback_load_failure_cleans_context(monkeypatch):
    fake = _FakeMatcha()
    import rkvoice_stream.engine.tts as tts_engine

    monkeypatch.setattr(tts_engine, "create_backend", lambda _name: fake)
    monkeypatch.setattr(
        piper.PiperRKNNBackend,
        "_require_soxr",
        staticmethod(lambda: object()),
    )
    monkeypatch.setattr(fake, "preload", lambda: (_ for _ in ()).throw(RuntimeError("load")))
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._chinese_fallback = None
    with pytest.raises(RuntimeError, match="load"):
        backend._load_chinese_fallback()
    assert backend._chinese_fallback is None
    assert fake.cleaned


def test_chinese_fallback_requires_soxr_only_when_enabled(monkeypatch):
    backend = piper.PiperRKNNBackend.__new__(piper.PiperRKNNBackend)
    backend._chinese_fallback = None
    called = []
    monkeypatch.setattr(
        backend,
        "_require_soxr",
        lambda: (_ for _ in ()).throw(RuntimeError("soxr missing")),
    )
    monkeypatch.setattr(
        piper.PiperRKNNBackend,
        "_synthesize_segment",
        lambda *_args, **_kwargs: (np.zeros(1, dtype=np.float32), {}),
    )
    with pytest.raises(RuntimeError, match="soxr missing"):
        backend._load_chinese_fallback()
    assert called == []

"""Window-commit behaviour of the true-streaming Qwen3 ASR session.

The rolling encoder buffer only holds ``QWEN3_ASR_TRUE_ROLL_SEC`` seconds.
Before window commits existed, overflow dropped the oldest frames outright, so
an utterance longer than the window came back missing its beginning.  These
tests use a fake encoder/decoder pair whose "transcript" is a deterministic
function of the audio, which makes word-level loss, duplication and ordering
directly assertable without an NPU.
"""

import numpy as np
import pytest

from rkvoice_stream.backends.asr.qwen3.streaming import (
    ENCODER_HOP_SAMPLES,
    WINDOW_COMMIT_RETRY_CHUNKS,
    Qwen3TrueStreamingASRStream,
)

SAMPLE_RATE = 16000
CHUNK_SEC = 0.4
LCTX_SEC = 0.8  # block-aligned left context keeps fake frame ids exact
CHUNK_SAMPLES = int(CHUNK_SEC * SAMPLE_RATE)
FRAMES_PER_CHUNK = CHUNK_SAMPLES // ENCODER_HOP_SAMPLES  # 5


def make_audio(seconds: float) -> np.ndarray:
    """Audio whose every 1280-sample block carries its own word id."""
    n_blocks = int(seconds * SAMPLE_RATE) // ENCODER_HOP_SAMPLES
    blocks = [
        np.full(ENCODER_HOP_SAMPLES, float(i), dtype=np.float32)
        for i in range(n_blocks)
    ]
    return np.concatenate(blocks)


class _FakeEncoder:
    def encode(self, audio):
        n = len(audio) // ENCODER_HOP_SAMPLES
        frames = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            frames[i, 0] = audio[i * ENCODER_HOP_SAMPLES]
        return frames


class _FakeDecoder:
    _early_stop_tokens = 0

    def __init__(self):
        self.calls = []

    def run_embed(self, full_embd, n_tokens, keep_history=0):
        words = ["w%d" % int(round(v)) for v in full_embd[:, 0]]
        if self._early_stop_tokens > 0:
            words = words[: self._early_stop_tokens]
        self.calls.append(("early" if self._early_stop_tokens else "full",
                           len(words)))
        return {
            "text": " ".join(words),
            "n_tokens_generated": len(words),
            "aborted": False,
            "perf": {},
        }

    def abort(self):
        pass


class _FakeEngine:
    def __init__(self):
        self.encoder = _FakeEncoder()
        self.decoder = _FakeDecoder()

    def build_embed(self, all_frames, **kwargs):
        return all_frames, all_frames.shape[0]


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("QWEN3_ASR_VAD_BACKEND", "silero")
    monkeypatch.setenv("QWEN3_ASR_TRUE_CHUNK_SEC", str(CHUNK_SEC))
    monkeypatch.setenv("QWEN3_ASR_TRUE_LCTX_SEC", str(LCTX_SEC))
    monkeypatch.setenv("QWEN3_ASR_TRUE_ROLL_SEC", "5")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_TOKENS", "1000")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_INTERVAL_MS", "0")
    monkeypatch.setenv("QWEN3_ASR_TRUE_PARTIAL_WARMUP", "0")
    monkeypatch.delenv("QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC", raising=False)
    monkeypatch.delenv("QWEN3_ASR_ACCUMULATE_SEGMENTS", raising=False)
    monkeypatch.delenv("QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT",
                       raising=False)
    return monkeypatch


def feed_all(stream, audio):
    texts = []
    for i in range(0, len(audio), CHUNK_SAMPLES):
        texts.append(stream.feed_audio(audio[i:i + CHUNK_SAMPLES])["text"])
    return texts


def expected_words(seconds: float) -> list[str]:
    n_chunks = int(seconds * SAMPLE_RATE) // CHUNK_SAMPLES
    return ["w%d" % i for i in range(n_chunks * FRAMES_PER_CHUNK)]


# ── long utterance ────────────────────────────────────────────────────


def test_long_utterance_keeps_every_word(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._max_encoder_frames == 65
    assert stream._roll_overlap_frames == 13

    feed_all(stream, make_audio(12.0))
    result = stream.finish(apply_itn_flag=False)

    words = result["text"].split()
    assert stream._window_commits >= 2
    assert words == expected_words(12.0)          # nothing lost, order kept
    assert len(words) == len(set(words))          # no duplicates from overlap
    assert "." not in result["text"]              # no mid-sentence terminator


def test_partials_grow_monotonically_and_keep_the_opening(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())

    texts = [t for t in feed_all(stream, make_audio(12.0)) if t]

    assert len(texts) > 5
    prev: list[str] = []
    for text in texts:
        cur = text.split()
        assert cur[0] == "w0", text[:40]
        assert cur[:len(prev)] == prev, (prev[-3:], cur[:len(prev)][-3:])
        prev = cur


def test_overlap_zero_still_loses_no_word(env):
    env.setenv("QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC", "0")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._roll_overlap_frames == 0

    feed_all(stream, make_audio(12.0))
    result = stream.finish(apply_itn_flag=False)

    assert stream._window_commits >= 1
    assert result["text"].split() == expected_words(12.0)


# ── unchanged behaviour when the window never overflows ───────────────


def test_short_utterance_takes_no_new_path(env):
    engine = _FakeEngine()
    stream = Qwen3TrueStreamingASRStream(engine)

    feed_all(stream, make_audio(4.0))
    result = stream.finish(apply_itn_flag=False)

    assert stream._window_commits == 0
    assert stream._window_committed_text == ""
    assert result["text"].split() == expected_words(4.0)
    # Every decode was either a throttled partial or the single final.
    assert sum(1 for kind, _ in engine.decoder.calls if kind == "full") == 1


# ── boundary stitching ────────────────────────────────────────────────


def test_commit_strips_invented_terminator_and_dedups_case(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    stream._window_committed_text = "computers that you are selling"

    # Next window re-decodes the overlap: same words, different case, and the
    # previous commit's invented full stop is already gone.
    merged = stream._join_text(
        stream._window_committed_text,
        "Selling in the online store",
        max_units=16,
    )

    assert merged == "computers that you are selling in the online store"


def test_midstream_terminator_does_not_survive_a_commit(env):
    engine = _FakeEngine()

    def run_embed(full_embd, n_tokens, keep_history=0):
        return {"text": "the product.", "aborted": False, "perf": {}}

    engine.decoder.run_embed = run_embed
    stream = Qwen3TrueStreamingASRStream(engine)
    stream._encoder_frames = [np.zeros((70, 4), dtype=np.float32)]
    stream._total_encoder_frames = 70

    assert stream._commit_window_overflow() is True
    assert stream._window_committed_text == "the product"


# ── lifecycle ─────────────────────────────────────────────────────────


def test_new_utterance_clears_committed_window_text(env):
    env.setenv("QWEN3_ASR_ALLOW_AUTO_RESUME_AFTER_ENDPOINT", "1")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    stream._window_committed_text = "first sentence"
    stream._archive_text = "first sentence"
    stream._episode_final = True

    stream.feed_audio(np.full(CHUNK_SAMPLES, 0.2, dtype=np.float32))

    assert stream._episode_final is False
    assert stream._window_committed_text == ""
    assert stream._archive_text == ""


def test_commit_is_skipped_while_a_final_decode_owns_the_decoder(env):
    engine = _FakeEngine()
    stream = Qwen3TrueStreamingASRStream(engine)
    stream._encoder_frames = [np.zeros((70, 4), dtype=np.float32)]
    stream._total_encoder_frames = 70

    for attr in ("_finalizing", "_episode_final", "_final_decode_in_progress"):
        setattr(stream, attr, True)
        assert stream._window_commit_allowed() is False
        assert stream._commit_window_overflow() is False
        setattr(stream, attr, False)

    assert engine.decoder.calls == []
    assert stream._window_commit_allowed() is True


def test_overflow_while_finalizing_falls_back_to_dropping(env):
    engine = _FakeEngine()
    stream = Qwen3TrueStreamingASRStream(engine)
    feed_all(stream, make_audio(2.0))
    stream._finalizing = True

    feed_all(stream, make_audio(12.0))

    assert stream._window_commits == 0
    assert stream._total_encoder_frames <= stream._max_encoder_frames


# ── review follow-ups ────────────────────────────────────────────────


class _PunctStopDecoder(_FakeDecoder):
    """Ends every window with "." and honours the final punctuation stop.

    Mirrors the RKLLM decoder: with ``_early_stop_tokens == 0`` and
    ``_final_stop_on_punctuation`` set, generation is aborted at the first
    sentence terminator.  Here a terminator follows every 10th word.
    """

    _final_stop_on_punctuation = True

    def run_embed(self, full_embd, n_tokens, keep_history=0):
        ids = [int(round(v)) for v in full_embd[:, 0]]
        out, aborted = [], False
        for i in ids:
            out.append("w%d" % i + ("." if i % 10 == 9 else ""))
            if (self._early_stop_tokens == 0 and self._final_stop_on_punctuation
                    and out[-1].endswith(".")):
                aborted = True
                break
        self.calls.append(("stopped" if aborted else "whole", len(out)))
        return {"text": " ".join(out), "n_tokens_generated": len(out),
                "aborted": aborted,
                "abort_reason": "final_punctuation" if aborted else "",
                "perf": {}}


def test_window_holding_two_sentences_commits_both(env):
    engine = _FakeEngine()
    engine.decoder = _PunctStopDecoder()
    stream = Qwen3TrueStreamingASRStream(engine)

    feed_all(stream, make_audio(8.0))

    assert stream._window_commits == 1
    committed = [w.rstrip(".") for w in stream._window_committed_text.split()]
    # The whole 65+ frame window, not just its first sentence (w0..w9).
    assert committed[:12] == ["w%d" % i for i in range(12)]
    assert len(committed) > 60
    # The punctuation stop is back on for the real final decode.
    assert engine.decoder._final_stop_on_punctuation is True


class _AbortingDecoder(_FakeDecoder):
    def __init__(self, abort_full_calls: int, flag_aborted: bool = False):
        super().__init__()
        self.abort_left = abort_full_calls
        self.flag_aborted = flag_aborted

    def run_embed(self, full_embd, n_tokens, keep_history=0):
        result = super().run_embed(full_embd, n_tokens, keep_history)
        if self._early_stop_tokens == 0 and self.abort_left > 0:
            self.abort_left -= 1
            # Exactly what the RKLLM decoder returns after ``abort()`` or an
            # async timeout: the reason is recorded, ``aborted`` stays False
            # (decoder.py sets ``_aborted`` only for its own stop rules).
            result.update(text=" ".join(result["text"].split()[:3]),
                          aborted=self.flag_aborted, abort_reason="external")
        return result


@pytest.mark.parametrize("flag_aborted", [False, True])
def test_aborted_commit_is_not_committed_and_is_retried(env, flag_aborted):
    engine = _FakeEngine()
    engine.decoder = _AbortingDecoder(abort_full_calls=1,
                                      flag_aborted=flag_aborted)
    stream = Qwen3TrueStreamingASRStream(engine)

    feed_all(stream, make_audio(12.0))
    result = stream.finish(apply_itn_flag=False)

    # The truncated decode was thrown away, the frames were kept, the retry on
    # the next chunk committed them: still every word, once.
    assert result["text"].split() == expected_words(12.0)


def test_commit_that_keeps_aborting_drops_with_a_warning(env, caplog):
    engine = _FakeEngine()
    engine.decoder = _AbortingDecoder(abort_full_calls=10**6)
    stream = Qwen3TrueStreamingASRStream(engine)

    with caplog.at_level("WARNING"):
        feed_all(stream, make_audio(12.0))

    assert stream._window_commits == 0
    assert stream._window_committed_text == ""
    # Bounded: the cap plus the chunks a retry is allowed to wait for.
    assert stream._total_encoder_frames <= (
        stream._max_encoder_frames
        + WINDOW_COMMIT_RETRY_CHUNKS * FRAMES_PER_CHUNK)
    assert any("without committing their text" in r.message for r in caplog.records)


def test_abort_partial_decode_cannot_reach_a_window_commit(env):
    engine = _FakeEngine()
    aborts = []
    engine.decoder.abort = lambda: aborts.append(1)
    stream = Qwen3TrueStreamingASRStream(engine)

    stream._window_commit_in_progress = True
    stream.abort_partial_decode()
    assert aborts == []
    stream._window_commit_in_progress = False
    stream.abort_partial_decode()
    assert aborts == [1]


def test_no_overlap_means_no_dedup(env):
    env.setenv("QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC", "0")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._window_dedup_units == 0
    # "no" on both sides of a seam with nothing decoded twice is the speaker.
    assert stream._join_text("he said no", "no I mean yes",
                             max_units=stream._window_dedup_units) == \
        "he said no no I mean yes"


def test_overlap_dedup_still_applies_with_overlap(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._window_dedup_units > 0
    assert stream._join_text("such as the speaker", "The speaker and a case",
                             max_units=stream._window_dedup_units) == \
        "such as the speaker and a case"


def test_cjk_dedup_keeps_the_remainders_own_spacing(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._join_text("我们去你好", "你好 New York 见",
                             max_units=stream._window_dedup_units) == \
        "我们去你好New York 见"


def test_real_decoder_reports_external_abort_by_reason_only():
    """Pin the contract the commit guard relies on, against the real class."""
    import inspect
    from rkvoice_stream.backends.asr.qwen3 import decoder as dec
    src = inspect.getsource(dec)
    abort_body = src[src.index("    def abort(self):"):]
    abort_body = abort_body[:abort_body.index("    def release(self):")]
    assert "_abort_reason" in abort_body and "self._aborted = True" not in abort_body


def test_retry_budget_does_not_leak_into_the_next_utterance(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    stream._window_commit_retries = 2
    stream._episode_final = True
    stream._maybe_resume_new_utterance(np.full(1600, 0.5, dtype=np.float32))
    assert stream._window_commit_retries == 0


# ── seam junk (measured through the service on RK3588, 2026-09-21) ────


def test_seam_survives_a_stray_token_at_the_start_of_the_window(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    left = ("I would like to know more about the speaker and the edge computers that you are "
            "selling in the online store today, because our team is planning to build a voice "
            "assistant for the factory")
    right = ("Assistant: Assistant for the factory floor, and we need hardware that can run "
             "speech recognition locally without any cloud connection.")
    assert stream._join_window(left, right) == (
        left + " floor, and we need hardware that can run speech recognition locally "
        "without any cloud connection.")


def test_seam_survives_a_clipped_first_word(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._join_window("please briefly introduce your corporate",
                               "Duce your corporate product such as this") == \
        "please briefly introduce your corporate product such as this"


def test_seam_match_ignores_punctuation_inside_the_overlap(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._join_window("for the factory floor, and we", "factory floor and we need hardware") == \
        "for the factory floor, and we need hardware"
    # ...and what remains may open on punctuation: no space before it.
    assert stream._join_window("store today", "Today, because our team") == \
        "store today, because our team"


def test_one_word_after_junk_is_not_a_seam(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    # "the" two units in is a coincidence; nothing may be dropped.
    assert stream._join_window("we went to the", "and then the dog barked") == \
        "we went to the and then the dog barked"


def test_junk_tolerance_is_bounded(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    # The repeat sits four words in -- past the skip limit -- so it is speech.
    assert stream._join_window("turn on the light", "now please go and turn on the light again") == \
        "turn on the light now please go and turn on the light again"


def test_no_junk_tolerance_without_overlap_or_outside_window_joins(env):
    env.setenv("QWEN3_ASR_TRUE_ROLL_OVERLAP_SEC", "0")
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._join_window("assistant for the factory", "Assistant: Assistant for the factory floor") == \
        "assistant for the factory Assistant: Assistant for the factory floor"
    # Dictation-mode joins keep their old exact-prefix behaviour.
    stream2 = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream2._join_text("assistant for the factory", "um assistant for the factory floor", max_units=8) == \
        "assistant for the factory um assistant for the factory floor"


def test_cjk_seam_skips_a_stray_character(env):
    stream = Qwen3TrueStreamingASRStream(_FakeEngine())
    assert stream._join_window("桥下垂直净空十五米", "嗯空十五米，该项目于二零一一年") == \
        "桥下垂直净空十五米，该项目于二零一一年"

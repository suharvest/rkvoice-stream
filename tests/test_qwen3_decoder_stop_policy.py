from rkvoice_stream.backends.asr.qwen3.decoder import (
    should_stop_after_punctuation,
)


def test_punctuation_stop_policy_is_opt_in():
    assert not should_stop_after_punctuation(
        "而且，太平洋海啸预警中心也表示。",
        enabled=False,
        min_chars=0,
    )


def test_punctuation_stop_policy_handles_chinese_and_english_boundaries():
    assert should_stop_after_punctuation("我们的非常震惊。", enabled=True)
    assert should_stop_after_punctuation("He refers to the rumors.", enabled=True)
    assert should_stop_after_punctuation("Is this finished?", enabled=True)
    assert not should_stop_after_punctuation("He refers to the rumors", enabled=True)


def test_punctuation_stop_policy_respects_min_chars():
    assert not should_stop_after_punctuation("好。", enabled=True, min_chars=4)
    assert should_stop_after_punctuation("我们的非常震惊。", enabled=True, min_chars=4)


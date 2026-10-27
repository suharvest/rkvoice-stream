"""Unit tests for Matcha-TTS Chinese Tone Sandhi (连续三声变调)."""

from __future__ import annotations

import pytest

from rkvoice_stream.backends.tts import matcha as m

Cls = m.RKNNMatchaVocoder

# Real IDs and tokens for tone sandhi testing
TOKENS = {
    ' ': 1, ',': 4, '.': 5, '!': 6, '?': 7,
    # pinyin tokens including tones 1-5
    'ke1': 876, 'ke2': 877, 'ke3': 878, 'ke4': 879,
    'yi1': 1946, 'yi2': 1947, 'yi3': 1948, 'yi4': 1949,
    'ni1': 1363, 'ni2': 1364, 'ni3': 1365, 'ni4': 1366,
    'hao1': 702, 'hao2': 703, 'hao3': 704, 'hao4': 705,
    'ma1': 1201, 'ma2': 1202, 'ma3': 1203,
    'shou1': 1600, 'shou2': 1601, 'shou3': 1602, 'shou4': 1603,
    'biao1': 200, 'biao2': 201, 'biao3': 202, 'biao4': 203,
    'mai1': 1100, 'mai2': 1101, 'mai3': 1102, 'mai4': 1103,
    'zhan1': 2100, 'zhan2': 2101, 'zhan3': 2102, 'zhan4': 2103,
    'lan1': 950, 'lan2': 951, 'lan3': 952, 'lan4': 953,
    'guan1': 620, 'guan2': 621, 'guan3': 622, 'guan4': 623,
    'qing1': 1431, 'qing2': 1432, 'qing3': 1433, 'qing4': 1434,
    'xuan1': 1900, 'xuan2': 1901, 'xuan3': 1902, 'xuan4': 1903,
    'wu1': 1850, 'wu2': 1851, 'wu3': 1852, 'wu4': 1853,
    'chi1': 301, 'fan4': 480,
    'bei3': 210, 'jing1': 820,
    'wo3': 1840, 'zai4': 2050, 'ne5': 1350, 'you3': 1980, 'shen2': 1580, 'me5': 1200,
    'bang1': 150, 'nin2': 1370,
}

LEXICON = {
    '可以': ['ke3', 'yi3'],
    '你好': ['ni3', 'hao3'],
    '手': ['shou3'],
    '表': ['biao3'],
    '买': ['mai3'],
    '展': ['zhan3'],
    '览': ['lan3'],
    '馆': ['guan3'],
    '请': ['qing3'],
    '你': ['ni3'],
    '好': ['hao3'],
    '吗': ['ma1'],
    '选': ['xuan3'],
    '五': ['wu3'],
    '吃': ['chi1'],
    '饭': ['fan4'],
    '北': ['bei3'],
    '京': ['jing1'],
    '我': ['wo3'],
    '在': ['zai4'],
    '呢': ['ne5'],
    '有': ['you3'],
    '什': ['shen2'],
    '么': ['me5'],
    '帮': ['bang1'],
    '您': ['nin2'],
}


@pytest.fixture
def vocoder():
    inst = object.__new__(Cls)
    inst._token_to_id = dict(TOKENS)
    inst._lexicon = dict(LEXICON)
    inst.data_dir = None
    inst._espeak_ipa = lambda w: ''
    return inst


def test_tone_sandhi_two_char_words(vocoder):
    """2-char word with tone 3+3 -> 2+3."""
    assert vocoder._chinese_token_strings("可以") == ["ke2", "yi3"]
    assert vocoder._chinese_token_strings("你好") == ["ni2", "hao3"]
    assert vocoder._chinese_token_strings("手表") == ["shou2", "biao3"]


def test_tone_sandhi_three_char_continuous(vocoder):
    """3 consecutive 3-tones (3+3+3) -> 2+2+3 (e.g. 展览馆)."""
    assert vocoder._chinese_token_strings("展览馆") == ["zhan2", "lan2", "guan3"]


def test_tone_sandhi_three_char_ends_with_other_tone(vocoder):
    """3-char sequence with 3+3+1 -> 2+3+1 (e.g. 你好吗)."""
    assert vocoder._chinese_token_strings("你好吗") == ["ni2", "hao3", "ma1"]


def test_tone_sandhi_cross_word_single_char(vocoder):
    """Cross-word sandhi across adjacent single-character 3-tone words."""
    assert vocoder._chinese_token_strings("请你") == ["qing2", "ni3"]


def test_tone_sandhi_non_three_tones_unchanged(vocoder):
    """Non-3 tones should not be modified."""
    assert vocoder._chinese_token_strings("吃饭") == ["chi1", "fan4"]
    assert vocoder._chinese_token_strings("北京") == ["bei3", "jing1"]


def test_tone_sandhi_unknown_char_barrier(vocoder):
    """An unknown character without lexicon entry must act as a barrier."""
    # '龘' is not in LEXICON. It should separate '你' and '好' so they do not sandhi.
    assert vocoder._chinese_token_strings("你龘好") == ["ni3", "hao3"]


def test_tone_sandhi_fallback_when_second_tone_missing(vocoder):
    """If a candidate tone 2 token is missing from tokens.txt, keep tone 3."""
    # Temporarily remove 'ke2'
    del vocoder._token_to_id['ke2']
    # '可以': ke3 would become ke2, but since ke2 is missing, it safely keeps ke3
    assert vocoder._chinese_token_strings("可以") == ["ke3", "yi3"]


def test_tone_sandhi_punctuation_isolation(vocoder):
    """Punctuation strictly isolates phrases: tone sandhi must not cross commas."""
    tokens = vocoder.text_to_tokens("选五，五")
    id_to_t = {v: k for k, v in vocoder._token_to_id.items()}
    tok_strs = [id_to_t.get(t, str(t)) for t in tokens]
    # '选五' -> xuan2 wu3. Then comma. Then '五' -> wu3.
    # The comma separates them: the second '五' stays wu3, first '五' is not affected by second.
    assert tok_strs == ["xuan2", "wu3", ",", "wu3"]


def test_full_greeting_sentence_tokens(vocoder):
    """Target greeting sentence: '你好，我在呢，有什么可以帮您？'"""
    tokens = vocoder.text_to_tokens("你好，我在呢，有什么可以帮您？")
    id_to_t = {v: k for k, v in vocoder._token_to_id.items()}
    tok_strs = [id_to_t.get(t, str(t)) for t in tokens]
    expected = [
        "ni2", "hao3", ",",
        "wo3", "zai4", "ne5", ",",
        "you3", "shen2", "me5",
        "ke2", "yi3", "bang1", "nin2", "?"
    ]
    assert tok_strs == expected

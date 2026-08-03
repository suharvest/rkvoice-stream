"""Text-frontend tests for the matcha zh-en backend.

The expected token sequences here are not invented: they are the exact int64
tensors sherpa-onnx's native frontend feeds this model, captured with
OfflineTtsModelConfig(debug=True) on the device.  Reproducing them is the whole
point -- driving the acoustic model with anything else is what produced 38.3%
English WER (`light` -> `lehshad`, `turn on` -> `10 down`) against 0.0% for the
reference frontend.

No espeak, no model files, no NPU: espeak output is stubbed with the IPA the
C API actually returns for these words.
"""

from __future__ import annotations

import pytest

from rkvoice_stream.backends.tts import matcha as m

Cls = m.RKNNMatchaVocoder


# Real ids, lifted from matcha-icefall-zh-en/tokens.txt.
TOKENS = {
    ' ': 1, ';': 2, ':': 3, ',': 4, '.': 5, '!': 6, '?': 7,
    'A': 22, 'I': 23, 'O': 24, 'Q': 25, 'S': 26, 'T': 27, 'W': 28, 'Y': 29,
    'a': 31, 'b': 32, 'd': 34, 'e': 35, 'f': 36, 'h': 37, 'i': 38,
    'l': 41, 'm': 42, 'n': 43, 'p': 45, 's': 48, 't': 49, 'u': 50,
    'v': 51, 'w': 52, 'x': 53, 'z': 55, 'ɐ': 57, 'ɔ': 61, 'ð': 65,
    'ʤ': 66, 'ə': 67, 'ɛ': 69, 'ɜ': 70, 'ɡ': 72, 'ɪ': 75, 'ŋ': 79,
    'ɹ': 87, 'ʧ': 95, 'ˈ': 105, 'ˌ': 106, 'ː': 107,
    # pinyin, for the mixed-script case
    'qing3': 1433, 'da3': 398, 'kai1': 856, 'de5': 420, 'deng1': 436,
}

# espeak-ng C API output, per word (NOT the CLI -- see _EspeakPhonemizer).
IPA = {
    'turn': 'tˈɜːn', 'on': 'ˈɔn', 'the': 'ðə', 'light': 'lˈaɪt',
    'living': 'lˈɪvɪŋ', 'room': 'ɹˈuːm',
    'weather': 'wˈɛðəɹ', 'water': 'wˈɔːɾɚ', 'day': 'dˈeɪ', 'go': 'ɡˈoʊ',
    'how': 'hˈaʊ', 'boy': 'bˈɔɪ', 'church': 'tʃˈɜːtʃ', 'judge': 'dʒˈʌdʒ',
    'my': 'mˈaɪ', 'hello': 'həlˈoʊ', 'world': 'wˈɜːld',
}


@pytest.fixture
def be():
    inst = object.__new__(Cls)
    inst._token_to_id = dict(TOKENS)
    inst._lexicon = {
        '请': ['qing3'], '打开': ['da3', 'kai1'],
        '的': ['de5'], '灯': ['deng1'],
    }
    inst.data_dir = None
    inst._espeak_ipa = lambda w: IPA.get(w.lower(), '')
    return inst


# ------------------------------------------------------------ IPA mapping


@pytest.mark.parametrize(
    "ipa,expected",
    [
        ('lˈaɪt', ['l', 'ˈ', 'I', 't']),          # aɪ -> single token I
        ('dˈeɪ', ['d', 'ˈ', 'A']),                # eɪ -> A
        ('ɡˈoʊ', ['ɡ', 'ˈ', 'O']),                # oʊ -> O
        ('hˈaʊ', ['h', 'ˈ', 'W']),                # aʊ -> W
        ('bˈɔɪ', ['b', 'ˈ', 'Y']),                # ɔɪ -> Y
        ('tʃˈɜːtʃ', ['ʧ', 'ˈ', 'ɜ', 'ʧ']),        # tʃ -> ʧ, ː dropped
        ('dʒˈʌdʒ', ['ʤ', 'ˈ', 'ʌ', 'ʤ']),         # dʒ -> ʤ
        ('wˈɔːɾɚ', ['w', 'ˈ', 'ɔ', 'ɾ', 'ə', 'ɹ']),  # ɚ EXPANDS to ə + ɹ
        ('tˈɜːn', ['t', 'ˈ', 'ɜ', 'n']),          # length mark dropped
    ],
)
def test_ipa_mapping(ipa, expected):
    assert m.ipa_to_token_strings(ipa) == expected


def test_stress_marks_are_kept_in_place():
    assert m.ipa_to_token_strings('mˈaɪ') == ['m', 'ˈ', 'I']
    assert m.ipa_to_token_strings('ɐbˌaʊt') == ['ɐ', 'b', 'ˌ', 'W', 't']


def test_internal_space_becomes_a_boundary():
    """espeak expands some inputs into several words in one call."""
    assert m.ipa_to_token_strings('ɡɹˈɪnɪŋ fˈeɪs') == [
        'ɡ', 'ɹ', 'ˈ', 'ɪ', 'n', 'ɪ', 'ŋ', ' ', 'f', 'ˈ', 'A', 's',
    ]


# ------------------------------------------------- ground-truth sequences


def test_matches_sherpa_for_a_plain_sentence(be):
    """Measured sherpa tensor for 'Turn on the light.'

    t ˈ ɜ n _ ˈ ɔ n _ ð ə _ l ˈ I t _ .
    """
    assert be.text_to_tokens("Turn on the light.") == [
        49, 105, 70, 43, 1, 105, 61, 43, 1, 65, 67, 1, 41, 105, 23, 49, 1, 5,
    ]


def test_matches_sherpa_for_mixed_scripts(be):
    """Measured sherpa tensor for '请打开 living room 的灯。'

    qing3 da3 kai1 l ˈ ɪ v ɪ ŋ _ ɹ ˈ u m _ de5 deng1 .
    Note: no boundary at the CN->EN edge, one at the EN->CN edge, and none
    before '.' because 灯 is a lexicon word rather than a Latin one.
    """
    assert be.text_to_tokens("请打开 living room 的灯。") == [
        1433, 398, 856,
        41, 105, 75, 51, 75, 79, 1,
        87, 105, 50, 42, 1,
        420, 436, 5,
    ]


# ------------------------------------------------------ word boundaries


def test_boundary_between_words_but_not_at_the_edges(be):
    ids = be.text_to_tokens("the light")
    assert ids == [65, 67, 1, 41, 105, 23, 49]
    assert ids[0] != 1 and ids[-1] != 1


def test_no_boundary_after_punctuation(be):
    """'hello; world' -> hello _ ; world  (nothing between ';' and 'world')."""
    ids = be.text_to_tokens("hello; world")
    assert ids == [
        37, 67, 41, 105, 24, 1,   # hello
        2,                        # ;
        52, 105, 70, 41, 34,      # world
    ]


def test_punctuation_is_emitted_not_dropped(be):
    assert 5 in be.text_to_tokens("Turn on the light.")
    assert 4 in be.text_to_tokens("the light, the light")


def test_full_width_punctuation_normalizes(be):
    assert be.text_to_tokens("请。")[-1] == TOKENS['.']
    assert be.text_to_tokens("请，")[-1] == TOKENS[',']
    assert be.text_to_tokens("请！")[-1] == TOKENS['!']
    assert be.text_to_tokens("请？")[-1] == TOKENS['?']


def test_colon_folds_onto_comma(be):
    """token 3 (':') exists but the reference frontend never emits it."""
    assert be.text_to_tokens("请：")[-1] == TOKENS[',']
    assert TOKENS[':'] not in be.text_to_tokens("请：")


def test_dropped_punctuation_produces_nothing(be):
    for ch in "-'（）《》":
        assert be.text_to_tokens(ch) == [], ch


# ------------------------------------------------- word resolution order


def test_single_letter_word_bypasses_espeak(be):
    """'a' is itself a token, so it must not become eɪ/A."""
    called = []
    be._espeak_ipa = lambda w: called.append(w) or 'ˈeɪ'
    assert be.text_to_tokens("a") == [TOKENS['a']]
    assert called == []


def test_unknown_word_yields_nothing_rather_than_garbage(be):
    assert be.text_to_tokens("zzzznotaword") == []

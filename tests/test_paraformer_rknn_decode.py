from rkvoice_stream.backends.asr.paraformer_rknn import decode_ids


def test_decode_ids_preserves_cjk_without_spaces():
    tokens = ["<blank>", "<s>", "</s>", "我", "们", "好"]

    assert decode_ids([3, 4, 5], tokens) == "我们好"


def test_decode_ids_restores_english_word_spaces():
    tokens = ["<blank>", "<s>", "</s>", "television", "reports", "show"]

    assert decode_ids([3, 4, 5], tokens) == "television reports show"


def test_decode_ids_merges_bpe_subwords():
    tokens = ["<blank>", "<s>", "</s>", "pol@@", "itical", "chatter"]

    assert decode_ids([3, 4, 5], tokens) == "political chatter"


def test_decode_ids_handles_sentencepiece_boundaries():
    tokens = ["<blank>", "<s>", "</s>", "▁hello", "▁world"]

    assert decode_ids([3, 4], tokens) == "hello world"

import numpy as np
import pytest

from rkvoice_stream.backends.tts.kokoro_istft import conv_post_to_waveform


def test_istft_self_contained_zero_fixture_and_shape():
    for frames in (2, 19, 19201):
        value = conv_post_to_waveform(np.zeros((1, 22, frames), dtype=np.float32))
        assert value.shape == (1, 5 * (frames - 1))
        assert np.isfinite(value).all()


def test_istft_rejects_invalid_abi():
    with pytest.raises(ValueError):
        conv_post_to_waveform(np.zeros((1, 11, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        conv_post_to_waveform(np.full((1, 22, 2), np.nan, dtype=np.float32))

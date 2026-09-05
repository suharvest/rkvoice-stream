import numpy as np
import pytest

from rkvoice_stream.backends.tts.kokoro_long32 import parse_route_ts, select_profile, snap_durations
from rkvoice_stream.backends.tts.kokoro_long32_frontend import LANG_ALIASES, ROUTES, VOICES


def test_frontend_public_language_and_voice_contract():
    assert {"en", "zh", "ja"}.issubset(LANG_ALIASES)
    assert ROUTES["a"] == "en-US" and ROUTES["z"] == "zh" and ROUTES["j"] == "ja"
    assert VOICES


def test_profile_and_duration_routing_is_bounded():
    profile = select_profile(160, platform="rk3576")
    assert profile["T"] in (320, 368, 416, 480, 576, 640)
    durations = snap_durations(np.array([3, 4, 5], dtype=np.int64), profile["T"])
    assert durations.dtype == np.int64 and int(durations.sum()) == profile["T"]


def test_route_parser_rejects_ambiguous_input():
    assert parse_route_ts("320,368", platform="rk3576") == (320, 368)
    with pytest.raises(ValueError):
        parse_route_ts("368,320", platform="rk3576")

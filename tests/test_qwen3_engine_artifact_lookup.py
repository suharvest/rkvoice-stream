from pathlib import Path

import pytest

from rkvoice_stream.backends.asr.qwen3.engine import Qwen3ASREngine


def test_rkllm_lookup_falls_back_to_rkllm_dir_when_decoder_has_matmul_only(tmp_path):
    model_dir = tmp_path / "models"
    (model_dir / "decoder" / "matmul_w8a16").mkdir(parents=True)
    rkllm_dir = model_dir / "rkllm"
    rkllm_dir.mkdir()
    expected = rkllm_dir / "decoder_qwen3.w4a16_g128.rk3576.rkllm"
    expected.write_bytes(b"stub")

    actual = Qwen3ASREngine._find_rkllm_decoder_model(
        model_dir,
        decoder_quant="w4a16_g128",
        platform="rk3576",
    )

    assert actual == expected


def test_rkllm_lookup_prefers_decoder_dir_when_matching_artifact_exists(tmp_path):
    model_dir = tmp_path / "models"
    decoder_dir = model_dir / "decoder"
    rkllm_dir = model_dir / "rkllm"
    decoder_dir.mkdir(parents=True)
    rkllm_dir.mkdir()
    expected = decoder_dir / "decoder_qwen3.fp16.rk3588.rkllm"
    expected.write_bytes(b"stub")
    (rkllm_dir / "decoder_qwen3.fp16.rk3588.rkllm").write_bytes(b"stub")

    actual = Qwen3ASREngine._find_rkllm_decoder_model(
        model_dir,
        decoder_quant="fp16",
        platform="rk3588",
    )

    assert actual == expected


def test_rkllm_lookup_error_mentions_both_layouts(tmp_path):
    model_dir = tmp_path / "models"
    (model_dir / "decoder").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc:
        Qwen3ASREngine._find_rkllm_decoder_model(
            model_dir,
            decoder_quant="w8a8",
            platform="rk3576",
        )

    message = str(exc.value)
    assert str(model_dir / "decoder") in message
    assert str(model_dir / "rkllm") in message


def test_rkllm_lookup_uses_exact_quant_token(tmp_path):
    model_dir = tmp_path / "models"
    rkllm_dir = model_dir / "rkllm"
    rkllm_dir.mkdir(parents=True)
    g128 = rkllm_dir / "decoder_qwen3.w4a16_g128.rk3576.rkllm"
    exact = rkllm_dir / "decoder_hf.w4a16.rk3576.rkllm"
    g128.write_bytes(b"stub")
    exact.write_bytes(b"stub")

    actual = Qwen3ASREngine._find_rkllm_decoder_model(
        model_dir,
        decoder_quant="w4a16",
        platform="rk3576",
    )

    assert actual == exact


def test_rkllm_lookup_w8a8_does_not_match_w8a8_g128(tmp_path):
    model_dir = tmp_path / "models"
    decoder_dir = model_dir / "decoder"
    decoder_dir.mkdir(parents=True)
    g128 = decoder_dir / "decoder_qwen3.w8a8_g128.rk3588.rkllm"
    g128.write_bytes(b"stub")

    with pytest.raises(FileNotFoundError):
        Qwen3ASREngine._find_rkllm_decoder_model(
            model_dir,
            decoder_quant="w8a8",
            platform="rk3588",
        )

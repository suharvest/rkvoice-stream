from __future__ import annotations

from pathlib import Path

import numpy as np

from models.tts.moss.probe_moss_rknn_runtime import (
    _case_from_name,
    _named_inputs_for_case,
    _inputs_for_case,
    _pass_through,
    _runtime_input_count,
    _stderr_runtime_error,
)


def test_probe_infers_moss_cases_from_filename():
    assert _case_from_name(Path("min_layernorm.fp16.rk3576.rknn"), "auto") == "layernorm"
    assert _case_from_name(Path("moss_block0_mlp.s32.fp16.rk3576.rknn"), "auto") == "island_float"
    assert _case_from_name(Path("moss_block11_ln2_mlp.s320.fp16.rk3576.rknn"), "auto") == "island_float"
    assert _case_from_name(Path("moss_block0_fc_out.s320.fp16.rk3576.rknn"), "auto") == "island_float"
    assert _case_from_name(Path("moss_block0_attn_residual.s320.fp16.rk3576.rknn"), "auto") == "attn_residual"
    assert _case_from_name(Path("moss_tts_prefill.s32.crop_Add_15_output_0.fp16.rk3576.rknn"), "auto") == "prefill_tokens"
    assert _case_from_name(Path("moss_tts_prefill.s32.suffix_from_Add15.fp16.rk3576.rknn"), "auto") == "prefill_hidden_mask"
    assert _case_from_name(Path("moss_tts_prefill.s32.fp16.rk3576.rknn"), "auto") == "prefill"
    assert _case_from_name(Path("moss_tts_decode_step.p128.fp16.rk3576.rknn"), "auto") == "decode"
    assert _case_from_name(Path("moss_tts_local_fixed_sampled_frame.fp16.rk3576.rknn"), "auto") == "sampler"
    assert _case_from_name(Path("moss_sampler_text_lm_head.fp16.rk3576.rknn"), "auto") == "sampler_island_float"
    assert _case_from_name(Path("moss_sampler_mlp0.fp16.rk3576.rknn"), "auto") == "sampler_island_float"
    assert _case_from_name(Path("moss_sampler_mlps0.fp16.rk3576.rknn"), "auto") == "sampler_mlps"
    assert _case_from_name(Path("moss_sampler_audio_heads0.fp16.rk3576.rknn"), "auto") == "sampler_audio_heads"
    assert _case_from_name(Path("codec_suffix_layer0_outproj_ffn.fp16.rk3576.rknn"), "auto") == "codec_suffix_outproj_ffn"
    assert _case_from_name(Path("codec_suffix_layer11_outproj_ffn.fp16.rk3576.rknn"), "auto") == "codec_suffix_outproj_ffn"
    assert _case_from_name(Path("codec_decode_step.f4.fp16.rk3576.rknn"), "auto") == "codec"


def test_probe_prefill_inputs_match_fixed_bucket_contract():
    inputs = _inputs_for_case("prefill", Path("moss_tts_prefill.s64.fp16.rk3576.rknn"))

    assert [x.shape for x in inputs] == [(1, 64, 17), (1, 64)]
    assert [x.dtype for x in inputs] == [np.dtype("int32"), np.dtype("int32")]
    assert np.all(inputs[0][:, :, 0] == 1)
    assert np.all(inputs[1] == 1)


def test_probe_decode_inputs_include_all_kv_pairs():
    inputs = _inputs_for_case("decode", Path("moss_tts_decode_step.p32.fp16.rk3576.rknn"))

    assert len(inputs) == 26
    assert inputs[0].shape == (1, 1, 17)
    assert inputs[1].tolist() == [32]
    assert all(x.shape == (1, 32, 12, 64) for x in inputs[2:])


def test_probe_prefill_crop_and_suffix_inputs_match_split_contracts():
    crop_inputs = _inputs_for_case("prefill_tokens", Path("moss_tts_prefill.s32.crop_Add_15_output_0.fp16.rk3576.rknn"))
    suffix_inputs = _inputs_for_case("prefill_hidden", Path("moss_tts_prefill.s32.suffix_from_Add15.fp16.rk3576.rknn"))
    suffix_mask_inputs = _inputs_for_case("prefill_hidden_mask", Path("moss_tts_prefill.s32.suffix_from_Add15.fp16.rk3576.rknn"))

    assert [x.shape for x in crop_inputs] == [(1, 32, 17)]
    assert crop_inputs[0].dtype == np.dtype("int32")
    assert [x.shape for x in suffix_inputs] == [(1, 32, 768)]
    assert suffix_inputs[0].dtype == np.dtype("float32")
    assert [x.shape for x in suffix_mask_inputs] == [(1, 32, 768), (1, 32)]
    assert [x.dtype for x in suffix_mask_inputs] == [np.dtype("float32"), np.dtype("int32")]


def test_probe_island_float_input_matches_seq_bucket():
    inputs = _inputs_for_case("island_float", Path("moss_block0_mlp.s64.fp16.rk3576.rknn"))

    assert [x.shape for x in inputs] == [(1, 64, 768)]
    assert inputs[0].dtype == np.dtype("float32")


def test_probe_fc_out_input_uses_mlp_expanded_width():
    inputs = _inputs_for_case("island_float", Path("moss_block0_fc_out.s320.fp16.rk3576.rknn"))

    assert [x.shape for x in inputs] == [(1, 320, 3072)]
    assert inputs[0].dtype == np.dtype("float32")


def test_probe_sampler_island_inputs_match_sampler_contracts():
    text = _inputs_for_case("sampler_island_float", Path("moss_sampler_text_lm_head.fp16.rk3576.rknn"))
    fc_out = _inputs_for_case("sampler_island_float", Path("moss_sampler_fc_out0.fp16.rk3576.rknn"))
    mlp = _inputs_for_case("sampler_island_float", Path("moss_sampler_mlp0.fp16.rk3576.rknn"))
    mlps = _inputs_for_case("sampler_mlps", Path("moss_sampler_mlps0.fp16.rk3576.rknn"))
    audio_heads = _inputs_for_case("sampler_audio_heads", Path("moss_sampler_audio_heads0.fp16.rk3576.rknn"))

    assert [x.shape for x in text] == [(1, 768)]
    assert [x.shape for x in fc_out] == [(1, 1, 3072)]
    assert [x.shape for x in mlp] == [(1, 1, 768)]
    assert [x.shape for x in mlps] == [(1, 1, 768)] * 17
    assert [x.shape for x in audio_heads] == [(1, 768)] * 16
    assert all(x.dtype == np.dtype("float32") for x in [*text, *fc_out, *mlp, *mlps, *audio_heads])


def test_probe_codec_input_includes_streaming_cache_state():
    inputs = _inputs_for_case("codec", Path("codec_decode_step.f1.fp16.rk3576.rknn"))

    assert len(inputs) == 54
    assert inputs[0].shape == (1, 1, 16)
    assert inputs[1].shape == (1,)
    assert [x.shape for x in inputs[2:6]] == [(1,)] * 4
    assert inputs[6].shape == (1,)
    assert inputs[7].shape == (1, 4, 500, 64)
    assert inputs[8].shape == (1, 4, 500, 64)
    assert inputs[9].shape == (1, 500)
    assert inputs[-4].shape == (1,)
    assert inputs[-3].shape == (1, 4, 1600, 64)
    assert inputs[-2].shape == (1, 4, 1600, 64)
    assert inputs[-1].shape == (1, 1600)
    assert inputs[0].dtype == np.dtype("int32")
    assert inputs[7].dtype == np.dtype("float32")


def test_probe_codec_named_inputs_match_runtime_tensor_names():
    inputs = _named_inputs_for_case("codec", Path("codec_decode_step.f1.fp16.rk3576.rknn"))

    assert inputs["audio_codes"].shape == (1, 1, 16)
    assert inputs["audio_code_lengths"].shape == (1,)
    assert inputs["transformer_offset_3"].shape == (1,)
    assert inputs["attn_offset_0"].shape == (1,)
    assert inputs["attn_cached_keys_0"].shape == (1, 4, 500, 64)
    assert inputs["attn_cached_values_11"].shape == (1, 4, 1600, 64)
    assert inputs["attn_cached_positions_11"].shape == (1, 1600)
    assert len(inputs) == 54


def test_probe_codec_int64input_uses_int64_for_cast_fixed_inputs():
    inputs = _named_inputs_for_case("codec", Path("codec_decode_step.f1.int64input.fp16.rk3576.rknn"))

    assert inputs["audio_codes"].dtype == np.dtype("int64")
    assert inputs["audio_code_lengths"].dtype == np.dtype("int64")
    assert inputs["attn_offset_0"].dtype == np.dtype("int32")


def test_probe_codec_int64offset_uses_int64_for_offset_and_position_inputs():
    inputs = _named_inputs_for_case("codec", Path("codec_decode_step.f1.int64offset.fp16.rk3576.rknn"))

    assert inputs["audio_codes"].dtype == np.dtype("int64")
    assert inputs["audio_code_lengths"].dtype == np.dtype("int64")
    assert inputs["transformer_offset_0"].dtype == np.dtype("int64")
    assert inputs["attn_offset_0"].dtype == np.dtype("int64")
    assert inputs["attn_cached_positions_0"].dtype == np.dtype("int64")
    assert inputs["attn_cached_keys_0"].dtype == np.dtype("float32")


def test_probe_codec_suffix_outproj_ffn_uses_two_float_inputs():
    first = _inputs_for_case("codec_suffix_outproj_ffn", Path("codec_suffix_layer0_outproj_ffn.fp16.rknn"))
    mid = _inputs_for_case("codec_suffix_outproj_ffn", Path("codec_suffix_layer6_outproj_ffn.fp16.rknn"))
    late = _inputs_for_case("codec_suffix_outproj_ffn", Path("codec_suffix_layer11_outproj_ffn.fp16.rknn"))

    assert [x.shape for x in first] == [(1, 4, 256), (1, 4, 256)]
    assert [x.shape for x in mid] == [(1, 16, 256), (1, 16, 256)]
    assert [x.shape for x in late] == [(1, 32, 256), (1, 32, 256)]
    assert [x.dtype for x in late] == [np.dtype("float32"), np.dtype("float32")]


def test_probe_attention_residual_inputs_include_mask():
    inputs = _inputs_for_case("attn_residual", Path("moss_block0_attn_residual.s320.fp16.rk3576.rknn"))

    assert [x.shape for x in inputs] == [(1, 320, 768), (1, 320)]
    assert [x.dtype for x in inputs] == [np.dtype("float32"), np.dtype("int32")]
    assert np.all(inputs[1] == 1)


def test_probe_pass_through_auto_only_marks_integer_inputs():
    inputs = [
        np.zeros((1, 4), dtype=np.int32),
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    ]

    assert _pass_through(inputs, "auto") == [1, 0, 1]
    assert _pass_through(inputs, "none") is None
    assert _pass_through(inputs, "all") == [1, 1, 1]


def test_probe_pass_through_auto_omits_all_float_inputs():
    inputs = [np.zeros((1, 32, 768), dtype=np.float32)]

    assert _pass_through(inputs, "auto") is None


def test_probe_treats_rknn_runtime_stderr_errors_as_failures():
    stderr = "E RKNN: input dtype is undefine!\nE RKNN: failed to submit!, op id: 4"

    assert _stderr_runtime_error(stderr) == "failed to submit"
    assert _stderr_runtime_error("W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID.") is None
    assert _stderr_runtime_error("W Query dynamic range failed. static shape RKNN model") is None


def test_probe_child_marks_inference_before_call():
    source = Path("models/tts/moss/probe_moss_rknn_runtime.py").read_text(encoding="utf-8")

    assert 'result["phase"] = "inference"' in source
    assert source.index('result["phase"] = "inference"') < source.index("rknn.inference(inputs=inputs")


def test_runtime_input_count_stops_at_first_missing_input_attr():
    class Runtime:
        def get_tensor_attr(self, index, is_output):
            assert is_output is False
            if index >= 3:
                raise RuntimeError("missing")
            return object()

    class RKNN:
        rknn_runtime = Runtime()

    assert _runtime_input_count(RKNN(), 54) == 3

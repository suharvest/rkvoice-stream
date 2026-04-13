#!/usr/bin/env python3
"""Re-export Qwen3-TTS sub-models with FIXED shapes for RKNN conversion.

Loads the original Qwen3-TTS model, creates the same wrapper classes as
export-onnx.py, and re-traces with torch.onnx.export WITHOUT dynamic_axes.

Fixed shapes:
  text_project:                input_ids [1, 128]
  codec_embed:                 token_ids [1, 1]
  code_predictor_embed:        token_id [1, 1], layer_idx scalar  (already static)
  code_predictor:              context [1, 2, 1024]
  talker_prefill:              inputs_embeds [1, 32, 1024], attention_mask [1, 32]
  talker_decode:               inputs_embeds [1, 1, 1024], attention_mask [1, 513],
                               past_KV [1, 8, 512, 128] x 56
  speaker_encoder:             mel [1, 300, 128]
  tokenizer12hz_encode:        audio [1, 72000]
  tokenizer12hz_decode_stream: audio_codes [1, 75, 16]  (context=50 + chunk=25)

Output: ~/qwen3-tts-export/qwen3-tts-0.6b-12hz-fixed/
"""

import functools
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# Force legacy TorchScript ONNX exporter
if hasattr(torch.onnx.export, "__wrapped__") or tuple(
    int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
) >= (2, 5):
    _orig_onnx_export = torch.onnx.export

    @functools.wraps(_orig_onnx_export)
    def _patched_onnx_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig_onnx_export(*args, **kwargs)

    torch.onnx.export = _patched_onnx_export

from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

OUTPUT_DIR = os.path.expanduser("~/qwen3-tts-export/qwen3-tts-0.6b-12hz-fixed")
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
OPSET = 14
DEVICE = "cpu"

# Fixed shape parameters
PREFILL_T = 32       # prefill sequence length
DECODE_PAST = 512    # decode past KV length
TEXT_T = 128         # text projection sequence length
SPEAKER_TIME = 300   # mel spectrogram time frames (~3s)
ENCODE_SAMPLES = 72000  # 3s at 24kHz
STREAM_CHUNK = 25    # streaming chunk frames
STREAM_CONTEXT = 50  # streaming context frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = {}


def report(name, status, detail=""):
    results[name] = (status, detail)
    print(f"  [{status}] {name}: {detail}")


def export_text_project(model):
    print("\n=== text_project (fixed T={}) ===".format(TEXT_T))
    t0 = time.time()

    class TextProject(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.text_embed = talker.model.text_embedding
            self.text_projection = talker.text_projection

        def forward(self, input_ids):
            return self.text_projection(self.text_embed(input_ids))

    wrapper = TextProject(model.talker)
    wrapper.eval()
    dummy = torch.tensor([[1] * TEXT_T], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper, (dummy,),
        os.path.join(OUTPUT_DIR, "text_project.onnx"),
        input_names=["input_ids"],
        output_names=["text_embed"],
        opset_version=OPSET,
    )
    report("text_project", "OK", f"[1,{TEXT_T}] -> [1,{TEXT_T},1024] ({time.time()-t0:.1f}s)")


def export_codec_embed(model):
    print("\n=== codec_embed (fixed [1,1]) ===")
    t0 = time.time()

    class CodecEmbed(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.embed_tokens = talker.model.codec_embedding

        def forward(self, token_ids):
            return self.embed_tokens(token_ids)

    wrapper = CodecEmbed(model.talker)
    wrapper.eval()
    dummy = torch.tensor([[100]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper, (dummy,),
        os.path.join(OUTPUT_DIR, "codec_embed.onnx"),
        input_names=["token_ids"],
        output_names=["embed"],
        opset_version=OPSET,
    )
    report("codec_embed", "OK", f"[1,1] -> [1,1,1024] ({time.time()-t0:.1f}s)")


def export_code_predictor_embed(model):
    print("\n=== code_predictor_embed (static) ===")
    t0 = time.time()

    num_groups = model.talker.config.num_code_groups
    code_predictor = model.talker.code_predictor

    class CodePredictorEmbed(nn.Module):
        def __init__(self, embed_layers):
            super().__init__()
            self.embed_layers = nn.ModuleList(embed_layers)

        def forward(self, token_id, layer_idx):
            embeds = []
            for layer in self.embed_layers:
                embeds.append(layer(token_id))
            stacked = torch.stack(embeds, dim=0)
            return stacked[layer_idx]

    embed_layers = list(code_predictor.get_input_embeddings())
    wrapper = CodePredictorEmbed(embed_layers)
    wrapper.eval()

    dummy_token = torch.tensor([[100]], dtype=torch.long, device=model.device)
    dummy_layer = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper, (dummy_token, dummy_layer),
        os.path.join(OUTPUT_DIR, "code_predictor_embed.onnx"),
        input_names=["token_id", "layer_idx"],
        output_names=["embed"],
        opset_version=OPSET,
    )
    report("code_predictor_embed", "OK", f"[1,1]+scalar -> [1,1,1024] ({time.time()-t0:.1f}s)")


def export_code_predictor(model):
    print("\n=== code_predictor (fixed ctx=2) ===")
    t0 = time.time()

    code_predictor = model.talker.code_predictor

    class CodePredictor(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, context, gen_step):
            out = self.predictor(
                inputs_embeds=context, use_cache=False, return_dict=True,
            )
            logits = out.logits
            return logits[:, -1:, :]

    wrapper = CodePredictor(code_predictor)
    wrapper.eval()

    D = model.talker.config.hidden_size
    dummy_ctx = torch.randn(1, 2, D, device=model.device)
    dummy_step = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper, (dummy_ctx, dummy_step),
        os.path.join(OUTPUT_DIR, "code_predictor.onnx"),
        input_names=["context", "gen_step"],
        output_names=["logits"],
        opset_version=OPSET,
    )
    report("code_predictor", "OK", f"[1,2,1024] -> [1,1,2048] ({time.time()-t0:.1f}s)")


def export_talker_prefill(model):
    T = PREFILL_T
    print(f"\n=== talker_prefill (fixed T={T}) ===")
    t0 = time.time()

    talker = model.talker
    num_layers = talker.config.num_hidden_layers
    D = talker.config.hidden_size
    num_kv_heads = talker.config.num_key_value_heads
    head_dim = getattr(talker.config, "head_dim", D // talker.config.num_attention_heads)

    class TalkerPrefill(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.talker = talker

        def forward(self, inputs_embeds, attention_mask):
            out = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = self.talker.codec_head(hidden[:, -1:, :])
            pkv = out.past_key_values
            return (logits, hidden) + tuple(
                t for i in range(len(pkv)) for t in pkv[i]
            )

    wrapper = TalkerPrefill(talker)
    wrapper.eval()

    dummy_embeds = torch.randn(1, T, D, device=model.device)
    dummy_mask = torch.ones(1, T, dtype=torch.long, device=model.device)

    kv_names = []
    for i in range(num_layers):
        kv_names += [f"past_key_{i}", f"past_value_{i}"]

    # NO dynamic_axes -> all shapes are fixed at trace time
    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask),
        os.path.join(OUTPUT_DIR, "talker_prefill.onnx"),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits", "last_hidden"] + kv_names,
        opset_version=OPSET,
    )
    elapsed = time.time() - t0
    report("talker_prefill", "OK",
           f"[1,{T},1024],[1,{T}] -> logits+hidden+{len(kv_names)} KV [{1},{num_kv_heads},{T},{head_dim}] ({elapsed:.1f}s)")


def export_talker_decode(model):
    T_PAST = DECODE_PAST
    print(f"\n=== talker_decode (fixed past_len={T_PAST}) ===")
    t0 = time.time()

    talker = model.talker
    num_layers = talker.config.num_hidden_layers
    D = talker.config.hidden_size
    num_kv_heads = talker.config.num_key_value_heads
    head_dim = getattr(talker.config, "head_dim", D // talker.config.num_attention_heads)

    class TalkerDecode(nn.Module):
        def __init__(self, talker, num_layers):
            super().__init__()
            self.talker = talker
            self.num_layers = num_layers

        def forward(self, inputs_embeds, attention_mask, *past_kv_flat):
            from transformers.cache_utils import DynamicCache
            legacy_tuples = tuple(
                (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
                for i in range(self.num_layers)
            )
            past_key_values = DynamicCache.from_legacy_cache(legacy_tuples)
            out = self.talker.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            hidden = out.last_hidden_state
            logits = self.talker.codec_head(hidden)
            new_pkv = out.past_key_values
            return (logits, hidden) + tuple(
                t for i in range(len(new_pkv)) for t in new_pkv[i]
            )

    wrapper = TalkerDecode(talker, num_layers)
    wrapper.eval()

    dummy_embeds = torch.randn(1, 1, D, device=model.device)
    dummy_mask = torch.ones(1, T_PAST + 1, dtype=torch.long, device=model.device)
    dummy_pkv = [
        torch.randn(1, num_kv_heads, T_PAST, head_dim, device=model.device)
        for _ in range(num_layers * 2)
    ]

    in_kv_names = []
    out_kv_names = []
    for i in range(num_layers):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    # NO dynamic_axes
    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask, *dummy_pkv),
        os.path.join(OUTPUT_DIR, "talker_decode.onnx"),
        input_names=["inputs_embeds", "attention_mask"] + in_kv_names,
        output_names=["logits", "last_hidden"] + out_kv_names,
        opset_version=OPSET,
    )
    elapsed = time.time() - t0
    report("talker_decode", "OK",
           f"[1,1,1024],[1,{T_PAST+1}]+{len(in_kv_names)} KV [{1},{num_kv_heads},{T_PAST},{head_dim}] ({elapsed:.1f}s)")


def export_speaker_encoder(model):
    T = SPEAKER_TIME
    print(f"\n=== speaker_encoder (fixed time={T}) ===")
    t0 = time.time()

    if model.speaker_encoder is None:
        report("speaker_encoder", "SKIP", "model has no speaker_encoder")
        return

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.encoder = speaker_encoder

        def forward(self, mel):
            return self.encoder(mel)[0]

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder)
    wrapper.eval()

    dummy_mel = torch.randn(1, T, 128, device=model.device, dtype=model.dtype)

    torch.onnx.export(
        wrapper, (dummy_mel,),
        os.path.join(OUTPUT_DIR, "speaker_encoder.onnx"),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        opset_version=OPSET,
    )
    report("speaker_encoder", "OK", f"[1,{T},128] -> [1024] ({time.time()-t0:.1f}s)")


def _register_diff_symbolic():
    def _diff_symbolic(g, x, n, dim, prepend, append):
        from torch.onnx.symbolic_helper import _get_const
        dim_val = _get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim_val], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        big = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))
        a = g.op("Slice", x, zero, neg1, axes, one)
        b = g.op("Slice", x, one, big, axes, one)
        diff_result = g.op("Sub", b, a)
        first = g.op("Slice", x, zero, one, axes, one)
        zero_pad = g.op("Sub", first, first)
        return g.op("Concat", zero_pad, diff_result, axis_i=dim_val)

    torch.onnx.register_custom_op_symbolic("aten::diff", _diff_symbolic, 18)


def _fix_bool_cumsum(onnx_model):
    import onnx
    name_to_node = {o: node for node in onnx_model.graph.node for o in node.output}
    cast_added = 0
    for i, node in enumerate(list(onnx_model.graph.node)):
        if node.op_type == "CumSum":
            data_input = node.input[0]
            src = name_to_node.get(data_input)
            if src and src.op_type in ("Not", "Equal", "Less", "Greater", "And", "Or"):
                cast_name = data_input + "_i64"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[data_input], outputs=[cast_name], to=7
                )
                node.input[0] = cast_name
                onnx_model.graph.node.insert(i, cast_node)
                cast_added += 1
    return cast_added


def export_tokenizer12hz_encode(model):
    print(f"\n=== tokenizer12hz_encode (fixed samples={ENCODE_SAMPLES}) ===")
    t0 = time.time()

    tokenizer_model = model.speech_tokenizer.model

    class TokenizerEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            return self.model.encode(audio)

    wrapper = TokenizerEncoder(tokenizer_model)
    wrapper.eval()

    dummy_audio = torch.randn(1, ENCODE_SAMPLES, device=model.device)

    torch.onnx.export(
        wrapper, (dummy_audio,),
        os.path.join(OUTPUT_DIR, "tokenizer12hz_encode.onnx"),
        input_names=["audio"],
        output_names=["codes"],
        opset_version=OPSET,
    )
    frames = ENCODE_SAMPLES // 1920
    report("tokenizer12hz_encode", "OK",
           f"[1,{ENCODE_SAMPLES}] -> [1,{frames},16] ({time.time()-t0:.1f}s)")


def export_tokenizer12hz_decode_stream(model):
    import onnx

    trace_T = STREAM_CONTEXT + STREAM_CHUNK
    print(f"\n=== tokenizer12hz_decode_stream (fixed T={trace_T}, chunk={STREAM_CHUNK}, ctx={STREAM_CONTEXT}) ===")
    t0 = time.time()

    _register_diff_symbolic()

    speech_model = model.speech_tokenizer.model
    decode_upsample_rate = speech_model.decode_upsample_rate

    class DecoderForward(nn.Module):
        def __init__(self, decoder, upsample_rate):
            super().__init__()
            self.decoder = decoder
            self.upsample_rate = upsample_rate

        def forward(self, audio_codes):
            wav = self.decoder(audio_codes.transpose(1, 2))
            audio_values = wav.squeeze(1)
            lengths = (audio_codes[..., 0] >= 0).sum(dim=1) * self.upsample_rate
            return audio_values, lengths

    wrapper = DecoderForward(speech_model.decoder, decode_upsample_rate)
    wrapper.eval()

    dummy_codes = torch.randint(0, 1024, (1, trace_T, 16), device=model.device)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper, (dummy_codes,),
        buf,
        input_names=["audio_codes"],
        output_names=["audio_values", "lengths"],
        opset_version=18,
        do_constant_folding=False,
    )
    buf.seek(0)

    onnx_model = onnx.load_model_from_string(buf.getvalue())
    n_casts = _fix_bool_cumsum(onnx_model)
    print(f"  Post-processing: inserted {n_casts} Cast(INT64) before CumSum")

    out_path = os.path.join(OUTPUT_DIR, "tokenizer12hz_decode_stream.onnx")
    onnx.save(onnx_model, out_path)
    report("tokenizer12hz_decode_stream", "OK",
           f"[1,{trace_T},16] -> [1,{trace_T*1920}]+[1] ({time.time()-t0:.1f}s)")


def main():
    print("=" * 60)
    print("Qwen3-TTS Fixed-Shape ONNX Export")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL_NAME} ...")
    t0 = time.time()

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Export all models with fixed shapes
    export_text_project(model)
    export_codec_embed(model)
    export_code_predictor_embed(model)
    export_code_predictor(model)
    export_speaker_encoder(model)
    export_talker_prefill(model)
    export_talker_decode(model)
    export_tokenizer12hz_encode(model)
    export_tokenizer12hz_decode_stream(model)

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    for name, (status, detail) in results.items():
        print(f"  {name:35s}  {status:6s}  {detail}")

    print(f"\nOutput files:")
    total = 0
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".onnx"):
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            total += sz
            print(f"  {f:45s}  {sz/(1024*1024):8.1f} MB")
    print(f"  {'TOTAL':45s}  {total/(1024*1024):8.1f} MB")


if __name__ == "__main__":
    main()

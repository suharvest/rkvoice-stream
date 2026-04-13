#!/usr/bin/env python3
# Copyright (c)  2026  Xiaomi Corporation
#
# Export Qwen3-TTS-12Hz-0.6B-Base to ONNX models for sherpa-onnx.
#
# Usage:
#   pip install qwen-tts torch onnx onnxruntime
#   python3 export-onnx.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --output-dir ./qwen3-tts-0.6b-12hz
#
# This script exports 9 ONNX sub-models:
#   1. text_project.onnx              - text token IDs -> text embeddings
#   2. codec_embed.onnx               - codec token ID -> codec embedding
#   3. code_predictor_embed.onnx      - residual code IDs -> embeddings (15 layers)
#   4. code_predictor.onnx            - sub-talker for residual codebook prediction
#   5. talker_prefill.onnx            - talker prefill (full context -> KV-cache)
#   6. talker_decode.onnx             - talker decode (one token -> next token + KV-cache)
#   7. speaker_encoder.onnx           - mel spectrogram -> speaker embedding
#   8. tokenizer12hz_encode.onnx      - audio waveform -> codec tokens
#   9. tokenizer12hz_decode.onnx      - codec tokens -> audio waveform (batch)
#  10. tokenizer12hz_decode_stream.onnx - same decoder traced with T=--streaming-chunk-frames
#                                       for fast per-chunk streaming decode
#
# Streaming decoder rationale
# ---------------------------
# The community pre-exported tokenizer12hz_decode*.onnx was traced with
# max_codes_length=1024 baked in, so every call computes ~82 s of audio
# regardless of actual input length (~16 s on M-series CPU).  Calling it
# per streaming chunk would be 10-40× slower than batch.
#
# tokenizer12hz_decode_stream.onnx is traced with T=--streaming-chunk-frames
# (default 25, i.e. 2 s of audio at 12.5 Hz).  Because the codec decoder is
# a stack of ConvTranspose1d layers (no attention), the ONNX graph size scales
# with the trace N, so the per-call cost is proportionally smaller:
#   batch  (T=1024): ~16 s → RTF 0.2  (good for batch)
#   stream (T=25):   ~0.4 s → RTF 5   (real-time first chunk)
#
# In sherpa-onnx, set cfg.extra["chunk_frames"] = "25" to use streaming mode
# with this model.

import argparse
import functools
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

# Force legacy TorchScript ONNX exporter (PyTorch 2.5+ defaults to dynamo
# which fails on data-dependent shapes like stacked[layer_idx]).
if hasattr(torch.onnx.export, "__wrapped__") or tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5):
    _orig_onnx_export = torch.onnx.export

    @functools.wraps(_orig_onnx_export)
    def _patched_onnx_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig_onnx_export(*args, **kwargs)

    torch.onnx.export = _patched_onnx_export

# Register Qwen3TTS model classes
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def export_text_project(model, output_dir, opset_version=14):
    """Export text_projection + text_embed_tokens as a single model.

    Input:  input_ids [1, T] int64  - text token IDs
    Output: text_embed [1, T, D] float32  - projected text embeddings
    """
    print("Exporting text_project.onnx ...")

    class TextProject(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.text_embed = talker.model.text_embedding
            self.text_projection = talker.text_projection

        def forward(self, input_ids):
            return self.text_projection(self.text_embed(input_ids))

    wrapper = TextProject(model.talker)
    wrapper.eval()

    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_input_ids,),
        os.path.join(output_dir, "text_project.onnx"),
        input_names=["input_ids"],
        output_names=["text_embed"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "text_embed": {1: "seq_len"},
        },
        opset_version=opset_version,
    )
    print("  Done: text_project.onnx")


def export_codec_embed(model, output_dir, opset_version=14):
    """Export codec token embedding layer.

    Input:  token_ids [1, T] int64  - codec token IDs
    Output: embed [1, T, D] float32  - codec embeddings
    """
    print("Exporting codec_embed.onnx ...")

    class CodecEmbed(nn.Module):
        def __init__(self, talker):
            super().__init__()
            self.embed_tokens = talker.model.codec_embedding

        def forward(self, token_ids):
            return self.embed_tokens(token_ids)

    wrapper = CodecEmbed(model.talker)
    wrapper.eval()

    dummy_ids = torch.tensor([[100]], dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_ids,),
        os.path.join(output_dir, "codec_embed.onnx"),
        input_names=["token_ids"],
        output_names=["embed"],
        dynamic_axes={
            "token_ids": {1: "seq_len"},
            "embed": {1: "seq_len"},
        },
        opset_version=opset_version,
    )
    print("  Done: codec_embed.onnx")


def export_code_predictor_embed(model, output_dir, opset_version=14):
    """Export code predictor embedding layers (15 layers for residual codebooks).

    Input:  token_id int64 scalar - residual codebook token ID
            layer_idx int64 scalar - which residual layer (0-14)
    Output: embed [1, 1, D] float32 - embedding
    """
    print("Exporting code_predictor_embed.onnx ...")

    num_groups = model.talker.config.num_code_groups  # typically 16
    code_predictor = model.talker.code_predictor

    # Export each embedding layer separately as a combined model
    class CodePredictorEmbed(nn.Module):
        def __init__(self, embed_layers):
            super().__init__()
            self.embed_layers = nn.ModuleList(embed_layers)

        def forward(self, token_id, layer_idx):
            # For ONNX, we'll use a simple approach: export all layers
            # and use gather at runtime
            embeds = []
            for layer in self.embed_layers:
                embeds.append(layer(token_id))
            stacked = torch.stack(embeds, dim=0)  # [num_layers, 1, 1, D]
            return stacked[layer_idx]  # [1, 1, D]

    embed_layers = list(code_predictor.get_input_embeddings())
    wrapper = CodePredictorEmbed(embed_layers)
    wrapper.eval()

    dummy_token = torch.tensor([[100]], dtype=torch.long, device=model.device)
    dummy_layer = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_token, dummy_layer),
        os.path.join(output_dir, "code_predictor_embed.onnx"),
        input_names=["token_id", "layer_idx"],
        output_names=["embed"],
        opset_version=opset_version,
    )
    print(f"  Done: code_predictor_embed.onnx ({num_groups - 1} layers)")


def export_code_predictor(model, output_dir, opset_version=14):
    """Export code predictor (sub-talker for residual codebook prediction).

    Input:  context [1, T, D] float32 - context embeddings (last_hidden + residual embeds)
            gen_step int64 scalar     - which residual group (0 to num_code_groups-2)
    Output: logits  [1, 1, vocab] float32 - next residual code logits
    """
    print("Exporting code_predictor.onnx ...")

    code_predictor = model.talker.code_predictor

    class CodePredictor(nn.Module):
        def __init__(self, predictor):
            super().__init__()
            self.predictor = predictor

        def forward(self, context, gen_step):
            # context: [1, T, D], gen_step: scalar
            out = self.predictor(
                inputs_embeds=context,
                use_cache=False,
                return_dict=True,
            )
            logits = out.logits  # [1, T, vocab]
            return logits[:, -1:, :]  # [1, 1, vocab]

    wrapper = CodePredictor(code_predictor)
    wrapper.eval()

    D = model.talker.config.hidden_size
    dummy_ctx = torch.randn(1, 2, D, device=model.device)
    dummy_step = torch.tensor(0, dtype=torch.long, device=model.device)

    torch.onnx.export(
        wrapper,
        (dummy_ctx, dummy_step),
        os.path.join(output_dir, "code_predictor.onnx"),
        input_names=["context", "gen_step"],
        output_names=["logits"],
        dynamic_axes={
            "context": {1: "ctx_len"},
        },
        opset_version=opset_version,
    )
    print("  Done: code_predictor.onnx")


def export_talker_prefill(model, output_dir, opset_version=14):
    """Export talker prefill: full sequence -> KV-cache + logits.

    Input:  inputs_embeds [1, T, D] float32 - prefill embeddings
            attention_mask [1, T] int64      - attention mask
    Output: logits      [1, 1, V] float32   - last-token logits
            last_hidden [1, T, D] float32   - all hidden states (for code predictor)
            past_key_values_*               - KV-cache tensors (2 per layer)
    """
    print("Exporting talker_prefill.onnx ...")

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
            hidden = out.last_hidden_state  # [1, T, D]
            logits = self.talker.codec_head(hidden[:, -1:, :])  # [1, 1, V]
            # past_key_values: DynamicCache; pkv[i] -> (key, value) per layer
            # each: [1, num_kv_heads, T, head_dim]
            pkv = out.past_key_values
            return (logits, hidden) + tuple(
                t for i in range(len(pkv)) for t in pkv[i]
            )

    wrapper = TalkerPrefill(talker)
    wrapper.eval()

    T = 8  # typical prefill length
    dummy_embeds = torch.randn(1, T, D, device=model.device)
    dummy_mask = torch.ones(1, T, dtype=torch.long, device=model.device)

    kv_names = []
    for i in range(num_layers):
        kv_names += [f"past_key_{i}", f"past_value_{i}"]

    kv_dynamic = {}
    for name in kv_names:
        kv_dynamic[name] = {2: "seq_len"}

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask),
        os.path.join(output_dir, "talker_prefill.onnx"),
        input_names=["inputs_embeds", "attention_mask"],
        output_names=["logits", "last_hidden"] + kv_names,
        dynamic_axes={
            "inputs_embeds": {1: "seq_len"},
            "attention_mask": {1: "seq_len"},
            "last_hidden": {1: "seq_len"},
            **kv_dynamic,
        },
        opset_version=opset_version,
    )
    print(f"  Done: talker_prefill.onnx ({num_layers} layers, {len(kv_names)} KV tensors)")


def export_talker_decode(model, output_dir, opset_version=14):
    """Export talker decode: single token + KV-cache -> next logits + updated KV-cache.

    Input:  inputs_embeds [1, 1, D] float32  - single-token embedding
            attention_mask [1, T+1] int64     - full attention mask (past + current)
            past_key_*    [1, H, T, head_dim] - KV-cache per layer
    Output: logits         [1, 1, V] float32  - next-token logits
            last_hidden    [1, 1, D] float32  - last hidden state (for code predictor)
            new_past_key_* [1, H, T+1, head_dim] - updated KV-cache
    """
    print("Exporting talker_decode.onnx ...")

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

            # Reconstruct past_key_values from flat list as DynamicCache
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
            hidden = out.last_hidden_state  # [1, 1, D]
            logits = self.talker.codec_head(hidden)  # [1, 1, V]
            new_pkv = out.past_key_values
            return (logits, hidden) + tuple(
                t for i in range(len(new_pkv)) for t in new_pkv[i]
            )

    wrapper = TalkerDecode(talker, num_layers)
    wrapper.eval()

    T_past = 8  # typical past length during tracing
    dummy_embeds = torch.randn(1, 1, D, device=model.device)
    dummy_mask = torch.ones(1, T_past + 1, dtype=torch.long, device=model.device)
    dummy_pkv = [
        torch.randn(1, num_kv_heads, T_past, head_dim, device=model.device)
        for _ in range(num_layers * 2)
    ]

    in_kv_names = []
    out_kv_names = []
    for i in range(num_layers):
        in_kv_names += [f"past_key_{i}", f"past_value_{i}"]
        out_kv_names += [f"new_past_key_{i}", f"new_past_value_{i}"]

    in_kv_dynamic = {name: {2: "past_len"} for name in in_kv_names}
    out_kv_dynamic = {name: {2: "new_len"} for name in out_kv_names}

    torch.onnx.export(
        wrapper,
        (dummy_embeds, dummy_mask, *dummy_pkv),
        os.path.join(output_dir, "talker_decode.onnx"),
        input_names=["inputs_embeds", "attention_mask"] + in_kv_names,
        output_names=["logits", "last_hidden"] + out_kv_names,
        dynamic_axes={
            "attention_mask": {1: "full_len"},
            **in_kv_dynamic,
            **out_kv_dynamic,
        },
        opset_version=opset_version,
    )
    print(f"  Done: talker_decode.onnx ({num_layers} layers)")


def export_speaker_encoder(model, output_dir, opset_version=14):
    """Export speaker encoder.

    Input:  mel [1, T, 128] float32 - mel spectrogram
    Output: speaker_embedding [D] float32 - speaker embedding vector
    """
    print("Exporting speaker_encoder.onnx ...")

    if model.speaker_encoder is None:
        print("  Skipped: model has no speaker_encoder (not a base model)")
        return

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, speaker_encoder):
            super().__init__()
            self.encoder = speaker_encoder

        def forward(self, mel):
            return self.encoder(mel)[0]

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder)
    wrapper.eval()

    dummy_mel = torch.randn(1, 100, 128, device=model.device, dtype=model.dtype)

    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        os.path.join(output_dir, "speaker_encoder.onnx"),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        dynamic_axes={
            "mel": {1: "time"},
        },
        opset_version=opset_version,
    )
    print("  Done: speaker_encoder.onnx")


def export_tokenizer_12hz_encode(speech_tokenizer, output_dir, device, opset_version=14):
    """Export tokenizer12hz_encode.onnx: audio waveform -> RVQ codec tokens.

    Input:  audio [1, num_samples] float32 - audio at 24 kHz
    Output: codes [1, T, 16] int64         - T codec frames (T = num_samples / 1920)
    """
    print("Exporting tokenizer12hz_encode.onnx ...")

    tokenizer_model = speech_tokenizer.model

    class TokenizerEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, audio):
            # audio: [1, num_samples] float32
            # Returns codes: [1, T, num_codebooks] int64
            return self.model.encode(audio)

    wrapper = TokenizerEncoder(tokenizer_model)
    wrapper.eval()

    # Dummy input: 1 second of audio at 24 kHz
    dummy_audio = torch.randn(1, 24000, device=device)

    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        os.path.join(output_dir, "tokenizer12hz_encode.onnx"),
        input_names=["audio"],
        output_names=["codes"],
        dynamic_axes={
            "audio": {1: "num_samples"},
            "codes": {1: "num_frames"},
        },
        opset_version=opset_version,
    )
    print("  Done: tokenizer12hz_encode.onnx")


def _register_diff_symbolic():
    """Register aten::diff ONNX symbolic.

    torch.diff(x, n=1, dim=-1) returns x[1:] - x[:-1], reducing size by 1.
    The Qwen3TTS sliding-window mask uses cumsum(diff(cache_position) != 1)
    as a sequence-ID array that must cover ALL N positions (0..N-1).
    We therefore prepend a zero element so the result has the same size as x.
    """

    def _diff_symbolic(g, x, n, dim, prepend, append):
        from torch.onnx.symbolic_helper import _get_const

        dim_val = _get_const(dim, "i", "dim")  # always -1 in practice
        axes = g.op("Constant", value_t=torch.tensor([dim_val], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        big = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))

        a = g.op("Slice", x, zero, neg1, axes, one)  # x[0:-1]
        b = g.op("Slice", x, one, big, axes, one)  # x[1:]
        diff_result = g.op("Sub", b, a)  # [N-1]

        # Prepend zero: first element minus itself = 0, same dtype/shape
        first = g.op("Slice", x, zero, one, axes, one)  # x[0:1] = [[0]]
        zero_pad = g.op("Sub", first, first)  # [[0]] of right dtype

        # Concat along dim to restore size N
        return g.op("Concat", zero_pad, diff_result, axis_i=dim_val)

    torch.onnx.register_custom_op_symbolic("aten::diff", _diff_symbolic, 18)


def _fix_bool_cumsum(onnx_model):
    """Insert Cast(INT64) before CumSum nodes whose data input is a boolean op.

    The Qwen3TTS mask creates a CumSum over a boolean (Not/Equal/…) tensor.
    ONNX requires the CumSum input to be a numeric type, so we insert an
    explicit Cast to INT64.
    """
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
                )  # 7 = INT64
                node.input[0] = cast_name
                onnx_model.graph.node.insert(i, cast_node)
                cast_added += 1
    return cast_added


def export_tokenizer_12hz_decode(
    speech_tokenizer,
    output_dir,
    device,
    opset_version=18,
    chunk_frames=None,
    context_frames=50,
):
    """Export tokenizer12hz_decode.onnx: RVQ codec tokens -> audio waveform.

    Input:  audio_codes  [1, T, 16] int64   - T frames of 16 RVQ codes each
    Output: audio_values [1, S]     float32 - decoded audio
            lengths      [1]        int64   - valid samples = T * 1920

    The decoder contains a Transformer with sliding-window attention (window=72).
    Tracing with a larger T captures more left-context for boundary frames so
    that chunk-by-chunk streaming produces output consistent with batch decode.

    Quality vs. speed tradeoff (measured on Apple M-series, chunk=25):
      context_frames=  0, trace_T= 25: decode ~0.6s, corr_vs_batch=0.83
      context_frames= 25, trace_T= 50: decode ~1.3s, corr_vs_batch=0.99
      context_frames= 50, trace_T= 75: decode ~1.9s, corr_vs_batch=1.00 ← default

    In C++, sherpa-onnx automatically passes the last context_frames codec
    frames as left context when calling the streaming decoder each chunk.
    The first chunk uses zero-padded context (or no context if T < trace_T).

      chunk_frames=None  → trace with T=100 → suited for batch decode
      chunk_frames=25    → streaming decoder (tokenizer12hz_decode_stream.onnx)

    When chunk_frames is set the output file is tokenizer12hz_decode_stream.onnx
    and trace_T = context_frames + chunk_frames.
    """
    import io
    import onnx

    if chunk_frames is None:
        trace_T = 100
        out_name = "tokenizer12hz_decode.onnx"
        label = "batch"
    else:
        trace_T = context_frames + chunk_frames
        out_name = "tokenizer12hz_decode_stream.onnx"
        label = f"streaming (chunk_frames={chunk_frames}, context_frames={context_frames}, trace_T={trace_T})"

    print(f"Exporting {out_name}  [{label}, traced at T={trace_T}] ...")

    # Register fixed diff symbolic (prepends zero to preserve N elements)
    _register_diff_symbolic()

    # Export decoder.forward directly — avoids Python list comprehension
    # inside Qwen3TTSTokenizerV2Model.decode() which ONNX cannot handle.
    speech_model = speech_tokenizer.model
    decode_upsample_rate = speech_model.decode_upsample_rate  # 1920

    class DecoderForward(nn.Module):
        def __init__(self, decoder, upsample_rate):
            super().__init__()
            self.decoder = decoder
            self.upsample_rate = upsample_rate

        def forward(self, audio_codes):
            # audio_codes: [1, T, 16]
            # decoder.forward expects [1, 16, T]
            wav = self.decoder(audio_codes.transpose(1, 2))  # [1, 1, S]
            audio_values = wav.squeeze(1)  # [1, S]
            # Compute lengths dynamically: count frames where codes >= 0.
            # torch.tensor([T * rate]) would bake T as a constant at trace time.
            # Using a reduction over the actual input keeps it dynamic at runtime.
            lengths = (audio_codes[..., 0] >= 0).sum(dim=1) * self.upsample_rate
            return audio_values, lengths

    wrapper = DecoderForward(speech_model.decoder, decode_upsample_rate)
    wrapper.eval()

    num_codebooks = 16
    dummy_codes = torch.randint(0, 1024, (1, trace_T, num_codebooks), device=device)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (dummy_codes,),
        buf,
        input_names=["audio_codes"],
        output_names=["audio_values", "lengths"],
        dynamic_axes={
            "audio_codes": {1: "codes_length"},
            "audio_values": {1: "audio_length"},
        },
        opset_version=opset_version,
        # Must disable constant folding: the decoder contains Shape nodes that
        # track the runtime sequence length for the attention mask.  With
        # constant folding enabled those are frozen to trace_T.
        do_constant_folding=False,
    )
    buf.seek(0)

    # Post-process: fix bool→CumSum type mismatch
    onnx_model = onnx.load_model_from_string(buf.getvalue())
    n_casts = _fix_bool_cumsum(onnx_model)
    print(f"  Post-processing: inserted {n_casts} Cast(INT64) before CumSum")

    out_path = os.path.join(output_dir, out_name)
    onnx.save(onnx_model, out_path)
    print(f"  Done: {out_name}")


def export_tokenizer_files(processor, output_dir):
    """Copy tokenizer files (vocab.json, merges.txt, etc.) to output directory."""
    print("Exporting tokenizer files ...")

    tokenizer = processor.tokenizer
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print("  Done: tokenizer/")


def get_config_json(model):
    """Extract key configuration values needed at inference time."""
    config = model.config
    talker_config = config.talker_config

    return {
        "model_type": "qwen3-tts-12hz",
        "tts_model_type": config.tts_model_type,
        "tts_model_size": config.tts_model_size,
        "tokenizer_type": config.tokenizer_type,
        "hidden_size": talker_config.hidden_size,
        "text_hidden_size": talker_config.text_hidden_size,
        "vocab_size": talker_config.vocab_size,
        "num_code_groups": talker_config.num_code_groups,
        "num_attention_heads": talker_config.num_attention_heads,
        "num_key_value_heads": talker_config.num_key_value_heads,
        "num_hidden_layers": talker_config.num_hidden_layers,
        "codec_bos_id": talker_config.codec_bos_id,
        "codec_eos_token_id": talker_config.codec_eos_token_id,
        "codec_pad_id": talker_config.codec_pad_id,
        "codec_nothink_id": talker_config.codec_nothink_id,
        "codec_think_id": talker_config.codec_think_id,
        "codec_think_bos_id": talker_config.codec_think_bos_id,
        "codec_think_eos_id": talker_config.codec_think_eos_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
        "tts_pad_token_id": config.tts_pad_token_id,
        "codec_language_id": dict(talker_config.codec_language_id),
        "spk_id": dict(talker_config.spk_id) if hasattr(talker_config, "spk_id") else {},
        "speaker_encoder_sample_rate": config.speaker_encoder_config.sample_rate,
        "output_sample_rate": 24000,
    }


def main():
    parser = argparse.ArgumentParser(description="Export Qwen3-TTS to ONNX for sherpa-onnx")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./qwen3-tts-0.6b-12hz",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on (cpu or cuda:0)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--streaming-chunk-frames",
        type=int,
        default=25,
        help=(
            "New audio frames delivered per streaming callback. "
            "25 frames = 2 s of audio at 12.5 Hz. "
            "Set to 0 to skip the streaming decoder export."
        ),
    )
    parser.add_argument(
        "--streaming-context-frames",
        type=int,
        default=50,
        help=(
            "Left-context frames prepended when decoding each streaming chunk. "
            "trace_T = context + chunk. The decoder sees this many previous "
            "codec frames so that boundary frames have full left context, "
            "making streaming output consistent with batch decode. "
            "context=0: fast (trace_T=chunk), corr≈0.83 vs batch. "
            "context=25: trace_T=50, corr≈0.99. "
            "context=50: trace_T=75, corr=1.00 (default, recommended). "
            "context=72: trace_T=97, corr=1.00 (full sliding window)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    dtype = torch.float32
    model = AutoModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model, fix_mistral_regex=True)

    print(f"Model loaded. Device: {model.device}")
    print(
        f"Config: tts_model_type={model.tts_model_type}, "
        f"tokenizer_type={model.tokenizer_type}, "
        f"tts_model_size={model.tts_model_size}"
    )

    # -----------------------------------------------------------------
    # Export all sub-models
    # -----------------------------------------------------------------
    export_text_project(model, args.output_dir, args.opset_version)
    export_codec_embed(model, args.output_dir, args.opset_version)
    export_code_predictor_embed(model, args.output_dir, args.opset_version)
    export_code_predictor(model, args.output_dir, args.opset_version)
    export_talker_prefill(model, args.output_dir, args.opset_version)
    export_talker_decode(model, args.output_dir, args.opset_version)
    export_speaker_encoder(model, args.output_dir, args.opset_version)

    # Speech tokenizer (encoder + decoder)
    speech_tokenizer = model.speech_tokenizer
    export_tokenizer_12hz_encode(
        speech_tokenizer, args.output_dir, model.device, args.opset_version
    )
    # Tokenizer decoder exports require opset 18 (Slice with dynamic axes,
    # aten::diff symbolic) regardless of --opset-version.
    # Batch decoder (traced at T=100, for full-sequence decode)
    export_tokenizer_12hz_decode(
        speech_tokenizer, args.output_dir, model.device,
        opset_version=18,
        chunk_frames=None,
    )
    # Streaming decoder (traced at T=context+chunk, for low-latency chunk decode)
    if args.streaming_chunk_frames > 0:
        export_tokenizer_12hz_decode(
            speech_tokenizer, args.output_dir, model.device,
            opset_version=18,
            chunk_frames=args.streaming_chunk_frames,
            context_frames=args.streaming_context_frames,
        )

    # Tokenizer vocab files
    export_tokenizer_files(processor, args.output_dir)

    # Save config
    config_data = get_config_json(model)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {config_path}")

    print("\n" + "=" * 60)
    print("Export complete.")
    print()
    print("Files written to:", args.output_dir)
    print()
    if args.streaming_chunk_frames > 0:
        print(
            f"  tokenizer12hz_decode_stream.onnx  ← streaming decoder "
            f"(chunk_frames={args.streaming_chunk_frames})"
        )
        print(
            "  To use streaming in sherpa-onnx, set:"
        )
        print(
            f'    cfg.extra["chunk_frames"] = "{args.streaming_chunk_frames}"'
        )
        print(
            '    cfg.model.qwen3.tokenizer12hz_decode = ".../tokenizer12hz_decode_stream.onnx"'
        )
    print("=" * 60)


if __name__ == "__main__":
    main()

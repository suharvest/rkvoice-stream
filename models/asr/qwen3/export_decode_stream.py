#!/usr/bin/env python3
"""Export tokenizer12hz_decode_stream with fixed shapes.
v3: Force eager attention on the tokenizer decoder to avoid SDPA tracing issues.
"""

import functools
import io
import os
import time

import torch
import torch.nn as nn

# Patch masking_utils to avoid __ior_
import transformers.masking_utils as _mu
_mu._is_torch_greater_or_equal_than_2_5 = True

# Force legacy ONNX exporter
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
DEVICE = "cpu"
STREAM_CHUNK = 25
STREAM_CONTEXT = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)


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


print("Loading model...")
t0 = time.time()

AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

model = AutoModel.from_pretrained(
    MODEL_NAME, device_map=DEVICE, dtype=torch.float32, attn_implementation="eager",
)
model.eval()

# Force eager attention on the speech tokenizer's decoder transformer
st_model = model.speech_tokenizer.model
st_model.config._attn_implementation = "eager"
for layer in st_model.decoder.pre_transformer.layers:
    layer.self_attn.config._attn_implementation = "eager"
    if hasattr(layer.self_attn, '_is_full_attention'):
        pass  # leave as-is
print(f"Model loaded in {time.time()-t0:.1f}s")
print(f"Tokenizer decoder attn_implementation: {st_model.config._attn_implementation}")

trace_T = STREAM_CONTEXT + STREAM_CHUNK
print(f"\n=== tokenizer12hz_decode_stream (fixed T={trace_T}) ===")
t0 = time.time()

import onnx

_register_diff_symbolic()

decode_upsample_rate = st_model.decode_upsample_rate

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

wrapper = DecoderForward(st_model.decoder, decode_upsample_rate)
wrapper.eval()

dummy_codes = torch.randint(0, 1024, (1, trace_T, 16), device=DEVICE)

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
elapsed = time.time() - t0
sz = os.path.getsize(out_path) / (1024*1024)
print(f"  OK: [1,{trace_T},16] -> audio+lengths ({elapsed:.1f}s, {sz:.1f} MB)")

# Final listing
print("\n=== All fixed ONNX files ===")
total = 0
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".onnx"):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        total += sz
        print(f"  {f:45s}  {sz/(1024*1024):8.1f} MB")
print(f"  {'TOTAL':45s}  {total/(1024*1024):8.1f} MB")

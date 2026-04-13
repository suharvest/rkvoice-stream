#!/usr/bin/env python3
"""Export vocoder with different context sizes.

Uses DecoderForwardNoEmbed wrapper that:
  - Takes pre-computed embeddings [1, 512, T] (skips quantizer)
  - Pre-computes causal mask to avoid vmap tracing issues in transformers 4.57+

Pipeline per variant:
  1. Export noembed ONNX with embeddings input [1, 512, T]
  2. Replace Sin with polynomial approximation
  3. Verify with ORT

Usage (on wsl2-local):
    ~/qwen3-tts-export/.venv/bin/python ~/qwen3-tts-export/export_decode_variants.py
"""

import functools
import io
import os
import time

import numpy as np
import torch
import torch.nn as nn

import transformers.masking_utils as _mu
_mu._is_torch_greater_or_equal_than_2_5 = True

if hasattr(torch.onnx.export, "__wrapped__") or tuple(
    int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
) >= (2, 5):
    _orig_onnx_export = torch.onnx.export

    @functools.wraps(_orig_onnx_export)
    def _patched_onnx_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig_onnx_export(*args, **kwargs)

    torch.onnx.export = _patched_onnx_export

import onnx
from onnx import TensorProto, helper
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)

OUTPUT_DIR = os.path.expanduser("~/qwen3-tts-export/vocoder_variants")
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEVICE = "cpu"
CHUNK = 25

VARIANTS = [
    ("ctx25", 25),  # trace_T=50
    ("ctx0", 0),    # trace_T=25
]

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
    name_to_node = {}
    for node in onnx_model.graph.node:
        for o in node.output:
            name_to_node[o] = node
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



class DecoderForwardNoEmbed(nn.Module):
    """Takes pre-computed embeddings [1, 512, T], skips quantizer.

    Also pre-computes causal mask to avoid vmap tracing issues in
    transformers 4.57+.
    """
    def __init__(self, decoder, upsample_rate, trace_T):
        super().__init__()
        self.pre_conv = decoder.pre_conv
        self.pre_transformer = decoder.pre_transformer
        self.upsample = decoder.upsample
        self.decoder_blocks = decoder.decoder
        self.upsample_rate = upsample_rate
        self.trace_T = trace_T

    def forward(self, embeddings):
        hidden = self.pre_conv(embeddings).transpose(1, 2)

        # Pre-compute causal mask (avoids vmap in create_causal_mask)
        seq_len = hidden.shape[1]
        dtype = hidden.dtype
        device = hidden.device
        min_val = torch.finfo(dtype).min
        causal = torch.triu(
            torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device),
            diagonal=1,
        )
        causal_mask = causal.unsqueeze(0).unsqueeze(0)
        mask_dict = {
            "full_attention": causal_mask,
            "sliding_attention": causal_mask,
        }

        hidden = self.pre_transformer(
            inputs_embeds=hidden, attention_mask=mask_dict
        ).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder_blocks:
            wav = block(wav)
        wav = wav.clamp(min=-1, max=1)
        audio_values = wav.squeeze(1)
        lengths = torch.tensor(
            [self.trace_T * self.upsample_rate], dtype=torch.long
        )
        return audio_values, lengths



# ---- Sin replacement ----

def make_scalar(name, value, graph):
    init = helper.make_tensor(name, TensorProto.FLOAT, [], [value])
    graph.initializer.append(init)
    return name


def is_rotary(node):
    name = node.output[0] if node.output else ""
    inp = node.input[0] if node.input else ""
    return "rotary" in name.lower() or "rotary" in inp.lower()


def sin_poly(inp, out, pfx, graph):
    nodes = []
    pi = float(np.pi)
    neg_pi = make_scalar(pfx + "_neg_pi", -pi, graph)
    pos_pi = make_scalar(pfx + "_pos_pi", pi, graph)
    cl = pfx + "_cl"
    nodes.append(helper.make_node("Clip", [inp, neg_pi, pos_pi], [cl]))
    c1 = make_scalar(pfx + "_c1", -1.0 / 6.0, graph)
    c2 = make_scalar(pfx + "_c2", 1.0 / 120.0, graph)
    c3 = make_scalar(pfx + "_c3", -1.0 / 5040.0, graph)
    one = make_scalar(pfx + "_one", 1.0, graph)
    x2 = pfx + "_x2"
    nodes.append(helper.make_node("Mul", [cl, cl], [x2]))
    t1 = pfx + "_t1"
    nodes.append(helper.make_node("Mul", [x2, c3], [t1]))
    t2 = pfx + "_t2"
    nodes.append(helper.make_node("Add", [t1, c2], [t2]))
    t3 = pfx + "_t3"
    nodes.append(helper.make_node("Mul", [t2, x2], [t3]))
    t4 = pfx + "_t4"
    nodes.append(helper.make_node("Add", [t3, c1], [t4]))
    t5 = pfx + "_t5"
    nodes.append(helper.make_node("Mul", [t4, x2], [t5]))
    t6 = pfx + "_t6"
    nodes.append(helper.make_node("Add", [t5, one], [t6]))
    nodes.append(helper.make_node("Mul", [cl, t6], [out]))
    return nodes


def cos_poly(inp, out, pfx, graph):
    nodes = []
    c0 = make_scalar(pfx + "_c0", 1.0, graph)
    c1 = make_scalar(pfx + "_c1", -0.5, graph)
    c2 = make_scalar(pfx + "_c2", 1.0 / 24.0, graph)
    c3 = make_scalar(pfx + "_c3", -1.0 / 720.0, graph)
    x2 = pfx + "_x2"
    nodes.append(helper.make_node("Mul", [inp, inp], [x2]))
    t1 = pfx + "_t1"
    nodes.append(helper.make_node("Mul", [x2, c3], [t1]))
    t2 = pfx + "_t2"
    nodes.append(helper.make_node("Add", [t1, c2], [t2]))
    t3 = pfx + "_t3"
    nodes.append(helper.make_node("Mul", [t2, x2], [t3]))
    t4 = pfx + "_t4"
    nodes.append(helper.make_node("Add", [t3, c1], [t4]))
    t5 = pfx + "_t5"
    nodes.append(helper.make_node("Mul", [t4, x2], [t5]))
    nodes.append(helper.make_node("Add", [t5, c0], [out]))
    return nodes


def replace_sin_cos(model):
    graph = model.graph
    sin_n = cos_n = skip_n = 0
    to_process = [(n, n.op_type) for n in list(graph.node) if n.op_type in ("Sin", "Cos")]
    for node, op in to_process:
        if is_rotary(node):
            skip_n += 1
            continue
        idx = list(graph.node).index(node)
        if op == "Sin":
            polys = sin_poly(node.input[0], node.output[0], "sp_%d" % sin_n, graph)
            sin_n += 1
        else:
            polys = cos_poly(node.input[0], node.output[0], "cp_%d" % cos_n, graph)
            cos_n += 1
        graph.node.remove(node)
        for j, pn in enumerate(polys):
            graph.node.insert(idx + j, pn)
    return sin_n, cos_n, skip_n


# ---- Main ----

def main():
    print("Loading model...")
    t0 = time.time()

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model = AutoModel.from_pretrained(
        MODEL_NAME, device_map=DEVICE, dtype=torch.float32, attn_implementation="eager",
    )
    model.eval()

    st_model = model.speech_tokenizer.model
    st_model.config._attn_implementation = "eager"
    for layer in st_model.decoder.pre_transformer.layers:
        layer.self_attn.config._attn_implementation = "eager"

    print("Model loaded in %.1fs" % (time.time() - t0))

    _register_diff_symbolic()
    decoder = st_model.decoder
    upsample_rate = st_model.decode_upsample_rate

    import onnxruntime as ort

    for name, ctx in VARIANTS:
        trace_T = ctx + CHUNK
        print("")
        print("=" * 60)
        print("VARIANT: %s (context=%d, chunk=%d, T=%d)" % (name, ctx, CHUNK, trace_T))
        print("=" * 60)

        # Step 1: Export noembed ONNX directly (embeddings input, pre-computed mask)
        noembed_path = os.path.join(OUTPUT_DIR, "decoder_%s_noembed.onnx" % name)
        print("")
        print("  Step 1: Export noembed ONNX [1, 512, %d]" % trace_T)
        t1 = time.time()

        wrapper = DecoderForwardNoEmbed(decoder, upsample_rate, trace_T)
        wrapper.eval()

        dummy = torch.randn(1, 512, trace_T, device=DEVICE)
        buf = io.BytesIO()
        torch.onnx.export(
            wrapper, (dummy,), buf,
            input_names=["embeddings"],
            output_names=["audio_values", "lengths"],
            opset_version=18, do_constant_folding=False,
        )
        buf.seek(0)
        m = onnx.load_model_from_string(buf.getvalue())
        nc = _fix_bool_cumsum(m)
        onnx.save(m, noembed_path)
        sz = os.path.getsize(noembed_path) / (1024 * 1024)
        print("    OK (%.1fs, %.1f MB, %d CumSum fixes)" % (time.time() - t1, sz, nc))

        # Verify noembed with ORT
        sess = ort.InferenceSession(noembed_path)
        test_embed = np.random.randn(1, 512, trace_T).astype(np.float32) * 0.5
        out = sess.run(None, {"embeddings": test_embed})
        print("    ORT verify: audio=%s, lengths=%s" % (str(out[0].shape), str(out[1])))

        # Step 2: Replace Sin/Cos
        nosin_path = os.path.join(OUTPUT_DIR, "decoder_%s_nosin.onnx" % name)
        print("")
        print("  Step 2: Replace Sin/Cos with polynomial")
        m2 = onnx.load(noembed_path)
        sr, cr, sk = replace_sin_cos(m2)
        onnx.save(m2, nosin_path)
        sz2 = os.path.getsize(nosin_path) / (1024 * 1024)
        sin_after = sum(1 for n in m2.graph.node if n.op_type == "Sin")
        cos_after = sum(1 for n in m2.graph.node if n.op_type == "Cos")
        print("    Replaced %d Sin + %d Cos (skipped %d rotary)" % (sr, cr, sk))
        print("    Remaining: %d Sin, %d Cos (%.1f MB)" % (sin_after, cos_after, sz2))

        # Verify nosin vs noembed
        sess_nosin = ort.InferenceSession(nosin_path)
        out_nosin = sess_nosin.run(None, {"embeddings": test_embed})
        out_noembed = sess.run(None, {"embeddings": test_embed})
        diff = np.abs(out_nosin[0] - out_noembed[0]).max()
        corr = np.corrcoef(out_nosin[0].flatten(), out_noembed[0].flatten())[0, 1]
        print("    Nosin vs noembed: max_diff=%.6f, corr=%.8f" % (diff, corr))

        # Cleanup noembed
        os.remove(noembed_path)
        print("")
        print("  Final: %s" % nosin_path)

    # Summary
    print("")
    print("=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".onnx"):
            sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024 * 1024)
            print("  %-50s %8.1f MB" % (f, sz))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze the SenseVoice encoder ONNX to a fixed sequence length for RKNN.

lovemefan encoder inputs:
  speech         [1, T, 560] float32   (4 prompt embeds + LFR feats)
  speech_lengths [1]         int64

RKNN has no dynamic dims, so we fix T = T_FIXED. ``speech_lengths`` is folded to
a constant initializer (= T_FIXED) since the runtime always pads to T_FIXED and
the encoder uses it only for masking (padded frames are masked anyway).

Usage:
  python sv_fix_shape.py <src.onnx> <dst.onnx> [T_FIXED=344]
"""
import sys

import numpy as np
import onnx
from onnx import numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "sense-voice-encoder.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "sense-voice-encoder.fixed.onnx"
T_FIXED = int(sys.argv[3]) if len(sys.argv) > 3 else 344

m = onnx.load(SRC)  # loads external data automatically
g = m.graph

# 1) fix speech input shape [1, T_FIXED, 560]
for i in g.input:
    if i.name == "speech":
        dims = i.type.tensor_type.shape.dim
        dims[0].dim_value = 1
        dims[1].dim_param = ""
        dims[1].dim_value = T_FIXED
        dims[2].dim_value = 560

# 2) fold speech_lengths into a constant initializer
for i in list(g.input):
    if i.name == "speech_lengths":
        g.input.remove(i)
g.initializer.append(
    numpy_helper.from_array(np.array([T_FIXED], dtype=np.int64), name="speech_lengths")
)

m = onnx.shape_inference.infer_shapes(m)
onnx.save(m, DST)
print("wrote", DST, "T_FIXED", T_FIXED)

mm = onnx.load(DST, load_external_data=False)
for i in mm.graph.input:
    print("INPUT", i.name, [(d.dim_param or d.dim_value) for d in i.type.tensor_type.shape.dim])
for o in mm.graph.output:
    print("OUTPUT", o.name, [(d.dim_param or d.dim_value) for d in o.type.tensor_type.shape.dim])

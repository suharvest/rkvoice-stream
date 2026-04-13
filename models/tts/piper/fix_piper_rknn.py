#!/usr/bin/env python3
"""Fix Piper VITS ONNX model for RKNN conversion.

Piper VITS is a fast text-to-speech model. This script applies a multi-step ONNX
graph surgery pipeline to make the ONNX model compatible with RKNN conversion:

Step 0: Inject dynamic inputs and masks (before onnxsim)
  - x_mask: replaces encoder sequence_mask subgraph (1, 1, seq_len) float
  - dp noise mask: inserts Mul(dp_noise, x_mask) to zero noise at padding positions
    (prevents dp normalizing flow from mixing padding noise into valid durations)
  - audio_length: replaces ReduceMax(audio_len) for y_mask Range limit (1,) int64
  - cumulative_durations: replaces CumSum(durations) for attn_mask (1, seq_len+1) int64

Step 1: onnxsim with fixed shapes (SEQ_LEN=128)
  - Resolves symbolic dims and folds static shape computations
  - Input shapes: input=[1,seq_len], input_lengths=[1], scales=[3],
                  x_mask=[1,1,seq_len], audio_length=[1],
                  cumulative_durations=[1,seq_len+1]

Step 2: Replace Range nodes with Constants
  - Range nodes for positional encodings are replaced with baked constant tensors

Step 3: Replace Erf(x) with Tanh approximation
  - RKNN runtime doesn't support Erf op on all platforms
  - Approximation: Tanh(x * 0.7978845608 * (1 + 0.044715 * x * x))

Step 4: Replace Softplus(x) with Log(1 + Exp(x))
  - RKNN runtime doesn't support Softplus op on all platforms
  - Mathematically equivalent decomposition

Step 5: Replace RandomNormalLike / RandomUniformLike with fixed constant tensors
  - Noise is baked at compile time (deterministic, seed=42)
  - RKNN runtime doesn't support these stochastic ops

Step 5a: Replace spline NonZero/GatherND/ScatterND with Clip+Where
  - Duration predictor flows.{3,5,7} use RQ spline coupling layers
  - Original: NonZero → GatherND (subset) → spline math → ScatterND (merge)
  - Replacement: Clip(-5,5) → Reshape → spline math on ALL positions → Where
  - Eliminates NonZero, GatherND, ScatterND_8/9 (60 nodes total)

Step 5a-sim: Re-run onnxsim to fold constant index computations

Step 5a2: Replace remaining ScatterND/GatherND/CumSum with NPU-native equivalents
  - ScatterND CHANNEL_SET → Slice+Concat; ScatterND 4D CONST → Slice+Concat
  - CumSum(axis=-1) → MatMul with upper-triangular ones matrix

After surgery, ORT verification is run and the model is saved.

Usage:
  python fix_piper_rknn.py \\
      --input /tmp/piper-analysis/en_US-lessac-medium.onnx \\
      --output /tmp/piper-analysis/piper-rknn-ready.onnx \\
      --seq-len 128 \\
      --wav /tmp/piper_final_fix.wav

Inference inputs:
  - input: phoneme IDs, shape (1, seq_len), padded with 0s
  - input_lengths: number of valid tokens, shape (1,)
  - scales: [noise_scale, length_scale, speaker_scale], shape (3,)
  - x_mask: encoder mask, (1, 1, seq_len), 1.0 for valid tokens, 0.0 for padding
  - audio_length: total audio frames, (1,), computed from duration predictor
  - cumulative_durations: cumulative frame counts, (1, seq_len+1), [0, d0, d0+d1, ...]

Note: audio_length and cumulative_durations must be computed from the duration
      predictor output before calling the fixed model. In typical inference:
      1. Run duration predictor to get durations
      2. audio_length = sum(Ceil(durations * length_scale))
      3. cumulative_durations = cumsum(Ceil(durations * length_scale))
      4. Pass these to the main model

Then convert with RKNN toolkit2:
  from rknn.api import RKNN
  rknn = RKNN()
  rknn.config(target_platform='rk3576', optimization_level=0)
  rknn.load_onnx(model='piper-rknn-ready.onnx')
  rknn.build(do_quantization=False)
  rknn.export_rknn('piper-fp16.rknn')
"""

import os
import sys
import argparse
import tempfile
import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper
import onnxruntime as ort

# Recognized ONNX standard domains (others are treated as custom)
_STANDARD_DOMAINS = {'', 'ai.onnx', 'ai.onnx.ml', 'com.microsoft', 'ai.onnx.training'}


def _stub_custom_ops(model_proto: onnx.ModelProto) -> onnx.ModelProto:
    """Replace custom-domain op nodes with Identity stubs so ORT can load the model.

    Used for probing intermediate tensor shapes when the model contains custom ops
    (e.g. PiperSplineCoupling) that ORT cannot execute. The stub maps the first
    input to the first output via Identity; any remaining outputs are left dangling
    (acceptable since we only probe specific upstream tensors).
    """
    patched = []
    for n in model_proto.graph.node:
        if n.domain not in _STANDARD_DOMAINS:
            if n.input and n.output:
                stub = onnx.helper.make_node(
                    'Identity', inputs=[n.input[0]], outputs=[n.output[0]],
                    name=(n.name or n.op_type) + '__stub')
                patched.append(stub)
            # Drop nodes with no inputs/outputs entirely
        else:
            patched.append(n)
    del model_proto.graph.node[:]
    model_proto.graph.node.extend(patched)
    return model_proto


def _ort_session_with_stub(model_proto: onnx.ModelProto) -> tuple:
    """Save model with custom op stubs to a temp file and return (session, tmp_path).

    Caller is responsible for unlinking tmp_path after use.
    """
    m_stub = onnx.ModelProto()
    m_stub.CopyFrom(model_proto)
    _stub_custom_ops(m_stub)
    del m_stub.graph.value_info[:]
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_stub, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    return sess, tmp


# Fixed parameters
SEQ_LEN = 128   # Max token sequence length (bucket size)


def inject_attention_mask(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace sequence_mask computation with a new model input 'x_mask'.

    The sequence_mask subgraph computes:
      input_lengths -> Unsqueeze -> Less(range, lengths) -> Unsqueeze -> Cast(float)
    producing tensor '/enc_p/Cast_1_output_0' with shape (1, 1, seq_len).

    When onnxsim fixes shapes, input_lengths becomes a known constant and the
    mask gets baked as all-ones, so padding tokens are treated as real input.
    By replacing this with an explicit input, the mask VALUES remain dynamic
    even after shape-fixing.

    At inference time the caller computes:
      x_mask = zeros(1, 1, SEQ_LEN); x_mask[0, 0, :n_tokens] = 1.0
    """
    MASK_TENSOR = '/enc_p/Cast_1_output_0'
    MASK_INPUT_NAME = 'x_mask'

    # Verify the mask tensor exists in the graph
    mask_found = False
    for node in model.graph.node:
        if MASK_TENSOR in node.output:
            mask_found = True
            break
    if not mask_found:
        print("  WARNING: mask tensor not found, skipping injection")
        return model

    # 1. Add x_mask as a new model input with symbolic shape (batch, 1, seq)
    x_mask_input = helper.make_tensor_value_info(
        MASK_INPUT_NAME, TensorProto.FLOAT, ['batch_size', 1, 'phonemes']
    )
    model.graph.input.append(x_mask_input)

    # 2. Rewire all consumers: replace references to the old mask tensor
    rewired = 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == MASK_TENSOR:
                node.input[i] = MASK_INPUT_NAME
                rewired += 1

    # 3. Remove the sequence_mask subgraph nodes (they are now dead code)
    #    Nodes: /enc_p/Cast_1, /enc_p/Unsqueeze_2, /enc_p/Less, /enc_p/Unsqueeze_1
    #    Also /enc_p/Unsqueeze and /enc_p/Range if only used by this subgraph
    mask_subgraph_names = {
        '/enc_p/Cast_1', '/enc_p/Unsqueeze_2', '/enc_p/Less', '/enc_p/Unsqueeze_1',
    }

    # Check if /enc_p/Range and /enc_p/Unsqueeze outputs are used elsewhere
    range_out = '/enc_p/Range_output_0'
    unsqueeze_out = '/enc_p/Unsqueeze_output_0'
    range_only_for_mask = all(
        node.name in mask_subgraph_names or node.name == '/enc_p/Unsqueeze'
        for node in model.graph.node if range_out in node.input
    )
    unsqueeze_only_for_mask = all(
        node.name in mask_subgraph_names
        for node in model.graph.node if unsqueeze_out in node.input
    )
    if range_only_for_mask and unsqueeze_only_for_mask:
        mask_subgraph_names.add('/enc_p/Range')
        mask_subgraph_names.add('/enc_p/Unsqueeze')

    new_nodes = [n for n in model.graph.node if n.name not in mask_subgraph_names]
    removed = len(model.graph.node) - len(new_nodes)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print(f"  Injected x_mask input, rewired {rewired} consumers, removed {removed} subgraph nodes")
    return model


def inject_audio_length(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace ReduceMax(audio_length) with audio_length input.

    VITS length regulator produces:
      durations -> Ceil -> sum = audio_length (total output frames)
      ReduceMax(audio_length) is used as Range limit for y_mask generation

    When onnxsim fixes shapes, this gets baked to a constant based on fixed
    duration values. By replacing with an explicit input, the Range limit
    stays dynamic.

    At inference time the caller computes:
      audio_length = sum(Ceil(durations)) = total number of output audio frames
    """
    AUDIO_LEN_TENSOR = '/ReduceMax_output_0'
    AUDIO_LEN_INPUT_NAME = 'audio_length'

    # Verify the tensor exists
    found = False
    for node in model.graph.node:
        if AUDIO_LEN_TENSOR in node.output:
            found = True
            break
    if not found:
        print("  WARNING: audio_length tensor not found, skipping injection")
        return model

    # Add audio_length as a new model input (scalar int64)
    audio_len_input = helper.make_tensor_value_info(
        AUDIO_LEN_INPUT_NAME, TensorProto.INT64, [1]
    )
    model.graph.input.append(audio_len_input)

    # Rewire: /Range uses /ReduceMax_output_0 as limit
    rewired = 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == AUDIO_LEN_TENSOR:
                node.input[i] = AUDIO_LEN_INPUT_NAME
                rewired += 1

    # DO NOT delete upstream nodes - they may be used elsewhere (e.g. /Cast by /Unsqueeze_1)
    # Let onnxsim clean up dead code after simplification
    print(f"  Injected audio_length input, rewired {rewired} consumers")
    return model


def inject_cumulative_durations(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace CumSum(Ceil(durations)) reshaped with cumulative_durations input.

    VITS length regulator uses CumSum(durations) to create alignment matrix:
      durations (per token) -> Ceil -> CumSum -> Reshape -> Unsqueeze_6
      
    Output shape analysis from ORT probing:
      /Ceil_output_0: (1, 1, seq_len)
      /CumSum_output_0: (1, 1, seq_len) 
      /Reshape_output_0: (seq_len,)
      /Unsqueeze_6_output_0: (seq_len, 1) - used in Less_1 comparison
      
    Less_1 compares:
      Unsqueeze_5 (audio_length frames): shape (1, audio_length)
      Unsqueeze_6 (seq_len tokens): shape (seq_len, 1)
      Result: attention mask (seq_len, audio_length)

    When onnxsim fixes shapes, durations become constant and CumSum output
    gets baked. By replacing with an explicit input, alignment stays dynamic.

    At inference time the caller computes:
      cumulative_durations = cumsum(durations)  # shape (seq_len,)
      Values: [d0, d0+d1, d0+d1+d2, ...] (end frame for each token)
      Input shape: [seq_len, 1] after Unsqueeze matching original tensor shape
    """
    CUMSUM_TENSOR = '/Unsqueeze_6_output_0'
    CUMSUM_INPUT_NAME = 'cumulative_durations'

    # Verify the tensor exists
    found = False
    for node in model.graph.node:
        if CUMSUM_TENSOR in node.output:
            found = True
            break
    if not found:
        print("  WARNING: cumulative_durations tensor not found, skipping injection")
        return model

    # Add cumulative_durations as a new model input
    # Shape: (seq_len, 1) to match Unsqueeze_6 output shape.
    # Original: CumSum(Ceil(durations)) -> Reshape(seq_len,) -> Unsqueeze(seq_len,1)
    # Each element = cumulative frame count at that token position.
    cumsum_input = helper.make_tensor_value_info(
        CUMSUM_INPUT_NAME, TensorProto.FLOAT, ['phonemes', 1]
    )
    model.graph.input.append(cumsum_input)

    # Rewire: /Less_1 uses the cumulative_durations directly
    rewired = 0
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == CUMSUM_TENSOR:
                node.input[i] = CUMSUM_INPUT_NAME
                rewired += 1

    print(f"  Injected cumulative_durations input (FLOAT, shape [phonemes, 1]), rewired {rewired} consumers")
    return model


def inject_dp_noise_mask(model: onnx.ModelProto) -> onnx.ModelProto:
    """Mask dp noise output with x_mask before it enters the dp normalizing flow.

    The duration predictor (dp) adds noise: dp/Mul_1 = RandomNormalLike * noise_scale_w.
    This noise is then sliced and fed into the dp normalizing flow (flows.7->5->3).

    Problem: when padding the input sequence to a fixed length (e.g. 128), the noise
    at padding positions is non-zero. The dp flow's coupling layers mix information
    across positions, so unmasked noise at padding positions corrupts the flow output
    and produces wrong durations.

    Fix: insert Mul(dp_noise, x_mask) right after dp/Mul_1, before the noise enters
    the flow. This zeros out noise at padding positions.

    The dp noise tensor '/dp/Mul_1_output_0' has shape (1, 2, seq_len).
    x_mask has shape (1, 1, seq_len) -- broadcasts correctly.
    """
    DP_NOISE_TENSOR = '/dp/Mul_1_output_0'
    MASK_INPUT_NAME = 'x_mask'  # Already injected by inject_attention_mask

    # Verify the noise tensor exists
    noise_found = False
    for node in model.graph.node:
        if DP_NOISE_TENSOR in node.output:
            noise_found = True
            break
    if not noise_found:
        print("  WARNING: dp noise tensor not found, skipping dp noise masking")
        return model

    # Create masked output name
    masked_noise = '/dp/Mul_1_masked_output_0'

    # 1. Add a Mul node: masked_noise = dp_noise * x_mask
    mask_node = helper.make_node(
        'Mul', [DP_NOISE_TENSOR, MASK_INPUT_NAME], [masked_noise],
        name='/dp/Mul_1_mask'
    )

    # 2. Rewire all consumers of dp_noise to use masked_noise instead
    rewired = 0
    for node in model.graph.node:
        # Skip the original dp/Mul_1 node (it produces DP_NOISE_TENSOR)
        if DP_NOISE_TENSOR in node.output:
            continue
        for i, inp in enumerate(node.input):
            if inp == DP_NOISE_TENSOR:
                node.input[i] = masked_noise
                rewired += 1

    # 3. Insert the mask node right after dp/Mul_1
    new_nodes = []
    for node in model.graph.node:
        new_nodes.append(node)
        if DP_NOISE_TENSOR in node.output:
            new_nodes.append(mask_node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    print(f"  Injected dp noise masking (Mul with x_mask), rewired {rewired} consumers")
    return model


def load_and_simplify(input_path: str, seq_len: int = SEQ_LEN) -> onnx.ModelProto:
    """Step 0+1: Inject dynamic inputs, then simplify with onnxsim."""
    import onnxsim
    model = onnx.load(input_path)
    print(f"  Original nodes: {len(model.graph.node)}")

    # Inject x_mask BEFORE onnxsim so the mask stays dynamic
    print("  Injecting x_mask input (replacing sequence_mask subgraph)...")
    model = inject_attention_mask(model)

    # Mask dp noise with x_mask to prevent padding contamination in dp flow
    print("  Injecting dp noise masking...")
    model = inject_dp_noise_mask(model)

    # Note: audio_length and cumulative_durations are NOT injected as inputs.
    # They stay inside the model and get folded by onnxsim to max values.
    # With correct x_mask + dp_noise_mask, the model produces correct audio
    # in the first portion of the output, with silence padding at the end.
    # Trim silence after inference to get the correct audio length.

    simplified, ok = onnxsim.simplify(
        model,
        overwrite_input_shapes={
            'input': [1, seq_len],
            'input_lengths': [1],
            'scales': [3],
            'x_mask': [1, 1, seq_len],
        },
    )
    print(f"  Simplified: ok={ok}, nodes={len(simplified.graph.node)}")
    return simplified


def fix_range_nodes(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 2: Replace Range nodes with constant tensors."""
    range_nodes = [n for n in model.graph.node if n.op_type == 'Range']
    if not range_nodes:
        print("  No Range nodes found")
        return model

    seq_len = test_inputs['input'].shape[1]

    # Build initializer lookup
    init_map = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    init_map[node.output[0]] = numpy_helper.to_array(attr.t)

    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Range':
            start_name, limit_name, delta_name = n.input[0], n.input[1], n.input[2]

            start_val = init_map.get(start_name)
            limit_val = init_map.get(limit_name)
            delta_val = init_map.get(delta_name)

            if start_val is not None and delta_val is not None:
                start = float(start_val.flat[0])
                delta = float(delta_val.flat[0])
                if limit_val is not None:
                    limit = float(limit_val.flat[0])
                else:
                    limit = float(seq_len)
                arr = np.arange(start, limit, delta)
                if start_val.dtype in (np.int64, np.int32):
                    arr = arr.astype(np.int64)
                else:
                    arr = arr.astype(np.float32)
            else:
                arr = np.arange(seq_len, dtype=np.int64)

            const_node = helper.make_node(
                'Constant', inputs=[], outputs=[n.output[0]],
                name=f'const_range_{i}',
                value=numpy_helper.from_array(arr, name=n.output[0])
            )
            new_nodes.insert(0, const_node)
            print(f"  Replaced Range {n.output[0]}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_erf_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Step 3: Replace Erf(x) with x * Sigmoid(1.702 * x) — simpler GELU.

    Avoids Tanh-based 6-op approximation which can trigger mixed-precision
    crashes in librknnrt when scalar constants are broadcast across FP16 ops.
    Sigmoid-based approximation needs only 3 ops: Mul → Sigmoid → Mul.

    The constant 1.702 is stored as FP16 initializer. To keep the ONNX graph
    valid (ORT requires uniform types in Mul), we Cast:
      x_f16   = Cast(x, fp16)          — activation to fp16
      scaled  = Mul(x_f16, c_f16)      — fp16 × fp16 = fp16
      sig     = Sigmoid(scaled)         — fp16
      gated   = Mul(x_f16, sig)         — fp16 × fp16 = fp16
      y       = Cast(gated, fp32)       — back to fp32 for graph continuity

    RKNN sees the entire island as FP16 ops — no mixed-precision issue.
    """
    erf_nodes = [n for n in model.graph.node if n.op_type == 'Erf']
    if not erf_nodes:
        print("  No Erf nodes found")
        return model

    c_name = '__gelu_c1702_f16'
    init_names = {init.name for init in model.graph.initializer}
    if c_name not in init_names:
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1.702, dtype=np.float16), name=c_name)
        )

    erf_node_ids = {id(n) for n in erf_nodes}
    new_nodes = []

    for n in model.graph.node:
        if id(n) not in erf_node_ids:
            new_nodes.append(n)
            continue

        x = n.input[0]
        y = n.output[0]
        pfx = n.name.replace('/', '_').replace(':', '_') if n.name else y.replace('/', '_').replace(':', '_')

        x_f16   = f'{pfx}__x_f16'
        scaled  = f'{pfx}__scaled_f16'
        sig     = f'{pfx}__sigmoid_f16'
        gated   = f'{pfx}__gated_f16'

        replacement = [
            # Cast activation to fp16
            helper.make_node('Cast', [x],       [x_f16],  name=f'{pfx}_cast_in',
                             to=TensorProto.FLOAT16),
            # Mul(x_f16, 1.702_f16) — all fp16
            helper.make_node('Mul',     [x_f16, c_name], [scaled], name=f'{pfx}_mul_scale'),
            helper.make_node('Sigmoid', [scaled],         [sig],   name=f'{pfx}_sigmoid'),
            helper.make_node('Mul',     [x_f16, sig],     [gated], name=f'{pfx}_mul_gate'),
            # Cast back to fp32 for the rest of the graph
            helper.make_node('Cast', [gated],   [y],      name=f'{pfx}_cast_out',
                             to=TensorProto.FLOAT),
        ]
        new_nodes.extend(replacement)
        print(f"  Replaced Erf {n.name} -> Cast+x*Sigmoid(1.702*x)+Cast (FP16 island)")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_softplus_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    """Step 4: Replace Softplus(x) with Log(1 + Exp(x)).

    Uses a FP16 constant for '1.0' with explicit Cast nodes to keep the
    computation in a consistent FP16 island and avoid mixed FP32/FP16 ops
    at RKNN runtime:
      x_f16       = Cast(x, fp16)
      exp_x       = Exp(x_f16)              — fp16
      one_plus_e  = Add(c1_f16, exp_x)      — fp16
      log_val     = Log(one_plus_e)          — fp16
      y           = Cast(log_val, fp32)      — back to fp32
    """
    sp_nodes = [n for n in model.graph.node if n.op_type == 'Softplus']
    if not sp_nodes:
        print("  No Softplus nodes found")
        return model

    c1_name = '__sp_c1_f16'
    init_names = {init.name for init in model.graph.initializer}
    if c1_name not in init_names:
        model.graph.initializer.append(
            numpy_helper.from_array(np.array(1.0, dtype=np.float16), name=c1_name)
        )

    sp_node_ids = {id(n) for n in sp_nodes}
    new_nodes = []

    for n in model.graph.node:
        if id(n) not in sp_node_ids:
            new_nodes.append(n)
            continue

        x = n.input[0]
        y = n.output[0]
        pfx = y.replace('/', '_').replace(':', '_')

        x_f16       = f'{pfx}__x_f16'
        exp_x       = f'{pfx}__exp_x_f16'
        one_plus_e  = f'{pfx}__one_plus_exp_f16'
        log_val     = f'{pfx}__log_f16'

        replacement = [
            helper.make_node('Cast', [x],                  [x_f16],      name=f'{pfx}_cast_in',
                             to=TensorProto.FLOAT16),
            helper.make_node('Exp',  [x_f16],              [exp_x],      name=f'{pfx}_exp'),
            helper.make_node('Add',  [c1_name, exp_x],     [one_plus_e], name=f'{pfx}_add_1'),
            helper.make_node('Log',  [one_plus_e],         [log_val],    name=f'{pfx}_log'),
            helper.make_node('Cast', [log_val],            [y],          name=f'{pfx}_cast_out',
                             to=TensorProto.FLOAT),
        ]
        new_nodes.extend(replacement)
        print(f"  Replaced Softplus {n.name} -> Cast+Log(1+Exp(x))+Cast (FP16 island)")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_ceil_ops(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Ceil(x) with Neg(Floor(Neg(x))). RKNN doesn't support Ceil."""
    ceil_nodes = [n for n in model.graph.node if n.op_type == 'Ceil']
    if not ceil_nodes:
        print("  No Ceil nodes found")
        return model

    new_nodes = []
    for i, n in enumerate(model.graph.node):
        if n.op_type == 'Ceil':
            x_in, y_out = n.input[0], n.output[0]
            neg1 = f'{y_out}__neg1'
            flr = f'{y_out}__floor'
            new_nodes.extend([
                helper.make_node('Neg', [x_in], [neg1], name=f'{n.name}_neg1'),
                helper.make_node('Floor', [neg1], [flr], name=f'{n.name}_floor'),
                helper.make_node('Neg', [flr], [y_out], name=f'{n.name}_neg2'),
            ])
            print(f"  Replaced Ceil {n.name}")
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def replace_spline_nz_ops(model: onnx.ModelProto, seq_len: int = SEQ_LEN) -> onnx.ModelProto:
    """Replace NonZero/GatherND/ScatterND spline pattern with Clip+Where.

    The RQ spline coupling layers in flows.{3,5,7} use NonZero to find
    in-range/out-of-range positions, GatherND to gather them, run spline math,
    and ScatterND to merge back. This pattern is incompatible with RKNN NPU.

    Replacement strategy (per flow layer):
      - Clip x_b to [-5, 5] so spline math is safe on all positions
      - Reshape 4D tensors to 2D for spline math (skip GatherND)
      - At the end, Where(mask, spline_result, original_x_b) replaces ScatterND merge
      - All removed ops (NonZero, GatherND, ScatterND_8/9, etc.) are deleted

    Also replaces:
      - CumSum → MatMul with upper-triangular ones matrix
      - Remaining ScatterND (constant-index channel writes) → Slice+Concat
    """
    flow_layers = ['flows.3', 'flows.5', 'flows.7']

    # Build node lookup by name
    node_by_name = {}
    for n in model.graph.node:
        if n.name:
            node_by_name[n.name] = n

    # Build consumer map
    consumer_map = {}
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                consumer_map.setdefault(inp, []).append(n)

    nodes_to_remove = set()  # set of node names
    new_nodes = []  # (insert_after_name, [nodes])
    new_initializers = []

    # Common shape constants
    shape_seq = np.array([seq_len], dtype=np.int64)
    shape_seq_name = '__spline_shape_seq'
    new_initializers.append(numpy_helper.from_array(shape_seq, name=shape_seq_name))

    shape_1_1_seq = np.array([1, 1, seq_len], dtype=np.int64)
    shape_1_1_seq_name = '__spline_shape_1_1_seq'
    new_initializers.append(numpy_helper.from_array(shape_1_1_seq, name=shape_1_1_seq_name))

    shape_seq_10 = np.array([seq_len, 10], dtype=np.int64)
    shape_seq_10_name = '__spline_shape_seq_10'
    new_initializers.append(numpy_helper.from_array(shape_seq_10, name=shape_seq_10_name))

    shape_seq_11 = np.array([seq_len, 11], dtype=np.int64)
    shape_seq_11_name = '__spline_shape_seq_11'
    new_initializers.append(numpy_helper.from_array(shape_seq_11, name=shape_seq_11_name))

    clip_min_name = '__spline_clip_min'
    clip_max_name = '__spline_clip_max'
    new_initializers.append(numpy_helper.from_array(np.array(-5.0, dtype=np.float32), name=clip_min_name))
    new_initializers.append(numpy_helper.from_array(np.array(5.0, dtype=np.float32), name=clip_max_name))

    for flow in flow_layers:
        pfx = f'/dp/{flow}/'
        sfx = flow.replace('.', '_')
        print(f"  Processing {flow}...")

        # --- Identify key nodes ---
        split_node = node_by_name.get(f'{pfx}Split')
        and_node = node_by_name.get(f'{pfx}And')
        not_node = node_by_name.get(f'{pfx}Not')
        nz_out = node_by_name.get(f'{pfx}NonZero')      # out-of-range
        nz_in = node_by_name.get(f'{pfx}NonZero_1')      # in-range
        tr2 = node_by_name.get(f'{pfx}Transpose_2')      # NonZero_out -> indices
        tr3 = node_by_name.get(f'{pfx}Transpose_3')      # NonZero_in -> indices
        gnd0 = node_by_name.get(f'{pfx}GatherND')        # out-of-range x_b
        gnd1 = node_by_name.get(f'{pfx}GatherND_1')      # in-range x_b
        gnd2 = node_by_name.get(f'{pfx}GatherND_2')      # widths
        gnd3 = node_by_name.get(f'{pfx}GatherND_3')      # heights
        gnd4 = node_by_name.get(f'{pfx}GatherND_4')      # derivatives
        snd8 = node_by_name.get(f'{pfx}ScatterND_8')     # merge out-of-range
        snd9 = node_by_name.get(f'{pfx}ScatterND_9')     # merge in-range
        sl23 = node_by_name.get(f'{pfx}Slice_23')        # slice gathered OOR
        sl24 = node_by_name.get(f'{pfx}Slice_24')        # slice spline output
        sh72 = node_by_name.get(f'{pfx}Shape_72')        # shape(Transpose_2)
        sh74 = node_by_name.get(f'{pfx}Shape_74')        # shape(Transpose_3)
        g31 = node_by_name.get(f'{pfx}Gather_31')        # length OOR
        g32 = node_by_name.get(f'{pfx}Gather_32')        # length IR
        u30 = node_by_name.get(f'{pfx}Unsqueeze_30')     # length as [1]
        u31 = node_by_name.get(f'{pfx}Unsqueeze_31')     # length as [1]

        if not all([split_node, and_node, gnd1, gnd2, gnd3, gnd4, snd9]):
            print(f"    WARNING: Could not find all required nodes for {flow}, skipping")
            continue

        # Key tensor names
        x_b = split_node.output[1]               # (1,1,128) float
        and_mask = and_node.output[0]             # (1,1,128) bool
        gnd1_out = gnd1.output[0]                 # GatherND_1 output name
        gnd2_out = gnd2.output[0]                 # GatherND_2 output name
        gnd3_out = gnd3.output[0]                 # GatherND_3 output name
        gnd4_out = gnd4.output[0]                 # GatherND_4 output name
        widths_4d = gnd2.input[0]                 # Div_output (1,1,128,10)
        heights_4d = gnd3.input[0]                # Div_1_output (1,1,128,10)
        derivs_4d = gnd4.input[0]                 # ScatterND_1_output (1,1,128,11)
        snd9_out = snd9.output[0]                 # ScatterND_9 output name (1,1,128)

        # The spline math output is Add_20, which Slice_24 slices from
        # Slice_24 input[0] is Add_20_output
        spline_out = sl24.input[0] if sl24 else f'{pfx}Add_20_output_0'

        # --- Mark nodes for removal ---
        remove_names = []
        for nd in [not_node, nz_out, nz_in, tr2, tr3, gnd0, gnd1, gnd2, gnd3, gnd4,
                   snd8, snd9, sl23, sl24, sh72, sh74, g31, g32, u30, u31]:
            if nd is not None:
                remove_names.append(nd.name)
        nodes_to_remove.update(remove_names)

        # --- Add replacement nodes ---
        layer_nodes = []

        # 1. Clip(x_b, -5, 5) → x_b_clipped (1,1,128)
        x_b_clipped = f'{pfx}__x_b_clipped'
        layer_nodes.append(helper.make_node(
            'Clip', [x_b, clip_min_name, clip_max_name], [x_b_clipped],
            name=f'{pfx}__clip_xb'))

        # 2. Reshape(x_b_clipped, [128]) → replaces GatherND_1 output
        layer_nodes.append(helper.make_node(
            'Reshape', [x_b_clipped, shape_seq_name], [gnd1_out],
            name=f'{pfx}__reshape_xb'))

        # 3. Reshape(widths_4d, [128, 10]) → replaces GatherND_2 output
        layer_nodes.append(helper.make_node(
            'Reshape', [widths_4d, shape_seq_10_name], [gnd2_out],
            name=f'{pfx}__reshape_widths'))

        # 4. Reshape(heights_4d, [128, 10]) → replaces GatherND_3 output
        layer_nodes.append(helper.make_node(
            'Reshape', [heights_4d, shape_seq_10_name], [gnd3_out],
            name=f'{pfx}__reshape_heights'))

        # 5. Reshape(derivs_4d, [128, 11]) → replaces GatherND_4 output
        layer_nodes.append(helper.make_node(
            'Reshape', [derivs_4d, shape_seq_11_name], [gnd4_out],
            name=f'{pfx}__reshape_derivs'))

        # 6. Where(mask_flat, spline_out, x_b_flat) → replaces ScatterND_9 output
        mask_flat = f'{pfx}__mask_flat'
        x_b_flat = f'{pfx}__xb_flat'
        where_out = f'{pfx}__where_out'

        layer_nodes.append(helper.make_node(
            'Reshape', [and_mask, shape_seq_name], [mask_flat],
            name=f'{pfx}__reshape_mask'))
        layer_nodes.append(helper.make_node(
            'Reshape', [x_b, shape_seq_name], [x_b_flat],
            name=f'{pfx}__reshape_xb_orig'))
        layer_nodes.append(helper.make_node(
            'Where', [mask_flat, spline_out, x_b_flat], [where_out],
            name=f'{pfx}__where_merge'))
        layer_nodes.append(helper.make_node(
            'Reshape', [where_out, shape_1_1_seq_name], [snd9_out],
            name=f'{pfx}__reshape_merged'))

        new_nodes.extend(layer_nodes)
        print(f"    Removed {len(remove_names)} nodes, added {len(layer_nodes)} replacement nodes")

    # Also remove ConstantOfShape_11 (shared by all 3 ScatterND_8 nodes)
    cos11 = node_by_name.get('/dp/flows.7/ConstantOfShape_11')
    if cos11 is not None:
        # Check if it's only used by ScatterND_8 nodes (which we're removing)
        consumers = consumer_map.get(cos11.output[0], [])
        all_removed = all(c.name in nodes_to_remove for c in consumers)
        if all_removed:
            nodes_to_remove.add(cos11.name)
            # Also remove its Shape input if no other consumers
            # ConstantOfShape_11 input is a Shape node
            for inp_name in cos11.input:
                for n in model.graph.node:
                    if n.output and n.output[0] == inp_name:
                        inp_consumers = consumer_map.get(inp_name, [])
                        if all(c.name in nodes_to_remove or c.name == cos11.name for c in inp_consumers):
                            nodes_to_remove.add(n.name)

    # Apply changes
    for init in new_initializers:
        model.graph.initializer.append(init)

    # Build new node list with correct topological ordering.
    # Strategy: insert each replacement node just before the first kept node
    # that needs its output. Handle transitive dependencies recursively.
    kept_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]

    # Build map: output_name -> replacement node that produces it
    repl_output_map = {}
    for nn in new_nodes:
        for o in nn.output:
            repl_output_map[o] = nn

    def _insert_repl(rn, inserted, result):
        """Recursively insert replacement node and its dependencies."""
        if id(rn) in inserted:
            return
        # First, insert any replacement nodes this one depends on
        for dep_inp in rn.input:
            if dep_inp in repl_output_map:
                _insert_repl(repl_output_map[dep_inp], inserted, result)
        result.append(rn)
        inserted.add(id(rn))

    inserted_repls = set()
    final_nodes = []
    for n in kept_nodes:
        # Before each kept node, insert any needed replacement nodes
        for inp in n.input:
            if inp in repl_output_map:
                _insert_repl(repl_output_map[inp], inserted_repls, final_nodes)
        final_nodes.append(n)

    # Append any remaining replacement nodes not yet inserted
    for nn in new_nodes:
        if id(nn) not in inserted_repls:
            _insert_repl(nn, inserted_repls, final_nodes)
    kept_nodes = final_nodes

    total_removed = sum(1 for n in model.graph.node if n.name in nodes_to_remove)
    del model.graph.node[:]
    model.graph.node.extend(kept_nodes)
    print(f"  Total: removed {total_removed} nodes, added {len(new_nodes)} nodes")
    print(f"  Graph now has {len(model.graph.node)} nodes")

    return model


def replace_remaining_cpu_ops(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Replace remaining ScatterND/GatherND/CumSum that are NOT part of the spline pattern.

    After replace_spline_nz_ops removes the NonZero-based gather/scatter,
    the remaining ops are:
      - ScatterND_0/1 per layer: constant-index boundary value writes
      - ScatterND_2-7 per layer: bin-edge grid construction
      - CumSum per layer: cumulative sum for bin edges
      - Top-level CumSum: duration prediction
    """
    target_ops = {'ScatterND', 'GatherND', 'CumSum'}
    target_nodes = [n for n in model.graph.node if n.op_type in target_ops]
    if not target_nodes:
        print("  No remaining ScatterND/GatherND/CumSum nodes found")
        return model

    op_counts = {}
    for n in target_nodes:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    print(f"  Found: {', '.join(f'{k}({v})' for k,v in op_counts.items())}")

    # Delegate to the existing replace_cpu_fallback_ops logic
    return replace_cpu_fallback_ops(model, test_inputs)


def fix_nonzero_nodes(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 5a: Replace NonZero nodes with baked constant tensors.

    RKNN requires NonZero inputs to be constant (static graph). We probe the
    actual output via ORT and bake the result as a Constant node.
    """
    nz_nodes = [n for n in model.graph.node if n.op_type == 'NonZero']
    if not nz_nodes:
        print("  No NonZero nodes found")
        return model

    # Probe all NonZero outputs
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for nz in nz_nodes:
        vi = helper.make_tensor_value_info(nz.output[0], TensorProto.INT64, None)
        m_probe.graph.output.append(vi)
    tmp = tempfile.mktemp(suffix='.onnx')
    onnx.save(m_probe, tmp)
    sess = ort.InferenceSession(tmp, providers=['CPUExecutionProvider'])
    all_out = sess.run(None, test_inputs)
    os.unlink(tmp)

    out_names = [o.name for o in m_probe.graph.output]
    nz_out_map = {}
    for name, val in zip(out_names, all_out):
        if any(nz.output[0] == name for nz in nz_nodes):
            nz_out_map[name] = val

    # Also collect downstream consumers that use NonZero outputs via Gather/GatherND
    # We need to find all nodes whose inputs are NonZero outputs and trace them
    nz_out_names = {nz.output[0] for nz in nz_nodes}

    # Find nodes that consume NonZero outputs directly and produce derived tensors
    # needed for downstream ops (e.g. Gather, Transpose, Squeeze used as indices)
    # We replace the NonZero nodes themselves; downstream ops will see a Constant input.
    nz_node_ids = {id(n) for n in nz_nodes}
    const_nodes = []
    for nz in nz_nodes:
        name = nz.output[0]
        if name in nz_out_map:
            val = nz_out_map[name]
            const_nodes.append(helper.make_node(
                'Constant', inputs=[], outputs=[name],
                name=f'const_nz_{name.replace("/", "_").replace(":", "_")}',
                value=numpy_helper.from_array(val.astype(np.int64), name=name)
            ))
            print(f"  Replaced NonZero {name}: shape={val.shape}")

    new_nodes = [n for n in model.graph.node if id(n) not in nz_node_ids]
    for cn in const_nodes:
        new_nodes.insert(0, cn)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def replace_cpu_fallback_ops(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Replace ScatterND/GatherND/CumSum with NPU-native equivalents.

    librknnrt 2.3.2 CPU fallback for these ops triggers double-free crashes.
    Instead of baking (which freezes dynamic computation), we replace each op
    with equivalent compositions of NPU-native ops (Reshape, Slice, Concat,
    MatMul, Squeeze, Unsqueeze).

    All 52 ops are in /dp/flows.{3,5,7} coupling layers of a normalizing flow,
    plus 1 top-level CumSum. Pattern classification:

    GatherND (15): 3 EMPTY (no-op) + 12 SEQUENTIAL (data[0,0,:] -> Squeeze)
    ScatterND (30): 3 EMPTY + 27 channel/sequential writes -> Slice+Concat
    CumSum (7): cumulative sum -> MatMul with lower-triangular ones matrix
    """
    target_ops = {'ScatterND', 'GatherND', 'CumSum'}
    target_nodes = [n for n in model.graph.node if n.op_type in target_ops]
    if not target_nodes:
        print("  No ScatterND/GatherND/CumSum nodes found")
        return model

    op_counts = {}
    for n in target_nodes:
        op_counts[n.op_type] = op_counts.get(n.op_type, 0) + 1
    print(f"  Found: {', '.join(f'{k}({v})' for k,v in op_counts.items())}")

    # Build initializer lookup for constant analysis
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == 'Constant':
            for attr in node.attribute:
                if attr.name == 'value':
                    init_map[node.output[0]] = numpy_helper.to_array(attr.t)

    # Probe shapes of all target node inputs/outputs via ORT
    probe_tensors = set()
    for n in target_nodes:
        for inp in n.input:
            if inp:
                probe_tensors.add(inp)
        probe_tensors.add(n.output[0])

    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    # Clear stale intermediate value_info to avoid type conflicts when adding
    # probe outputs (e.g. int64 tensors mistakenly annotated as float after surgery).
    del m_probe.graph.value_info[:]

    # Replace any custom-domain ops (e.g. PiperSplineCoupling) with Identity stubs
    # so that ORT can load and execute the model. Custom ops are only needed at
    # RKNN runtime; the probing session only needs to run through upstream nodes.
    existing = {o.name for o in m_probe.graph.output}
    for tname in probe_tensors:
        if tname not in existing:
            # Use UNDEFINED type — ORT will infer the actual type from the graph.
            vi = onnx.ValueInfoProto()
            vi.name = tname
            m_probe.graph.output.append(vi)

    # Use stub helper to handle any custom domain ops (e.g. PiperSplineCoupling)
    sess, tmp = _ort_session_with_stub(m_probe)
    # Filter test_inputs to only what this model actually requires
    model_input_names = {i.name for i in sess.get_inputs()}
    filtered_inputs = {k: v for k, v in test_inputs.items() if k in model_input_names}
    all_out = sess.run(None, filtered_inputs)
    os.unlink(tmp)

    out_names = [o.name for o in m_probe.graph.output]
    probed_shapes = {}
    for name, val in zip(out_names, all_out):
        probed_shapes[name] = val.shape

    # Build consumer map: output_name -> list of nodes that consume it
    consumer_map = {}
    for n in model.graph.node:
        for inp in n.input:
            if inp:
                consumer_map.setdefault(inp, []).append(n)

    # Collect all nodes to replace and their replacements
    nodes_to_remove = set()
    replacement_nodes = []  # (original_node_id, [new_nodes])
    new_initializers = []

    def _safe_name(s):
        return s.replace('/', '_').replace(':', '_').replace('.', '_')

    # --- Replace GatherND ---
    gathernd_nodes = [n for n in target_nodes if n.op_type == 'GatherND']
    for n in gathernd_nodes:
        data_name = n.input[0]
        indices_name = n.input[1]
        out_name = n.output[0]
        pfx = _safe_name(out_name)

        indices = init_map.get(indices_name)
        if indices is None:
            print(f"  WARNING: GatherND {out_name} has dynamic indices, skipping")
            continue

        nodes_to_remove.add(id(n))

        if indices.size == 0:
            # EMPTY: output is empty tensor, rewire consumers to use data input
            # But we need to check what consumes this. If nothing meaningful, just
            # create an identity.
            # Actually for empty GatherND with indices (0,3), output is (0,) empty tensor.
            # Check if downstream is a ScatterND with empty indices (also a no-op).
            # For safety, create a constant empty tensor matching probed output shape.
            out_shape = probed_shapes.get(out_name, (0,))
            empty_val = np.zeros(out_shape, dtype=np.float32)
            replacement_nodes.append((id(n), [
                helper.make_node('Constant', inputs=[], outputs=[out_name],
                                 name=f'{pfx}_empty',
                                 value=numpy_helper.from_array(empty_val, name=f'{pfx}_empty_val'))
            ]))
            print(f"  GatherND {out_name}: EMPTY indices -> constant empty tensor {out_shape}")

        else:
            # SEQUENTIAL: indices like [[0,0,0],[0,0,1],...,[0,0,N-1]]
            # = data[0, 0, :] = Squeeze(data, axes=[0,1]) or Reshape
            data_shape = probed_shapes.get(data_name)
            out_shape = probed_shapes.get(out_name)
            if data_shape is not None and len(data_shape) == 3:
                # data is (1, 1, N), extract to (N,) via Reshape
                target_shape = np.array(list(out_shape), dtype=np.int64)
                shape_name = f'{pfx}_reshape_shape'
                new_initializers.append(
                    numpy_helper.from_array(target_shape, name=shape_name))
                replacement_nodes.append((id(n), [
                    helper.make_node('Reshape', [data_name, shape_name], [out_name],
                                     name=f'{pfx}_reshape')
                ]))
                print(f"  GatherND {out_name}: SEQUENTIAL {data_shape} -> Reshape to {out_shape}")
            else:
                # Fallback: reshape to output shape
                target_shape = np.array(list(out_shape), dtype=np.int64)
                shape_name = f'{pfx}_reshape_shape'
                new_initializers.append(
                    numpy_helper.from_array(target_shape, name=shape_name))
                replacement_nodes.append((id(n), [
                    helper.make_node('Reshape', [data_name, shape_name], [out_name],
                                     name=f'{pfx}_reshape')
                ]))
                print(f"  GatherND {out_name}: SEQUENTIAL {data_shape} -> Reshape to {out_shape}")

    def _is_sequential_scatter_indices(indices):
        """Check if indices are sequential: [[0,0,0],[0,0,1],...,[0,0,N-1]].

        Handles both 2D (N, ndim) and 3D (N, 1, ndim) shapes.
        """
        idx = indices.reshape(-1, indices.shape[-1]) if len(indices.shape) > 2 else indices
        if idx.shape[-1] < 2:
            return False
        N = idx.shape[0]
        # All leading dims must be 0, last dim must be 0..N-1
        expected = np.zeros_like(idx)
        expected[:, -1] = np.arange(N)
        return np.array_equal(idx, expected)

    # --- Replace ScatterND ---
    scatternd_nodes = [n for n in target_nodes if n.op_type == 'ScatterND']
    for n in scatternd_nodes:
        data_name = n.input[0]
        indices_name = n.input[1]
        updates_name = n.input[2]
        out_name = n.output[0]
        pfx = _safe_name(out_name)

        indices = init_map.get(indices_name)
        if indices is None:
            print(f"  WARNING: ScatterND {out_name} has dynamic indices, skipping")
            continue

        nodes_to_remove.add(id(n))

        if indices.size == 0:
            # EMPTY: output = data (identity), rewire consumers
            replacement_nodes.append((id(n), [
                helper.make_node('Identity', [data_name], [out_name],
                                 name=f'{pfx}_identity')
            ]))
            print(f"  ScatterND {out_name}: EMPTY indices -> Identity")

        elif len(indices.shape) == 3 and indices.shape[-1] == 2:
            # CHANNEL_SET: indices shape (seq_len, 1, 2) = [[i, ch] for i in range(seq_len)]
            # Writing updates to data[:, ch]
            # data shape: (seq_len, num_channels), e.g. (128, 20)
            ch = int(indices[0, 0, 1])  # target channel
            data_shape = probed_shapes.get(data_name)
            if data_shape is None or len(data_shape) != 2:
                print(f"  WARNING: ScatterND CHANNEL_SET {out_name} unexpected data shape {data_shape}")
                continue

            seq_len_dim, num_ch = data_shape

            # Check if updates are constant or dynamic
            updates_const = init_map.get(updates_name)
            updates_shape = probed_shapes.get(updates_name)

            # updates shape: (seq_len, 1) — need to keep as (seq_len, 1) for Concat
            # Build: before = data[:, :ch], after = data[:, ch+1:], out = Concat([before, updates, after])

            new_nodes_list = []

            # Slice constants for this node
            zero_1d = np.array([0], dtype=np.int64)
            ch_1d = np.array([ch], dtype=np.int64)
            ch_plus1_1d = np.array([ch + 1], dtype=np.int64)
            num_ch_1d = np.array([num_ch], dtype=np.int64)
            axis1_1d = np.array([1], dtype=np.int64)

            zero_name = f'{pfx}_zero'
            ch_name = f'{pfx}_ch{ch}'
            ch1_name = f'{pfx}_ch{ch}_plus1'
            nch_name = f'{pfx}_nch{num_ch}'
            ax1_name = f'{pfx}_axis1'

            new_initializers.extend([
                numpy_helper.from_array(zero_1d, name=zero_name),
                numpy_helper.from_array(ch_1d, name=ch_name),
                numpy_helper.from_array(ch_plus1_1d, name=ch1_name),
                numpy_helper.from_array(num_ch_1d, name=nch_name),
                numpy_helper.from_array(axis1_1d, name=ax1_name),
            ])

            before_name = f'{pfx}_before'
            after_name = f'{pfx}_after'

            if ch > 0:
                new_nodes_list.append(
                    helper.make_node('Slice', [data_name, zero_name, ch_name, ax1_name],
                                     [before_name], name=f'{pfx}_slice_before'))

            if ch + 1 < num_ch:
                new_nodes_list.append(
                    helper.make_node('Slice', [data_name, ch1_name, nch_name, ax1_name],
                                     [after_name], name=f'{pfx}_slice_after'))

            # Ensure updates are (seq_len, 1) shaped for concat
            # updates from ScatterND are typically (seq_len, 1) already
            updates_reshaped = updates_name
            if updates_shape is not None and len(updates_shape) == 1:
                # Need to unsqueeze to (seq_len, 1)
                updates_reshaped = f'{pfx}_updates_unsq'
                unsq_axis = np.array([1], dtype=np.int64)
                unsq_axis_name = f'{pfx}_unsq_axis'
                new_initializers.append(numpy_helper.from_array(unsq_axis, name=unsq_axis_name))
                new_nodes_list.append(
                    helper.make_node('Unsqueeze', [updates_name, unsq_axis_name],
                                     [updates_reshaped], name=f'{pfx}_unsqueeze'))

            # Concat parts
            concat_inputs = []
            if ch > 0:
                concat_inputs.append(before_name)
            concat_inputs.append(updates_reshaped)
            if ch + 1 < num_ch:
                concat_inputs.append(after_name)

            new_nodes_list.append(
                helper.make_node('Concat', concat_inputs, [out_name],
                                 name=f'{pfx}_concat', axis=1))

            replacement_nodes.append((id(n), new_nodes_list))
            print(f"  ScatterND {out_name}: CHANNEL_SET ch={ch} data={data_shape} -> Slice+Concat")

        elif _is_sequential_scatter_indices(indices):
            # SEQUENTIAL: indices like [[0,0,0],[0,0,1],...,[0,0,N-1]]
            # Can be 2D shape (N, ndim) or 3D shape (N, 1, ndim)
            # Writing updates to data[0, 0, :] — replacing the inner content
            # data shape: (1, 1, N), updates shape: (N,) or (1, 1, N)
            data_shape = probed_shapes.get(data_name)
            updates_shape = probed_shapes.get(updates_name)

            if data_shape is not None:
                # Reshape updates to match data shape
                target_shape = np.array(list(data_shape), dtype=np.int64)
                shape_name = f'{pfx}_target_shape'
                new_initializers.append(numpy_helper.from_array(target_shape, name=shape_name))
                replacement_nodes.append((id(n), [
                    helper.make_node('Reshape', [updates_name, shape_name], [out_name],
                                     name=f'{pfx}_reshape')
                ]))
                print(f"  ScatterND {out_name}: SEQUENTIAL updates={updates_shape} -> Reshape to {data_shape}")
            else:
                print(f"  WARNING: ScatterND SEQUENTIAL {out_name} unexpected data shape {data_shape}")

        elif len(indices.shape) == 5 and indices.shape[-1] == 4:
            # 4D indices: (1, 1, N, 1, 4) with const updates
            # Writing constant values at positions [0, 0, i, ch] for all i
            # data shape: (1, 1, N, num_ch), updates: (1, 1, N, 1)
            # This overwrites a specific channel slice with constants
            data_shape = probed_shapes.get(data_name)
            updates_const = init_map.get(updates_name)

            if data_shape is not None and len(data_shape) == 4 and updates_const is not None:
                # Extract target channel from indices
                ch = int(indices[0, 0, 0, 0, 3])
                _, _, seq_dim, num_ch = data_shape

                zero_1d = np.array([0], dtype=np.int64)
                ch_1d = np.array([ch], dtype=np.int64)
                ch1_1d = np.array([ch + 1], dtype=np.int64)
                nch_1d = np.array([num_ch], dtype=np.int64)
                ax3_1d = np.array([3], dtype=np.int64)

                zero_name = f'{pfx}_4d_zero'
                ch_name = f'{pfx}_4d_ch{ch}'
                ch1_name = f'{pfx}_4d_ch{ch}_plus1'
                nch_name = f'{pfx}_4d_nch{num_ch}'
                ax3_name = f'{pfx}_4d_axis3'

                new_initializers.extend([
                    numpy_helper.from_array(zero_1d, name=zero_name),
                    numpy_helper.from_array(ch_1d, name=ch_name),
                    numpy_helper.from_array(ch1_1d, name=ch1_name),
                    numpy_helper.from_array(nch_1d, name=nch_name),
                    numpy_helper.from_array(ax3_1d, name=ax3_name),
                ])

                # updates_const reshaped to (1, 1, N, 1) for the channel
                updates_val = updates_const.reshape(1, 1, seq_dim, 1).astype(np.float32)
                updates_init_name = f'{pfx}_4d_updates_const'
                new_initializers.append(
                    numpy_helper.from_array(updates_val, name=updates_init_name))

                before_name = f'{pfx}_4d_before'
                after_name = f'{pfx}_4d_after'
                new_nodes_list = []

                if ch > 0:
                    new_nodes_list.append(
                        helper.make_node('Slice', [data_name, zero_name, ch_name, ax3_name],
                                         [before_name], name=f'{pfx}_4d_slice_before'))
                if ch + 1 < num_ch:
                    new_nodes_list.append(
                        helper.make_node('Slice', [data_name, ch1_name, nch_name, ax3_name],
                                         [after_name], name=f'{pfx}_4d_slice_after'))

                concat_inputs = []
                if ch > 0:
                    concat_inputs.append(before_name)
                concat_inputs.append(updates_init_name)
                if ch + 1 < num_ch:
                    concat_inputs.append(after_name)

                new_nodes_list.append(
                    helper.make_node('Concat', concat_inputs, [out_name],
                                     name=f'{pfx}_4d_concat', axis=3))

                replacement_nodes.append((id(n), new_nodes_list))
                print(f"  ScatterND {out_name}: 4D CONST ch={ch} data={data_shape} -> Slice+Concat")
            else:
                print(f"  WARNING: ScatterND 4D {out_name} unexpected shapes data={data_shape}")

        else:
            print(f"  WARNING: ScatterND {out_name} unrecognized indices shape {indices.shape}, skipping")
            nodes_to_remove.discard(id(n))

    # --- Replace CumSum ---
    cumsum_nodes = [n for n in target_nodes if n.op_type == 'CumSum']
    for n in cumsum_nodes:
        x_name = n.input[0]
        out_name = n.output[0]
        pfx = _safe_name(out_name)

        # Get last dimension from probed shape
        x_shape = probed_shapes.get(x_name)
        if x_shape is None:
            print(f"  WARNING: CumSum {out_name} cannot determine input shape, skipping")
            continue

        nodes_to_remove.add(id(n))

        # CumSum along last axis = MatMul(x, triu(ones(N, N)))
        # y[i,j] = sum_{k=0}^{j} x[i,k] requires M[k,j]=1 if k<=j = upper triangular
        N = x_shape[-1]
        triu_matrix = np.triu(np.ones((N, N), dtype=np.float32))
        tril_name = f'{pfx}_triu_{N}'  # keep var name for minimal diff
        new_initializers.append(numpy_helper.from_array(triu_matrix, name=tril_name))

        # If x has more than 2 dims, we need to reshape for MatMul then reshape back
        if len(x_shape) > 2:
            # Reshape to (batch, N) where batch = product of all dims except last
            batch = int(np.prod(x_shape[:-1]))
            flat_shape = np.array([batch, N], dtype=np.int64)
            orig_shape = np.array(list(x_shape), dtype=np.int64)
            flat_shape_name = f'{pfx}_flat_shape'
            orig_shape_name = f'{pfx}_orig_shape'
            new_initializers.extend([
                numpy_helper.from_array(flat_shape, name=flat_shape_name),
                numpy_helper.from_array(orig_shape, name=orig_shape_name),
            ])

            x_flat = f'{pfx}_flat'
            mm_out = f'{pfx}_mm'
            replacement_nodes.append((id(n), [
                helper.make_node('Reshape', [x_name, flat_shape_name], [x_flat],
                                 name=f'{pfx}_flatten'),
                helper.make_node('MatMul', [x_flat, tril_name], [mm_out],
                                 name=f'{pfx}_matmul'),
                helper.make_node('Reshape', [mm_out, orig_shape_name], [out_name],
                                 name=f'{pfx}_unflatten'),
            ]))
        else:
            # Simple 2D case
            replacement_nodes.append((id(n), [
                helper.make_node('MatMul', [x_name, tril_name], [out_name],
                                 name=f'{pfx}_matmul')
            ]))

        print(f"  CumSum {out_name}: shape={x_shape} -> MatMul with triu({N}x{N})")

    # --- Apply all replacements ---
    # Build replacement map: original node id -> list of new nodes
    repl_map = {nid: nodes for nid, nodes in replacement_nodes}

    # Add new initializers
    for init in new_initializers:
        model.graph.initializer.append(init)

    new_nodes = []
    for n in model.graph.node:
        if id(n) in nodes_to_remove:
            if id(n) in repl_map:
                new_nodes.extend(repl_map[id(n)])
        else:
            new_nodes.append(n)

    replaced_count = len(nodes_to_remove)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    print(f"  Replaced {replaced_count} ops total, graph now has {len(model.graph.node)} nodes")

    return model


def fix_3d_matmul_for_rknn(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 5c: Fix 3D MatMul nodes that trigger RKNN exMatMul broadcast bug.

    RKNN toolkit2's exMatMul incorrectly handles 3D batched MatMul when the
    first operand has shape (B, N, N) — it remaps to (B*N*N, 1, 1) and
    broadcast-multiplies with (N, 1, 1), causing a shape mismatch.

    This affects multiple matmuls in Piper VITS:
      - /MatMul, /MatMul_1: (1, 128, 128) mask @ (1, 128, 192) encoder  [3D]
      - attn_layers.*/MatMul_2: (1, 2, 128, 128) attn_wt @ (1, 2, 128, 96) V  [4D]

    Fix: Decompose A @ B by splitting along the K-dimension (inner product split).

    For A @ B where A has square last-two dims (M==K==N):
      C = A @ B = A[:, :, :, :N/2] @ B[:, :, :N/2, :] + A[:, :, :, N/2:] @ B[:, :, N/2:, :]

    Each sub-MatMul has shape (..., N, N/2) @ (..., N/2, D), so M=N, K=N/2, M≠K.
    This avoids the exMatMul square-first-input bug entirely.

    This approach (vs row-splitting) ensures neither Softmax→MatMul nor Split(A)+parallel
    pattern is seen by RKNN's attention or mesh-backward fusion passes.

    For 3D tensors (B, N, N) @ (B, N, D) — K-split on A axis=2, B axis=1:
      A_lo: (B, N, N/2)  B_lo: (B, N/2, D)  → MatMul: (B, N, D)
      A_hi: (B, N, N/2)  B_hi: (B, N/2, D)  → MatMul: (B, N, D)
      C = Add(MatMul(A_lo, B_lo), MatMul(A_hi, B_hi))

    For 4D tensors (B, H, N, N) @ (B, H, N, D) — K-split on A axis=3, B axis=2:
      A_lo: (B, H, N, N/2)  B_lo: (B, H, N/2, D)  → MatMul: (B, H, N, D)
      A_hi: (B, H, N, N/2)  B_hi: (B, H, N/2, D)  → MatMul: (B, H, N, D)
      C = Add(MatMul(A_lo, B_lo), MatMul(A_hi, B_hi))

    N must be even (N=128 → N/2=64). Each sub-MatMul: M=N=128, K=N/2=64, M≠K. Safe.

    We identify candidates by probing ORT: a 3D or 4D float MatMul whose first
    input has equal last-two dims (N x N square matrix).
    """
    # Probe shapes of all MatMul outputs via ORT
    matmul_nodes = [n for n in model.graph.node
                    if n.op_type == 'MatMul' and len(n.input) >= 2
                    and n.input[0] and n.input[1]]
    if not matmul_nodes:
        print("  No MatMul nodes found")
        return model

    # Probe both A and B inputs of each MatMul
    probe_tensors = set()
    for mm in matmul_nodes:
        probe_tensors.add(mm.input[0])
        probe_tensors.add(mm.input[1])

    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for tname in probe_tensors:
        vi = helper.make_tensor_value_info(tname, TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)
    try:
        sess, tmp = _ort_session_with_stub(m_probe)
        model_input_names = {i.name for i in sess.get_inputs()}
        filtered = {k: v for k, v in test_inputs.items() if k in model_input_names}
        all_out = sess.run(None, filtered)
        out_names = [o.name for o in m_probe.graph.output]
        input_shapes = {name: val.shape for name, val in zip(out_names, all_out)
                        if name in probe_tensors}
    except Exception as e:
        print(f"  Warning: could not probe MatMul shapes: {e}")
        input_shapes = {}
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass

    # Find MatMul nodes where first input has square last two dims (3D or 4D)
    # RKNN exMatMul bug: when A[-2] == A[-1] (i.e. M == K), exMatMul_infer remaps
    # A to (M*K, 1, 1) = (N^2, 1, 1) and K-dim to (N, 1, 1), failing to broadcast.
    target_matmul_ids = set()
    for mm in matmul_nodes:
        sh = input_shapes.get(mm.input[0])
        if sh is None:
            continue
        ndim = len(sh)
        if ndim == 3 and sh[1] == sh[2]:
            target_matmul_ids.add(id(mm))
            print(f"  Will fix 3D square MatMul {mm.name}: A_shape={sh}")
        elif ndim == 4 and sh[2] == sh[3]:
            target_matmul_ids.add(id(mm))
            print(f"  Will fix 4D square MatMul {mm.name}: A_shape={sh}")

    if not target_matmul_ids:
        print("  No problematic square MatMul nodes found")
        return model

    new_nodes = []
    for n in model.graph.node:
        if id(n) not in target_matmul_ids:
            new_nodes.append(n)
            continue

        a_in, b_in = n.input[0], n.input[1]
        y_out = n.output[0]
        pfx = y_out.replace('/', '_').replace(':', '_')

        a_shape = input_shapes.get(a_in)
        b_shape = input_shapes.get(b_in)
        ndim = len(a_shape)

        # Concat+1 trick: extend A with a zero-column and B with a zero-row, making
        # K → K+1 so M ≠ K.  Uses Concat instead of Pad so that RKNN's
        # swap_pad_transpose / swap_reshape_pad passes cannot undo the shape change.
        # RKNN has no swap_concat_transpose rule.
        #
        # A: (..., M, K) concat (..., M, 1) → (..., M, K+1)   [along last axis]
        # B: (..., K, D) concat (..., 1, D) → (..., K+1, D)   [along second-to-last]
        # (A_ext) @ (B_ext) = A @ B  (zero col × zero row contributes nothing)

        N = a_shape[-1]        # = a_shape[-2] since square (K = M = N)
        if ndim == 3:
            a_col_shape = list(a_shape); a_col_shape[-1] = 1   # (B, M, 1)
            b_row_shape = list(b_shape); b_row_shape[-2] = 1   # (B, 1, D)
            concat_a_axis = 2   # last dim
            concat_b_axis = 1   # second-to-last
        else:  # ndim == 4
            a_col_shape = list(a_shape); a_col_shape[-1] = 1   # (B, H, M, 1)
            b_row_shape = list(b_shape); b_row_shape[-2] = 1   # (B, H, 1, D)
            concat_a_axis = 3   # last dim
            concat_b_axis = 2   # second-to-last

        a_zeros_name = f'{pfx}__a_zeros'
        b_zeros_name = f'{pfx}__b_zeros'
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.zeros(a_col_shape, dtype=np.float32), name=a_zeros_name))
        model.graph.initializer.append(
            numpy_helper.from_array(
                np.zeros(b_row_shape, dtype=np.float32), name=b_zeros_name))

        a_ext = f'{pfx}__a_ext'
        b_ext = f'{pfx}__b_ext'

        replacement = [
            # A_ext = Concat([A, zeros_col], axis=last)  → (..., M, K+1)
            helper.make_node('Concat', [a_in, a_zeros_name], [a_ext],
                             name=f'{pfx}_cat_a', axis=concat_a_axis),
            # B_ext = Concat([B, zeros_row], axis=second-to-last)  → (..., K+1, D)
            helper.make_node('Concat', [b_in, b_zeros_name], [b_ext],
                             name=f'{pfx}_cat_b', axis=concat_b_axis),
            # out = A_ext @ B_ext  → (..., M, D)  [M=128, K+1=129, M≠K]
            helper.make_node('MatMul', [a_ext, b_ext], [y_out],
                             name=f'{pfx}_mm'),
        ]
        new_nodes.extend(replacement)
        print(f"  Fixed {ndim}D MatMul {n.name} -> Concat(K+1)+MatMul: A={a_shape} B={b_shape} K+1={N+1}")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def fix_random_noise(model: onnx.ModelProto, test_inputs: dict) -> onnx.ModelProto:
    """Step 5b: Replace RandomNormalLike / RandomUniformLike with fixed constant tensors (seed=42)."""
    target_ops = {'RandomNormalLike', 'RandomUniformLike'}
    rn_nodes = [n for n in model.graph.node if n.op_type in target_ops]
    if not rn_nodes:
        print("  No RandomNormalLike / RandomUniformLike nodes found")
        return model

    # Probe to get output shapes (use stub to handle any custom ops in the graph)
    m_probe = onnx.ModelProto()
    m_probe.CopyFrom(model)
    for rn in rn_nodes:
        vi = helper.make_tensor_value_info(rn.output[0], TensorProto.FLOAT, None)
        m_probe.graph.output.append(vi)
    sess, tmp = _ort_session_with_stub(m_probe)
    model_input_names = {i.name for i in sess.get_inputs()}
    filtered_inputs = {k: v for k, v in test_inputs.items() if k in model_input_names}
    all_out = sess.run(None, filtered_inputs)
    os.unlink(tmp)
    shapes = {name: val.shape for name, val in zip([o.name for o in m_probe.graph.output], all_out)
              if any(rn.output[0] == name for rn in rn_nodes)}

    rng = np.random.default_rng(42)
    nodes_to_remove = set()
    const_nodes = []
    for rn in rn_nodes:
        name = rn.output[0]
        if name in shapes:
            if rn.op_type == 'RandomNormalLike':
                noise = rng.standard_normal(size=shapes[name]).astype(np.float32)
            else:  # RandomUniformLike
                noise = rng.uniform(0, 1, size=shapes[name]).astype(np.float32)
            const_nodes.append(helper.make_node(
                'Constant', inputs=[], outputs=[name],
                name=f'const_{rn.name}',
                value=numpy_helper.from_array(noise, name=name)
            ))
            nodes_to_remove.add(id(rn))
            print(f"  Replaced {rn.op_type} {name}: shape={shapes[name]}")

    new_nodes = [n for n in model.graph.node if id(n) not in nodes_to_remove]
    for cn in const_nodes:
        new_nodes.insert(0, cn)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Fix Piper VITS ONNX model for RKNN conversion (5-step pipeline)'
    )
    parser.add_argument('--input',
                        default='/tmp/piper-analysis/en_US-lessac-medium.onnx',
                        help='Path to input ONNX model')
    parser.add_argument('--output',
                        default='/tmp/piper-analysis/piper-rknn-ready.onnx',
                        help='Path to output fixed ONNX model')
    parser.add_argument('--seq-len', type=int, default=SEQ_LEN,
                        help=f'Max token sequence length / bucket size (default: {SEQ_LEN})')
    parser.add_argument('--wav', default=None,
                        help='Generate a test WAV file at this path')
    parser.add_argument('--stop-after-step4b', action='store_true',
                        help='Stop after Step 4b (Ceil fix), before spline ops (use with surgery_piper_custom_ops.py)')
    parser.add_argument('--start-from-step5b', action='store_true',
                        help='Skip steps 0-5a2, only apply 5b (3D MatMul) + 5c (RandomNoise) on --input')
    args = parser.parse_args()

    input_path = os.path.expanduser(args.input)
    output_path = os.path.expanduser(args.output)
    seq_len = args.seq_len

    print(f"=== Piper VITS ONNX RKNN Fix Pipeline ===")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"SEQ_LEN={seq_len}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Reference test inputs (with x_mask and audio_length for dynamic masking)
    # For testing, use synthetic duration values to compute audio_length and cumulative_durations
    n_ref = 5
    tokens = np.zeros((1, seq_len), dtype=np.int64)
    tokens[0, :n_ref] = [1, 2, 3, 4, 5]
    
    # Compute x_mask (encoder mask)
    x_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
    x_mask[0, 0, :n_ref] = 1.0
    
    # For testing: use synthetic durations (each token ~10 frames at scale=1.0)
    # audio_length = sum of all durations
    # In real inference, durations come from duration predictor output
    # For test: assume each token has duration = 10 * scale (scale[1] = length_scale)
    length_scale = 1.0  # scales[1]
    synthetic_durations = np.array([10, 12, 8, 15, 11], dtype=np.float32) * length_scale
    synthetic_durations_int = np.ceil(synthetic_durations).astype(np.int64)
    audio_length = np.array([int(np.sum(synthetic_durations_int))], dtype=np.int64)
    
    # cumulative_durations: [d0, d0+d1, d0+d1+d2, ...] - each token's cumulative end frame
    # Shape: (seq_len, 1) to match Unsqueeze_6 output in original model
    # Values are the end frame position for each token
    cumulative_durations = np.zeros((seq_len, 1), dtype=np.float32)
    cumsum_vals = np.cumsum(synthetic_durations_int).astype(np.float32)
    cumulative_durations[:n_ref, 0] = cumsum_vals
    
    test_inputs = {
        'input': tokens,
        'input_lengths': np.array([n_ref], dtype=np.int64),
        'scales': np.array([0.667, 1.0, 0.8], dtype=np.float32),
        'x_mask': x_mask,
        'audio_length': audio_length,
        'cumulative_durations': cumulative_durations,
    }

    # Jump-start from step 5a2 or 5b (for custom-op surgery workflow)
    if args.start_from_step5b:
        print(f"\n[start-from-step5b] Loading model (skipping steps 0-5a2)...")
        model = onnx.load(input_path)
        print(f"  Nodes: {len(model.graph.node)}")
        # Step 5a2: Replace remaining ScatterND/GatherND/CumSum
        print("\n[5a2] Replacing remaining ScatterND/GatherND/CumSum with NPU-native ops...")
        model = replace_remaining_cpu_ops(model, test_inputs)
        # Step 5b: Fix 3D MatMul for RKNN exMatMul broadcast bug
        print("\n[5b] Fixing 3D MatMul nodes for RKNN compatibility...")
        model = fix_3d_matmul_for_rknn(model, test_inputs)
        # Step 5c: RandomNormalLike / RandomUniformLike
        print("\n[5c] Replacing RandomNormalLike / RandomUniformLike...")
        model = fix_random_noise(model, test_inputs)
        # Check remaining problematic ops
        warn_ops = ['Range', 'Erf', 'Softplus', 'NonZero', 'RandomNormalLike', 'RandomUniformLike',
                    'ScatterND', 'GatherND', 'CumSum']
        for op in warn_ops:
            n = sum(1 for node in model.graph.node if node.op_type == op)
            if n:
                print(f"  WARNING: {n} {op} node(s) remain")
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(model, output_path)
        sz = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSaved: {output_path} ({sz:.1f} MB), {len(model.graph.node)} nodes")
        return

    # Step 1: onnxsim
    print("\n[1/5] onnxsim simplification...")
    model = load_and_simplify(input_path, seq_len)

    # Step 2: Range nodes
    print("\n[2/5] Replacing Range nodes...")
    model = fix_range_nodes(model, test_inputs)

    # Step 3: Erf nodes
    print("\n[3/5] Replacing Erf nodes with Tanh approximation...")
    model = fix_erf_nodes(model)

    # Step 4: Softplus nodes
    print("\n[4/5] Replacing Softplus nodes with Log(1+Exp(x))...")
    model = fix_softplus_nodes(model)

    # Step 4b: Ceil ops
    print("\n[4b] Replacing Ceil with Neg(Floor(Neg(x)))...")
    model = fix_ceil_ops(model)

    # Early exit for custom-op surgery workflow
    if args.stop_after_step4b:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        onnx.save(model, output_path)
        sz = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n[stop-after-step4b] Saved intermediate model: {output_path} ({sz:.1f} MB)")
        print(f"  Nodes: {len(model.graph.node)}")
        print(f"  Next: python surgery_piper_custom_ops.py --input {output_path} ...")
        return

    # Step 5a: Replace spline NonZero/GatherND/ScatterND with Clip+Where
    print("\n[5a/6] Replacing spline NonZero/GatherND/ScatterND with Clip+Where...")
    model = replace_spline_nz_ops(model, seq_len)

    # Re-run onnxsim to fold constant propagation (Range constants → index computations)
    print("\n[5a-sim] Re-running onnxsim to fold constant index computations...")
    import onnxsim
    tmp_sim = tempfile.mktemp(suffix='.onnx')
    onnx.save(model, tmp_sim)
    model2, ok2 = onnxsim.simplify(onnx.load(tmp_sim))
    os.unlink(tmp_sim)
    print(f"  Re-simplified: ok={ok2}, nodes={len(model2.graph.node)}")
    model = model2

    # Step 5a2: Replace remaining ScatterND/GatherND/CumSum with NPU-native equivalents
    print("\n[5a2] Replacing remaining ScatterND/GatherND/CumSum with NPU-native ops...")
    model = replace_remaining_cpu_ops(model, test_inputs)

    # Step 5b: Fix 3D MatMul for RKNN exMatMul broadcast bug
    print("\n[5b/6] Fixing 3D MatMul nodes for RKNN compatibility...")
    model = fix_3d_matmul_for_rknn(model, test_inputs)

    # Step 5c: RandomNormalLike / RandomUniformLike
    print("\n[5c/6] Replacing RandomNormalLike / RandomUniformLike...")
    model = fix_random_noise(model, test_inputs)

    # Shape inference + ORT verify
    print("\n[verify] Shape inference and ORT verification...")
    try:
        model = onnx.shape_inference.infer_shapes(model)
        print("  Shape inference OK")
    except Exception as e:
        print(f"  Warning (shape inference): {e}")

    try:
        sess = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
        out = sess.run(None, test_inputs)
        audio = out[0]
        print(f"  Output shape: {audio.shape}")
        print(f"  Output RMS: {float(np.sqrt(np.mean(audio**2))):.4f}")
        print("  ORT OK")
    except Exception as e:
        print(f"  ORT FAIL: {e}")
        sys.exit(1)

    # Check for remaining problematic ops
    warn_ops = ['Range', 'Erf', 'Softplus', 'NonZero', 'RandomNormalLike', 'RandomUniformLike',
                'If', 'Loop', 'SplitToSequence', 'SequenceEmpty', 'ConcatFromSequence',
                'ScatterND', 'GatherND', 'CumSum']
    remaining_issues = 0
    for op in warn_ops:
        n = sum(1 for node in model.graph.node if node.op_type == op)
        if n:
            print(f"  WARNING: {n} {op} node(s) remain -- may cause RKNN conversion failure")
            remaining_issues += n

    if remaining_issues == 0:
        print("  All problematic ops eliminated successfully")

    # Multi-input verification: test with different token counts
    print("\n[verify-multi] Multi-input verification...")
    for n_tokens in [5, 30, 64]:
        test_tok = np.zeros((1, seq_len), dtype=np.int64)
        test_tok[0, :n_tokens] = range(1, n_tokens + 1)
        test_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
        test_mask[0, 0, :n_tokens] = 1.0
        
        # Compute synthetic audio_length and cumulative_durations
        synth_dur = np.array([10 + i % 5 for i in range(n_tokens)], dtype=np.float32)
        synth_dur_int = np.ceil(synth_dur).astype(np.int64)
        test_audio_len = np.array([int(np.sum(synth_dur_int))], dtype=np.int64)
        # Shape: (seq_len, 1) to match Unsqueeze_6 output
        test_cumsum = np.zeros((seq_len, 1), dtype=np.float32)
        cumsum_vals = np.cumsum(synth_dur_int).astype(np.float32)
        test_cumsum[:n_tokens, 0] = cumsum_vals
        
        test_in = {
            'input': test_tok,
            'input_lengths': np.array([n_tokens], dtype=np.int64),
            'scales': np.array([0.667, 1.0, 0.8], dtype=np.float32),
            'x_mask': test_mask,
            'audio_length': test_audio_len,
            'cumulative_durations': test_cumsum,
        }
        out_multi = sess.run(None, test_in)
        a = out_multi[0].flatten()
        rms = float(np.sqrt(np.mean(a ** 2)))
        has_nan = bool(np.any(np.isnan(a)))
        has_inf = bool(np.any(np.isinf(a)))
        print(f"  n_tokens={n_tokens}: shape={a.shape}, rms={rms:.4f}, "
              f"max={float(np.max(np.abs(a))):.4f}, nan={has_nan}, inf={has_inf}")
        if has_nan or has_inf:
            print(f"  FAIL: NaN or Inf detected for n_tokens={n_tokens}")
            sys.exit(1)

    # Compare with original model (deterministic, noise_scale=0)
    orig_path = os.path.join(os.path.dirname(input_path), os.path.basename(input_path))
    # Try to find original model for comparison
    orig_candidates = [
        input_path,
        '/tmp/piper-rknn-models/en_US/en_US-lessac-medium.onnx',
    ]
    orig_sess = None
    for orig_path in orig_candidates:
        if os.path.exists(orig_path):
            try:
                orig_sess = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
                break
            except Exception:
                pass

    if orig_sess is not None:
        print(f"\n[compare] Comparing with original ({orig_path}), noise_scale=0, noise_scale_w=0...")

        # Probe original model's intermediate durations for correct alignment
        orig_model_probe = onnx.load(orig_path)
        for pname in ['/Ceil_output_0', '/CumSum_output_0']:
            vi = onnx.ValueInfoProto()
            vi.name = pname
            orig_model_probe.graph.output.append(vi)
        orig_probe_sess = ort.InferenceSession(
            orig_model_probe.SerializeToString(), providers=['CPUExecutionProvider'])

        for n_tokens in [5, 13, 30]:
            # Original: dynamic shape, both noise scales = 0 for determinism
            tokens_orig = np.zeros((1, n_tokens), dtype=np.int64)
            tokens_orig[0, :n_tokens] = range(1, n_tokens + 1)
            orig_scales = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            orig_probe_out = orig_probe_sess.run(None, {
                'input': tokens_orig,
                'input_lengths': np.array([n_tokens], dtype=np.int64),
                'scales': orig_scales,
            })
            orig_out_names = [o.name for o in orig_model_probe.graph.output]
            orig_vals = dict(zip(orig_out_names, orig_probe_out))

            a_orig = orig_vals['output'].flatten()
            orig_ceil = orig_vals['/Ceil_output_0'].flatten()  # per-token durations
            orig_cumsum = orig_vals['/CumSum_output_0'].flatten()  # cumulative durations
            orig_audio_len = int(orig_cumsum[-1])

            # Fixed: pad to seq_len with correct masks from original durations
            tokens_fixed = np.zeros((1, seq_len), dtype=np.int64)
            tokens_fixed[0, :n_tokens] = range(1, n_tokens + 1)
            cmp_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
            cmp_mask[0, 0, :n_tokens] = 1.0

            cmp_audio_len = np.array([orig_audio_len], dtype=np.int64)

            # Build cumulative_durations with shape (seq_len, 1)
            # Values are the cumulative end frame for each token
            cmp_cumsum = np.zeros((seq_len, 1), dtype=np.float32)
            cmp_cumsum[:n_tokens, 0] = orig_cumsum[:n_tokens]
            # Fill remaining positions with audio_len (so Less_1 yields 0 for padding tokens)
            cmp_cumsum[n_tokens:, 0] = float(orig_audio_len)

            fixed_out = sess.run(None, {
                'input': tokens_fixed,
                'input_lengths': np.array([n_tokens], dtype=np.int64),
                'scales': np.array([0.0, 1.0, 0.0], dtype=np.float32),
                'x_mask': cmp_mask,
                'audio_length': cmp_audio_len,
                'cumulative_durations': cmp_cumsum,
            })
            a_fixed = fixed_out[0].flatten()
            # Crop fixed model output to actual audio length (audio_length * 256 upsampling factor)
            # VITS upsampling: each duration unit = 256 audio samples (hop_size=256)
            actual_audio_len = int(cmp_audio_len[0] * 256)  # upsampling factor
            a_fixed_cropped = a_fixed[:actual_audio_len]
            
            min_len = min(len(a_orig), len(a_fixed_cropped))
            if min_len > 100:
                corr = float(np.corrcoef(a_orig[:min_len], a_fixed_cropped[:min_len])[0, 1])
            else:
                corr = 0.0
            print(f"  n_tokens={n_tokens}: orig_len={len(a_orig)} fixed_len={len(a_fixed)} "
                  f"cropped={len(a_fixed_cropped)} corr(det)={corr:.4f}")
            if corr < 0.9:
                print(f"  WARNING: correlation {corr:.4f} < 0.9, may indicate audio errors")
    else:
        print("\n[compare] Original model not found, skipping comparison")

    # Save
    onnx.save(model, output_path)
    sz = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({sz:.1f} MB)")

    # Generate WAV if --wav is specified
    if args.wav:
        import wave
        import json as json_mod
        wav_path = args.wav

        # Load phoneme_id_map from config JSON
        config_path = input_path + '.json'
        phoneme_ids = [0, 20, 61, 24, 27, 100, 3, 35, 62, 122, 24, 17, 0]
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json_mod.load(f)
            id_map = config.get('phoneme_id_map', {})
            # Phonemize "hello world" using IPA: ^ h ɛ l oʊ   w ɜː l d $
            text_phonemes = ['^', 'h', 'ɛ', 'l', 'oʊ', ' ', 'w', 'ɜː', 'l', 'd', '$']
            mapped_ids = []
            for ph in text_phonemes:
                if ph in id_map:
                    mapped_ids.extend(id_map[ph])
                else:
                    for c in ph:
                        if c in id_map:
                            mapped_ids.extend(id_map[c])
            if mapped_ids:
                phoneme_ids = mapped_ids
            print(f"  Phoneme IDs ({len(phoneme_ids)}): {phoneme_ids[:20]}...")

        n_ph = len(phoneme_ids)

        # 2-pass: get durations from original model, then run fixed model
        if orig_sess is not None:
            tok_wav_orig = np.array([phoneme_ids], dtype=np.int64)
            orig_wav_out = orig_probe_sess.run(None, {
                'input': tok_wav_orig,
                'input_lengths': np.array([n_ph], dtype=np.int64),
                'scales': np.array([0.667, 1.0, 0.8], dtype=np.float32),
            })
            orig_wav_vals = dict(zip([o.name for o in orig_model_probe.graph.output],
                                     orig_wav_out))
            wav_cumsum_vals = orig_wav_vals['/CumSum_output_0'].flatten()
            wav_audio_len_val = int(wav_cumsum_vals[-1])

            wav_tokens = np.zeros((1, seq_len), dtype=np.int64)
            wav_tokens[0, :n_ph] = phoneme_ids
            wav_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
            wav_mask[0, 0, :n_ph] = 1.0
            wav_audio_len = np.array([wav_audio_len_val], dtype=np.int64)
            # Shape: (seq_len, 1) to match Unsqueeze_6 output
            wav_cumsum = np.zeros((seq_len, 1), dtype=np.float32)
            wav_cumsum[:n_ph, 0] = wav_cumsum_vals[:n_ph]
            wav_cumsum[n_ph:, 0] = float(wav_audio_len_val)
        else:
            # Fallback: synthetic durations (~8 frames per phoneme)
            wav_tokens = np.zeros((1, seq_len), dtype=np.int64)
            wav_tokens[0, :n_ph] = phoneme_ids
            wav_mask = np.zeros((1, 1, seq_len), dtype=np.float32)
            wav_mask[0, 0, :n_ph] = 1.0
            synth_dur = np.full(n_ph, 8, dtype=np.float32)
            synth_int = np.ceil(synth_dur).astype(np.int64)
            wav_audio_len = np.array([int(synth_int.sum())], dtype=np.int64)
            # Shape: (seq_len, 1) to match Unsqueeze_6 output
            wav_cumsum = np.zeros((seq_len, 1), dtype=np.float32)
            wav_cumsum[:n_ph, 0] = np.cumsum(synth_int).astype(np.float32)

        wav_inputs = {
            'input': wav_tokens,
            'input_lengths': np.array([n_ph], dtype=np.int64),
            'scales': np.array([0.667, 1.0, 0.8], dtype=np.float32),
            'x_mask': wav_mask,
            'audio_length': wav_audio_len,
            'cumulative_durations': wav_cumsum,
        }
        wav_out = sess.run(None, wav_inputs)
        wav_audio = wav_out[0].flatten()
        audio_int16 = (wav_audio * 32767).clip(-32768, 32767).astype(np.int16)
        with wave.open(wav_path, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(audio_int16.tobytes())
        print(f"WAV: {wav_path} ({len(audio_int16) / 22050:.2f}s, rms={float(np.sqrt(np.mean(wav_audio ** 2))):.4f})")

    print("Done.")


if __name__ == '__main__':
    main()

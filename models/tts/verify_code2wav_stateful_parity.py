#!/usr/bin/env python3
"""PyTorch parity verifier for Qwen3-TTS Code2Wav stateful streaming.

This script is intentionally local-device free: it loads the PyTorch decoder,
introspects the stateful operators, and compares full-sequence decode with a
Python-managed stateful chunked decode.  The Qwen3-TTS PyTorch Code2Wav class is
not vendored in this repository, so model-specific entry points are discovered
dynamically and guarded with assertions that explain what to patch if upstream
remote code changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS"
DEFAULT_NUM_CODEBOOKS = 16
DEFAULT_CODEBOOK_SIZE = 2048
MANIFEST_PATH = Path(__file__).with_name("manifest.json")


@dataclass
class StateEntry:
    index: int
    category: str
    layer_name: str
    shape: list[int | str]
    dtype: str
    zero_init: bool
    semantics: str
    evidence: str


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def lazy_import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - import guard for local setup
        raise AssertionError(
            "PyTorch is required. Install it first, for example: "
            "uv add torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def lazy_import_transformers():
    try:
        from transformers import AutoModel, AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - import guard for local setup
        raise AssertionError(
            "transformers is required. Install it first: uv add transformers accelerate huggingface_hub"
        ) from exc
    return AutoModel, AutoModelForCausalLM


def select_device(device_name: str):
    torch = lazy_import_torch()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise AssertionError("--device cuda requested, but CUDA is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise AssertionError("--device mps requested, but MPS is not available")
    return torch.device(device_name)


def torch_dtype(dtype_name: str):
    torch = lazy_import_torch()
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    raise AssertionError(f"Unsupported dtype: {dtype_name}")


def load_decoder(model_id: str, cache_dir: str | None, device_name: str, dtype_name: str):
    """Load tokenizer-12Hz Code2Wav decoder with best-effort dynamic discovery.

    The RK runtime in this repository consumes exported RKNN variants, while the
    PyTorch Code2Wav implementation lives in HuggingFace remote code.  We avoid
    importing transformers at module import time so this script remains
    importable on hosts that have not installed the model stack yet.
    """

    torch = lazy_import_torch()
    AutoModel, AutoModelForCausalLM = lazy_import_transformers()
    device = select_device(device_name)
    dtype = torch_dtype(dtype_name)

    errors: list[str] = []
    model = None
    for cls in (AutoModel, AutoModelForCausalLM):
        try:
            model = cls.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as exc:
            errors.append(f"{cls.__name__}: {type(exc).__name__}: {exc}")

    if model is None:
        raise AssertionError(
            "Unable to load Qwen3-TTS PyTorch model. Tried AutoModel and "
            f"AutoModelForCausalLM for {model_id!r}.\n" + "\n".join(errors)
        )

    decoder = find_code2wav_decoder(model)
    decoder.to(device)
    decoder.eval()

    # Decode-only export pitfall: the ONNX/RKNN decode-only path drops these
    # weights.  PyTorch reference parity must load a full decoder with them.
    names = {name for name, _ in decoder.named_parameters()}
    missing = [
        n
        for n in (
            "quantizer.rvq_first.input_proj.weight",
            "quantizer.rvq_rest.input_proj.weight",
        )
        if not any(name.endswith(n) or n in name for name in names)
    ]
    if missing:
        print(
            json.dumps(
                {
                    "warning": "decode-only export weight check",
                    "missing_or_renamed": missing,
                    "note": "If this is an exported decode-only module, it may have dropped quantizer.rvq_first/rest.input_proj.weight.",
                }
            ),
            file=sys.stderr,
        )

    return decoder


def find_code2wav_decoder(model: Any):
    """Find the decoder module using common Qwen3-TTS remote-code names."""

    candidate_attrs = [
        "code2wav",
        "code2wav_model",
        "Code2WavModel",
        "tokenizer12hz_decode",
        "tokenizer12hz_decoder",
        "tokenizer_12hz_decoder",
        "decoder",
        "vocoder",
    ]
    for attr in candidate_attrs:
        if hasattr(model, attr):
            return getattr(model, attr)

    for name, module in model.named_modules():
        lowered = name.lower()
        cls_name = module.__class__.__name__.lower()
        if "code2wav" in lowered or "code2wav" in cls_name:
            return module
        if "tokenizer12hz" in lowered and "decode" in lowered:
            return module

    raise AssertionError(
        "Could not locate Code2Wav/tokenizer-12Hz decoder inside the loaded model. "
        "TODO: verify module name via introspect; expected attrs include "
        "code2wav, tokenizer12hz_decoder, decoder, or vocoder."
    )


def module_out_channels(module: Any) -> int | None:
    return int(getattr(module, "out_channels", 0) or 0) or None


def module_in_channels(module: Any) -> int | None:
    return int(getattr(module, "in_channels", 0) or 0) or None


def snake_beta_raw(x, alpha, beta):
    """SnakeBeta activation using raw alpha/beta.

    Pitfall assertion: do not apply exp/log conversion here. Qwen3-TTS exported
    parameters are already raw alpha/beta for reference parity.
    """

    torch = lazy_import_torch()
    assert torch.is_tensor(alpha) and torch.is_tensor(beta), "SnakeBeta alpha/beta must be raw tensors"
    return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x).pow(2)


def is_probably_causal_conv1d(name: str, module: Any) -> bool:
    cls = module.__class__.__name__.lower()
    lname = name.lower()
    padding = getattr(module, "padding", (0,))
    pad = padding[0] if isinstance(padding, tuple) else padding
    kernel = getattr(module, "kernel_size", (1,))
    kernel_size = kernel[0] if isinstance(kernel, tuple) else kernel
    if module.__class__.__name__ != "Conv1d":
        return False
    if int(kernel_size) <= 1:
        return False
    if int(pad) == 0:
        # Mandatory pitfall: padding=0 convs must NOT expose state. Exporters
        # trim zero-length state tensors and create in/out count mismatches.
        return False
    return any(tok in lname or tok in cls for tok in ("causal", "resblock", "conv", "decoder"))


def is_attention_module(name: str, module: Any) -> bool:
    lname = name.lower()
    cls = module.__class__.__name__.lower()
    return (
        "attention" in lname
        or "attn" in lname
        or "attention" in cls
        or "attn" in cls
    ) and any(hasattr(module, attr) for attr in ("num_heads", "num_key_value_heads", "head_dim", "q_proj", "k_proj"))


def attention_shape(module: Any) -> list[int | str]:
    heads = getattr(module, "num_key_value_heads", None) or getattr(module, "num_heads", "H")
    head_dim = getattr(module, "head_dim", "D")
    window = (
        getattr(module, "sliding_window", None)
        or getattr(module, "window_size", None)
        or getattr(getattr(module, "config", None), "sliding_window", None)
        or "W"
    )
    return ["B", int(heads) if isinstance(heads, int) else heads, window, int(head_dim) if isinstance(head_dim, int) else head_dim]


def introspect_state_layout(model: Any) -> list[StateEntry]:
    """Walk the decoder and build a state manifest.

    Target state count from Jetson parity work is around 37.  The true number is
    architecture-dependent, so we persist evidence rather than hard-coding it.
    """

    torch = lazy_import_torch()
    entries: list[StateEntry] = []

    for name, module in model.named_modules():
        if not name:
            continue
        if is_attention_module(name, module):
            shape = attention_shape(module)
            entries.append(
                StateEntry(
                    index=len(entries),
                    category="swa_kv_cache_k",
                    layer_name=name,
                    shape=shape,
                    dtype="model_dtype",
                    zero_init=True,
                    semantics="append+slice_left by sliding window; absolute position_offset increments by chunk length",
                    evidence=module.__class__.__name__,
                )
            )
            entries.append(
                StateEntry(
                    index=len(entries),
                    category="swa_kv_cache_v",
                    layer_name=name,
                    shape=shape,
                    dtype="model_dtype",
                    zero_init=True,
                    semantics="append+slice_left by sliding window; paired with K cache",
                    evidence=module.__class__.__name__,
                )
            )
        elif module.__class__.__name__ == "Conv1d":
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            pad = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            if int(pad) == 0:
                # Mandatory assertion for exporter parity: do not allocate state
                # for padding=0 convs, because zero-length state inputs are
                # pruned and cause RKNN/ONNX in/out count mismatches.
                assert int(pad) == 0 and not is_probably_causal_conv1d(name, module)
                continue
            if int(kernel) > 1 and is_probably_causal_conv1d(name, module):
                hist = int(kernel) - 1
                assert hist > 0, "Causal Conv1d state must be non-empty"
                entries.append(
                    StateEntry(
                        index=len(entries),
                        category="causal_conv1d_history",
                        layer_name=name,
                        shape=["B", module_in_channels(module) or "C", hist],
                        dtype=str(getattr(next(module.parameters(), torch.empty((), dtype=torch.float32)), "dtype", "model_dtype")),
                        zero_init=True,
                        semantics="left history; prepend to chunk input, update to last kernel_size-1 pre-conv input frames",
                        evidence=f"Conv1d kernel={kernel} padding={pad}",
                    )
                )
        elif module.__class__.__name__ == "ConvTranspose1d":
            kernel = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            dilation = module.dilation[0] if isinstance(module.dilation, tuple) else module.dilation
            padding = module.padding[0] if isinstance(module.padding, tuple) else module.padding
            out_pad = module.output_padding[0] if isinstance(module.output_padding, tuple) else module.output_padding
            tail = max(0, int(dilation) * (int(kernel) - 1) + int(out_pad) - 2 * int(padding) - (int(stride) - 1))
            entries.append(
                StateEntry(
                    index=len(entries),
                    category="convtranspose_pending_tail",
                    layer_name=name,
                    shape=["B", module_out_channels(module) or "Cout", tail],
                    dtype=str(getattr(next(module.parameters(), torch.empty((), dtype=torch.float32)), "dtype", "model_dtype")),
                    zero_init=True,
                    semantics="overlap-add pending tail; add bias once globally after chunk stitching, never per chunk",
                    evidence=f"ConvTranspose1d kernel={kernel} stride={stride} dilation={dilation} padding={padding} output_padding={out_pad}",
                )
            )

    for idx, entry in enumerate(entries):
        entry.index = idx

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)

    return entries


def state_tensors_from_manifest(manifest: list[StateEntry], batch: int, device: Any, dtype: Any):
    torch = lazy_import_torch()
    states = []
    for entry in manifest:
        shape = [batch if dim == "B" else (1 if isinstance(dim, str) else int(dim)) for dim in entry.shape]
        states.append(torch.zeros(*shape, device=device, dtype=dtype))
    return states


def random_codes_input(input_frames: int, device: Any):
    """Generate RVQ codes in Code2Wav layout [B, 16, T].

    The repository RKNN path stores codes as [T, 16] before vocoding.  The
    PyTorch Code2Wav contract uses [B, 16, T], so this verifier transposes only
    when calling a decoder API that explicitly requests [T, 16].
    """

    torch = lazy_import_torch()
    return torch.randint(
        low=0,
        high=DEFAULT_CODEBOOK_SIZE,
        size=(1, DEFAULT_NUM_CODEBOOKS, int(input_frames)),
        device=device,
        dtype=torch.long,
    )


def find_callable(model: Any, names: Iterable[str]) -> Callable[..., Any] | None:
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            if callable(value):
                return value
    return None


def normalize_waveform(output: Any):
    torch = lazy_import_torch()
    if isinstance(output, dict):
        for key in ("waveform", "audio", "wav", "x", "output"):
            if key in output:
                output = output[key]
                break
    elif isinstance(output, (tuple, list)):
        output = output[0]
    assert torch.is_tensor(output), f"Decoder output must be a tensor, got {type(output).__name__}"
    return output.detach()


def call_with_codes(fn: Callable[..., Any], codes: Any, position_offset: int | None = None, states: list[Any] | None = None):
    """Call a discovered decoder function while tolerating common signatures."""

    attempts = []
    if states is not None and position_offset is not None:
        attempts.extend(
            [
                lambda: fn(codes=codes, position_offset=position_offset, states=states),
                lambda: fn(codes, position_offset, *states),
                lambda: fn(codes, position_offset=position_offset, past_key_values=states),
            ]
        )
    if position_offset is not None:
        attempts.extend(
            [
                lambda: fn(codes=codes, position_offset=position_offset),
                lambda: fn(codes, position_offset),
            ]
        )
    attempts.extend([lambda: fn(codes=codes), lambda: fn(codes)])

    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            errors.append(str(exc).splitlines()[0])
    raise AssertionError(
        "Could not call decoder with discovered signatures. "
        "TODO: verify module name/signature via introspect. Last TypeErrors: "
        + " | ".join(errors[-3:])
    )


def run_full_forward(model: Any, codes: Any):
    """Run full-sequence decode."""

    torch = lazy_import_torch()
    fn = find_callable(
        model,
        (
            "decode_codes",
            "decode",
            "code2wav",
            "codes_to_wav",
            "forward",
        ),
    )
    assert fn is not None, "No full decode callable found on Code2Wav decoder"
    with torch.no_grad():
        return normalize_waveform(call_with_codes(fn, codes, position_offset=0))


def split_stateful_output(output: Any) -> tuple[Any, list[Any] | None]:
    if isinstance(output, dict):
        wav = normalize_waveform(output)
        states = output.get("states") or output.get("new_states") or output.get("past_key_values")
        return wav, list(states) if states is not None else None
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        return normalize_waveform(output[0]), list(output[1]) if output[1] is not None else None
    return normalize_waveform(output), None


def run_stateful_forward(model: Any, codes: Any, chunk_frames: int, manifest: list[StateEntry]):
    """Run chunked decode with Python-managed state list.

    Preferred path: call a low-level stream/chunk decoder when remote code
    exposes one.  Fallback path intentionally asserts instead of silently using
    top-level full forward, because that would not verify the state contract.
    """

    torch = lazy_import_torch()
    stream_fn = find_callable(
        model,
        (
            "forward_chunk",
            "decode_chunk",
            "stream_decode",
            "decode_stream",
            "forward_stream",
            "stateful_forward",
        ),
    )
    assert stream_fn is not None, (
        "No low-level stateful chunk decoder found. TODO: verify module name via "
        "introspect and bind it here; do not fall back to top-level forward for parity."
    )

    states = state_tensors_from_manifest(manifest, batch=codes.shape[0], device=codes.device, dtype=torch.float32)
    chunks = []
    position_offset = 0
    with torch.no_grad():
        for start in range(0, codes.shape[-1], int(chunk_frames)):
            chunk = codes[..., start : start + int(chunk_frames)]
            output = call_with_codes(stream_fn, chunk, position_offset=position_offset, states=states)
            wav_chunk, new_states = split_stateful_output(output)
            if new_states is not None:
                assert len(new_states) == len(states), (
                    f"State count mismatch: model returned {len(new_states)} states, "
                    f"manifest has {len(states)}"
                )
                states = new_states
            chunks.append(wav_chunk)
            position_offset += chunk.shape[-1]
    return torch.cat(chunks, dim=-1)


def convtranspose_overlap_add_chunk(module: Any, x: Any, pending_tail: Any):
    """Reference ConvTranspose1d overlap-add primitive.

    Bias pitfall: run conv_transpose1d with bias=None per chunk.  The caller must
    add module.bias exactly once after stitching the whole waveform.  Repeated
    per-chunk bias causes a fixed DC/step error at every seam.
    """

    torch = lazy_import_torch()
    y = torch.nn.functional.conv_transpose1d(
        x,
        module.weight,
        bias=None,
        stride=module.stride,
        padding=module.padding,
        output_padding=module.output_padding,
        groups=module.groups,
        dilation=module.dilation,
    )
    overlap = min(y.shape[-1], pending_tail.shape[-1])
    if overlap:
        y[..., :overlap] = y[..., :overlap] + pending_tail[..., :overlap]
    stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
    emit = min(int(stride) * x.shape[-1], y.shape[-1])
    new_tail = y[..., emit:].detach()
    return y[..., :emit], new_tail


def compare(waveform_full: Any, waveform_chunked: Any, chunk_frames: int, samples_per_frame: int | None = None):
    torch = lazy_import_torch()
    full = waveform_full.float().flatten()
    chunked = waveform_chunked.float().flatten()
    n = min(full.numel(), chunked.numel())
    assert n > 0, "Empty waveform comparison"
    full = full[:n]
    chunked = chunked[:n]
    diff = full - chunked
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())
    signal = float((full.pow(2).mean() + 1e-12).item())
    noise = float((diff.pow(2).mean() + 1e-12).item())
    snr_db = 10.0 * math.log10(signal / noise)

    seam_rms_jump_pct = 0.0
    if samples_per_frame is not None and samples_per_frame > 0:
        seam_period = int(chunk_frames) * int(samples_per_frame)
        jumps = []
        for seam in range(seam_period, n, seam_period):
            left = chunked[max(0, seam - 64) : seam]
            right = chunked[seam : min(n, seam + 64)]
            if left.numel() and right.numel():
                denom = float(left.pow(2).mean().sqrt().item()) + 1e-9
                jumps.append(abs(float(right.pow(2).mean().sqrt().item()) - denom) / denom * 100.0)
        seam_rms_jump_pct = max(jumps) if jumps else 0.0

    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "snr_db": snr_db,
        "seam_rms_jump_pct": seam_rms_jump_pct,
    }


def infer_samples_per_frame(waveform: Any, input_frames: int) -> int | None:
    n = int(waveform.flatten().numel())
    if input_frames > 0 and n % int(input_frames) == 0:
        return n // int(input_frames)
    return None


def validate_manifest(manifest: list[StateEntry]) -> None:
    padding_zero_states = [
        e for e in manifest if e.category == "causal_conv1d_history" and str(e.evidence).find("padding=0") >= 0
    ]
    assert not padding_zero_states, "padding=0 Conv1d layers must not expose state"
    if len(manifest) != 37:
        print(
            json.dumps(
                {
                    "warning": "state_count differs from Jetson target",
                    "target": 37,
                    "observed": len(manifest),
                    "manifest": str(MANIFEST_PATH),
                    "evidence": "runtime introspection of attention/causal conv/convtranspose modules",
                }
            ),
            file=sys.stderr,
        )


def run_cases(args: argparse.Namespace) -> int:
    torch = lazy_import_torch()
    random.seed(0)
    torch.manual_seed(0)
    device = select_device(args.device)

    model = load_decoder(args.model_id, args.cache_dir, args.device, args.dtype)
    manifest = introspect_state_layout(model)
    validate_manifest(manifest)

    failures = 0
    case_id = 0
    for input_frames in parse_csv_ints(args.input_frames):
        for chunk_frames in parse_csv_ints(args.chunk_frames):
            for _ in range(int(args.cases)):
                codes = random_codes_input(input_frames, device=device)
                full = run_full_forward(model, codes)
                chunked = run_stateful_forward(model, codes, chunk_frames, manifest)
                samples_per_frame = infer_samples_per_frame(full, input_frames)
                metrics = compare(full, chunked, chunk_frames, samples_per_frame)
                passed = (
                    metrics["max_abs"] <= float(args.max_abs_threshold)
                    and metrics["snr_db"] >= float(args.snr_threshold_db)
                )
                if not passed:
                    failures += 1
                print(
                    json.dumps(
                        {
                            "case_id": case_id,
                            "chunk_frames": chunk_frames,
                            "input_frames": input_frames,
                            **metrics,
                            "pass": passed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                case_id += 1
    return 1 if failures else 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--chunk-frames", default="1,4,8,16")
    parser.add_argument("--input-frames", default="8,16,44")
    parser.add_argument("--max-abs-threshold", type=float, default=2e-4)
    parser.add_argument("--snr-threshold-db", type=float, default=60.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        return run_cases(args)
    except AssertionError as exc:
        print(json.dumps({"error": str(exc), "manifest": str(MANIFEST_PATH)}), file=sys.stderr)
        return 1
    except Exception as exc:  # survive to actionable failure instead of raw import crash
        print(
            json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=8),
                    "manifest": str(MANIFEST_PATH),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

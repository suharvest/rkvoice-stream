#!/usr/bin/env python3
"""Extract and optionally convert small MOSS ONNX islands for RKNN probing."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import time
from pathlib import Path

import onnx
from onnx import helper, shape_inference, utils

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.tts.moss.convert_moss_rknn import (
    DEFAULT_RKNN_WORKSPACE,
    DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB,
    check_output_workspace,
    convert_onnx,
    prepare_prefill_onnx,
    sha256_file,
)


PRESET_CHOICES = (
    "ln2",
    "ln1",
    "mlp",
    "ln2_mlp",
    "fc_in_act",
    "fc_out",
    "ln1_cattn",
    "cattn",
    "attn_after_cattn",
    "cproj",
    "attn_residual",
    "embedding_prefix",
    "final_norm",
    "sampler_fc_in_act",
    "sampler_fc_out",
    "sampler_mlp",
    "sampler_mlps",
    "sampler_audio_head",
    "sampler_audio_heads",
    "sampler_text_lm_head",
)


def _suffix(layer: int) -> str:
    return "" if layer == 0 else f"_{layer}"


def _layer_add_before_ln2(layer: int) -> str:
    return f"/Add_{19 + layer * 5}_output_0"


def _layer_attention_input(layer: int) -> str:
    if layer == 0:
        return "/Add_15_output_0"
    return f"/Mul_{22 + (layer - 1) * 6}_output_0"


def _layer_ln1_input(layer: int) -> str:
    return f"/Mul_{16 + layer * 6}_output_0"


def _preset_spec(preset: str, layer: int) -> dict[str, object]:
    suf = _suffix(layer)
    sampler_gather = 25 + layer * 14
    if preset == "embedding_prefix":
        return {
            "inputs": ["input_ids"],
            "outputs": ["/Add_15_output_0"],
            "shape": [1, 32, 17],
            "input_shapes": {
                "input_ids": [1, 32, 17],
            },
            "case": "onnx_cpu",
            "description": "prefill embedding prefix: input_ids -> block0 attention input",
        }
    if preset == "final_norm":
        return {
            "inputs": ["/Mul_88_output_0"],
            "outputs": ["/ln_f/LayerNormalization_output_0"],
            "shape": [1, 32, 768],
            "input_shapes": {
                "/Mul_88_output_0": [1, 32, 768],
            },
            "case": "onnx_cpu",
            "description": "prefill final norm before output mask: final block hidden -> ln_f output",
        }
    if preset == "mlp":
        return {
            "inputs": [f"/ln_2{suf}/LayerNormalization_output_0"],
            "outputs": [f"/mlp/fc_out{suf}/Add_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} MLP only: normalized hidden -> mlp output",
        }
    if preset == "ln2":
        return {
            "inputs": [_layer_add_before_ln2(layer)],
            "outputs": [f"/ln_2{suf}/LayerNormalization_output_0"],
            "shape": [1, 32, 768],
            "case": "onnx_cpu",
            "description": f"block{layer} ln2 only: attention residual hidden -> normalized hidden",
        }
    if preset == "ln1":
        return {
            "inputs": [_layer_ln1_input(layer)],
            "outputs": [f"/ln_1{suf}/LayerNormalization_output_0"],
            "shape": [1, 32, 768],
            "case": "onnx_cpu",
            "description": f"block{layer} ln1 only: masked block input hidden -> normalized attention hidden",
        }
    if preset == "ln2_mlp":
        return {
            "inputs": [_layer_add_before_ln2(layer)],
            "outputs": [f"/mlp/fc_out{suf}/Add_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} ln2 + MLP: attention residual hidden -> mlp output",
        }
    if preset == "fc_in_act":
        return {
            "inputs": [f"/ln_2{suf}/LayerNormalization_output_0"],
            "outputs": [f"/mlp/act{suf}/Mul_3_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} MLP first projection + GELU approximation",
        }
    if preset == "fc_out":
        return {
            "inputs": [f"/mlp/act{suf}/Mul_3_output_0"],
            "outputs": [f"/mlp/fc_out{suf}/Add_output_0"],
            "shape": [1, 32, 3072],
            "case": "island_float",
            "description": f"block{layer} MLP output projection: activation -> mlp output",
        }
    if preset == "cproj":
        # Attention projection inputs come from the attention reshape output.
        reshape_idx = 5 if layer == 0 else 5 + layer * 6
        return {
            "inputs": [f"/Reshape_{reshape_idx}_output_0"],
            "outputs": [f"/c_proj{suf}/Add_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} attention output projection only",
        }
    if preset == "cattn":
        return {
            "inputs": [f"/ln_1{suf}/LayerNormalization_output_0"],
            "outputs": [f"/c_attn{suf}/Add_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} fused attention qkv projection only: ln1 hidden -> qkv",
        }
    if preset == "ln1_cattn":
        return {
            "inputs": [_layer_ln1_input(layer)],
            "outputs": [f"/c_attn{suf}/Add_output_0"],
            "shape": [1, 32, 768],
            "case": "island_float",
            "description": f"block{layer} ln1 + fused attention qkv projection: masked hidden -> qkv",
        }
    if preset == "sampler_fc_in_act":
        return {
            "inputs": [f"/ln_2{suf}/LayerNormalization_output_0"],
            "outputs": [f"/mlp/act{suf}/Mul_3_output_0"],
            "shape": [1, 1, 768],
            "case": "island_float",
            "description": f"sampler local block {layer} MLP first projection + GELU approximation",
        }
    if preset == "sampler_fc_out":
        return {
            "inputs": [f"/mlp/act{suf}/Mul_3_output_0"],
            "outputs": [f"/mlp/fc_out{suf}/Add_output_0"],
            "shape": [1, 1, 3072],
            "case": "island_float",
            "description": f"sampler local block {layer} MLP output projection",
        }
    if preset == "sampler_mlp":
        return {
            "inputs": [f"/ln_2{suf}/LayerNormalization_output_0"],
            "outputs": [f"/mlp/fc_out{suf}/Add_output_0"],
            "shape": [1, 1, 768],
            "case": "island_float",
            "description": f"sampler local block {layer} fused MLP: normalized hidden -> mlp output",
        }
    if preset == "sampler_mlps":
        return {
            "inputs": [
                "/ln_2/LayerNormalization_output_0",
                *[f"/ln_2_{i}/LayerNormalization_output_0" for i in range(1, 17)],
            ],
            "outputs": [
                "/mlp/fc_out/Add_output_0",
                *[f"/mlp/fc_out_{i}/Add_output_0" for i in range(1, 17)],
            ],
            "input_shapes": {
                "/ln_2/LayerNormalization_output_0": [1, 1, 768],
                **{f"/ln_2_{i}/LayerNormalization_output_0": [1, 1, 768] for i in range(1, 17)},
            },
            "case": "island_float",
            "description": "sampler local MLPs 0-16: 17 normalized hidden rows -> 17 MLP outputs",
        }
    if preset == "sampler_audio_head":
        if not 0 <= layer <= 15:
            raise ValueError("sampler_audio_head layer must be in [0, 15]")
        return {
            "inputs": [f"/Gather_{sampler_gather}_output_0"],
            "outputs": [f"/audio_lm_heads.{layer}/MatMul_output_0"],
            "shape": [1, 768],
            "case": "island_float",
            "description": f"sampler audio head {layer}: hidden -> 1024 logits",
        }
    if preset == "sampler_audio_heads":
        return {
            "inputs": [f"/Gather_{25 + i * 14}_output_0" for i in range(16)],
            "outputs": [f"/audio_lm_heads.{i}/MatMul_output_0" for i in range(16)],
            "input_shapes": {f"/Gather_{25 + i * 14}_output_0": [1, 768] for i in range(16)},
            "case": "island_float",
            "description": "sampler audio heads 0-15: 16 hidden rows -> 16x1024 logits",
        }
    if preset == "sampler_text_lm_head":
        return {
            "inputs": ["/Gather_11_output_0"],
            "outputs": ["/text_lm_head/MatMul_output_0"],
            "shape": [1, 768],
            "case": "island_float",
            "description": "sampler text LM head: hidden -> 16384 logits",
        }
    if preset == "attn_residual":
        return {
            "inputs": [_layer_attention_input(layer), "attention_mask"],
            "outputs": [
                _layer_add_before_ln2(layer),
                f"present_key_{layer}",
                f"present_value_{layer}",
            ],
            "shape": [1, 32, 768],
            "input_shapes": {
                _layer_attention_input(layer): [1, 32, 768],
                "attention_mask": [1, 32],
            },
            "case": "onnx_cpu",
            "description": f"block{layer} attention + residual CPU slice: hidden -> ln2_mlp input + KV",
        }
    if preset == "attn_after_cattn":
        return {
            "inputs": [f"/c_attn{suf}/Add_output_0", _layer_attention_input(layer), "attention_mask"],
            "outputs": [
                _layer_add_before_ln2(layer),
                f"present_key_{layer}",
                f"present_value_{layer}",
            ],
            "input_shapes": {
                f"/c_attn{suf}/Add_output_0": [1, 32, 2304],
                _layer_attention_input(layer): [1, 32, 768],
                "attention_mask": [1, 32],
            },
            "case": "onnx_cpu",
            "description": f"block{layer} attention suffix after qkv projection: qkv + residual hidden -> ln2_mlp input + KV",
        }
    raise ValueError(f"Unsupported preset: {preset}")


def _sanitize(text: str) -> str:
    return text.strip("/").replace("/", "_").replace(":", "_")


def _set_dynamic_input_shapes(path: Path, shapes: dict[str, list[int]]) -> None:
    model = onnx.load(str(path), load_external_data=True)
    for inp in model.graph.input:
        if inp.name not in shapes:
            continue
        dims = inp.type.tensor_type.shape.dim
        for dim, value in zip(dims, shapes[inp.name], strict=True):
            dim.dim_param = ""
            dim.dim_value = int(value)
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )


def extract_island(source: Path, out_path: Path, inputs: list[str], outputs: list[str], shapes: dict[str, list[int]]) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        inferred = Path(td) / "source.inferred.onnx"
        try:
            shape_inference.infer_shapes_path(str(source), str(inferred), data_prop=True)
            data_files = list(source.parent.glob("*.data"))
            for data_file in data_files:
                target = inferred.parent / data_file.name
                if not target.exists():
                    target.write_bytes(data_file.read_bytes())
            extract_source = inferred
        except Exception:
            extract_source = source
        utils.extract_model(
            str(extract_source),
            str(out_path),
            input_names=inputs,
            output_names=outputs,
            check_model=False,
        )
    _set_dynamic_input_shapes(out_path, shapes)
    onnx.checker.check_model(str(out_path))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--preset", required=True, choices=PRESET_CHOICES)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--convert-rknn", action="store_true")
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--require-rknn-workspace",
        action="store_true",
        help="require --out-dir to be under a prepared non-root RKNN workspace before writing artifacts",
    )
    parser.add_argument("--rknn-workspace", type=Path, default=DEFAULT_RKNN_WORKSPACE)
    parser.add_argument("--rknn-workspace-min-free-mb", type=int, default=DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.preset == "sampler_audio_head" and not 0 <= args.layer <= 15:
        raise SystemExit("--layer must be in [0, 15] for sampler_audio_head")
    if args.preset in {"sampler_fc_in_act", "sampler_fc_out", "sampler_mlp"} and not 0 <= args.layer <= 16:
        raise SystemExit("--layer must be in [0, 16] for sampler local block presets")
    if args.preset not in {
        "embedding_prefix",
        "final_norm",
        "sampler_audio_head",
        "sampler_audio_heads",
        "sampler_text_lm_head",
        "sampler_fc_in_act",
        "sampler_fc_out",
        "sampler_mlp",
        "sampler_mlps",
    } and not 0 <= args.layer <= 11:
        raise SystemExit("--layer must be in [0, 11]")
    if args.require_rknn_workspace:
        workspace_report = check_output_workspace(
            out_dir=args.out_dir,
            workspace=args.rknn_workspace,
            min_free_mb=args.rknn_workspace_min_free_mb,
        )
        if not workspace_report["passed"]:
            raise RuntimeError(
                "RKNN workspace preflight failed: " + "; ".join(str(error) for error in workspace_report["errors"])
            )
    spec = _preset_spec(args.preset, args.layer)
    if "input_shapes" in spec:
        input_shapes = {
            str(name): [args.seq_len if int(dim) == 32 else int(dim) for dim in shape]
            for name, shape in dict(spec["input_shapes"]).items()
        }
    else:
        input_shapes = {
            name: [args.seq_len if dim == 32 else dim for dim in spec["shape"]]
            for name in spec["inputs"]
        }
    if args.preset.startswith("sampler_"):
        if args.preset == "sampler_text_lm_head":
            base = "moss_sampler_text_lm_head"
        else:
            base = f"moss_{args.preset}{args.layer}"
    elif args.preset in {"embedding_prefix", "final_norm"}:
        base = f"moss_{args.preset}.s{args.seq_len}"
    else:
        base = f"moss_block{args.layer}_{args.preset}.s{args.seq_len}"
    onnx_path = args.out_dir / f"{base}.onnx"
    t0 = time.perf_counter()
    extract_island(args.onnx, onnx_path, list(spec["inputs"]), list(spec["outputs"]), input_shapes)
    if args.preset in {"attn_residual", "attn_after_cattn"}:
        prepare_prefill_onnx(onnx_path, onnx_path, args.seq_len, output_mode="full")
    result: dict[str, object] = {
        "preset": args.preset,
        "layer": args.layer,
        "description": spec["description"],
        "onnx": str(onnx_path),
        "inputs": spec["inputs"],
        "outputs": spec["outputs"],
        "input_shapes": input_shapes,
        "case": spec["case"],
        "extract_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_sha256": sha256_file(onnx_path),
    }
    if args.convert_rknn:
        rknn_path = args.out_dir / f"{base}.{args.precision}.{args.target}.rknn"
        log_path = args.out_dir / "build_logs" / f"{rknn_path.name}.log"
        log_path.parent.mkdir(exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            build = convert_onnx(
                onnx_path=onnx_path,
                rknn_path=rknn_path,
                target=args.target,
                precision=args.precision,
                overrides={"batch": 1, "prefill_seq": args.seq_len},
                outputs=None,
                optimization_level=args.optimization_level,
                disable_rules=[],
                dataset=None,
                force=args.force,
                verbose=args.verbose,
            )
        result["rknn"] = str(rknn_path)
        result["rknn_build"] = build
        result["build_log"] = str(log_path)
    report_path = args.out_dir / f"{base}.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

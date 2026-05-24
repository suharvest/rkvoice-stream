#!/usr/bin/env python3
"""Convert MOSS-TTS-Nano paged-FP16 ONNX bundle to fixed-bucket RKNN.

The expected input is the Jetson ONNX bundle produced by the MOSS port:

  <onnx-bundle>/
    MOSS-TTS-Nano-100M-ONNX/
      moss_tts_prefill.onnx
      moss_tts_decode_step.onnx
      moss_tts_local_fixed_sampled_frame.onnx
      moss_tts_global_shared.data
      moss_tts_local_shared.data
      tokenizer.model
      tts_browser_onnx_meta.json
    MOSS-Audio-Tokenizer-Nano-ONNX/
      moss_audio_tokenizer_decode_step.onnx
      moss_audio_tokenizer_decode_shared.data
      codec_browser_onnx_meta.json

RKNN has much weaker dynamic shape support than TensorRT, so this script builds
fixed buckets and writes a production manifest with sha256/size metadata.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from models.tts.moss.verify_rknn_artifact_workspace import (
    DEFAULT_MIN_FREE_MB as DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB,
)
from models.tts.moss.verify_rknn_artifact_workspace import verify_workspace as verify_rknn_workspace


FLOAT_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "tf32": "tfloat32",
}

PREFILL_BUCKETS = (32, 64, 128, 256)
DECODE_PAST_BUCKETS = (1, 32, 64, 128, 256, 512)
CODEC_FRAME_BUCKETS = (1, 4, 8)
DEFAULT_RKNN_WORKSPACE = Path("/mnt/rknn-workspace/moss-rknn-workspace")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_int_list(text: str, default: Iterable[int]) -> tuple[int, ...]:
    if not text:
        return tuple(default)
    return tuple(int(part) for part in text.split(",") if part.strip())


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def check_output_workspace(out_dir: Path, workspace: Path, min_free_mb: int) -> dict:
    report = verify_rknn_workspace(workspace=workspace, min_free_mb=min_free_mb)
    errors = list(report.get("errors") or [])
    if not _is_under(out_dir, workspace):
        errors.append(f"out_dir must be under RKNN workspace {workspace}: {out_dir}")
    return {
        "passed": not errors,
        "errors": errors,
        "out_dir": str(out_dir),
        "workspace": str(workspace),
        "workspace_report": report,
    }


def input_shapes(onnx_path: Path, overrides: dict[str, int]) -> tuple[list[str], list[list[int]]]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    names: list[str] = []
    shapes: list[list[int]] = []
    for inp in model.graph.input:
        tt = inp.type.tensor_type
        names.append(inp.name)
        shape: list[int] = []
        for dim in tt.shape.dim:
            if dim.dim_value:
                shape.append(int(dim.dim_value))
            elif dim.dim_param in overrides:
                shape.append(int(overrides[dim.dim_param]))
            elif dim.dim_param == "batch":
                shape.append(1)
            else:
                raise RuntimeError(f"{onnx_path.name}: no fixed value for input {inp.name} dim {dim.dim_param!r}")
        shapes.append(shape)
    return names, shapes


def input_dtypes(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [
        TensorProto.DataType.Name(inp.type.tensor_type.elem_type).lower()
        for inp in model.graph.input
    ]


def prepare_prefill_onnx(input_path: Path, output_path: Path, seq_len: int, output_mode: str) -> Path:
    """Replace CumSum(attention_mask, axis=-1) with MatMul for RKNN runtime.

    RKNN toolkit can export the original CumSum as a CPU op, but RK3576
    librknnrt 2.3.2 cannot execute that CPU fallback and segfaults on inference.
    For fixed prefill buckets, CumSum over the last axis is equivalent to
    MatMul with an upper-triangular all-ones matrix.
    """

    model = onnx.load(str(input_path), load_external_data=True)
    if output_mode == "global_hidden":
        outputs = [out for out in model.graph.output if out.name == "global_hidden"]
        if len(outputs) != 1:
            raise RuntimeError(f"Expected one global_hidden output in {input_path}, found {len(outputs)}")
        del model.graph.output[:]
        model.graph.output.extend(outputs)
    elif output_mode != "full":
        raise ValueError(f"Unsupported prefill output mode: {output_mode}")

    cumsum_nodes = [node for node in model.graph.node if node.op_type == "CumSum"]
    if not cumsum_nodes:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(output_path), save_as_external_data=True, all_tensors_to_one_file=True)
        return output_path
    if len(cumsum_nodes) != 1:
        raise RuntimeError(f"Expected one CumSum in {input_path}, found {len(cumsum_nodes)}")

    node = cumsum_nodes[0]
    data_input = node.input[0]
    cumsum_output = node.output[0]
    base_name = node.name or "CumSum"
    mat_name = f"{base_name}_upper_triangular_ones"
    mat_value = np.triu(np.ones((seq_len, seq_len), dtype=np.float32))
    model.graph.initializer.append(numpy_helper.from_array(mat_value, name=mat_name))

    cast_in = helper.make_node(
        "Cast",
        inputs=[data_input],
        outputs=[f"{base_name}_input_float"],
        name=f"{base_name}_input_cast_float",
        to=TensorProto.FLOAT,
    )
    matmul_out = f"{base_name}_matmul_float"
    matmul = helper.make_node(
        "MatMul",
        inputs=[cast_in.output[0], mat_name],
        outputs=[matmul_out],
        name=f"{base_name}_as_matmul",
    )
    cast_out = helper.make_node(
        "Cast",
        inputs=[matmul_out],
        outputs=[cumsum_output],
        name=f"{base_name}_output_cast_int64",
        to=TensorProto.INT64,
    )
    idx = list(model.graph.node).index(node)
    model.graph.node.remove(node)
    model.graph.node.insert(idx, cast_out)
    model.graph.node.insert(idx, matmul)
    model.graph.node.insert(idx, cast_in)

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return output_path


def _constant_bool(model: onnx.ModelProto, output_name: str) -> bool | None:
    for node in model.graph.node:
        if node.op_type != "Constant" or output_name not in node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                value = numpy_helper.to_array(attr.t)
                if value.shape == () or value.size == 1:
                    return bool(value.reshape(-1)[0])
    return None


def _cast_to(model: onnx.ModelProto, node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _graph_inputs_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    return {inp.name: inp for inp in model.graph.input}


def _rewrite_input_casts_to_int64(model: onnx.ModelProto) -> int:
    inputs = _graph_inputs_by_name(model)
    rewritten = 0
    for node in model.graph.node:
        if len(node.input) != 1 or node.input[0] not in inputs:
            continue
        if _cast_to(model, node) != TensorProto.INT64:
            continue
        tensor_type = inputs[node.input[0]].type.tensor_type
        tensor_type.elem_type = TensorProto.INT64
        del node.attribute[:]
        node.op_type = "Identity"
        node.name = f"{node.name or 'Cast'}_int64_input_identity"
        rewritten += 1
    return rewritten


def prepare_codec_onnx(input_path: Path, output_path: Path) -> Path:
    """Apply exact RKNN compatibility rewrites to codec decode-step ONNX."""

    model = onnx.load(str(input_path), load_external_data=True)
    rewritten = 0
    rewritten += _rewrite_input_casts_to_int64(model)
    for node in model.graph.node:
        if node.op_type != "Xor" or len(node.input) != 2:
            continue
        const_value = _constant_bool(model, node.input[1])
        data_input = node.input[0]
        if const_value is None:
            const_value = _constant_bool(model, node.input[0])
            data_input = node.input[1]
        if const_value is False:
            del node.input[:]
            node.input.extend([data_input])
            node.op_type = "Identity"
            node.name = f"{node.name or 'Xor'}_as_identity"
            rewritten += 1
        elif const_value is True:
            del node.input[:]
            node.input.extend([data_input])
            node.op_type = "Not"
            node.name = f"{node.name or 'Xor'}_as_not"
            rewritten += 1

    if rewritten:
        onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return output_path


def simplify_static_onnx(input_path: Path, output_path: Path, overrides: dict[str, int]) -> Path:
    """Run onnxsim with the fixed RKNN input shapes used for conversion."""

    import onnxsim

    input_names, shape_list = input_shapes(input_path, overrides)
    input_shapes_by_name = dict(zip(input_names, shape_list))
    model = onnx.load(str(input_path), load_external_data=True)
    simplified, ok = onnxsim.simplify(model, input_shapes=input_shapes_by_name)
    if not ok:
        raise RuntimeError(f"onnxsim failed validation for {input_path}")
    onnx.checker.check_model(simplified)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        simplified,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path.with_suffix(".data").name,
        size_threshold=1024,
        convert_attribute=False,
    )
    return output_path


def convert_onnx(
    onnx_path: Path,
    rknn_path: Path,
    target: str,
    precision: str,
    overrides: dict[str, int],
    outputs: list[str] | None,
    optimization_level: int,
    disable_rules: list[str],
    dataset: str | None,
    force: bool,
    verbose: bool,
) -> dict:
    from rknn.api import RKNN

    if rknn_path.exists() and not force:
        return {
            "name": rknn_path.name,
            "status": "SKIP",
            "reason": "exists",
            "size_bytes": rknn_path.stat().st_size,
            "sha256": sha256_file(rknn_path),
        }

    input_names, shape_list = input_shapes(onnx_path, overrides)
    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    if rknn_path.exists():
        rknn_path.unlink()

    t0 = time.perf_counter()
    rknn = RKNN(verbose=verbose)
    try:
        config_kwargs = {
            "target_platform": target,
            "optimization_level": optimization_level,
        }
        if disable_rules:
            config_kwargs["disable_rules"] = disable_rules
        if precision in FLOAT_DTYPES:
            config_kwargs["float_dtype"] = FLOAT_DTYPES[precision]
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.config returned {ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=input_names,
            input_size_list=shape_list,
            outputs=outputs,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx returned {ret}")

        do_quantization = precision == "int8"
        if do_quantization and not dataset:
            raise RuntimeError("INT8 conversion requires --dataset")
        ret = rknn.build(do_quantization=do_quantization, dataset=dataset)
        if ret != 0:
            raise RuntimeError(f"rknn.build returned {ret}")

        ret = rknn.export_rknn(str(rknn_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn returned {ret}")
    finally:
        rknn.release()

    elapsed = time.perf_counter() - t0
    return {
        "name": rknn_path.name,
        "status": "OK",
        "elapsed_s": round(elapsed, 3),
        "input_names": input_names,
        "input_shapes": shape_list,
        "input_dtypes": input_dtypes(onnx_path),
        "disable_rules": disable_rules,
        "size_bytes": rknn_path.stat().st_size,
        "sha256": sha256_file(rknn_path),
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def artifact_entry(root: Path, rel: str, required: bool = True) -> dict:
    path = root / rel
    entry = {"path": rel, "required": required}
    if path.exists() and path.is_file():
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def write_manifest(out_dir: Path, target: str, precision: str, artifact_set: str, build_results: list[dict]) -> None:
    required = [
        "tokenizer.model",
        "tts_browser_onnx_meta.json",
        "codec_browser_onnx_meta.json",
        *[f"moss_tts_prefill.s{s}.{precision}.{target}.rknn" for s in PREFILL_BUCKETS],
        *[f"moss_tts_decode_step.p{p}.{precision}.{target}.rknn" for p in DECODE_PAST_BUCKETS],
        f"moss_tts_local_fixed_sampled_frame.{precision}.{target}.rknn",
        *[f"codec_decode_step.f{f}.int64offset.{precision}.{target}.rknn" for f in CODEC_FRAME_BUCKETS],
    ]
    manifest = {
        "model_id": "moss-tts-nano-rknn",
        "target_platform": target,
        "precision": precision,
        "sample_rate": 48000,
        "channels": 2,
        "artifact_set": artifact_set,
        "source": "MOSS paged-FP16 ONNX bundle",
        "buckets": {
            "prefill_seq": list(PREFILL_BUCKETS),
            "decode_past_seq": list(DECODE_PAST_BUCKETS),
            "codec_frames": list(CODEC_FRAME_BUCKETS),
        },
        "artifacts": [artifact_entry(out_dir, rel) for rel in required],
        "build_results": build_results,
        "production_gates": {
            "max_ttfa_ms": 500,
            "max_rtf": 0.75,
            "max_asr_cer": 0.15,
            "min_non_silent_rms": 0.02,
        },
    }
    (out_dir / "moss-rknn-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-bundle", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "tf32", "int8"])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument(
        "--disable-rules",
        default="",
        help="Comma-separated RKNN optimizer rules to disable, e.g. merge_conv_channel_inner_perm",
    )
    parser.add_argument("--prefill-buckets", default="32,64,128,256")
    parser.add_argument("--prefill-output-mode", choices=["full", "global_hidden"], default="full")
    parser.add_argument(
        "--prefill-crop-output",
        default="",
        help="Optional intermediate tensor name used as the only RKNN output for prefill crash probes.",
    )
    parser.add_argument("--decode-past-buckets", default="1,32,64,128,256,512")
    parser.add_argument("--codec-frame-buckets", default="1,4,8")
    parser.add_argument("--codec-output-mode", choices=["full", "audio_only"], default="full")
    parser.add_argument(
        "--codec-crop-output",
        default="",
        help="Optional intermediate tensor name used as the only RKNN output for codec crash probes.",
    )
    parser.add_argument(
        "--codec-simplify",
        action="store_true",
        help="Run onnxsim after codec graph surgery with fixed bucket shapes before RKNN conversion.",
    )
    parser.add_argument("--only", choices=["all", "prefill", "decode", "sampler", "codec"], default="all")
    parser.add_argument("--artifact-set", default="")
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

    global PREFILL_BUCKETS, DECODE_PAST_BUCKETS, CODEC_FRAME_BUCKETS
    PREFILL_BUCKETS = parse_int_list(args.prefill_buckets, PREFILL_BUCKETS)
    DECODE_PAST_BUCKETS = parse_int_list(args.decode_past_buckets, DECODE_PAST_BUCKETS)
    CODEC_FRAME_BUCKETS = parse_int_list(args.codec_frame_buckets, CODEC_FRAME_BUCKETS)
    disable_rules = parse_string_list(args.disable_rules)

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

    tts_dir = args.onnx_bundle / "MOSS-TTS-Nano-100M-ONNX"
    codec_dir = args.onnx_bundle / "MOSS-Audio-Tokenizer-Nano-ONNX"
    if not tts_dir.exists():
        tts_dir = args.onnx_bundle
    if not codec_dir.exists():
        codec_dir = args.onnx_bundle

    prefill_onnx = tts_dir / "moss_tts_prefill.onnx"
    decode_onnx = tts_dir / "moss_tts_decode_step.onnx"
    sampler_onnx = tts_dir / "moss_tts_local_fixed_sampled_frame.onnx"
    codec_onnx = codec_dir / "moss_audio_tokenizer_decode_step.onnx"
    required_inputs = {
        "prefill": prefill_onnx,
        "decode": decode_onnx,
        "sampler": sampler_onnx,
        "codec": codec_onnx,
    }
    for name, path in required_inputs.items():
        if (args.only == "all" or args.only == name) and not path.exists():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "build_logs"
    log_dir.mkdir(exist_ok=True)

    for name in ("tokenizer.model", "tts_browser_onnx_meta.json", "browser_poc_manifest.json"):
        copy_if_exists(tts_dir / name, args.out_dir / name)
    for name in ("codec_browser_onnx_meta.json",):
        copy_if_exists(codec_dir / name, args.out_dir / name)

    build_results: list[dict] = []

    def run_one(
        label: str,
        onnx_path: Path,
        rknn_name: str,
        overrides: dict[str, int],
        outputs: list[str] | None = None,
    ) -> None:
        log_path = log_dir / f"{rknn_name}.log"
        print(f"[build] {label} -> {rknn_name}")
        try:
            with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                result = convert_onnx(
                    onnx_path=onnx_path,
                    rknn_path=args.out_dir / rknn_name,
                    target=args.target,
                    precision=args.precision,
                    overrides=overrides,
                    outputs=outputs,
                    optimization_level=args.optimization_level,
                    disable_rules=disable_rules,
                    dataset=args.dataset,
                    force=args.force,
                    verbose=args.verbose,
                )
            print(f"[{result['status']}] {rknn_name} size={result.get('size_bytes', 0)} log={log_path}")
            build_results.append(result)
        except Exception as exc:
            print(f"[FAIL] {rknn_name}: {exc} log={log_path}")
            build_results.append({"name": rknn_name, "status": "FAIL", "error": str(exc), "log": str(log_path)})
            raise

    if args.only in ("all", "prefill"):
        for seq in PREFILL_BUCKETS:
            patched_prefill = prepare_prefill_onnx(
                prefill_onnx,
                args.out_dir / "_fixed_onnx" / f"moss_tts_prefill.s{seq}.{args.prefill_output_mode}.onnx",
                seq,
                args.prefill_output_mode,
            )
            crop_outputs = [args.prefill_crop_output] if args.prefill_crop_output else None
            mode_suffix = "" if args.prefill_output_mode == "full" else f".{args.prefill_output_mode}"
            if args.prefill_crop_output:
                safe_crop = args.prefill_crop_output.strip("/").replace("/", "_").replace(":", "_")
                mode_suffix = f".crop_{safe_crop}"
            run_one(
                f"prefill seq={seq} output={args.prefill_output_mode} crop={args.prefill_crop_output or 'none'}",
                patched_prefill,
                f"moss_tts_prefill.s{seq}{mode_suffix}.{args.precision}.{args.target}.rknn",
                {"batch": 1, "prefill_seq": seq},
                outputs=crop_outputs,
            )

    if args.only in ("all", "decode"):
        for past in DECODE_PAST_BUCKETS:
            run_one(
                f"decode past={past}",
                decode_onnx,
                f"moss_tts_decode_step.p{past}.{args.precision}.{args.target}.rknn",
                {"batch": 1, "step_seq": 1, "past_seq": past},
            )

    if args.only in ("all", "sampler"):
        run_one(
            "local fixed sampled frame",
            sampler_onnx,
            f"moss_tts_local_fixed_sampled_frame.{args.precision}.{args.target}.rknn",
            {"batch": 1},
        )

    if args.only in ("all", "codec"):
        for frames in CODEC_FRAME_BUCKETS:
            codec_overrides = {"batch": 1, "code_length": frames}
            patched_codec = prepare_codec_onnx(
                codec_onnx,
                args.out_dir / "_fixed_onnx" / f"moss_audio_tokenizer_decode_step.f{frames}.onnx",
            )
            if args.codec_simplify:
                patched_codec = simplify_static_onnx(
                    patched_codec,
                    args.out_dir / "_fixed_onnx" / f"moss_audio_tokenizer_decode_step.f{frames}.sim.onnx",
                    codec_overrides,
                )
            codec_outputs = ["audio", "audio_lengths"] if args.codec_output_mode == "audio_only" else None
            output_suffix = ".int64offset"
            if args.codec_output_mode == "audio_only":
                output_suffix += ".audio_only"
            if args.codec_crop_output:
                codec_outputs = [args.codec_crop_output]
                safe_crop = args.codec_crop_output.strip("/").replace("/", "_").replace(":", "_")
                output_suffix += f".crop_{safe_crop}"
            run_one(
                f"codec decode frames={frames} output={args.codec_output_mode} crop={args.codec_crop_output or 'none'}",
                patched_codec,
                f"codec_decode_step.f{frames}{output_suffix}.{args.precision}.{args.target}.rknn",
                codec_overrides,
                outputs=codec_outputs,
            )

    write_manifest(args.out_dir, args.target, args.precision, args.artifact_set, build_results)
    print(f"[done] manifest={args.out_dir / 'moss-rknn-manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Probe MOSS RKNN artifacts on a real RK3576/RK3588 device.

Each inference runs in a child process so an RKNN runtime segfault is recorded
as a failed probe instead of killing the whole sweep.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def _parse_bucket(pattern: str, text: str, default: int) -> int:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


def _case_from_name(path: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    name = path.name
    if "min_layernorm" in name:
        return "layernorm"
    if re.search(r"block\d+_(mlp|ln2_mlp|fc_in_act|fc_out|ln1_cattn|cattn|cproj)", name):
        return "island_float"
    if "moss_sampler_audio_heads" in name:
        return "sampler_audio_heads"
    if "moss_sampler_mlps" in name:
        return "sampler_mlps"
    if re.search(r"moss_sampler_(fc_in_act|fc_out|mlp|audio_head|text_lm_head)", name):
        return "sampler_island_float"
    if re.search(r"block\d+_attn_residual", name):
        return "attn_residual"
    if "codec_decode_step" in name or "audio_tokenizer_decode" in name:
        return "codec"
    if ".crop_" in name:
        return "prefill_tokens"
    if ".suffix_" in name:
        return "prefill_hidden_mask"
    if "prefill" in name:
        return "prefill"
    if "decode_step" in name:
        return "decode"
    if "sampled_frame" in name or "sampler" in name:
        return "sampler"
    raise ValueError(f"Cannot infer MOSS RKNN probe case from filename: {path.name}")


def _inputs_for_case(case: str, path: Path) -> list[np.ndarray]:
    name = path.name
    if case == "layernorm":
        return [np.linspace(-1.0, 1.0, num=32 * 768, dtype=np.float32).reshape(1, 32, 768)]
    if case == "island_float":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        width = 768
        if "fc_out" in name:
            width = 3072
        return [np.linspace(-0.5, 0.5, num=seq * width, dtype=np.float32).reshape(1, seq, width)]
    if case == "sampler_island_float":
        width = 3072 if "sampler_fc_out" in name else 768
        if "sampler_fc_" in name or "sampler_mlp" in name:
            return [np.linspace(-0.5, 0.5, num=width, dtype=np.float32).reshape(1, 1, width)]
        return [np.linspace(-0.5, 0.5, num=width, dtype=np.float32).reshape(1, width)]
    if case == "sampler_audio_heads":
        base = np.linspace(-0.5, 0.5, num=768, dtype=np.float32).reshape(1, 768)
        return [base + np.float32(i * 0.01) for i in range(16)]
    if case == "sampler_mlps":
        base = np.linspace(-0.5, 0.5, num=768, dtype=np.float32).reshape(1, 1, 768)
        return [base + np.float32(i * 0.01) for i in range(17)]
    if case == "attn_residual":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        hidden = np.linspace(-0.5, 0.5, num=seq * 768, dtype=np.float32).reshape(1, seq, 768)
        attention_mask = np.ones((1, seq), dtype=np.int32)
        return [hidden, attention_mask]
    if case == "prefill":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        input_ids = np.zeros((1, seq, 17), dtype=np.int32)
        input_ids[:, :, 0] = 1
        attention_mask = np.ones((1, seq), dtype=np.int32)
        return [input_ids, attention_mask]
    if case == "prefill_tokens":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        input_ids = np.zeros((1, seq, 17), dtype=np.int32)
        input_ids[:, :, 0] = 1
        return [input_ids]
    if case == "prefill_hidden":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        return [np.zeros((1, seq, 768), dtype=np.float32)]
    if case == "prefill_hidden_mask":
        seq = _parse_bucket(r"\.s(\d+)", name, 32)
        return [
            np.zeros((1, seq, 768), dtype=np.float32),
            np.ones((1, seq), dtype=np.int32),
        ]
    if case == "decode":
        past = _parse_bucket(r"\.p(\d+)", name, 1)
        inputs: list[np.ndarray] = [
            np.ones((1, 1, 17), dtype=np.int32),
            np.asarray([past], dtype=np.int32),
        ]
        for _layer in range(12):
            inputs.append(np.zeros((1, past, 12, 64), dtype=np.float32))
            inputs.append(np.zeros((1, past, 12, 64), dtype=np.float32))
        return inputs
    if case == "sampler":
        return [
            np.zeros((1, 768), dtype=np.float32),
            np.zeros((1, 16, 1024), dtype=np.int32),
            np.asarray([0.5], dtype=np.float32),
            np.full((1, 16), 0.5, dtype=np.float32),
        ]
    if case == "codec":
        frames = _parse_bucket(r"\.f(\d+)", name, 1)
        inputs: list[np.ndarray] = [
            np.zeros((1, frames, 16), dtype=np.int32),
            np.asarray([frames], dtype=np.int32),
        ]
        for _ in range(4):
            inputs.append(np.zeros((1,), dtype=np.int32))
        # Streaming codec decode_step carries attention cache state as explicit
        # inputs. The current MOSS codec export uses progressively larger cache
        # buckets across its 12 attention blocks.
        for cache_len in [500, 500, 500, 500, 800, 800, 1200, 1200, 1600, 1600, 1600, 1600]:
            inputs.append(np.zeros((1,), dtype=np.int32))
            inputs.append(np.zeros((1, 4, cache_len, 64), dtype=np.float32))
            inputs.append(np.zeros((1, 4, cache_len, 64), dtype=np.float32))
            inputs.append(np.full((1, cache_len), -1, dtype=np.int32))
        return inputs
    raise ValueError(f"Unsupported probe case: {case}")


def _pass_through(inputs: list[np.ndarray], mode: str) -> list[int] | None:
    if mode == "none":
        return None
    if mode == "all":
        return [1] * len(inputs)
    if mode == "auto":
        values = [1 if np.issubdtype(x.dtype, np.integer) else 0 for x in inputs]
        return values if any(values) else None
    raise ValueError(f"Unsupported pass-through mode: {mode}")


def _summarize_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    summary = []
    for out in outputs:
        arr = np.asarray(out)
        finite = bool(np.isfinite(arr).all()) if np.issubdtype(arr.dtype, np.floating) else True
        item: dict[str, Any] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "finite": finite,
        }
        if arr.size and np.issubdtype(arr.dtype, np.number):
            item["min"] = float(np.nanmin(arr))
            item["max"] = float(np.nanmax(arr))
            item["mean"] = float(np.nanmean(arr))
        summary.append(item)
    return summary


def _stderr_runtime_error(stderr: str | None) -> str | None:
    text = stderr or ""
    patterns = [
        "failed to submit",
        "input dtype is undefine",
        "failed to set inputs",
    ]
    lowered = text.lower()
    for pattern in patterns:
        if pattern.lower() in lowered:
            return pattern
    return None


def _child_main(args: argparse.Namespace) -> int:
    from rknnlite.api import RKNNLite

    rknn_path = Path(args.rknn)
    case = _case_from_name(rknn_path, args.case)
    inputs = _inputs_for_case(case, rknn_path)
    pass_through = _pass_through(inputs, args.pass_through)

    result: dict[str, Any] = {
        "path": str(rknn_path),
        "name": rknn_path.name,
        "case": case,
        "pass_through": args.pass_through,
        "input_shapes": [list(x.shape) for x in inputs],
        "input_dtypes": [str(x.dtype) for x in inputs],
        "status": "FAIL",
    }
    rknn = RKNNLite(verbose=False)
    try:
        t0 = time.perf_counter()
        ret = rknn.load_rknn(str(rknn_path))
        result["load_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        result["load_ret"] = int(ret)
        print(json.dumps({**result, "phase": "load"}, ensure_ascii=False), flush=True)
        if ret != 0:
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 2

        t0 = time.perf_counter()
        ret = rknn.init_runtime()
        result["init_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        result["init_ret"] = int(ret)
        print(json.dumps({**result, "phase": "init"}, ensure_ascii=False), flush=True)
        if ret != 0:
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 3

        t0 = time.perf_counter()
        if pass_through is None:
            outputs = rknn.inference(inputs=inputs)
        else:
            outputs = rknn.inference(inputs=inputs, inputs_pass_through=pass_through)
        result["infer_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        if outputs is None:
            result["error"] = "rknn.inference returned None"
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 5
        result["outputs"] = _summarize_outputs(outputs or [])
        result["status"] = "OK"
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        result["error"] = repr(exc)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 4
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def _run_child(script: Path, rknn: Path, case: str, pass_through: str, timeout: float) -> dict[str, Any]:
    inferred_case = _case_from_name(rknn, case)
    cmd = [
        sys.executable,
        str(script),
        "--child",
        "--rknn",
        str(rknn),
        "--case",
        case,
        "--pass-through",
        pass_through,
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "path": str(rknn),
            "name": rknn.name,
            "case": case,
            "pass_through": pass_through,
            "status": "TIMEOUT",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "timeout_s": timeout,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }

    parsed = None
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if parsed is None:
        parsed = {
            "path": str(rknn),
            "name": rknn.name,
            "case": inferred_case,
            "pass_through": pass_through,
            "status": "CRASH" if proc.returncode < 0 or proc.returncode == 139 else "FAIL",
        }
    parsed["returncode"] = proc.returncode
    parsed["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    if proc.returncode in (139, -11):
        parsed["status"] = "CRASH"
        parsed["signal"] = "SIGSEGV"
    if proc.stderr:
        parsed["stderr_tail"] = proc.stderr[-4000:]
    runtime_error = _stderr_runtime_error(proc.stderr)
    if runtime_error and parsed.get("status") == "OK":
        parsed["status"] = "FAIL"
        parsed["error"] = f"RKNN runtime stderr contained {runtime_error!r}"
    return parsed


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*", help="RKNN paths or glob patterns")
    parser.add_argument("--rknn", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--case",
        default="auto",
        choices=[
            "auto",
            "layernorm",
            "island_float",
            "sampler_island_float",
            "sampler_audio_heads",
            "sampler_mlps",
            "prefill",
            "prefill_tokens",
            "prefill_hidden",
            "prefill_hidden_mask",
            "decode",
            "sampler",
            "codec",
        ],
    )
    parser.add_argument("--pass-through", default="auto", choices=["auto", "none", "all"])
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        if args.rknn is None:
            raise SystemExit("--child requires --rknn")
        return _child_main(args)

    paths = _expand_inputs(args.patterns)
    if not paths:
        raise SystemExit("No RKNN paths matched")

    script = Path(__file__).resolve()
    results = []
    for path in paths:
        if not path.exists():
            item = {
                "path": str(path),
                "name": path.name,
                "case": args.case,
                "pass_through": args.pass_through,
                "status": "MISSING",
            }
        else:
            item = _run_child(script, path, args.case, args.pass_through, args.timeout)
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "OK"),
        "crash": sum(1 for r in results if r.get("status") == "CRASH"),
        "timeout": sum(1 for r in results if r.get("status") == "TIMEOUT"),
        "missing": sum(1 for r in results if r.get("status") == "MISSING"),
    }
    report = {"summary": summary, "results": results}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    return 0 if summary["ok"] == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

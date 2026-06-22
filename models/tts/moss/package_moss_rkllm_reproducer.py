#!/usr/bin/env python3
"""Package MOSS RKLLM parity evidence for an upstream Rockchip report."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = (
    "rk3576-moss-rkllm-folded-runtime-probe.json",
    "rk3576-moss-rkllm-hidden-vs-onnx-s8.json",
    "rk3576-moss-rkllm-hidden-vs-onnx-s8-embedflash0.json",
    "rk3576-moss-rkllm-token-hidden-s8.json",
    "wsl2-moss-hf-variants-vs-rkllm-s8.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _copy_file(src: Path, dst_dir: Path) -> dict[str, Any]:
    item = {
        "source": str(src),
        "target": str(dst_dir / src.name),
        "exists": src.exists(),
        "size_bytes": src.stat().st_size if src.exists() else None,
    }
    if src.exists():
        shutil.copy2(src, dst_dir / src.name)
    return item


def _metric_line(label: str, metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return f"- {label}: missing"
    return (
        f"- {label}: rel_l2={metrics.get('rel_l2')}, cosine={metrics.get('cosine')}, "
        f"max_abs={metrics.get('max_abs')}, finite={metrics.get('finite')}"
    )


def _render_issue(manifest: dict[str, Any]) -> str:
    hidden = manifest["summary"].get("hidden_parity") or {}
    token = manifest["summary"].get("token_input") or {}
    embed_flash0 = manifest["summary"].get("embed_flash0") or {}
    variants = manifest["summary"].get("hf_variants") or {}
    runtime = manifest["summary"].get("runtime_probe") or {}

    hf_original = (variants.get("variants") or {}).get("hf_original") or {}
    hf_rmsnorm = (variants.get("variants") or {}).get("hf_rmsnorm") or {}

    return (
        "# RKLLM custom model hidden-state mismatch for MOSS/GPT2-style LayerNorm block\n\n"
        "## Summary\n\n"
        "A MOSS-TTS global model scaffold can be exported and loaded as `.rkllm` on RK3576. "
        "The runtime returns finite hidden states for both prefill and decode, but hidden-state "
        "parity against the ONNX/HuggingFace reference is very poor. This blocks using RKLLM "
        "as the NPU runtime for low-latency streaming TTS.\n\n"
        "The model is GPT2-style: LayerNorm with bias, fused qkv/Conv1D-style projections, "
        "GELU MLP, and external 17-column MOSS row embeddings. It is not LLaMA/Qwen-style "
        "RMSNorm + gated FFN.\n\n"
        "## Environment\n\n"
        f"- RK device model evidence: {hidden.get('rkllm_model') or runtime.get('model_path')}\n"
        "- Target SoC: RK3576\n"
        "- RKLLM runtime ABI used by wrapper: v1.2.3 ctypes layout\n"
        "- Reference checker: ONNX Runtime CPU and HuggingFace scaffold\n\n"
        "## Observed Behavior\n\n"
        f"{_metric_line('embedding input prefill hidden vs ONNX', hidden.get('prefill_metrics'))}\n"
        f"{_metric_line('embedding input decode hidden vs ONNX', hidden.get('decode_metrics'))}\n"
        f"{_metric_line('embedding input with embed_flash=0 prefill hidden vs ONNX', embed_flash0.get('prefill_metrics'))}\n"
        f"{_metric_line('token input prefill hidden vs ONNX', token.get('prefill_metrics'))}\n"
        f"{_metric_line('HF original scaffold vs ONNX', hf_original.get('vs_onnx'))}\n"
        f"{_metric_line('HF original scaffold vs RKLLM hidden', hf_original.get('vs_rkllm'))}\n"
        f"{_metric_line('HF RMSNorm variant vs RKLLM hidden', hf_rmsnorm.get('vs_rkllm'))}\n\n"
        "Runtime smoke succeeds:\n\n"
        f"- `.rkllm` loaded: {runtime.get('loaded')}\n"
        f"- prefill callback hidden finite: {(runtime.get('prefill_hidden') or {}).get('finite')}\n"
        f"- decode callback hidden finite: {(runtime.get('decode_hidden') or {}).get('finite')}\n\n"
        "## Expected Behavior\n\n"
        "For a supported custom model, `RKLLM_INFER_GET_LAST_HIDDEN_LAYER` should match the "
        "same HuggingFace/ONNX hidden state within FP16 tolerance. The local gate is "
        "`rel_l2 <= 0.02` and `cosine >= 0.999` for prefill hidden.\n\n"
        "## Fixes Already Tried\n\n"
        "- Disable chat template with `rkllm_set_chat_template(handle, b\"\", b\"\", b\"\")`.\n"
        "- Force `RKLLM_EMBED_FLASH=0`.\n"
        "- Use `RKLLM_INPUT_TOKEN` instead of external embeddings.\n"
        "- Compare against a HuggingFace RMSNorm variant to test whether LayerNorm was being "
        "lowered as RMSNorm. This does not explain the mismatch.\n\n"
        "## Question\n\n"
        "Does RKLLM custom-model export/runtime support GPT2-style LayerNorm-with-bias and "
        "bias-capable attention/MLP projections, or is custom export currently limited to "
        "LLaMA/Qwen-like RMSNorm/gated-FFN templates? If supported, what custom_config/modeling "
        "changes are required to make hidden parity pass?\n\n"
        "## Attachments\n\n"
        "The package manifest lists copied JSON evidence and optional NPZ hidden dumps. "
        "The NPZ contains fixed input IDs, ONNX hidden, RKLLM hidden, decode input row, and "
        "decode hidden for an 8-token repro.\n"
    )


def package_reproducer(evidence_dir: Path, out_dir: Path, include_npz: bool = True) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = [_copy_file(evidence_dir / name, out_dir) for name in DEFAULT_EVIDENCE]
    npz_files: list[dict[str, Any]] = []
    if include_npz:
        npz_files.append(_copy_file(evidence_dir / "rk3576-moss-rkllm-hidden-vs-onnx-s8-dump.npz", out_dir))

    runtime = _optional_json(evidence_dir / "rk3576-moss-rkllm-folded-runtime-probe.json")
    hidden = _optional_json(evidence_dir / "rk3576-moss-rkllm-hidden-vs-onnx-s8.json")
    embed_flash0 = _optional_json(evidence_dir / "rk3576-moss-rkllm-hidden-vs-onnx-s8-embedflash0.json")
    token = _optional_json(evidence_dir / "rk3576-moss-rkllm-token-hidden-s8.json")
    variants = _optional_json(evidence_dir / "wsl2-moss-hf-variants-vs-rkllm-s8.json")

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": str(out_dir),
        "purpose": "upstream RKLLM custom-model hidden parity reproducer for MOSS-TTS",
        "summary": {
            "runtime_probe": runtime,
            "hidden_parity": hidden,
            "embed_flash0": embed_flash0,
            "token_input": token,
            "hf_variants": variants,
        },
        "files": copied,
        "npz_files": npz_files,
        "commands": {
            "rk3576_hidden_parity": (
                "RKLLM_DISABLE_CHAT_TEMPLATE=1 PYTHONPATH=/home/cat/rkvoice-stream "
                "/home/cat/rknn-venv/bin/python "
                "models/tts/moss/compare_moss_rkllm_hidden_runtime.py "
                "--model-dir /home/cat/moss-onnx-baseline "
                "--rkllm-model /home/cat/moss-rkllm/moss_global_embed_only_folded_fp16_rk3576.rkllm "
                "--assets /home/cat/moss-rkllm/moss_rkllm_runtime_assets.npz "
                "--npz-out docs/evidence/moss/rk3576-moss-rkllm-hidden-vs-onnx-s8-dump.npz"
            ),
            "rk3576_token_input_parity": (
                "RKLLM_DISABLE_CHAT_TEMPLATE=1 RKLLM_EMBED_FLASH=0 "
                "PYTHONPATH=/home/cat/rkvoice-stream /home/cat/rknn-venv/bin/python "
                "models/tts/moss/compare_moss_rkllm_token_runtime.py "
                "--model-dir /home/cat/moss-onnx-baseline "
                "--rkllm-model /home/cat/moss-rkllm/moss_global_embed_only_folded_fp16_rk3576.rkllm"
            ),
            "wsl2_hf_variant_probe": (
                "/home/harve/projects/MOSS-TTS-Nano/.venv/bin/python "
                "models/tts/moss/compare_moss_hf_variants_to_rkllm_hidden.py "
                "--model-dir /home/harve/models/moss-rkllm-embed-only-folded-scaffold "
                "--dump docs/evidence/moss/rk3576-moss-rkllm-hidden-vs-onnx-s8-dump.npz"
            ),
        },
    }
    manifest["upstream_issue_markdown"] = "UPSTREAM_ISSUE_DRAFT.md"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "UPSTREAM_ISSUE_DRAFT.md").write_text(_render_issue(manifest), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path("docs/evidence/moss"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--no-npz", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = package_reproducer(args.evidence_dir, args.out_dir, include_npz=not args.no_npz)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

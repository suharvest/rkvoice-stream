#!/usr/bin/env python3
"""Write a MOSS-TTS-Nano hybrid ORT+RKNN prefill manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _parse_layers(raw: str | None) -> set[int]:
    text = str(raw or "all").strip().lower()
    if text in {"", "all"}:
        return set(range(12))
    layers: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(part))
    invalid = sorted(layer for layer in layers if layer < 0 or layer > 11)
    if invalid:
        raise ValueError(f"invalid MOSS hybrid layer ids: {invalid}")
    return layers


def default_required(
    seq_len: int,
    target: str,
    split: str = "ln2_mlp",
    layers: set[int] | None = None,
    split_root: str = "artifact_dir",
) -> list[tuple[str, str]]:
    selected_layers = set(range(12)) if layers is None else set(layers)
    artifacts: list[tuple[str, str]] = [
        ("artifact_dir", f"moss_embedding_prefix.s{seq_len}.onnx"),
        ("artifact_dir", f"moss_final_norm.s{seq_len}.onnx"),
    ]
    for layer in range(12):
        artifacts.append(("artifact_dir", f"moss_block{layer}_attn_residual.s{seq_len}.onnx"))
        if layer not in selected_layers:
            artifacts.append(("artifact_dir", f"moss_block{layer}_ln2_mlp.s{seq_len}.onnx"))
        elif split == "ln2_mlp":
            artifacts.append((split_root, f"moss_block{layer}_ln2_mlp.s{seq_len}.fp16.{target}.rknn"))
        elif split == "mlp_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_mlp.s{seq_len}.fp16.{target}.rknn"),
                ]
            )
        elif split == "fc_in_act_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_in_act.s{seq_len}.fp16.{target}.rknn"),
                    (split_root, f"moss_block{layer}_fc_out.s{seq_len}.onnx"),
                ]
            )
        elif split == "fc_out_only":
            artifacts.extend(
                [
                    (split_root, f"moss_block{layer}_ln2.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_in_act.s{seq_len}.onnx"),
                    (split_root, f"moss_block{layer}_fc_out.s{seq_len}.fp16.{target}.rknn"),
                ]
            )
        else:
            raise ValueError(f"unsupported MOSS hybrid split: {split}")
    return artifacts


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_entry(root: Path, rel: str, root_key: str = "artifact_dir") -> dict:
    path = root / rel
    entry = {"path": rel, "required": True}
    if root_key != "artifact_dir":
        entry["root"] = root_key
    if path.exists() and path.is_file():
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--out", default="moss-hybrid-manifest.json")
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument(
        "--split",
        default="ln2_mlp",
        choices=["ln2_mlp", "mlp_only", "fc_in_act_only", "fc_out_only"],
    )
    parser.add_argument("--layers", default="all", help="RKNN layer ids, e.g. all or 0,1,4,5,6")
    parser.add_argument("--rknn-dir", type=Path, help="directory for split RKNN artifacts when separate")
    args = parser.parse_args()
    layers = _parse_layers(args.layers)
    roots = {
        "artifact_dir": args.artifact_dir,
        "rknn_dir": args.rknn_dir or args.artifact_dir,
    }
    split_root = "rknn_dir" if args.rknn_dir else "artifact_dir"
    required = default_required(args.seq_len, args.target, args.split, layers, split_root)

    manifest = {
        "model_id": "moss-tts-nano-hybrid-rknn",
        "target_platform": args.target,
        "seq_len": args.seq_len,
        "split": f"prefill_{args.split}",
        "rknn_layers": sorted(layers),
        "composition": "embedding_prefix_ort -> 12*(attn_residual_ort + split MLP path) -> final_norm_ort",
        "artifacts": [artifact_entry(roots[root_key], rel, root_key) for root_key, rel in required],
        "quality_status": {
            "production_default": False,
            "reason": "RKNN FP16 MLP drift still fails ASR roundtrip quality gates after sampler",
        },
    }
    out_path = args.artifact_dir / args.out
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

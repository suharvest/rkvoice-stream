#!/usr/bin/env python3
"""Validate a MOSS-TTS-Nano hybrid ORT+RKNN prefill artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_hybrid_artifacts


def _parse_layers(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
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
    return layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--manifest", default="moss-hybrid-manifest.json")
    parser.add_argument("--seq-len", type=int, default=320)
    parser.add_argument("--target", default="rk3576", choices=["rk3576", "rk3588"])
    parser.add_argument("--split", choices=["ln2_mlp", "mlp_only", "fc_in_act_only", "fc_out_only"])
    parser.add_argument("--layers", help="expected RKNN layer ids, e.g. all or 0,1,4,5,6")
    parser.add_argument("--rknn-dir", type=Path, help="directory for split RKNN artifacts when separate")
    parser.add_argument("--json", action="store_true", help="print parsed manifest as JSON")
    args = parser.parse_args()

    try:
        manifest = validate_moss_hybrid_artifacts(
            args.artifact_dir,
            args.manifest,
            args.seq_len,
            args.target,
            split=args.split,
            layers=_parse_layers(args.layers),
            rknn_dir=args.rknn_dir,
        )
    except MossORTArtifactError as exc:
        print(f"MOSS_HYBRID_ARTIFACTS_FAIL {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        artifacts = [a for a in manifest.get("artifacts", []) if a.get("required", True)]
        print(
            "MOSS_HYBRID_ARTIFACTS_OK "
            f"target={manifest.get('target_platform')} "
            f"seq_len={manifest.get('seq_len')} "
            f"required={len(artifacts)} "
            f"production_default={manifest.get('quality_status', {}).get('production_default')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate a MOSS-TTS-Nano ONNX Runtime production artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_ort_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", default="moss-ort-manifest.json")
    parser.add_argument("--json", action="store_true", help="print parsed manifest as JSON")
    args = parser.parse_args()

    try:
        manifest = validate_moss_ort_artifacts(args.model_dir, args.manifest)
    except MossORTArtifactError as exc:
        print(f"MOSS_ORT_ARTIFACTS_FAIL {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        artifacts = [a for a in manifest.get("artifacts", []) if a.get("required", True)]
        print(
            "MOSS_ORT_ARTIFACTS_OK "
            f"target={manifest.get('target_platform')} "
            f"required={len(artifacts)} "
            f"sample_rate={manifest.get('sample_rate')} "
            f"streaming_required={manifest.get('streaming_required', True)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate a MOSS-TTS-Nano RKNN production artifact bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream.backends.tts.moss_rknn import MossArtifactError, validate_moss_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--manifest", default="moss-rknn-manifest.json")
    parser.add_argument(
        "--require-production-default",
        action="store_true",
        help="require quality_status.production_default=true with full production evidence",
    )
    parser.add_argument("--json", action="store_true", help="print parsed manifest as JSON")
    args = parser.parse_args()

    try:
        manifest = validate_moss_artifacts(
            args.model_dir,
            args.manifest,
            require_production_default=args.require_production_default,
        )
    except MossArtifactError as exc:
        print(f"MOSS_RKNN_ARTIFACTS_FAIL {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        artifacts = [a for a in manifest.get("artifacts", []) if a.get("required", True)]
        print(
            "MOSS_RKNN_ARTIFACTS_OK "
            f"target={manifest.get('target_platform')} "
            f"required={len(artifacts)} "
            f"sample_rate={manifest.get('sample_rate')} "
            f"production_default={manifest.get('quality_status', {}).get('production_default', None)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

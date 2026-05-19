#!/usr/bin/env python3
"""Upload Paraformer hybrid RKNN artifacts to the existing RK HF repo.

Requires:
  pip install huggingface_hub
  huggingface-cli login  # or export HF_TOKEN=...

By default this integrates into the existing RK artifact repository under the
``paraformer-hybrid/`` prefix instead of creating a new model repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATTERNS = (
    "tokens.txt",
    "decoder-rknn.onnx",
    "encoder_prefix_to_block30.onnx",
    "encoder_suffix_from_block30.onnx",
    "manifest-*.json",
    "rknn/**/encoder_prefix_to_block30.*.fp16.rknn",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="harvestsu/seeed-local-voice-rk-artifacts")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--path-in-repo", default="paraformer-hybrid")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--commit-message", default="Add Paraformer RKNN hybrid artifacts")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        folder_path=str(args.artifact_dir),
        path_in_repo=args.path_in_repo.strip("/"),
        allow_patterns=list(DEFAULT_PATTERNS),
        commit_message=args.commit_message,
    )
    print(f"Uploaded {args.artifact_dir} to hf://{args.repo_id}/{args.path_in_repo.strip('/')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

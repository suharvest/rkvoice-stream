#!/usr/bin/env python3
"""Create or verify the portable Kokoro ConvOnly bundle receipt.

The command only records files already staged by the model release process; it
does not download, compile, or invent model artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path

SCHEMA = "kokoro.convonly.bundle.v1"
FULL_SOURCE_SHA256 = "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb"
GENERATOR_SOURCE_SHA256 = "301fc9d3fa7db258028ce895230b6f1782662bd0bdc8bd7dcdf5a9808bcf193c"
MANIFEST = "convonly.manifest.json"
CPU_GENERATOR = "generator-dynamic-hardened.onnx"
CPU_GENERATOR_SIZE = 78_971_566


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST and not p.is_symlink())


def validate_entries(root: Path) -> None:
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"bundle contains symlink: {path.relative_to(root)}")
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ValueError(f"bundle contains special file: {path.relative_to(root)}")


def build_manifest(root: Path, platform: str) -> dict:
    validate_entries(root)
    required = (root / "frontend", root / "prefix", root / "tail")
    missing = [str(p.relative_to(root)) for p in required if not p.is_dir()]
    if missing:
        raise ValueError("bundle is missing required directories: " + ", ".join(missing))
    required_files = (root / "frontend/frontend.manifest.json", root / "prefix/build-manifest.json",
                      root / "tail/library.so", root / f"tail/merge-official-v9.{platform}.fp16.rknn")
    missing = [str(p.relative_to(root)) for p in required_files if not p.is_file() or p.is_symlink()]
    if missing:
        raise ValueError("bundle is missing required artifacts: " + ", ".join(missing))
    rows = [{"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)} for p in files(root)]
    cpu_generator = root / CPU_GENERATOR
    manifest = {
        "schema": SCHEMA, "platform": platform, "precision": "FP16",
        "full_source_sha256": FULL_SOURCE_SHA256,
        "generator_source_sha256": GENERATOR_SOURCE_SHA256,
        "frontend": "frontend", "prefix": "prefix",
        "tail": {"model_dir": "tail/models", "slopes_dir": "tail/slopes",
                 "merge": f"tail/merge-official-v9.{platform}.fp16.rknn",
                 "library": "tail/library.so",
                 "implementation": "native" if platform == "rk3576" else "python"},
        "files": rows,
    }
    if cpu_generator.is_file() and not cpu_generator.is_symlink():
        if cpu_generator.stat().st_size != CPU_GENERATOR_SIZE:
            raise ValueError(f"{CPU_GENERATOR} size mismatch")
        if sha256(cpu_generator) != GENERATOR_SOURCE_SHA256:
            raise ValueError(f"{CPU_GENERATOR} SHA256 mismatch")
        manifest["cpu_generator"] = cpu_generator.name
    return manifest


def verify(root: Path, expected: dict) -> None:
    path = root / MANIFEST
    if not path.is_file():
        raise ValueError(f"manifest missing: {path}")
    actual = json.loads(path.read_text())
    if actual != expected:
        raise ValueError("manifest does not match current bundle inventory")
    listed = {row["path"] for row in actual["files"]} | {MANIFEST}
    present = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not p.is_symlink()}
    if listed != present:
        raise ValueError("manifest file closure mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify a Kokoro ConvOnly bundle manifest")
    parser.add_argument("root", type=Path, help="staged bundle root")
    parser.add_argument("--platform", choices=("rk3576", "rk3588"), required=True)
    parser.add_argument("--check", action="store_true", help="verify the existing manifest instead of writing it")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"bundle root is not a directory: {root}")
    manifest = build_manifest(root, args.platform)
    if args.check:
        verify(root, manifest)
    else:
        (root / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        verify(root, manifest)
    print(json.dumps({"manifest": str(root / MANIFEST), "sha256": sha256(root / MANIFEST), "files": len(manifest["files"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

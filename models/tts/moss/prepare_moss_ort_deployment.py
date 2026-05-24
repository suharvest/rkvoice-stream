#!/usr/bin/env python3
"""Prepare the canonical RK3576 MOSS ORT deployment path.

Default mode is dry-run. The recommended mode creates a symlink from the
canonical production path to an already validated model bundle, avoiding a large
copy on RK3576 root storage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_ort_artifacts


CONFIRMATION = "INSTALL_MOSS_ORT_DEPLOYMENT"


def _command_result(cmd: list[str], *, executed: bool, returncode: int | None = None, stderr: str = "") -> dict[str, Any]:
    return {
        "cmd": cmd,
        "executed": executed,
        "returncode": returncode,
        "stdout": "",
        "stderr": stderr,
    }


def _validate_source(source: Path, manifest: str) -> tuple[list[str], dict[str, Any] | None]:
    try:
        parsed = validate_moss_ort_artifacts(source, manifest)
        return [], {
            "model_dir": str(source),
            "manifest": manifest,
            "target_platform": parsed.get("target_platform"),
            "sample_rate": parsed.get("sample_rate"),
            "channels": parsed.get("channels"),
            "required_artifacts": len([item for item in parsed.get("artifacts", []) if item.get("required", True)]),
        }
    except MossORTArtifactError as exc:
        return [f"source artifact validation failed: {exc}"], None


def _destination_errors(source: Path, destination: Path) -> list[str]:
    errors: list[str] = []
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return []
        errors.append(f"destination already exists and is not the expected symlink: {destination}")
    if not str(destination).startswith("/opt/tts/models/"):
        errors.append(f"destination must be under /opt/tts/models, got {destination}")
    return errors


def prepare_deployment(
    *,
    source: Path = Path("/home/cat/moss-onnx-baseline"),
    destination: Path = Path("/opt/tts/models/moss-tts-nano-onnx"),
    manifest: str = "moss-ort-manifest.json",
    mode: str = "symlink",
    execute: bool = False,
    confirm: str = "",
) -> dict[str, Any]:
    source = source.resolve()
    destination = Path(destination)
    errors, source_summary = _validate_source(source, manifest)
    if mode != "symlink":
        errors.append(f"unsupported deployment mode: {mode}")
    errors.extend(_destination_errors(source, destination))
    if execute and confirm != CONFIRMATION:
        errors.append(f"--confirm must equal {CONFIRMATION!r} when --execute is used")

    commands = [
        ["mkdir", "-p", str(destination.parent)],
        ["ln", "-s", str(source), str(destination)],
    ]
    results: list[dict[str, Any]] = []

    if execute and not errors:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            results.append(_command_result(commands[0], executed=True, returncode=0))
            os.symlink(source, destination)
            results.append(_command_result(commands[1], executed=True, returncode=0))
        except OSError as exc:
            errors.append(f"deployment command failed: {exc}")
            if len(results) < len(commands):
                results.append(_command_result(commands[len(results)], executed=True, returncode=1, stderr=str(exc)))
    else:
        results = [_command_result(command, executed=False) for command in commands]

    deployed = destination.is_symlink() and destination.resolve() == source
    return {
        "passed": not errors,
        "errors": errors,
        "execute": execute,
        "requires_confirmation": True,
        "confirmation": CONFIRMATION,
        "mode": mode,
        "source": str(source),
        "destination": str(destination),
        "deployed": deployed,
        "source_manifest": source_summary,
        "commands": commands,
        "results": results,
        "disk_note": "symlink mode avoids copying the large MOSS ONNX bundle onto root storage",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/home/cat/moss-onnx-baseline"))
    parser.add_argument("--destination", type=Path, default=Path("/opt/tts/models/moss-tts-nano-onnx"))
    parser.add_argument("--manifest", default="moss-ort-manifest.json")
    parser.add_argument("--mode", default="symlink")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = prepare_deployment(
        source=args.source,
        destination=args.destination,
        manifest=args.manifest,
        mode=args.mode,
        execute=args.execute,
        confirm=args.confirm,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Start the RK3576 MOSS production service behind the release audit gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.tts.moss.audit_moss_release import (  # noqa: E402
    DEFAULT_EVIDENCE,
    DEFAULT_ENTRYPOINT_EVIDENCE,
    DEFAULT_ORT_CONFIG,
    DEFAULT_RKNN_CONFIG,
    DEFAULT_RKNN_WORKSPACE_MOUNT_POINT,
    DEFAULT_RKNN_WORKSPACE,
    DEFAULT_ROUNDTRIP_EVIDENCE,
    audit,
)
from models.tts.moss.verify_rknn_artifact_workspace import DEFAULT_MIN_FREE_MB as DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB


APP_TARGET = "rkvoice_stream.app.server:app"


def build_command(*, python: str, host: str, port: int, app: str = APP_TARGET) -> list[str]:
    return [
        python,
        "-m",
        "uvicorn",
        app,
        "--host",
        host,
        "--port",
        str(port),
    ]


def production_env(config: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    env["CONFIG"] = str(config)
    return env


def preflight(
    *,
    config: Path = DEFAULT_ORT_CONFIG,
    rknn_config: Path = DEFAULT_RKNN_CONFIG,
    evidence: Path = DEFAULT_EVIDENCE,
    roundtrip_evidence: Path = DEFAULT_ROUNDTRIP_EVIDENCE,
    entrypoint_evidence: Path = DEFAULT_ENTRYPOINT_EVIDENCE,
    min_root_free_mb: int | None = None,
    disk_path: Path = Path("/"),
    validate_artifacts: bool = False,
    require_rknn_workspace: bool = False,
    rknn_workspace: Path = DEFAULT_RKNN_WORKSPACE,
    rknn_workspace_min_free_mb: int = DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB,
    require_rknn_workspace_deployment: bool = False,
    rknn_workspace_mount_point: Path = DEFAULT_RKNN_WORKSPACE_MOUNT_POINT,
    require_performance_mode: bool = False,
    sysfs_root: Path = Path("/sys"),
) -> dict[str, Any]:
    report = audit(
        ort_config=config,
        rknn_config=rknn_config,
        evidence=evidence,
        roundtrip_evidence=roundtrip_evidence,
        entrypoint_evidence=entrypoint_evidence,
        min_root_free_mb=min_root_free_mb,
        disk_path=disk_path,
        validate_ort_artifacts=validate_artifacts,
        require_rknn_workspace=require_rknn_workspace,
        rknn_workspace=rknn_workspace,
        rknn_workspace_min_free_mb=rknn_workspace_min_free_mb,
        require_rknn_workspace_deployment=require_rknn_workspace_deployment,
        rknn_workspace_mount_point=rknn_workspace_mount_point,
        require_performance_mode=require_performance_mode,
        sysfs_root=sysfs_root,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_ORT_CONFIG)
    parser.add_argument("--rknn-config", type=Path, default=DEFAULT_RKNN_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--roundtrip-evidence", type=Path, default=DEFAULT_ROUNDTRIP_EVIDENCE)
    parser.add_argument("--entrypoint-evidence", type=Path, default=DEFAULT_ENTRYPOINT_EVIDENCE)
    parser.add_argument("--min-root-free-mb", type=int)
    parser.add_argument("--disk-path", type=Path, default=Path("/"))
    parser.add_argument(
        "--require-rknn-workspace",
        action="store_true",
        help="require a prepared non-root RKNN artifact workspace before reporting success",
    )
    parser.add_argument("--rknn-workspace", type=Path, default=DEFAULT_RKNN_WORKSPACE)
    parser.add_argument("--rknn-workspace-min-free-mb", type=int, default=DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB)
    parser.add_argument(
        "--require-rknn-workspace-deployment",
        action="store_true",
        help="require persistent fstab + mounted non-root RKNN workspace before reporting success",
    )
    parser.add_argument("--rknn-workspace-mount-point", type=Path, default=DEFAULT_RKNN_WORKSPACE_MOUNT_POINT)
    parser.add_argument(
        "--require-performance-mode",
        action="store_true",
        help="require RK3576 CPU/NPU/DDR fixed max-frequency performance mode before starting",
    )
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8621)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="audit and print the uvicorn command without starting")
    parser.add_argument(
        "--validate-artifacts",
        action="store_true",
        help="validate configured model_dir, manifest, and artifact hashes before reporting success",
    )
    args = parser.parse_args()

    report = preflight(
        config=args.config,
        rknn_config=args.rknn_config,
        evidence=args.evidence,
        roundtrip_evidence=args.roundtrip_evidence,
        entrypoint_evidence=args.entrypoint_evidence,
        min_root_free_mb=args.min_root_free_mb,
        disk_path=args.disk_path,
        validate_artifacts=args.validate_artifacts or not args.dry_run,
        require_rknn_workspace=args.require_rknn_workspace,
        rknn_workspace=args.rknn_workspace,
        rknn_workspace_min_free_mb=args.rknn_workspace_min_free_mb,
        require_rknn_workspace_deployment=args.require_rknn_workspace_deployment,
        rknn_workspace_mount_point=args.rknn_workspace_mount_point,
        require_performance_mode=args.require_performance_mode,
        sysfs_root=args.sysfs_root,
    )
    command = build_command(python=args.python, host=args.host, port=args.port)
    report["service"] = {
        "config": str(args.config),
        "env": {"CONFIG": str(args.config)},
        "command": command,
        "dry_run": bool(args.dry_run),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if not report["passed"]:
        return 2
    if args.dry_run:
        return 0
    os.execvpe(command[0], command, production_env(args.config))
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

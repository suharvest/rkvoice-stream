#!/usr/bin/env python3
"""Audit checked-in MOSS RK3576 release readiness.

This is a fast, offline release gate. It verifies that the checked-in
production fallback config, production evidence, and RKNN production-candidate
config agree on the current safety contract. Device artifact hashes and live
service behavior are still validated by the dedicated verifiers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rkvoice_stream import load_config
from rkvoice_stream.backends.tts.moss_ort import MossORTArtifactError, validate_moss_ort_artifacts
from models.tts.moss.verify_moss_ort_config import EXPECTED as ORT_EXPECTED
from models.tts.moss.verify_moss_ort_config import _check_config as check_ort_config
from models.tts.moss.verify_rk3576_performance_mode import verify_performance_mode
from models.tts.moss.verify_rknn_artifact_workspace import (
    DEFAULT_MIN_FREE_MB as DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB,
)
from models.tts.moss.verify_rknn_artifact_workspace import verify_workspace as verify_rknn_workspace
from models.tts.moss.verify_rknn_workspace_deployment import (
    DEFAULT_MOUNT_POINT as DEFAULT_RKNN_WORKSPACE_MOUNT_POINT,
)
from models.tts.moss.verify_rknn_workspace_deployment import verify_deployment as verify_rknn_workspace_deployment


DEFAULT_ORT_CONFIG = Path("configs/rk3576-moss-ort-stream.yaml")
DEFAULT_RKNN_CONFIG = Path("configs/rk3576-moss-rknn-stream.yaml")
DEFAULT_EVIDENCE = Path("docs/evidence/moss/rk3576-moss-ort-production-current-rerun.json")
DEFAULT_ROUNDTRIP_EVIDENCE = Path("docs/evidence/moss/rk3576-moss-ort-production-current-rerun-roundtrip.json")
DEFAULT_ENTRYPOINT_EVIDENCE = Path("docs/evidence/moss/rk3576-moss-production-entrypoint-profile.json")
DEFAULT_DEPLOY_SOURCE = Path("/home/cat/moss-onnx-baseline")
DEFAULT_DEPLOY_DESTINATION = Path("/opt/tts/models/moss-tts-nano-onnx")
DEFAULT_RKNN_WORKSPACE = Path("/mnt/rknn-workspace/moss-rknn-workspace")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc


def _check_ort_evidence(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summary = report.get("summary", {})
    expected_checks = {
        "artifact_manifest": True,
        "service_streaming": True,
        "backend_stage": True,
        "roundtrip_quality": True,
    }
    if summary.get("passed") is not True:
        errors.append("ORT evidence summary.passed must be true")
    if summary.get("errors") not in ([], None):
        errors.append(f"ORT evidence summary.errors must be empty, got {summary.get('errors')!r}")
    if summary.get("checks") != expected_checks:
        errors.append(f"ORT evidence checks mismatch: {summary.get('checks')!r}")

    profile = report.get("profile", {})
    for key in ("backend", "manifest", "voice", "seed", "threads", "prefill_seq", "codec_batch_frames"):
        expected = ORT_EXPECTED.get(key)
        if profile.get(key) != expected:
            errors.append(f"ORT evidence profile.{key}={profile.get(key)!r}, expected {expected!r}")
    if profile.get("codec_streaming") != 1:
        errors.append(f"ORT evidence profile.codec_streaming={profile.get('codec_streaming')!r}, expected 1")

    runtime = (
        report.get("service_streaming", {})
        .get("health", {})
        .get("tts_info", {})
        .get("profile", {})
    )
    hybrid = (
        report.get("service_streaming", {})
        .get("health", {})
        .get("tts_info", {})
        .get("hybrid", {})
    )
    runtime_expected = {
        "codec_streaming": True,
        "codec_full_loaded": False,
        "codec_batch_frames": 3,
        "codec_async": False,
        "cache_voice_prefix": False,
    }
    for key, expected in runtime_expected.items():
        if runtime.get(key) != expected:
            errors.append(f"ORT runtime profile.{key}={runtime.get(key)!r}, expected {expected!r}")
    if hybrid.get("enabled") is not False:
        errors.append(f"ORT runtime hybrid.enabled={hybrid.get('enabled')!r}, expected false")

    service = report.get("service_streaming", {})
    backend = report.get("backend_stage", {}).get("summary", {})
    roundtrip = report.get("roundtrip_quality", {})
    thresholds = [
        ("service.tts_stream.first_payload_ms", service.get("tts_stream", {}).get("first_payload_ms"), 1500, "<="),
        ("service.dialogue.first_payload_ms", service.get("dialogue", {}).get("first_payload_ms"), 1500, "<="),
        ("service.dialogue.max_payload_gap_ms", service.get("dialogue", {}).get("max_payload_gap_ms"), 1500, "<="),
        ("backend.ttfa_ms", backend.get("ttfa_ms"), 1500, "<="),
        ("backend.prefill_ms", backend.get("prefill_ms"), 1200, "<="),
        ("backend.max_codec_ms", backend.get("max_codec_ms"), 170, "<="),
        ("roundtrip.avg_cer", roundtrip.get("avg_cer"), 0.5, "<="),
        ("roundtrip.max_cer", roundtrip.get("max_cer"), 1.0, "<="),
        ("roundtrip.min_rms", roundtrip.get("min_rms"), 0.02, ">="),
        ("roundtrip.max_ttfa_ms", roundtrip.get("max_ttfa_ms"), 1500, "<="),
        ("roundtrip.max_codec_ms", roundtrip.get("max_codec_ms"), 170, "<="),
    ]
    for name, value, limit, op in thresholds:
        if value is None:
            errors.append(f"{name} is missing")
            continue
        if op == "<=" and float(value) > float(limit):
            errors.append(f"{name}={value} exceeds {limit}")
        if op == ">=" and float(value) < float(limit):
            errors.append(f"{name}={value} below {limit}")
    chunks = service.get("dialogue", {}).get("binary_chunks")
    if chunks is None or int(chunks) < 7:
        errors.append(f"service.dialogue.binary_chunks={chunks!r}, expected >= 7")
    frames = backend.get("audio_frames")
    if frames is None or int(frames) < 20:
        errors.append(f"backend.audio_frames={frames!r}, expected >= 20")
    return errors


def _check_roundtrip_evidence(report: dict[str, Any], roundtrip: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if roundtrip.get("gates", {}).get("passed") is not True:
        errors.append("roundtrip evidence gates.passed must be true")
    source = report.get("roundtrip_quality", {})
    for key in ("avg_cer", "max_cer", "min_rms", "max_ttfa_ms", "max_codec_ms"):
        if roundtrip.get(key) != source.get(key):
            errors.append(f"roundtrip evidence {key}={roundtrip.get(key)!r}, expected {source.get(key)!r}")
    return errors


def _check_production_entrypoint_evidence(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    gates = report.get("gates", {})
    if gates.get("passed") is not True:
        errors.append("production entrypoint gates.passed must be true")
    if gates.get("errors") not in ([], None):
        errors.append(f"production entrypoint gates.errors must be empty, got {gates.get('errors')!r}")

    entrypoint = report.get("entrypoint", {})
    runner = str(entrypoint.get("runner", ""))
    if not runner.endswith("models/tts/moss/run_moss_production_server.py"):
        errors.append(f"production entrypoint runner={runner!r}, expected run_moss_production_server.py")
    if entrypoint.get("config") != str(DEFAULT_ORT_CONFIG):
        errors.append(f"production entrypoint config={entrypoint.get('config')!r}, expected {str(DEFAULT_ORT_CONFIG)!r}")

    tts_info = report.get("health", {}).get("tts_info", {})
    if tts_info.get("backend") != "moss_ort":
        errors.append(f"production entrypoint backend={tts_info.get('backend')!r}, expected 'moss_ort'")
    if tts_info.get("model_dir") != str(DEFAULT_DEPLOY_DESTINATION):
        errors.append(
            "production entrypoint health.tts_info.model_dir="
            f"{tts_info.get('model_dir')!r}, expected {str(DEFAULT_DEPLOY_DESTINATION)!r}"
        )
    manifest = tts_info.get("manifest", {})
    if manifest.get("validated") is not True:
        errors.append(f"production entrypoint manifest.validated={manifest.get('validated')!r}, expected true")
    if manifest.get("required_artifacts") != 11:
        errors.append(
            "production entrypoint manifest.required_artifacts="
            f"{manifest.get('required_artifacts')!r}, expected 11"
        )

    runtime = tts_info.get("profile", {})
    runtime_expected = {
        "codec_streaming": True,
        "codec_full_loaded": False,
        "codec_batch_frames": 3,
        "codec_async": False,
        "cache_voice_prefix": False,
    }
    for key, expected in runtime_expected.items():
        if runtime.get(key) != expected:
            errors.append(f"production entrypoint profile.{key}={runtime.get(key)!r}, expected {expected!r}")
    if tts_info.get("hybrid", {}).get("enabled") is not False:
        errors.append(
            "production entrypoint hybrid.enabled="
            f"{tts_info.get('hybrid', {}).get('enabled')!r}, expected false"
        )

    stats = report.get("health_after", {}).get("tts_info", {}).get("streaming_stats", {})
    stats_expected = {"requests": 2, "completed": 2, "errors": 0, "active": 0}
    for key, expected in stats_expected.items():
        if stats.get(key) != expected:
            errors.append(f"production entrypoint streaming_stats.{key}={stats.get(key)!r}, expected {expected!r}")
    if int(stats.get("chunks", 0) or 0) < 14:
        errors.append(f"production entrypoint streaming_stats.chunks={stats.get('chunks')!r}, expected >= 14")

    thresholds = [
        ("production entrypoint tts_stream.first_payload_ms", report.get("tts_stream", {}).get("first_payload_ms"), 1500),
        ("production entrypoint dialogue.first_payload_ms", report.get("dialogue", {}).get("first_payload_ms"), 1500),
        ("production entrypoint dialogue.max_payload_gap_ms", report.get("dialogue", {}).get("max_payload_gap_ms"), 1500),
    ]
    for name, value, limit in thresholds:
        if value is None:
            errors.append(f"{name} is missing")
        elif float(value) > float(limit):
            errors.append(f"{name}={value} exceeds {limit}")
    chunks = report.get("dialogue", {}).get("binary_chunks")
    if chunks is None or int(chunks) < 7:
        errors.append(f"production entrypoint dialogue.binary_chunks={chunks!r}, expected >= 7")
    return errors


def _check_rknn_candidate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    tts = config.get("tts") or {}
    asr = config.get("asr") or {}
    expected = {
        "backend": "moss_rknn",
        "manifest": "moss-rknn-manifest.json",
        "worker_bin": "/opt/rkvoice-workers/moss_rknn_worker",
        "require_production_default": 1,
    }
    for key, value in expected.items():
        if tts.get(key) != value:
            errors.append(f"RKNN config tts.{key}={tts.get(key)!r}, expected {value!r}")
    if int(tts.get("chunk_frames", 0)) <= 0:
        errors.append(f"RKNN config tts.chunk_frames={tts.get('chunk_frames')!r}, expected positive")
    if asr.get("backend") not in {None, "disabled"}:
        errors.append(f"RKNN config asr.backend must be null/disabled, got {asr.get('backend')!r}")
    return errors


def _check_root_disk(path: Path, min_free_mb: int) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_mb = usage.free // (1024 * 1024)
    total_mb = usage.total // (1024 * 1024)
    used_mb = usage.used // (1024 * 1024)
    errors: list[str] = []
    if free_mb < min_free_mb:
        errors.append(f"disk.free_mb={free_mb} below required {min_free_mb} for {path}")
    return {
        "passed": not errors,
        "errors": errors,
        "path": str(path),
        "free_mb": free_mb,
        "used_mb": used_mb,
        "total_mb": total_mb,
        "min_free_mb": min_free_mb,
    }


def _check_ort_artifacts(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    summary: dict[str, Any] | None = None
    try:
        config = load_config(str(config_path))
        errors.extend(check_ort_config(config, require_model_dir=True))
        if not errors:
            tts = config["tts"]
            manifest = validate_moss_ort_artifacts(tts["model_dir"], tts["manifest"])
            summary = {
                "model_dir": str(tts["model_dir"]),
                "manifest": str(tts["manifest"]),
                "target_platform": manifest.get("target_platform"),
                "sample_rate": manifest.get("sample_rate"),
                "channels": manifest.get("channels"),
                "required_artifacts": len([a for a in manifest.get("artifacts", []) if a.get("required", True)]),
            }
    except (RuntimeError, MossORTArtifactError, KeyError) as exc:
        errors.append(f"artifact validation failed: {exc}")
    check: dict[str, Any] = {
        "passed": not errors,
        "errors": errors,
        "path": str(config_path),
        "summary": summary,
    }
    if errors:
        check["remediation"] = {
            "dry_run": [
                sys.executable,
                "models/tts/moss/prepare_moss_ort_deployment.py",
                "--source",
                str(DEFAULT_DEPLOY_SOURCE),
                "--destination",
                str(DEFAULT_DEPLOY_DESTINATION),
                "--json-out",
                "/tmp/moss_ort_deployment_dry_run.json",
            ],
            "execute_requires_confirmation": [
                sys.executable,
                "models/tts/moss/prepare_moss_ort_deployment.py",
                "--source",
                str(DEFAULT_DEPLOY_SOURCE),
                "--destination",
                str(DEFAULT_DEPLOY_DESTINATION),
                "--execute",
                "--confirm",
                "INSTALL_MOSS_ORT_DEPLOYMENT",
            ],
        }
    return check


def audit(
    *,
    ort_config: Path = DEFAULT_ORT_CONFIG,
    rknn_config: Path = DEFAULT_RKNN_CONFIG,
    evidence: Path = DEFAULT_EVIDENCE,
    roundtrip_evidence: Path = DEFAULT_ROUNDTRIP_EVIDENCE,
    entrypoint_evidence: Path = DEFAULT_ENTRYPOINT_EVIDENCE,
    min_root_free_mb: int | None = None,
    disk_path: Path = Path("/"),
    validate_ort_artifacts: bool = False,
    require_rknn_workspace: bool = False,
    rknn_workspace: Path = DEFAULT_RKNN_WORKSPACE,
    rknn_workspace_min_free_mb: int = DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB,
    require_rknn_workspace_deployment: bool = False,
    rknn_workspace_mount_point: Path = DEFAULT_RKNN_WORKSPACE_MOUNT_POINT,
    require_performance_mode: bool = False,
    sysfs_root: Path = Path("/sys"),
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    ort_errors = check_ort_config(load_config(str(ort_config)))
    checks["ort_config"] = {"passed": not ort_errors, "errors": ort_errors, "path": str(ort_config)}

    evidence_report = _load_json(evidence)
    evidence_errors = _check_ort_evidence(evidence_report)
    checks["ort_evidence"] = {"passed": not evidence_errors, "errors": evidence_errors, "path": str(evidence)}

    roundtrip_report = _load_json(roundtrip_evidence)
    roundtrip_errors = _check_roundtrip_evidence(evidence_report, roundtrip_report)
    checks["roundtrip_evidence"] = {
        "passed": not roundtrip_errors,
        "errors": roundtrip_errors,
        "path": str(roundtrip_evidence),
    }

    entrypoint_report = _load_json(entrypoint_evidence)
    entrypoint_errors = _check_production_entrypoint_evidence(entrypoint_report)
    checks["production_entrypoint"] = {
        "passed": not entrypoint_errors,
        "errors": entrypoint_errors,
        "path": str(entrypoint_evidence),
    }

    rknn_errors = _check_rknn_candidate_config(load_config(str(rknn_config)))
    checks["rknn_candidate_config"] = {"passed": not rknn_errors, "errors": rknn_errors, "path": str(rknn_config)}

    if min_root_free_mb is not None:
        checks["root_disk"] = _check_root_disk(disk_path, min_root_free_mb)
    if validate_ort_artifacts:
        checks["ort_artifacts"] = _check_ort_artifacts(ort_config)
    if require_rknn_workspace:
        checks["rknn_workspace"] = verify_rknn_workspace(
            workspace=rknn_workspace,
            min_free_mb=rknn_workspace_min_free_mb,
        )
    if require_rknn_workspace_deployment:
        checks["rknn_workspace_deployment"] = verify_rknn_workspace_deployment(
            workspace=rknn_workspace,
            mount_point=rknn_workspace_mount_point,
            min_free_mb=rknn_workspace_min_free_mb,
        )
    if require_performance_mode:
        checks["performance_mode"] = verify_performance_mode(sysfs_root=sysfs_root)

    errors = [error for item in checks.values() for error in item["errors"]]
    return {
        "passed": not errors,
        "errors": errors,
        "checks": checks,
        "production_default": "moss_ort",
        "rknn_status": "production_candidate_requires_manifest_evidence",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ort-config", type=Path, default=DEFAULT_ORT_CONFIG)
    parser.add_argument("--rknn-config", type=Path, default=DEFAULT_RKNN_CONFIG)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--roundtrip-evidence", type=Path, default=DEFAULT_ROUNDTRIP_EVIDENCE)
    parser.add_argument("--entrypoint-evidence", type=Path, default=DEFAULT_ENTRYPOINT_EVIDENCE)
    parser.add_argument(
        "--min-root-free-mb",
        type=int,
        help="optional device preflight: require this much free space on --disk-path",
    )
    parser.add_argument("--disk-path", type=Path, default=Path("/"))
    parser.add_argument(
        "--validate-ort-artifacts",
        action="store_true",
        help="validate configured ORT model_dir, manifest, and artifact hashes",
    )
    parser.add_argument(
        "--require-rknn-workspace",
        action="store_true",
        help="require a prepared non-root RKNN artifact workspace before passing",
    )
    parser.add_argument("--rknn-workspace", type=Path, default=DEFAULT_RKNN_WORKSPACE)
    parser.add_argument("--rknn-workspace-min-free-mb", type=int, default=DEFAULT_RKNN_WORKSPACE_MIN_FREE_MB)
    parser.add_argument(
        "--require-rknn-workspace-deployment",
        action="store_true",
        help="require persistent fstab + mounted non-root RKNN workspace before passing",
    )
    parser.add_argument("--rknn-workspace-mount-point", type=Path, default=DEFAULT_RKNN_WORKSPACE_MOUNT_POINT)
    parser.add_argument(
        "--require-performance-mode",
        action="store_true",
        help="require RK3576 CPU/NPU/DDR fixed max-frequency performance mode before passing",
    )
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    try:
        report = audit(
            ort_config=args.ort_config,
            rknn_config=args.rknn_config,
            evidence=args.evidence,
            roundtrip_evidence=args.roundtrip_evidence,
            entrypoint_evidence=args.entrypoint_evidence,
            min_root_free_mb=args.min_root_free_mb,
            disk_path=args.disk_path,
            validate_ort_artifacts=args.validate_ort_artifacts,
            require_rknn_workspace=args.require_rknn_workspace,
            rknn_workspace=args.rknn_workspace,
            rknn_workspace_min_free_mb=args.rknn_workspace_min_free_mb,
            require_rknn_workspace_deployment=args.require_rknn_workspace_deployment,
            rknn_workspace_mount_point=args.rknn_workspace_mount_point,
            require_performance_mode=args.require_performance_mode,
            sysfs_root=args.sysfs_root,
        )
    except RuntimeError as exc:
        report = {
            "passed": False,
            "errors": [str(exc)],
            "checks": {},
            "production_default": "moss_ort",
            "rknn_status": "production_candidate_requires_manifest_evidence",
        }

    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

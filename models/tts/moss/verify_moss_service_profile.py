#!/usr/bin/env python3
"""Start rkvoice service and verify MOSS service-level streaming gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_moss_service_streaming import _read_json, main as verify_streaming_main


def _parse_env(values: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--set-env value must be KEY=VALUE, got {value!r}")
        key, val = value.split("=", 1)
        if not key:
            raise ValueError(f"--set-env key must not be empty: {value!r}")
        env[key] = val
    return env


def _wait_ready(base_url: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            health = _read_json(base_url.rstrip("/") + "/health", timeout=5.0)
            if health.get("tts") and health.get("streaming_tts"):
                return health
            last_error = RuntimeError(f"service not ready: {health}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for service readiness: {last_error}")


def _write_failure_report(path: str | None, *, base_url: str, ws_url: str, error: str, log_file: Path) -> None:
    if not path:
        return
    report = {
        "base_url": base_url,
        "ws_url": ws_url,
        "health": {"error": error},
        "tts_stream": {"passed": False},
        "dialogue": {"error": "not run"},
        "gates": {
            "passed": False,
            "errors": [error],
            "log_file": str(log_file),
        },
    }
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="", help="runtime CONFIG profile or YAML path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8623)
    parser.add_argument("--app", default="rkvoice_stream.app.server:app")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--expected-backend", default="moss_ort")
    parser.add_argument("--expected-sample-rate", type=int, default=48000)
    parser.add_argument("--min-payload-bytes", type=int, default=30720)
    parser.add_argument("--max-tts-wall-ms", type=float, default=2000.0)
    parser.add_argument("--max-dialogue-wall-ms", type=float, default=2000.0)
    parser.add_argument("--max-tts-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-payload-gap-ms", type=float, default=0.0)
    parser.add_argument("--require-manifest-validated", action="store_true")
    parser.add_argument("--expected-voice")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-manifest")
    parser.add_argument("--expected-codec-batch-frames", type=int)
    parser.add_argument("--require-production-runtime", action="store_true")
    parser.add_argument("--min-dialogue-binary-chunks", type=int, default=0)
    parser.add_argument("--set-env", action="append", default=[], help="extra service env KEY=VALUE")
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/moss_service_profile_uvicorn.log"))
    parser.add_argument("--json-out")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/dialogue"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.config:
        env["CONFIG"] = args.config
    env.update(_parse_env(args.set_env))

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("wb") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                args.app,
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--log-level",
                "info",
            ],
            cwd=args.repo_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            try:
                _wait_ready(base_url, args.startup_timeout)
            except Exception as exc:
                error = f"service readiness failed: {exc}"
                _write_failure_report(args.json_out, base_url=base_url, ws_url=ws_url, error=error, log_file=args.log_file)
                print(error, file=sys.stderr)
                return 1
            old_argv = sys.argv
            sys.argv = [
                "verify_moss_service_streaming.py",
                "--base-url",
                base_url,
                "--ws-url",
                ws_url,
                "--text",
                args.text,
                "--expected-backend",
                args.expected_backend,
                "--expected-sample-rate",
                str(args.expected_sample_rate),
                "--min-payload-bytes",
                str(args.min_payload_bytes),
                "--max-tts-wall-ms",
                str(args.max_tts_wall_ms),
                "--max-dialogue-wall-ms",
                str(args.max_dialogue_wall_ms),
                "--max-tts-first-payload-ms",
                str(args.max_tts_first_payload_ms),
                "--max-dialogue-first-payload-ms",
                str(args.max_dialogue_first_payload_ms),
                "--max-dialogue-payload-gap-ms",
                str(args.max_dialogue_payload_gap_ms),
                "--timeout",
                str(args.request_timeout),
            ]
            if args.require_manifest_validated:
                sys.argv.append("--require-manifest-validated")
            if args.expected_voice:
                sys.argv.extend(["--expected-voice", args.expected_voice])
            if args.expected_seed is not None:
                sys.argv.extend(["--expected-seed", str(args.expected_seed)])
            if args.expected_manifest:
                sys.argv.extend(["--expected-manifest", args.expected_manifest])
            if args.expected_codec_batch_frames is not None:
                sys.argv.extend(["--expected-codec-batch-frames", str(args.expected_codec_batch_frames)])
            if args.require_production_runtime:
                sys.argv.append("--require-production-runtime")
            if args.min_dialogue_binary_chunks:
                sys.argv.extend(["--min-dialogue-binary-chunks", str(args.min_dialogue_binary_chunks)])
            if args.json_out:
                sys.argv.extend(["--json-out", args.json_out])
            try:
                return verify_streaming_main()
            finally:
                sys.argv = old_argv
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10.0)
            if proc.returncode not in {0, -15, -2, 143, 130}:
                print(f"uvicorn exited with code {proc.returncode}; log={args.log_file}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

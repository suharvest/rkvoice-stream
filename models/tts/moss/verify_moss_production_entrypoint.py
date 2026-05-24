#!/usr/bin/env python3
"""Start the audited MOSS production entrypoint and verify streaming gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_moss_service_profile import _wait_ready, _write_failure_report  # noqa: E402
from verify_moss_service_streaming import main as verify_streaming_main  # noqa: E402


def build_runner_command(
    *,
    runner: Path,
    python: str,
    config: Path,
    host: str,
    port: int,
    preflight_json: Path,
) -> list[str]:
    return [
        python,
        str(runner),
        "--config",
        str(config),
        "--host",
        host,
        "--port",
        str(port),
        "--python",
        python,
        "--json-out",
        str(preflight_json),
    ]


def _augment_report(path: Path | None, *, entrypoint: dict) -> None:
    if path is None or not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    report["entrypoint"] = entrypoint
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/rk3576-moss-ort-stream.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8628)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runner", type=Path, default=Path("models/tts/moss/run_moss_production_server.py"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--text", default="你好")
    parser.add_argument("--expected-backend", default="moss_ort")
    parser.add_argument("--expected-sample-rate", type=int, default=48000)
    parser.add_argument("--min-payload-bytes", type=int, default=30720)
    parser.add_argument("--max-tts-wall-ms", type=float, default=6000.0)
    parser.add_argument("--max-dialogue-wall-ms", type=float, default=6500.0)
    parser.add_argument("--max-tts-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-first-payload-ms", type=float, default=1500.0)
    parser.add_argument("--max-dialogue-payload-gap-ms", type=float, default=1500.0)
    parser.add_argument("--require-manifest-validated", action="store_true")
    parser.add_argument("--expected-voice")
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument("--expected-manifest")
    parser.add_argument("--expected-codec-batch-frames", type=int)
    parser.add_argument("--require-production-runtime", action="store_true")
    parser.add_argument("--min-dialogue-binary-chunks", type=int, default=0)
    parser.add_argument("--log-file", type=Path, default=Path("/tmp/moss_production_entrypoint.log"))
    parser.add_argument("--preflight-json", type=Path, default=Path("/tmp/moss_production_entrypoint_preflight.json"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    ws_url = f"ws://{args.host}:{args.port}/dialogue"
    runner = args.runner if args.runner.is_absolute() else args.repo_root / args.runner
    command = build_runner_command(
        runner=runner,
        python=args.python,
        config=args.config,
        host=args.host,
        port=args.port,
        preflight_json=args.preflight_json,
    )
    entrypoint = {
        "runner": str(runner),
        "command": command,
        "preflight_json": str(args.preflight_json),
        "config": str(args.config),
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    proc: subprocess.Popen | None = None
    with args.log_file.open("wb") as log:
        proc = subprocess.Popen(command, cwd=args.repo_root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            try:
                _wait_ready(base_url, args.startup_timeout)
            except Exception as exc:
                error = f"production entrypoint readiness failed: {exc}"
                _write_failure_report(
                    args.json_out,
                    base_url=base_url,
                    ws_url=ws_url,
                    error=error,
                    log_file=args.log_file,
                )
                _augment_report(Path(args.json_out) if args.json_out else None, entrypoint=entrypoint)
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
                sys.argv.extend(["--json-out", str(args.json_out)])
            try:
                rc = verify_streaming_main()
                _augment_report(Path(args.json_out) if args.json_out else None, entrypoint=entrypoint)
                return rc
            finally:
                sys.argv = old_argv
        finally:
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10.0)
                if proc.returncode not in {0, -15, -2, 143, 130}:
                    print(f"production entrypoint exited with code {proc.returncode}; log={args.log_file}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

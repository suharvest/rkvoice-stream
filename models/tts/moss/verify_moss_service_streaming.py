#!/usr/bin/env python3
"""Verify MOSS service-level streaming contracts over HTTP and WebSocket."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
import urllib.error
import urllib.request
from typing import Any

import websockets


def _read_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_stream(url: str, text: str, timeout: float) -> tuple[bytes, float, float, float]:
    req = urllib.request.Request(
        url,
        data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        header = resp.read(4)
        header_ms = (time.perf_counter() - start) * 1000.0
        first_payload = resp.read(1)
        first_payload_ms = (time.perf_counter() - start) * 1000.0 if first_payload else None
        rest = resp.read()
    body = header + first_payload + rest
    return body, header_ms, float(first_payload_ms or 0.0), (time.perf_counter() - start) * 1000.0


async def _dialogue_ws(url: str, text: str, timeout: float) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"text": text}, ensure_ascii=False))
        chunks: list[bytes] = []
        payload_times_ms: list[float] = []
        sample_rate = None
        first_payload_ms = None
        last_text: dict[str, Any] | None = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
            if isinstance(msg, str):
                data = json.loads(msg)
                last_text = data
                if data.get("done") or data.get("type") in {"done", "error"} or data.get("error"):
                    return (
                        {
                            "sample_rate": sample_rate,
                            "binary_chunks": len(chunks),
                            "payload_bytes": sum(len(item) for item in chunks[1:]) if len(chunks) > 1 else 0,
                            "first_payload_ms": first_payload_ms,
                            "payload_chunk_times_ms": [round(item, 3) for item in payload_times_ms],
                            "max_payload_gap_ms": round(
                                max(
                                    (
                                        payload_times_ms[idx] - payload_times_ms[idx - 1]
                                        for idx in range(1, len(payload_times_ms))
                                    ),
                                    default=0.0,
                                ),
                                3,
                            ),
                            "last_text": last_text,
                        },
                        (time.perf_counter() - start) * 1000.0,
                    )
                continue
            chunks.append(msg)
            if sample_rate is None and len(msg) == 4:
                sample_rate = struct.unpack("<I", msg)[0]
            elif first_payload_ms is None:
                first_payload_ms = (time.perf_counter() - start) * 1000.0
                payload_times_ms.append(first_payload_ms)
            else:
                payload_times_ms.append((time.perf_counter() - start) * 1000.0)


def _check_tts_stream(body: bytes, expected_sample_rate: int, min_payload_bytes: int) -> dict[str, Any]:
    sample_rate = struct.unpack("<I", body[:4])[0] if len(body) >= 4 else None
    payload_bytes = max(0, len(body) - 4)
    return {
        "bytes": len(body),
        "sample_rate": sample_rate,
        "payload_bytes": payload_bytes,
        "int16_samples": payload_bytes // 2,
        "passed": sample_rate == expected_sample_rate and payload_bytes >= min_payload_bytes and payload_bytes % 2 == 0,
    }


def _collect_gate_errors(
    *,
    health: dict[str, Any],
    health_after: dict[str, Any] | None = None,
    expected_backend: str,
    expected_sample_rate: int,
    min_payload_bytes: int,
    tts_stream: dict[str, Any],
    tts_wall_ms: float | None,
    tts_first_payload_ms: float | None,
    max_tts_wall_ms: float,
    max_tts_first_payload_ms: float,
    dialogue: dict[str, Any],
    dialogue_wall_ms: float | None,
    max_dialogue_wall_ms: float,
    max_dialogue_first_payload_ms: float,
    max_dialogue_payload_gap_ms: float = 0.0,
    require_manifest_validated: bool = False,
    expected_voice: str | None = None,
    expected_seed: int | None = None,
    expected_manifest: str | None = None,
    expected_codec_batch_frames: int | None = None,
    require_production_runtime: bool = False,
    min_dialogue_binary_chunks: int = 0,
) -> list[str]:
    errors: list[str] = []
    if not health.get("tts"):
        errors.append("health.tts is not true")
    if health.get("tts_backend") != expected_backend:
        errors.append(f"health.tts_backend={health.get('tts_backend')!r}")
    if not health.get("streaming_tts"):
        errors.append("health.streaming_tts is not true")
    tts_info = health.get("tts_info") if isinstance(health.get("tts_info"), dict) else {}
    if require_manifest_validated:
        manifest = tts_info.get("manifest") if isinstance(tts_info.get("manifest"), dict) else {}
        if not manifest.get("validated"):
            errors.append("health.tts_info.manifest.validated is not true")
        if expected_manifest and manifest.get("name") != expected_manifest:
            errors.append(f"health.tts_info.manifest.name={manifest.get('name')!r}")
    profile = tts_info.get("profile") if isinstance(tts_info.get("profile"), dict) else {}
    if expected_voice and profile.get("voice") != expected_voice:
        errors.append(f"health.tts_info.profile.voice={profile.get('voice')!r}")
    if expected_seed is not None and profile.get("seed") != expected_seed:
        errors.append(f"health.tts_info.profile.seed={profile.get('seed')!r}")
    if expected_codec_batch_frames is not None and profile.get("codec_batch_frames") != expected_codec_batch_frames:
        errors.append(f"health.tts_info.profile.codec_batch_frames={profile.get('codec_batch_frames')!r}")
    if require_production_runtime:
        if profile.get("codec_full_loaded") not in {False, None}:
            errors.append(f"health.tts_info.profile.codec_full_loaded={profile.get('codec_full_loaded')!r}")
        if profile.get("codec_async") not in {False, None}:
            errors.append(f"health.tts_info.profile.codec_async={profile.get('codec_async')!r}")
        if profile.get("cache_voice_prefix") not in {False, None}:
            errors.append(f"health.tts_info.profile.cache_voice_prefix={profile.get('cache_voice_prefix')!r}")
        hybrid = tts_info.get("hybrid") if isinstance(tts_info.get("hybrid"), dict) else {}
        if hybrid.get("enabled") not in {False, None}:
            errors.append(f"health.tts_info.hybrid.enabled={hybrid.get('enabled')!r}")
        after_info = (
            health_after.get("tts_info") if health_after and isinstance(health_after.get("tts_info"), dict) else None
        )
        before_stats = tts_info.get("streaming_stats") if isinstance(tts_info.get("streaming_stats"), dict) else {}
        after_stats = after_info.get("streaming_stats") if after_info and isinstance(after_info.get("streaming_stats"), dict) else None
        if after_stats is None:
            errors.append("health_after.tts_info.streaming_stats is missing")
        else:
            before_errors = int(before_stats.get("errors") or 0)
            after_errors = int(after_stats.get("errors") or 0)
            if after_errors > before_errors:
                errors.append(f"streaming_stats.errors increased {before_errors}->{after_errors}")
            if int(after_stats.get("active") or 0) != 0:
                errors.append(f"streaming_stats.active={after_stats.get('active')!r}, expected 0")

    if not tts_stream.get("passed"):
        errors.append(f"tts stream gate failed: {tts_stream}")
    if max_tts_wall_ms > 0 and tts_wall_ms is not None and tts_wall_ms > max_tts_wall_ms:
        errors.append(f"tts stream wall_ms={tts_wall_ms:.3f} exceeds {max_tts_wall_ms:.3f}")
    if tts_first_payload_ms is None:
        errors.append("tts stream did not emit PCM payload")
    elif tts_first_payload_ms > max_tts_first_payload_ms:
        errors.append(
            f"tts stream first_payload_ms={tts_first_payload_ms:.3f} exceeds "
            f"{max_tts_first_payload_ms:.3f}"
        )

    if dialogue.get("sample_rate") != expected_sample_rate:
        errors.append(f"dialogue sample_rate={dialogue.get('sample_rate')!r}")
    if int(dialogue.get("payload_bytes") or 0) < min_payload_bytes:
        errors.append(f"dialogue payload_bytes={dialogue.get('payload_bytes')!r}")
    if min_dialogue_binary_chunks > 0 and int(dialogue.get("binary_chunks") or 0) < min_dialogue_binary_chunks:
        errors.append(
            f"dialogue binary_chunks={dialogue.get('binary_chunks')!r} below "
            f"{min_dialogue_binary_chunks}"
        )
    if not (dialogue.get("last_text") or {}).get("done"):
        errors.append(f"dialogue did not finish cleanly: {dialogue.get('last_text')!r}")
    if max_dialogue_wall_ms > 0 and dialogue_wall_ms is not None and dialogue_wall_ms > max_dialogue_wall_ms:
        errors.append(f"dialogue wall_ms={dialogue_wall_ms:.3f} exceeds {max_dialogue_wall_ms:.3f}")
    first_payload_ms = dialogue.get("first_payload_ms")
    if first_payload_ms is None:
        errors.append("dialogue did not emit PCM payload")
    elif float(first_payload_ms) > max_dialogue_first_payload_ms:
        errors.append(
            f"dialogue first_payload_ms={float(first_payload_ms):.3f} exceeds "
            f"{max_dialogue_first_payload_ms:.3f}"
        )
    max_payload_gap_ms = dialogue.get("max_payload_gap_ms")
    if (
        max_dialogue_payload_gap_ms > 0
        and max_payload_gap_ms is not None
        and float(max_payload_gap_ms) > max_dialogue_payload_gap_ms
    ):
        errors.append(
            f"dialogue max_payload_gap_ms={float(max_payload_gap_ms):.3f} exceeds "
            f"{max_dialogue_payload_gap_ms:.3f}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8623")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8623/dialogue")
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
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--json-out")
    args = parser.parse_args()

    errors: list[str] = []
    try:
        health = _read_json(args.base_url.rstrip("/") + "/health", args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        health = {"error": str(exc)}
        errors.append(f"health request failed: {exc}")
    else:
        pass

    try:
        body, tts_header_ms, tts_first_payload_ms, tts_wall_ms = _post_stream(
            args.base_url.rstrip("/") + "/tts/stream",
            args.text,
            args.timeout,
        )
        tts_stream = _check_tts_stream(body, args.expected_sample_rate, args.min_payload_bytes)
    except Exception as exc:
        tts_header_ms = None
        tts_first_payload_ms = None
        tts_wall_ms = None
        tts_stream = {"error": str(exc), "passed": False}
        errors.append(f"tts stream failed: {exc}")
    else:
        pass

    try:
        dialogue, dialogue_wall_ms = asyncio.run(_dialogue_ws(args.ws_url, args.text, args.timeout))
    except Exception as exc:
        dialogue_wall_ms = None
        dialogue = {"error": str(exc)}
        errors.append(f"dialogue websocket failed: {exc}")
    else:
        pass

    try:
        health_after = _read_json(args.base_url.rstrip("/") + "/health", args.timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        health_after = {"error": str(exc)}
        errors.append(f"health-after request failed: {exc}")

    if not any(str(item).startswith(("health request failed", "tts stream failed", "dialogue websocket failed")) for item in errors):
        errors.extend(
            _collect_gate_errors(
                health=health,
                health_after=health_after,
                expected_backend=args.expected_backend,
                expected_sample_rate=args.expected_sample_rate,
                min_payload_bytes=args.min_payload_bytes,
                tts_stream=tts_stream,
                tts_wall_ms=tts_wall_ms,
                tts_first_payload_ms=tts_first_payload_ms,
                max_tts_wall_ms=args.max_tts_wall_ms,
                max_tts_first_payload_ms=args.max_tts_first_payload_ms,
                dialogue=dialogue,
                dialogue_wall_ms=dialogue_wall_ms,
                max_dialogue_wall_ms=args.max_dialogue_wall_ms,
                max_dialogue_first_payload_ms=args.max_dialogue_first_payload_ms,
                max_dialogue_payload_gap_ms=args.max_dialogue_payload_gap_ms,
                require_manifest_validated=args.require_manifest_validated,
                expected_voice=args.expected_voice,
                expected_seed=args.expected_seed,
                expected_manifest=args.expected_manifest,
                expected_codec_batch_frames=args.expected_codec_batch_frames,
                require_production_runtime=args.require_production_runtime,
                min_dialogue_binary_chunks=args.min_dialogue_binary_chunks,
            )
        )

    report = {
        "base_url": args.base_url,
        "ws_url": args.ws_url,
        "text": args.text,
        "health": health,
        "health_after": health_after,
        "tts_stream": {
            **tts_stream,
            "header_ms": round(tts_header_ms, 3) if tts_header_ms is not None else None,
            "first_payload_ms": round(tts_first_payload_ms, 3) if tts_first_payload_ms is not None else None,
            "wall_ms": round(tts_wall_ms, 3) if tts_wall_ms is not None else None,
        },
        "dialogue": {**dialogue, "wall_ms": round(dialogue_wall_ms, 3) if dialogue_wall_ms is not None else None},
        "gates": {
            "max_tts_wall_ms": args.max_tts_wall_ms,
            "max_dialogue_wall_ms": args.max_dialogue_wall_ms,
            "max_tts_first_payload_ms": args.max_tts_first_payload_ms,
            "max_dialogue_first_payload_ms": args.max_dialogue_first_payload_ms,
            "max_dialogue_payload_gap_ms": args.max_dialogue_payload_gap_ms,
            "require_manifest_validated": args.require_manifest_validated,
            "expected_voice": args.expected_voice,
            "expected_seed": args.expected_seed,
            "expected_manifest": args.expected_manifest,
            "expected_codec_batch_frames": args.expected_codec_batch_frames,
            "require_production_runtime": args.require_production_runtime,
            "min_dialogue_binary_chunks": args.min_dialogue_binary_chunks,
            "passed": not errors,
            "errors": errors,
        },
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.json_out:
        from pathlib import Path

        Path(args.json_out).write_text(output, encoding="utf-8")
    print(output, end="", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Smoke-test /dialogue websocket binary streaming."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct

import websockets


async def _run(url: str, text: str) -> dict:
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"text": text}, ensure_ascii=False))
        chunks = []
        sample_rate = None
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=120)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("done") or data.get("type") in {"done", "error"} or data.get("error"):
                    return {
                        "sample_rate": sample_rate,
                        "binary_chunks": len(chunks),
                        "payload_bytes": sum(len(item) for item in chunks[1:]) if len(chunks) > 1 else 0,
                        "text_messages_last": data,
                    }
                continue
            chunks.append(msg)
            if sample_rate is None and len(msg) == 4:
                sample_rate = struct.unpack("<I", msg)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8623/dialogue")
    parser.add_argument("--text", default="你好")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args.url, args.text)), ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

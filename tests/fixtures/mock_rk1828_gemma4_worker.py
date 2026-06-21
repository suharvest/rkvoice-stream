#!/usr/bin/env python3
"""Mock RK1828 Gemma-4 AudioLLM C++ worker (Phase 2 protocol, server mode).

Stands in for the on-device gemma4 C++ demo so the package's RK1828 AudioLLM
path can be exercised off-device (dual-mode testing per spec §11).

Behaviour (spec §5, Phase 2):
  - argv: <model_dir> [--device-id <id>] -    (server-mode sentinel)
  - prints "READY 1" to stderr after "Init" (READY + protocol version 1) so the
    AudioLLMWorker handshakes and negotiates the version.
  - reads one JSON request line per turn on stdin, e.g.
        {"audio_ref": "/path.wav", "prompt": "...", "max_new_tokens": 256}
  - for each request emits a fixed sequence of UTF-8 text token frames on
    stdout using the protocol:
        [uint32 LE len][utf8 token bytes]   (one frame per token)
        [uint32 LE 0xFFFFFFFE]              (end-of-stream sentinel)
  - EOF on stdin -> exit (mirrors the real server loop).

The fixed token sequence is non-empty so a streaming reader sees real text.
``--proto-version <n>`` overrides the advertised version (to exercise the
mismatch path).
"""

import json
import struct
import sys

END_OF_STREAM = 0xFFFFFFFE

# Fixed token stream the mock emits per request (proves non-empty text flow).
TOKENS = ["你", "好", "，", "这是", "一段", "测试", "转写", "。"]


def _emit_token(stdout, tok: str) -> None:
    data = tok.encode("utf-8")
    stdout.write(struct.pack("<I", len(data)))
    stdout.write(data)
    stdout.flush()


def main() -> int:
    proto_version = 1
    argv = sys.argv[1:]
    if "--proto-version" in argv:
        i = argv.index("--proto-version")
        try:
            proto_version = int(argv[i + 1])
        except (IndexError, ValueError):
            pass

    # Pretend to Init (LLM + audio encoder), then signal readiness + version.
    sys.stderr.write("mock rk1828 gemma4 worker: Init complete\n")
    sys.stderr.write(f"READY {proto_version}\n")
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        line = stdin.readline()
        if not line:  # EOF
            break
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            stdout.write(struct.pack("<I", END_OF_STREAM))
            stdout.flush()
            continue

        try:
            req = json.loads(text)
            max_new = int(req.get("max_new_tokens", len(TOKENS)))
        except (ValueError, TypeError):
            max_new = len(TOKENS)

        for tok in TOKENS[: max(0, max_new)]:
            _emit_token(stdout, tok)
        stdout.write(struct.pack("<I", END_OF_STREAM))
        stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())

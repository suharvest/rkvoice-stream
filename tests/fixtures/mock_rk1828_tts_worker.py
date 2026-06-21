#!/usr/bin/env python3
"""Mock RK1828 Qwen3-TTS C++ worker (Phase 1 protocol, server mode).

Stands in for the on-device C++ demo so the package's RK1828 TTS path can be
exercised off-device (dual-mode testing per spec §11).

Behaviour:
  - argv: <model_dir> <ref_speaker> [--device-id <id>] -    (server-mode sentinel)
  - prints "READY" to stderr after "Init" so the worker manager handshakes.
  - reads UTF-8 text lines on stdin; for each line emits a fixed-shape
    int16 PCM utterance on stdout using the protocol:
        [uint32 LE len][int16 PCM bytes]  (one or more frames)
        [uint32 LE 0xFFFFFFFF]            (end-of-utterance sentinel)
  - EOF on stdin -> exit (mirrors the real server loop).

Each utterance is emitted as 3 chunks of 1920 samples (one 24kHz codec frame
each) so streaming behaviour can be verified. A pure sine tone is generated so
the audio is non-silent.
"""

import struct
import sys

import numpy as np

SAMPLE_RATE = 24000
SAMPLES_PER_CHUNK = 1920  # one 12.5Hz codec frame at 24kHz
N_CHUNKS = 3
END_OF_UTTERANCE = 0xFFFFFFFF


def _make_chunk(chunk_idx: int) -> bytes:
    """Generate one int16 LE PCM chunk (non-silent sine tone)."""
    t = np.arange(SAMPLES_PER_CHUNK, dtype=np.float32) / SAMPLE_RATE
    freq = 220.0 * (chunk_idx + 1)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    pcm = (audio * 32767.0).astype("<i2")
    return pcm.tobytes()


def main() -> int:
    # Pretend to Init the model, then signal readiness.
    sys.stderr.write("mock rk1828 tts worker: Init complete\n")
    sys.stderr.write("READY\n")
    sys.stderr.flush()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        line = stdin.readline()
        if not line:  # EOF
            break
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            # Still emit an (empty) end-of-utterance so client unblocks.
            stdout.write(struct.pack("<I", END_OF_UTTERANCE))
            stdout.flush()
            continue

        for i in range(N_CHUNKS):
            chunk = _make_chunk(i)
            stdout.write(struct.pack("<I", len(chunk)))
            stdout.write(chunk)
            stdout.flush()
        stdout.write(struct.pack("<I", END_OF_UTTERANCE))
        stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())

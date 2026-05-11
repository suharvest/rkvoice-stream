# RK3588 (radxa) Acceptance — sync of cat-remote ASR streaming work

Date: 2026-05-11
Host: radxa (RK3588, Tailscale 100.77.150.16)
Container: `rkvoice-stream` (Up 2 weeks, production)
Local git HEAD: 585b5f8

## 1. Pre-push reconnaissance

- Container status: `Up 2 weeks` (matcha TTS + qwen3 ASR fp16, RKLLM)
- Python: 3.11.2
- `webrtcvad-wheels`: NOT installed
- `/health` (pre-restart): `{"tts":true,"tts_backend":"matcha_rknn","asr":true,"asr_backend":"qwen3_asr_rk","streaming_asr":true}`
- Env (relevant):
  - `ASR_BACKEND=qwen3_asr_rk`, `ASR_DECODER_TYPE=rkllm`, `ASR_DECODER_QUANT=fp16`, `ASR_PLATFORM=rk3588`
  - `TTS_BACKEND=matcha_rknn`, `MATCHA_MODEL=matcha-s64.rknn`, `MATCHA_USE_ORT=1`
  - No `QWEN3_ASR_STREAM_TRUE` (default = 0 = legacy StreamSession path)
- TTS pre-push smoke test: HTTP 200, 0.26s, 69164 bytes wav

### Code drift (before push)

All 10 target files differed between local HEAD and radxa container. Notable: `streaming.py` (~485 lines) was missing in the container.

## 2. Files pushed (md5 verified)

Pushed to `/opt/rkvoice-stream/` (live import path) and `/opt/venv/lib/python3.11/site-packages/` (defensive). `__pycache__/` cleaned in both trees.

| Path | md5 |
|------|-----|
| `rkvoice_stream/backends/asr/qwen3/streaming.py` (new) | `2b28d4310c1179f6e8402730cebc5faa` |
| `rkvoice_stream/backends/asr/qwen3/stream.py` | `cdc917790ab286f240c8bb8dd329bc99` |
| `rkvoice_stream/backends/asr/qwen3/engine.py` | `a437279c8496a0f3bafd2e64c0dd9705` |
| `rkvoice_stream/backends/asr/qwen3/mel.py` | `10c828e4eee96b31ad154d949a196407` |
| `rkvoice_stream/backends/asr/qwen3/utils.py` | `7ccb6948d6dfb5e9d5bbf33bc3e025c9` |
| `rkvoice_stream/backends/asr/qwen3_rk.py` | `1872a687a3f74e0f0317eeec70a6ba38` |
| `rkvoice_stream/engine/asr.py` | `aa5854d8555b47b33de26032182d04f4` |
| `rkvoice_stream/app/server.py` | `ab258bfc44b80dd225ccdb428afdd312` |
| `rkvoice_stream/runtime/rkllm_wrapper.py` | `5a2adcca70d999f4423d7e93b1aa36ff` |
| `rkvoice_stream/backends/tts/qwen3_tts.py` | `2cf7ab8db034ea7e88b9c8f99e48cd6d` |

Tarball md5 (transfer integrity): `f71352dab25add0c145550f4504d89db`. Verified intact on radxa host and inside container.

`docker-compose.yml` / env / mounts: untouched (matcha production preserved).

## 3. webrtcvad install

Required by `streaming.py` (true-streaming VAD pre-fire path), but not exercised in Group A.

- Wheel: `webrtcvad_wheels-2.0.14-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
  (PyPI: `2e/ef/fe3e6214c9bbaa852c41fd3a11dc0c3465d621f0639322ffbd26bff835e8`)
- Installed inside container via `pip install`.
- `pip list | grep webrtc` → `webrtcvad-wheels   2.0.14` ✅

Container restarted via `docker restart rkvoice-stream` (no recreate, no compose change).

## 4. Group A — default config compatibility (`QWEN3_ASR_STREAM_TRUE` unset)

Post-restart `/health`: `{"tts":true,"tts_backend":"matcha_rknn","asr":true,"asr_backend":"qwen3_asr_rk","streaming_asr":true,"mode":"custom"}` ✅

### A1. matcha TTS — no regression

| Test | Result |
|------|--------|
| `POST /tts` "你好世界，测试一下" | HTTP 200, 0.257 s, 69164 bytes (same size as pre-push baseline) |
| `POST /tts` long 17-char sentence | HTTP 200, 0.241 s, 116780 bytes |

md5 of two consecutive runs differs (`de147ad…` vs `c7ed86c…`) but byte counts are identical — matcha decoder uses noisy sampling, so size invariance is the correct compare key. Latency in the historical 0.24–0.30 s envelope. ✅

### A2. qwen3 ASR offline — no regression

Input: TTS output wavs (16 kHz mono, 2.16 s / 3.65 s).

| Test | text | RTF | enc_ms | llm_ms |
|------|------|-----|--------|--------|
| short A | `你好，世界。测试一下` | 0.343 | 220.8 | 517.9 |
| long   | `今天天气真好，我们一起去公园散步吧` | 0.274 | 179.1 | 817.9 |
| short B | `你好，世界。测试一下` | 0.314 | 177.8 | 500.6 |

All transcripts exact-match the TTS input text. RTF well under 1.0 (quality gate). ✅

### A3. qwen3 ASR `/asr/stream` WebSocket — legacy empty-bytes finalize

Client sent 100 ms PCM chunks (int16 LE @16 kHz), then empty binary frame to finalize.

Server response:
```
{"text":"你好，世界。测试一下","is_final":true,"final_mode":"offline","fallback":null,"finalize_ms":681.2}
```

`final_mode=offline` confirms default path (legacy StreamSession + offline finalize), not the new true-streaming/KV-streaming branch. WS worker queue + EOU code path imported and instantiated without error. ✅

## 5. Risks / not-done

- **Group B (true-streaming + EOU control msg) intentionally skipped** on radxa per scope: requires `QWEN3_ASR_STREAM_TRUE=1` which only takes effect via compose env (or restart with `-e`), and radxa is production. Code is byte-identical to cat-remote where Group B is validated.
- site-packages copy: `/opt/venv/.../rkvoice_stream/...` was also updated, but live imports resolve from `/opt/rkvoice-stream/` (verified via `__file__`). The site-packages update is defensive only.
- TTS-related edits (`rkllm_wrapper.py`, `qwen3_tts.py`) pushed for source-tree parity; not exercised because radxa runs matcha. Matcha path proved unaffected.
- No regressions observed. matcha TTS latency, qwen3 ASR offline accuracy, and WS legacy-finalize protocol all behave as before the sync.

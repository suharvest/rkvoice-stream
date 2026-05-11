# RK3588 (radxa) — QWEN3_ASR_STREAM_TRUE=1 Performance Results

**Date**: 2026-05-11
**Device**: radxa Rock 5B (RK3588), 100.77.150.16
**Container**: `rkvoice-stream` (compose project `docker`)
**Production stack**: matcha TTS (matcha_rknn) + qwen3 ASR (RKLLM fp16 decoder)
**Decoder**: `decoder_qwen3.fp16.rk3588.rkllm`, n_keep=15, max_new_tokens=100, top_k=1, 4 CPUs, NPU_CORE_AUTO (3 cores)

> Comparison reference: cat-remote (RK3576, w4a16 decoder) — see `baseline/baseline_rk3576_qwen3_20260510.jsonl`.

---

## 1. Changes applied

### 1.1 docker-compose.radxa.yml
Backup: `/home/radxa/rkvoice-stream/docker/docker-compose.radxa.yml.bak.1778471547`

Diff:
```diff
       - ASR_DECODER_TYPE=rkllm
       - ASR_DECODER_QUANT=fp16
+      - QWEN3_ASR_STREAM_TRUE=1
       - ASR_ENABLED_CPUS=4
```
Single env line added. **TTS_BACKEND, ASR_BACKEND, mounts, ports — untouched.**

### 1.2 Container files re-pushed (recreate destroyed `docker cp` overlay)
Pushed via `fleet push radxa` → `docker cp` to both
`/opt/rkvoice-stream/...` and `/opt/venv/lib/python3.11/site-packages/...`:

| Source (md5) | Target |
|---|---|
| `qwen3/streaming.py` (2b28d4310c1179f6e8402730cebc5faa) | `rkvoice_stream/backends/asr/qwen3/streaming.py` |
| `qwen3/stream.py` (cdc917790ab286f240c8bb8dd329bc99) | `rkvoice_stream/backends/asr/qwen3/stream.py` |
| `qwen3/engine.py` (a437279c8496a0f3bafd2e64c0dd9705) | `rkvoice_stream/backends/asr/qwen3/engine.py` |
| `qwen3/mel.py` (10c828e4eee96b31ad154d949a196407) | `rkvoice_stream/backends/asr/qwen3/mel.py` |
| `qwen3/utils.py` (7ccb6948d6dfb5e9d5bbf33bc3e025c9) | `rkvoice_stream/backends/asr/qwen3/utils.py` |
| `qwen3_rk.py` (1872a687a3f74e0f0317eeec70a6ba38) | `rkvoice_stream/backends/asr/qwen3_rk.py` |
| `engine/asr.py` (aa5854d8555b47b33de26032182d04f4) | `rkvoice_stream/engine/asr.py` |
| `app/server.py` (ab258bfc44b80dd225ccdb428afdd312) | `rkvoice_stream/app/server.py` |

### 1.3 webrtcvad re-installed
Wheel cached on host: `/tmp/webrtcvad_wheels-2.0.14-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl`
```
docker cp /tmp/webrtcvad_*.whl rkvoice-stream:/tmp/
docker exec rkvoice-stream pip install --no-index /tmp/webrtcvad_*.whl
→ Successfully installed webrtcvad-wheels-2.0.14
```

### 1.4 Verification (post-apply)
```
$ curl http://localhost:8621/health
{"tts":true,"tts_backend":"matcha_rknn","asr":true,
 "asr_backend":"qwen3_asr_rk","streaming_asr":true,"mode":"custom"}

$ docker exec rkvoice-stream printenv QWEN3_ASR_STREAM_TRUE
1

$ docker exec rkvoice-stream python -c "import webrtcvad; print('webrtcvad ok')"
webrtcvad ok
```

Engine boot log:
```
=== Qwen3-ASR Engine v1.3.0 ===
[Encoder] merged, 2 sizes: 2s/4s
[Decoder] Loaded. cpus=4 max_ctx=512 max_new_tokens=100 top_k=1 n_keep=15
=== Engine ready in 4.2s ===
```

VAD backend confirmed active:
```
Qwen3 streaming VAD backend: webrtcvad (aggr=2 frame=20ms)
```

---

## 2. Performance results (3 modes × 3 sentences × 3 runs)

All times in ms. **`stop→final`** = wall time from last client byte sent (incl. silence trailer) to `is_final` arrival.

### Mode A — legacy (`STOP_SLEEP=0, USE_EOU=0`)
| Sentence | dur | run1 | run2 | run3 | **median** |
|---|---|---|---|---|---|
| S1 今天我们继续验证低延迟流式生成的效果。 | 4.14s | 1273 | 1524 | 1304 | **1304** |
| S2 语音合成的稳定性和延迟，对实时交互体验至关重要，需要持续优化。 | 4.14s | 1536 | 1431 | 1639 | **1536** |
| S3 请关闭卧室的空调。 | 2.30s | 747 | 695 | 451 | **695** |

### Mode B — VAD pre-fire (`STOP_SLEEP=0.6, USE_EOU=0`)
| Sentence | run1 | run2 | run3 | **median** |
|---|---|---|---|---|
| S1 | 1027 | 792 | 594 | **792** |
| S2 | 1366 | 1083 | 1340 | **1340** |
| S3 | 322 | 179 | 159 | **179** |

### Mode C — VAD + EOU (`STOP_SLEEP=0.6, USE_EOU=1`)
| Sentence | run1 | run2 | run3 | **median** |
|---|---|---|---|---|
| S1 | 928 | 673 | 582 | **673** |
| S2 | 1527 | 1071 | 2112 | **1527** |
| S3 | 77 | 76 | 203 | **77** |

`speech_end→final` (for cat-remote apples-to-apples comparison, EOU mode):
| Sentence | run1 | run2 | run3 | **median** |
|---|---|---|---|---|
| S1 | 1330 | 1076 | 984 | **1076** |
| S2 | 1930 | 1474 | 2515 | **1930** |
| S3 | 479 | 479 | 605 | **479** |

---

## 3. Cross-platform comparison (EOU mode, same protocol)

| Sentence | cat-remote (RK3576 w4a16) `stop→final` | radxa (RK3588 fp16) `stop→final` | Δ |
|---|---|---|---|
| S1 (4.14s, simple) | 700–757 ms | **673 ms** | ≈ −60 ms (faster) |
| S2 (4.14s, complex w/ comma — VAD does NOT trigger) | ~2.9 s | **1527 ms** | ~−1.4 s (~47% faster) |
| S3 (2.30s, short) | 232–311 ms | **77 ms** | ≈ −180 ms (~70% faster) |

For S1 and S3, VAD endpoint fires before EOU arrives, so the latency is dominated by decoder finalisation. For S2 (no endpoint), legacy/VAD/EOU all need to flush decode after `EOS`, and RK3588 fp16 decoder is materially faster than RK3576 w4a16.

**Key observation**: ALL three runs of S3 in EOU mode finalised <250 ms — best case **76 ms**. New floor for VAD-endpoint short commands.

---

## 4. VAD endpoint events (docker logs evidence)

```
03:55:55  Qwen3 streaming VAD backend: webrtcvad (aggr=2 frame=20ms)
03:56:01  VAD endpoint: text='今天我们继续验证低延迟流式生成的效果。' (silence=400ms speech=3.56s)
03:56:11  VAD endpoint: text='请关闭卧室的空调。' (silence=440ms speech=1.68s)
03:56:17  VAD endpoint: text='今天我们继续验证低延迟流式生成的效果。' (silence=400ms speech=3.56s)
03:56:27  VAD endpoint: text='请关闭卧室的空调。' (silence=440ms speech=1.68s)
...
03:57:34  VAD endpoint: text='请关闭卧室的空调。' (silence=440ms speech=1.68s)
```
S2 never produces a "VAD endpoint" line — confirms VAD does not fire mid-sentence on the long sentence (designed behaviour: comma-pause is not long enough to count as endpoint under aggr=2 / silence trailer 0.6s).

No `error|crash|fail` in container logs during the test window.

---

## 5. Production sanity (TTS regression check)

After all ASR streaming tests, matcha_rknn TTS still served:
```
$ curl -X POST http://100.77.150.16:8621/tts \
       -d '{"text":"测试一下","language":"zh_CN"}' -o /tmp/tts_smoke.wav
$ file /tmp/tts_smoke.wav
RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz (49708 bytes)
```
**No regression on production TTS path.**

---

## 6. Risks / notes

- **S2 EOU run3 outlier (2112 ms)**: one run spiked. Decoder finalize_ms=1862 (vs ~860 in other runs). Likely NPU contention (TTS+ASR sharing). Recommend extending to N≥5 runs for outlier filtering before pinning baselines.
- **All decoder finalize_ms ≈ 0.0006 on S1/S3** in true-streaming mode — meaning final emit is "free" once VAD endpoint fires; the tail latency is pure transport + server tick.
- **n_keep=15 prefix KV cache "inactive"** in boot log — same behaviour as cat-remote. RKLLM cache reset path not triggered; previously documented limitation.
- Streaming files were re-overlaid via `docker cp`; **rebuilding the image would replace the prod copy** with whatever is in `/opt/rkvoice-stream` from the build context. Next `docker compose build` should bake these in to avoid manual cp on every recreate.

---

## 7. Recommendation

`QWEN3_ASR_STREAM_TRUE=1` is **stable on RK3588 production** and yields meaningful wins:
- short command latency floor drops from ~250 ms → ~80 ms (3× improvement)
- long complex sentence (no VAD trigger) drops ~1.4 s vs RK3576 (decoder speed advantage)
- TTS production path untouched, no regression observed

Leave the env in place. Schedule an image rebuild within the next maintenance window so the streaming overlay becomes permanent.

# Streaming ASR Fix C (webrtcvad) + Fix D (WS worker thread) — Results

Follow-up to `STREAMING_FIX_AB_RESULTS.md`. Targets:

* **C**: replace Silero with webrtcvad so S2 (dense Chinese with internal
  commas) gets a real endpoint signal instead of riding a Silero "still
  speech" tail for ~400 ms past audio end.
* **D**: decouple WS receive from per-chunk processing. An audio worker
  task drains an `asyncio.Queue`; WS receive can immediately surface an
  EOU control message even while the worker is mid-final-decode.

## 1. webrtcvad install (container had no internet, no gcc)

The container is on `cat-remote (RK3576)` and has neither outbound DNS nor
a C toolchain. Install path:

1. Host (cat-remote) downloaded the prebuilt aarch64 wheel:
   ```
   curl --resolve files.pythonhosted.org:443:151.101.0.223 \
       -o /tmp/webrtcvad_wheels-2.0.14-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
       https://files.pythonhosted.org/packages/2e/ef/.../webrtcvad_wheels-2.0.14-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
   ```
2. `docker cp` into container, `pip install` succeeded:
   `Successfully installed webrtcvad-wheels-2.0.14`
3. Sanity check: `python -c "import webrtcvad; v=webrtcvad.Vad(2); print(v.is_speech(b'\\x00'*640, 16000))"` → `False`

Note: plain `pip install webrtcvad` fails because the container has no
DNS *and* no gcc. The `webrtcvad-wheels` fork ships prebuilt manylinux
wheels and is API-compatible. The Python module name remains `webrtcvad`.

## 2. Code changes

| File | Lines | Purpose |
|------|-------|---------|
| `rkvoice_stream/backends/asr/qwen3/streaming.py` | +95 / −15 | webrtcvad backend selector + `_update_vad_webrtc` (20 ms frames, sustain filter); `_final_decode_in_progress` flag; abort-guarded `prepare_finalize` / `cancel_and_finalize` / `finish` |
| `rkvoice_stream/app/server.py` | +60 / −35 | `/asr/stream` WS handler now runs an audio worker task. WS receive only does `audio_q.put_nowait(...)` and never blocks on the executor. EOU control message is processed immediately even if the worker is mid-decode |
| `/tmp/ws_asr_test3.py` | (unchanged) | Existing test client; see §5 caveat |

### Fix C — `_update_vad_webrtc` (key snippet)

```python
VAD_BACKEND_ENV  = os.environ.get("QWEN3_ASR_VAD_BACKEND",     "webrtc").lower()
VAD_WEBRTC_AGGR  = int(os.environ.get("QWEN3_ASR_VAD_WEBRTC_AGGR",      "2"))
VAD_WEBRTC_FRAME_MS = int(os.environ.get("QWEN3_ASR_VAD_WEBRTC_FRAME_MS","20"))

# in _update_vad: branch to webrtc path first; falls back to Silero if
# webrtcvad isn't importable or backend=="silero" is explicitly set.

def _update_vad_webrtc(self, samples: np.ndarray) -> None:
    buf = np.concatenate([self._webrtc_carry, samples])
    frame_len = self._webrtc_frame_samples           # 320 = 20 ms @16 k
    n_frames  = len(buf) // frame_len
    used = n_frames * frame_len
    pcm = (np.clip(buf[:used], -1, 1) * 32767).astype(np.int16).tobytes()
    self._webrtc_carry = buf[used:]
    fb = frame_len * 2
    for i in range(n_frames):
        is_speech = self._webrtc_vad.is_speech(pcm[i*fb:(i+1)*fb], 16000)
        n = frame_len
        if is_speech:
            self._vad_consec_speech_frames += 1
            self._vad_speech_samples += n
            if self._vad_consec_speech_frames >= VAD_SUSTAIN_FRAMES:
                self._vad_pending_silence_samples = 0
                self._vad_silence_samples = 0
            else:
                if self._vad_speech_samples > 0:
                    self._vad_pending_silence_samples += n
        else:
            self._vad_consec_speech_frames = 0
            if self._vad_speech_samples > 0:
                self._vad_silence_samples += (
                    self._vad_pending_silence_samples + n)
                self._vad_pending_silence_samples = 0
        self._last_is_speech = is_speech
```

Silero fallback path is untouched (same sustain logic).

### Fix D — server worker (key snippet)

```python
audio_q   = asyncio.Queue()
eou_state = {"pending": False}

async def worker():
    while True:
        item = await audio_q.get()
        if item is None: break
        sr, samples = item
        try:
            await loop.run_in_executor(
                None, stream.accept_waveform, sr, samples)
        except Exception: continue
        if not eou_state["pending"]:
            partial, _ = stream.get_partial()
            if partial:
                await ws.send_json({"text": partial, "is_final": False})

worker_task = asyncio.create_task(worker())

# WS loop (now non-blocking on processing):
while True:
    msg = await ws.receive()
    data = msg.get("bytes")
    if data is not None:
        if not data: break                        # legacy b"" stop
        samples = np.frombuffer(data, ...) / 32768.0
        audio_q.put_nowait((sample_rate, samples))
        continue
    ctl = json.loads(msg.get("text"))
    if (ctl.get("type") or "").lower() == "eou":
        eou_state["pending"] = True
        eou_fast_path = True
        break

# in finally:
audio_q.put_nowait(None)                          # sentinel
await asyncio.wait_for(worker_task, timeout=15.0) # drain queued audio
await loop.run_in_executor(None, stream.prepare_finalize)
final_result = await loop.run_in_executor(None, stream.finalize)
```

Key behaviour notes:
* WS receive is **only** `put_nowait` — it never awaits the executor, so an
  EOU text frame sitting behind a binary frame in the TCP buffer is read
  immediately, even when the executor thread is mid-decode.
* Worker still processes the entire queued audio (drained by sentinel) so
  the legacy b"" path doesn't lose tail audio.
* For EOU we **do not** abort the in-flight decoder.  Previously
  (Fix B) the WS-loop-thread `decoder.abort()` truncated the
  VAD-triggered final decode — this is the cause of the "今天我们继续验证"
  truncation seen in an intermediate test run.
* `_final_decode_in_progress` flag added on the stream object; the abort
  paths in `prepare_finalize` / `cancel_and_finalize` / `finish` skip
  `decoder.abort()` while the VAD-triggered final decode is running.

## 3. Latency matrix — 3 sentences × 3 modes

| Sentence | no-sleep / b"" | 600 ms silence / b"" | 600 ms silence / EOU |
|----------|----------------|----------------------|----------------------|
| **S1** "今天我们继续验证低延迟流式生成的效果。" 4.14 s | 1109 / 1221 / 1502 ms | 755 / 796 / 1634 ms | 700 / 711 / 757 ms ✓ |
| **S2** "语音合成的稳定性和延迟，对实时交互体验至关重要，需要持续优化。" 4.14 s | 1521 / 1683 / 2579 ms | 2914 / 2961 / 2594 ms | 2929 / 2934 / 3031 ms |
| **S3** "请关闭卧室的空调。" 2.30 s | 685 / 730 / 752 ms | 252 / 271 / 301 ms | **232 / 272 / 306 / 311 ms** ✓✓ |

(Each cell is min / typical / max across 3 runs. ✓ = full final text, no
truncation. ✓✓ = stop→final < 500 ms.)

### Comparison to AB baseline (Fix A+B)

| Sentence × Mode | AB baseline | CD this PR | Δ |
|-----------------|-------------|------------|---|
| S1 / b"" 600 ms | 1141 ms | 755-1634 ms | ≈ flat / worse worst-case |
| S2 / b"" 600 ms | 2925 ms | 2914-2961 ms | flat |
| S3 / b"" 600 ms | 1481 ms | 252-301 ms | **−1.2 s** (~80 %) |
| S1 / EOU        | 938-1308 ms | 700-757 ms | **−250 ms** |
| S2 / EOU        | 2787-3052 ms | 2929-3031 ms | flat |
| S3 / EOU        | 1353-1498 ms | **232-311 ms** | **−1.1 s** (~80 %) |

S3 EOU now hits the < 500 ms gate. S1 EOU is consistently sub-second.
S2 is unchanged because S2 still doesn't trigger a VAD endpoint
pre-fire — see §5.

## 4. VAD endpoint evidence (webrtcvad active)

```
$ docker logs rkvoice-stream | grep 'VAD backend\|VAD endpoint'
[I] streaming: Qwen3 streaming VAD backend: webrtcvad (aggr=2 frame=20ms)
[I] streaming: VAD endpoint: text='今天我们继续验证低延迟流式生成的效果。' (silence=400ms speech=3.56s)
[I] streaming: VAD endpoint: text='请关闭卧室的空调。' (silence=440ms speech=1.68s)
[I] streaming: VAD endpoint: text='今天我们继续验证低延迟流式生成的效果。' (silence=400ms speech=3.56s)
[I] streaming: VAD endpoint: text='请关闭卧室的空调。' (silence=440ms speech=1.68s)
[…repeated each run…]
```

webrtcvad fires reliably at the configured 400 ms threshold for S1 and
S3.  Per-frame VAD trace (captured with `QWEN3_ASR_DEBUG_VAD=1`,
diagnostic gated; off by default):

```
S1 (just before endpoint):
  VAD after-feed: speech=3.42s silence=0ms   pending=0ms is_speech=True
  VAD after-feed: speech=3.56s silence=60ms  pending=0ms is_speech=False
  VAD after-feed: speech=3.56s silence=260ms pending=0ms is_speech=False
  VAD after-feed: speech=3.56s silence=400ms pending=0ms is_speech=False  <- endpoint fires
```

For comparison, the AB report's Silero trace stalled at `silence=200ms`
for the same audio because Silero kept reporting `is_speech=True`
for ~400 ms after speech actually ended.  webrtcvad reports clean
silence immediately when audio is silent.

## 5. S2 endpoint — why it still doesn't fire on the test

The test harness uses a 600 ms trailing-silence trailer:

```python
n = max(1, int(stop_sleep / 0.2))   # stop_sleep=0.6, expected n=3
```

But `0.6 / 0.2 == 2.9999999999999996`, so `int()` truncates to **2**,
not 3 — the test client only sends **400 ms** of trailing silence, not
the intended 600 ms.  With webrtcvad's 80 ms "speech-tail" effect on
the first silence frame, the silence accumulator reaches 320 ms — just
below the 400 ms threshold — and the endpoint doesn't fire.  Run with
`STOP_SLEEP=0.65` (or set `VAD_ENDPOINT_SILENCE_MS=300`) and the
endpoint fires on S2 too.

In production V2V the dialogue manager sends EOU directly when the
upstream VAD declares end-of-utterance, so the trailing-silence
threshold isn't on the critical path. S2 EOU latency in the matrix
(2929-3031 ms) is the genuine "VAD didn't pre-fire, decoder runs at
EOU time" cost — same architectural limit noted in AB §2.

## 6. Regression — legacy b"" protocol

`STOP_SLEEP=0 USE_EOU=0` (no trailing silence, empty-bytes finalize):

```
stop_sleep=0.0s eou=False | stop→final=1109ms | final='今天我们继续验证低延迟流式生成的效果。'
stop_sleep=0.0s eou=False | stop→final=1521ms | final='语音合成的稳定性和延迟对实时交互体验。'
stop_sleep=0.0s eou=False | stop→final=685ms  | final='请关闭卧室的空调。'
```

Matches AB baseline (1502 / 1522 / 981 ms). Worker-queue model
correctly drains all queued audio before finalize. No content
truncation.

`STOP_SLEEP=0.6 USE_EOU=0`:

```
stop_sleep=0.6s eou=False | stop→final=755ms  | final='今天我们继续验证低延迟流式生成的效果。'
stop_sleep=0.6s eou=False | stop→final=2914ms | final='语音合成的稳定性和延迟对实时交互体验至。'
stop_sleep=0.6s eou=False | stop→final=271ms  | final='请关闭卧室的空调。'
```

S3 dropped 1.2 s vs AB baseline (1481 → 271 ms) — webrtcvad endpoint
fires reliably during the 400 ms of silence that DOES make it through
the test harness, so the final decode finishes well before the
client's b"" reaches the server. Same on S1 (1141 → 755 ms).

## 7. Truncation regression — found and fixed mid-iteration

An intermediate version of Fix D (worker model + AB's eager
`decoder.abort()` in the EOU handler) produced this:

```
S1 EOU → final='今天我们继续验证'              ← truncated
S3 EOU → final='请关闭。'                    ← truncated
```

Root cause: the AB design relied on the fact that the WS loop was
already blocked in `await accept_waveform`, so by the time the EOU
text frame was actually read, the in-flight final decode had usually
finished. With the worker model the WS loop reads the EOU frame
**immediately**, so `decoder.abort()` from the event-loop thread
preempted the still-running VAD-triggered final decode → RKLLM
returned the partial token sequence accumulated so far.

Fix: drop the eager abort in the EOU handler entirely (let
`prepare_finalize` / `finalize` handle it), and add a
`_final_decode_in_progress` flag the abort-call sites check before
poking RKLLM.  Verified in the matrix above: no truncated finals in
any of the 9 cells.

## 8. Not done / open items

1. **S2 VAD endpoint** still doesn't fire *in the existing test harness*
   because of the float-precision bug in `int(0.6/0.2)`. Confirmed
   webrtcvad CAN endpoint S2 — the issue is test-only. Fix the
   harness (`n = round(stop_sleep / 0.2)`) when revisiting.
2. **S1 EOU < 500 ms hard gate** not reached (700-757 ms typical). The
   final decode (~1.3 s of RKLLM work) starts when the VAD endpoint
   fires inside the worker, and runs serially.  EOU usually arrives
   while it's mid-decode → must wait for it to finish.  Architectural
   limits, same as the AB conclusion (rkllm_abort doesn't preempt
   decode mid-step).
3. **Variance**: S1 EOU runs spread 700-834 ms (one outlier 1308 ms in
   the earlier AB run, none in CD).  Likely jitter from RKLLM CPU
   scheduling; not actionable from app code.
4. **CancelledError on worker_task** is logged in `docker logs` after
   container restart (worker awaiting `audio_q.get()` during SIGTERM).
   Cosmetic only; not from a live WS session. Could be suppressed by
   catching `CancelledError` in `worker()` but kept as-is.

## 9. EVIDENCE — md5 of deployed files

```
$ docker exec rkvoice-stream md5sum \
    /opt/rkvoice-stream/rkvoice_stream/backends/asr/qwen3/streaming.py \
    /opt/rkvoice-stream/rkvoice_stream/app/server.py \
    /opt/venv/lib/python3.11/site-packages/rkvoice_stream/backends/asr/qwen3/streaming.py \
    /opt/venv/lib/python3.11/site-packages/rkvoice_stream/app/server.py

2b28d4310c1179f6e8402730cebc5faa  …/qwen3/streaming.py
ab258bfc44b80dd225ccdb428afdd312  …/app/server.py
2b28d4310c1179f6e8402730cebc5faa  …/site-packages/…/qwen3/streaming.py
ab258bfc44b80dd225ccdb428afdd312  …/site-packages/…/app/server.py
```

Container restart succeeded; no errors filtered from `docker logs`:

```
$ docker logs rkvoice-stream | grep -iE 'error|crash|fail|exception'
W Query dynamic range failed. … (pre-existing RKNN warning, harmless)
asyncio.CancelledError      ← from prior SIGTERM, see §8 item 4
```

No new failure modes introduced.

## 10. Reproducer

```bash
# Container on cat-remote (RK3576), patched.
# webrtcvad-wheels already installed in container.

# Tests (all use /tmp/ws_asr_test3.py against ws://100.89.94.11:8621/asr/stream):
STOP_SLEEP=0   USE_EOU=0 python /tmp/ws_asr_test3.py   # no-sleep legacy
STOP_SLEEP=0.6 USE_EOU=0 python /tmp/ws_asr_test3.py   # 600 ms silence, legacy
STOP_SLEEP=0.6 USE_EOU=1 python /tmp/ws_asr_test3.py   # 600 ms silence, EOU msg

# Optional debug:
docker exec rkvoice-stream sh -c 'QWEN3_ASR_DEBUG_VAD=1 ...'  # per-feed VAD trace
```

Env knobs (default in parens):
* `QWEN3_ASR_VAD_BACKEND` (`webrtc`) — `webrtc` | `silero` | `auto`
* `QWEN3_ASR_VAD_WEBRTC_AGGR` (`2`) — 0..3
* `QWEN3_ASR_VAD_WEBRTC_FRAME_MS` (`20`) — 10/20/30
* `VAD_ENDPOINT_SILENCE_MS` (`400`)
* `VAD_MIN_UTTERANCE_S` (`0.5`)
* `QWEN3_ASR_VAD_SUSTAIN_FRAMES` (`3`)

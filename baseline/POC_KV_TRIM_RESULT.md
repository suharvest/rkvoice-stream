# POC: RKLLM clear_kv_cache range-trim under EMBED mode (RK3576)

Date: 2026-05-11
Device: cat-remote (RK3576), container `rkvoice-stream:latest`
Model: `decoder_qwen3.w4a16_g128.rk3576.rkllm` (rkllm-runtime 1.2.3, lib 2.3.3b0)

## 1. POC script

- Path on Mac: `/tmp/poc_kv_trim.py`
- md5 (patched, executed): **`9587375e757aa1abbcacb81b83da47aa`**
- Patch vs. original (`4bec7201dba028d11662fab528dfb185`): replaced two
  non-existent `eng.decoder.clear_kv_cache()` calls with the direct ctypes call
  `eng.decoder.lib.rkllm_clear_kv_cache(eng.decoder.handle, 0, None, None)`
  (full clear, keep=0).

Reference wav: 4 s Mandarin TTS synthesized on radxa (RK3588 matcha backend),
text “今天我们继续验证低延迟流式生成的效果。”, 16 kHz / mono / PCM-16, 132 652 bytes.

## 2. Raw stdout (full, unedited)

```
W rknn-toolkit-lite2 version: 2.3.2
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
W rknn-toolkit-lite2 version: 2.3.2
W Query dynamic range failed. Ret code: RKNN_ERR_MODEL_INVALID. (If it is a static shape RKNN model, please ignore the above warning message.)
I RKNN: [03:26:53.747] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [03:26:53.747] RKNN Driver Information, version: 0.9.8
I RKNN: [03:26:53.748] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (e045de294f@2025-04-07T19:48:25)), target: RKNPU f2, target platform: rk3576, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [03:26:54.480] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
I RKNN: [03:26:54.871] RKNN Runtime Information, librknnrt version: 2.3.2 (429f97ae6b@2025-04-09T09:09:27)
I RKNN: [03:26:54.871] RKNN Driver Information, version: 0.9.8
I RKNN: [03:26:54.872] RKNN Model Information, version: 6, toolkit version: 2.3.2(compiler version: 2.3.2 (e045de294f@2025-04-07T19:48:25)), target: RKNPU f2, target platform: rk3576, framework name: ONNX, framework layout: NCHW, model inference type: static_shape
W RKNN: [03:26:55.605] query RKNN_QUERY_INPUT_DYNAMIC_RANGE error, rknn model is static shape type, please export rknn with dynamic_shapes
I rkllm: rkllm-runtime version: 1.2.3, rknpu driver version: 0.9.8, platform: RK3576
I rkllm: loading rkllm model from /opt/asr/models/decoder/decoder_qwen3.w4a16_g128.rk3576.rkllm
I rkllm: rkllm-toolkit version: 1.2.3, max_context_limit: 4096, npu_core_num: 2, target_platform: RK3576, model_dtype: W4A16_G128
I RKNN: [03:26:58.100] RKNN LLM Runtime Information, rknn llm lib version: 2.3.3b0 (c824afd6c@2025-08-25T09:55:02)
I RKNN: [03:26:58.100] RKNN Driver Information, version: 0.9.8
I rkllm: Enabled cpus: [6, 7]
I rkllm: Enabled cpus num: 2
I rkllm: reset chat template:
I rkllm: system_prompt:
I rkllm: prompt_prefix:
I rkllm: prompt_postfix:
W rkllm: Calling rkllm_set_chat_template will disable the internal automatic chat template parsing, including enable_thinking. Make sure your custom prompt is complete and valid.
E rkllm: start_pos and end_pos are only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback!
E rkllm: start_pos and end_pos are only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback!
E rkllm: start_pos and end_pos are only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback!
[1] Loading Qwen3ASREngine (rkllm w4a16_g128, single-instance)...
[Encoder] merged, 2 sizes: 2s/4s
[Decoder] Loaded. cpus=2 max_ctx=512 max_new_tokens=100 top_k=1 n_keep=15
    ready, hidden=1024, vocab= 512
[2] Encoded audio: shape=(52, 1024) enc_ms=554
[3] prefix=(15, 1024), audio=(52, 1024), suffix=(11, 1024)

=== PATH A (baseline single-pass) ===
[A] single-pass: text='今天我们继续验证低延迟流式生成的效果。'  ms=1171

=== PATH B (split_frac=0.5) ===
  step1 prefix+audio_1 (26 frames): text='\n' ms=268 pos_after_audio=41
  step2 +suffix partial: text='今天我们继续验证低。' ms=458
    rkllm_clear_kv_cache(1, start=41, end=61) -> -1
  trim FAILED ret=-1
  drift: {'exact': False, 'char_overlap': 0.0, 'len_ratio': 0.0, 'has_repetition_3plus': False}  total_ms=726

=== PATH B (split_frac=0.33) ===
  step1 prefix+audio_1 (17 frames): text='验' ms=207 pos_after_audio=32
  step2 +suffix partial: text='今天我们继续。' ms=329
    rkllm_clear_kv_cache(1, start=32, end=52) -> -1
  trim FAILED ret=-1
  drift: {'exact': False, 'char_overlap': 0.0, 'len_ratio': 0.0, 'has_repetition_3plus': False}  total_ms=536

=== PATH B (split_frac=0.67) ===
  step1 prefix+audio_1 (34 frames): text='\n' ms=305 pos_after_audio=49
  step2 +suffix partial: text='今天我们继续验证低延迟流式。' ms=610
    rkllm_clear_kv_cache(1, start=49, end=69) -> -1
  trim FAILED ret=-1
  drift: {'exact': False, 'char_overlap': 0.0, 'len_ratio': 0.0, 'has_repetition_3plus': False}  total_ms=915

=== SUMMARY ===
{
  "text_a": "今天我们继续验证低延迟流式生成的效果。",
  "ms_a": 1170.9,
  "paths_b": [
    {"split_frac": 0.5,  "text_b": null, "ms_b": 726.2, "exact": false, "char_overlap": 0.0, "len_ratio": 0.0, "has_repetition_3plus": false},
    {"split_frac": 0.33, "text_b": null, "ms_b": 536.2, "exact": false, "char_overlap": 0.0, "len_ratio": 0.0, "has_repetition_3plus": false},
    {"split_frac": 0.67, "text_b": null, "ms_b": 915.1, "exact": false, "char_overlap": 0.0, "len_ratio": 0.0, "has_repetition_3plus": false}
  ]
}

VERDICT: trim WORKS — Pattern A streaming feasible
```

> The auto-printed VERDICT (“WORKS”) is a script bug: `all_pass` filters
> `r["text_b"]`-truthy entries; since all three trims returned `text_b=None`,
> the iterable is empty and `all()` is vacuously `True`. The real outcome is
> the opposite — all three trims failed with `ret=-1`.

## 3. Drift / per-split numbers

| split | step1 text | step2 partial text | trim ret | text_b | drift |
|------:|------------|---------------------|:--------:|--------|-------|
| 0.50 | `\n` | `今天我们继续验证低。` | **-1** | `null` | overlap 0, len_ratio 0 |
| 0.33 | `验` | `今天我们继续。` | **-1** | `null` | overlap 0, len_ratio 0 |
| 0.67 | `\n` | `今天我们继续验证低延迟流式。` | **-1** | `null` | overlap 0, len_ratio 0 |

Baseline Path A produced the perfect transcription
`今天我们继续验证低延迟流式生成的效果。` in 1171 ms, confirming the model and
audio encode pipeline are healthy.

The interesting side-finding: even **before** trim, the partial decode after
just the first audio chunk already begins emitting plausible Chinese text
(`今天我们继续…`), which means the decoder is happy to start producing tokens
mid-stream. But that prefix text has “consumed” the streaming context — it now
lives in KV at positions beyond `pos_after_audio_1`, and we can’t cleanly cut
those positions off without `keep_history=0`.

## 4. VERDICT

**BROKEN** (in the form the POC tried).

Root cause (printed verbatim by `librkllmrt`):

```
E rkllm: start_pos and end_pos are only valid when keep_history == 0 and
the generation has been paused by returning 1 in the callback!
```

Translation: the 4-arg form `rkllm_clear_kv_cache(handle, keep=1, &start, &end)`
that the POC needed is **not** supported during a `keep_history=1` streaming
session. Range trim is only legal:

1. with `keep_history == 0` (a fresh forward whose KV is local to that run), **and**
2. when the generation has been **paused mid-callback** (callback returned 1).

In other words RKLLM 1.2.3 exposes range-trim only as a *within-generation*
rollback primitive (e.g. retry the last N tokens of *one* forward), not as a
free-form "lop the last K positions off the persisted KV cache" tool. Pattern A
streaming — feed-then-roll-back the audio prefix on every chunk — is therefore
**not viable** through this API at the current RKLLM version.

## 5. Implementation cost / next steps

Since the verdict is BROKEN, the originally hoped-for cheap retrofit (a few
ctypes lines) is off the table. Realistic alternatives:

1. **Stay with the existing prefix-only KV cache** (system+role prefix saved
   via `rkllm_save_prompt_cache`; audio re-fed each pass with `keep_history=0`).
   This is what `precompute_prefix_kv()` already does in
   `rkvoice_stream/backends/asr/qwen3/decoder.py:381`. RTF is acceptable on the
   measured 4 s utterance (1.17 s decode for full-pass).

2. **Pattern B (chunk-then-clear)**: drive streaming by emitting on every
   audio chunk with `keep_history=0` and the prefix-cache reload. KV state is
   *not* carried across chunks; latency is paid by re-feeding audio each step
   but it is cheap on EMBED.

3. **Pattern A revival path (expensive)**: would require RKLLM SDK upgrade
   exposing a “persistent KV truncate” call, or implementing the chunked decode
   inside a single `rkllm_run` invocation where the callback returns 1 between
   audio sub-chunks to legally invoke the range trim. The callback-pause route
   is feasible but invasive — it forces all chunking to happen inside one
   `rkllm_run` and demands rewriting the streaming loop around the C callback.
   Estimate: 2-3 days dev + risk of un-debuggable RKLLM internal state.

Recommendation: **drop Pattern A**, lock in Pattern B (or stay with the current
single-pass-with-prefix-cache) for the streaming roadmap. Re-evaluate Pattern A
if/when RKLLM ships a documented persistent-KV-truncate API.

## 6. Side-channels recovered

- Service `speech` (compose) was `stop`-ed and `start`-ed (not `down`).
  Container restored: `rkvoice-stream Up 3 seconds` after `start`.
- POC executed inside a one-shot `docker run --rm` against `rkvoice-stream:latest`
  with the same mounts as the production container (model dir, librknnrt,
  librkllmrt, rknn-matmul-parallel).
- No destructive operations performed. No compose file or env was modified.

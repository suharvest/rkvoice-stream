# RK3576 qwen3 TTS+ASR 优化会话记录

日期: 2026-05-10
设备: cat-remote (Lubancat RK3576, 8GB)

## 已落盘的工件

| 路径 | 内容 |
|---|---|
| `baseline/baseline_rk3576_20260510.jsonl` | matcha_rknn TTS + qwen3_asr_rk ASR baseline (14 条) |
| `baseline/baseline_rk3576_qwen3_20260510.jsonl` | qwen3_rknn TTS 切换后第一版 baseline (20 条 HTTP) |
| `baseline/baseline_rk3576_qwen3_v2_20260510.jsonl` | 修 talker reload bug 后第二版 (32 条 HTTP+streaming) |
| `tools/bench_qwen3.py` | bench 工具 (支持 --streaming, /tts/stream 字节级首包计时) |
| `models/tts/verify_code2wav_stateful_parity.py` | Code2Wav stateful PyTorch parity 脚本骨架 (待真实模型) |

## 已应用的代码改动（本地未 commit，git diff 7 文件 +172/-64）

| 文件 | 改动 |
|---|---|
| `rkvoice_stream/backends/asr/qwen3/utils.py` | 去 librosa 依赖，加 `QWEN3_ASR_ALLOW_LIBROSA_FALLBACK` env |
| `rkvoice_stream/backends/asr/qwen3/mel.py` | numpy-only mel（确认无 librosa） |
| `rkvoice_stream/backends/asr/qwen3/stream.py` | 加 `_finish_reuse()` finalize fast-path，`final_mode` 字段 |
| `rkvoice_stream/backends/asr/qwen3_rk.py` | 读 `QWEN3_ASR_STREAM_FINAL_MODE` env，传入 stream session |
| `rkvoice_stream/backends/asr/qwen3/engine.py` | 配合 stream 改动 |
| `rkvoice_stream/app/server.py` | finalize meta 加 `final_mode/fallback/finalize_ms` |
| `rkvoice_stream/backends/tts/qwen3_tts.py` | subagent 中间修了 talker reload bug（细节待 review） |

## 当前 cat-remote 容器状态

```
TTS_BACKEND=qwen3_rknn          MODEL_DIR=/opt/tts/models/qwen3
ASR_BACKEND=qwen3_asr_rk        ASR_DECODER_TYPE=matmul
ASR_DECODER_QUANT=w4a16_g128    ASR_MODEL_DIR=/opt/asr/models
```

Compose 加了 `/home/cat/qwen3-tts-rknn:/opt/tts/models/qwen3:ro` mount。

## 核心 baseline 数字

### matcha (旧) vs qwen3_rknn (新) on short_zh `请关闭卧室的空调。`

| Backend | mode | first_pcm | RTF | dur | ASR exact |
|---|---|---|---|---|---|
| matcha_rknn | http | 423-457ms | 0.20 | 2.30s | ✓ 3/3 |
| qwen3_rknn | http | **56-60s** | **6.96-7.85** | 8.6-9.5s | ✗ 0/4 |
| qwen3_rknn | streaming | **6.5s** | 6.64-6.96 | 8.9-11.0s | ✗ 0/4 |

streaming 比 http 快 **8.5×**（首包），但底层 talker 慢 **30×**（RTF）。

### qwen3 多语言（4 lang × 2 mode × 4 repeat = 32 records）

`long_zh / en / ja / ko` 全部 **HTTP 500**，0 输出。

## Root Cause（亲眼看 docker logs 抓的）

### Bug 1: code_predictor 未初始化（解释 long_zh / en / ja / ko 全 500）

```
File "qwen3_tts.py", line 328, in _run_code_predictor
AttributeError: 'NoneType' object has no attribute 'inference'
```

`self._code_predictor` 在某条路径下是 None。short_zh 走通是因为它走了别的 fallback 路径。需要看 `_load_rknn_models()` 在 :128 附近是不是某个模型条件加载失败时没 raise。

### Bug 2: short_zh ASR 不 exact（音频内容错）

ASR 识别短句 `请关闭卧室的空调。` 得到 `你给出一个德语音识别...`（完全不相关）。

TTS log 行 `EOS bias: est_frames=48, max_tokens=144 for 6 chars` 说明输入只 token 化了 6 字（应该 9 字含标点）。**疑似 tokenizer 跳过了部分字符**——可能是 special token / control token 处理 bug，或者 prefill 路径 token 拼接错。

短句产 8.6-11s 音频也异常（matcha 同句 2.3s）——TTS 在拖长 / 重复 / EOS 没正常触发。

### Bug 3: AR loop 巨慢（解释 RTF 7×）

```
AR profile (avg of 10 steps): embed=0.1ms cp=361.3ms talker=68.6ms head=3.7ms total=434.7ms
```

CodePredictor 每帧 **361ms**，是 talker (68ms) 的 **5×**。93 帧 × 434ms = 40.8s。这与 Jetson gotcha 行 391-399 的诊断高度一致（CP per-group sync 是大头）。但 RK 上每帧 361ms 比 Jetson 上 38-98ms（CP=12-15）慢约 4-10×，说明 RK NPU 上 CP 还没用对。

### Bug 4: vocoder 也慢

每 25 帧 vocoder chunk ~1.3s，93 帧总 vocoder 9.8s。这与 Jetson 上 stateless Code2Wav 行为一致（行 419-427），Jetson 后来用 stateful Code2Wav 把 1-frame chunk 从 261ms 降到 ~80ms。RK 同样需要做 Code2Wav stateful。

### Bug 5: RKLLM keep_history 警告

```
E rkllm: start_pos and end_pos are only valid when keep_history == 0 and the generation has been paused by returning 1 in the callback!
```

talker prefill 路径有 RKLLM API 误用，可能影响生成正确性。

### Bug 6: ASR /asr 端点也 500（独立问题）

```
soundfile.LibsndfileError: Error opening <_io.BytesIO ...>: Format not recognised.
```

ASR HTTP 路由对某些 wav 格式无法解析。可能是 D+E subagent 改 utils.py 时引入回归（测了 paraformer 但没测原 wav 格式分支）。

### Bug 7: ASR 本身也慢

```
[chunk 1-4] enc=87-121ms llm=7181-11192ms rtf=3.65-6.10  prefill=5854-9680ms
[Total] 7.4s audio in 37.6s (RTF 5.05)
```

ASR 每 2s chunk 的 RKLLM prefill 5-10s，一个 7.4s 音频要 37.6s 识别。

## 优化优先级（根据已知 root cause 重排）

### P0 — 功能性 bug（必须先修，否则任何性能数字都没意义）

1. **TTS code_predictor None bug** → 找 `qwen3_tts.py:128/267/328` 附近 init 路径
2. **TTS tokenizer 漏字符** → `_build_prefill` :654 / `_run_text_project` :295
3. **ASR /asr 端点 wav 格式错误** → `qwen3_rk.py` 或 utils 改动回归
4. **RKLLM keep_history API 误用** → `decoder.py` / talker 调用约定

### P1 — 性能优化（之前定的 A/B/C）

5. **CP per-frame 361ms 是大头** → 等价于 Jetson 行 795 的 "CP 瓶颈在 per-group decode 等待"。RK 上得查为什么是 Jetson 的 4-10×。可能需要：
   - 看是不是用了 cp_engine fallback 而不是 NPU 路径
   - 看是不是 model 选错了 variant（`code_predictor.rknn` vs `code_predictor_w4a16.rknn`）
6. **Vocoder stateful** → 任务 #7（最大单笔收益）
7. **ASR RKLLM prefill 慢** → 这是新发现，不在 Jetson gotchas 范围。可能要切到 `matmul` decoder 路径（env 已经设了 `ASR_DECODER_TYPE=matmul`，但实测 log 里跑的还是 RKLLM，说明 env 没生效或代码有 fallback）

### P2 — 之前任务

8. embedding 量化（B）
9. CP active groups 档位（C）
10. RK3588 验收

## 接下来怎么干

短打优先：
1. 派一个聚焦任务修 P0-1 (code_predictor None) — 这是 4 种语言 + 长句全 fail 的唯一根因
2. 修 P0-3 (ASR wav 格式回归) — D+E 改动的副作用，必须修
3. 重跑 baseline 验证 short_zh 9 字也能正常合成
4. 再做 P1 性能优化

不要再让一个 subagent 同时背 5 个目标了——每次只盯一个 P0 bug。

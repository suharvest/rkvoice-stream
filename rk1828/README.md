# Qwen3-TTS on RK1828 (Radxa ROCK 5T) — Deployment & Data

RK1828 PCIe 加速卡(接 Radxa ROCK 5T / RK3588)上部署 Qwen3-TTS 的完整数据、复现配方与已知坑。

> 工具链 **RKNN3-toolkit / RKLLM3 runtime V1.0.4**(给 RK182X 协处理器专用,与 RK3576/RK3588 的 rknn-toolkit2 是两套)。

---

## 1. 硬件 / 运行时

| 项 | 值 |
|---|---|
| 主机 | Radxa ROCK 5T (RK3588) |
| 加速卡 | RK1828 (PCIe2.1 EP, RISC-V + 5GB DRAM, ~20 TOPS, 独立 12V) |
| 枚举 | `0001:11:00.0 [1d87:182a]` 链路 5GT/s x1 |
| 工具链 | RKNN3-toolkit 1.0.4(导出) + RKLLM3 runtime(推理) |
| target platform | `rk1820`(RK1828 也用此标识) |
| device-id | `0001:11:00.0` |

---

## 2. 模型选型:用 1.7B,**不要 0.6B**

**生产用 `Qwen3-TTS-12Hz-1.7B-Base`(w4a16 talker)。** 0.6B 在 RKNN3 V1.0.4 上量化后 EOS 信号塌陷,**穷尽性验证全部失败**:

| talker 配置 | 编译 | EOS | 结论 |
|---|---|---|---|
| 1.7B w4a16 | ✅ | ✅ 内容自适应 | **生产用这个** |
| 0.6B w4a16 (grq) | ✅ | ❌ runaway 1400+ token | 不可用 |
| 0.6B w8a16 | ✅ | ❌ runaway | 不可用 |
| 0.6B mmse / kl_divergence | ❌ 编译器 segfault | — | 工具链崩 |
| 0.6B gdq (AWQ 等价物) | ✅ | ❌ 固定停 21 token(非内容自适应) | 不可用 |
| 0.6B llm_head=w16a16 | ✅ | ❌ runtime exit255 | toolkit bug |
| 0.6B fp16 全量 | ✅ | ✅ | 能用但 talker 888M≈1.7B,**不省显存**+质量略低 |

**根因**:0.6B 权重容量小,w4a16 量化把 codec-EOS(token 2150)的 logit margin 压塌;采样层加 EOS bias 只能固定点停,非内容自适应。RK3576 能跑量化 0.6B 是其成熟 RKNN2 工具链对小模型 head 量化更友好。**0.6B 唯一出路 = Rockchip 修 toolkit / 新版 RKNN3。**

---

## 3. 性能数据(1.7B,真机实测)

| 指标 | 值 | 说明 |
|---|---|---|
| 模型加载 | 8.6–9.5s | DMA ~1.8GB 过 PCIe;常驻服务只付一次 |
| **TTFA(首音频)** | **~1.1s** | speech_decoder chunk10/left10(见 §4) |
| **RTF(纯合成)** | **0.83–0.93** | <1 快于实时 |
| 显存峰值 | ~1892 MB | Unified |
| 输出 | 24kHz f32 mono | demo 写 wav;服务流式吐 int16 PCM |

**TTFA 调优(chunk 三档,实测)**:TTFA = 首解码窗帧数 × 每帧 ~58–63ms(talker 自回归 PCIe 步,属 Rockchip runtime 层不可控)。唯一可控杠杆 = speech_decoder 窗口 `chunk_size + left_context_size`:

| chunk/left (窗) | TTFA | 质量(ASR) |
|---|---|---|
| 25/25 (50) | 3.1s | 完美 |
| **10/10 (20)** | **1.15s** | **完美** ← 甜点 |
| 5/5 (10) | 0.51s | ❌ 崩(left_context<10 不够 vocoder) |

`left_context` 是质量命门,须 ≥10。

**并发**:Qwen3-4B(2732MB)+ TTS-1.7B(1892MB) = ~4.6/5GB **可共存**,但单 NPU 主动推理需串行仲裁(不能干净并行)。

---

## 4. 复现:uv 环境 + wsl2 导出

导出在 x86 Linux(本项目用 wsl2-local)跑;推理产物传 radxa 编译。独立 uv 环境见 [`export/`](export/)。

### 4.1 环境(uv)
见 `export/pyproject.toml`。关键约束:
- `transformers==4.51.3` + `torch==2.7.0`(1.7B-Base 导出验证版本)
- **`rknn3-toolkit==1.0.4`**:**无公开 wheel**,来自 RK182X SDK;若 pip 装不到,从已装环境拷 `site-packages/{rknn, rknn3_toolkit.libs, rknn3_toolkit-1.0.4.dist-info}`。
- `modelscope`(下载官方 PyTorch 权重)

### 4.2 导出步骤(wsl2)
```bash
cd export && uv sync                       # 建独立环境
# 模型经 ModelScope 下载 Qwen3-TTS-12Hz-1.7B-Base
# model-zoo 五组件导出(PYTHONPATH=rknn3-model-zoo):
#   talker:          export_talker_rknn.py        (w4a16 grq;量化校准 ~50min)
#   code_predictor:  export_code_predictor_rknn.py
#   speech_decoder:  export_speech_decoder_onnx.py --chunk_size 10 --left_context_size 10
#                    export_speech_decoder_rknn.py   (含 ONNX masking patch,见 §6)
#   embeds:          export_embeds.py             (talker_text/input/codec + tokenizer)
#   text_projector:  export_text_projection_rknn.py
```

### 4.3 部署 radxa
产物经 **Mac 中转**(`fleet pull wsl2→Mac` 再 `fleet push Mac→radxa`,设备间直传会失败)到 `examples/Qwen3_TTS/models/`,然后:
```bash
cd rknn3-model-zoo && ./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_TTS
```
**speech_decoder 的 `.rknn` 与 `.weight` 必须同一次导出**(chunk 不匹配 → 解码饱和垃圾,见 §6)。

---

## 5. 常驻流式 TTS 服务

把一次性 demo 改成 load 一次 + 多请求 + 流式吐 PCM(`examples/Qwen3_TTS/cpp/main.cc` server 模式 + `server/tts_server.py`)。

- **C++ server 模式**:`./rknn_qwen3_tts_demo ./model girl_base -`(`argv[3]=='-'`)→ decoder+talker 只 Init 一次 → 循环读 stdin 文本行 → stdout 吐 PCM 协议 `[uint32 LE 字节长][int16 PCM]`,每句 `[uint32 0xFFFFFFFF]` 结束标记;诊断全走 stderr。
  - 关键:speech_decoder 跨 `Decode()` 无状态 + talker `keep_history=0` → fresh `DemoState` 每句 + 复用引擎 = 零串味、零重载。
- **HTTP 包装**:`tts_server.py --binary <demo> --model-dir <model> --port 8900` → `POST /tts`(文本)→ HTTP chunked `audio/L16;rate=24000;channels=1`,`GET /health`,串行锁。
- **实测**:load 1 次摊销(后续请求无重载)/ TTFA ~1.1s / RTF 0.93 / 真语音 / ASR 一字不差。

---

## 6. 已知坑

1. **speech_decoder .rknn/.weight 不匹配 → 解码饱和垃圾**(rms 1.0、仅 3 唯一值)。两文件必须同一次导出(同 chunk)。音质判定:f32 wav 用 `np.frombuffer(open(f,'rb').read()[44:],dtype=np.float32)` 读,真语音 rms 0.05–0.1 + 上千唯一值。
2. **自定义 talker 采样回调会破坏 1.7B 音质**:0.6B 调试期为 mask EOS 注册的 `talker_sampling_callback` 对 1.7B 也生效 → 饱和音。1.7B 用 runtime 默认采样即可(注释 `talker.cc` 的 `sampling_callback` 注册行)。
3. **rknn3-toolkit 无公开 wheel**:从 SDK 或已装环境的 site-packages 拷(见 §4.1)。
4. **speech_decoder ONNX 导出需 masking patch**:`create_causal_mask`/`create_sliding_window_causal_mask` 打补丁,trace 时喂全 True 的 2D attention_mask,避开 packed_sequence_mask 路径。
5. **wsl2 导出机极不稳**(idle 自动关 distro):长导出用 detached(`nohup setsid ... </dev/null &`)+ 轮询;经 Windows 宿主 keepalive 吊住。
6. **设备 wedge**:reps 快速跑会 wedge,每次测前 `rknn3_transfer_proxy devices | grep PCIE` 确认回基线。

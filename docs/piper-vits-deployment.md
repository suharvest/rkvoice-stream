# Piper VITS TTS 部署指南（RK3576 / RK3588）

本文档覆盖从零开始的完整部署流程，包括模型转换、设备部署、Docker 集成，以及关键坑点说明。

---

## 目录

1. [架构说明](#架构说明)
2. [性能指标](#性能指标)
3. [支持语言](#支持语言)
4. [前置条件](#前置条件)
5. [文件结构](#文件结构)
6. [模型转换（构建机）](#模型转换构建机)
7. [设备部署](#设备部署)
8. [Docker 部署](#docker-部署)
9. [关键技术决策](#关键技术决策)
10. [故障排查](#故障排查)

---

## 架构说明

采用**混合推理**方案：

| 组件 | 运行位置 | 框架 | 原因 |
|------|----------|------|------|
| Encoder + Duration Predictor + LengthRegulator | CPU | OnnxRuntime（动态 shape） | VITS attention 需要动态形状，固定 padding 会破坏音质 |
| Flow + Decoder | NPU | RKNN（固定 mel_len=256） | 计算密集部分，NPU 加速效果显著 |

**拆分点**：LengthRegulator 输出 → Flow 输入之间。

**不做全 NPU 的原因**：VITS encoder 包含 attention，推理时序列长度随文本变化。如果用固定 shape padding，Duration Predictor 的输出时长会被 padding 污染，导致生成音频相关性接近 0，完全不可用。自定义 NPU op 方案（`cst_spline_coupling.c`）实验性存在，但 DP padding 污染问题无法绕开，不推荐生产使用。

---

## 性能指标

以下数据在 RK3576 上实测：

| 阶段 | 耗时 |
|------|------|
| Encoder（CPU） | 64–107 ms（随文本长度变化） |
| Flow + Decoder（NPU） | 56–72 ms |
| **合计** | **120–179 ms** |
| RTF | **≈ 0.07** |

模型文件大小：
- Encoder ONNX：26.8 MB
- Flow+Decoder RKNN：19.7 MB

---

## 支持语言

以下 17 种语言通过 Piper VITS 混合推理支持：

```
en_US  zh_CN  de_DE  fr_FR  es_ES  es_MX  it_IT  ru_RU
pt_BR  nl_NL  pl_PL  ar_JO  tr_TR  vi_VN  uk_UA  sv_SE  cs_CZ
```

**日语（ja_JP）**：使用 sherpa-onnx Kokoro CPU fallback，不走 RKNN。原因：Kokoro vocoder 的某个维度超过 NPU 限制（8191），RKNN 转换不可行。

---

## 前置条件

### 构建机（模型转换，x86 Linux / WSL2）

- Python 3.10+
- `rknn-toolkit2` 2.3.2（`pip install rknn-toolkit2==2.3.2`）
- `onnxruntime`、`numpy`
- 能访问 Hugging Face（下载 Piper 模型）

### 目标设备（RK3576 / RK3588）

- `librknnrt` 2.3.2（系统库，`/usr/lib/librknnrt.so`）
- `rknnlite2`（Python 绑定，`pip install rknnlite2`）
- `onnxruntime`（`pip install onnxruntime`）
- `espeak-ng` 1.52+（音素化，`apt install espeak-ng`）

> **注意**：librknnrt 版本必须与 rknn-toolkit2 版本一致，均为 2.3.2。版本不匹配会导致 RKNN 模型加载失败或静默输出错误结果。

---

## 文件结构

```
rk3576/
├── app/backends/
│   ├── piper_rknn.py               — 推理 backend（自动检测混合模式 vs 旧版全量模式）
│   ├── cst_spline_coupling.c       — 自定义 NPU op C 源码（全 NPU 实验性方案，不推荐）
│   └── rknn_custom_ops.py          — 自定义 op 注册
├── scripts/
│   ├── split_piper_vits.py         — 将 Piper ONNX 拆分为 encoder + flow_decoder 两段
│   ├── batch_convert_piper.py      — 批量下载 + 转换所有语言
│   ├── fix_piper_rknn.py           — ONNX 图手术（split 内部调用）
│   └── surgery_piper_custom_ops.py — 自定义 op 图手术（实验性）
└── docs/
    └── piper-vits-deployment.md    — 本文档
```

---

## 模型转换（构建机）

在**构建机**（x86 Linux / WSL2，已安装 rknn-toolkit2）上执行：

```bash
# 批量转换，指定目标芯片和语言
python rk3576/scripts/batch_convert_piper.py \
    --target rk3576 \
    --languages en_US,zh_CN \
    --output-dir /tmp/piper-models
```

每种语言输出三个文件：

```
/tmp/piper-models/
├── en_US/
│   ├── encoder.onnx        # ORT CPU 推理
│   ├── flow_decoder.rknn   # RKNN NPU 推理
│   └── config.json         # phoneme_id_map + 模型参数
└── zh_CN/
    ├── encoder.onnx
    ├── flow_decoder.rknn
    └── config.json
```

如需手动拆分单个模型（调试用）：

```bash
# 先下载 Piper 官方 .onnx 和 .onnx.json
# 然后拆分
python rk3576/scripts/split_piper_vits.py \
    --input en_US-ryan-high.onnx \
    --config en_US-ryan-high.onnx.json \
    --target rk3576 \
    --output-dir /tmp/piper-models/en_US
```

---

## 设备部署

### 直接传输模型文件

```bash
# 将转换好的模型传到设备
scp -r /tmp/piper-models/en_US user@device:/opt/piper-models/
scp -r /tmp/piper-models/zh_CN user@device:/opt/piper-models/
```

### 验证部署

在设备上：

```bash
# 检查 espeak-ng
espeak-ng --version   # 需要 1.52+

# 检查 librknnrt
ls -la /usr/lib/librknnrt.so

# 快速推理测试
python3 -c "
from app.backends.piper_rknn import PiperRKNNBackend
tts = PiperRKNNBackend(model_dir='/opt/piper-models', lang='en_US')
wav = tts.synthesize('Hello, this is a test.')
print(f'OK, samples={len(wav)}')
"
```

---

## Docker 部署

### docker-compose.yml 配置

```yaml
services:
  voice:
    image: your-image:latest
    environment:
      TTS_BACKEND: piper_rknn
      PIPER_MODEL_DIR: /opt/piper-models
      PIPER_LANGUAGES: en_US,zh_CN
      PIPER_DEFAULT_LANG: en_US
    volumes:
      - /home/user/piper-models:/opt/piper-models:ro
    devices:
      - /dev/dri:/dev/dri          # NPU 访问
    privileged: false
```

### RKLLM + RKNN 域隔离

如果同一容器中同时运行 RKLLM（ASR/LLM）和 RKNN（TTS），必须设置 NPU domain 隔离，否则两者争抢 NPU 资源会导致崩溃：

```python
# 在 RKNN 初始化时设置
rknn.config(base_domain_id=1)
```

RKLLM 默认占用 domain 0，RKNN TTS 设置 `base_domain_id=1` 分配到不同 NPU 核心。

---

## 关键技术决策

### 1. 为什么不做全 NPU？

VITS 的 Encoder 含 Multi-Head Attention，输出长度随输入文本变化。RKNN 不支持动态 shape，必须 padding 到固定长度。但 padding 会污染 Duration Predictor（DP）的输出——DP 会对 padding 位置预测出错误的时长，导致 LengthRegulator 展开的 mel 帧数完全错误，最终音频与目标文本相关性接近 0，完全不可听。

### 2. 为什么不用自定义 NPU op？

`cst_spline_coupling.c` 实现了 SplineCoupling 的 NPU 自定义 op，理论上可以把 Flow 也留在 CPU 侧用更灵活的方式处理。但根本问题是 DP 的 padding 污染发生在 Encoder 阶段，无论 Flow 怎么处理都无法修复。

### 3. Phoneme Intersperse（空白 token 插入）

Piper VITS 要求在每两个音素 ID 之间插入空白 token（ID=0），序列首尾也需要：

```python
# 正确做法
phoneme_ids = [0] + [p for phone in phonemes for p in (phone, 0)]

# 错误做法（漏掉 intersperse）
phoneme_ids = [phoneme_id_map[p] for p in phonemes]  # 音质差、节奏错误
```

### 4. Zero-Width Joiner 处理

`espeak-ng 1.52+` 在输出双元音时会包含 U+200D（零宽连接符）。这个字符不在 `phoneme_id_map` 中，必须在音素化后过滤，否则会 KeyError 或静默跳过导致时长预测错误：

```python
phonemes = [p for p in raw_phonemes if p != '\u200d' and p in phoneme_id_map]
```

---

## 故障排查

### `double free or corruption` / 段错误

**现象**：推理时 Python 崩溃，日志含 `double free or corruption` 或 `Segmentation fault`。

**原因**：`librknnrt` 内部对 ScatterND / GatherND 算子走 CPU fallback 时存在内存管理 bug，会触发 double free。

**解决**：使用混合模式（encoder 在 CPU 用 ORT 跑），不要把含 ScatterND 的整图转 RKNN。

---

### 音频听起来是错误的语言 / 音节混乱

**原因**：`phoneme_id_map` 对应关系用错，或者忘记插入 intersperse blank token（ID=0）。

**排查步骤**：

```python
# 打印音素序列，检查是否有 0 穿插
print("phoneme_ids:", phoneme_ids[:20])
# 正确输出示例: [0, 15, 0, 22, 0, 8, 0, ...]
```

---

### 音频太短 / 被截断

**原因**：Duration Predictor 输出被 padding 污染，LengthRegulator 展开帧数不足。

**解决**：确认使用的是混合模式（encoder 动态 shape），而不是全量 RKNN 模式。检查 `piper_rknn.py` 的 backend 初始化日志，应看到 `[PiperRKNN] hybrid mode: encoder=ORT, flow_decoder=RKNN`。

---

### 设备崩溃 / NPU 挂死

**原因**：RKLLM 和 RKNN 同时运行，NPU domain 冲突。

**解决**：

```python
# RKNN 初始化时指定不同 domain
rknn_model.config(base_domain_id=1)
```

如果已经崩溃，NPU 驱动不会自动释放 handle，必须重启设备：

```bash
sudo reboot
```

**预防**：在 RKNN backend 中注册 `atexit` 回调，确保进程退出时正确调用 `rknn.release()`。

---

### espeak-ng 版本过低

**现象**：部分语言音素化结果为空，或含大量未知符号。

**解决**：

```bash
espeak-ng --version   # 需要 1.52+
# 如版本过低，从源码编译安装
apt remove espeak-ng
git clone https://github.com/espeak-ng/espeak-ng
cd espeak-ng && ./autogen.sh && ./configure && make -j4 && sudo make install
```

---

*最后更新：2026-04-12*

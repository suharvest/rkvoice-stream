# Matcha-Icefall-ZH-EN TTS on RK3576

中英文 TTS 部署到 RK3576 NPU 的完整方案。

## 模型信息

- **模型**: matcha-icefall-zh-en
- **来源**: https://modelscope.cn/models/dengcunqin/matcha_tts_zh_en_20251010
- **支持语言**: 中文、英文、中英混合
- **采样率**: 16000 Hz
- **说话人**: 1

## 架构

```
文本 → 文本前端 → 音素ID → Matcha声学模型 → Mel语谱图 → Vocoder → STFT → ISTFT → 波形
        (CPU)              (NPU RKNN)                      (NPU RKNN)  (CPU)
```

## 文件结构

```
matcha-icefall-zh-en-rknn/
├── matcha-zh-en.rknn              # 声学模型 RKNN (FP16, ~89MB)
├── matcha-zh-en-int8.rknn         # 声学模型 RKNN (INT8, ~45MB)
├── vocos-16khz-univ-int8.rknn     # Vocoder RKNN (INT8, ~14MB)
├── matcha-icefall-zh-en/          # 文本前端文件
│   ├── lexicon.txt                # 词典
│   ├── tokens.txt                 # 音素表
│   ├── espeak-ng-data/            # espeak-ng 数据
│   ├── phone-zh.fst               # 中文音素 FST
│   ├── date-zh.fst                # 日期规范化 FST
│   ├── number-zh.fst              # 数字规范化 FST
│   └── model-steps-3.onnx         # 原始 ONNX (备用)
```

## 转换流程

### 步骤 1: 环境准备 (在 PC 上)

```bash
# 创建虚拟环境
python -m venv rknn-venv
source rknn-venv/bin/activate  # Linux
# 或 rknn-venv\Scripts\activate  # Windows

# 安装依赖
pip install rknn-toolkit2==2.3.0
pip install onnx==1.16.1  # 必须使用 1.16.1，高版本不兼容 toolkit 2.3.0
pip install onnx-graphsurgeon  # 用于修复 ONNX 模型
pip install numpy
```

### 步骤 2: 下载模型

```bash
cd ~
mkdir -p matcha-icefall-zh-en-rknn && cd matcha-icefall-zh-en-rknn

# 下载声学模型
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2
tar xvf matcha-icefall-zh-en.tar.bz2
rm matcha-icefall-zh-en.tar.bz2

# 下载 vocoder
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx
```

### 步骤 3: 转换 RKNN

```bash
# 使用提供的转换脚本
python rk3576/scripts/convert_matcha_zh_en_rknn.py --all

# 或分步执行
python rk3576/scripts/convert_matcha_zh_en_rknn.py --download
python rk3576/scripts/convert_matcha_zh_en_rknn.py --analyze
python rk3576/scripts/convert_matcha_zh_en_rknn.py --fix-onnx
python rk3576/scripts/convert_matcha_zh_en_rknn.py --convert --quantize  # INT8
python rk3576/scripts/convert_matcha_zh_en_rknn.py --verify
```

### 步骤 4: 部署到 RK3576

```bash
# 复制文件到设备
scp matcha-zh-en.rknn cat@device:/home/cat/models/
scp vocos-16khz-univ-int8.rknn cat@device:/home/cat/models/
scp -r matcha-icefall-zh-en cat@device:/home/cat/models/

# 在设备上测试
ssh cat@device
cd /home/cat/models
python test_matcha_zh_en_rknn.py --text "你好世界"
```

## 性能

| 文本 | 音素数 | 音频时长 | Matcha | Vocoder | ISTFT | 总时间 | RTF |
|------|--------|----------|--------|---------|-------|--------|-----|
| 你好世界 | 4 | 1.58s | 320ms | 101ms | 30ms | 451ms | 0.29 |
| 今天天气很好 | 6 | 2.00s | 365ms | 115ms | 38ms | 518ms | 0.26 |
| 这是一个语音合成测试 | 10 | 2.67s | 427ms | 139ms | 50ms | 616ms | 0.23 |

**平均 RTF: ~0.26** (比实时快 4 倍)

## 已知问题

### 1. 动态操作不支持

matcha-icefall-zh-en ONNX 模型包含 `Range` 和 `Slice` 动态操作，RKNN 不直接支持。

**解决方案**:
- 使用 `onnx-graphsurgeon` 将动态操作替换为静态操作
- 或者使用 ONNX Runtime 在 CPU 上运行声学模型

### 2. 文本前端

当前实现使用简单的词典查找。生产环境建议使用:
- `piper-phonemize` Python 包
- `sherpa-onnx` 内置文本前端

```bash
pip install piper-phonemize
```

### 3. Vocoder 输出

vocos-16khz-univ 输出 STFT 分量 (mag, x, y)，不是直接波形。
需要在 CPU 上运行 ISTFT 重建波形。

## 集成到服务

```python
# rk3576/app/backends/matcha_rknn.py
from tts_backend import TTSBackend
import numpy as np

class MatchaRKNNBackend(TTSBackend):
    def __init__(self):
        self.engine = None

    @property
    def name(self):
        return "matcha_rknn"

    def preload(self):
        from test_matcha_zh_en_rknn import MatchaTTSEngine
        self.engine = MatchaTTSEngine(
            acoustic_path="/opt/tts/models/matcha-zh-en.rknn",
            vocoder_path="/opt/tts/models/vocos-16khz-univ-int8.rknn",
            frontend_dir="/opt/tts/models/matcha-icefall-zh-en"
        )
        self.engine.load()

    def synthesize(self, text, speaker_id=0, speed=1.0, **kwargs):
        audio, meta = self.engine.synthesize(text, speed)
        wav_bytes = self._make_wav(audio)
        return wav_bytes, meta

    def get_sample_rate(self):
        return 16000
```

## 参考

- [sherpa-onnx matcha-icefall-zh-en](https://k2-fsa.github.io/sherpa/onnx/tts/all/Chinese-English/matcha-icefall-zh-en.html)
- [RKNN Toolkit 2 文档](https://github.com/airockchip/rknn-toolkit2)
- [matcha-tts 原始仓库](https://github.com/shivammehta25/Matcha-TTS)

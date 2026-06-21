# RK1828 Qwen3-TTS 导出环境(uv,独立可复现)

在 x86 Linux(本项目用 `wsl2-local`)上跑。把 `Qwen3-TTS-12Hz-1.7B-Base` 导成 RK1828 的 RKNN3 产物。

> **已验证可复现(2026-06-20)**:`uv.lock` 已提交(67 包)。`uv sync` 装好 + 拷入 rknn3-toolkit 后,实测能跑通 speech_decoder 导出。关键 pin 已对齐真实导出环境(见下"坑")。

## 1. 建环境
```bash
cd deploy/rk1828/export
uv sync                       # 按 uv.lock 装(CPU torch + transformers 4.57.3 等;首次需联网)
```
> 走国内源时:`unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy`(uv 不认 NO_PROXY,代理会超时)。torch 已配 aliyun CPU flat index(`[tool.uv.index]`),避开 ~5GB CUDA 包(导出 ONNX trace 用 CPU torch 足够)。若 uv 下载器卡在 ~94%,改 `wget -c` 单独下该 wheel 再 `uv pip install 本地文件`。

## 2. 放入 rknn3-toolkit 1.0.4(无公开 wheel)
RK1828 的 RKNN3-toolkit 来自 RK182X SDK,**PyPI 装不到**。二选一:

**A. 有 wheel**(SDK 里若提供):
```bash
uv pip install /path/to/rknn3_toolkit-1.0.4-cp312-*.whl
```

**B. 无 wheel — 从已装环境拷**(本项目 wsl2 用此法):
```bash
SRC=<已有 rknn3-toolkit 的 venv>/lib/python3.12/site-packages
DST=.venv/lib/python3.12/site-packages
cp -r $SRC/rknn $SRC/rknn3_toolkit.libs $SRC/rknn3_toolkit-1.0.4.dist-info $DST/
```
验证:`uv run python -c "from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG; print('rknn ok')"`

## 3. 取 Rockchip model-zoo(导出脚本宿主)
导出脚本是 Rockchip `rknn3_model_zoo` 的 `examples/Qwen3_TTS/python/`(vendored,不在本仓)。
```bash
export PYTHONPATH=/path/to/rknn3-model-zoo
```

## 4. 五组件导出
```bash
cd /path/to/rknn3-model-zoo/examples/Qwen3_TTS/python
# 模型经 ModelScope 下载 Qwen3-TTS-12Hz-1.7B-Base(export 脚本 --model_path 默认即 1.7B)

# talker(w4a16,grq 量化校准 ~50min;慢,detached 跑)
cd talker && uv run --project ../../../../../deploy/rk1828/export python export_talker_rknn.py

# code_predictor
cd ../code_predictor && uv run ... python export_code_predictor_rknn.py

# speech_decoder(chunk10/left10 = 低 TTFA 甜点;ONNX 含 masking patch)
cd ../speech_decoder
uv run ... python export_speech_decoder_onnx.py --chunk_size 10 --left_context_size 10
uv run ... python export_speech_decoder_rknn.py

# embeds(talker_text/input/codec + tokenizer)
cd ../embeds && uv run ... python export_embeds.py

# text_projector
cd ../text_projector && uv run ... python export_text_projection_rknn.py
```

## 5. 部署 radxa
产物 `examples/Qwen3_TTS/models/` 经 Mac 中转到 radxa(`fleet pull wsl2→Mac`,再 `fleet push Mac→radxa`;设备间直传会失败),然后 radxa 上:
```bash
cd rknn3-model-zoo && ./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_TTS
```
> **铁律**:speech_decoder 的 `.rknn` 与 `.weight` 必须同一次导出(chunk 不一致 → 解码饱和垃圾)。详见上级 [`../README.md`](../README.md) §6 坑。

## 注意 / 可复现关键坑(实测)
- **transformers 必须 4.57.3,不能 4.51.3**:模型代码 `modeling_qwen3_tts_tokenizer_v2.py` import `transformers.masking_utils`(4.52+ 才有);4.51.3 会 `ModuleNotFoundError: transformers.masking_utils`。4.57.3 一个 env 同时满足 ONNX trace + rknn3-toolkit(rknn 声明 pin 4.51.3 但拷贝装不强制)。
- **librosa / soundfile 必装**:tokenizer 依赖,漏了 ONNX 导出报缺模块。
- **torch 用 CPU wheel**:`[tool.uv.sources]` 配 aliyun CPU flat index(`format="flat"`,aliyun `/cpu/` 是文件目录非 PEP503 registry,必须 flat)。避开 ~5GB CUDA 包(naive pin 会拉 34 个 nvidia 包还卡下载)。CPU torch 跑 ONNX trace 足够。
- **rknn3-toolkit 1.0.4 必须手动拷**(无公开 wheel,见 §2),`uv sync` 装不来。
- **不要导 0.6B**:RKNN3 V1.0.4 量化 0.6B EOS 塌陷,全量化算法失败(详见 `../README.md` §2)。
- wsl2 极不稳:长导出 detached + keepalive,见 `../README.md` §6。
- `uv.lock` 已提交(67 包,CPU torch、0 nvidia),保证可复现;改依赖后重跑 `uv lock`。

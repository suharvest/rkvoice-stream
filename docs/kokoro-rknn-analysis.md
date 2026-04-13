# Kokoro TTS RKNN 可行性分析

> 2026-04-11

## 结论：RKNN 整图转换不可行

Kokoro 82M 的 ONNX graph surgery 成功（5039→1712 节点，所有不兼容算子消除），但 RKNN build 阶段失败。

## 根因：NPU 寄存器维度上限

RK3576 NPU 单层最大时间维度为 **8191**（寄存器位宽 0x1fff），Kokoro 的 ISTFTNet vocoder 内部时间维度达到 **13081**（上采样后更达 65400），超过硬件限制。

```
REGTASK: The bit width of field value exceeds the limit,
         target: f2, limit: 0x1fff, value: 0x3318 (13080)
```

这不是算子支持问题（99.4% 算子 RKNN 支持），而是硬件维度上限。

## 算力分布

| 模块 | GFLOPs | 占比 | 能上 RKNN？ |
|------|--------|------|------------|
| BERT encoder | 2.12 | 23.7% | 能 |
| Vocoder 后半段 (resblocks.3-5 + noise_res.1) | 4.51 | 50.2% | 不能 (dim=13081) |
| Vocoder 前半段 (resblocks.0-2 + noise_res.0) | 1.51 | 16.8% | 能 |
| text_encoder + decoder | ~0.15 | 1.7% | 能 |
| 其他 | ~0.65 | 7.6% | 混合 |

63% 算力在 RKNN 不能跑的 vocoder 后半段，拆模型收益有限。

## 推荐方案

sherpa-onnx CPU 全量推理（82M 模型足够轻量）。

## 验证过程

1. ONNX graph surgery: `fix_kokoro_rknn.py` — 消除 If/Loop/Sequence/Range/Random → 成功
2. onnxsim: 5039→1712 节点 → 成功
3. ORT 验证: 输出正常 → 成功
4. RKNN build: **失败** (REGTASK bit width)
5. 缩小 seq_len=16: 仍然失败（问题在 vocoder 内部维度，不受输入长度影响）

## 文件

- `rk3576/scripts/fix_kokoro_rknn.py` — ONNX graph surgery 脚本（可复用于其他模型）
- WSL2: `/home/harve/kokoro-analysis/kokoro-multi-lang-v1_1/model-rknn-ready.onnx` — 修复后的 ONNX（311MB）

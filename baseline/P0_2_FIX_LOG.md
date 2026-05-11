# P0-2 修复尝试日志 (2026-05-11)

## 改动总结（rkllm_wrapper.py + qwen3_tts.py）

按 codex Q1-Q6 调研结果做了 5 处对齐 ASR 的改动 + 1 处后撤：

| # | 改动 | 文件 | 效果 |
|---|---|---|---|
| 1 | `rkllm_clear_kv_cache` argtypes 改成 4-arg ABI `(handle, keep, start_pos*, end_pos*)` | `runtime/rkllm_wrapper.py:190` | ✅ docker logs 的 `start_pos and end_pos are only valid...` 警告消失 |
| 2 | `param.extend_param.base_domain_id = 1` | `runtime/rkllm_wrapper.py:219` | RKLLM/RKNN 共存（memory project_rkllm_domain_coexist） |
| 3 | `param.extend_param.embed_flash = 1` (原 0) | `runtime/rkllm_wrapper.py:222` | 跟 ASR 对齐 |
| 4 | `inp.role = b""` + `inp.enable_thinking = False` | `runtime/rkllm_wrapper.py:283` | 跟 ASR 对齐 |
| 5 | Prefill `keep_history=0` → `keep_history=1` | `backends/tts/qwen3_tts.py:436, 989` | ✅ **关键：prefill KV cache 现在能保留给 decode 步用** |
| 6 | `rkllm_set_chat_template(handle, b"", b"", b"")` | `runtime/rkllm_wrapper.py` | ⚠️ 实测加上 vs 不加，结果差不多。**已撤销不加**，保留默认 template |

## 测试结果对照（同输入：`请关闭卧室的空调。`）

| 阶段 | TTS 输出 (ASR 识别) | RTF | 备注 |
|---|---|---|---|
| 原始（5 处修复前） | `你是一个智能语言交互你` | 6.42 | 完整 AI 助手训练先验，prefill 完全被忽略 |
| 4 处 API 对齐后（无 keep_history fix） | `你是一个智能语言交互你` | 6.58 | API 对齐了但输出不变，因为 KV cache 还是被清 |
| + keep_history=1 fix | `你\n\n` | 6.50 | **prefill 现在影响 decode 了**，但音频仍大部分不可识别 |
| + 移除 set_chat_template | `你催催\n\n` | 6.50 | 跟上一步差不多 |

## 已修复

- ✅ `start_pos and end_pos` 警告消失（codex Q2，置信度 0.88）
- ✅ Prefill KV cache 正确传递给 decode 步（codex Q5，置信度 0.82）—— 这是真正解锁内容差异的关键
- ✅ TTS 现在能产生**与 prefill 相关**的输出（不再是无条件训练先验）

## 未修复

- ❌ TTS 音频内容仍大部分不可识别（ASR 只能识别 1-2 个汉字）
- ❌ RTF 仍 6.5（性能问题）

## 剩余可能根因（codex 报告里覆盖的）

1. **Q1 prompt 格式**（置信度 0.65）—— 元素级 add codec_prefix + role 是不是 Qwen3-TTS 的真实用法？需要对照 Jetson tensorrt-edge-llm `qwen3OmniTTSRuntime.cpp` 的具体调用
2. **Q4 codec_prefix IDs**（置信度 0.75）—— `CODEC_THINK_ID=2154, CODEC_BOS_ID=2149` 等是硬编码，需要从 model config 读取验证
3. **Q6 w4a16 quant**（置信度 0.85 不是主因，但既然其他都修了还不对，可能就是它）—— talker_fullvocab w4a16 模型本身质量差，需要换 fp16 或不同 quant 试

## 下一步建议

应该派 codex 再深入做：
- 实际拉 wsl-local 上 tensorrt-edge-llm 的 qwen3OmniTTSRuntime.cpp，逐行对比 Python 实现的 codec_prefix / prefill_embed 构造
- 用调试脚本逐 token 验证 RKLLM 收到的 input embeddings 跟期望一致
- 评估是否需要重新转 talker (w8a8 / fp16 / 其他量化)

或者用户根据"qwen3 TTS 一直没真正可用过" 决定是否值得继续。

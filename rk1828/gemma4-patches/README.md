# Gemma-4 RK1828 demo 改动(长音频 / KV cache / 实时流式)

对 Rockchip `rknn3_model_zoo` 的 `examples/gemma4/cpp/` 的改动,给 RK1828(Radxa ROCK 5T)上的 Gemma-4 E2B demo 加了长音频 chunk、KV/prefix cache、实时捕获流水线。**全部 opt-in(env 控制),默认行为不变。**

- `gemma4-rk1828-mods.patch` — 相对 SDK V1.0.4 原版 gemma4/cpp 的统一 diff。
- `full-files/` — 改动后的完整文件(备份;patch 对不同 model-zoo 版本不适用时直接用)。

## 应用
```bash
cd rknn3_model_zoo
git apply /path/to/gemma4-rk1828-mods.patch     # 或手动用 full-files/ 覆盖 examples/gemma4/cpp/
cd .. && ./build-linux.sh -t rk3588 -a aarch64 -d gemma4
```

## 功能(env 开关)
| env | 功能 | 说明 |
|---|---|---|
| (默认) | 原版单轮 MULTIMODAL | 行为与原版逐字节一致 |
| `GEMMA4_KV_DEMO=1` | KV/prefix cache 多轮验证 | keep_history=1 多轮不 clear,实证 follow-up 省 82% prefill |
| `GEMMA4_STREAM_DEMO=1` | 流水线 vs 批处理对比 | off-path 模拟,证编码重叠省 post-speech 延迟 |
| `GEMMA4_RTSTREAM=1` | **真实时捕获 loop** | 增量读+满 7s 即编(藏捕获里)+流结束尾块编+一次 prefill |
| `GEMMA4_RT_PACE=1` | (配 RTSTREAM)真实时速率 | nanosleep 模拟音频按真实时到达 |

## 改动概要
- `rknn_gemma4.cc`(最大):长音频 chunk 化(`gemma4_encode_one_chunk` + 切块循环,7s/块=112000样本@16k)+ 实时捕获 loop(`gemma4_rtstream_loop`,增量读→满块即编→流结束收尾)+ 流水线对比。
- `llm/rknn_gemma4_llm.cc`:KV cache 路径(rknn3_session keep_history + query_state)。
- `rknn_gemma4.h`:导出 `gemma4_audio_num_chunks`。
- `main.cc`:audio_embeds buffer 按 N_chunks 扩容 + `g_first_token_time_us` 计时。
- `CMakeLists.txt`:`-no-pie`(Debian12 gcc 默认 PIE + 预编译 libfftw3f.a 非 PIC,不加链接失败)。

## 部署/运行坑(必读)
1. **build-linux.sh 的 install 会清空 `install/.../rknn_gemma4_demo/model/`** → 模型必须 **build 之后**放(8 文件)。设备上用 **move-aside**(`mv model 别处` → build → `mv` 回;**用 mv 不用 cp**,radxa 磁盘紧 cp 会产生坏文件)。
2. **运行参数顺序**见 `print_usage`(main.cc):`llm.rknn llm.weight 0xff tokenizer embed max_ctx max_new per_layer safetensors audio.rknn audio.weight 0xf "" "" 0 audio.wav "" prompt`。
3. **core_mask**:LLM=`0xff`(8核),**audio=`0xf`(4核,不同!)**;给错报 `core_mask not match npu core number N`。
4. demo 跑需 root(EP node `/dev/pcie-rkep-*` root-only)。
5. **LLM+audio 需 context≤8192** 才装下 RK1828 5GB(16384+audio 会 MODEL_SETUP fail);与 Qwen3-TTS 单卡不共存(纯文本 LLM 与 TTS 可共存)。
6. **反复失败的 model load 会 wedge EP** → host `reboot` 恢复(boot 时 rknn3.service 重刷固件)。

## 实测性能(E2B,radxa RK1828)
- LLM:decode 58–63 tok/s,prefill ~916 tok/s。
- 音频编码:RTF ~0.04(154ms/3.76s,482ms/14s)= 比实时快 24×。
- 长音频:chunk 化完整处理(14s/3块,转写贯穿到结尾)。
- KV cache:多轮 follow-up prefill 省 82%。
- 实时 loop:编码藏进说话期间(RT_PACE 实测 chunk1 在 +7.3s 编),post-stream = 尾块编码 + prefill。
- 流式输出:逐 token(result_callback 每 token emit)。

## 已知边界:增量 prefill 不可行(闭源 runtime 限制)
想把 audio prefill 也藏进捕获(INPUT_EMBED 逐块进 KV)**做不到**:Gemma-4 需 per-token `per_layer_inputs` 流(靠 input_callback 逐位置查表),而 `RKNN3_LLM_INPUT_EMBED` 会禁掉 input_callback → 二者互斥,runtime 明确拒绝(`input_callback does not support input_embeddings and per_layer_inputs when input_type is RKNN3_LLM_INPUT_EMBED`)。所以**实时 TTFT 下限 = 尾块编码 + 完整 audio prefill + 首 token**(短句 ~300ms,14s ~558ms);编码已藏掉(RTSTREAM),prefill 藏不掉。唯一解需 Rockchip 出 EMBED+per_layer_inputs 共存的 runtime 变体。
</content>

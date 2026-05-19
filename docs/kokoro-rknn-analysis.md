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

## 2026-05-18 复测：WSL2 导出环境可用，但全图仍不可导

WSL2 环境：

- 可用 Python：`/home/harve/rknn-build/.venv/bin/python`
- 可导入：`rknn.api`、`onnx`、`onnxruntime`、`onnxsim`
- 原始模型目录：`/home/harve/kokoro-analysis/kokoro-multi-lang-v1_1`

新增导出入口：`models/tts/kokoro/export_kokoro_rknn.py`。

复测结果：

- 复用旧 `model-rknn-ready.onnx` 的 `seq_len=128` fixed ONNX，RKNN build 在 constant folding 阶段失败：
  `Reshape /text_encoder_1/Transpose_output_0_rs input_shape={1,128,1,128}, requested={1,512,1,128}`。
- 从原始 `model.onnx` 重新生成 `seq_len=512` fixed ONNX，ONNX surgery、onnxsim、ORT verification 均通过；输出 shape 为 `(1569600,)`，fixed ONNX 约 410MB。
- `seq_len=512` 继续 RKNN build 仍失败在同一个 text encoder reshape 折叠点：
  `input_shape={1,128,1,512}, requested={1,512,1,512}`。

结论：WSL2 可以承担导出任务，工具链环境已确认；但 Kokoro 全图 RKNN 当前先被 text encoder 的 reshape/constant-fold 问题卡住，即使绕过该点，后续仍有 vocoder 维度超过 NPU 寄存器上限的已知风险。下一步应按 Matcha/Paraformer 的模式做 hybrid split：先拆出能稳定上 NPU 的前段子图，保留 text frontend / 敏感 reshape / 超长 vocoder 段在 CPU 或其他运行时。

## 2026-05-18 hybrid split 真机验证

按 Paraformer block split 的方式进一步拆 Kokoro：

1. CPU prefix：`tokens/style/speed -> /MatMul_1_output_0, /Slice_2_output_0`
2. RKNN decoder-front：`/MatMul_1_output_0, /Slice_2_output_0 -> /decoder/decode.3/Mul_output_0`
3. CPU generator tail：`/decoder/decode.3/Mul_output_0, /Slice_2_output_0 -> audio`

全 decoder suffix 曾能导出 519MB RKNN，但 Radxa RK3588 真机不可用：运行时报
`failed to submit!, op id: 272, op name: Mul:/decoder/generator/noise_res.0/adain1.0/Mul_2`，
输出全零。该 op 输出 shape 为 `[1,256,52320]`，属于 vocoder/generator tail 的超长时间维，和既有 REGTASK 风险一致。

可用 split 是 decoder-front：

- 导出脚本：`models/tts/kokoro/export_kokoro_decoder_front.py`
- WSL2 v2 产物：
  - `kokoro-decoder-front.onnx`：129MB
  - `kokoro-generator-tail-cpu.onnx`：196MB
  - `rk3588/kokoro-decoder-front.rknn`：83MB
- Radxa RK3588 真机验证：
  - `load_rknn=0`，`init_runtime=0`
  - real prefix input 推理耗时约 `0.673s`
  - RKNN output shape `(1,512,5232)`
  - 对 ORT golden：`mae=0.00528346`，`max=0.08997178`，`rel_l2=0.00227114`
- 后端完整 smoke：`kokoro_rknn` hybrid 在 Radxa 上可完成 `synthesize("abc.")`，输出 WAV `3,128,364` bytes。
  固定图输出音频时长 `65.17s`，端到端耗时 `45.31s`，RTF `0.70`。分段耗时：
  CPU prefix `159ms`，RKNN decoder-front `709ms`，CPU generator tail `44.44s`。

因此当前可落地路径是混合后端：CPU/ONNX Runtime 处理 prefix 和 generator tail，RKNN 只承接 decoder-front。它能验证 Rockchip NPU 子图正确性，但 generator tail 仍在 CPU，完整 TTS 延迟会受 CPU tail 限制。

## 2026-05-18 bucket 复测

固定 `seq_len=512` 的主要问题是短文本也会生成完整 `1,569,600` samples，约 `65.17s` 音频。按 bucket 重新导出后，输出长度会跟随 bucket 缩短：

| bucket | audio samples | 音频时长 | Radxa 端到端 | RTF | CPU tail |
|---|---:|---:|---:|---:|---:|
| 512 | 1,569,600 | 65.17s | 45.31s | 0.70 | 44.44s |
| 64 | 286,800 | 10.37s | 8.72s | 0.84 | 8.60s |
| 32 | 126,000 | 5.25s | 3.94s | 0.75 | 3.85s |
| 16 | 65,400 | 2.71s | 2.14s | 0.79 | 2.09s |

测试文本均为 `abc.`，`num_tokens=4`。bucket 能把短句等待时间从 `45s` 降到 `2-4s`，但 RTF 没有稳定改善；主要原因仍是 CPU generator tail 占绝大部分时间。实际上线应按 token 数选择最小可容纳 bucket，而不是固定使用 `seq_len=512`。

## 2026-05-18 tail 内部继续 RKNN 化尝试

在 `seq_len=16` bucket 上继续拆 generator tail：

- `noise_res.0/Add_8_output_0`：ORT 子图可跑，输出 `[1,256,2180]`；RKNN 可 build，但 Radxa `inference()` 返回 `None`。该子图包含 AdaIN/InstanceNormalization/Snake (`Sin/Pow`) 路径，RKNN build 日志有 `Unkown op target`。
- `noise_res.0/adain1.0/Add_2_output_0`：更早切到 AdaIN 后，Radxa 仍返回 `None`。
- `noise_convs.0/Conv_output_0`：RKNN build 失败，toolkit 判定输出被 constant-fold 成常量；这是噪声分支，不是可独立加速的主路径。
- `ups.0/ConvTranspose_output_0`：主路径子图能在 Radxa 上运行，输入 `[1,512,218]`，输出 `[1,256,2180]`，单次约 `45ms`。但它只覆盖 tail 的很小一段，不能解释 `2.09s` 的 CPU tail 瓶颈。

结论：第 3 条“继续把 tail 前半段上 RKNN”在 `seq16` 下只找到一个稳定但收益很小的主路径 ConvTranspose；真正耗时的 AdaIN/Snake/InstanceNorm/resblock 分支仍卡在 RKNN runtime/CPU fallback。要继续推进，需要替换 Snake/InstanceNorm 为 RKNN 友好的近似或自定义 op，再重新验证数值和真机输出。

第 5 条“真 streaming/windowed vocoder”不能简单通过 slicing 当前 tail input 完成。当前 ONNX bucket 的 tail 输入是固定 shape，例如 `seq16` 为 `[1,512,218]`，模型内部还 bake 了对应的 noise、上采样和重建长度。真正 windowed 需要重新导出窗口图，并处理 Conv/ConvTranspose、InstanceNorm/AdaIN、noise source 和重建边界；当前 bucket selector 是可落地的近似方案，不是严格 streaming。

### 继续验证结果：RKNN tail 不是当前收益点，ORT tail 参数可小幅降低 RTF

进一步把 `Sin/Pow/InstanceNormalization` 改写为 RKNN 友好算子后，`noise_res.0/adain1.0/Add_2_output_0` 与 `noise_res.0/Add_8_output_0` 的 style-only 子图能在 Radxa 上运行：

- `adain1.0/Add_2`：约 `12.9ms`，输出 `[1,256,2180]`
- `Add_8`：约 `110ms`，输出 `[1,256,2180]`

但这些子图被 RKNN compiler 裁成 style/noise-only；传入真实 `hidden+style` 会返回 `None`，不能直接并入主流水线。继续按 ORT profiling 定位真实热点后，`seq16/seq32` tail 的主要耗时来自后半段 `ConvTranspose` 和 1D Conv：

- `ups.1/ConvTranspose`
- `resblocks.5/*/Conv`
- `noise_res.1/*/Conv`
- `resblocks.2/4/1/*/Conv`

针对热点做最小子图验证：

- 单个 `resblocks.5/convs1.0/Conv` RKNN 可运行，但 `seq16` 上约 `57.5ms`，比 CPU ORT profile 中同类节点约 `45.9ms` 更慢。
- `ups.1/ConvTranspose` 是最大热点，但 RKNN build 阶段直接退出，无法产出可运行 RKNN。

因此长期方案的当前落点不是“继续切更多小 RKNN 子图”，而是：

1. 使用最小可容纳 bucket，避免短句固定跑 `seq512`。
2. 保留已验证的 `decoder-front` RKNN。
3. CPU tail 使用 ORT 线程参数调优。

Radxa RK3588 上 ORT tail 参数基准：

| bucket | 最优参数 | tail avg | tail RTF |
|---|---|---:|---:|
| 16 | `intra=3, inter=1, graph_opt=all` | `2002ms` | `0.735` |
| 32 | `intra=4, inter=1, graph_opt=all` | `3357ms` | `0.639` |

已在 `kokoro_rknn` backend 加入环境变量：

- `KOKORO_PREFIX_ORT_INTRA_OP`
- `KOKORO_PREFIX_ORT_INTER_OP`
- `KOKORO_TAIL_ORT_INTRA_OP`
- `KOKORO_TAIL_ORT_INTER_OP`
- `KOKORO_ORT_GRAPH_OPT`

端到端单独运行 `seq32 + tail intra=4`，`abc.` 结果：

- 音频时长 `5.248s`
- 端到端 `3.62s`
- RTF `0.69`
- tail `3.52s`

相比早前 `seq32` 约 `3.94s / RTF 0.75` 有小幅改善。并行跑多个 TTS 会抢 CPU，RTF 会明显变差；生产部署应避免多个 Kokoro tail 同时跑在同一组 CPU 核上。

### 2026-05-19 int8 验证

社区 `onnx-community/Kokoro-82M-v1.0-ONNX` 的量化模型不是 RKNN 可直接消费的 int8：

- `model_quantized.onnx` / `model_q8f16.onnx` 为 opset 20，RKNN Toolkit 2.3.2 只支持到 opset 19。
- 图中包含 `DynamicQuantizeLinear`、`MatMulInteger`、`ConvInteger`、`DynamicQuantizeLSTM`、Microsoft domain fused ops，以及动态 `sequence_length/num_samples`。
- ONNX version converter 降到 opset 19 失败在 `ConstantOfShape` adapter。
- Radxa ORT 可运行社区 q8，但随机短输入测试 `0.325s` 音频需 `0.661s`，RTF `2.03`，不适合作为当前 RK3588 路线。

对我们自己的静态 split 子图做 RKNN int8 是可行的：

| 子图 | fp RKNN | int8 RKNN | 结论 |
|---|---:|---:|---|
| `seq32 decoder-front` | `48.3ms` | `33.7ms` | 可用，约 30% 提升 |
| `resblocks.5/convs1.0/Conv` | `57.5ms` | `40.5ms` | isolated 子图可提升 |
| `ups.1/ConvTranspose` direct | `62.1ms` | `46.6ms` | isolated 子图可提升 |

端到端 `seq32 + int8 decoder-front + ORT tail intra=4`，`abc.` 结果：

- 音频时长 `5.248s`
- 端到端 `3.44s`
- RTF `0.655`
- prefix `19ms`
- int8 front `41ms`
- CPU tail `3374ms`

结论：int8 对可上 NPU 的子图有效，但总 RTF 仍由 CPU tail 决定。当前默认配置应使用 `rk3588/kokoro-decoder-front.int8.rknn`；继续把 tail isolated int8 子图并入主流水线需要重切 tail 边界并处理 Snake/AdaIN/InstanceNorm，收益仍需谨慎评估。

### tail RKNN island 真实 split 验证

为了避免只看 isolated 子图误判性能，进一步验证了真实 CPU/RKNN/CPU split：

1. 原始 CPU tail：`hidden/style -> audio`
2. split tail：`CPU pre -> ups.1 ConvTranspose RKNN island -> CPU post`

`seq16` bucket、Radxa RK3588、`runs=3`：

| 方案 | full / split 平均耗时 | island 耗时 | 数值 |
|---|---:|---:|---|
| 原始 CPU tail | `1849.6ms` | - | baseline |
| `ups.1` int8 island | `1885.5ms` | `43.9ms` | `rel_l2=0.636`，不可接受 |
| `ups.1` fp island | `1913.0ms` | `56.0ms` | `rel_l2=0.00099`，数值可接受但更慢 |

结论：即使 isolated `ups.1` int8 子图更快，真实切入 tail 后仍不提升；pre/post ONNX Runtime 分段、数据往返和 session 边界开销会吃掉收益。fp island 数值可接受但更慢，int8 island 又有明显精度风险。因此不建议把 tail hotspot island 工程化接入主链路。当前有确定收益的 NPU 扩大范围仍限于 `decoder-front.int8.rknn`。

### 参考 Jetson TensorRT split-generator 后的 RKNN 复核

参考 `~/project/seeed-local-voice` 的 Kokoro TensorRT 路线：

```text
TRT encoder -> CPU length regulator -> TRT decoder FP16
-> TRT source BF16 -> TRT generator FP16 -> CPU post-spec/ISTFT
```

对应到当前 fixed bucket tail，TensorRT 的关键切分不是 `conv_post`，而是：

- source：`-> /decoder/generator/Concat_1_output_0`
- generator-rest-preexp：
  `/decoder/decode.3/Mul_output_0, /decoder/generator/Concat_1_output_0, /Slice_2_output_0`
  `-> /decoder/generator/Slice_1_output_0, /decoder/generator/Slice_2_output_0`
- post-spec/ISTFT：
  `/decoder/generator/Slice_1_output_0, /decoder/generator/Slice_2_output_0 -> audio`

新增脚本：`models/tts/kokoro/extract_trt_style_generator_split.py`。

`seq16` 结果：

| 方案 | Radxa RK3588 耗时 | 结论 |
|---|---:|---|
| CPU full tail ORT | `1961ms` | baseline |
| CPU generator-rest-preexp ORT | `1819ms` | tail 主要耗时 |
| CPU post-spec/ISTFT ORT | `7ms` | 不是瓶颈 |
| RKNN generator-rest-preexp FP16 | `3464ms` | 比 CPU 慢 |
| RKNN generator-rest-preexp INT8 | `2656ms` | 比 FP16 快，但仍慢于 CPU |

source 子图在 fixed bucket 中已被常量折叠，ORT 约 `0.24ms`，RKNN build 会拒绝“全常量输出”。它适合预计算/缓存，不需要上 NPU。

`generator-rest-preexp` 虽然能在 Radxa 上运行并输出非零结果，但 build 日志仍出现：

```text
REGTASK: limit: 0x1fff, value: 0x3318
Unkown op target: 0
channel is too large, may produce thousands of regtask, fallback to cpu
```

结论：Jetson TRT 的 split-generator 思路在 TensorRT 上合理，但迁移到 RKNN 不产生性能收益。原因是 RKNN 对 `128 x 13081` 长时间维的 generator 主体会产生大量 fallback/regtask，NPU 运行反而慢于 ORT CPU；CPU post-spec/ISTFT 只有毫秒级，不值得作为优化重点。当前不应接入 `generator-rest-preexp.rknn`，继续保持 `decoder-front.int8.rknn + ORT tail`。

### 复用 TRT Pow2->Mul 图手术验证

TRT 侧有 `rewrite_onnx_pow2_to_mul.py`，把 `Pow(x, 2)` 替换为严格等价的 `Mul(x, x)`，用于避免 FP16 Pow 的 NaN/慢路径。RKNN 侧已复用到 `generator-rest-preexp`：

- 新增脚本：`models/tts/kokoro/rewrite_onnx_pow2_to_mul.py`
- 替换数量：`48` 个 `Pow`
- ORT 对比：两个输出 `mae=0`、`max_abs=0`、`rel_l2=0`
- 改写后 op 统计：`Pow=0`，`Mul=192`

Radxa RK3588 `seq16` 真机结果：

| 方案 | 原始 | Pow2->Mul 后 | 结论 |
|---|---:|---:|---|
| RKNN FP16 generator-rest | `3464ms` | `3473ms` | 无提升 |
| RKNN INT8 generator-rest | `2656ms` | `2779ms` | 略慢 |

build 日志中 `REGTASK limit 0x1fff value 0x3318`、`Unkown op target`、`channel is too large ... fallback to cpu` 仍存在。结论：`Pow` 不是当前 generator 慢的主因；主要瓶颈仍是长时间维下的 `Sin`、`InstanceNormalization/AdaIN`、大量大张量 `Add/Mul` 以及 RKNN fallback/regtask 调度。

### 复用 TRT/RKNN Sin 图手术验证

进一步验证两种 `Sin` 改写，输入基于 `Pow2->Mul` 后的 `generator-rest-preexp`：

- `rewrite_onnx_sin_poly.py`：`Sin(x)` 改为 `Clip[-pi, pi] + 7阶多项式`
- `rewrite_onnx_sin_poly_range_reduce.py`：先用 `Floor((x+pi)/(2pi))` 做周期 range-reduction，再做 7阶多项式

ORT 数值对比：

| 方案 | output0 rel_l2 | output1 rel_l2 | 结论 |
|---|---:|---:|---|
| clip sin-poly | `0.0543` | `0.0490` | 误差约 5%，风险较高 |
| range-reduced sin-poly | `0.00138` | `0.00119` | 数值可进入音质评估 |

Radxa RK3588 `seq16` 真机，同一输入、`runs=3`：

| 方案 | FP16 RKNN | INT8 RKNN | 结论 |
|---|---:|---:|---|
| Pow2->Mul baseline | `3446ms` | `2665ms` | 仍慢于 CPU generator-rest |
| clip sin-poly | `2713ms` | `1860ms` | 有性能提升，但数值误差过大 |
| range-reduced sin-poly | `3417ms` | `2404ms` | 数值较好，但性能仍不够 |

结论：`Sin` 确实是 RKNN generator 的一部分慢因，但不是能单独解决的主因。简单 clip 多项式能把 INT8 从 `2665ms` 降到 `1860ms`，但 5% 级输出误差不适合直接接入；range-reduction 把误差压到约 `0.1%`，但额外 `Floor/Add/Mul/Sub` 节点把耗时拉回 `2404ms`，仍慢于 CPU generator-rest `1819ms`。因此当前不建议接入 generator RKNN，也不建议把近似 `Sin` 放进生产链路；继续保留 `decoder-front.int8.rknn + ORT tail`。

### ORT CPU tail 继续优化验证

在 Radxa RK3588 上顺序复测 ORT session 参数，避免并发 benchmark 抢 CPU。`seq16/32/64` 的最佳结果：

| bucket | 原默认附近 | 最佳 session 参数 | 最佳 tail RTF |
|---|---:|---:|---:|
| seq16 | `1827ms` (`intra=4, all`) | `intra=4, all, mem_pattern=false` | `0.670` |
| seq32 | `3183ms` (`intra=4, all`) | `intra=4, all` / `arena=false` 波动 | `0.606` |
| seq64 | `7971ms` (`intra=4, all`) | `intra=4, all, mem_pattern=false` | `0.639` |

已在 `kokoro_rknn` backend 增加 ORT 内存选项环境变量：

- `KOKORO_ORT_ENABLE_CPU_MEM_ARENA`
- `KOKORO_ORT_ENABLE_MEM_PATTERN`
- `KOKORO_ORT_ENABLE_MEM_REUSE`

真实后端 `abc.` 对比：`mem_pattern=false` RTF `0.721`，`mem_pattern=true` RTF `0.734`，收益约 `1.8%`，不是根本解。

继续尝试 ORT CPU int8 tail，不走 RKNN：

| bucket | 模型 | 最快 tail 耗时 | tail RTF | 结论 |
|---|---|---:|---:|---|
| seq16 | 原 FP32 ORT | `1826ms` | `0.670` | baseline |
| seq16 | dynamic int8 | `7842ms` | `2.878` | 明显更慢 |
| seq16 | static QDQ/QOperator int8 | `854-858ms` | `0.313-0.315` | 速度接近 2.1x |
| seq32 | 原 FP32 ORT | `3183ms` | `0.606` | baseline |
| seq32 | dynamic int8 | `14794ms` | `2.818` | 明显更慢 |
| seq32 | static QDQ/QOperator int8 | `1666-1684ms` | `0.317-0.321` | 速度接近 1.9x |

但真实 prefix+front 产生的 seq32 tail 输入重新校准后，static int8 的输出仍不合格：

- `rel_l2` 多数在 `0.92~1.11`
- `cosine` 约 `0.14~0.41`

结论：ORT static int8 证明 ARM CPU 有足够 kernel 性能潜力，但全量 Conv/Gemm/MatMul int8 会破坏 Kokoro generator 数值，不能直接接入。下一步若继续保 Kokoro，应做“选择性量化”：只量化误差低的 Conv/ConvTranspose 分组，逐组测速度和输出误差；全量 int8 不可用。

选择性量化复测：

| 候选 | seq32 tail 耗时 | 误差概况 | 结论 |
|---|---:|---|---|
| `post` QDQ | `3206ms` | sample005 `rel_l2=0.091` | 无速度收益 |
| `resblocks_0_2` QDQ（Conv+Gemm） | `2713ms` | `rel_l2=0.07~0.41` | 有收益但误差不稳 |
| `resblocks_3_5` QDQ（Conv+Gemm） | `2665ms` | `rel_l2=0.11~0.27` | 有收益但误差偏大 |
| `resblocks_0_2+3_5` Conv-only QDQ | `2233ms` | sample005 `rel_l2=0.308` | 速度好，误差仍偏大 |
| 单 `resblock0` Conv-only QDQ | `3124ms` | sample005 `rel_l2=0.014` | 精度好但几乎没收益 |
| 单 `resblock5` Conv-only QDQ | `3107ms` | sample005 `rel_l2=0.199` | 小收益但误差偏大 |
| `ups` QOperator | `3586ms` | `rel_l2=0` | 转换后未提速，排除 |

结论：选择性 ORT int8 的可用窗口很窄。量化越多速度越好，但误差迅速上升；误差低的单 block 速度收益接近噪声。当前还没有找到可直接接入的 int8 tail 子集。

MNN runtime 验证：

- Radxa 安装 `MNN==3.5.0`，ONNX -> MNN 转换成功。
- seq32 tail MNN FP32：`3559ms`
- seq32 tail MNN fp16 权重：`3490ms`
- 同机 ORT FP32 baseline：约 `3150ms`

结论：MNN 在这个 Kokoro tail 上慢于 ORT，排除作为短期加速路线。

### 后续 TODO：降低感知延迟 / streaming

当前先不实施，但保留为后续可快速 follow-up 的方向。

目标区分：

- 降低总 RTF：目前 Kokoro tail 仍受 generator CPU 计算限制，streaming 本身不会让同一段计算更少。
- 降低首包/感知延迟：streaming 或更小 bucket 有价值，能让用户更早听到声音。

短期可做：

1. 句子级 streaming
   - 按标点/短句切分，生成完一句立即播放一句。
   - 工程风险低，适合长回复。
   - 不解决单个短句首包仍需完整 tail 的问题。

2. 更小 bucket
   - 继续导出并验证 `seq8/12/16/24/32`。
   - 对短回复优先选择最小可容纳 bucket。
   - 这是当前最现实的首包延迟优化，优先级高于真 streaming。

长期可做：

1. windowed tail prototype
   - 不能直接把当前 fixed tail input 切片运行。
   - 需要重新定义窗口图，处理 Conv/ConvTranspose padding、InstanceNorm/AdaIN 统计、source/noise 一致性、overlap-add 和边界 click/pop。
   - 先做 seq16 小窗口实验，比较拼接音频和整图音频的误差与听感。

2. 真 streaming vocoder
   - 如果 windowed prototype 误差和听感可接受，再产品化为 chunked generator。
   - 目标是把首包从秒级压到几百毫秒级；总 RTF 仍需配合 CPU/RKNN/NEON 优化。

后续接续建议：

- 先补 `seq8/12/16/24/32` bucket 导出和端到端 benchmark。
- 若短 bucket 仍不满足首包，再启动 windowed tail prototype。
- 不要从“当前 tail 简单切片”开始，那条路已知风险高，容易得到错误音频。

## 文件

- `models/tts/kokoro/fix_kokoro_rknn.py` — ONNX graph surgery 脚本（可复用于其他模型）
- `models/tts/kokoro/export_kokoro_rknn.py` — WSL2 RKNN 导出入口
- `models/tts/kokoro/export_kokoro_decoder_front.py` — hybrid split 导出入口
- `models/tts/kokoro/probe_kokoro_tail_splits.py` — generator tail 内部切点探测
- `models/tts/kokoro/profile_onnx_nodes.py` — ORT node profiling
- `models/tts/kokoro/bench_ort_tail_options.py` — ORT tail 线程参数基准
- `models/tts/kokoro/quantize_rknn_probe.py` — 静态 ONNX 子图 RKNN int8 量化探针
- `models/tts/kokoro/bench_tail_rknn_island.py` / `run_tail_rknn_island_bench.py` — tail RKNN island 真实 split 性能验证
- `models/tts/kokoro/extract_trt_style_generator_split.py` — 参考 Jetson TensorRT 的 source/generator/post-spec split 提取脚本
- `models/tts/kokoro/inspect_onnx_tensors.py` — 按张量/节点名检查内部边界和 shape
- `models/tts/kokoro/bench_ort_subgraphs.py` — 多个 ONNX 子图 ORT 静态 shape 计时
- `models/tts/kokoro/rewrite_onnx_pow2_to_mul.py` — 复用 TRT 经验，将 `Pow(x, 2)` 改写为 `Mul(x, x)`
- `models/tts/kokoro/rewrite_onnx_sin_poly.py` — `Sin` 到 clip 7阶多项式的可控验证脚本
- `models/tts/kokoro/rewrite_onnx_sin_poly_range_reduce.py` — 带周期 range-reduction 的 `Sin` 多项式验证脚本
- `models/tts/kokoro/compare_onnx_outputs.py` — 两个 ONNX 模型同输入输出误差对比
- `models/tts/kokoro/convert_onnx_to_rknn.py` — 静态 ONNX 到 RKNN 的小型转换入口
- `models/tts/kokoro/bench_rknn_multi_input.py` — Radxa 上多输入 RKNN 真机 benchmark
- `models/tts/kokoro/bench_ort_tail_advanced.py` / `bench_ort_tail_selected.py` — ORT CPU tail session 参数复测
- `models/tts/kokoro/quantize_ort_tail.py` — ORT CPU dynamic/static int8 tail 候选生成
- `models/tts/kokoro/dump_kokoro_tail_inputs.py` — 从真实 prefix+front 路径导出 tail 校准输入
- `models/tts/kokoro/compare_ort_pair_random.py` / `compare_ort_pair_dataset.py` — ORT 原图/量化图输出误差对比
- `models/tts/kokoro/list_quantizable_nodes.py` / `quantize_ort_tail_selective.py` — tail 选择性量化分组与候选生成
- `models/tts/kokoro/bench_ort_single.py` — 单 ONNX 模型固定 ORT 参数 benchmark
- `models/tts/kokoro/bench_mnn_tail.py` — MNN tail 候选 benchmark
- WSL2: `/home/harve/kokoro-analysis/kokoro-multi-lang-v1_1/model-rknn-ready.onnx` — 修复后的 ONNX（311MB）

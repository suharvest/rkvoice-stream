# Piper 前半段 NPU 迁移状态

状态：text-only 前端迁移已完成可复现验收；DP128 fused 已作为可选路径完成真实 compile/runtime/full-audio 验收。DP256 fused 已编译并完成隔离 defaults 的同 feed 音频对照，但不推荐启用。cat 原生 RKNN Toolkit2 2.3.2 已在真实 128/256 图上完成 6/6 attention 的 compact rewrite 和编译；formal/prototype RKNN 两档五输出 shape 相同、finite、`maxabs=0`、bit-exact，full-audio smoke 已完成。当前正式 NPU service 使用 `PIPER_ENABLE_FRONTEND_NPU=1`，同 service TTS/ASR roundtrip 已完成；CPU `0` 已完成短时 healthy 与 ORT CPU 验证后恢复 `1`。广泛音质、其他语言和 LLM/V2V 验收仍不在已完成范围。

## 当前边界

输入必须是已经 split 的 `encoder.onnx`，不是完整 Piper `model.onnx`。提取器通过 dependency closure 生成：

- `text_encoder.onnx`：CPU/NPU 前端候选图，保留显式 `x_mask`。
- `remainder.onnx`：CPU remainder，输出 `z` 和 `y_mask`。
- `manifest.json`：bucket、输入 dtype/shape schema、边界输出和 remainder 输出语义。

默认五个边界是 mask、hidden、m_p、logs_p，以及 `/enc_p/encoder/Unsqueeze_output_0` shape/mask 控制量。后者来自 cat 实机诊断，当前只增加约 512 bytes。此前的 23 boundary 结论来自 namespace 和 graph input 声明，不能证明真实 consumer；五边界 dependency closure 消除了无名 attention 张量跨界。旧 `Erf` 手术、随机采样烘焙和未知 `Range` 常量化不属于本路径。

## Runtime 条件

runtime 只有在 `PIPER_ENABLE_FRONTEND_NPU=1` 时启用，并且要求同一语言目录同时具备：

`config.json`（或 Piper config）、`text_encoder.rknn`、`remainder.onnx`、`manifest.json`、`flow_decoder.rknn`。

执行路径是 `text_encoder.rknn → remainder.onnx(z/y_mask) → flow_decoder.rknn → audio`。前端和 decoder 使用独立 RKNN context；加载、probe 或 ORT 初始化失败会释放已建立 context。关闭开关只能在仍保留原 `encoder.onnx + flow_decoder.rknn` 的原模型目录中回滚；实验目录若不含原 encoder，必须切回原 artifact 目录，不能只依赖关闭开关。profile 的 `PIPER_SEQ_LEN=256` 保留原 artifact contract 和 hybrid 回退语义；frontend loader 以 manifest 的 bucket（当前 staged bucket 为 128）覆盖运行时 `seq_len`。所有固定输入 Piper 路径（CPU hybrid、legacy 和 frontend）都按实际 phonemized token 数在逗号、分号、空白或 Unicode 字符边界切段，每个最终段重新实测，单个最小单元仍超过 `seq_len` 时明确报错；Kokoro 保持普通句子切分。token 数不超过 256 时仍可单独验证多窗口 mel 音频。

无标点长句的分段实现以有限字符窗口产生候选，再用真实 phonemizer 校验自然边界或 Unicode 字符边界；不会对整个剩余文本重复 phonemize，也不假设 token 数全局单调。短句仍走一次 phonemize/cache 路径。该规则现在同时覆盖 CPU hybrid、legacy 和 frontend；直接调用单段超限会明确抛出 `ValueError`，不再静默截断。Kokoro 仍走原句子切分。

提取命令（示例路径需替换为当前实验中已知的 split `encoder.onnx` 路径；cat runtime/formal128 和 runtime/formal256 目录已 ready）：

```bash
python models/tts/piper/extract_piper_frontend.py \
  --input /path/to/already-split/encoder.onnx \
  --output-dir /path/to/artifacts-128 \
  --seq-len 128 --check

python models/tts/piper/extract_piper_frontend.py \
  --input /path/to/already-split/encoder.onnx \
  --output-dir /path/to/artifacts-256 \
  --seq-len 256 --check
```

正式 relative-position compact rewrite 命令：

```bash
python models/tts/piper/rewrite_piper_relative_position.py \
  --input /path/to/text_encoder.onnx \
  --output /path/to/text_encoder-compact.onnx \
  --seq-len 128 --layers 6 --compact-relative-value
```

256 bucket 将 `--seq-len` 改为 `256`。cat 使用的 native compile 脚本为 `bench/perf/results/rk-tts-cat-20260921/native/convert_text_encoder.py`；上面的配置是其 manifest 驱动的最小可复制形式。

DP conditioner 只在明确 opt-in 时加入 prefix：

```bash
python models/tts/piper/extract_piper_frontend.py \
  --input /path/to/already-split/encoder.onnx \
  --output-dir /path/to/experimental-artifacts-128 \
  --seq-len 128 --include-dp-conditioner --check
```

该开关默认关闭；它只改变 extractor 的边界输出，不改变原 hybrid 路径。

双语 fallback 也是独立 opt-in：设置 `PIPER_CHINESE_FALLBACK=matcha_rknn` 后，Piper 保留英文路径并预加载 Matcha 中文 fallback；未设置或设置为 `off` 时行为不变。中文 Matcha 的 16 kHz 输出通过 `soxr==1.1.0` HQ 重采样到 Piper 的 22.05 kHz，普通和流式路径都使用同一采样率契约；非法非空配置、fallback preload 失败或 fallback 未 ready 会 fail-closed。依赖只在项目环境中启用：

```bash
uv sync --extra piper-bilingual
```

当前锁文件保持 `numpy<2`。正式 production profile 当前使用 NPU toggle `1`；library fallback 开关仍默认 off。CPU toggle `0` 的验证已完成并已恢复 `1`。

配置切换操作沿用正式 profile 的 `PIPER_ENABLE_FRONTEND_NPU` literal `0/1`：`0` 保持 CPU/hybrid 路径，`1` 启用同语言目录中 manifest 驱动的 frontend。同一 profile 只切换这个开关时不需要重跑 `profile-init`，直接按下列方式重建 `speech`；切换 profile 或语言时才先重跑 `profile-init` 写入 `ovs.env`，再重建 `speech`（若 agent 也读取 `ASR_LANGUAGE/TTS_LANGUAGE`，同步重建该服务）。Piper backend 不支持 hot reload。现有运行环境的等价操作为：

```bash
# 仅切换 profile 或语言时执行 profile-init；同一 profile 只切换 NPU/CPU 开关时跳过它。
docker compose -p conversational_voice_ai \
  --project-directory <device-home>/conversational_voice_ai/cloud_rk3576 \
  -f <device-home>/conversational_voice_ai/cloud_rk3576/docker-compose.rk3576.yml \
  run --rm --no-deps profile-init
# 同一 profile 仅切换 PIPER_ENABLE_FRONTEND_NPU=0/1 时，从这里开始执行。
docker compose -p conversational_voice_ai \
  --project-directory <device-home>/conversational_voice_ai/cloud_rk3576 \
  -f <device-home>/conversational_voice_ai/cloud_rk3576/docker-compose.rk3576.yml \
  up -d --no-deps --force-recreate speech
```

执行前先读取现有 `init`/`speech` 容器的 `com.docker.compose.project` label，并确认两者使用同一个 resolved volume；必须都是 `conversational_voice_ai`，再执行上述命令，避免 profile-init 写入错误的 resolved volume。上面的 `profile-init` 仅用于 profile/语言切换；只切换同一 profile 的 NPU/CPU 开关时跳过该步并执行 `speech` 重建。生产当前 toggle 为 `1`，CPU `0` 验证后已恢复 `1`。

RK3576 语言矩阵的 zh/en 两个 cell 现已标为 `measured`：两者使用同一 `rk3576-piper` 双语 profile，zh 保持 `ASR_LANGUAGE/TTS_LANGUAGE=zh` 并使用 Matcha 中文 fallback，en 保持对应的 `en` overlay 并使用 PiperDP128 English。matrix 不写死 `PIPER_ENABLE_FRONTEND_NPU`；该开关仍由 profile 控制。此前 isolated-service evidence 阶段曾标为 `untested`，现已由同 service TTS/ASR roundtrip 补齐；这不构成新的 LLM/V2V 验收。

当前部署记录：image SHA256 `c2e2a14d1b47c4235b44cc024990c48bfddb9bbff28e3be7a99cdc0d13d21af0`，source SHA256 `735c4d30b79bb0f5f53b40ab7068ae5678dae705d64a2417516a5dc709a9b082`，final toggle-1 validation JSON SHA256 `9a6140e6d9769ed97e7fa732880b8d4e1603a7d778a12c95167c420d31b190da`，Compose project 为 `conversational_voice_ai`，stable profile 为 `<device-home>/conversational_voice_ai/cloud_rk3576/config/profiles/rk3576-piper.json`。实际 service `rk3576-piper` 已标记 `verified=true`，ASR/TTS ready，中英文 HTTP 输出均为 `22050` Hz。原配置卷项目名根因已修复；原 profile-init 错卷曾导致的状态已用旧 image SHA256 前缀 `d167...` 的 rollback 和 readyz 验证恢复。CPU `0` 已短时 healthy 并完成 ORT CPU 验证，随后按同一 profile 只重建 speech 恢复 toggle `1`。manifest 为 bucket 128、六输出、DP conditioner enabled。

## 证据和 gate

已有证据目录：

- [`bench/perf/results/rk-tts-cat-20260921/raw/frontend_structure.json`](../../../bench/perf/results/rk-tts-cat-20260921/raw/frontend_structure.json)
- [`bench/perf/results/rk-tts-cat-20260921/raw/graph_boundaries.json`](../../../bench/perf/results/rk-tts-cat-20260921/raw/graph_boundaries.json)
- [`bench/perf/results/rk-tts-cat-20260921/raw/remainder_encoder_residual_diagnostic.json`](../../../bench/perf/results/rk-tts-cat-20260921/raw/remainder_encoder_residual_diagnostic.json)

ORT zero-noise parity 已完成：128 和 256 bucket 各有 3 组 synthetic 样本，以及 `Hello world`（33 tokens）和 `weather`（79 tokens）真实文本；full graph 与 split graph 的 z/mask shape、frame 数和 finite 检查全部一致，`max_abs=0`。证据为 [`frontend128-v2-parity-real.json`](../../../bench/perf/results/rk-tts-cat-20260921/raw/frontend128-v2-parity-real.json)、[`frontend256-v2-parity-real.json`](../../../bench/perf/results/rk-tts-cat-20260921/raw/frontend256-v2-parity-real.json) 和 [`frontend_v2_parity.py`](../../../bench/perf/results/rk-tts-cat-20260921/frontend_v2_parity.py)，脚本 SHA256 为 `9eea2a4a5a26ba5f7f64369b7c5e4b7d67a9a671dc5566bc624a8db7ff10ad0d`。

static Gather 修复后的 NPU value 证据：compact 图保留实际 B 的全部非零 9 行；128/256 模型约 `15.83/17.36 MB`，prefix p50 为 `25.76/83.04 ms`，mask exact、其他输出约 `relL2 0.003–0.005`。编译配置为非量化浮点模型，未据此强制标注 FP16。formal 工具 SHA256 为 `5f8fd00e9a1f263877546547f76d16b80714165986f60438e71aff1f044af600`。128 artifact `formal-128-compact-v3.rknn` 为 `15827659` bytes、SHA256 `7bba0d92cb9e57aa57a9dda22566947a827ea8c00cddf689f4f92798af9ba813`；256 artifact `formal-256-compact-v3.rknn` 为 `17355787` bytes、SHA256 `0c109ebbb275f46d11b374cdf64423da71d6e31ac45a41f17696e1ae9d4d375b`。

formal 编译配置是 RK3576、`optimization_level=3`、`do_quantization=False`，未显式指定 dtype，因此文档将产物描述为非量化浮点模型，不强制称为 FP16。最小可复制配置沿用 manifest 的 prefix 输入名和 shape：

```python
import json
from rknn.api import RKNN

manifest = json.load(open("manifest.json", encoding="utf-8"))
prefix_inputs = manifest["prefix"]["inputs"]
inputs = [item["name"] for item in prefix_inputs]
input_size_list = [item["shape"] for item in prefix_inputs]

rknn = RKNN(verbose=False)
rknn.config(target_platform="rk3576", optimization_level=3)
rknn.load_onnx(
    model="text_encoder-compact.onnx",
    inputs=inputs,
    input_size_list=input_size_list,
)
rknn.build(do_quantization=False)
rknn.export_rknn("text_encoder.rknn")
```

formal/prototype RKNN 对照证据保留在 cat 的 `artifacts/formal-vs-prototype-128.json` 和 `artifacts/formal-vs-prototype-256.json`：两档五输出 shape 相同、finite、`maxabs=0`、bit-exact。该结论适用于已测 bucket 和输入集合，不外推为任意输入的数学 bit-exact。

英文 canary 当前为 `20/20` 格式检查通过；该结果只覆盖输入/输出格式，不构成音质或端到端质量验收。

full-audio smoke 证据保留在 `artifacts/formal-ab128/frontend-ab.json` 和 `artifacts/formal-ab256/frontend-ab.json`；两档均为 zero-noise、warmup 2、reps 7。128 `Hello` `216.381 → 154.367 ms`、`78/78` frames，`weather` `216.438 → 151.761 ms`、`147/147` frames，`numbers` `218.970 → 207.060 ms`、`171/171` frames；256 `Hello` `284.926 → 220.701 ms`、`78 → 77` frames，`weather` `287.154 → 219.161 ms`、`147 → 146` frames，`numbers` `281.662 → 224.076 ms`、`171 → 170` frames。边界交换诊断显示差帧来自 hidden 边界值：forward-only 路径使用 NPU hidden，输出 `77/146/170`；另一条路径仅将 hidden 换回 CPU 原图值，其余 boundary 仍使用 NPU 值，恢复 `78/147/171`。两条路径的 remainder 都在 CPU；样本范围内 hidden 数值偏差经 CPU DP 放大并触发 `ceil` 翻转，形成帧差。NPU 不直接输出 `logw`。原始诊断为 `rawframe-diagnostic-20260922-boundary-swap-v1.json`，SHA256 `f134cfe93594e1573f1cdae0d84dfba2e28da9bb646460bfa44147b8929c4ef4`。这些数据不概括为所有文本均有 30% 收益。

当前 gate：模型保留在 cat 原机，不转移到 WSL。cat 原生 RKNN Toolkit2 2.3.2 已完成 128/256 转换；native 产物路径和 SHA 可核对 [`native-summary.md`](../../../bench/perf/results/rk-tts-cat-20260921/native/native-summary.md)、[`cat-frontend128-source.sha256`](../../../bench/perf/results/rk-tts-cat-20260921/native/cat-frontend128-source.sha256) 和 [`cat-frontend256-source.sha256`](../../../bench/perf/results/rk-tts-cat-20260921/native/cat-frontend256-source.sha256)。原始转换和 runtime 日志保留在 cat；归档操作曾被 auto-review 拒绝，相关授权仍 pending。

cat 原生 Toolkit2.3.2 已完成真实 128/256 图编译验证。此前 hidden/split 偏差来自 static Gather 处理，修复后已取得上述 value 结果；正式工具相关测试当前为 `25`。当前本地相关集合为 `123 passed`，包含前端、extractor、DP conditioner 和 relative-position rewrite 测试；profile contract 相关测试为 `26 passed`，覆盖 frontend literal `0/1`、缺失、非法值和 whitespace；silence trim 额外覆盖 partial frame 的实际 RMS、尾部封顶、短输入和整除边界。此前 `100 passed`、cat worker 的 `60 passed` 属于旧快照，保留作历史记录。formal/prototype RKNN 五输出对照和 text-only full-audio smoke 已完成；256 差帧已有边界交换诊断，DP256 已完成编译和隔离 defaults 同 feed 对照但不推荐启用。正式 NPU service、CPU `0` 短时验证和 matrix 同 service ASR roundtrip 已完成，当前已恢复 toggle `1`。ready scratch runtime 位于 cat `<device-home>/piper-frontend-migration-20260921`，含 formal128/formal256 的 manifest、remainder、decoder 和 config。统一 CPU/NPU 分段修复 source SHA256 为 `735c4d30b79bb0f5f53b40ab7068ae5678dae705d64a2417516a5dc709a9b082`。

同环境 runtime A/B 已有 finite、非空和帧数证据：128 `Hello` 为 `216.25/217.48 ms (p50/p90) → 151.98/159.83 ms`，`weather` 为 `216.51/218.21 ms → 152.43/175.64 ms`；两组均为 zero `78/78` 帧和 `147/147` 帧。256 短/中输入 baseline `282.6–284.9 ms`，NPU `219.3–221.4 ms`，NPU 各少 1 帧。7 reps 内结果 finite 且非空。128 bucket 的 ASR 中，NPU 与 baseline 的 `Hello`、`weather` 文本一致。

显式压力测试证据保留在 cat 的 `artifacts/ab256-stress/frontend-ab.json`：256 bucket、249 tokens、`length_scale=2.0`、warmup 2、reps 7；原始路径 840 frames，p50/p90 为 `396.289/411.266 ms`，frontend 路径 839 frames，p50/p90 为 `332.779/337.142 ms`，frontend 输出为 `214784` samples（`839×256`）。7/7 结果 finite 且非空，路径内确定性 `maxabs=0`。baseline 和 frontend 两路均使用 legacy 端点 `/asr?language=en`，完整文本均为 `Today the weather is clear and warm, and everyone can enjoy a calm walk through the park beside the river at sunset`。这些是带压力参数的独立结果，不能与默认速度收益混算；文本一致也不等于主观音质完全无损。具体原始日志见 native 目录中的 `cat-runtime-*.json`、`cat-compare-256.json` 和对应 build/compare logs。

无标点长输入 cat runtime smoke 已完成。证据为 `<device-home>/piper-dp-followup-20260922/frame-diagnostic-20260922-runtime-smoke-v2/runtime-smoke.json`，SHA256 `e839d6ec0dfa5a1c900aaae806c99968b466dd74c980e47481c8a7e91f6c7e82`；source+overlay SHA256 为 `19bb6aa32d63b5a094c49fc2acae075f6351b8eae9278bee2e119bedafdda2da`。1721-token 无标点输入在 `noise_scale=0.667`、`noise_w=0.8`、`length_scale=1` 下，ordinary 和 stream 都得到 15 段，token 序列完全相同：`[115,119,127,123,117,123,113,123,113,121,109,127,127,127,83]`；整句音素化为 1721，分段后各段 tokens 合计 1767，最大段长 127，文本重拼 exact，结果 finite 且非空。ordinary 输出 `901632` samples；stream 输出 `900608` samples，末段 `43008` samples，差异为 `1024` samples。两次独立推理包含随机噪声，因此本次只验证分段完整性和数值有效性，不声称逐样本一致；差异原因待进一步核查，不能归因于 normalize（normalize 只改变幅值）。ordinary segmentation 为 `3542.7327 ms`，total 为 `6168.0691 ms`；该成本来自真实长输入预处理，不与短句 benchmark 混用，后续仍可优化。无 ASR 或听感验收，不据此声称全文音质准确。

新增双语 final smoke 使用独立的 326-word 输入，不能与上述旧 1721-token 对照混用。双语 final smoke 证据 SHA256 为 `8ce504b44736c4b5429d4a3e0f80ac36e15077908714f96b9ccdb6c274e06aa1`；long-300 JSON SHA256 为 `2cf9fc3d946bf6d520b906a5299dc3226fbee83b34a9b54e81104dbabfbfcf94`，输入样本 SHA256 为 `596099d2a904c95ed881ed8b5ffec4226d331bf591ee5a3eed5de54f485b78a1`。采样率为 `22050`；ordinary 记录 `2018304` frames、`7977.3 ms`，stream 记录 `4077568` PCM bytes、TTFA `553.4 ms`、total `7860.6 ms`。本轮尚无广泛内容质量验收，结果只作为运行完整性和计时证据；正式 NPU service 已上线，CPU `0` 已短时验证后恢复 `1`。ASR 实机调用按约 5 秒自动切为 2 片、10.4 秒自动切为 3 片；该行为不归因于 ASR 的 64-token 上限。

统一分段修复后的本地计时记录为：digits candidate `4.712 s / 461.8 ms`，CPU `5.060 s / 630.3 ms`；long 输入 `10.286 s / 1055.8 ms` 对比 `10.170 s / 1322.9 ms`。long 结果是单次运行，不是 benchmark。此前 CPU 基线含静默超限截断，不能继续作为质量对照；CPU `0` 已完成短时 healthy 与 ORT CPU 验证，不能据此外推广泛质量。

最新实机 HTTP 性能记录为 `perf3-fix735`，raw JSON SHA256 为 `6f3ddf9cad8feed3108ab7b293df318ef4c8a3fb476592f2c38170e595ba283e`；warmup 2、reps 7。以下均为 p50/p90，顺序为 NPU frontend / CPU 对照：`Hello` `177.3/189.3` / `247.0/265.4 ms`，`weather` `181.6/197.7` / `253.4/273.0 ms`，`numbers` `185.1/192.3` / `252.4/269.5 ms`。这些是三条已测英文样本的 HTTP 计时，不外推为任意输入收益。

中文对照中，数字句耗时 `5.696 s`，两路 ASR 文本一致；混合句耗时 `5.984 s`，存在文本规范化差异。中文证据 SHA256 为 `cb9535d65dd96ac6ea574680d925bb853b79c655c9a33d2d19ba69944deca1c1`。长句尾部仍在定位，因此当前不把中文或长句记录作为内容质量通过证据。

后续 candidate 尾部检查已识别完整尾词；对应证据 SHA256 为 `a5f1a3c3767f5fd74576e8a281992e8c32bb3f18cd6ddb96f5e0be6406765827` 和 `d81b9f336f79876f45e80de13e083fb68b8091d5097c2c1ca076263c685f8e99`。该检查的四段 token 序列为 `[115,119,127,83]`，文本重拼 exact。它只说明分段尾部和文本完整性，不能将 CPU 对照差异归因于 TTS 质量或单一运行时组件。

诊断纠正：已撤销的最小 MatMul 默认 layout 判断曾被误报为硬件缺陷；该判断已从结论中移除，不再传播为硬件问题。

WSL CPU-only torch 状态仅作为此前环境记录保留，不是当前 compile gate：torch 2.4.0+cpu import 成功，wheel 大小 `194980071` bytes，SHA256 为 `78dbf5f2789933a7ea2dabeead4daa44679b1e0d8eb35ddb7071c8ab7b181eb3`；跨设备模型传输不执行，除非获得具体授权。

DP conditioning 独立实验：cat 中 conditioner CPU 约 `6.26 ms`、NPU 约 `2.17 ms`，独立算子相对 `relL2=0.003979`；这不是端到端已兑现的 4 ms 收益。按正确的 mask、`scales=[0,1,0]` 和 `input_lengths`，33/79/120 tokens 的原图与 candidate 六输出及 full `z/y_mask` 均 exact `0`；此前基于 `155/162` 差异推断缺少 `input_lengths`，已被正确 feed 对照否定。独立 conditioner 已完成编译。[`extract_piper_dp_conditioner.py`](../models/tts/piper/extract_piper_dp_conditioner.py) 和 frontend 的 `--include-dp-conditioner` 已有本地 `13` 个测试通过。

DP128 fused 可选路径已完成真实 compile/runtime/full-audio 验收。artifact 位于 cat `<device-home>/piper-frontend-migration-20260921/dp128-fused-scratch/text_encoder.rknn`，大小 `16666236` bytes，SHA256 `9d5d605eadbc77c82fcc457adf40ff181011d07dc8da192826c684f96f930085`；当前性能证据固定为 `dp-paired-ab.json`，SHA256 `a555ac4efcbeb4501439bdb1dce9e3f8d3c1a5dadcb36ce35854a685e980e433`，mtime `2026-09-22 01:58:54.958491958+0800`，大小 `9454` bytes。六个输出 finite 且 shape 一致，mask/control exact；第六个 conditioner 输出 NPU 对 ORT 为 `relL2=0.0049355258`、`maxabs=0.0097341985`。该 raw 记录 7 次测量，脚本强制至少 2 次预热，但文件未保留实际 warmup 字段。当前 text-only → fused 结果为：`Hello` `154.709/165.633 → 148.183/149.194 ms (p50/p90)`，`weather` `153.819/164.643 → 148.682/150.452 ms`；同样本、同帧音频逐样本 `maxabs=0`、`rms=0`。当前 run 的 p50 约下降 `3.3–4.2%`，p90 也下降；样本只有两条，不外推为泛化收益。旧快照性能数字已撤销，不与当前 raw 混用。建议需要尽量使用 NPU 的场景采用可选 DP128 融合；text-only128/256 为已测基线。随机 DP 流仍在 CPU。runtime 已确认按六输出 manifest 加载该模型。运行时确认实际 decoder 窗口为 `512` mel 帧；`256` 帧预探测失败属于窗口探测过程。

DP256 fused 已编译，并在 `<device-home>/piper-dp-followup-20260922/dp256/textonly-fused-iso-defaults.json` 完成隔离 defaults 的同 feed 对照；raw SHA256 为 `d2e16f5d03e75fcf3eb5a97e545a2fc6d651a028580390d99e53bd2a348bf1f9`。`Hello`（33 tokens）text-only/fused 为 `77→78` frames，`weather`（79）为 `146→145`，`numbers`（87）为 `170→170`。三句前五个 prefix outputs（包括 hidden）均 bit-exact，因此新增差异归因于 DP conditioner；共享 encoder 误差仍存在，不能表述为 hidden 没有误差。CPU 原图 `78/147/171` 仅作为既有诊断参照，未在这次同 feed raw 中重新运行。DP256 不推荐启用；此前 synthetic conditioner 证据中 valid `129` 的 `z relL2=0.0625` 与真实音频对照分开记录。DP256 的 compile 成功不改变默认关闭状态。

DP256 stage probe 的边界结论是：三条真实文本的 prefix 前五输出（含 hidden）保持 bit-exact，新增差异只出现在第六个 conditioner 输出；hidden swap 诊断中的 `77/146/170` 与 `78/147/171` 帧变化仍由 CPU DP 放大后的 `ceil` 翻转解释。该 probe 只覆盖 `33/79/87` tokens，不能替代更广泛质量验收。

## 未完成项

- DP256 融合进入 review：已完成编译和同 feed 对照，但当前差异与 synthetic conditioner 误差仍不支持推荐启用。
- 更广泛音质和其他语言验证。
- production profile 的更广泛内容质量验收；当前 same-service roundtrip 仅覆盖记录的中英文样本。
- 1721-token 长句的旧 runtime smoke 只作为分段完整性证据；统一 CPU/NPU 分段修复后的广泛内容质量验收仍待完成。

单个最小文本单元超过 bucket 时明确抛出 `ValueError`；无标点长句兜底已通过本地测试和 cat 1721-token runtime smoke 的完整性验证。正式 service 当前开关为 `1`，library fallback 默认关闭，原 hybrid 执行路径保留；同一 profile 的 `0` 切换已完成短时健康和 ORT CPU 验证。已测得的收益只针对文档列出的样本和 bucket，不外推到任意输入或广泛内容质量。

## 当前改动

本文件记录前端 extractor、relative-position rewrite、runtime hybrid 接入、DP conditioner opt-in 和双语 fallback 的当前状态；对应实现文件分别位于 `models/tts/piper/` 与 `rkvoice_stream/backends/tts/piper.py`。查看状态：

```bash
git -C third_party/rkvoice-stream status --short
```

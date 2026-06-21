# RK1828 一等公民集成设计(Spec)

把 RK1828(RKNN3,PCIe 协处理器)从"项目支持 + standalone demo"提升为 **`rkvoice_stream` 包的一等平台**:`create_tts(backend="qwen3_tts_rk1828")` / `create_audio_llm(backend="gemma4_rk1828")` 像 RK3576/88 一样开箱可用,严格对齐包现有契约。

- **Phase 1**:Qwen3-TTS → `TTSBackend`(干净匹配现有 ABC)
- **Phase 2**:Gemma-4 多模态(audio→文本) → 新 `AudioLLMEngine` 抽象

---

## 0. 现状与契约(实施基准,file:line)

包是 RK3576/88(RKNN2/RKLLM)的流式语音服务,三层 app→engine→backends + platform + runtime。关键契约:

- `rkvoice_stream/engine/tts.py:15` `TTSBackend(ABC)`:`name`/`is_ready`/`preload`/`synthesize`/`synthesize_stream`/`get_sample_rate`/`runtime_info`
- `rkvoice_stream/engine/tts.py:80` `create_backend(name)` 显式 if/elif 分发;`:114` `create_tts = create_backend`
- `rkvoice_stream/backends/tts/qwen3_rknn.py:19` 样板:**薄壳 backend 委托给一个 Service**(`Qwen3RKNNBackend` → `qwen3_tts.TTSService`),backend 只管 ABC + NPU lock,Service 管模型
- `rkvoice_stream/engine/asr.py:33` `ASRStream` / `ASRBackend` + `TranscriptionResult`(Phase 2 新引擎参照其风格)
- `rkvoice_stream/platform/base.py:9` `PlatformProfile`(frozen dataclass:`name`/`npu_cores`/`npu_memory_limit_mb`/cpu cores+masks/`default_rkllm_domain`);`platform/rk3576.py`、`rk3588.py` 是实例
- `rkvoice_stream/runtime/rkllm_wrapper.py` = RKNN2/RKLLM 底层(Phase 1 的 `rknn3_worker.py` 与之平级)
- `rkvoice_stream/app/capability.py` 冲突检测(按 platform NPU 资源)

RK1828 现有交付物(**不在包内**,在 `models/rk1828/` + radxa 设备上):
- **Qwen3-TTS C++ demo,已有 server 模式**:`rknn_qwen3_tts_demo ./model girl_base -`(argv[3]=='-')→ Init 一次 → 循环读 stdin 文本行 → stdout 协议 `[uint32 LE 字节长][int16 PCM]`,每句以 `[uint32 0xFFFFFFFF]` 结束;stderr=诊断。详见 `docs/rk1828-qwen3-tts.md` §5。
- **Gemma-4 C++ demo**:目前**一次性**(非 server 模式),patch 在 `models/rk1828/gemma4-patches/`。流式输出确认逐 token(result_callback)。

---

## 1. 桥接决策:subprocess worker(已定;codex-hardened)

Python backend → 子进程 worker → C++ demo(server 模式),复用 stdin/stdout 流式协议。

- 不在 Python 重写 RKNN3 pipeline;不依赖 HTTP(tts_server.py 那层不需要)。
- **注意契约差异(codex)**:现有 `qwen3_rknn.py:32` 的 `preload()` 是**进程内** `TTSService.load()`、就绪标志是内存 `_ready`(`qwen3_tts.py:99,112`);subprocess IPC 打破这个。**不声称"同构"** —— 而是沿用"backend 薄壳 + Service"的**形状**,但 Service 是 IPC-backed,且 Service 负责把 C++ 的 **int16 PCM 帧转成 `TTSBackend` 流式契约(float32 ndarray @24kHz)**(见 §3.1 + §10 流式契约)。
- 与主项目验证过的 TTS worker CLI-shim 模式一致。

---

## 2. 共享:`runtime/rknn3_worker.py`(新)

通用 RK1828 C++ worker 管理器,Phase 1/2 复用。

职责:
- **spawn**:`Popen([binary, *args], stdin=PIPE, stdout=PIPE, stderr=PIPE)`,server 模式。`device_id` 经 args 传(C++ demo 接 `--device-id 0001:11:00.0`;TTS 现以 argv 位置参数,需统一成带 device-id 的形参——见 §6 C++ 改动)。
- **lifecycle**:`start()`(spawn + 等 ready 标记)、`is_alive()`、`stop()`(EOF stdin + 等退出 + kill 兜底)、崩溃检测(stdout EOF / 进程退出 → `is_ready()=False`)。
- **串行锁**:RK1828 单设备串行,worker 内 `threading.Lock`(独立于 host 的 `get_npu_lock`)。
- **协议解析**:
  - 二进制流式(TTS):读 `[uint32 LE len]`,len==0xFFFFFFFF 为句末;否则读 len 字节 = int16 PCM chunk,yield。
  - 文本流式(AudioLLM):每 token/段一行 或 `[uint32 len][utf8]`,句末/EOS 标记。**统一协议**:见 §5。
- **stderr 抽水**:后台线程读 stderr → logging(诊断不堵塞 stdout)。

设计点:worker 进程模型 = 一个 backend 实例持有一个 worker;并发请求经锁串行(RK1828 单卡)。

---

## 3. Phase 1:Qwen3-TTS → `TTSBackend`

### 3.1 `backends/tts/qwen3_tts_rk1828.py`(新)
`class Qwen3TTSRK1828Backend(TTSBackend)`,薄壳委托 `Qwen3TTSRK1828Service`(同 `qwen3_rknn.py:19` 模式)。ABC 映射:
- `preload()`:从 config 取 `binary_path`/`model_dir`/`device_id`/`ref_speaker` → `rknn3_worker.start()`(C++ Init decoder+talker 一次)
- `synthesize(text)`:写一行文本 → 读 PCM 直到 `0xFFFFFFFF` → 拼成 wav bytes 返回
- `synthesize_stream(text)`:写一行 → 逐 `[len][PCM]` chunk `yield (pcm_bytes, meta)`(对齐 `engine/tts.py:47` 的流式签名)
- `get_sample_rate()`:24000
- `is_ready()`:worker.is_alive()
- `name()`:`"qwen3_tts_rk1828"`;`runtime_info()`:device_id/binary/model 版本

### 3.2 `platform/rk1828.py`(新)+ `platform/base.py` 扩展 + 注册表
- `base.py:9` `PlatformProfile` **加 defaulted 可选字段**(不破坏 `rk3576.py:5` 等现有构造):`device_id: str | None = None`(RK1828=PCIe EP `0001:11:00.0`),`is_coprocessor: bool = False`。
- `rk1828.py`:`PlatformProfile(name="rk1828", npu_cores=<EP核>, npu_memory_limit_mb=5120, is_coprocessor=True, device_id="0001:11:00.0", cpu_*=host RK3588 拓扑)`。
- **codex#2:CPU 字段语义错位**——`base.py:12` 注释说 CPU 字段用于 NPU core mask + CPU 亲和;RK1828 填的是宿主拓扑。**规则:消费 CPU mask/亲和的逻辑遇 `is_coprocessor=True` 必须跳过这些字段**(RK1828 不做宿主 CPU 亲和;NPU core mask 走 device_id)。实施时审 capability/affinity 所有 CPU 字段消费点加此守卫。
- **codex#6b 注册表(day-1 阻塞)**:`platform/__init__.py:7` 现只注册 `rk3576`/`rk3588`,`__init__.py:11` 的 config 加载只解析 `asr`/`tts` 节。**必须**:① `platform/__init__.py` 注册 `rk1828`;② config 加载认 `platform: rk1828` 并把它喂给 backend 选择/profile 查找。否则 factory 选不中。

### 3.3 `configs/rk1828-qwen3-tts.yaml`(新)
`platform: rk1828`,`tts.backend: qwen3_tts_rk1828`,`tts.binary_path`、`tts.model_dir`、`tts.device_id`、`tts.ref_speaker: girl_base`。

### 3.4 factory
`engine/tts.py:80` 加 `elif backend_name=="qwen3_tts_rk1828": from ...qwen3_tts_rk1828 import Qwen3TTSRK1828Backend; return ...()`。

### 3.5 capability.py —— **移出 Phase 1(codex#3,#7)**
原想标注 RK1828 与 host 可并存。但 `app/capability.py:55,57` 现按**单 platform** 合并 ASR+TTS 内存限额,`:64,72` NPU core/domain 命名空间也是统一的 —— **没有"设备 ID"维度,代码里无法干净表达跨设备并存**;且 capability 目前根本没在 server 启动里被触发(`capability.py:6`)。
**决定**:Phase 1 **不碰 capability**。跨设备资源模型(给 capability 加 device_id 维度,使 RK1828 的 5GB / core 与宿主 NPU 分账)**推到 Phase 2** 单独 scope + 实现。Phase 1 的 RK1828 TTS 不经 capability 门控(单设备串行锁已足够)。

### 3.6 验收
radxa(RK3588 host + RK1828 EP)真机:`create_tts(backend="qwen3_tts_rk1828").synthesize("...")` → 能量/ASR 校验出真语音(非静音字节);TTFA/RTF 对齐 standalone demo;多请求串行无崩。

---

## 4. Phase 2:Gemma-4 多模态 → `AudioLLMEngine`

Gemma-4 是 audio(+可选文本 prompt)→ 文本 的多模态 LLM(在 V2V 里塌缩 ASR+LLM 两步)。不匹配 ASR(纯转写)/TTS,需新引擎抽象。

### 4.1 `engine/audio_llm.py`(新 ABC)
```
class AudioLLMBackend(ABC):
    def name() -> str
    def is_ready() -> bool
    def preload() -> None
    def generate(audio: np.ndarray, sample_rate: int, prompt: str|None=None,
                 max_new_tokens: int=...) -> str           # 一次性
    def generate_stream(audio, sample_rate, prompt=None, ...) -> Iterator[str]  # 流式 token
    def get_capabilities() -> set                          # {audio, vision?, multiturn?}
def create_audio_llm(name=None) -> AudioLLMBackend         # 同 tts factory 模式
```
风格对齐 `engine/asr.py`(audio 输入)+ `engine/tts.py`(流式 yield)。

### 4.2 `backends/audio_llm/gemma4_rk1828.py`(新)
薄壳委托 `Gemma4RK1828Service`(用 `rknn3_worker`)。`generate_stream`:写 `{audio_ref, prompt}` → 逐 token 读 → yield。复用 Phase 1 worker。

### 4.3 C++:给 gemma4 demo 加 server 模式(扩展 `models/rk1828/gemma4-patches/`)
现 gemma4 demo 一次性。需加 server 模式(同 TTS):Init(LLM+audio encoder)一次 → 循环读 `[audio 路径或 PCM][prompt 文本]` → 流式 emit token(文本行协议)+ EOS 标记。复用已实现的 chunk 化(长音频)+ RTSTREAM。**这是 Phase 2 的主要 C++ 工作**,产物 patch 进 gemma4-patches。

### 4.4 factory + config + app(codex#4 具体改动面)
- `create_audio_llm` factory(if/elif)。
- `configs/rk1828-gemma4.yaml`。
- **server.py 改动(codex 定位)**:现在启动时只加载 `_backend`(TTS)和 `_asr_backend`(`server.py:37,98,117`),`/dialogue` 只接受文本(`server.py:526,531`)。需:① 加**第三个全局后端槽** `_audio_llm_backend`;② 新增端点 `/audio_dialogue`(收 audio → AudioLLM 流式文本 → TTS);③ 编排器"理解阶段"可插拔:(ASR→LLM) 或 (AudioLLM)。RK1828 AudioLLM + host/RK1828 TTS 串成 V2V。

### 4.5 验收
radxa:`create_audio_llm("gemma4_rk1828").generate_stream(audio, prompt="转写")` → 流式文本;长音频 chunk 化贯穿;V2V(gemma4→TTS)端到端出语音。

---

## 5. worker IPC 协议(codex#5:Phase 1 不动现有协议)

- **Phase 1(TTS)**:**完全沿用现有 server 模式协议,不改** —— stdin 纯文本行 in,stdout `[uint32 LE len][int16 PCM]`、`0xFFFFFFFF` 句末。不引入 JSON、不碰 standalone tts_server.py。Python worker 只是这个协议的客户端。
- **Phase 2(AudioLLM)**:gemma4 需传 audio+prompt(纯文本行不够)→ 用**独立的请求格式**(JSON 行 或 length-prefixed),且**显式协议版本协商**(worker 启动握手报版本)。text token 响应:`[uint32 LE len][utf8]`,`0xFFFFFFFE`=EOS。
- **不在同一协议里宣称"双向兼容"**:Phase 1 协议 = TTS 现状原样;Phase 2 协议 = gemma4 专用、版本隔离。两者是不同 worker 二进制的不同协议,不强行统一。

---

## 6. 构建/部署边界

- C++ 二进制(TTS server 模式已有;gemma4 server 模式 Phase 2 新增)由 `models/rk1828/` 的 recipe/patch 在**设备上构建**;包**不含二进制**,经 config `binary_path` 定位 + 启动 worker。
- device-id 统一成 `--device-id` 形参(TTS 现为 argv 位置参数,Phase 1 顺带规整)。
- 包不 vendor RKNN3 SDK/model-zoo(引用式,延续现有原则)。

---

## 7. 风险 / 待定

1. **worker 启动延迟 / 首请求阻塞(codex#6a)**:server 启动时**同步**调 `_backend.preload()`(`server.py:102,104`)→ worker spawn + C++ Init(decoder+talker)发生在启动期,启动变慢但首请求不额外等。**解方**:Phase 1 接受"启动期同步 spawn worker"(对齐现有同步 preload 语义);`is_ready()` 反映 worker 存活 + Init 完成握手;若要异步,单独加 lazy-preload flag(非 Phase 1 必须)。
2. **崩溃恢复**:worker 挂 → backend 重 spawn(带退避);请求期间挂 → 抛错(不静默)。
3. **gemma4 server 模式的 C++ 工作量**(Phase 2 主成本):需把一次性 demo 改 Init-once + 请求循环 + 文本流协议;复用现有 chunk/RTSTREAM。
4. **AudioLLM 引擎抽象**是否过度设计?备选:Phase 2 先把 gemma4 当"带 prompt 的 ASR"塞进 ASRBackend(loose fit,快但语义不纯)。**本 spec 选新引擎**(语义干净、为 V2V 留口)——请 codex 评估这个取舍。
5. **app 层 V2V 编排**改动范围(理解阶段可插拔)需评估。

## 8. 非目标
- 不在 Python 重写 RKNN3 推理。
- 不把 RK1828 塞进 RK3576/88 的 RKNN2 runtime。
- 不动 RK3576/88 现有路径。

## 9. 分期落地(codex#7 收紧)
- **Phase 1(可独立交付,不依赖 Phase 2)**:§2 worker + §3 TTS backend + platform 注册(`platform/__init__.py` + config 认 rk1828)+ factory + §10 流式契约转换 + §11 直连/HTTP smoke。**不含 capability 跨设备语义**(那不在 Phase 1)。`/tts` 端点只依赖 TTS(`server.py:196`),技术上独立。RK1828 即成包的一等 TTS 平台。
- **Phase 2**:§4 AudioLLM 引擎 + gemma4 server 模式 C++ + app V2V 编排(server.py 第三槽 + /audio_dialogue)+ **capability 加 device_id 维度**(§3.5 推迟的跨设备资源模型)+ 真机。

## 10. 流式契约转换(codex Top#1,Phase 1 必解)
C++ worker 输出 `[uint32 len][int16 PCM @24kHz]`;`TTSBackend.synthesize_stream`(`engine/tts.py:47`)的现有 yield 契约需核对 `qwen3_rknn.py:93`→`qwen3_tts.py` Service 实际 yield 类型(float32 ndarray 还是 PCM bytes + meta)。**`Qwen3TTSRK1828Service` 负责适配**:
- int16 PCM bytes → 与现有 Service 一致的 yield 类型(若是 float32 ndarray:`int16 → /32768.0 → float32`);保持 `(chunk, meta)` 形状一致。
- **采样率**:C++ 出 24kHz;若下游/现有 backend 统一某速率则在 Service 重采样,否则 `get_sample_rate()=24000` 透传。
- 实施第一步**先读 `qwen3_tts.py` 的 `synthesize_stream` 确认确切 yield 类型**,再写转换,保证与 `matcha`/`qwen3_rknn` 对 app 暴露的契约逐位一致。

## 12. Phase 2d:真机 V2V 端到端(实测落地)

audio → gemma4 AudioLLM(RK1828,理解+回应)→ 文本 → Qwen3-TTS → 真语音。radxa(RK3588 host + RK1828 EP `0001:11:00.0`)实测跑通。

### 12.1 单 EP 摆放结论(实测)
V2V 需 AudioLLM + streaming TTS 同时可用,但二者在 RK1828 上的内存互斥(gemma4 ~4200MB + Qwen3-TTS ~1700MB > 5120MB EP 上限,见 §rk1828-gemma4 部署坑 5)。两条路径都调查过:

- **路径 A(host RK3588 NPU TTS,真并发)— 实测不现成**:radxa host *有* RKNN2 runtime(`/usr/lib/librknnrt.so → 2.3.0`,实跑 2.3.2)+ `rknnlite` Python + host TTS 模型(`/home/radxa/models/tts/`:`matcha-icefall-zh-en`、`matcha-s64.rknn`、`vocos-16khz-600.rknn`、多个 kokoro bucket)。**但**:① host 上的 `matcha-s64.rknn` 与包 `backends/tts/matcha.py` 的 `run_matcha` 输入装配漂移(`input[3] need 3dims, got 1dims`,模型导出版本不匹配);② matcha/kokoro 的干净 fallback(ORT 声学模型 `model-steps-3.onnx`)需 `onnxruntime`,而生产 host 是 PEP-668 externally-managed + 磁盘 96%,装包属"大动干戈"。结论:Path A 原理可行(runtime+模型齐),但现成资产已漂移,**本次不强行修 RKNN 模型 I/O / 不污染生产 host python**。
- **路径 B(单 EP 时分)— 实测采用**:顺序 gemma4(RK1828)→ 文本 → 释放 gemma4 worker → Qwen3-TTS(RK1828)。用全部已验证资产(gemma4 server-mode binary + 生产 tts_server)。代价是每次切模型 ~13s load,但端到端演示 V2V 能力可靠。

### 12.2 实测链路与时序
- 输入音频:用生产 `tts_server :8900` 的 `POST /tts`(int16 PCM @24k)合成 "今天天气怎么样"(1.98s,rms 0.050),resample 到 16k mono WAV 喂 gemma4。
- **gemma4(RK1828)**:`create_audio_llm("gemma4_rk1828").generate(audio, prompt="...")`。preload **13.6s**(worker spawn + C++ Init),generate **0.56s**;输出文本语义正确:
  - 输入 "今天天气怎么样" → gemma4 "请提供您所在的城市或地区，我才能告诉您今天的天气。"(理解了天气提问并要求补地点 — 贴原文语义)。
- EP 在 gemma4 退出后干净释放(`rknn3_transfer_proxy devices` 仍报 `0001:11:00.0 ... PCIE`,无 wedge)。
- **Qwen3-TTS**:两种驱动方式实测:
  - 经生产 `tts_server :8900` 的 `POST /tts`(**推荐,已验证路径**):产出真语音(rms 0.059,有能量),tts_server 自身保持 health。
  - 经包 `create_tts("qwen3_tts_rk1828")` 直驱 server-mode binary:worker Init 成功但 `rknn3_worker` 的 length-prefix 解析与该 binary 的 stdout 帧不同步(`wanted 1852143441 bytes` = ASCII 误读),把诊断行当成二进制长度前缀 →`WorkerCrashError`。**这是包 worker 协议解析 vs 该 binary 实际 stdout 帧的对齐 bug,非模型问题**(binary 确实吐了 ~4MB PCM)。生产路径走 tts_server HTTP 不受影响;若要包直驱需对齐 `rknn3_worker` 的 §5 Phase 1 帧解析与 binary 实际输出。

### 12.3 已知问题:Qwen3-TTS 长句 runaway
对较长回应文本(如上面 25 字的 gemma4 回复),Qwen3-TTS demo 出现 **runaway/garbled 合成**:输出 64.5s 持续能量但 ASR 回读为乱码 + "我們在這裡" 重复(faster-whisper 回读)。同一 server 对短句("今天天气怎么样")合成干净。这是 Qwen3-TTS talker 的 max_frames/EOS runaway(与主项目记忆里的 CJK 长度失控同源)。**V2V 演示用短回应 prompt 规避**(`prompt="用一句简短的话回答"`);根治需 talker EOS/分句层(超出 Phase 2d scope)。

### 12.4 编排实现
- 路径 A(若 host TTS 现成)经 `/audio_dialogue` WS:起 server,`AUDIO_LLM_BACKEND=gemma4_rk1828` + TTS backend = host(matcha_rknn/kokoro_rknn),真并发。
- 路径 B 经脚本时分:stop tts_server(释放 EP)→ `create_audio_llm("gemma4_rk1828")` 出文本 → 释放 gemma4 → 重启 tts_server / `create_tts("qwen3_tts_rk1828")` 合成。**铁律:生产 tts_server :8900 测试动过必恢复 health**(脚本用 EXIT trap 保证 restore + EP 不 wedge)。

## 11. 测试策略(codex#6c)
- **mock worker**:提供一个 Python 假 worker 脚本(说 §5 Phase 1 协议:读文本行 → 吐固定 `[len][int16 PCM]` + `0xFFFFFFFF`),`rknn3_worker` 可经 config `binary_path` 指向它 → **direct + HTTP dual-mode 测试无需真设备**(对齐 CLAUDE.md 包测试 dual-mode 约定)。
- **真机 smoke**:radxa(RK3588 host + RK1828 EP)指向真二进制,能量/ASR 校验出真语音。
- 单测:协议解析(截断帧/句末标记/崩溃 EOF)、Service int16→float32 转换、factory 选中 `qwen3_tts_rk1828`、platform 注册解析。

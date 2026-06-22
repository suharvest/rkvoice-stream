<p align="center">
  <img src="media/logo.png" alt="rkvoice-stream" width="200">
</p>

<h1 align="center">rkvoice-stream</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/Platform-RK3576%20%7C%20RK3588%20%7C%20RK1828%20(NPU%20coprocessor)-orange.svg" alt="Platform">
</p>

<p align="center">
  Deploy streaming ASR + TTS on RK3576/RK3588 — <strong>120ms TTS latency, 52-language ASR, one Docker command.</strong>
</p>

<!-- TODO: Add demo GIF here — record a terminal session showing:
     1. docker-compose up (service starts)
     2. curl POST /tts with Chinese text → WAV file plays
     3. WebSocket /asr/stream with microphone → real-time transcription
     Target length: ~15 seconds -->

## What is this?

rkvoice-stream is a ready-to-deploy speech AI service for Rockchip NPU devices. It runs ASR and TTS entirely on-device via RKNN/RKLLM acceleration — no cloud, no GPU, no internet required. Ship it as a Python library or a Docker container.

It also supports the **RK1828 PCIe NPU coprocessor** (an accelerator card attached to an RK3576/RK3588 host) for on-device TTS (`qwen3_tts_rk1828`) and a multimodal **AudioLLM** (`gemma4_rk1828`, Gemma-4) that takes audio and streams text — collapsing ASR + LLM into a single model.

## Table of Contents

- [Performance](#performance)
- [AudioLLM — Gemma-4 (RK1828)](#audiollm--gemma-4-rk1828)
- [Features](#features)
- [Supported Platforms](#supported-platforms)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Architecture](#architecture)
- [Model Preparation](#model-preparation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Performance

### ASR — five backends

| Backend | Languages | Type | RK3576 RTF | RK3588 RTF |
|---------|:---------:|------|:----------:|:----------:|
| **Qwen3-ASR** (NPU) | 52 | RKNN + RKLLM | 0.44 | 0.34 |
| **Paraformer** (Hybrid) | 4 | RKNN encoder prefix + CPU suffix/decoder | 0.29 | 0.33 |
| **Paraformer** (CPU) | 4 | sherpa-onnx streaming | 0.50 | 0.24 |
| **SenseVoice** (NPU) | 50+ | RKNN encoder + CPU CTC | — | — |
| **SenseVoice** (CPU) | 50+ | sherpa-onnx | 0.36 | 0.11 |

### TTS — six backends

| Backend | Languages | Type | RK3576 RTF | RK3588 RTF |
|---------|-----------|------|:----------:|:----------:|
| **Matcha + Vocos** | zh, en | RKNN vocoder (NPU) | 0.19 | 0.10 |
| **Piper VITS** | en, zh, de, fr, ja, … | Hybrid CPU + NPU | ~0.05 | ~0.03 |
| **Kokoro** | en, zh | RKNN (NPU) | — | — |
| **Qwen3-TTS** | zh, en | RKNN (NPU) | — | — |
| **Qwen3-TTS (RK1828)** | zh, en | RKNN3 on RK1828 PCIe NPU coprocessor | — | — |
| **MOSS-TTS-Nano** (experimental) | zh, en | ORT (CPU) / RKNN hybrid | — | — |

> **Qwen3-TTS (RK1828)** — `qwen3_tts_rk1828` runs Qwen3-TTS (~1.7 GB) on the RK1828 PCIe
> NPU coprocessor via the RKNN3 toolchain (driven by a subprocess worker over PCIe). See
> [`docs/rk1828-qwen3-tts.md`](docs/rk1828-qwen3-tts.md).

> **MOSS-TTS-Nano (experimental)** — supported via **ORT (CPU) and hybrid** routes only; the
> ORT path is the production-correctness fallback. **NPU acceleration is not production-ready**
> (RKNN hybrid prefill is slower than full ORT and/or fails the ASR roundtrip quality gate; the
> RKLLM custom export fails hidden-state parity). Expect **performance issues** vs the Matcha/Piper
> NPU paths. See [`docs/moss-rknn-rk3576.md`](docs/moss-rknn-rk3576.md).

### Voice-to-Voice (EOS → First Audio)

Streaming V2V latency: time from user stops speaking to first TTS audio chunk.
Audio streamed at real-time pace (simulating live microphone). Qwen3-ASR (NPU) + Matcha TTS.

| Sentence | RK3576 | RK3588 |
|----------|:------:|:------:|
| 你好世界 (1.5s) | 949 ms | **685 ms** |
| 今天天气真不错 (2.0s) | 1604 ms | **1429 ms** |
| 语音识别测试 (3.1s) | 1700 ms | **1408 ms** |
| Hello world (1.7s) | 1289 ms | **644 ms** |
| **Average** | **1385 ms** | **1042 ms** |

## AudioLLM — Gemma-4 (RK1828)

A new engine type beyond ASR and TTS: an **AudioLLM** consumes audio (plus an optional
text prompt) and **streams text** back, collapsing the ASR + LLM "understanding" steps of a
voice-to-voice pipeline into a single multimodal model.

| Backend | Input → Output | Type | Platform | Status |
|---------|----------------|------|----------|--------|
| **Gemma-4** (`gemma4_rk1828`) | audio (+ optional prompt) → streaming text | RKNN3 (subprocess worker over PCIe) | RK1828 PCIe NPU coprocessor (~4.2 GB) | experimental |

Used by the `/audio_dialogue` WebSocket endpoint for V2V: audio in → AudioLLM streaming text
→ TTS → audio out.

> **Single-EP constraint** — the RK1828 has a single ~5 GB NPU. Gemma-4 (~4.2 GB) and
> Qwen3-TTS (~1.7 GB) do not fit at the same time, so V2V runs Gemma-4 on the RK1828 with
> TTS on the host SoC's NPU, or time-shares the coprocessor between them.

> **Experimental — firmware flakiness.** Loading Gemma-4 on the RK1828 occasionally hits a
> firmware `ACK_FAIL` during model setup (an RKNN3 V1.0.4 firmware-layer issue, mitigated by
> worker-level retry). See [`docs/rk1828-upstream-modelsetup-ackfail.md`](docs/rk1828-upstream-modelsetup-ackfail.md)
> and [`docs/rk1828-gemma4.md`](docs/rk1828-gemma4.md).

## Features

- **ASR: Qwen3-ASR** — streaming + offline, 52 languages, RKNN encoder + RKLLM decoder on NPU
- **ASR: Paraformer RKNN** — experimental hybrid split: FP16 RKNN encoder prefix through block30, CPU ONNX encoder suffix and decoder; boundary parity verified on RK3588 and RK3576
- **ASR: SenseVoice RKNN** — offline, 50+ languages, RKNN encoder + CPU CTC decode
- **ASR: SenseVoice** — offline + VAD streaming, 50+ languages, CPU (sherpa-onnx)
- **ASR: Paraformer** — native streaming, zh/en/ja/ko, CPU (sherpa-onnx)
- **TTS: Matcha + Vocos** — high-quality Chinese/English synthesis, NPU-accelerated vocoder
- **TTS: Piper VITS** — lightweight multi-language TTS (en, zh, de, fr, ja, …), hybrid CPU+NPU
- **TTS: Kokoro RKNN** — multi-stage RKNN synthesis (en, zh), NPU-accelerated
- **TTS: Qwen3-TTS** — RKNN streaming TTS (zh, en) on NPU
- **TTS: Qwen3-TTS (RK1828)** — `qwen3_tts_rk1828`, Qwen3-TTS on the RK1828 PCIe NPU coprocessor (RKNN3)
- **AudioLLM: Gemma-4 multimodal (audio→text) on RK1828 PCIe NPU** — `gemma4_rk1828`, streams text from audio (+ optional prompt), powering single-model V2V (experimental)
- **TTS: MOSS-TTS-Nano** (experimental) — ORT (CPU) and RKNN-hybrid routes only; ORT is the
  production-correctness fallback, NPU path is not production-ready (see note above)
- **Streaming everywhere** — WebSocket ASR (real-time partials), streaming TTS (sentence-by-sentence PCM)
- **Voice-to-voice pipeline** — ASR → LLM → TTS dialogue orchestrator with sub-second first-audio latency
- **NPU accelerated** — runs on Rockchip RKNN/RKLLM, not CPU
- **Config profiles** — pre-validated YAML configs for common setups (ASR-only, TTS-only, full stack)
- **jetson-voice compatible** — same HTTP/WebSocket API, drop-in replacement for RK platforms

## Supported Platforms

| Platform | NPU | CPU | RKLLM Quant | V2V Latency | Status |
|----------|-----|-----|-------------|:-----------:|--------|
| RK3576 | 2 cores, 6 TOPS | 2x A72 + 4x A55 | W4A16 | ~1.1s | Tested |
| RK3588 | 3 cores, 6 TOPS | 4x A76 + 4x A55 | FP16 | **~0.7s** | Tested |
| RK1828 | PCIe NPU coprocessor (~5 GB, RKNN3), attached to RK3576/RK3588 host; device `0001:11:00.0` | host SoC | — | — | Experimental |

> **RK1828** is a PCIe NPU coprocessor (accelerator card), **not** a standalone SoC. It plugs
> into an RK3576/RK3588 host and uses a **different toolchain (RKNN3)** than the host's RKNN2;
> it is driven by a subprocess worker over PCIe (C++ server-mode demo). Backends:
> `qwen3_tts_rk1828` (TTS) and `gemma4_rk1828` (AudioLLM). See
> [`docs/rk1828-package-integration.md`](docs/rk1828-package-integration.md) for the full
> first-class-platform design.

## Quick Start

### Option 1: Docker (recommended)

```bash
# Build
cd docker && docker build -t rkvoice-stream -f Dockerfile .. && cd ..

# Run with pre-validated config
docker-compose -f docker/docker-compose.yml up
```

### Option 2: Python library

```bash
pip install /path/to/rkvoice-stream
```

```python
from rkvoice_stream import create_asr, create_tts

# ASR
asr = create_asr(backend="qwen3_asr_rk", model_dir="/opt/models/asr", platform="rk3576")
result = asr.transcribe("audio.wav", language="Chinese")
print(result.text)

# Streaming ASR
stream = asr.create_stream(language="Chinese")
stream.feed_audio(audio_chunk)
final = stream.finish()

# TTS
tts = create_tts(backend="matcha_rknn", model_dir="/opt/models/tts", platform="rk3576")
wav_bytes = tts.synthesize("Hello world")

# Streaming TTS
for chunk, meta in tts.synthesize_stream("Hello world"):
    play(chunk)

# AudioLLM (Gemma-4 on RK1828): audio (+ optional prompt) -> streaming text
from rkvoice_stream.engine.audio_llm import create_audio_llm

audio_llm = create_audio_llm("gemma4_rk1828")  # or AUDIO_LLM_BACKEND env
audio_llm.preload()
for token in audio_llm.generate_stream(audio, sample_rate=16000, prompt="Reply in English."):
    print(token, end="", flush=True)
```

### Option 3: Config profile

```python
from rkvoice_stream import load_config, create_from_config

config = load_config("configs/rk3576-full.yaml")
asr, tts = create_from_config(config)
```

## API Reference

All endpoints are compatible with [jetson-voice](https://github.com/dusty-nv/jetson-voice) clients.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tts` | Synthesize text to WAV |
| POST | `/tts/stream` | Streaming TTS (PCM chunks) |
| POST | `/asr` | Transcribe audio file |
| WS | `/asr/stream` | Streaming ASR (real-time partials) |
| WS | `/dialogue` | Voice-to-voice dialogue pipeline (ASR → LLM → TTS) |
| WS | `/audio_dialogue` | Voice-to-voice via AudioLLM (audio → AudioLLM streaming text → TTS → audio) |
| GET | `/health` | Service health + backend status |
| GET | `/capabilities` | NPU resource usage + conflict info |

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Application Layer                                │
│  FastAPI server, WebSocket streaming, dialogue,   │
│  capability/conflict detection                    │
├──────────────────────────────────────────────────┤
│  Engine Layer (public API)                        │
│  ASREngine ABC + TTSEngine ABC + AudioLLM ABC     │
│  + factories                                      │
├──────────────────────────────────────────────────┤
│  Backend Layer                                    │
│  Qwen3-ASR (RKNN + RKLLM)  │  Matcha+Vocos      │
│  Piper VITS (RKNN)         │  Qwen3-TTS          │
│  Gemma-4 (RK1828 AudioLLM) │  Qwen3-TTS (RK1828) │
├──────────────────────────────────────────────────┤
│  Platform + Runtime                               │
│  RK3576/RK3588 configs, RKNN/RKLLM wrappers,    │
│  RK1828 (RKNN3, subprocess worker over PCIe)      │
└──────────────────────────────────────────────────┘
```

Data flow (streaming ASR):
```
Mic → int16 PCM → [WebSocket] → VAD → Mel → RKNN Encoder → RKLLM Decoder → Text
                                                  NPU                NPU
```

Data flow (streaming TTS):
```
Text → Phonemes → Matcha (NPU) → Mel → Vocos (NPU) → ISTFT (CPU) → PCM → [WebSocket]
```

## Model Preparation

Models are not bundled — use the conversion scripts in `models/` to generate them.
These are **build-time scripts only** (run on x86/WSL2 with rknn-toolkit2); they are not
part of the pip package and are never deployed to the device.

```
models/
├── asr/qwen3/       # RKNN encoder, RKLLM decoder, matmul weights
├── asr/paraformer/  # Paraformer RKNN conversion + hybrid-bucket export
├── asr/sensevoice/  # SenseVoice RKNN export
├── tts/matcha/      # Matcha+Vocos RKNN conversion + ONNX fixes
├── tts/piper/       # Piper VITS split (CPU encoder + NPU decoder)
├── tts/kokoro/      # Kokoro RKNN fixes
├── tts/moss/        # MOSS-TTS island/bucket build, parity & smoke scripts
├── rk1828/          # RK1828 PCIe coprocessor model production
└── common/          # Shared tools: Sin→polynomial, ScatterND bake, Erf→Tanh
```

Expected model layout on the device:

```
/opt/models/
├── asr/
│   ├── encoder/rk3576/*.rknn
│   ├── decoder/rk3576/*.rkllm
│   ├── embed_tokens.npy
│   ├── mel_filters.npy
│   └── tokenizer.json
└── tts/rk3576/
    ├── matcha.fp16.rknn
    └── vocos.w4a16.rknn
```

## Configuration

Pre-validated profiles in `configs/`:

| Profile | Description |
|---------|-------------|
| `rk3576-full.yaml` | ASR + Matcha TTS (split NPU cores) |
| `rk3576-paraformer-matcha.yaml` | Paraformer RKNN ASR (hybrid encoder + RKNN decoder) + Matcha TTS |
| `rk3576-asr-only.yaml` | ASR with both NPU cores |
| `rk3576-tts-only.yaml` | Matcha TTS only |
| `rk3576-piper-multilang.yaml` | Piper multi-language TTS |
| `rk3588-full.yaml` | RK3588 full stack |
| `rk3588-paraformer-matcha.yaml` | RK3588 Paraformer RKNN ASR (hybrid encoder + RKNN decoder) + Matcha TTS |

Use via Docker:
```bash
docker run -e CONFIG=rk3576-full rkvoice-stream
```

Enable the validated Paraformer hybrid ASR container profile with an artifact
directory mounted at `/opt/asr/paraformer`. This uses RKNN encoder prefix,
CPU encoder suffix, and RKNN decoder:

```bash
PARAFORMER_HOST_MODEL_DIR=/home/cat/models/paraformer-hybrid \
PARAFORMER_CONTAINER_RKNN_DIR=/opt/asr/paraformer/rknn/rk3576 \
docker compose -f docker/docker-compose.yml \
  -f docker/docker-compose.paraformer-hybrid.yml \
  --profile paraformer-hybrid up -d
```

This profile uses the published arm64 image
`sensecraft-missionpack.seeed.cn/solution/seeed-local-voice:rk-v1.5-paraformer-hybrid`.
Published digest: `sha256:8dec7528ed4e08b919f0b2fd9192b8564d2b713df8552aed3eb98202c0a2c194`.
For RK3588 set `PARAFORMER_CONTAINER_RKNN_DIR=/opt/asr/paraformer/rknn/rk3588`.
Export/upload scripts live under `models/asr/paraformer/`; generated artifacts
are stored in the existing RK artifact repo under
`harvestsu/seeed-local-voice-rk-artifacts/rk3576/paraformer-hybrid/` and
`harvestsu/seeed-local-voice-rk-artifacts/rk3588/paraformer-hybrid/`.

Measured hybrid ASR performance uses the same Python pipeline baseline with
full ONNX Runtime vs RKNN prefix + ONNX suffix/decoder. RK3576 improved from
0.58 RTF to 0.29 RTF, and RK3588 improved from 0.63 RTF to 0.33 RTF. The
actual `paraformer_rknn` backend entry measured 0.21 RTF on the 10.05s
validation sample on RK3576. The container profile has also been rebuilt and
validated on real RK3576/RK3588 devices with the same validation sample.

## Testing

Dual-mode test suite — works against a live container or directly on device:

```bash
# On device (direct mode)
pytest tests/ -v

# Against running container (HTTP mode)
SERVICE_URL=http://192.168.1.100:8621 pytest tests/ -v
```

Quality gates: CER < 0.5 per sentence, RTF < 1.0.

## Acknowledgements

Built on top of these projects:

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — speech inference engine (Piper, Kokoro, VAD)
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3 TTS model
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) — non-autoregressive TTS
- [Piper](https://github.com/rhasspy/piper) — fast local neural TTS
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) — Rockchip NPU SDK
- [RKLLM-Toolkit](https://github.com/airockchip/rkllm-toolkit) — Rockchip LLM SDK

## License

[Apache-2.0](LICENSE)

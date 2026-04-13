# rkvoice-stream

Streaming speech AI service for Rockchip NPU platforms — ASR and TTS via a Python library or Docker container.

## Features

- Streaming ASR with real-time partial results
- Streaming TTS with low-latency audio chunking
- Multiple backends: Qwen3-ASR, Matcha+Vocos, Piper VITS, Qwen3-TTS
- Multi-platform: RK3576 and RK3588
- HTTP and WebSocket API compatible with jetson-voice clients
- Docker ready with pre-validated config profiles
- NPU conflict detection and capability reporting

## Supported Platforms

| Platform | NPU Cores | NPU Memory Limit | CPU Big Cores |
|----------|-----------|-----------------|---------------|
| RK3576   | 2         | 180 MB          | 2x A72        |
| RK3588   | 3         | 512 MB          | 4x A76        |

## Available Backends

### ASR

| Backend   | Languages | Streaming | RTF (RK3576) |
|-----------|-----------|-----------|--------------|
| Qwen3-ASR | 52        | Yes       | ~0.44        |

### TTS

| Backend      | Languages | Streaming | RTF (RK3576) |
|--------------|-----------|-----------|--------------|
| Matcha+Vocos | zh/en     | Yes       | ~0.07        |
| Piper VITS   | multi     | Yes       | ~0.034       |
| Qwen3-TTS    | zh/en     | Yes       | TBD          |

## Quick Start

### Python Library

```python
from rkvoice_stream import create_asr, create_tts

asr = create_asr(
    backend="qwen3",
    model_dir="/opt/models/asr",
    platform="rk3576",
)

tts = create_tts(
    backend="matcha",
    model_dir="/opt/models/tts",
    platform="rk3576",
)

# Offline transcription
result = asr.transcribe("audio.wav", language="Chinese")
print(result.text)

# Streaming ASR
stream = asr.create_stream(language="Chinese")
stream.feed_audio(audio_chunk)
final = stream.finish()

# TTS synthesis
wav_bytes = tts.synthesize("Hello world")

# Streaming TTS
for audio_chunk, meta in tts.synthesize_stream("Hello world"):
    play(audio_chunk)
```

Load a pre-validated config profile:

```python
from rkvoice_stream import load_config, create_from_config

config = load_config("configs/rk3576-full.yaml")
asr, tts = create_from_config(config)
```

### Docker Deployment

```yaml
# docker/docker-compose.yml
services:
  speech:
    image: rkvoice-stream:latest
    privileged: true
    network_mode: host
    volumes:
      - /path/to/models:/opt/models:ro
    environment:
      - PLATFORM=rk3576
      - CONFIG=rk3576-full
      - ASR_BACKEND=qwen3
      - ASR_MODEL_DIR=/opt/models/asr
      - TTS_BACKEND=matcha
      - TTS_MODEL_DIR=/opt/models/tts
```

```bash
cd docker
docker build -t rkvoice-stream -f Dockerfile ..
docker-compose up
```

### Config Profiles

Five pre-validated profiles are included in `configs/`:

| Profile                    | Description                          |
|----------------------------|--------------------------------------|
| `rk3576-asr-only.yaml`     | ASR only, full NPU for encoder       |
| `rk3576-tts-only.yaml`     | TTS only                             |
| `rk3576-full.yaml`         | ASR + Matcha TTS (sequential NPU)    |
| `rk3576-piper-multilang.yaml` | Piper multi-language TTS          |
| `rk3588-full.yaml`         | RK3588 full ASR + TTS stack          |

Select a profile at runtime:

```bash
docker run -e CONFIG=rk3576-full rkvoice-stream
```

## API Reference

All endpoints are compatible with existing jetson-voice clients.

| Method | Path            | Description                                           |
|--------|-----------------|-------------------------------------------------------|
| POST   | `/tts`          | Synthesize text to WAV (`{"text": "...", "sid": 0}`)  |
| POST   | `/tts/stream`   | Streaming TTS: PCM stream (uint32 LE header + int16)  |
| POST   | `/asr`          | Transcribe audio file (multipart) to JSON             |
| WS     | `/asr/stream`   | Streaming ASR: int16 PCM in, JSON partials out        |
| WS     | `/dialogue`     | Full V2V dialogue: text JSON in, PCM audio stream out |
| GET    | `/health`       | Service health including backend info and conflicts   |
| GET    | `/capabilities` | Loaded backends, resource usage, parallel capability  |

## Model Preparation

Models are not bundled. Conversion scripts are in `models/`:

```
models/
├── asr/qwen3/         # Export RKNN encoder, RKLLM decoder, matmul weights
├── tts/matcha/        # Convert and split Matcha+Vocos for RKNN
├── tts/piper/         # Batch convert Piper VITS voices for RKNN
└── common/            # Shared ONNX fix utilities (sin, ScatterND, erf)
```

Each subdirectory contains a `README.md` with step-by-step conversion instructions.

Expected model layout under `/opt/models/`:

```
/opt/models/
├── asr/
│   ├── encoder/rk3576/encoder.4s.fp16.rknn
│   ├── decoder/rk3576/decoder.w4a16.rkllm
│   ├── embed_tokens.npy
│   ├── mel_filters.npy
│   └── tokenizer.json
└── tts/
    └── rk3576/
        ├── matcha.fp16.rknn
        └── vocos.w4a16.rknn
```

## Project Structure

```
rkvoice-stream/
├── rkvoice_stream/    # Python package
│   ├── engine/        # Public API: ABCs and factory functions
│   ├── backends/      # ASR and TTS backend implementations
│   ├── app/           # FastAPI server, dialogue orchestrator, capability detection
│   ├── platform/      # RK3576/RK3588 device config constants
│   ├── runtime/       # Low-level RKNN/RKLLM wrappers
│   └── vad/           # Voice activity detection (Silero)
├── models/            # Model conversion scripts (not pip-packaged)
├── configs/           # Pre-validated YAML config profiles
├── docker/            # Dockerfile and docker-compose.yml
├── docs/              # Guides: quickstart, model conversion, API reference
└── tests/             # Test suite (dual-mode: direct backend or HTTP)
```

## Testing

```bash
# On device (direct backend mode, requires NPU + models)
pytest tests/ -v

# Against running container (HTTP mode)
SERVICE_URL=http://192.168.1.100:8621 pytest tests/ -v
```

Quality gates: CER < 0.5, RTF < 1.0.

## License

Apache-2.0

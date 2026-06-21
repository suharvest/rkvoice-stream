# rkvoice-stream

## Project Overview

Streaming speech AI service for Rockchip NPU platforms (RK3576/RK3588), plus the
RK1828 PCIe NPU coprocessor (RKNN3 accelerator card attached to an RK3576/RK3588 host,
`is_coprocessor=True`, addressed by PCIe BDF `0001:11:00.0`) for TTS and a multimodal
AudioLLM (audio -> streaming text).
Python package: `rkvoice_stream`. Three-layer architecture: app -> engine -> backends.

## Architecture

- `rkvoice_stream/engine/` — Public API (ABCs + factories: asr, tts, audio_llm)
- `rkvoice_stream/backends/` — Concrete implementations (ASR: qwen3, TTS: matcha/piper/qwen3/qwen3_tts_rk1828, AudioLLM: gemma4_rk1828)
- `rkvoice_stream/app/` — FastAPI server, dialogue orchestrator, capability detection
- `rkvoice_stream/platform/` — Device config constants (RK3576, RK3588, RK1828)
- `rkvoice_stream/runtime/` — Low-level RKNN/RKLLM wrappers; `rknn3_worker` (RK1828 subprocess worker over PCIe)
- `models/` — Model conversion scripts (not part of pip package)
- `configs/` — Pre-validated YAML config profiles

## Key Rules

- Use absolute imports (`from rkvoice_stream.xxx import`) not sys.path hacks
- Backend implementations must not import from each other; only from engine/ ABCs
- NPU resource management stays simple (threading.Lock + manual domain_id)
- Conflict detection is in app/capability.py, not in backends
- Model paths are user-configured, no auto-detection
- Tests support dual-mode: HTTP (SERVICE_URL env) or direct backend loading

## Testing

- On device: `pytest tests/ -v`
- Against container: `SERVICE_URL=http://host:8621 pytest tests/ -v`
- Quality gates: CER < 0.5, RTF < 1.0

## Docker

- Build: `cd docker && docker build -t rkvoice-stream -f Dockerfile ..`
- Run: `docker-compose -f docker/docker-compose.yml up`

# rkvoice-stream

## Project Overview

Streaming speech AI service for Rockchip NPU platforms (RK3576/RK3588).
Python package: `rkvoice_stream`. Three-layer architecture: app -> engine -> backends.

## Architecture

- `rkvoice_stream/engine/` — Public API (ABCs + factories)
- `rkvoice_stream/backends/` — Concrete implementations (ASR: qwen3, TTS: matcha/piper/qwen3)
- `rkvoice_stream/app/` — FastAPI server, dialogue orchestrator, capability detection
- `rkvoice_stream/platform/` — Device config constants (RK3576, RK3588)
- `rkvoice_stream/runtime/` — Low-level RKNN/RKLLM wrappers
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

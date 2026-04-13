# RK3576 Speech Service Docker Container (TTS + ASR)
#
# TTS: Matcha-icefall-zh-en (RKNN NPU) + Vocos vocoder (RKNN NPU)
# ASR: Qwen3-ASR encoder (RKNN NPU) + matmul decoder (rknn-matmul-parallel)
#
# Build on cat-remote (aarch64):
#   docker build --network=host -t rk3576-speech -f Dockerfile .
#
# Prereqs on host:
#   cd ~/project/rknn-matmul-parallel && make python   # builds matmul_decoder .so
#
# Run via docker-compose (see docker-compose.yml) or manually:
#   docker run --privileged --network=host \
#     -v /dev:/dev \
#     -v /home/cat/models:/opt/tts/models:ro \
#     -v /home/cat/qwen3-asr-models:/opt/asr/models:ro \
#     -v /home/cat/rknn-matmul-parallel:/opt/rknn-matmul-parallel:ro \
#     -e TTS_BACKEND=matcha_rknn -e ASR_BACKEND=qwen3_asr_rk \
#     -e ASR_DECODER_TYPE=matmul -e PYTHONPATH=/opt/rknn-matmul-parallel \
#     rk3576-speech

FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    libsndfile1 \
    libgomp1 \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Python venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Python deps (install in stages for better caching)
RUN pip install --no-cache-dir \
    'numpy<2' \
    soundfile \
    librosa

RUN pip install --no-cache-dir \
    fastapi \
    'uvicorn[standard]' \
    python-multipart

RUN pip install --no-cache-dir \
    transformers \
    tokenizers

RUN pip install --no-cache-dir pybind11

RUN pip install --no-cache-dir sherpa-onnx onnxruntime

# RKNN toolkit lite2 (aarch64 wheel from PyPI)
RUN pip install --no-cache-dir \
    rknn-toolkit-lite2

# RKLLM + RKNN runtime libraries (copied from host /usr/lib/)
COPY librkllmrt.so /usr/lib/librkllmrt.so
COPY librknnrt.so /usr/lib/librknnrt.so
RUN ldconfig

RUN mkdir -p /opt/tts/models /opt/asr/models

# matmul_decoder C extension: pre-built on host from ~/project/rknn-matmul-parallel
# and mounted via docker-compose volume. See: make python in that repo.

# Scripts (extract_codec_embed_table.py etc.)
COPY scripts/ /opt/app/scripts/

# Application code
COPY app/ /opt/tts/app/

ENV MODEL_DIR=/opt/tts/models
ENV TTS_BACKEND=matcha_rknn
ENV TTS_MODEL_DIR=/opt/tts/models
ENV ASR_MODEL_DIR=/opt/asr/models
# ASR_BACKEND is intentionally unset here — enable at runtime via env var
# e.g. -e ASR_BACKEND=qwen3_asr_rk
ENV LD_LIBRARY_PATH=/usr/lib
WORKDIR /opt/tts/app
EXPOSE 8621

CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8621"]

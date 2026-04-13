#!/bin/bash
# Deploy Qwen3-TTS service on RK3576 (cat-remote)
#
# Usage:
#   ./deploy.sh [docker|venv]
#
# docker: Build and run in Docker container (requires Docker with iptables support)
# venv:   Run directly in Python venv (default, more compatible)

set -euo pipefail

FLEET="uv run --project /Users/harvest/project/_hub python /Users/harvest/project/_hub/fleet.py"
DEVICE="cat-remote"
REMOTE_DIR="/home/cat/rk3576-tts"
MODEL_DIR="/home/cat/qwen3-tts-rknn"
RKLLM_MODEL="/home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm"
LOCAL_APP_DIR="$(cd "$(dirname "$0")/app" && pwd)"

MODE="${1:-venv}"

echo "=== Deploying RK3576 TTS Service (mode: $MODE) ==="

# Check device is online
echo "Checking device status..."
if ! $FLEET exec --timeout 10 "$DEVICE" -- "echo ok" >/dev/null 2>&1; then
    echo "ERROR: $DEVICE is offline"
    exit 1
fi

# Upload app files
echo "Uploading app files..."
$FLEET exec --timeout 30 "$DEVICE" -- "mkdir -p $REMOTE_DIR/app" 2>/dev/null
for f in main.py tts_service.py rkllm_wrapper.py; do
    $FLEET push "$DEVICE" "$LOCAL_APP_DIR/$f" "$REMOTE_DIR/app/$f" 2>/dev/null
done

if [ "$MODE" = "docker" ]; then
    # Docker deployment
    echo "=== Docker Build ==="
    $FLEET push "$DEVICE" "$(dirname "$0")/Dockerfile" "$REMOTE_DIR/Dockerfile" 2>/dev/null
    $FLEET push "$DEVICE" "$(dirname "$0")/docker-compose.yml" "$REMOTE_DIR/docker-compose.yml" 2>/dev/null
    $FLEET exec --timeout 10 "$DEVICE" -- "cp /tmp/librkllmrt.so $REMOTE_DIR/librkllmrt.so" 2>/dev/null

    echo "Building Docker image (this takes a few minutes)..."
    $FLEET exec --timeout 600 "$DEVICE" -- \
        "cd $REMOTE_DIR && docker build --network=host -t rk3576-tts:latest . 2>&1 | tail -5" 2>/dev/null

    echo "Starting container..."
    $FLEET exec --sudo --timeout 60 "$DEVICE" -- \
        "docker stop rk3576-tts 2>/dev/null; docker rm rk3576-tts 2>/dev/null; \
         docker run -d --name rk3576-tts --privileged \
         -v /dev:/dev \
         -v $MODEL_DIR:/opt/tts/models:ro \
         -v $RKLLM_MODEL:/opt/tts/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm:ro \
         -p 8621:8000 rk3576-tts:latest" 2>/dev/null
else
    # Venv deployment (no Docker)
    echo "=== Venv Setup ==="
    $FLEET exec --timeout 300 "$DEVICE" -- "
        cd $REMOTE_DIR
        if [ ! -d venv ]; then
            python3 -m venv venv
            source venv/bin/activate
            pip install --no-cache-dir 'numpy<2' soundfile fastapi 'uvicorn[standard]' python-multipart transformers tokenizers rknn-toolkit-lite2
        else
            source venv/bin/activate
        fi
        echo 'Venv ready'
    " 2>/dev/null

    echo "Creating systemd service..."
    $FLEET exec --sudo --timeout 30 "$DEVICE" -- "cat > /etc/systemd/system/rk3576-tts.service << 'EOF'
[Unit]
Description=RK3576 Qwen3-TTS Service
After=network.target

[Service]
Type=simple
User=cat
WorkingDirectory=$REMOTE_DIR/app
Environment=MODEL_DIR=$MODEL_DIR
Environment=LD_LIBRARY_PATH=/usr/lib
ExecStart=$REMOTE_DIR/venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8621
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable rk3576-tts
systemctl restart rk3576-tts
" 2>/dev/null

    echo "Service started. Check with: systemctl status rk3576-tts"
fi

echo ""
echo "=== Deploy Complete ==="
echo "Test: curl -X POST http://\$DEVICE_IP:8621/tts -H 'Content-Type: application/json' -d '{\"text\":\"你好\"}' -o test.wav"

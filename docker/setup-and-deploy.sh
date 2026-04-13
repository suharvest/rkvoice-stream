#!/bin/bash
# Full setup and deployment for RK3576 Qwen3-TTS service.
#
# This script:
#   1. Fixes Docker iptables issue (RK3576 kernel lacks iptable_raw)
#   2. Uploads app code to the device
#   3. Sets up Python venv with dependencies
#   4. Starts the TTS service
#
# Usage:
#   ./setup-and-deploy.sh
#
# Prerequisites:
#   - cat-remote device must be online (via Tailscale)
#   - Model files already in place on device
#   - fleet.py configured with device credentials

set -euo pipefail

FLEET="uv run --project /Users/harvest/project/_hub python /Users/harvest/project/_hub/fleet.py"
DEVICE="cat-remote"
REMOTE_DIR="/home/cat/rk3576-tts"
MODEL_DIR="/home/cat/qwen3-tts-rknn"
RKLLM_MODEL="/home/cat/models/talker_fullvocab_fixed_w4a16_rk3576.rkllm"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8621

echo "=== RK3576 Qwen3-TTS: Setup & Deploy ==="
echo ""

# Step 0: Check device
echo "[1/6] Checking device connectivity..."
if ! $FLEET exec --timeout 15 "$DEVICE" -- "echo ok" >/dev/null 2>&1; then
    echo "ERROR: $DEVICE is offline. Cannot proceed."
    exit 1
fi
echo "  Device is online."

# Step 1: Fix Docker (if needed)
echo ""
echo "[2/6] Fixing Docker iptables configuration..."
$FLEET exec --sudo --timeout 60 "$DEVICE" -- "
    # Ensure iptables-legacy
    update-alternatives --set iptables /usr/sbin/iptables-legacy 2>/dev/null || true
    update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy 2>/dev/null || true

    # Configure Docker
    mkdir -p /etc/docker
    cat > /etc/docker/daemon.json << 'INNER_EOF'
{
  \"iptables\": false,
  \"registry-mirrors\": [\"https://docker.1ms.run\", \"https://docker.xuanyuan.me\"]
}
INNER_EOF

    # Restart Docker only if config changed
    if docker ps >/dev/null 2>&1; then
        echo 'Docker already running'
    else
        systemctl restart docker 2>&1
        sleep 3
        if docker ps >/dev/null 2>&1; then
            echo 'Docker restarted OK'
        else
            echo 'Docker restart failed - will use venv mode'
        fi
    fi
" 2>/dev/null
echo "  Done."

# Step 2: Upload app code
echo ""
echo "[3/6] Uploading application code..."
$FLEET exec --timeout 30 "$DEVICE" -- "mkdir -p $REMOTE_DIR/app" 2>/dev/null

for f in main.py tts_service.py rkllm_wrapper.py; do
    $FLEET push "$DEVICE" "$LOCAL_DIR/app/$f" "$REMOTE_DIR/app/$f" 2>/dev/null
done
echo "  App code uploaded."

# Step 3: Verify model files
echo ""
echo "[4/6] Verifying model files..."
$FLEET exec --timeout 30 "$DEVICE" -- "
    echo 'Checking model files...'
    for f in text_project.rknn codec_embed.rknn code_predictor.rknn code_predictor_embed.rknn decoder_ctx25_int8.rknn codec_head_weight.npy; do
        if [ -f $MODEL_DIR/\$f ]; then
            echo \"  OK: \$f\"
        else
            echo \"  MISSING: \$f\"
        fi
    done
    if [ -f $RKLLM_MODEL ]; then
        echo '  OK: talker RKLLM'
    else
        echo '  MISSING: talker RKLLM'
    fi
    if [ -d $MODEL_DIR/tokenizer ]; then
        echo '  OK: tokenizer/'
    else
        echo '  MISSING: tokenizer/'
    fi
    if [ -d $MODEL_DIR/codebook_embeds ]; then
        echo '  OK: codebook_embeds/'
    else
        echo '  MISSING: codebook_embeds/'
    fi
" 2>/dev/null

# Step 4: Setup Python venv
echo ""
echo "[5/6] Setting up Python venv..."
$FLEET exec --timeout 300 "$DEVICE" -- "
    cd $REMOTE_DIR
    if [ ! -d venv ]; then
        echo 'Creating venv...'
        python3 -m venv venv
        source venv/bin/activate
        pip install --no-cache-dir 'numpy<2' soundfile fastapi 'uvicorn[standard]' python-multipart transformers tokenizers rknn-toolkit-lite2 2>&1 | tail -5
    else
        echo 'Venv already exists'
        source venv/bin/activate
    fi
    echo 'Python packages:'
    pip list 2>/dev/null | grep -iE 'numpy|soundfile|fastapi|uvicorn|transform|rknn'
" 2>/dev/null

# Step 5: Create symlinks for model files and start service
echo ""
echo "[6/6] Starting TTS service..."
$FLEET exec --timeout 30 "$DEVICE" -- "
    # Create symlinks so app can find models in MODEL_DIR
    ln -sf $RKLLM_MODEL $MODEL_DIR/talker_fullvocab_fixed_w4a16_rk3576.rkllm 2>/dev/null || true

    # Kill any existing service
    pkill -f 'uvicorn main:app.*$PORT' 2>/dev/null || true
    sleep 1

    # Start service in background
    cd $REMOTE_DIR
    source venv/bin/activate
    export MODEL_DIR=$MODEL_DIR
    export LD_LIBRARY_PATH=/usr/lib
    nohup python3 -m uvicorn app.main:app --host 0.0.0.0 --port $PORT > /tmp/rk3576-tts.log 2>&1 &
    echo \"PID: \$!\"
    sleep 2
    if curl -s http://localhost:$PORT/health >/dev/null 2>&1; then
        echo 'Service started and responding!'
    else
        echo 'Service starting (may take time to load models)...'
        echo 'Monitor with: tail -f /tmp/rk3576-tts.log'
    fi
" 2>/dev/null

echo ""
echo "=== Deploy Complete ==="
echo ""
echo "Endpoints:"
echo "  Health: curl http://\$(tailscale ip -4 cat-remote):$PORT/health"
echo "  TTS:    curl -X POST http://\$(tailscale ip -4 cat-remote):$PORT/tts -H 'Content-Type: application/json' -d '{\"text\":\"你好\"}' -o test.wav"
echo ""
echo "Logs: fleet exec cat-remote -- 'tail -50 /tmp/rk3576-tts.log'"

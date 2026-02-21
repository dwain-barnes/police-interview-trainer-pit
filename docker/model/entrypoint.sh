#!/bin/bash
set -e

MODEL_REPO="${MODEL_REPO:-EryriLabs/PIT-GGUF}"
GGUF_FILE="${GGUF_FILE:-pit_q8_0.gguf}"
MODEL_DIR="${MODEL_DIR:-/models/pit}"
N_CTX="${N_CTX:-16384}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

# Disable Xet storage protocol — causes 404s in some environments
export HF_HUB_DISABLE_XET=1

mkdir -p "${MODEL_DIR}"

GGUF_PATH="${MODEL_DIR}/${GGUF_FILE}"

# Download GGUF from HuggingFace if not already cached
if [ ! -f "${GGUF_PATH}" ]; then
    echo "==> GGUF not found at ${GGUF_PATH}, downloading ${GGUF_FILE} from ${MODEL_REPO}..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('${MODEL_REPO}', '${GGUF_FILE}', local_dir='${MODEL_DIR}')
"
    echo "==> Download complete."
else
    echo "==> GGUF already cached at ${GGUF_PATH}"
fi

if [ ! -f "${GGUF_PATH}" ]; then
    echo "ERROR: GGUF not found at ${GGUF_PATH} after download."
    exit 1
fi

# Start llama-server backend on internal port 8080
echo "==> Starting llama-server backend on port 8080..."
echo "    Model: ${GGUF_PATH}"
echo "    Context: ${N_CTX}"
echo "    GPU layers: ${N_GPU_LAYERS}"

llama-server \
    --model "${GGUF_PATH}" \
    --host 127.0.0.1 --port 8080 \
    --ctx-size "${N_CTX}" \
    -ngl "${N_GPU_LAYERS}" \
    --jinja \
    --chat-template-file /app/chat_template.jinja &

BACKEND_PID=$!

# Wait for backend to be ready
echo "==> Waiting for llama-server backend..."
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
        echo "==> Backend ready."
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo "ERROR: Backend process exited unexpectedly."
        exit 1
    fi
    sleep 2
done

if ! curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; then
    echo "ERROR: Backend failed to start within 240 seconds."
    exit 1
fi

# Start Harmony proxy on port 8000 (exposed to other containers)
echo "==> Starting Harmony proxy on port 8000..."
exec python3 proxy.py

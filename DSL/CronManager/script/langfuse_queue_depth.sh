#!/bin/bash
# Langfuse queue-depth check: logs BullMQ queue depth from Dragonfly
# to Loki. Scheduled every minute by langfuse_queue_depth.yml.

PYTHON="/app/python_virtual_env/bin/python3"
UV_BIN="/root/.local/bin/uv"

# The Dragonfly client library (redis-py; Dragonfly speaks its protocol) and requests
# (used by LokiLogger) are not part of the image;
# install them once into the venv. uv is installed on demand, the same way
# vector_indexer_pipeline.sh does it.
if ! "$PYTHON" -c "import redis, requests" 2>/dev/null; then
    if [ ! -x "$UV_BIN" ]; then
        echo "[UV] Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || {
            echo "[ERROR] Failed to install uv"
            exit 1
        }
    fi
    echo "[PACKAGES] Installing the Dragonfly client library and requests..."
    "$UV_BIN" pip install --python "$PYTHON" "redis>=5.0,<6.0" "requests>=2.32.5" || {
        echo "[ERROR] Could not install the Dragonfly client library/requests"
        exit 1
    }
fi

exec "$PYTHON" /app/scripts/langfuse_queue_depth.py

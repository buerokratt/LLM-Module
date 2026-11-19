#!/bin/bash
# Dragonfly metrics: logs memory, clients, load and uptime from the
# Redis-compatible store to Loki. Scheduled every minute by dragonfly_metrics.yml.

PYTHON="/app/python_virtual_env/bin/python3"
UV_BIN="/root/.local/bin/uv"

# redis (store client) and requests (used by LokiLogger) are not part of the image;
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
    echo "[PACKAGES] Installing redis and requests..."
    "$UV_BIN" pip install --python "$PYTHON" "redis>=5.0,<6.0" "requests>=2.32.5" || {
        echo "[ERROR] Could not install redis/requests"
        exit 1
    }
fi

exec "$PYTHON" /app/scripts/dragonfly_metrics.py

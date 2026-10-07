#!/bin/bash

echo "Starting vector indexer pipeline..."

if [ -z "$signedUrl" ] || [ -z "$clientDataHash" ]; then
  echo "Please set the signedUrl and clientDataHash environment variables."
  exit 1
fi

PYTHON_SCRIPT="/app/src/vector_indexer/main_indexer.py"

echo "Using signedUrl: $signedUrl"
echo "Using clientDataHash: $clientDataHash"

# Install uv if not found
UV_BIN="/root/.local/bin/uv"
if [ ! -f "$UV_BIN" ]; then
    echo "[UV] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || {
        echo "[ERROR] Failed to install uv"
        exit 1
    }
fi

# Activate Python virtual environment
VENV_PATH="/app/python_virtual_env"
echo "[VENV] Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate" || {
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
}

# Install required packages
echo "[PACKAGES] Installing required packages..."

"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "numpy>=1.21.0,<2.0" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "requests>=2.32.5" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "pydantic>=2.11.7" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "qdrant-client>=1.15.1" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "rank-bm25>=0.2.2" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "tiktoken>=0.11.0" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "dvc[s3]>=3.55.2" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "loguru>=0.7.3" || exit 1

echo "[PACKAGES] All packages installed successfully"

export PYTHONPATH="/app:/app/src:/app/src/vector_indexer:$PYTHONPATH"

# Service endpoint URLs; also exports RAG_SEARCH_CONSTANTS so the Python
# process below resolves the same constants.ini.
source /app/scripts/load_constants.sh

[ ! -f "$PYTHON_SCRIPT" ] && { echo "[ERROR] Python script not found"; exit 1; }

echo "[FOUND] Python script at: $PYTHON_SCRIPT"

# Run vector indexer with signed URL parameter
echo "[STARTING] Vector indexer processing..."

echo "[DEBUG] About to execute main_indexer.py..."
if [ -n "$signedUrl" ]; then
    echo "[SIGNED_URL] Using signed URL for dataset processing"
    echo "[COMMAND] python3 -u $PYTHON_SCRIPT --signed-url $signedUrl"
    python3 -u "$PYTHON_SCRIPT" --signed-url "$signedUrl" 2>&1
    PYTHON_EXIT_CODE=$?
else
    echo "[NO_URL] Running without signed URL"
    echo "[COMMAND] python3 -u $PYTHON_SCRIPT"
    python3 -u "$PYTHON_SCRIPT" 2>&1
    PYTHON_EXIT_CODE=$?
fi

echo "[DEBUG] Python execution completed with exit code: $PYTHON_EXIT_CODE"

# Handle exit codes
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Vector indexer completed successfully"
    exit 0
elif [ $PYTHON_EXIT_CODE -eq 2 ]; then
    echo "[WARNING] Vector indexer completed with some failures"
    exit 2
elif [ $PYTHON_EXIT_CODE -eq 130 ]; then
    echo "[INTERRUPTED] Vector indexer was interrupted by user"
    exit 130
else
    echo "[ERROR] Vector indexer failed with exit code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi
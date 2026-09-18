#!/bin/bash

echo "Starting API Tool Indexer pipeline..."

# Validate required environment variables
if [ -z "$endpoint_id" ] || [ -z "$name" ] || [ -z "$description" ] || [ -z "$url" ]; then
  echo "[ERROR] Missing required environment variables: endpoint_id, name, description, or url"
  exit 1
fi

PYTHON_SCRIPT="/app/src/api_tool_indexer/main_indexer.py"

echo "[INFO] Endpoint ID: $endpoint_id"
echo "[INFO] Endpoint Name: $name"

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
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "httpx>=0.27.0" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "pydantic>=2.11.7" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "qdrant-client>=1.15.1" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "requests>=2.32.5" || exit 1

echo "[PACKAGES] All packages installed successfully"

# Set Python path
export PYTHONPATH="/app:/app/src:/app/src/api_tool_indexer:$PYTHONPATH"

# Verify Python script exists
[ ! -f "$PYTHON_SCRIPT" ] && { echo "[ERROR] Python script not found at $PYTHON_SCRIPT"; exit 1; }

echo "[FOUND] Python script at: $PYTHON_SCRIPT"

# Run indexing script with arguments
echo "[STARTING] API Tool indexing processing..."

# URL decode function using Python
url_decode() {
    python3 -c "import sys; from urllib.parse import unquote; print(unquote(sys.argv[1]))" "$1"
}

# Write JSON arrays to temporary files to avoid bash parsing issues
TEMP_DIR=$(mktemp -d)
PARAMS_FILE="$TEMP_DIR/params.json"

if [ -n "$params" ]; then
    url_decode "$params" > "$PARAMS_FILE"
fi

# Decode URL to restore path parameter templates that were
# URL-encoded by Ruuter to prevent Spring URI template expansion errors.
DECODED_URL=$(url_decode "$url")

# Build Python command arguments array
PYTHON_ARGS=(
    "$PYTHON_SCRIPT"
    --endpoint-id "$endpoint_id"
    --service-id "${service_id:-""}"
    --name "$name"
    --description "$description"
    --url "$DECODED_URL"
    --method "${method:-"GET"}"
    --visibility "${visibility:-"public"}"
    --type "${type:-"custom_endpoint"}"
)

# Add params file if provided
[ -n "$params" ] && PYTHON_ARGS+=(--params-file "$PARAMS_FILE")

# Execute Python script
python3 -u "${PYTHON_ARGS[@]}" 2>&1
PYTHON_EXIT_CODE=$?

# Cleanup temporary files
rm -rf "$TEMP_DIR"

# Handle exit codes
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] API Tool indexing completed successfully"
    exit 0
else
    echo "[ERROR] API Tool indexing failed with exit code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi
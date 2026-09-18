#!/bin/bash

echo "Starting service data enrichment pipeline..."

# Validate required environment variables
if [ -z "$service_id" ] || [ -z "$name" ] || [ -z "$description" ]; then
  echo "[ERROR] Missing required environment variables: service_id, name, or description"
  exit 1
fi

PYTHON_SCRIPT="/app/src/intent_data_enrichment/main_enrichment.py"

echo "[INFO] Service ID: $service_id"
echo "[INFO] Service Name: $name"

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

# Install required packages (minimal for Phase 1)
echo "[PACKAGES] Installing required packages..."

"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "httpx>=0.27.0" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "pydantic>=2.11.7" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "qdrant-client>=1.15.1" || exit 1
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "requests>=2.32.5" || exit 1

echo "[PACKAGES] All packages installed successfully"

# Set Python path
export PYTHONPATH="/app:/app/src:/app/src/intent_data_enrichment:$PYTHONPATH"

# Verify Python script exists
[ ! -f "$PYTHON_SCRIPT" ] && { echo "[ERROR] Python script not found at $PYTHON_SCRIPT"; exit 1; }

echo "[FOUND] Python script at: $PYTHON_SCRIPT"

# Run enrichment script with arguments
echo "[STARTING] Service enrichment processing..."

# URL decode function using Python
url_decode() {
    python3 -c "import sys; from urllib.parse import unquote; print(unquote(sys.argv[1]))" "$1"
}

# Write JSON arrays to temporary files to avoid bash parsing issues
# Arrays are URL-encoded from Ruuter, need to decode them
TEMP_DIR=$(mktemp -d)
EXAMPLES_FILE="$TEMP_DIR/examples.json"
ENTITIES_FILE="$TEMP_DIR/entities.json"

if [ -n "$examples" ]; then
    url_decode "$examples" > "$EXAMPLES_FILE"
fi

if [ -n "$entities" ]; then
    url_decode "$entities" > "$ENTITIES_FILE"
fi

# Build Python command arguments array
PYTHON_ARGS=(
    "$PYTHON_SCRIPT"
    --service-id "$service_id"
    --name "$name"
    --description "$description"
)

# Add optional fields
[ -n "$ruuter_type" ] && PYTHON_ARGS+=(--ruuter-type "$ruuter_type")
[ -n "$current_state" ] && PYTHON_ARGS+=(--current-state "$current_state")
[ -n "$is_common" ] && PYTHON_ARGS+=(--is-common "$is_common")
[ -n "$examples" ] && PYTHON_ARGS+=(--examples-file "$EXAMPLES_FILE")
[ -n "$entities" ] && PYTHON_ARGS+=(--entities-file "$ENTITIES_FILE")

# Execute Python script directly (no eval to avoid parsing issues)
python3 -u "${PYTHON_ARGS[@]}" 2>&1
PYTHON_EXIT_CODE=$?

# Cleanup temporary files
rm -rf "$TEMP_DIR"

# Handle exit codes
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Service enrichment completed successfully"
    exit 0
else
    echo "[ERROR] Service enrichment failed with exit code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi

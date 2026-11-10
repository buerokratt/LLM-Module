#!/bin/bash
#
# API Tool indexer job (run by CronManager: api_tool_indexer.yml -> index_endpoint).
#
# Result reporting:
#   - Only when attempt_no is set (request body "attemptNo", optional, passed
#     through Ruuter (api-tools/index.yml) and CronManager (allowedEnvs)), the
#     EXIT trap sends exactly one result callback, on exit for any reason:
#       {"entityType": "ENDPOINT", "entityId": <endpoint_id>, "attemptNo": <attempt_no>,
#        "status": "SUCCESS" | "FAILED", "failureReason": null | <string>}
#     SUCCESS only when indexing finished (Qdrant write done). On failure,
#     failureReason is the exact error (with the failed step), read from the
#     result file that main_indexer.py writes (--result-file). If Python never
#     ran, it is the reason this script stopped, or the last log lines.
#   - No attempt_no: indexing runs normally, no callback is sent.
#   - The callback goes to our ruuter-private: POST {RAG_SEARCH_RUUTER_PRIVATE}/indexing/callback
#     (Ruuter.private/rag-search/POST/indexing/callback.yml, no auth: indexing/.guard allows all),
#     which validates the body and forwards it to the Service Module:
#     {RAG_SEARCH_RUUTER_PUBLIC_INTERNAL_SERVICE}/llm/index-callback.
#     A failed forward returns a 5xx (a bare 500 in this Ruuter build), so curl retries it.
#     Callback failures are logged only; they never change the job's exit code.
#   - JOB_ID is generated here for log tracking only. It is not sent in the callback.
#
# Notes:
#   - CronManager starts this script with an empty environment. Only the vars in
#     allowedEnvs (api_tool_indexer.yml) are set; config comes from constants.ini.
#   - CronManager logs stdout only, so everything here is echoed to stdout.

echo "Starting API Tool Indexer pipeline..."

PYTHON_SCRIPT="/app/src/api_tool_indexer/main_indexer.py"
CALLBACK_PATH="/indexing/callback"
ENTITY_TYPE="ENDPOINT"
LOG_TAIL_LINES=20
MAX_ERROR_CHARS=2000

TEMP_DIR=$(mktemp -d)
JOB_ID=$(cat /proc/sys/kernel/random/uuid 2>/dev/null || date +%s%N)
PARAMS_FILE="$TEMP_DIR/params.json"
RESULT_FILE="$TEMP_DIR/result.json"
LOG_FILE="$TEMP_DIR/indexer.log"
FAIL_REASON=""

# Python used for building callback payloads. CronManager starts this script
# with an empty environment, and the image has Python only inside the venv,
# so plain `python3` is not found before the venv is activated.
VENV_PATH="/app/python_virtual_env"
PY_BIN="$VENV_PATH/bin/python3"
[ -x "$PY_BIN" ] || PY_BIN=$(command -v python3 || command -v python || echo python3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Stop the job with a clear reason (used as the error in the callback).
fail() {
    FAIL_REASON="$1"
    echo "[ERROR] $1"
    exit 1
}

# Build the callback JSON. Python is used so all values are escaped correctly.
# Args: status(SUCCESS|FAILED) entity_type entity_id attempt_no result_file fallback_error max_chars
build_payload() {
    "$PY_BIN" -c '
import json, sys
status, entity_type, entity_id, attempt_no, result_file, fallback_error, max_chars = sys.argv[1:8]
payload = {
    "entityType": entity_type,
    "entityId": entity_id,
    "attemptNo": int(attempt_no) if attempt_no.isdigit() else attempt_no,
    "status": status,
    "failureReason": None,
}
if status == "FAILED":
    step, error = None, None
    try:
        with open(result_file, encoding="utf-8") as f:
            result = json.load(f)
        step, error = result.get("step"), result.get("error")
    except Exception:
        pass
    error = error or fallback_error or "Indexing failed (no error details available)"
    if step:
        error = f"Failed at {step}: {error}"
    payload["failureReason"] = error[: int(max_chars)]
print(json.dumps(payload, ensure_ascii=False))
' "$@"
}

# POST the result. Retries 3 times on timeout / connection refused / 5xx
# (curl's own transient-error rules, with backoff). A 4xx is not retried.
send_callback() {
    local status="$1"
    local fallback_error="${2:-}"

    # The callback is only wanted when the Service Module sent an attemptNo.
    if [ -z "$attempt_no" ]; then
        echo "[CALLBACK] job=$JOB_ID No attempt_no, callback not requested ($status)"
        return 0
    fi
    if [ -z "$RAG_SEARCH_RUUTER_PRIVATE" ]; then
        echo "[CALLBACK] job=$JOB_ID ERROR: RAG_SEARCH_RUUTER_PRIVATE not set, skipping $status callback"
        return 0
    fi
    if [ -z "$endpoint_id" ]; then
        echo "[CALLBACK] job=$JOB_ID ERROR: endpoint_id not set, skipping $status callback"
        return 0
    fi

    local callback_url="${RAG_SEARCH_RUUTER_PRIVATE}${CALLBACK_PATH}"
    local payload http_code
    payload=$(build_payload "$status" "$ENTITY_TYPE" "$endpoint_id" "$attempt_no" "$RESULT_FILE" \
        "$fallback_error" "$MAX_ERROR_CHARS" 2>&1) || {
        echo "[CALLBACK] job=$JOB_ID Failed to build payload with $PY_BIN: $payload"
        return 0
    }
    echo "[CALLBACK] job=$JOB_ID Sending to $callback_url: $payload"
    http_code=$(curl -sS -o /dev/null -w '%{http_code}' \
        --max-time 10 --retry 3 --retry-connrefused \
        -X POST -H "Content-Type: application/json" \
        --data "$payload" \
        "$callback_url")
    local curl_exit=$?

    if [ $curl_exit -eq 0 ] && [ "${http_code:0:1}" = "2" ]; then
        echo "[CALLBACK] job=$JOB_ID Result sent (status=$status, HTTP $http_code)"
    else
        echo "[CALLBACK] job=$JOB_ID Result callback failed (curl exit $curl_exit, HTTP ${http_code:-none})"
    fi
    return 0
}

# Runs on every exit: decide the final status, send it, clean up.
on_exit() {
    local exit_code=$?
    trap - EXIT

    if [ $exit_code -eq 0 ]; then
        send_callback "SUCCESS"
    else
        local fallback="$FAIL_REASON"
        if [ -z "$fallback" ] && [ -s "$LOG_FILE" ]; then
            fallback=$(tail -n "$LOG_TAIL_LINES" "$LOG_FILE")
        fi
        [ -z "$fallback" ] && fallback="Indexer script exited with code $exit_code"
        send_callback "FAILED" "$fallback"
    fi

    rm -rf "$TEMP_DIR"
    exit $exit_code
}

trap on_exit EXIT
# Make sure the EXIT trap also runs when the job is stopped with a signal.
# (SIGKILL cannot be caught; the Service Module timeout covers that case.)
trap 'FAIL_REASON="Job interrupted (SIGTERM)"; exit 143' TERM
trap 'FAIL_REASON="Job interrupted (SIGINT)"; exit 130' INT

# ---------------------------------------------------------------------------
# Config first, so the callback URL (RAG_SEARCH_RUUTER_PRIVATE)
# is known before anything can fail.
# Also exports RAG_SEARCH_CONSTANTS so Python resolves the same constants.ini.
# (If the constants file is missing, load_constants.sh exits 1 and no callback
# can be sent; that case is covered by the Service Module timeout.)
# ---------------------------------------------------------------------------
source /app/scripts/load_constants.sh

# Validate required environment variables
if [ -z "$endpoint_id" ] || [ -z "$name" ] || [ -z "$description" ] || [ -z "$url" ]; then
    fail "Missing required environment variables: endpoint_id, name, description, or url"
fi

echo "[INFO] Job ID: $JOB_ID"
echo "[INFO] Endpoint ID: $endpoint_id"
echo "[INFO] Endpoint Name: $name"
echo "[INFO] Attempt No: ${attempt_no:-<not set, no callback>}"

# Install uv if not found
UV_BIN="/root/.local/bin/uv"
if [ ! -f "$UV_BIN" ]; then
    echo "[UV] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "Failed to install uv"
fi

# Activate Python virtual environment
echo "[VENV] Activating virtual environment at: $VENV_PATH"
source "$VENV_PATH/bin/activate" || fail "Failed to activate virtual environment at $VENV_PATH"

# Install required packages
echo "[PACKAGES] Installing required packages..."
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "httpx>=0.27.0" || fail "Failed to install httpx"
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "pydantic>=2.11.7" || fail "Failed to install pydantic"
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "qdrant-client>=1.15.1" || fail "Failed to install qdrant-client"
"$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "requests>=2.32.5" || fail "Failed to install requests"

echo "[PACKAGES] All packages installed successfully"

# Set Python path
export PYTHONPATH="/app:/app/src:/app/src/api_tool_indexer:$PYTHONPATH"

# Verify Python script exists
[ ! -f "$PYTHON_SCRIPT" ] && fail "Python script not found at $PYTHON_SCRIPT"

echo "[FOUND] Python script at: $PYTHON_SCRIPT"

# Run indexing script with arguments
echo "[STARTING] API Tool indexing processing..."

# URL decode function using Python
url_decode() {
    python3 -c "import sys; from urllib.parse import unquote; print(unquote(sys.argv[1]))" "$1"
}

# Write JSON arrays to temporary files to avoid bash parsing issues
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
    --result-file "$RESULT_FILE"
)

# Add params file if provided
[ -n "$params" ] && PYTHON_ARGS+=(--params-file "$PARAMS_FILE")
[ -n "$attempt_no" ] && PYTHON_ARGS+=(--attempt-no "$attempt_no")

# Execute Python script. Output goes to stdout (CronManager log) and to
# LOG_FILE (used as the error when no result file was written).
python3 -u "${PYTHON_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

# Handle exit codes (the EXIT trap sends the final callback and cleans up)
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] API Tool indexing completed successfully"
    exit 0
else
    echo "[ERROR] API Tool indexing failed with exit code: $PYTHON_EXIT_CODE"
    exit $PYTHON_EXIT_CODE
fi

#!/bin/bash

# Vault Secrets Storage Script
# This script stores LLM and embedding credentials in HashiCorp Vault

set -e  # Exit on any error

# Configuration
# Resolve Vault Agent URL:
# 1. Use vaultAgentUrl env var if set (from container env or CronManager request)
# 2. Auto-detect Kubernetes via KUBERNETES_SERVICE_HOST (injected by kubelet, cannot be disabled)
# 3. Auto-detect Kubernetes via service account token (mounted by default in every pod)
# 4. Fallback to Docker Compose hostname
if [ -n "$vaultAgentUrl" ]; then
    VAULT_ADDR="$vaultAgentUrl"
elif [ -n "$KUBERNETES_SERVICE_HOST" ] || [ -f "/var/run/secrets/kubernetes.io/serviceaccount/token" ]; then
    VAULT_ADDR="http://localhost:8203"
else
    VAULT_ADDR="http://vault-agent-cron:8203"
fi

echo "DEBUG: VAULT_ADDR=$VAULT_ADDR vaultAgentUrl=$vaultAgentUrl KUBERNETES_SERVICE_HOST=$KUBERNETES_SERVICE_HOST"

# Decryption Configuration
PRIVATE_KEY_CACHE=""
PRIVATE_KEY_PATH="secret/encryption/private_key"

# Python decryption script path
DECRYPT_SCRIPT="/app/src/utils/decrypt_vault_secrets.py"

# Virtual environment setup
VENV_PATH="/app/python_virtual_env"
UV_BIN="/root/.local/bin/uv"

# Logging function (writes to stderr to avoid command substitution capture)
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

# Fetch private key from Vault
fetch_private_key() {
    if [ -n "$PRIVATE_KEY_CACHE" ]; then
        # Key already cached
        return 0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Fetching private key from Vault..." >&2
    
    # Convert path for KV v2 API
    local api_path=$(echo "$PRIVATE_KEY_PATH" | sed 's|^secret/|secret/data/|')
    
    # Fetch private key from Vault
    local response=$(curl -s --max-time 10 -w "HTTPSTATUS:%{http_code}" \
        -X GET \
        "$VAULT_ADDR/v1/$api_path")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_code" -ne 200 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to fetch private key from Vault (HTTP $http_code)" >&2
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Response: $body" >&2
        return 1
    fi
    
    # Extract private key from JSON response using jq for proper JSON parsing
    if command -v jq >/dev/null 2>&1; then
        PRIVATE_KEY_CACHE=$(echo "$body" | jq -r '.data.data.key // empty')
    else
        # Fallback to grep/sed if jq not available
        PRIVATE_KEY_CACHE=$(echo "$body" | grep -o '"key":"[^"]*"' | sed 's/"key":"//; s/"$//' | sed 's/\\n/\n/g')
    fi
    
    if [ -z "$PRIVATE_KEY_CACHE" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Private key is empty or could not be extracted" >&2
        return 1
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Private key fetched and cached successfully" >&2
}

# Decrypt RSA-OAEP encrypted value using Python
decrypt_rsa_oaep() {
    local encrypted_base64="$1"
    
    if [ -z "$encrypted_base64" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: decrypt_rsa_oaep called with empty value" >&2
        return 1
    fi
    
    # Ensure private key is fetched
    if ! fetch_private_key; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to fetch private key" >&2
        return 1
    fi
    
    # Call Python script for in-memory decryption
    # Private key passed as argument
    local decrypted_value
    local python_stderr
    python_stderr=$(mktemp)
    decrypted_value=$("$VENV_PATH/bin/python3" "$DECRYPT_SCRIPT" "$encrypted_base64" "$PRIVATE_KEY_CACHE" 2>"$python_stderr")
    local exit_code=$?
    
    # Show Python errors if any
    if [ -s "$python_stderr" ]; then
        cat "$python_stderr" >&2
    fi
    rm -f "$python_stderr"
    
    if [ $exit_code -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Python decryption failed with exit code: $exit_code" >&2
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output: $decrypted_value" >&2
        return 1
    fi
    
    if [ -z "$decrypted_value" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Decrypted value is empty" >&2
        return 1
    fi
    
    # Output decrypted value to stdout only
    echo "$decrypted_value"
}

# Setup Python environment and install dependencies
setup_python_environment() {
    # Check if already setup
    if [ -f "$VENV_PATH/.setup_complete" ]; then
        return 0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up Python environment..."
    
    # Install uv if not found
    if [ ! -f "$UV_BIN" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh || {
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to install uv" >&2
            return 1
        }
    fi
    
    # Activate virtual environment
    if [ ! -d "$VENV_PATH" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Virtual environment not found at $VENV_PATH" >&2
        return 1
    fi
    
    source "$VENV_PATH/bin/activate" || {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to activate virtual environment" >&2
        return 1
    }
    
    # Install cryptography library for RSA-OAEP decryption
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing cryptography library..."
    "$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "cryptography>=46.0.3" 2>&1 || {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to install cryptography" >&2
        return 1
    }
    
    # Install loguru for structured logging
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing loguru library..."
    "$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "loguru>=0.7.3" 2>&1 || {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to install loguru" >&2
        return 1
    }

    # Install requests, required by LokiLogger which decrypt_vault_secrets.py imports
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing requests library..."
    "$UV_BIN" pip install --python "$VENV_PATH/bin/python3" "requests>=2.32" 2>&1 || {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Failed to install requests" >&2
        return 1
    }

    # Mark setup as complete
    touch "$VENV_PATH/.setup_complete"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Python environment setup complete"
    return 0
}

log "=== Starting Vault Secrets Storage ==="
log "Received parameters:"
log "  vaultUuid: $vaultUuid"
log "  llmPlatform: $llmPlatform"
log "  llmModel: $llmModel"
log "  deploymentEnvironment: $deploymentEnvironment"
log "  Vault Address: $VAULT_ADDR"

# Validate required vaultUuid parameter
if [ -z "$vaultUuid" ]; then
    log "ERROR: vaultUuid is required but not provided"
    exit 1
fi

# Redirect stderr to stdout so cron-manager can capture all logs
exec 2>&1

# Setup Python environment once at startup (before any decryption)
if ! setup_python_environment; then
    log "ERROR: Failed to setup Python environment"
    exit 1
fi

# Set Python path
export PYTHONPATH="/app:/app/src:/app/src/vector_indexer:$PYTHONPATH"

# Function to determine platform name
get_platform_name() {
    case "$llmPlatform" in
        "aws") echo "aws_bedrock" ;;
        "azure") echo "azure_openai" ;;
        *) 
            log "ERROR: Unsupported platform: $llmPlatform"
            exit 1
            ;;
    esac
}

# Function to get model name (first element from array)
get_model_name() {
    # Remove brackets and quotes, get first element
    echo "$llmModel" | sed 's/\[//g' | sed 's/\]//g' | sed 's/"//g' | cut -d',' -f1 | xargs
}

# Function to build vault path (uses vaultUuid as stable path terminal)
build_vault_path() {
    local secret_type=$1  # "llm" or "embeddings"
    local platform=$(get_platform_name)
    
    # UUID-based path: no environment in path, swap is DB-only
    echo "secret/$secret_type/connections/$platform/$vaultUuid"
}

# Function to store LLM secrets
store_llm_secrets() {
    local vault_path=$(build_vault_path "llm")
    log "Storing LLM secrets at path: $vault_path"
    
    case "$llmPlatform" in
        "aws")
            store_aws_llm_secrets "$vault_path"
            ;;
        "azure")
            store_azure_llm_secrets "$vault_path"
            ;;
    esac
}

# Function to store embedding secrets
store_embedding_secrets() {
    local vault_path=$(build_vault_path "embeddings")
    log "Storing embedding secrets at path: $vault_path"
    
    case "$embeddingPlatform" in
        "aws")
            store_aws_embedding_secrets "$vault_path"
            ;;
        "azure")
            store_azure_embedding_secrets "$vault_path"
            ;;
        *)
            log "WARNING: Embedding platform '$embeddingPlatform' not supported, skipping embedding secrets"
            ;;
    esac
}

# Function to store AWS LLM secrets
store_aws_llm_secrets() {
    local vault_path=$1
    local model=$(get_model_name)
    
    log "Storing AWS LLM secrets..."
    
    # Decrypt sensitive fields
    local decrypted_access_key=$(decrypt_rsa_oaep "$accessKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_access_key" ]; then
        log "ERROR: Failed to decrypt access key"
        exit 1
    fi
    
    local decrypted_secret_key=$(decrypt_rsa_oaep "$secretKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_secret_key" ]; then
        log "ERROR: Failed to decrypt secret key"
        exit 1
    fi
    
    # Build JSON payload using jq for proper escaping
    local json_payload=$(jq -n \
        --arg conn_id "$vaultUuid" \
        --arg access_key "$decrypted_access_key" \
        --arg secret_key "$decrypted_secret_key" \
        --arg model "$model" \
        '{data: {
            connection_id: $conn_id,
            access_key: $access_key,
            secret_key: $secret_key,
            model: $model,
            tags: "aws,bedrock,\($model)"
        }}')
    
    log "Storing secrets at path: $vault_path"
    
    # Convert path for KV v2 API (secret/path -> secret/data/path)
    local api_path=$(echo "$vault_path" | sed 's|^secret/|secret/data/|')
    log "API URL: $VAULT_ADDR/v1/$api_path"
    
    # Execute HTTP API call
    # No X-Vault-Token header needed - vault agent proxy auto-injects it
    local response=$(curl -s --max-time 10 -w "HTTPSTATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$VAULT_ADDR/v1/$api_path")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        log "AWS LLM secrets stored successfully (HTTP $http_code)"
    else
        log "ERROR: Failed to store AWS LLM secrets (HTTP $http_code)"
        log "Response: $body"
        # Clear sensitive variables before exit
        unset decrypted_access_key decrypted_secret_key json_payload
        exit 1
    fi
    
    # Clear sensitive variables after successful storage
    unset decrypted_access_key decrypted_secret_key json_payload
}

# Function to store Azure LLM secrets
store_azure_llm_secrets() {
    local vault_path=$1
    local model=$(get_model_name)
    
    log "Storing Azure LLM secrets..."
    
    # Decrypt sensitive fields
    local decrypted_api_key=$(decrypt_rsa_oaep "$apiKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_api_key" ]; then
        log "ERROR: Failed to decrypt API key"
        exit 1
    fi
    
    # Build JSON payload using jq for proper escaping
    local json_payload=$(jq -n \
        --arg conn_id "$vaultUuid" \
        --arg endpoint "$targetUrl" \
        --arg api_key "$decrypted_api_key" \
        --arg deploy_name "$deploymentName" \
        --arg model "$model" \
        '{data: {
            connection_id: $conn_id,
            endpoint: $endpoint,
            api_key: $api_key,
            deployment_name: $deploy_name,
            model: $model,
            api_version: "2024-05-01-preview",
            tags: "azure,\($model)"
        }}')
    
    log "Storing secrets at path: $vault_path"
    
    # Convert path for KV v2 API (secret/path -> secret/data/path)
    local api_path=$(echo "$vault_path" | sed 's|^secret/|secret/data/|')
    log "API URL: $VAULT_ADDR/v1/$api_path"
    
    # Execute HTTP API call
    # No X-Vault-Token header needed - vault agent proxy auto-injects it
    local response=$(curl -s --max-time 10 -w "HTTPSTATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$VAULT_ADDR/v1/$api_path")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        log "Azure LLM secrets stored successfully (HTTP $http_code)"
    else
        log "ERROR: Failed to store Azure LLM secrets (HTTP $http_code)"
        log "Response: $body"
        # Clear sensitive variables before exit
        unset decrypted_api_key json_payload
        exit 1
    fi
    
    # Clear sensitive variables after successful storage
    unset decrypted_api_key json_payload
}

# Function to store AWS embedding secrets
store_aws_embedding_secrets() {
    local vault_path=$1
    
    log "Storing AWS embedding secrets..."
    
    # Decrypt sensitive fields
    local decrypted_embedding_access_key=$(decrypt_rsa_oaep "$embeddingAccessKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_embedding_access_key" ]; then
        log "ERROR: Failed to decrypt embedding access key"
        exit 1
    fi
    
    local decrypted_embedding_secret_key=$(decrypt_rsa_oaep "$embeddingSecretKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_embedding_secret_key" ]; then
        log "ERROR: Failed to decrypt embedding secret key"
        exit 1
    fi
    
    # Build JSON payload using jq for proper escaping
    local json_payload=$(jq -n \
        --arg conn_id "$vaultUuid" \
        --arg access_key "$decrypted_embedding_access_key" \
        --arg secret_key "$decrypted_embedding_secret_key" \
        --arg model "$embeddingModel" \
        '{data: {
            connection_id: $conn_id,
            access_key: $access_key,
            secret_key: $secret_key,
            model: $model,
            tags: "aws,bedrock,embedding,\($model)"
        }}')
    
    log "Storing secrets at path: $vault_path"
    
    # Convert path for KV v2 API (secret/path -> secret/data/path)
    local api_path=$(echo "$vault_path" | sed 's|^secret/|secret/data/|')
    log "API URL: $VAULT_ADDR/v1/$api_path"
    
    # Execute HTTP API call
    # No X-Vault-Token header needed - vault agent proxy auto-injects it
    local response=$(curl -s --max-time 10 -w "HTTPSTATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$VAULT_ADDR/v1/$api_path")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        log "AWS embedding secrets stored successfully (HTTP $http_code)"
    else
        log "ERROR: Failed to store AWS embedding secrets (HTTP $http_code)"
        log "Response: $body"
        # Clear sensitive variables before exit
        unset decrypted_embedding_access_key decrypted_embedding_secret_key json_payload
        exit 1
    fi
    
    # Clear sensitive variables after successful storage
    unset decrypted_embedding_access_key decrypted_embedding_secret_key json_payload
}

# Function to store Azure embedding secrets
store_azure_embedding_secrets() {
    local vault_path=$1
    
    log "Storing Azure embedding secrets..."
    
    # Decrypt sensitive fields
    local decrypted_embedding_api_key=$(decrypt_rsa_oaep "$embeddingAzureApiKey")
    if [ $? -ne 0 ] || [ -z "$decrypted_embedding_api_key" ]; then
        log "ERROR: Failed to decrypt embedding API key"
        exit 1
    fi
    
    # Build JSON payload using jq for proper escaping
    local json_payload=$(jq -n \
        --arg conn_id "$vaultUuid" \
        --arg endpoint "$embeddingTargetUri" \
        --arg api_key "$decrypted_embedding_api_key" \
        --arg deploy_name "$embeddingDeploymentName" \
        --arg model "$embeddingModel" \
        '{data: {
            connection_id: $conn_id,
            endpoint: $endpoint,
            api_key: $api_key,
            deployment_name: $deploy_name,
            model: $model,
            api_version: "2024-12-01-preview",
            tags: "azure,embedding,\($model)"
        }}')
    
    log "Storing secrets at path: $vault_path"
    
    # Convert path for KV v2 API (secret/path -> secret/data/path)
    local api_path=$(echo "$vault_path" | sed 's|^secret/|secret/data/|')
    log "API URL: $VAULT_ADDR/v1/$api_path"
    
    # Execute HTTP API call
    # No X-Vault-Token header needed - vault agent proxy auto-injects it
    local response=$(curl -s --max-time 10 -w "HTTPSTATUS:%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$VAULT_ADDR/v1/$api_path")
    
    local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
    local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*$//')
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        log "Azure embedding secrets stored successfully (HTTP $http_code)"
    else
        log "ERROR: Failed to store Azure embedding secrets (HTTP $http_code)"
        log "Response: $body"
        # Clear sensitive variables before exit
        unset decrypted_embedding_api_key json_payload
        exit 1
    fi
    
    # Clear sensitive variables after successful storage
    unset decrypted_embedding_api_key json_payload
}

# Main execution
log "Platform: $(get_platform_name)"
log "Model: $(get_model_name)"

# Store LLM secrets
store_llm_secrets

# Store embedding secrets if embedding platform is provided
if [ -n "$embeddingPlatform" ]; then
    store_embedding_secrets
else
    log "No embedding platform specified, skipping embedding secrets"
fi

# Clear private key from memory
unset PRIVATE_KEY_CACHE

log "=== Vault secrets storage completed successfully ==="
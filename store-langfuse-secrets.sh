#!/bin/sh
set -e

# ============================================================================
# Langfuse Secrets Storage Script for Vault
# ============================================================================
# This script stores Langfuse configuration secrets in HashiCorp Vault.
# Run this script AFTER vault-init.sh has completed successfully.
#
# Prerequisites:
# 1. Vault must be initialized and unsealed
# 2. Environment variables must be set (LANGFUSE_INIT_PROJECT_PUBLIC_KEY, etc.)
# 3. Root token must be available in /vault/file/unseal-keys.json
#
# Usage:
#   ./store-langfuse-secrets.sh
#
# Or with custom values:
#   LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-xxx \
#   LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-xxx \
#   LANGFUSE_HOST=http://langfuse-web:3000 \
#   ./store-langfuse-secrets.sh
# ============================================================================

VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
UNSEAL_KEYS_FILE="/vault/file/unseal-keys.json"

echo "========================================"
echo "Langfuse Secrets Storage Script"
echo "========================================"

# Check if Vault is available
echo "Checking Vault availability..."
if ! wget -q -O- "$VAULT_ADDR/v1/sys/health" >/dev/null 2>&1; then
    echo "Error: Vault is not available at $VAULT_ADDR"
    echo "  Please ensure Vault is running and accessible."
    exit 1
fi
echo "Vault is available"

# Check if Vault is sealed
SEALED=$(wget -q -O- "$VAULT_ADDR/v1/sys/seal-status" | grep -o '"sealed":[^,}]*' | cut -d':' -f2)
if [ "$SEALED" = "true" ]; then
    echo "Error: Vault is sealed"
    echo "  Please unseal Vault first using vault-init.sh or manual unseal process."
    exit 1
fi
echo "Vault is unsealed"

# Get root token
echo "Loading Vault root token..."
if [ ! -f "$UNSEAL_KEYS_FILE" ]; then
    echo "Error: Unseal keys file not found at $UNSEAL_KEYS_FILE"
    echo "  Please run vault-init.sh first to initialize Vault."
    exit 1
fi

ROOT_TOKEN=$(grep -o '"root_token":"[^"]*"' "$UNSEAL_KEYS_FILE" | cut -d':' -f2 | tr -d '"')
if [ -z "$ROOT_TOKEN" ]; then
    echo "Error: Could not extract root token from $UNSEAL_KEYS_FILE"
    exit 1
fi
echo "Root token loaded"

# Check required environment variables
echo "Checking Langfuse environment variables..."
if [ -z "$LANGFUSE_INIT_PROJECT_PUBLIC_KEY" ]; then
    echo "Error: LANGFUSE_INIT_PROJECT_PUBLIC_KEY is not set"
    echo "  Please set this environment variable before running the script."
    echo ""
    echo "  Example:"
    echo "    export LANGFUSE_INIT_PROJECT_PUBLIC_KEY='pk-lf-...'"
    exit 1
fi

if [ -z "$LANGFUSE_INIT_PROJECT_SECRET_KEY" ]; then
    echo "Error: LANGFUSE_INIT_PROJECT_SECRET_KEY is not set"
    echo "  Please set this environment variable before running the script."
    echo ""
    echo "  Example:"
    echo "    export LANGFUSE_INIT_PROJECT_SECRET_KEY='sk-lf-...'"
    exit 1
fi

# Use default host if not specified
LANGFUSE_HOST="${LANGFUSE_HOST:-http://langfuse-web:3000}"

echo "Langfuse environment variables found"
echo "  Public Key: ${LANGFUSE_INIT_PROJECT_PUBLIC_KEY:0:10}..."
echo "  Secret Key: ${LANGFUSE_INIT_PROJECT_SECRET_KEY:0:10}..."
echo "  Host: $LANGFUSE_HOST"

# Update Vault policy to include Langfuse secrets access
echo ""
echo "Updating llm-orchestration-policy to include Langfuse secrets..."
# Preserve the production policy paths (see vault-init.sh) and add Langfuse read.
# This is a full overwrite of the policy, so the existing grants must be repeated.
POLICY='path "secret/data/llm/connections/*" { capabilities = ["read", "list"] }
path "secret/metadata/llm/connections/*" { capabilities = ["read", "list"] }
path "secret/data/embeddings/connections/*" { capabilities = ["read", "list"] }
path "secret/metadata/embeddings/connections/*" { capabilities = ["read", "list"] }
path "secret/data/encryption/*" { capabilities = ["deny"] }
path "secret/data/langfuse/*" { capabilities = ["read"] }
path "secret/metadata/langfuse/*" { capabilities = ["read", "list"] }
path "auth/token/lookup-self" { capabilities = ["read"] }'

# Create JSON without jq (using printf for proper escaping)
POLICY_ESCAPED=$(printf '%s' "$POLICY" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | sed ':a;N;$!ba;s/\n/\\n/g')
POLICY_JSON='{"policy":"'"$POLICY_ESCAPED"'"}'

if wget -q -O- --post-data="$POLICY_JSON" \
    --header="X-Vault-Token: $ROOT_TOKEN" \
    --header='Content-Type: application/json' \
    "$VAULT_ADDR/v1/sys/policies/acl/llm-orchestration-policy" >/dev/null 2>&1; then
    echo "Policy updated successfully"
else
    echo "Error: Failed to update llm-orchestration-policy"
    echo "  Langfuse secrets would be stored but the agent would be denied access."
    exit 1
fi

# Store Langfuse secrets in Vault
echo ""
echo "Storing Langfuse secrets in Vault..."

# Create JSON payload
LANGFUSE_SECRET='{"data":{"public_key":"'"$LANGFUSE_INIT_PROJECT_PUBLIC_KEY"'","secret_key":"'"$LANGFUSE_INIT_PROJECT_SECRET_KEY"'","host":"'"$LANGFUSE_HOST"'"}}'

# Store in Vault
if wget -q -O- --post-data="$LANGFUSE_SECRET" \
    --header="X-Vault-Token: $ROOT_TOKEN" \
    --header='Content-Type: application/json' \
    "$VAULT_ADDR/v1/secret/data/langfuse/config" >/dev/null 2>&1; then
    echo "Langfuse secrets stored successfully"
else
    echo "Error: Failed to store Langfuse secrets"
    exit 1
fi

# Verify secrets were stored
echo ""
echo "Verifying stored secrets..."
VERIFICATION=$(wget -q -O- \
    --header="X-Vault-Token: $ROOT_TOKEN" \
    "$VAULT_ADDR/v1/secret/data/langfuse/config" 2>/dev/null)

if echo "$VERIFICATION" | grep -q '"public_key"'; then
    echo "Secrets verified successfully"
    echo ""
    echo "========================================"
    echo "SUCCESS"
    echo "========================================"
    echo "Langfuse secrets have been stored in Vault at:"
    echo "  Path: secret/data/langfuse/config"
    echo ""
    echo "The LLM Orchestration Service will now be able to:"
    echo "  - Initialize Langfuse client automatically"
    echo "  - Track LLM usage and costs"
    echo "  - Monitor orchestration pipelines"
    echo ""
    echo "Next steps:"
    echo "  1. Restart llm-orchestration-service container (if running)"
    echo "  2. Check logs for 'Langfuse client initialized successfully'"
    echo "========================================"
else
    echo "Warning: Secrets stored but verification failed"
    echo "  The secrets may still be accessible, but verification could not confirm."
fi

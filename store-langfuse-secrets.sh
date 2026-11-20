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
# 3. A token: either VAULT_TOKEN in the environment, or a root token in
#    /vault/file/unseal-keys.json
#
# Usage:
#   ./store-langfuse-secrets.sh
#
# Or with custom values:
#   LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-xxx \
#   LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-xxx \
#   LANGFUSE_HOST=http://langfuse-web:3000 \
#   ./store-langfuse-secrets.sh
#
# Seeding the shared central Vault (no unseal-keys file of ours), per decision
# D5 in VAULT_USAGE_AND_MIGRATION_CONCERNS.md -- this is a one-time manual step
# run by whoever holds a token with write access to the prefixed path:
#   VAULT_ADDR=http://<central-vault>:8200 \
#   VAULT_TOKEN=<token> \
#   VAULT_PATH_PREFIX=llm-module \
#   LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-xxx \
#   LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-xxx \
#   LANGFUSE_HOST=http://langfuse-web:3000 \
#   ./store-langfuse-secrets.sh
# ============================================================================

VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
UNSEAL_KEYS_FILE="/vault/file/unseal-keys.json"

# KV path prefix. Empty for the module-owned Vault; set to "llm-module" when
# seeding the shared central Vault, so the write lands where the orchestration
# policy grants read.
VAULT_PATH_PREFIX="$(echo "${VAULT_PATH_PREFIX:-}" | sed 's|^/*||; s|/*$||')"
if [ -n "$VAULT_PATH_PREFIX" ]; then
    P="$VAULT_PATH_PREFIX/"
else
    P=""
fi

# Optional: a token supplied by the operator, for a central Vault where the
# unseal-keys file is not ours to read. Takes precedence over UNSEAL_KEYS_FILE.
VAULT_TOKEN="${VAULT_TOKEN:-}"

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

# Get a token: an explicitly supplied VAULT_TOKEN wins, otherwise fall back to
# the root token on this Vault's own data volume. Against the central Vault the
# unseal-keys file does not belong to us, so VAULT_TOKEN is the only option.
if [ -n "$VAULT_TOKEN" ]; then
    echo "Using VAULT_TOKEN supplied by the environment"
    ROOT_TOKEN="$VAULT_TOKEN"
else
    echo "Loading Vault root token..."
    if [ ! -f "$UNSEAL_KEYS_FILE" ]; then
        echo "Error: Unseal keys file not found at $UNSEAL_KEYS_FILE"
        echo "  Run vault-init.sh first, or pass VAULT_TOKEN explicitly when"
        echo "  seeding a Vault whose init material is not stored here."
        exit 1
    fi

    ROOT_TOKEN=$(grep -o '"root_token":"[^"]*"' "$UNSEAL_KEYS_FILE" | cut -d':' -f2 | tr -d '"')
    if [ -z "$ROOT_TOKEN" ]; then
        echo "Error: Could not extract root token from $UNSEAL_KEYS_FILE"
        exit 1
    fi
    echo "Root token loaded"
fi

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

# NOTE: this script no longer touches llm-orchestration-policy.
# It used to overwrite the whole policy to add the Langfuse read grant, which
# meant re-asserting every other grant as a side effect. That is now redundant
# (vault-init.sh already grants langfuse read/list) and would be actively wrong
# against the central Vault, where the policy is owned by the platform team and
# we have no permission to rewrite it. This script only writes the secret.

# Store Langfuse secrets in Vault
echo ""
echo "Storing Langfuse secrets in Vault..."

# Create JSON payload
LANGFUSE_SECRET='{"data":{"public_key":"'"$LANGFUSE_INIT_PROJECT_PUBLIC_KEY"'","secret_key":"'"$LANGFUSE_INIT_PROJECT_SECRET_KEY"'","host":"'"$LANGFUSE_HOST"'"}}'

# Store in Vault
if wget -q -O- --post-data="$LANGFUSE_SECRET" \
    --header="X-Vault-Token: $ROOT_TOKEN" \
    --header='Content-Type: application/json' \
    "$VAULT_ADDR/v1/secret/data/${P}langfuse/config" >/dev/null 2>&1; then
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
    "$VAULT_ADDR/v1/secret/data/${P}langfuse/config" 2>/dev/null)

if echo "$VERIFICATION" | grep -q '"public_key"'; then
    echo "Secrets verified successfully"
    echo ""
    echo "========================================"
    echo "SUCCESS"
    echo "========================================"
    echo "Langfuse secrets have been stored in Vault at:"
    echo "  Path: secret/data/${P}langfuse/config"
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

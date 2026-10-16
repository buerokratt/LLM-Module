#!/bin/bash
# Central path contract shared by the real store/delete workflows.
central_vault_configure() {
    VAULT_KV_MOUNT=${VAULT_KV_MOUNT:-secret}
    [[ "$VAULT_KV_MOUNT" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
    [[ -n "$VAULT_PATH_PREFIX" ]] || return 1
    local part
    IFS='/' read -ra parts <<< "$VAULT_PATH_PREFIX"
    [[ "$VAULT_PATH_PREFIX" != /* && "$VAULT_PATH_PREFIX" != */ ]] || return 1
    for part in "${parts[@]}"; do
        [[ "$part" =~ ^[A-Za-z0-9_-]+$ ]] || return 1
    done
}

central_vault_connection_path() {
    local kind=$1 provider=$2 uuid=$3
    [[ "$kind" == llm || "$kind" == embeddings ]] || return 1
    [[ "$provider" == aws_bedrock || "$provider" == azure_openai ]] || return 1
    [[ "$uuid" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || return 1
    printf '%s/%s/%s/connections/%s/%s\n' "$VAULT_KV_MOUNT" "$VAULT_PATH_PREFIX" "$kind" "$provider" "$uuid"
}

central_vault_api_path() {
    local logical=$1 operation=$2
    [[ "$operation" == data || "$operation" == metadata ]] || return 1
    [[ "$logical" == "$VAULT_KV_MOUNT/$VAULT_PATH_PREFIX/"* ]] || return 1
    printf '%s/%s/%s\n' "$VAULT_KV_MOUNT" "$operation" "${logical#"$VAULT_KV_MOUNT/"}"
}

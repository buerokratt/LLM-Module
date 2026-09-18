UPDATE rag_search.llm_connections 
SET 
    connection_status = 'inactive'
WHERE vault_uuid = :vault_uuid::uuid
RETURNING 
    vault_uuid,
    connection_name,
    connection_status,
    used_budget,
    stop_budget_threshold,
    disconnect_on_budget_exceed;
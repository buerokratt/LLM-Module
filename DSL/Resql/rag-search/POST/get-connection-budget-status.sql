SELECT 
    id,
    vault_uuid,
    connection_name,
    connection_status,
    used_budget,
    monthly_budget,
    warn_budget_threshold,
    stop_budget_threshold,
    disconnect_on_budget_exceed,
    environment
FROM rag_search.llm_connections
WHERE vault_uuid = :vault_uuid::uuid
  AND connection_status <> 'deleted';

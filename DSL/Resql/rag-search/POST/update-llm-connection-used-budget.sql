UPDATE rag_search.llm_connections 
SET 
    used_budget = used_budget + :usage
WHERE vault_uuid = :vault_uuid::uuid
RETURNING 
    vault_uuid,
    connection_name,
    monthly_budget,
    used_budget,
    (monthly_budget - used_budget) AS remaining_budget,
    warn_budget_threshold,
    stop_budget_threshold,
    disconnect_on_budget_exceed,
    connection_status,
    (used_budget >= stop_budget_threshold) AS budget_exceeded;
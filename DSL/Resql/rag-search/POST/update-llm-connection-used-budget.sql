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
    CASE
        WHEN stop_budget_threshold = 0 THEN (used_budget >= monthly_budget)
        ELSE (used_budget::DECIMAL / monthly_budget::DECIMAL) >= (stop_budget_threshold::DECIMAL / 100.0)
    END AS budget_exceeded;
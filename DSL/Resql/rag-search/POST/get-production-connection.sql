SELECT 
    id,
    vault_uuid,
    connection_name,
    used_budget,
    monthly_budget,
    warn_budget_threshold,
    stop_budget_threshold,
    environment,
    connection_status,
    created_at,
    llm_platform,
    llm_model,
    embedding_platform,
    embedding_model,
    CASE 
        WHEN used_budget IS NULL OR used_budget = 0 OR (used_budget::DECIMAL / monthly_budget::DECIMAL) < (warn_budget_threshold::DECIMAL / 100.0) THEN 'within_budget'
        WHEN stop_budget_threshold != 0 AND (used_budget::DECIMAL / monthly_budget::DECIMAL) >= (stop_budget_threshold::DECIMAL / 100.0) THEN 'over_budget'
        WHEN stop_budget_threshold = 0 AND (used_budget::DECIMAL / monthly_budget::DECIMAL) >= 1 THEN 'over_budget'
        WHEN (used_budget::DECIMAL / monthly_budget::DECIMAL) >= (warn_budget_threshold::DECIMAL / 100.0) THEN 'close_to_exceed'
        ELSE 'within_budget'
    END AS budget_status
FROM rag_search.llm_connections
WHERE environment = 'production'
ORDER BY created_at DESC
LIMIT 1;

SELECT 
    id,
    vault_uuid,
    connection_name,
    llm_platform,
    llm_model,
    embedding_platform,
    embedding_model,
    monthly_budget,
    warn_budget_threshold,
    stop_budget_threshold,
    disconnect_on_budget_exceed,
    used_budget,
    environment,
    connection_status,
    created_at,
    CEIL(COUNT(*) OVER() / :page_size::DECIMAL) AS totalPages,
    CASE 
        WHEN used_budget IS NULL OR used_budget = 0 OR (used_budget::DECIMAL / monthly_budget::DECIMAL) < (warn_budget_threshold::DECIMAL / 100.0) THEN 'within_budget'
        WHEN stop_budget_threshold != 0 AND (used_budget::DECIMAL / monthly_budget::DECIMAL) >= (stop_budget_threshold::DECIMAL / 100.0) THEN 'over_budget'
        WHEN stop_budget_threshold = 0 AND (used_budget::DECIMAL / monthly_budget::DECIMAL) >= 1 THEN 'over_budget'
        WHEN (used_budget::DECIMAL / monthly_budget::DECIMAL) >= (warn_budget_threshold::DECIMAL / 100.0) THEN 'close_to_exceed'
        ELSE 'within_budget'
    END AS budget_status
FROM rag_search.llm_connections
WHERE connection_status <> 'deleted'
    -- AND environment = 'testing'
    AND (:llm_platform IS NULL OR :llm_platform = '' OR llm_platform = :llm_platform)
    AND (:llm_model IS NULL OR :llm_model = '' OR llm_model = :llm_model)
    AND (:environment IS NULL OR :environment = '' OR environment = :environment)
ORDER BY
    CASE WHEN :sorting = 'connection_name asc' THEN connection_name END ASC,
    CASE WHEN :sorting = 'connection_name desc' THEN connection_name END DESC,
    CASE WHEN :sorting = 'llm_platform asc' THEN llm_platform END ASC,
    CASE WHEN :sorting = 'llm_platform desc' THEN llm_platform END DESC,
    CASE WHEN :sorting = 'llm_model asc' THEN llm_model END ASC,
    CASE WHEN :sorting = 'llm_model desc' THEN llm_model END DESC,
    CASE WHEN :sorting = 'monthly_budget asc' THEN monthly_budget END ASC,
    CASE WHEN :sorting = 'monthly_budget desc' THEN monthly_budget END DESC,
    CASE WHEN :sorting = 'environment asc' THEN environment END ASC,
    CASE WHEN :sorting = 'environment desc' THEN environment END DESC,
    CASE WHEN :sorting = 'created_at asc' THEN created_at END ASC,
    CASE WHEN :sorting = 'created_at desc' THEN created_at END DESC,
    created_at DESC  -- Default fallback sorting
OFFSET ((GREATEST(:page, 1) - 1) * :page_size) LIMIT :page_size;
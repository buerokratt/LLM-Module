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
    used_budget,
    disconnect_on_budget_exceed,
    environment,
    connection_status,
    created_at,
    -- Azure credentials
    deployment_name,
    target_uri,
    api_key,
    -- AWS Bedrock credentials
    secret_key,
    access_key,
    -- Embedding model credentials
    embedding_access_key,
    embedding_secret_key,
    embedding_deployment_name,
    embedding_target_uri,
    embedding_azure_api_key
FROM rag_search.llm_connections
WHERE id = :connection_id
  AND connection_status <> 'deleted';

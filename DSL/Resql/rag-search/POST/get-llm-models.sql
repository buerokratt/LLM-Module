SELECT 
	id,
	platform_id,
    model_key as value,
    model_name as label
FROM rag_search.llm_models 
ORDER BY model_name;
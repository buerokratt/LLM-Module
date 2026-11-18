-- liquibase formatted sql

-- changeset gpt5-models:rag-script-v9-changeset1
-- Add the Azure GPT-5 reasoning models to the selectable model catalogue
INSERT INTO rag_search.llm_models (platform_id, model_key, model_name)
VALUES
    ((SELECT id FROM rag_search.llm_platforms WHERE platform_key = 'azure'), 'gpt-5.2', 'GPT-5.2'),
    ((SELECT id FROM rag_search.llm_platforms WHERE platform_key = 'azure'), 'gpt-5.4', 'GPT-5.4')
ON CONFLICT (platform_id, model_key) DO NOTHING;
-- rollback DELETE FROM rag_search.llm_models WHERE model_key IN ('gpt-5.2', 'gpt-5.4') AND platform_id = (SELECT id FROM rag_search.llm_platforms WHERE platform_key = 'azure');

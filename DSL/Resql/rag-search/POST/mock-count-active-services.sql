-- Count active services for tool classifier
-- Used by Service Workflow to determine search strategy:
-- - If count <= 50: Use all services for LLM context
-- - If count > 50: Use Qdrant semantic search for top 20

SELECT 
    COUNT(*) AS active_service_count
FROM 
    public.services
WHERE 
    current_state = 'active'
    AND discoverable = true;

-- Get all active services for intent detection
-- Used when active_service_count <= 50
-- Returns all service metadata needed for LLM intent detection

SELECT 
    service_id,
    name,
    description,
    ruuter_type,
    slot,
    entities,
    examples,
    structure,
    endpoints
FROM 
    public.services
WHERE 
    current_state = 'active'
    AND discoverable = true
ORDER BY 
    name ASC;

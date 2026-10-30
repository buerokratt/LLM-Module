-- Get specific service by service_id for validation
-- Used after LLM detects intent to validate the service exists and is active
-- Returns all service details needed to trigger the external service call

SELECT 
    id,
    service_id,
    name,
    description,
    ruuter_type,
    current_state,
    slot,
    entities,
    examples,
    structure,
    endpoints,
    created_at,
    updated_at
FROM 
    public.services
WHERE 
    service_id = :serviceId
    AND current_state = 'active'
    AND discoverable = true;

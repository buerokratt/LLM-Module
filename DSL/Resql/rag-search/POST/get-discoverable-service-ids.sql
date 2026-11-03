SELECT
    service_id
FROM
    public.services
WHERE
    current_state = 'active'
    AND discoverable = true
    AND deleted = false;

-- One-time nonce for service-to-service calls to the Service Module (x-ruuter-nonce header).
-- request_nonces lives in the shared database; the Service Module consumes it with use-nonce.
INSERT INTO request_nonces (valid_until) VALUES ( now() + interval '1 day') RETURNING nonce;

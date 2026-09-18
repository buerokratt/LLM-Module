-- liquibase formatted sql

-- changeset uuid-vault-path:rag-script-v8-changeset1
-- Add UUID column to llm_connections for stable Vault secret paths
ALTER TABLE rag_search.llm_connections
    ADD COLUMN IF NOT EXISTS vault_uuid UUID DEFAULT gen_random_uuid();

-- Backfill existing rows with unique UUIDs
UPDATE rag_search.llm_connections SET vault_uuid = gen_random_uuid() WHERE vault_uuid IS NULL;

-- Make it NOT NULL after backfill
ALTER TABLE rag_search.llm_connections ALTER COLUMN vault_uuid SET NOT NULL;

-- Add unique constraint
ALTER TABLE rag_search.llm_connections ADD CONSTRAINT llm_connections_vault_uuid_unique UNIQUE (vault_uuid);
-- rollback ALTER TABLE rag_search.llm_connections DROP CONSTRAINT IF EXISTS llm_connections_vault_uuid_unique; ALTER TABLE rag_search.llm_connections DROP COLUMN IF EXISTS vault_uuid;

-- liquibase formatted sql

-- changeset reset-agency-hash:rag-script-v9-changeset1
-- Issue #805: before the fix, the data/update sync stored the CKB data hash
-- BEFORE the vector indexer ran. When that run failed (e.g. no production LLM
-- connection), the hash stayed stored with an empty knowledge base, and every
-- later sync answered "No sync required". Clearing the hash once makes the next
-- hourly sync run the indexer again.
-- Safe for healthy environments: the indexer's diff detector skips files that
-- were already indexed, so nothing is re-embedded and no LLM calls are made.
UPDATE rag_search.agency_sync SET agency_data_hash = NULL;
-- rollback SELECT 1;

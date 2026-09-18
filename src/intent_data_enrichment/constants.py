"""Constants for data enrichment service."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnrichmentConstants:
    """Constants for enrichment pipeline."""

    # API Configuration
    DEFAULT_API_BASE_URL = "http://llm-orchestration-service:8100"
    DEFAULT_ENVIRONMENT = "production"
    # None → orchestration service resolves the embedding model via the
    # DB-fetched vault UUID (path: embeddings/connections/{provider}/{vault_uuid}).
    DEFAULT_CONNECTION_ID = None

    # Retry Configuration
    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 2  # Exponential backoff base (2^attempt seconds)
    REQUEST_TIMEOUT = 60  # seconds

    # Qdrant Configuration
    COLLECTION_NAME = "intent_collections"
    DEFAULT_QDRANT_HOST = "qdrant"
    DEFAULT_QDRANT_PORT = 6333
    VECTOR_SIZE = 3072  # Azure text-embedding-3-large dimension
    DISTANCE_METRIC = "Cosine"

    # Named Vector Configuration (for hybrid search)
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    # Context Generation
    CONTEXT_TEMPLATE = """<document>
{full_service_info}
</document>

Here is the service intent we want to enrich for better search retrieval:
<service>
Name: {name}
Description: {description}
Examples: {examples}
</service>

Please generate a rich, detailed context that describes this service intent comprehensively for semantic search. 
Include information about:
- What the user wants to accomplish
- Key terms and synonyms
- Related concepts
- Common ways users might express this intent

IMPORTANT: Generate the context in the SAME LANGUAGE as the service description above. If the description is in Estonian, respond in Estonian. If in English, respond in English. If in Russian, respond in Russian.

Answer only with the enriched context and nothing else."""

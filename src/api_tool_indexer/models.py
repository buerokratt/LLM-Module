"""Data models for the API Tool Indexer."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ParamSchema(BaseModel):
    """Schema for a single API endpoint parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(
        ...,
        description="Parameter type: string, date, integer, boolean, number",
    )
    required: bool = Field(..., description="Whether this parameter is required")
    description: str = Field(
        ..., description="Human-readable description of the parameter"
    )


class EndpointData(BaseModel):
    """Raw endpoint data fetched from the mock_endpoints table.

    This is the input to the indexing pipeline.
    """

    endpoint_id: str = Field(..., description="UUID of the endpoint")
    name: str = Field(..., description="Endpoint name (snake_case)")
    description: str = Field(..., description="Human-readable description")
    url: str = Field(..., description="Full URL of the external API")
    method: str = Field(..., description="HTTP method: GET or POST")
    params: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of parameter schemas [{name, type, required, description}]",
    )
    service_id: Optional[str] = Field(
        default=None, description="Optional parent service UUID"
    )
    visibility: str = Field(default="private", description="public or private")
    type: str = Field(default="custom_endpoint", description="Endpoint type")
    cacheable: bool = Field(
        default=True,
        description=(
            "Set False for endpoints returning sensitive/personal data "
            "(e.g. document status). Disables all L1/L2 cache writes."
        ),
    )
    cache_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-endpoint L1 TTL override in seconds. "
            "None = use ATC_CACHE_DEFAULT_TTL_SECONDS."
        ),
    )


class EnrichedEndpoint(BaseModel):
    """Enriched endpoint data ready for storage in Qdrant api_tool_collection.

    Multiple points are stored per endpoint:
    - One 'example' point per example query — embedded from that sentence alone,
      so the vector sits in the correct language region of the embedding space.
    - One 'summary' point — embedded from name + description + full enriched context.

    All points share the same endpoint_id payload field so the searcher can
    deduplicate results back to a single endpoint after retrieval.
    The payload on every point contains all fields needed by the agentic loop
    so no additional DB roundtrip is needed after a semantic match.
    """

    endpoint_id: str = Field(..., description="UUID of the endpoint")
    name: str = Field(..., description="Endpoint name")
    description: str = Field(..., description="Raw description from DB")
    url: str = Field(..., description="Full URL of the external API")
    method: str = Field(..., description="HTTP method: GET or POST")
    params: List[Dict[str, Any]] = Field(
        ...,
        description="Full params schema — stored in payload for direct use by agentic loop",
    )
    enriched_context: str = Field(
        ..., description="LLM-generated rich context for better semantic matching"
    )
    service_id: Optional[str] = Field(default=None, description="Parent service UUID")

    cacheable: bool = Field(
        default=True,
        description=(
            "Propagated from EndpointData. False disables all L1/L2 cache writes "
            "for this endpoint at query time."
        ),
    )
    cache_ttl_seconds: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-endpoint L1 TTL override propagated from EndpointData. "
            "None = use ATC_CACHE_DEFAULT_TTL_SECONDS."
        ),
    )

    # Point type — controls which text was embedded for this point
    point_type: str = Field(
        default="summary",
        description="'example' (individual query text) or 'summary' (full context)",
    )
    example_text: Optional[str] = Field(
        default=None,
        description="The example query string for 'example' points; None for 'summary'",
    )

    # Vector fields (populated by indexer pipeline)
    embedding: List[float] = Field(
        default_factory=list, description="Dense embedding vector (3072-dim)"
    )
    sparse_indices: List[int] = Field(
        default_factory=list, description="Sparse vector indices (BM25)"
    )
    sparse_values: List[float] = Field(
        default_factory=list, description="Sparse vector values (BM25)"
    )


class IndexingResult(BaseModel):
    """Result of a single endpoint indexing operation."""

    success: bool = Field(..., description="Whether indexing succeeded")
    endpoint_id: str = Field(..., description="Endpoint UUID")
    message: str = Field(..., description="Result message")
    error: Optional[str] = Field(default=None, description="Error message if failed")

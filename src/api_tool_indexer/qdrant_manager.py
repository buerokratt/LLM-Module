"""Qdrant manager for api_tool_collection with hybrid search support.
for the api_tool_collection used by the API Tool Calling workflow.
"""

import uuid
from typing import Any, Dict, List, Optional

from src.loki_logger import LokiLogger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from api_tool_indexer.constants import ApiToolIndexerConstants
from api_tool_indexer.models import EnrichedEndpoint

logger = LokiLogger(service_name="api-tool-calling")

# Error messages
_CLIENT_NOT_INITIALIZED = "Qdrant client not initialized"


class ApiToolQdrantManager:
    """Manages Qdrant operations for api_tool_collection with hybrid search.

    Multiple points are stored per endpoint:
    - One 'example' point per example query
    - One 'summary' point for the full combined context
    All points share the same endpoint_id payload field for deduplication.
    """

    def __init__(
        self,
        host: str = ApiToolIndexerConstants.DEFAULT_QDRANT_HOST,
        port: int = ApiToolIndexerConstants.DEFAULT_QDRANT_PORT,
        collection_name: str = ApiToolIndexerConstants.COLLECTION_NAME,
    ) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None

    def connect(self) -> None:
        """Connect to Qdrant."""
        try:
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                timeout=30,
                prefer_grpc=False,
                api_key=None,
            )
            logger.success("Successfully connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def ensure_collection(self) -> None:
        """Ensure api_tool_collection exists with hybrid vector config.

        The collection uses named vectors:
        - 'dense': 3072-dim cosine similarity vectors for semantic matching
        - 'sparse': BM25-style sparse vectors for keyword matching
        """
        if not self.client:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)

        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name in collection_names:
                self._validate_existing_collection()
            else:
                self._create_collection()

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def _validate_existing_collection(self) -> None:
        """Validate that the existing API Tool collection has correct hybrid vector config."""
        if not self.client:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)

        collection_info = self.client.get_collection(self.collection_name)
        vectors_config = collection_info.config.params.vectors

        if isinstance(vectors_config, dict):
            if ApiToolIndexerConstants.DENSE_VECTOR_NAME in vectors_config:
                existing_size = vectors_config[
                    ApiToolIndexerConstants.DENSE_VECTOR_NAME
                ].size
                if existing_size != ApiToolIndexerConstants.VECTOR_SIZE:
                    logger.error(
                        f"Collection '{self.collection_name}' has incompatible vector size: "
                        f"{existing_size} (expected {ApiToolIndexerConstants.VECTOR_SIZE})"
                    )
                    raise RuntimeError(
                        f"Collection '{self.collection_name}' has incompatible vector size: "
                        f"{existing_size} (expected {ApiToolIndexerConstants.VECTOR_SIZE}). "
                        "Delete the collection and re-index all endpoints."
                    )
                logger.info(
                    f"Collection '{self.collection_name}' already exists "
                    f"with correct hybrid vector config (dense: {existing_size}d + sparse)"
                )
            else:
                # Old collection format (unnamed/single vector) — needs migration
                logger.error(
                    f"Collection '{self.collection_name}' exists but uses old single-vector format. "
                    "Migration to named vectors (dense + sparse) required."
                )
                raise RuntimeError(
                    f"Collection '{self.collection_name}' uses old single-vector format. "
                    "Please delete the collection and re-index all endpoints. "
                    f"Delete with: qdrant.client.delete_collection('{self.collection_name}') "
                    "or via Qdrant UI/API."
                )
        elif vectors_config is not None:
            # Direct VectorParams object (old single-vector format)
            logger.error(
                f"Collection '{self.collection_name}' exists but uses old single-vector format."
            )
            raise RuntimeError(
                f"Collection '{self.collection_name}' uses old single-vector format. "
                "Please delete the collection and re-index all endpoints. "
                f"Delete with: qdrant.client.delete_collection('{self.collection_name}') "
                "or via Qdrant UI/API."
            )
        else:
            logger.error(
                f"Collection '{self.collection_name}' exists but vector config cannot be determined."
            )
            raise RuntimeError(
                f"Collection '{self.collection_name}' vector config cannot be determined. "
                "Manual intervention required."
            )

    def _create_collection(self) -> None:
        """Create api_tool_collection with hybrid vector configuration (dense + sparse)."""
        if not self.client:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)

        logger.info(
            f"Creating collection '{self.collection_name}' "
            f"with hybrid vectors (dense: {ApiToolIndexerConstants.VECTOR_SIZE}d + sparse)"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                ApiToolIndexerConstants.DENSE_VECTOR_NAME: VectorParams(
                    size=ApiToolIndexerConstants.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                ApiToolIndexerConstants.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.success(f"Collection '{self.collection_name}' created successfully")

    def delete_endpoint_points(self, endpoint_id: str) -> bool:
        """Delete all Qdrant points for a given endpoint.

        Uses a payload filter on 'endpoint_id' to remove all example and summary
        points belonging to this endpoint. Called before re-indexing to ensure
        idempotent updates, and when an endpoint is removed from the DB.

        Args:
            endpoint_id: UUID of the endpoint whose points should be deleted.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if not self.client:
                raise RuntimeError(_CLIENT_NOT_INITIALIZED)

            logger.info(f"Deleting all points for endpoint '{endpoint_id}' from Qdrant")
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="endpoint_id",
                                match=MatchValue(value=endpoint_id),
                            )
                        ]
                    )
                ),
            )
            logger.success(
                f"Successfully deleted all points for endpoint '{endpoint_id}'"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete points for endpoint '{endpoint_id}': {e}")
            return False

    def upsert_endpoint_points(self, enriched_points: List[EnrichedEndpoint]) -> bool:
        """Upsert multiple enriched endpoint points to Qdrant.

        Each point gets a deterministic UUID derived from endpoint_id + index so
        upserts are idempotent. All points carry the full endpoint payload so no
        additional DB roundtrip is needed after a semantic match.

        Payload fields stored on every point:
            endpoint_id, name, description, url, method, params,
            enriched_context, service_id, point_type, cacheable,
            cache_ttl_seconds, example_text (example only)

        Args:
            enriched_points: List of EnrichedEndpoint instances (examples + summary).

        Returns:
            True if all points upserted successfully, False otherwise.
        """
        try:
            if not self.client:
                raise RuntimeError(_CLIENT_NOT_INITIALIZED)

            if not enriched_points:
                logger.warning("No points to upsert")
                return True

            endpoint_id = enriched_points[0].endpoint_id
            logger.info(
                f"Upserting {len(enriched_points)} points for endpoint '{endpoint_id}'"
            )

            points: List[PointStruct] = []
            for idx, enriched in enumerate(enriched_points):
                # Deterministic UUID: same input always produces the same point ID
                point_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, f"{enriched.endpoint_id}_{idx}")
                )

                payload: Dict[str, Any] = {
                    "endpoint_id": enriched.endpoint_id,
                    "name": enriched.name,
                    "description": enriched.description,
                    "url": enriched.url,
                    "method": enriched.method,
                    "params": enriched.params,
                    "enriched_context": enriched.enriched_context,
                    "service_id": enriched.service_id,
                    "point_type": enriched.point_type,
                    "cacheable": enriched.cacheable,
                    "cache_ttl_seconds": enriched.cache_ttl_seconds,
                }
                if enriched.example_text is not None:
                    payload["example_text"] = enriched.example_text

                vectors: Dict[str, Any] = {
                    ApiToolIndexerConstants.DENSE_VECTOR_NAME: enriched.embedding,
                }
                if enriched.sparse_indices:
                    vectors[ApiToolIndexerConstants.SPARSE_VECTOR_NAME] = SparseVector(
                        indices=enriched.sparse_indices,
                        values=enriched.sparse_values,
                    )

                points.append(PointStruct(id=point_id, vector=vectors, payload=payload))

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            n_examples = sum(1 for p in enriched_points if p.point_type == "example")
            n_summary = sum(1 for p in enriched_points if p.point_type == "summary")
            logger.success(
                f"Successfully upserted {len(points)} points for endpoint '{endpoint_id}' "
                f"({n_examples} example + {n_summary} summary)"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to upsert points for endpoint "
                f"'{enriched_points[0].endpoint_id if enriched_points else '?'}': {e}"
            )
            return False

    def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            logger.info("Closing Qdrant connection")
            self.client.close()

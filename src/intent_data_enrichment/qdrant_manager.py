"""Qdrant manager for intent collections with hybrid search support."""

import uuid
from typing import Optional, List
from src.loki_logger import LokiLogger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)

from intent_data_enrichment.constants import EnrichmentConstants
from intent_data_enrichment.models import EnrichedService

# Initialize Loki logger
logger = LokiLogger(service_name="intent-qdrant-manager")

# Error messages
_CLIENT_NOT_INITIALIZED = "Qdrant client not initialized"


class QdrantManager:
    """Manages Qdrant operations for intent collections with hybrid search."""

    def __init__(
        self,
        host: str = EnrichmentConstants.DEFAULT_QDRANT_HOST,
        port: int = EnrichmentConstants.DEFAULT_QDRANT_PORT,
        collection_name: str = EnrichmentConstants.COLLECTION_NAME,
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
        """Ensure the intent_collections collection exists with hybrid vector config.

        The collection uses named vectors:
        - 'dense': 3072-dim cosine similarity vectors for semantic matching
        - 'sparse': BM25-style sparse vectors for keyword matching
        """
        try:
            if not self.client:
                raise RuntimeError(_CLIENT_NOT_INITIALIZED)

            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name in collection_names:
                collection_info = self.client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors

                # Check if collection has the expected named vector configuration
                if isinstance(vectors_config, dict):
                    if EnrichmentConstants.DENSE_VECTOR_NAME in vectors_config:
                        existing_vector_size = vectors_config[
                            EnrichmentConstants.DENSE_VECTOR_NAME
                        ].size
                        if existing_vector_size != EnrichmentConstants.VECTOR_SIZE:
                            logger.error(
                                f"Collection '{self.collection_name}' has incompatible vector size: "
                                f"{existing_vector_size} (expected {EnrichmentConstants.VECTOR_SIZE})"
                            )
                            raise RuntimeError(
                                f"Collection '{self.collection_name}' has incompatible vector size "
                                f"({existing_vector_size} vs expected {EnrichmentConstants.VECTOR_SIZE}). "
                                "To recreate the collection, manually delete it first using: "
                                f"qdrant.client.delete_collection('{self.collection_name}') or via Qdrant UI/API."
                            )
                        logger.info(
                            f"Collection '{self.collection_name}' already exists "
                            f"with correct hybrid vector config (dense: {existing_vector_size}d + sparse)"
                        )
                    else:
                        # Old collection format (unnamed/single vector) — needs migration
                        logger.error(
                            f"Collection '{self.collection_name}' exists but uses old single-vector format. "
                            "Migration to named vectors (dense + sparse) required."
                        )
                        raise RuntimeError(
                            f"Collection '{self.collection_name}' uses old single-vector format. "
                            "Please delete the collection and re-index all services. "
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
                        "Please delete the collection and re-index all services. "
                        f"Delete with: qdrant.client.delete_collection('{self.collection_name}') "
                        "or via Qdrant UI/API."
                    )
                else:
                    logger.error(
                        f"Collection '{self.collection_name}' exists but vector config cannot be determined"
                    )
                    raise RuntimeError(
                        f"Collection '{self.collection_name}' exists but vector config cannot be determined. "
                        "Manual intervention required."
                    )
            else:
                self._create_collection()

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def _create_collection(self) -> None:
        """Create the collection with hybrid vector configuration (dense + sparse)."""
        if not self.client:
            raise RuntimeError(_CLIENT_NOT_INITIALIZED)

        logger.info(
            f"Creating collection '{self.collection_name}' "
            f"with hybrid vectors (dense: {EnrichmentConstants.VECTOR_SIZE}d + sparse)"
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                EnrichmentConstants.DENSE_VECTOR_NAME: VectorParams(
                    size=EnrichmentConstants.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                EnrichmentConstants.SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.success(f"Collection '{self.collection_name}' created successfully")

    def delete_service_points(self, service_id: str) -> bool:
        """Delete all points belonging to a service.

        Used before re-indexing to ensure idempotent updates, and when
        a service is deactivated.

        Args:
            service_id: Service identifier to delete all points for

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.client:
                raise RuntimeError(_CLIENT_NOT_INITIALIZED)

            logger.info(
                f"Deleting existing points for service '{service_id}' from Qdrant"
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="service_id",
                                match=MatchValue(value=service_id),
                            )
                        ]
                    )
                ),
            )

            logger.success(f"Successfully deleted points for service '{service_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to delete points for service '{service_id}': {e}")
            return False

    def upsert_service_points(self, enriched_points: List[EnrichedService]) -> bool:
        """Upsert multiple enriched service points to Qdrant.

        Each point contains both dense and sparse vectors for hybrid search.
        Points are identified by a deterministic UUID based on service_id + point_index.

        Args:
            enriched_points: List of EnrichedService instances (examples + summary)

        Returns:
            True if all points upserted successfully, False otherwise
        """
        try:
            if not self.client:
                raise RuntimeError(_CLIENT_NOT_INITIALIZED)

            if not enriched_points:
                logger.warning("No points to upsert")
                return True

            service_id = enriched_points[0].id
            logger.info(
                f"Upserting {len(enriched_points)} points for service '{service_id}'"
            )

            from typing import Any, Dict

            points: List[PointStruct] = []
            for idx, enriched_service in enumerate(enriched_points):
                # Deterministic UUID based on service_id + index
                point_id_source = f"{enriched_service.id}_{idx}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id_source))

                # Prepare payload
                payload = {
                    "service_id": enriched_service.id,
                    "name": enriched_service.name,
                    "description": enriched_service.description,
                    "examples": enriched_service.examples,
                    "entities": enriched_service.entities,
                    "context": enriched_service.context,
                    "point_type": enriched_service.point_type,
                    "is_common": enriched_service.is_common,
                }

                # Add example_text for example points
                if enriched_service.example_text:
                    payload["example_text"] = enriched_service.example_text

                # Build named vectors (dense always, sparse if present)
                vectors: Dict[str, Any] = {
                    EnrichmentConstants.DENSE_VECTOR_NAME: enriched_service.embedding,
                }
                if enriched_service.sparse_indices:
                    vectors[EnrichmentConstants.SPARSE_VECTOR_NAME] = SparseVector(
                        indices=enriched_service.sparse_indices,
                        values=enriched_service.sparse_values,
                    )

                point = PointStruct(
                    id=point_id,
                    vector=vectors,
                    payload=payload,
                )

                points.append(point)

            # Bulk upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            logger.success(
                f"Successfully upserted {len(points)} points for service '{service_id}' "
                f"({sum(1 for p in enriched_points if p.point_type == 'example')} examples + "
                f"{sum(1 for p in enriched_points if p.point_type == 'summary')} summary)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to upsert service points: {e}")
            return False

    def upsert_service(self, enriched_service: EnrichedService) -> bool:
        """Upsert a single enriched service to Qdrant.

        Backward-compatible wrapper that delegates to upsert_service_points.

        Args:
            enriched_service: EnrichedService instance

        Returns:
            True if successful, False otherwise
        """
        return self.upsert_service_points([enriched_service])

    def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            logger.info("Closing Qdrant connection")
            self.client.close()

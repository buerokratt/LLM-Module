"""Qdrant vector database manager for storing contextual chunks."""

from typing import List, Dict, Any, Optional
from loki_logger import LokiLogger
import httpx
import uuid
from typing_extensions import Self

from vector_indexer.config.config_loader import VectorIndexerConfig
from vector_indexer.models import ContextualChunk

logger = LokiLogger(service_name="qdrant-manager")


class QdrantOperationError(Exception):
    """Custom exception for Qdrant operations."""

    pass


class QdrantManager:
    """Manages Qdrant vector database operations for contextual chunks."""

    def __init__(self, config: VectorIndexerConfig) -> None:
        self.config = config
        self.qdrant_url: str = getattr(config, "qdrant_url", "http://localhost:6333")
        self.client = httpx.AsyncClient(timeout=30.0)

        # Collection configurations based on embedding models
        self.collections_config: Dict[str, Dict[str, Any]] = {
            "contextual_chunks_azure": {
                "vector_size": 3072,  # text-embedding-3-large
                "distance": "Cosine",
                "models": ["text-embedding-3-large", "text-embedding-ada-002"],
            },
            "contextual_chunks_aws": {
                "vector_size": 1024,  # amazon.titan-embed-text-v2:0
                "distance": "Cosine",
                "models": [
                    "amazon.titan-embed-text-v2:0",
                    "amazon.titan-embed-text-v1",
                ],
            },
        }

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit."""
        await self.client.aclose()

    async def ensure_collections_exist(self) -> None:
        """Create collections if they don't exist."""
        logger.info("Ensuring Qdrant collections exist")

        for collection_name, config in self.collections_config.items():
            await self._create_collection_if_not_exists(collection_name, config)

    async def _create_collection_if_not_exists(
        self, collection_name: str, collection_config: Dict[str, Any]
    ) -> None:
        """Create a collection if it doesn't exist."""

        try:
            # Check if collection exists
            response = await self.client.get(
                f"{self.qdrant_url}/collections/{collection_name}"
            )

            if response.status_code == 200:
                logger.debug(f"Collection {collection_name} already exists")
                return
            elif response.status_code == 404:
                logger.info(f"Creating collection {collection_name}")

                # Create collection
                create_payload = {
                    "vectors": {
                        "size": collection_config["vector_size"],
                        "distance": collection_config["distance"],
                    },
                    "optimizers_config": {"default_segment_number": 2},
                    "replication_factor": 1,
                }

                response = await self.client.put(
                    f"{self.qdrant_url}/collections/{collection_name}",
                    json=create_payload,
                )

                if response.status_code in [200, 201]:
                    logger.info(f"Successfully created collection {collection_name}")
                else:
                    logger.error(
                        f"Failed to create collection {collection_name}: {response.status_code} {response.text}"
                    )

            else:
                logger.error(
                    f"Unexpected response checking collection {collection_name}: {response.status_code}"
                )

        except Exception as e:
            logger.error(f"Error ensuring collection {collection_name} exists: {e}")
            raise

    async def store_chunks(self, chunks: List[ContextualChunk]) -> None:
        """
        Store contextual chunks in appropriate Qdrant collection.

        Args:
            chunks: List of contextual chunks to store
        """
        if not chunks:
            logger.warning("No chunks to store")
            return

        logger.info(f"Storing {len(chunks)} chunks in Qdrant")

        # Group chunks by embedding model
        chunks_by_model: Dict[str, List[ContextualChunk]] = {}
        for chunk in chunks:
            model_key = self._get_collection_for_model(chunk.embedding_model)
            if model_key not in chunks_by_model:
                chunks_by_model[model_key] = []
            chunks_by_model[model_key].append(chunk)

        # Store chunks in appropriate collections
        for collection_name, chunk_list in chunks_by_model.items():
            await self._store_chunks_in_collection(collection_name, chunk_list)

    async def _store_chunks_in_collection(
        self, collection_name: str, chunks: List[ContextualChunk]
    ) -> None:
        """Store chunks in specific collection."""

        logger.debug(f"Storing {len(chunks)} chunks in collection {collection_name}")

        # Prepare points for upsert
        points: List[Dict[str, Any]] = []
        for chunk in chunks:
            if not chunk.embedding:
                logger.warning(f"Skipping chunk {chunk.chunk_id} - no embedding")
                continue

            # Convert chunk_id to UUID for Qdrant compatibility
            # Qdrant requires point IDs to be either integers or UUIDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            point = {
                "id": point_id,
                "vector": chunk.embedding,
                "payload": self._create_chunk_payload(chunk),
            }
            points.append(point)

        if not points:
            logger.warning(f"No valid points to store in {collection_name}")
            return

        try:
            # Upsert points in batches to avoid request size limits
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]

                upsert_payload = {"points": batch}

                # Log batch summary only (avoid massive logs)
                logger.debug(
                    f"Storing batch {i // batch_size + 1}: {len(batch)} points to {collection_name}"
                )

                response = await self.client.put(
                    f"{self.qdrant_url}/collections/{collection_name}/points",
                    json=upsert_payload,
                )

                if response.status_code in [200, 201]:
                    logger.debug(
                        f"Successfully stored batch {i // batch_size + 1} in {collection_name}"
                    )
                else:
                    logger.error(
                        f"Failed to store batch in {collection_name}: {response.status_code} {response.text}"
                    )
                    raise QdrantOperationError(
                        f"Qdrant upsert failed: {response.status_code}"
                    )

            logger.info(
                f"Successfully stored {len(points)} chunks in {collection_name}"
            )

        except Exception as e:
            logger.error(f"Error storing chunks in {collection_name}: {e}")
            raise

    def _create_chunk_payload(self, chunk: ContextualChunk) -> Dict[str, Any]:
        """Create payload for Qdrant point."""

        return {
            # Core identifiers
            "chunk_id": chunk.chunk_id,
            "document_hash": chunk.document_hash,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            # Content
            "original_content": chunk.original_content,
            "contextual_content": chunk.contextual_content,
            "context_only": chunk.context,
            # Embedding info
            "embedding_model": chunk.embedding_model,
            "vector_dimensions": chunk.vector_dimensions,
            # Document metadata
            "document_url": chunk.source_url,
            "dataset_collection": chunk.dataset_collection,
            # Processing metadata
            "processing_timestamp": chunk.processing_timestamp.isoformat(),
            "tokens_count": chunk.tokens_count,
            # Additional metadata from source
            "file_type": chunk.metadata.get("file_type"),
            "created_at": chunk.metadata.get("created_at"),
        }

    def _get_collection_for_model(self, embedding_model: Optional[str]) -> str:
        """Determine which collection to use based on embedding model."""

        if not embedding_model:
            logger.warning("No embedding model specified, using azure collection")
            return "contextual_chunks_azure"

        model_lower = embedding_model.lower()

        # Check Azure models
        for azure_model in self.collections_config["contextual_chunks_azure"]["models"]:
            if azure_model.lower() in model_lower:
                return "contextual_chunks_azure"

        # Check AWS models
        for aws_model in self.collections_config["contextual_chunks_aws"]["models"]:
            if aws_model.lower() in model_lower:
                return "contextual_chunks_aws"

        # Default to Azure if no match
        logger.warning(
            f"Unknown embedding model {embedding_model}, using azure collection"
        )
        return "contextual_chunks_azure"

    async def get_collection_info(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get information about a collection."""

        try:
            response = await self.client.get(
                f"{self.qdrant_url}/collections/{collection_name}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(
                    f"Failed to get collection info for {collection_name}: {response.status_code}"
                )
                return None

        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None

    async def count_points(self, collection_name: str) -> int:
        """Count points in a collection."""

        try:
            response = await self.client.get(
                f"{self.qdrant_url}/collections/{collection_name}"
            )

            if response.status_code == 200:
                collection_info = response.json()
                return collection_info.get("result", {}).get("points_count", 0)
            else:
                logger.error(
                    f"Failed to get point count for {collection_name}: {response.status_code}"
                )
                return 0

        except Exception as e:
            logger.error(f"Error counting points in {collection_name}: {e}")
            return 0

    async def delete_chunks_by_document_hash(
        self, collection_name: str, document_hash: str
    ) -> int:
        """
        Delete all chunks associated with a specific document hash.

        Args:
            collection_name: Name of the Qdrant collection
            document_hash: SHA256 hash of the document to delete chunks for

        Returns:
            Number of chunks deleted (estimated as 1 if deletion successful, 0 if nothing to delete)

        Raises:
            QdrantOperationError: If deletion fails
        """
        try:
            logger.info(
                f"🗑️  Attempting to delete chunks for document: {document_hash[:12]}... from {collection_name}"
            )

            # Step 1: Check if chunks exist BEFORE deletion (for accurate reporting)
            pre_check_payload = {
                "filter": {
                    "must": [
                        {"key": "document_hash", "match": {"value": document_hash}}
                    ]
                },
                "limit": 100,  # Get up to 100 to count
                "with_payload": False,
                "with_vector": False,
            }

            pre_check_response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                json=pre_check_payload,
            )

            chunks_found_before = 0
            if pre_check_response.status_code == 200:
                pre_check_data = pre_check_response.json()
                chunks_found_before = len(
                    pre_check_data.get("result", {}).get("points", [])
                )
                logger.info(f"🔍 Found {chunks_found_before} chunks to delete")
            else:
                logger.warning(
                    f"⚠️  Pre-check query failed with status {pre_check_response.status_code}"
                )

            # Step 2: Execute deletion using filter
            delete_payload = {
                "filter": {
                    "must": [
                        {"key": "document_hash", "match": {"value": document_hash}}
                    ]
                }
            }

            logger.debug(f"🔍 Executing delete with filter: {delete_payload}")

            response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/delete",
                json=delete_payload,
            )

            if response.status_code in [200, 201]:
                result = response.json()

                if result.get("status") == "ok":
                    # Step 3: Verify deletion by checking if chunks still exist
                    verify_payload = {
                        "filter": {
                            "must": [
                                {
                                    "key": "document_hash",
                                    "match": {"value": document_hash},
                                }
                            ]
                        },
                        "limit": 1,
                        "with_payload": False,
                        "with_vector": False,
                    }

                    verify_response = await self.client.post(
                        f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                        json=verify_payload,
                    )

                    if verify_response.status_code == 200:
                        verify_data = verify_response.json()
                        remaining_chunks = len(
                            verify_data.get("result", {}).get("points", [])
                        )

                        if remaining_chunks == 0:
                            if chunks_found_before > 0:
                                logger.info(
                                    f"✅ Successfully deleted {chunks_found_before} chunk(s) from {collection_name}"
                                )
                                return chunks_found_before
                            else:
                                logger.info(
                                    f"ℹ️  No chunks found for document {document_hash[:12]}... in {collection_name}"
                                )
                                return 0
                        else:
                            logger.error(
                                f"❌ Delete verification failed: {remaining_chunks} chunk(s) still exist!"
                            )
                            return 0
                    else:
                        # Verification query failed, but delete was accepted
                        # Assume success based on pre-check count
                        if chunks_found_before > 0:
                            logger.warning(
                                f"⚠️  Delete succeeded but verification failed - assuming {chunks_found_before} chunks deleted"
                            )
                            return chunks_found_before
                        else:
                            logger.info(
                                f"ℹ️  Delete completed (verification query failed, found {chunks_found_before} before)"
                            )
                            return 0
                else:
                    raise QdrantOperationError(
                        f"Qdrant delete returned error status: {result}"
                    )
            else:
                raise QdrantOperationError(
                    f"Delete request failed with HTTP {response.status_code}: {response.text}"
                )

        except QdrantOperationError:
            # Re-raise QdrantOperationError as-is
            raise
        except Exception as e:
            logger.error(
                f"❌ Failed to delete chunks for document {document_hash[:12]}...: {e}"
            )
            raise QdrantOperationError(
                f"Failed to delete chunks by document hash: {str(e)}"
            ) from e

    async def delete_chunks_by_file_path(
        self, collection_name: str, file_path: str
    ) -> int:
        """
        Delete all chunks associated with a specific file path (fallback method).

        Args:
            collection_name: Name of the Qdrant collection
            file_path: Original file path to delete chunks for

        Returns:
            Number of chunks deleted

        Raises:
            QdrantOperationError: If deletion fails
        """
        try:
            logger.info(
                f"🗑️  Deleting chunks for file path: {file_path} from {collection_name}"
            )

            # Count chunks first
            scroll_payload = {
                "filter": {
                    "must": [{"key": "document_url", "match": {"value": file_path}}]
                },
                "limit": 1000,
                "with_payload": False,
                "with_vector": False,
            }

            scroll_response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                json=scroll_payload,
            )

            chunks_to_delete = 0
            if scroll_response.status_code == 200:
                scroll_data = scroll_response.json()
                chunks_to_delete = len(scroll_data.get("result", {}).get("points", []))

            # Delete chunks using filter
            delete_payload = {
                "filter": {
                    "must": [{"key": "document_url", "match": {"value": file_path}}]
                }
            }

            response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/delete",
                json=delete_payload,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                if result.get("status") == "ok":
                    logger.info(
                        f"✅ Successfully deleted {chunks_to_delete} chunks for file {file_path}"
                    )
                    return chunks_to_delete
                else:
                    raise QdrantOperationError(f"Qdrant returned error: {result}")
            else:
                raise QdrantOperationError(
                    f"HTTP {response.status_code}: {response.text}"
                )

        except Exception as e:
            logger.error(f"Failed to delete chunks for file {file_path}: {e}")
            raise QdrantOperationError(
                f"Failed to delete chunks by file path: {str(e)}"
            ) from e

    async def get_chunks_for_document(
        self, collection_name: str, document_hash: str
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks associated with a specific document hash.

        Args:
            collection_name: Name of the Qdrant collection
            document_hash: SHA256 hash of the document

        Returns:
            List of chunk records with their metadata
        """
        try:
            scroll_payload = {
                "filter": {
                    "must": [
                        {"key": "document_hash", "match": {"value": document_hash}}
                    ]
                },
                "limit": 1000,
                "with_payload": True,
                "with_vector": False,
            }

            response = await self.client.post(
                f"{self.qdrant_url}/collections/{collection_name}/points/scroll",
                json=scroll_payload,
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("result", {}).get("points", [])
            else:
                logger.warning(
                    f"Failed to get chunks for document {document_hash[:12]}...: HTTP {response.status_code}"
                )
                return []

        except Exception as e:
            logger.warning(
                f"Error getting chunks for document {document_hash[:12]}...: {e}"
            )
            return []

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection (for cleanup/testing)."""

        try:
            response = await self.client.delete(
                f"{self.qdrant_url}/collections/{collection_name}"
            )

            if response.status_code in [200, 404]:  # 404 means already deleted
                logger.info(f"Successfully deleted collection {collection_name}")
                return True
            else:
                logger.error(
                    f"Failed to delete collection {collection_name}: {response.status_code}"
                )
                return False

        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

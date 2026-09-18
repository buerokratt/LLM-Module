"""
Qdrant Contextual Search Client

Handles semantic search against contextual chunk collections using
existing contextual embeddings created by the vector indexer.
"""

from typing import List, Dict, Any, Optional, Protocol, TYPE_CHECKING
from src.loki_logger import LokiLogger

import asyncio

from contextual_retrieval.contextual_retrieval_api_client import get_http_client_manager
from contextual_retrieval.error_handler import SecureErrorHandler
from contextual_retrieval.constants import (
    HttpStatusConstants,
    ErrorContextConstants,
    LoggingConstants,
)
from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Initialize Loki logger
logger = LokiLogger(service_name="qdrant-search")

if TYPE_CHECKING:
    from contextual_retrieval.contextual_retrieval_api_client import HTTPClientManager


class LLMServiceProtocol(Protocol):
    """Protocol defining the interface required from LLM service for embedding operations."""

    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """Create embeddings for text inputs using the configured embedding model.

        Args:
            texts: List of text strings to embed
            environment: Environment for model resolution
            connection_id: Optional connection ID for service selection
            batch_size: Number of texts to process in each batch

        Returns:
            Dictionary containing embeddings list and metadata
        """
        ...


class QdrantContextualSearch:
    """Semantic search client for contextual chunk collections."""

    def __init__(
        self, qdrant_url: str, config: Optional["ContextualRetrievalConfig"] = None
    ) -> None:
        self.qdrant_url = qdrant_url
        self._config = config if config is not None else ConfigLoader.load_config()
        self._http_client_manager = None

    async def _get_http_client_manager(self) -> "HTTPClientManager":
        """Get the HTTP client manager instance."""
        if self._http_client_manager is None:
            self._http_client_manager = await get_http_client_manager()
        return self._http_client_manager

    async def search_contextual_embeddings(
        self,
        query_embedding: List[float],
        collections: List[str],
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search contextual embeddings across specified collections.

        Args:
            query_embedding: Query vector embedding
            collections: List of collection names to search
            limit: Number of results per collection (uses config default if None)
            score_threshold: Minimum similarity score (uses config default if None)

        Returns:
            List of chunks with similarity scores and metadata
        """
        # Use configuration defaults if not specified
        if limit is None:
            limit = self._config.search.topk_semantic
        if score_threshold is None:
            score_threshold = self._config.search.score_threshold

        return await self.search_contextual_embeddings_direct(
            query_embedding, collections, limit, score_threshold
        )

    async def search_contextual_embeddings_direct(
        self,
        query_embedding: List[float],
        collections: List[str],
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search contextual embeddings using pre-computed embedding vector.
        This method skips embedding generation and directly performs vector search.

        Args:
            query_embedding: Pre-computed query vector embedding
            collections: List of collection names to search
            limit: Number of results per collection (uses config default if None)
            score_threshold: Minimum similarity score (uses config default if None)

        Returns:
            List of chunks with similarity scores and metadata
        """
        # Use configuration defaults if not specified
        if limit is None:
            limit = self._config.search.topk_semantic
        if score_threshold is None:
            score_threshold = self._config.search.score_threshold

        all_results: List[Dict[str, Any]] = []

        # Search collections in parallel for performance
        search_tasks = [
            self._search_single_collection(
                collection_name, query_embedding, limit, score_threshold
            )
            for collection_name in collections
        ]

        try:
            collection_results = await asyncio.gather(
                *search_tasks, return_exceptions=True
            )

            for i, result in enumerate(collection_results):
                if isinstance(result, BaseException):
                    logger.warning(
                        f"Search failed for collection {collections[i]}: {result}"
                    )
                    continue

                if result:
                    # Tag results with source collection - type checked above
                    for chunk in result:
                        chunk["search_type"] = "semantic"
                    all_results.extend(result)

            # Sort by similarity score (descending)
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

            logger.info(
                f"Semantic search found {len(all_results)} chunks across {len(collections)} collections"
            )

            # Detailed results at DEBUG level (filters based on log level config)
            logger.debug("=== SEMANTIC SEARCH RESULTS BREAKDOWN ===")
            for i, chunk in enumerate(all_results[:10]):  # Show top 10 results
                content_preview = (
                    (chunk.get("original_content", "")[:150] + "...")
                    if len(chunk.get("original_content", "")) > 150
                    else chunk.get("original_content", "")
                )
                logger.debug(
                    f"  Rank {i + 1}: score={chunk['score']:.4f}, collection={chunk.get('source_collection', 'unknown')}, id={chunk['chunk_id']}"
                )
                logger.debug(f"           content: '{content_preview}'")
            logger.debug("=== END SEMANTIC SEARCH RESULTS ===")

            return all_results

        except Exception as e:
            logger.error(f"Contextual semantic search failed: {e}")
            return []

    async def _search_single_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        limit: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Search a single collection for contextual chunks."""
        try:
            search_payload = {
                "vector": query_embedding,
                "limit": limit,
                "score_threshold": score_threshold,
                "with_payload": True,
            }

            client_manager = await self._get_http_client_manager()
            client = await client_manager.get_client()

            search_url = (
                f"{self.qdrant_url}/collections/{collection_name}/points/search"
            )
            search_headers = {"Content-Type": "application/json"}

            response = await client.post(
                search_url, json=search_payload, headers=search_headers
            )

            if response.status_code != HttpStatusConstants.OK:
                SecureErrorHandler.log_secure_error(
                    error=Exception(
                        f"Qdrant search failed with status {response.status_code}"
                    ),
                    context=ErrorContextConstants.PROVIDER_DETECTION,
                    request_url=search_url,
                    request_headers=search_headers,
                    level=LoggingConstants.ERROR,
                )
                return []

            search_results = response.json()
            points = search_results.get("result", [])

            # Transform Qdrant results to our format
            chunks: List[Dict[str, Any]] = []
            for point in points:
                payload = point.get("payload", {})
                chunk = {
                    "id": point.get("id"),
                    "score": float(point.get("score", 0)),
                    "chunk_id": payload.get("chunk_id"),
                    "document_hash": payload.get("document_hash"),
                    "original_content": payload.get("original_content", ""),
                    "contextual_content": payload.get("contextual_content", ""),
                    "context_only": payload.get("context_only", ""),
                    "embedding_model": payload.get("embedding_model"),
                    "document_url": payload.get("document_url"),
                    "chunk_index": payload.get("chunk_index", 0),
                    "total_chunks": payload.get("total_chunks", 1),
                    "tokens_count": payload.get("tokens_count", 0),
                    "processing_timestamp": payload.get("processing_timestamp"),
                    "metadata": payload,  # Full payload for additional context
                }
                chunks.append(chunk)

            # Debug logging for retrieved chunks
            logger.info(f"Found {len(chunks)} chunks in {collection_name}")
            for i, chunk in enumerate(chunks):
                content_preview = (
                    (chunk.get("original_content", "")[:100] + "...")
                    if len(chunk.get("original_content", "")) > 100
                    else chunk.get("original_content", "")
                )
                logger.info(
                    f"  Chunk {i + 1}/{len(chunks)}: score={chunk['score']:.4f}, id={chunk['chunk_id']}, content='{content_preview}'"
                )

            return chunks

        except Exception as e:
            SecureErrorHandler.log_secure_error(
                error=e,
                context="qdrant_search_collection",
                request_url=f"{self.qdrant_url}/collections/{collection_name}",
                level="error",
            )
            return []

    def get_embedding_for_query_with_service(
        self,
        query: str,
        llm_service: LLMServiceProtocol,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> Optional[List[float]]:
        """
        Get embedding for query using provided LLMOrchestrationService instance.
        This avoids creating new service instances and enables connection pooling.

        Args:
            query: Text to embed
            llm_service: Pre-initialized LLMOrchestrationService instance
            environment: Environment for model resolution
            connection_id: Optional connection ID

        Returns:
            Query embedding vector or None if failed
        """
        try:
            # Use provided service instance for connection pooling
            embedding_result = llm_service.create_embeddings_for_indexer(
                texts=[query],
                environment=environment,
                connection_id=connection_id,
                batch_size=self._config.performance.batch_size,
            )

            embeddings = embedding_result.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            else:
                logger.error("No embedding returned for query")
                return None

        except Exception as e:
            logger.error(f"Failed to get query embedding with provided service: {e}")
            return None

    def get_embeddings_for_queries_batch(
        self,
        queries: List[str],
        llm_service: LLMServiceProtocol,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> Optional[List[List[float]]]:
        """
        Get embeddings for multiple queries in a single batch call.
        This significantly reduces API latency by batching all queries together.

        Args:
            queries: List of query texts to embed
            llm_service: Pre-initialized LLMOrchestrationService instance
            environment: Environment for model resolution
            connection_id: Optional connection ID

        Returns:
            List of query embedding vectors in same order as input queries, or None if failed
        """
        if not queries:
            logger.warning("Empty queries list provided for batch embedding")
            return []

        try:
            logger.info(f"Creating batch embeddings for {len(queries)} queries")

            # Use provided service instance for batch embedding
            embedding_result = llm_service.create_embeddings_for_indexer(
                texts=queries,
                environment=environment,
                connection_id=connection_id,
                batch_size=len(queries),  # Process all queries in single batch
            )

            embeddings = embedding_result.get("embeddings", [])
            if embeddings and len(embeddings) == len(queries):
                logger.info(f"Successfully created {len(embeddings)} batch embeddings")
                return embeddings
            else:
                logger.error(
                    f"Batch embedding mismatch: expected {len(queries)}, got {len(embeddings) if embeddings else 0}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to get batch embeddings: {e}")
            return None

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client_manager:
            await self._http_client_manager.close()

    # Context Manager Protocol
    async def __aenter__(self) -> "QdrantContextualSearch":
        """Async context manager entry."""
        # Ensure HTTP client manager is initialized
        await self._get_http_client_manager()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit with cleanup."""
        await self.close()

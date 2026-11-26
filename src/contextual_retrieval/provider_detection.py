"""
Dynamic Provider Detection for Contextual Retrieval

Intelligently selects optimal Qdrant collections based on:
- Environment's default embedding model
- Collection health and availability
- No hardcoded weights or preferences
"""

import asyncio
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from src.loki_logger import LokiLogger

from contextual_retrieval.contextual_retrieval_api_client import get_http_client_manager
from contextual_retrieval.error_handler import SecureErrorHandler
from contextual_retrieval.constants import (
    HttpClientConstants,
    HttpStatusConstants,
    ErrorContextConstants,
    LoggingConstants,
)
from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Initialize Loki logger
logger = LokiLogger(service_name="provider-detection")

if TYPE_CHECKING:
    from contextual_retrieval.contextual_retrieval_api_client import HTTPClientManager


class DynamicProviderDetection:
    """Dynamic collection selection without hardcoded preferences."""

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

    async def detect_optimal_collections(
        self, environment: str, connection_id: Optional[str] = None
    ) -> List[str]:
        """
        Dynamically detect optimal collections based on environment config.

        Args:
            environment: Environment (production, development, test)
            connection_id: Optional connection ID

        Returns:
            List of collection names to search
        """
        try:
            # Get default embedding model from environment
            default_model = self._get_default_embedding_model(
                environment, connection_id
            )

            if default_model:
                logger.info(f"Detected default embedding model: {default_model}")
                collections = self._map_model_to_collections(default_model)
            else:
                logger.warning("Could not detect default model, using all collections")
                collections = [
                    self._config.collections.azure_collection,
                    self._config.collections.aws_collection,
                ]

            # Verify collections are healthy
            healthy_collections = await self._filter_healthy_collections(collections)

            if not healthy_collections:
                # No fallback to the other collection: it holds vectors of
                # another size (3072 vs 1024), so a query embedded with this
                # connection's model cannot be compared with them.
                await self._log_no_matching_embeddings(default_model, collections)
                return []

            logger.info(f"Selected collections: {healthy_collections}")
            return healthy_collections

        except Exception as e:
            logger.error(f"Provider detection failed: {e}")
            # Safe fallback - search all collections
            return [
                self._config.collections.azure_collection,
                self._config.collections.aws_collection,
            ]

    async def _log_no_matching_embeddings(
        self, model: Optional[str], searched: List[str]
    ) -> None:
        """Explain why nothing can be searched: empty knowledge base, or the
        connection's embedding model differs from the one that built it."""
        all_collections = [
            self._config.collections.azure_collection,
            self._config.collections.aws_collection,
        ]
        others = await self._filter_healthy_collections(
            [name for name in all_collections if name not in searched],
            include_unchecked=False,
        )
        if others:
            logger.warning(
                f"No knowledge base embeddings for embedding model {model or 'unknown'} "
                f"(collection {', '.join(searched)} is empty). The knowledge base is "
                f"built in {', '.join(others)} with another embedding model, which this "
                "connection cannot search. Use the knowledge base's embedding model for "
                "this connection, or wait until the re-index to this model has finished. "
                "Retrieval skipped."
            )
        else:
            logger.warning(
                "Knowledge base is empty (no embeddings in any collection) - nothing "
                "to retrieve until the indexer has run"
            )

    def _get_default_embedding_model(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[str]:
        """Get default embedding model from existing infrastructure."""
        try:
            # Import here to avoid circular dependencies
            from src.llm_orchestrator_config.config.loader import ConfigurationLoader

            config_loader = ConfigurationLoader()
            provider_name, model_name = config_loader.resolve_embedding_model(
                environment, connection_id
            )

            return f"{provider_name}/{model_name}"

        except Exception as e:
            logger.warning(f"Could not resolve default embedding model: {e}")
            return None

    def _map_model_to_collections(self, model: str) -> List[str]:
        """Map embedding model to appropriate collections."""
        model_lower = model.lower()

        # Azure OpenAI models
        if any(
            keyword in model_lower
            for keyword in self._config.collections.azure_keywords
        ):
            return [self._config.collections.azure_collection]

        # AWS Bedrock models
        elif any(
            keyword in model_lower for keyword in self._config.collections.aws_keywords
        ):
            return [self._config.collections.aws_collection]

        # Unknown model - search both collections
        else:
            logger.info(f"Unknown model {model}, searching all collections")
            return [
                self._config.collections.azure_collection,
                self._config.collections.aws_collection,
            ]

    async def _filter_healthy_collections(
        self, collections: List[str], include_unchecked: bool = True
    ) -> List[str]:
        """Keep the collections that hold embeddings.

        A collection Qdrant reports as empty or missing is dropped. One that
        could not be checked (408, 5xx, dropped connection) is kept when
        ``include_unchecked``: a transient Qdrant error must not turn into
        "the knowledge base is empty" and an out-of-scope answer. Searching it
        is harmless - a failed search returns no chunks.
        """
        healthy: List[str] = []

        for collection_name in collections:
            points_count = await self._collection_points_count(collection_name)

            if points_count is None:
                if include_unchecked:
                    logger.warning(
                        f"Could not check collection {collection_name} - "
                        "searching it anyway"
                    )
                    healthy.append(collection_name)
            elif points_count > 0:
                healthy.append(collection_name)
                logger.debug(f"Collection {collection_name}: {points_count} points")
            else:
                logger.warning(f"Collection {collection_name} is empty")

        return healthy

    async def _collection_points_count(self, collection_name: str) -> Optional[int]:
        """Points in a collection: 0 if it does not exist, None if Qdrant could
        not answer after HEALTH_CHECK_ATTEMPTS tries."""
        health_check_url = f"{self.qdrant_url}/collections/{collection_name}"

        for attempt in range(1, HttpClientConstants.HEALTH_CHECK_ATTEMPTS + 1):
            try:
                client_manager = await self._get_http_client_manager()
                client = await client_manager.get_client()
                response = await client.get(health_check_url)

                if response.status_code == HttpStatusConstants.OK:
                    return int(
                        response.json().get("result", {}).get("points_count") or 0
                    )
                if response.status_code == HttpStatusConstants.NOT_FOUND:
                    logger.debug(f"Collection {collection_name} does not exist")
                    return 0

                error: Exception = Exception(
                    f"Collection not accessible with status {response.status_code}"
                )
            except Exception as e:
                error = e

            SecureErrorHandler.log_secure_error(
                error=error,
                context=ErrorContextConstants.PROVIDER_HEALTH_CHECK,
                request_url=health_check_url,
                level=LoggingConstants.WARNING,
            )
            if attempt < HttpClientConstants.HEALTH_CHECK_ATTEMPTS:
                await asyncio.sleep(HttpClientConstants.HEALTH_CHECK_RETRY_DELAY)

        return None

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all contextual collections."""
        stats: Dict[str, Any] = {}
        collections = [
            self._config.collections.azure_collection,
            self._config.collections.aws_collection,
        ]

        for collection_name in collections:
            try:
                client_manager = await self._get_http_client_manager()
                client = await client_manager.get_client()
                response = await client.get(
                    f"{self.qdrant_url}/collections/{collection_name}"
                )

                if response.status_code == HttpStatusConstants.OK:
                    collection_info = response.json()
                    stats[collection_name] = {
                        "points_count": collection_info.get("result", {}).get(
                            "points_count", 0
                        ),
                        "status": collection_info.get("result", {}).get(
                            "status", "unknown"
                        ),
                    }
                else:
                    stats[collection_name] = {
                        "points_count": 0,
                        "status": "unavailable",
                    }

            except Exception as e:
                logger.warning(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = {"points_count": 0, "status": "error"}

        return stats

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client_manager:
            await self._http_client_manager.close()

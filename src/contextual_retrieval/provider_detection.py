"""
Dynamic Provider Detection for Contextual Retrieval

Intelligently selects optimal Qdrant collections based on:
- Environment's default embedding model
- Collection health and availability
- No hardcoded weights or preferences
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING
from src.loki_logger import LokiLogger

from contextual_retrieval.contextual_retrieval_api_client import get_http_client_manager
from contextual_retrieval.error_handler import SecureErrorHandler
from contextual_retrieval.constants import (
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
                logger.warning("No healthy collections found, falling back to all")
                return [
                    self._config.collections.azure_collection,
                    self._config.collections.aws_collection,
                ]

            logger.info(f"Selected collections: {healthy_collections}")
            return healthy_collections

        except Exception as e:
            logger.error(f"Provider detection failed: {e}")
            # Safe fallback - search all collections
            return [
                self._config.collections.azure_collection,
                self._config.collections.aws_collection,
            ]

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

    async def _filter_healthy_collections(self, collections: List[str]) -> List[str]:
        """Filter collections to only healthy/available ones."""
        healthy: List[str] = []

        for collection_name in collections:
            try:
                client_manager = await self._get_http_client_manager()
                client = await client_manager.get_client()

                health_check_url = f"{self.qdrant_url}/collections/{collection_name}"
                response = await client.get(health_check_url)

                if response.status_code == HttpStatusConstants.OK:
                    collection_info = response.json()
                    points_count = collection_info.get("result", {}).get(
                        "points_count", 0
                    )

                    if points_count > 0:
                        healthy.append(collection_name)
                        logger.debug(
                            f"Collection {collection_name}: {points_count} points"
                        )
                    else:
                        logger.warning(f"Collection {collection_name} is empty")
                else:
                    SecureErrorHandler.log_secure_error(
                        error=Exception(
                            f"Collection not accessible with status {response.status_code}"
                        ),
                        context=ErrorContextConstants.PROVIDER_HEALTH_CHECK,
                        request_url=health_check_url,
                        level=LoggingConstants.WARNING,
                    )

            except Exception as e:
                SecureErrorHandler.log_secure_error(
                    error=e,
                    context=ErrorContextConstants.PROVIDER_HEALTH_CHECK,
                    request_url=f"{self.qdrant_url}/collections/{collection_name}",
                    level=LoggingConstants.WARNING,
                )

        return healthy

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

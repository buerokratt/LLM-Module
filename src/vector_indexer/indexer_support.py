"""Vector-indexer support: embeddings and contextual-chunk generation.

Extracted from LLMOrchestrationService. This is a self-contained collaborator
used by the `/embeddings`, `/generate-context`, and `/embedding-models` API
routes (called by the external vector-indexer service), and reused as the
general-purpose embedding provider injected into RAG retrieval and the tool
classifier (`llm_service` / `orchestration_service` / `embedding_service`
duck-typed references). It has no dependency on the RAG/streaming pipeline.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import ContextGenerationRequest
from llm_orchestrator_config.llm_manager import LLMManager
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    PRODUCTION_DEPLOYMENT_ENVIRONMENT,
)

if TYPE_CHECKING:
    from src.llm_orchestrator_config.embedding_manager import EmbeddingManager
    from src.llm_orchestrator_config.context_manager import (
        ContextGenerationManager,
    )
    from src.llm_orchestrator_config.config.loader import ConfigurationLoader

logger = LokiLogger(service_name="llm-orchestration-service")


class IndexerSupport:
    """Embeddings and context-generation support for the vector indexer.

    Completely isolated from the RAG pipeline and uses lazy initialization so
    construction doesn't interfere with the main orchestration flow.
    """

    def __init__(self) -> None:
        self._embedding_manager: Optional["EmbeddingManager"] = None
        self._context_manager: Optional["ContextGenerationManager"] = None
        self._config_loader: Optional["ConfigurationLoader"] = None

    @observe(name="create_embeddings_for_indexer", as_type="span")
    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Create embeddings for vector indexer using vault-driven model resolution.

        Args:
            texts: List of texts to embed
            environment: Environment (production, development, testing)
            connection_id: Optional connection ID for dev/test environments
            batch_size: Batch size for processing

        Returns:
            Dictionary with embeddings and metadata
        """
        logger.info(
            f"Creating embeddings for vector indexer: {len(texts)} texts in {environment} environment"
        )

        try:
            embedding_manager = self._get_embedding_manager()

            return embedding_manager.create_embeddings(
                texts=texts,
                environment=environment,
                connection_id=connection_id,
                batch_size=batch_size,
            )
        except Exception as e:
            logger.error(f"Vector indexer embedding creation failed: {e}")
            raise

    def generate_context_for_chunks(
        self, request: ContextGenerationRequest
    ) -> Dict[str, Any]:
        """Generate context for chunks using Anthropic methodology.

        Args:
            request: Context generation request with document and chunk prompts

        Returns:
            Dictionary with generated context and metadata
        """
        logger.info("Generating context for chunks using Anthropic methodology")

        try:
            context_manager = self._get_context_manager()

            return context_manager.generate_context_with_caching(request)
        except Exception as e:
            logger.error(f"Vector indexer context generation failed: {e}")
            raise

    def get_available_embedding_models_for_indexer(
        self, environment: str = PRODUCTION_DEPLOYMENT_ENVIRONMENT
    ) -> Dict[str, Any]:
        """Get available embedding models for vector indexer.

        Args:
            environment: Environment (production, development, testing)

        Returns:
            Dictionary with available models and default model info
        """
        try:
            embedding_manager = self._get_embedding_manager()
            config_loader = self._get_config_loader()

            available_models: List[str] = embedding_manager.get_available_models(
                environment
            )

            try:
                provider_name, model_name = config_loader.resolve_embedding_model(
                    environment
                )
                default_model: str = f"{provider_name}/{model_name}"
            except Exception as e:
                logger.warning(f"Could not resolve default embedding model: {e}")
                default_model = "azure_openai/text-embedding-3-large"  # Fallback

            return {
                "available_models": available_models,
                "default_model": default_model,
                "environment": environment,
            }
        except Exception as e:
            logger.error(f"Failed to get embedding models for vector indexer: {e}")
            raise

    def _get_embedding_manager(self) -> "EmbeddingManager":
        """Lazy initialization of EmbeddingManager for vector indexer."""
        if self._embedding_manager is None:
            from src.llm_orchestrator_config.embedding_manager import EmbeddingManager
            from src.llm_orchestrator_config.vault.vault_client import get_vault_client

            vault_client = get_vault_client()
            config_loader = self._get_config_loader()

            self._embedding_manager = EmbeddingManager(vault_client, config_loader)
            logger.debug("Lazy initialized EmbeddingManager for vector indexer")

        return self._embedding_manager

    def _get_context_manager(self) -> "ContextGenerationManager":
        """Lazy initialization of ContextGenerationManager for vector indexer."""
        if self._context_manager is None:
            from src.llm_orchestrator_config.context_manager import (
                ContextGenerationManager,
            )
            from src.utils.connection_id_fetcher import get_connection_id_fetcher

            fetcher = get_connection_id_fetcher()
            connection_id = fetcher.fetch_vault_uuid_sync("production")

            llm_manager = LLMManager(
                environment="production", connection_id=connection_id
            )
            self._context_manager = ContextGenerationManager(llm_manager)
            logger.debug("Lazy initialized ContextGenerationManager for vector indexer")

        return self._context_manager

    def _get_config_loader(self) -> "ConfigurationLoader":
        """Lazy initialization of ConfigurationLoader for vector indexer."""
        if self._config_loader is None:
            from src.llm_orchestrator_config.config.loader import ConfigurationLoader

            self._config_loader = ConfigurationLoader()
            logger.debug("Lazy initialized ConfigurationLoader for vector indexer")

        return self._config_loader

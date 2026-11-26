"""Vector-indexer support: embeddings and contextual-chunk generation.

Extracted from LLMOrchestrationService. This is a self-contained collaborator
used by the `/embeddings`, `/generate-context`, and `/embedding-models` API
routes (called by the external vector-indexer service), and reused as the
general-purpose embedding provider injected into RAG retrieval and the tool
classifier (`llm_service` / `orchestration_service` / `embedding_service`
duck-typed references). It has no dependency on the RAG/streaming pipeline.
"""

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import ContextGenerationRequest, EmbeddingLockCheckRequest
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
        # One context manager per (environment, vault_uuid); see _get_context_manager.
        self._context_managers: Dict[Tuple[str, str], "ContextGenerationManager"] = {}
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
            # Passing the request's connection_id lets dev/test indexers work
            # without Resql; otherwise the environment's current connection is used.
            context_manager = self._get_context_manager(
                environment=request.environment,
                connection_id=request.connection_id,
            )

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

    def get_knowledge_base_embedding_model(self) -> Dict[str, Any]:
        """The embedding model(s) locked by the knowledge base (issue #806)."""
        from src.llm_orchestrator_config.knowledge_base_lock import read_locked_models
        from src.llm_orchestrator_config.llm_ochestrator_constants import QDRANT_URL

        locked = read_locked_models(QDRANT_URL)
        if locked:
            logger.info(
                "Knowledge base embedding model: "
                + " and ".join(model.describe() for model in locked)
            )
        else:
            logger.info("Knowledge base is empty - no embedding model locked yet")
        return {"locked": bool(locked), "models": [asdict(model) for model in locked]}

    def check_embedding_lock(
        self, request: EmbeddingLockCheckRequest
    ) -> Dict[str, Any]:
        """Check a connection's embedding model against the knowledge base (#806).

        Raises:
            KnowledgeBaseLockError: Qdrant or Resql could not be read, so the
                check cannot be made (callers treat this as blocked).
        """
        from src.llm_orchestrator_config.knowledge_base_lock import (
            EmbeddingCheckResult,
            check_embedding_compatibility,
            read_locked_models,
            read_other_connections,
        )
        from src.llm_orchestrator_config.llm_ochestrator_constants import (
            QDRANT_URL,
            RAG_SEARCH_RESQL,
        )

        locked = read_locked_models(QDRANT_URL)
        # Ruuter sends "" / 0 for fields it was not given (it cannot send null)
        others = read_other_connections(
            RAG_SEARCH_RESQL, request.exclude_connection_id or None
        )

        # Form values first: a platform/model conflict is reported as such, without
        # spending a test embedding that might fail for unrelated reasons.
        environment = request.deployment_environment or None
        declared = check_embedding_compatibility(
            platform=request.embedding_platform,
            embedding_model=request.embedding_model,
            detected_vector_dimensions=None,
            locked_models=locked,
            other_connections=others,
            environment=environment,
        )

        # A temporary secret is deleted by Ruuter as soon as this answers. When the
        # form values already fail there is no test embedding, so wait for the
        # (asynchronous) store to finish first; otherwise the delete can run
        # before the secret is written and leave it behind in Vault.
        if request.vault_uuid and not declared.allowed:
            if not self._get_embedding_manager().wait_until_embedding_secret_written(
                request.vault_uuid
            ):
                logger.warning(
                    "Embedding lock check: temporary secret "
                    f"{request.vault_uuid} did not appear in Vault within the wait"
                )

        detected_dimensions: Optional[int] = None
        test_vault_uuid = request.vault_uuid or request.existing_vault_uuid
        if test_vault_uuid and declared.allowed:
            overrides = {
                "deployment_name": request.embedding_deployment_name or "",
                "endpoint": request.embedding_target_uri or "",
            }
            try:
                measured = self._get_embedding_manager().measure_test_embedding(
                    test_vault_uuid,
                    overrides=overrides
                    if request.embedding_platform == "azure"
                    else None,
                )
                detected_dimensions = measured["vector_dimensions"]
            except Exception as e:  # missing secret, bad key, wrong endpoint, ...
                logger.warning(f"Embedding lock check: test embedding failed: {e}")
                return EmbeddingCheckResult(
                    allowed=False,
                    code="test_embedding_failed",
                    reason="A test embedding could not be made with these credentials.",
                    locked_models=[asdict(model) for model in locked],
                ).to_dict()

        result = check_embedding_compatibility(
            platform=request.embedding_platform,
            embedding_model=request.embedding_model,
            detected_vector_dimensions=detected_dimensions,
            locked_models=locked,
            other_connections=others,
            environment=environment,
        )

        locked_text = (
            " and ".join(model.describe() for model in locked)
            if locked
            else "none (knowledge base empty)"
        )
        tested = (
            f", test embedding {detected_dimensions} dimensions"
            if detected_dimensions is not None
            else ""
        )
        message = (
            f"Embedding lock check for {request.embedding_model} ({request.embedding_platform}"
            f"{tested}, {environment or 'testing'}): "
            f"{'allowed' if result.allowed else 'BLOCKED'}. "
            f"Knowledge base embedding model: {locked_text}"
        )
        if result.allowed:
            logger.info(message)
        else:
            logger.warning(f"{message}. Reason: {result.reason}")
        if result.reindex_required:
            logger.warning(
                f"Knowledge base re-index required: production connection uses "
                f"{request.embedding_model} ({request.embedding_platform}), knowledge "
                f"base embedding model: {locked_text}. The next indexer run "
                "re-indexes all content with the new model."
            )
        for other in result.mismatched_connections:
            logger.warning(
                f"After this change the {other['environment']} connection "
                f"'{other['connection_name']}' ({other['embedding_model']}, "
                f"{other['platform']}) no longer matches the knowledge base's "
                "embedding model; its queries will find nothing until it uses "
                f"{request.embedding_model} ({request.embedding_platform})."
            )
        return result.to_dict()

    def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Whether the knowledge base must be (re-)indexed (issues #586, #588).

        Compares the embedding platform of the connection the indexer uses
        (production, otherwise testing) with the platforms that hold embeddings.
        The data/update sync runs the indexer when ``reindex_required`` is true,
        even if the CKB data hash has not changed.

        Raises:
            KnowledgeBaseLockError: Qdrant or Resql could not be read.
        """
        from src.llm_orchestrator_config.knowledge_base_lock import (
            read_locked_models,
            read_other_connections,
            reindex_required_for,
        )
        from src.llm_orchestrator_config.llm_ochestrator_constants import (
            QDRANT_URL,
            RAG_SEARCH_RESQL,
        )

        locked = read_locked_models(QDRANT_URL)
        connections = read_other_connections(RAG_SEARCH_RESQL)
        # Same choice as the indexer: production when it exists, else testing.
        indexing = next(
            (c for c in connections if c.environment == "production"),
            next((c for c in connections if c.environment == "testing"), None),
        )

        reindex_required = bool(
            indexing and reindex_required_for(indexing.platform, locked)
        )
        locked_text = (
            " and ".join(model.describe() for model in locked)
            if locked
            else "none (knowledge base empty)"
        )
        if indexing is None:
            logger.info(
                "Knowledge base status: no active LLM connection; "
                f"knowledge base embedding model: {locked_text}"
            )
        elif reindex_required:
            logger.warning(
                f"Knowledge base status: re-index required - the {indexing.environment} "
                f"connection uses {indexing.embedding_model} ({indexing.platform}), "
                f"knowledge base embedding model: {locked_text}"
            )
        else:
            logger.info(
                f"Knowledge base status: up to date with the {indexing.environment} "
                f"connection's {indexing.embedding_model} ({indexing.platform}); "
                f"knowledge base embedding model: {locked_text}"
            )

        return {
            "reindex_required": reindex_required,
            "indexing_environment": indexing.environment if indexing else None,
            "indexing_platform": indexing.platform if indexing else None,
            "indexing_embedding_model": indexing.embedding_model if indexing else None,
            "locked_models": [asdict(model) for model in locked],
        }

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

    def _get_context_manager(
        self,
        environment: str = PRODUCTION_DEPLOYMENT_ENVIRONMENT,
        connection_id: Optional[str] = None,
    ) -> "ContextGenerationManager":
        """Return the ContextGenerationManager for the connection to index with.

        Mirrors ``create_embeddings_for_indexer``: a ``connection_id`` supplied by
        the caller (dev/test indexers, the deepeval stack) is used as-is, so no
        Resql lookup is needed; only when it is absent is the environment's
        current vault_uuid looked up in the DB (Resql has routes for
        ``production`` and ``testing`` only, so ``development`` without a
        connection_id fails with ``ConfigurationError``, exactly like the
        embeddings path).

        The connection is resolved on every call, so after a connection switch or
        ``/cache/clear`` the next call uses the current connection. ``LLMManager``
        caches one instance per connection, so this is cheap; a context manager
        is created again only when its LLMManager has been reloaded.
        """
        if not connection_id:
            from src.utils.connection_id_fetcher import get_connection_id_fetcher

            fetcher = get_connection_id_fetcher()
            connection_id = fetcher.fetch_vault_uuid_sync(environment)

        llm_manager = LLMManager(environment=environment, connection_id=connection_id)
        key = (environment, llm_manager.connection_id or "")

        context_manager = self._context_managers.get(key)
        if context_manager is None or context_manager.llm_manager is not llm_manager:
            from src.llm_orchestrator_config.context_manager import (
                ContextGenerationManager,
            )

            context_manager = ContextGenerationManager(llm_manager)
            self._context_managers[key] = context_manager
            logger.debug(
                f"Initialized ContextGenerationManager for vector indexer "
                f"({environment}, vault_uuid: {llm_manager.connection_id})"
            )

        return context_manager

    def _get_config_loader(self) -> "ConfigurationLoader":
        """Lazy initialization of ConfigurationLoader for vector indexer."""
        if self._config_loader is None:
            from src.llm_orchestrator_config.config.loader import ConfigurationLoader

            self._config_loader = ConfigurationLoader()
            logger.debug("Lazy initialized ConfigurationLoader for vector indexer")

        return self._config_loader

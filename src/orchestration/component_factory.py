"""Per-request component construction and caching (Group D).

Extracted from LLMOrchestrationService. Builds the LLM manager, guardrails
adapter, contextual retriever, and response generator for a request, and owns
the ContextualRetriever cache.

The factory holds a back-reference to the orchestration-service façade because
several collaborators are startup-owned state that must remain publicly
readable on the façade (``shared_guardrails_adapters``, ``shared_bm25_search``,
``prompt_config_loader``), and because ContextualRetriever receives
``llm_service=<façade>`` to break a circular dependency.
"""

import threading
from typing import Any, Dict, Optional

from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import OrchestrationRequest
from llm_orchestrator_config.llm_manager import LLMManager
from src.llm_orchestrator_config.llm_ochestrator_constants import QDRANT_URL
from src.contextual_retrieval import ContextualRetriever
from src.guardrails import NeMoRailsAdapter
from src.response_generator.response_generate import ResponseGeneratorAgent

logger = LokiLogger(service_name="llm-orchestration-service")


class ComponentFactory:
    """Constructs and caches the per-request service components."""

    def __init__(self, orchestration_service: Any) -> None:  # noqa: ANN401 — façade, avoids circular import
        self._service = orchestration_service
        self._retriever_cache: Dict[tuple, ContextualRetriever] = {}
        self._cache_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Aggregate entry point
    # ------------------------------------------------------------------

    def initialize_service_components(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Initialize all service components and return them as a dictionary."""
        components: Dict[str, Any] = {}

        llm_manager = self.initialize_llm_manager(
            environment=request.environment, connection_id=request.connection_id
        )
        components["llm_manager"] = llm_manager

        # Store resolved connection_id on request for downstream use
        # (budget, inference storage)
        if llm_manager.connection_id and not request.connection_id:
            request.connection_id = llm_manager.connection_id
            logger.debug(
                f"Stored resolved vault_uuid on request: {llm_manager.connection_id}"
            )

        shared_adapters = self._service.shared_guardrails_adapters
        if request.environment in shared_adapters:
            logger.info(
                f" Using shared guardrails adapter for environment='{request.environment}' "
                f"(startup-initialized, zero overhead)"
            )
            components["guardrails_adapter"] = shared_adapters[request.environment]
        else:
            logger.warning(
                f" Shared guardrails unavailable for environment='{request.environment}', "
                f"initializing per-request (slower)"
            )
            components["guardrails_adapter"] = self.safe_initialize_guardrails(
                request.environment, request.connection_id
            )

        components["contextual_retriever"] = self._get_or_create_retriever(
            request.environment, request.connection_id
        )

        # Response Generator is created fresh per request - NOT cached.
        # ResponseGeneratorAgent uses dspy.streamify() which has internal state
        # that doesn't reset between calls, causing 0-token streaming on reuse.
        # Only costs ~0.02s to create, so caching is not worth the risk.
        components["response_generator"] = self.safe_initialize_response_generator(
            components["llm_manager"]
        )

        return components

    def _get_or_create_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[ContextualRetriever]:
        """Return a cached ContextualRetriever, creating it under lock if absent."""
        retriever_key = (environment, connection_id)

        if retriever_key in self._retriever_cache:
            logger.info(f"Using cached ContextualRetriever for key={retriever_key}")
            return self._retriever_cache[retriever_key]

        with self._cache_lock:
            if retriever_key in self._retriever_cache:
                return self._retriever_cache[retriever_key]

            retriever = self.safe_initialize_contextual_retriever(
                environment, connection_id
            )
            if retriever is not None:
                self._retriever_cache[retriever_key] = retriever
            return retriever

    # ------------------------------------------------------------------
    # Safe (error-swallowing) wrappers
    # ------------------------------------------------------------------

    def safe_initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[NeMoRailsAdapter]:
        """Safely initialize guardrails adapter with error handling."""
        try:
            adapter = self.initialize_guardrails(environment, connection_id)
            logger.info("Guardrails adapter initialization successful")
            return adapter
        except Exception as guardrails_error:
            logger.warning(f"Guardrails initialization failed: {str(guardrails_error)}")
            logger.warning("Continuing without guardrails protection")
            return None

    @observe(name="safe_initialize_contextual_retriever", as_type="span")
    def safe_initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[ContextualRetriever]:
        """Safely initialize contextual retriever with error handling."""
        try:
            retriever = self.initialize_contextual_retriever(environment, connection_id)
            logger.info("Contextual Retriever initialization successful")
            return retriever
        except Exception as retriever_error:
            logger.warning(
                f"Contextual Retriever initialization failed: {str(retriever_error)}"
            )
            logger.warning("Continuing without chunk retrieval capabilities")
            return None

    @observe(name="safe_initialize_response_generator", as_type="span")
    def safe_initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> Optional[ResponseGeneratorAgent]:
        """Safely initialize response generator with error handling."""
        try:
            generator = self.initialize_response_generator(llm_manager)
            logger.info("Response Generator initialization successful")
            return generator
        except Exception as generator_error:
            logger.warning(
                f"Response Generator initialization failed: {str(generator_error)}"
            )
            return None

    # ------------------------------------------------------------------
    # Raw constructors
    # ------------------------------------------------------------------

    @observe(name="initialize_llm_manager", as_type="span")
    def initialize_llm_manager(
        self, environment: str, connection_id: Optional[str]
    ) -> LLMManager:
        """Initialize LLM Manager with proper configuration.

        For production, resolves vault_uuid from the DB when not provided.
        For testing, connection_id (vault_uuid) must be provided.
        """
        try:
            logger.info(f"Initializing LLM Manager for environment: {environment}")

            resolved_connection_id = connection_id

            if environment == "production" and not connection_id:
                from src.utils.connection_id_fetcher import get_connection_id_fetcher

                fetcher = get_connection_id_fetcher()
                resolved_connection_id = fetcher.fetch_vault_uuid_sync("production")
                if not resolved_connection_id:
                    raise ValueError(
                        "No production connection found in database. "
                        "Please create a production LLM connection first."
                    )
                logger.info(
                    f"Resolved production vault_uuid from DB: {resolved_connection_id}"
                )

            llm_manager = LLMManager(
                environment=environment, connection_id=resolved_connection_id
            )
            llm_manager.ensure_global_config()

            logger.info("LLM Manager initialized successfully")
            return llm_manager

        except Exception as e:
            logger.error(f"Failed to initialize LLM Manager: {str(e)}")
            raise

    def initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> NeMoRailsAdapter:
        """Initialize a NeMo Guardrails adapter.

        Raises:
            Exception: For initialization errors
        """
        try:
            logger.info(f"Initializing Guardrails for environment: {environment}")
            return NeMoRailsAdapter(
                environment=environment, connection_id=connection_id
            )
        except Exception as e:
            logger.error(f"Failed to initialize Guardrails adapter: {str(e)}")
            raise

    @observe(name="initialize_contextual_retriever", as_type="span")
    def initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> ContextualRetriever:
        """Initialize contextual retriever for enhanced document retrieval."""
        logger.info("Initializing contextual retriever")

        try:
            contextual_retriever = ContextualRetriever(
                qdrant_url=QDRANT_URL,
                environment=environment,
                connection_id=connection_id,
                # Inject the façade to eliminate a circular dependency
                llm_service=self._service,
                # Inject the startup pre-warmed BM25 index
                shared_bm25=self._service.shared_bm25_search,
            )

            logger.info("Contextual retriever initialized successfully")
            return contextual_retriever

        except Exception as e:
            logger.error(f"Failed to initialize contextual retriever: {str(e)}")
            raise

    @observe(name="initialize_response_generator", as_type="span")
    def initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> ResponseGeneratorAgent:
        """Initialize Response Generator with the provided LLM manager."""
        logger.info("Initializing response generator")

        try:
            custom_prefix = self.get_custom_instructions_for_response_generation()

            with llm_manager.use_task_local():
                response_generator = ResponseGeneratorAgent(
                    custom_instructions_prefix=custom_prefix
                )

            logger.info("Response generator initialized successfully")
            return response_generator

        except Exception as e:
            logger.error(f"Failed to initialize response generator: {str(e)}")
            raise

    def get_custom_instructions_for_response_generation(self) -> str:
        """Get custom prompt instructions for response generation only.

        Note: Applied only to ResponseGeneratorAgent, not PromptRefinerAgent.
        PromptRefiner focuses on query optimization for retrieval, while
        ResponseGenerator needs to follow language policy and interaction style
        for user-facing content.
        """
        try:
            loader = self._service.prompt_config_loader
            custom_prompt = loader.get_custom_instructions() if loader else None
            if custom_prompt:
                return f"[SYSTEM INSTRUCTIONS]\n{custom_prompt}\n\n[USER QUESTION]\n"
            return ""
        except Exception as e:
            logger.error(f"Error retrieving custom instructions: {e}")
            return ""

"""LLM Orchestration Service - Business logic for LLM orchestration."""

from typing import Optional, List, Dict, Union, Any, AsyncIterator, TYPE_CHECKING
import os
import time
import asyncio
import threading
from src.loki_logger import LokiLogger
from langfuse import Langfuse, observe
import dspy
from datetime import datetime
import json as json_module

from llm_orchestrator_config.llm_manager import LLMManager
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    ConversationItem,
    PromptRefinerOutput,
    ContextGenerationRequest,
    TestOrchestrationResponse,
    ChunkInfo,
    DocumentReference,
)
from prompt_refine_manager.prompt_refiner import PromptRefinerAgent
from src.response_generator.response_generate import ResponseGeneratorAgent
from src.response_generator.response_generate import stream_response_native
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    OUT_OF_SCOPE_MESSAGES,
    TECHNICAL_ISSUE_MESSAGE,
    TECHNICAL_ISSUE_MESSAGES,
    INPUT_GUARDRAIL_VIOLATION_MESSAGE,
    INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGE,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
    QUERY_VALIDATION_FAILED_MESSAGES,
    get_localized_message,
    GUARDRAILS_BLOCKED_PHRASES,
    TEST_DEPLOYMENT_ENVIRONMENT,
    STREAM_TOKEN_LIMIT_MESSAGE,
    PRODUCTION_DEPLOYMENT_ENVIRONMENT,
    RUUTER_PROMPT_CONFIG_ENDPOINT,
    PROMPT_CONFIG_CACHE_TTL,
)
from src.llm_orchestrator_config.stream_config import StreamConfig
from src.vector_indexer.constants import ResponseGenerationConstants
from src.utils.error_utils import generate_error_id, log_error_with_context
from src.utils.stream_manager import stream_manager, StreamContext
from src.utils.cost_utils import (
    calculate_total_costs,
    get_lm_usage_since,
    get_lm_usage_since_split,
)

if TYPE_CHECKING:
    from src.llm_orchestrator_config.embedding_manager import EmbeddingManager
    from src.llm_orchestrator_config.context_manager import (
        ContextGenerationManager,
    )
    from src.llm_orchestrator_config.config.loader import ConfigurationLoader
from src.utils.time_tracker import log_step_timings
from src.utils.budget_tracker import get_budget_tracker
from src.utils.production_store import get_production_store
from src.utils.language_detector import detect_language, get_language_name
from src.utils.prompt_config_loader import PromptConfigurationLoader
from src.utils.query_validator import validate_query_basic
from src.utils.sse_utils import extract_content_from_sse
from src.utils.conversation_history_store import should_save_history, save_history_round
from src.utils.conversation_history_helpers import get_conversation_history
from src.guardrails import NeMoRailsAdapter, GuardrailCheckResult
from src.contextual_retrieval import ContextualRetriever
from src.contextual_retrieval.bm25_search import SmartBM25Search
from src.llm_orchestrator_config.exceptions import (
    ContextualRetrieverInitializationError,
    ContextualRetrievalFailureError,
)
from src.llm_orchestrator_config.feature_flags import FeatureFlags
from src.tool_classifier import ToolClassifier, WorkflowType
from src.tool_classifier.constants import SERVICE_STEP_PREFIXES
from src.tool_classifier.workflows.service_workflow import ServiceWorkflowExecutor

# Initialize Loki logger for orchestration service
logger = LokiLogger(service_name="llm-orchestration-service")

REFERENCES_SECTION_HEADER = "\n\n**References:**\n"

# Set of content strings that must NOT be persisted in conversation history.
# Covers all multilingual error / OOS / guardrail-violation messages so that
# failed or blocked exchanges are never written to Redis.
_HISTORY_EXCLUDED_MESSAGES: frozenset[str] = frozenset(
    {
        *OUT_OF_SCOPE_MESSAGES.values(),
        *TECHNICAL_ISSUE_MESSAGES.values(),
        *INPUT_GUARDRAIL_VIOLATION_MESSAGES.values(),
        *OUTPUT_GUARDRAIL_VIOLATION_MESSAGES.values(),
        *QUERY_VALIDATION_FAILED_MESSAGES.values(),
        STREAM_TOKEN_LIMIT_MESSAGE,
    }
)


class LangfuseConfig:
    """Configuration for Langfuse integration."""

    def __init__(self) -> None:
        self.langfuse_client: Optional[Langfuse] = None
        self._initialize_langfuse()

    def _initialize_langfuse(self) -> None:
        """Initialize Langfuse client with Vault secrets."""
        try:
            from llm_orchestrator_config.vault.vault_client import get_vault_client

            vault = get_vault_client()
            if vault.is_vault_available():
                langfuse_secrets = vault.get_secret("langfuse/config")
                if langfuse_secrets:
                    self.langfuse_client = Langfuse(
                        public_key=langfuse_secrets.get("public_key"),
                        secret_key=langfuse_secrets.get("secret_key"),
                        host=langfuse_secrets.get("host", "http://langfuse-web:3000"),
                    )
                    logger.info("Langfuse client initialized successfully")
                else:
                    logger.warning("Langfuse secrets not found in Vault")
            else:
                logger.warning("Vault not available, Langfuse tracing disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize Langfuse: {e}")


class LLMOrchestrationService:
    """
    Service class for handling LLM orchestration with integrated guardrails.
    Features:
    - Input guardrails before prompt refinement
    - Output guardrails after response generation
    - Comprehensive cost tracking for all components
    """

    def __init__(self) -> None:
        """Initialize the orchestration service."""
        self.langfuse_config = LangfuseConfig()

        # Initialize prompt configuration loader
        self.prompt_config_loader = PromptConfigurationLoader(
            ruuter_endpoint=RUUTER_PROMPT_CONFIG_ENDPOINT,
            cache_ttl_seconds=PROMPT_CONFIG_CACHE_TTL,
            max_retries=3,
            timeout_seconds=10,
        )

        try:
            custom_instructions = self.prompt_config_loader.get_custom_instructions()
            if custom_instructions:
                logger.info(
                    f"Custom prompt configuration loaded at startup "
                    f"({len(custom_instructions)} chars)"
                )
            else:
                logger.info("ℹNo custom prompt configuration found - using defaults")
        except Exception as e:
            logger.warning(
                f"Failed to load custom prompts at startup: {e}. "
                f"Service will continue with default behavior."
            )

        # Initialize tool classifier (lazy initialization - will be created when first needed)
        # This allows components to be initialized per-request with proper context
        self.tool_classifier = None

        # Redis-backed session store for API Tool Calling agentic loop.
        # Set to None here; the FastAPI lifespan injects the live store after
        # Redis initialises (app.state.orchestration_service.session_store = ...).
        # Workflow executors access it via self.orchestration_service.session_store.
        self.session_store: Any = None

        # Redis-backed conversation history store.
        # Set to None here; the FastAPI lifespan injects the live store after
        # Redis initialises (app.state.orchestration_service.conversation_history_store = ...).
        self.conversation_history_store: Any = None

        # Shared BM25 search index pre-warmed at startup.
        # Populated by _prewarm_shared_bm25() which is called from the FastAPI
        # lifespan so it runs inside the async event loop.  Until then it is None
        # and each ContextualRetriever will build the index on first query (graceful
        # degradation path).
        self.shared_bm25_search: Optional[SmartBM25Search] = None

        self._retriever_cache: Dict[tuple, ContextualRetriever] = {}
        self._component_cache_lock = threading.Lock()

        # Initialize shared guardrails adapters at startup (production and testing)
        self.shared_guardrails_adapters = (
            self._initialize_shared_guardrails_at_startup()
        )

        # Log feature flag configuration
        FeatureFlags.log_configuration()

    def _initialize_shared_guardrails_at_startup(self) -> Dict[str, NeMoRailsAdapter]:
        """
        Initialize shared guardrails adapters at startup for production and testing environments.

        Returns:
            Dictionary mapping environment names to NeMoRailsAdapter instances.
            Empty dict on failure (graceful degradation).
        """
        adapters: Dict[str, NeMoRailsAdapter] = {}

        # Initialize adapters for commonly-used environments
        environments_to_initialize = ["production", "testing"]

        logger.info("  Initializing shared guardrails at startup...")
        total_start_time = time.time()

        for env in environments_to_initialize:
            try:
                logger.info(f"  Initializing guardrails for environment: {env}")
                start_time = time.time()

                # Initialize with specific environment and no connection (shared config)
                guardrails_adapter = self._initialize_guardrails(
                    environment=env,
                    connection_id=None,  # Shared configuration, not user-specific
                )

                # Eagerly trigger the full internal initialization (NeMo config
                # loading, LLMRails creation, embedding model download) so that
                # the first user query is not penalised by the cold-start cost.
                # Without this, _ensure_initialized() runs lazily on the first
                guardrails_adapter._ensure_initialized()

                elapsed_time = time.time() - start_time
                adapters[env] = guardrails_adapter
                logger.info(
                    f" Guardrails for '{env}' fully initialized in {elapsed_time:.3f}s "
                    f"(NeMo Rails + embedding model loaded)"
                )

            except Exception as e:
                logger.error(f" Failed to initialize guardrails for '{env}': {e}")
                logger.warning(
                    f"  Service will fall back to per-request initialization for '{env}' environment"
                )
                # Continue with other environments - partial success is acceptable
                continue

        total_elapsed = time.time() - total_start_time

        if adapters:
            logger.info(
                f" Shared guardrails initialized for {len(adapters)} environment(s) "
                f"in {total_elapsed:.3f}s total"
            )
        else:
            logger.error(
                "  Failed to initialize any shared guardrails - "
                "service will use per-request initialization (slower)"
            )

        return adapters

    async def _prewarm_shared_bm25(self) -> None:
        """
        Pre-warm the shared BM25 index at application startup.

        Must be called from an async context (e.g. FastAPI lifespan) so that
        asyncio is available for the HTTP calls to Qdrant.  Absorbs the
        cold-start latency (fetching all chunks + building BM25Okapi corpus)
        at deploy time so that the first real user query is not penalised.

        On any failure the method logs a warning and leaves
        self.shared_bm25_search as None — the ContextualRetriever will then
        fall back to building the index on the first query (graceful degradation).
        """
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        logger.info("Pre-warming shared BM25 index at startup...")
        prewarm_start = time.time()
        try:
            bm25 = SmartBM25Search(qdrant_url=qdrant_url)
            success = await bm25.initialize_index()
            if success:
                self.shared_bm25_search = bm25
                elapsed = time.time() - prewarm_start
                logger.info(
                    f"Shared BM25 index pre-warmed in {elapsed:.2f}s "
                    f"({len(bm25.chunk_mapping)} chunks indexed)"
                )
            else:
                logger.warning(
                    "BM25 pre-warming produced an empty index - "
                    "index will be built on first query instead"
                )
        except Exception as e:
            logger.warning(
                f"BM25 pre-warming failed: {e} - "
                f"index will be built on first query (graceful degradation)"
            )

    async def aclose(self) -> None:
        """Release all long-lived async resources held by the service.

        Must be awaited during application shutdown (FastAPI lifespan teardown)
        to avoid connection leaks from the ToolClassifier's httpx client.
        """
        if self.tool_classifier is not None:
            await self.tool_classifier.aclose()
            logger.debug("LLMOrchestrationService async resources closed")

    def _get_service_workflow_executor(self) -> ServiceWorkflowExecutor:
        """Return the ServiceWorkflowExecutor, reusing the ToolClassifier instance
        when available, or creating a lightweight standalone executor otherwise.

        Direct MCQ steps do not invoke any LLM, so llm_manager=None is safe.
        orchestration_service=self is needed for format_sse() in the streaming path.
        """
        if self.tool_classifier is not None:
            return self.tool_classifier.service_workflow
        return ServiceWorkflowExecutor(
            llm_manager=None,
            orchestration_service=self,
        )

    @observe(name="orchestration_request", as_type="agent")
    async def process_orchestration_request(
        self, request: OrchestrationRequest
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """
        Process an orchestration request with guardrails and return response.

        Pipeline:
        1. Input Guardrails Check
        2. Prompt Refinement (if input allowed)
        3. Chunk Retrieval
        4. Response Generation
        5. Output Guardrails Check
        6. Cost Logging

        Args:
            request: The orchestration request containing user message and context

        Returns:
            OrchestrationResponse: Response with LLM output and status flags

        Raises:
            Exception: For any processing errors
        """
        costs_metric: Dict[str, Dict[str, Any]] = {}
        time_metric: Dict[str, float] = {}

        try:
            logger.info(
                f"Processing orchestration request for chatId: {request.chatId}, "
                f"authorId: {request.authorId}, environment: {request.environment}"
            )

            # STEP 0: Detect language from user message (with timing)
            start_time = time.time()
            detected_language = detect_language(request.message)
            language_name = get_language_name(detected_language)
            time_metric["language_detection"] = time.time() - start_time
            logger.info(
                f"[{request.chatId}] Detected language: {language_name} ({detected_language})"
            )

            # Store detected language in request for use throughout pipeline
            # Using setattr for type safety - adds dynamic attribute to Pydantic model instance
            setattr(request, "_detected_language", detected_language)  # noqa: B010

            # STEP 0.1: Multi-step service prefix check (bypass NLU pipeline)
            if request.message.startswith(SERVICE_STEP_PREFIXES):
                logger.info(
                    f"[{request.chatId}] #service prefix detected - direct step execution"
                )
                executor = self._get_service_workflow_executor()
                direct_response = await executor.execute_direct_step(
                    request=request,
                    time_metric=time_metric,
                )
                if direct_response is not None:
                    log_step_timings(time_metric, request.chatId)
                    return direct_response
                # Parse failed — fall through to normal pipeline
                logger.warning(
                    f"[{request.chatId}] Direct step failed, falling through to normal pipeline"
                )

            # STEP 0.5: Basic Query Validation (before expensive component initialization)
            start_time = time.time()
            validation_result = validate_query_basic(request.message)
            time_metric["query_validation"] = time.time() - start_time
            if not validation_result.is_valid:
                logger.info(
                    f"[{request.chatId}] Query validation failed: {validation_result.rejection_reason}"
                )
                # Get localized message
                validation_msg = get_localized_message(
                    QUERY_VALIDATION_FAILED_MESSAGES, detected_language
                )

                # Return appropriate response type without initializing components
                if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                    return TestOrchestrationResponse(
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=validation_msg,
                        chunks=None,
                    )
                else:
                    return OrchestrationResponse(
                        chatId=request.chatId,
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=validation_msg,
                    )

            # Initialize all service components (only for valid queries, with timing)
            start_time = time.time()
            components = self._initialize_service_components(request)
            time_metric["initialization"] = time.time() - start_time

            if components["guardrails_adapter"]:
                start_time = time.time()
                input_blocked_response = await self.handle_input_guardrails(
                    components["guardrails_adapter"], request, costs_metric
                )
                time_metric["input_guardrails_check"] = time.time() - start_time

                if input_blocked_response:
                    logger.warning(
                        f"[{request.chatId}] Input blocked before classifier - "
                        f"saved expensive service discovery"
                    )
                    log_step_timings(time_metric, request.chatId)
                    return input_blocked_response
            else:
                logger.info(
                    f"[{request.chatId}] Guardrails not available - "
                    f"proceeding without input validation"
                )

            # TOOL CLASSIFIER INTEGRATION
            # Route through tool classifier if enabled, otherwise use existing RAG pipeline
            if FeatureFlags.TOOL_CLASSIFIER_ENABLED:
                try:
                    logger.info(
                        f"[{request.chatId}] Tool classifier enabled - routing query"
                    )

                    # Initialize tool classifier if not already done
                    if self.tool_classifier is None:
                        self.tool_classifier = ToolClassifier(
                            llm_manager=components["llm_manager"],
                            orchestration_service=self,
                        )
                        logger.info("Tool classifier initialized")

                    # Classify query to determine workflow (with timing)
                    start_time = time.time()
                    classification = await self.tool_classifier.classify(
                        query=request.message,
                        language=detected_language,
                        request=request,
                    )
                    time_metric["classifier.classify"] = time.time() - start_time

                    logger.info(
                        f"[{request.chatId}] Classification: {classification.workflow.value} "
                        f"(confidence: {classification.confidence:.2f})"
                    )

                    # Route to appropriate workflow (with timing)
                    start_time = time.time()
                    response = await self.tool_classifier.route_to_workflow(
                        classification=classification,
                        request=request,
                        is_streaming=False,
                        time_metric=time_metric,
                    )
                    time_metric["classifier.route"] = time.time() - start_time

                except Exception as classifier_error:
                    logger.error(
                        f"[{request.chatId}] Tool classifier error: {classifier_error}",
                        exc_info=True,
                    )

                    if FeatureFlags.FALLBACK_TO_RAG_ON_ERROR:
                        logger.info(
                            f"[{request.chatId}] Falling back to RAG pipeline due to classifier error"
                        )
                        # Execute existing RAG pipeline as fallback
                        response = await self._execute_orchestration_pipeline(
                            request, components, costs_metric, time_metric
                        )
                    else:
                        raise
            else:
                # Tool classifier disabled - use existing RAG pipeline
                logger.debug(
                    f"[{request.chatId}] Tool classifier disabled - using RAG pipeline"
                )
                response = await self._execute_orchestration_pipeline(
                    request, components, costs_metric, time_metric
                )

            # Log final costs and return response
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)

            # Update budget for the LLM connection
            self._update_connection_budget(
                request.connection_id, costs_metric, request.environment
            )

            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                total_costs = calculate_total_costs(costs_metric)

                langfuse.update_current_generation(
                    metadata={
                        "total_calls": total_costs.get("total_calls", 0),
                        "cost_breakdown": costs_metric,
                        "chat_id": request.chatId,
                        "author_id": request.authorId,
                        "environment": request.environment,
                    },
                )
                langfuse.flush()

            # Persist successful exchange to conversation history (non-streaming)
            if should_save_history(
                self.conversation_history_store, response, _HISTORY_EXCLUDED_MESSAGES
            ):
                await save_history_round(
                    self.conversation_history_store,
                    request.chatId,
                    request.message,
                    response.content,
                )

            return response

        except Exception as e:
            error_id = generate_error_id()
            log_error_with_context(
                logger, error_id, "orchestration_request", request.chatId, e
            )
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_generation(
                    metadata={
                        "error_id": error_id,
                        "error_type": type(e).__name__,
                        "response_type": "technical_issue",
                    }
                )
                langfuse.flush()
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)

            # Update budget even on error
            self._update_connection_budget(
                request.connection_id, costs_metric, request.environment
            )

            return self._create_error_response(request)

    async def stream_orchestration_response(
        self, request: OrchestrationRequest
    ) -> AsyncIterator[str]:
        """
        Stream orchestration response with validation-first guardrails.

        Pipeline:
        1. Input Guardrails Check (blocking)
        2. Prompt Refinement (blocking)
        3. Chunk Retrieval (blocking)
        4. Out-of-scope Check (blocking, quick)
        5. Stream through NeMo Guardrails (validation-first)

        Args:
            request: The orchestration request containing user message and context

        Yields:
            SSE-formatted strings: "data: {json}\\n\\n"

        SSE Message Format:
            {
                "chatId": "...",
                "payload": {"content": "..."},
                "timestamp": "...",
                "sentTo": []
            }

        Content Types:
            - Regular token: "Python", " is", " awesome"
            - Stream complete: "END"
            - Input blocked: INPUT_GUARDRAIL_VIOLATION_MESSAGE
            - Out of scope: OUT_OF_SCOPE_MESSAGE
            - Guardrail failed: OUTPUT_GUARDRAIL_VIOLATION_MESSAGE
            - Technical error: TECHNICAL_ISSUE_MESSAGE
        """

        # Track costs after streaming completes
        costs_metric: Dict[str, Dict[str, Any]] = {}
        time_metric: Dict[str, float] = {}

        # Capture DSPy history baseline before any LLM calls.
        # Used at the end of the request to compute the total cost delta,
        _lm = dspy.settings.lm
        initial_history_length = (
            len(_lm.history) if _lm and hasattr(_lm, "history") else 0
        )

        # STEP 0: Detect language from user message (with timing)
        start_time = time.time()
        detected_language = detect_language(request.message)
        language_name = get_language_name(detected_language)
        time_metric["language_detection"] = time.time() - start_time
        logger.info(
            f"[{request.chatId}] Streaming request - Detected language: {language_name} ({detected_language})"
        )

        # Store detected language in request for use throughout pipeline
        # Using setattr for type safety - adds dynamic attribute to Pydantic model instance
        setattr(request, "_detected_language", detected_language)  # noqa: B010

        # STEP 0.1: Multi-step service prefix check (bypass NLU pipeline)
        if request.message.startswith(SERVICE_STEP_PREFIXES):
            logger.info(
                f"[{request.chatId}] #service prefix detected - direct step stream"
            )
            executor = self._get_service_workflow_executor()
            step_stream = await executor.execute_direct_step_streaming(
                request=request,
                time_metric=time_metric,
            )
            if step_stream is not None:
                async for chunk in step_stream:
                    yield chunk
                log_step_timings(time_metric, request.chatId)
                return
            # Parse failed — fall through to normal pipeline
            logger.warning(
                f"[{request.chatId}] Direct step stream failed, falling through to normal pipeline"
            )

        # Step 0.5: Basic Query Validation (before guardrails, with timing)
        start_time = time.time()
        validation_result = validate_query_basic(request.message)
        time_metric["query_validation"] = time.time() - start_time
        if not validation_result.is_valid:
            logger.info(
                f"[{request.chatId}] Streaming - Query validation failed: {validation_result.rejection_reason}"
            )
            # Get localized message
            validation_msg = get_localized_message(
                QUERY_VALIDATION_FAILED_MESSAGES, detected_language
            )

            # Yield SSE format error + END marker
            yield self.format_sse(request.chatId, validation_msg)
            yield self.format_sse(request.chatId, "END")
            return  # Stop processing

        # Use StreamManager for centralized tracking and guaranteed cleanup
        async with stream_manager.managed_stream(
            chat_id=request.chatId, author_id=request.authorId
        ) as stream_ctx:
            try:
                logger.info(
                    f"[{request.chatId}] [{stream_ctx.stream_id}] Starting streaming orchestration "
                    f"(environment: {request.environment})"
                )

                # Initialize all service components (with timing)
                start_time = time.time()
                components = self._initialize_service_components(request)
                time_metric["initialization"] = time.time() - start_time

                # This implements fail-fast principle - block malicious/policy-violating inputs
                # before expensive operations (service discovery, LLM calls, streaming setup)
                logger.info(
                    f"[{request.chatId}] [{stream_ctx.stream_id}] Checking input guardrails (before classifier)"
                )

                if components["guardrails_adapter"]:
                    start_time = time.time()
                    input_check_result = await self._check_input_guardrails_async(
                        guardrails_adapter=components["guardrails_adapter"],
                        user_message=request.message,
                        costs_metric=costs_metric,
                    )
                    time_metric["input_guardrails_check"] = time.time() - start_time

                    if not input_check_result.allowed:
                        logger.warning(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Input blocked before classifier - "
                            f"saved expensive service discovery. Reason: {input_check_result.reason}"
                        )
                        yield self.format_sse(
                            request.chatId, INPUT_GUARDRAIL_VIOLATION_MESSAGE
                        )
                        yield self.format_sse(request.chatId, "END")
                        self.log_costs(costs_metric)
                        # Log timings before returning (for visibility)
                        log_step_timings(time_metric, request.chatId)
                        stream_ctx.mark_completed()
                        return
                else:
                    logger.info(
                        f"[{request.chatId}] [{stream_ctx.stream_id}] Guardrails not available - "
                        f"proceeding without input validation"
                    )

                logger.info(
                    f"[{request.chatId}] [{stream_ctx.stream_id}] Input guardrails passed"
                )

                # TOOL CLASSIFIER INTEGRATION (STREAMING)
                # Route through tool classifier if enabled, otherwise use existing RAG pipeline
                if FeatureFlags.TOOL_CLASSIFIER_ENABLED:
                    try:
                        logger.info(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Tool classifier enabled - routing query (streaming)"
                        )

                        # Initialize tool classifier if not already done
                        if self.tool_classifier is None:
                            self.tool_classifier = ToolClassifier(
                                llm_manager=components["llm_manager"],
                                orchestration_service=self,
                            )
                            logger.info(
                                f"[{request.chatId}] [{stream_ctx.stream_id}] Tool classifier initialized"
                            )

                        # Classify query to determine workflow
                        start_time = time.time()
                        classification = await self.tool_classifier.classify(
                            query=request.message,
                            language=detected_language,
                            request=request,
                        )
                        time_metric["classifier.classify"] = time.time() - start_time

                        logger.info(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Classification: {classification.workflow.value} "
                            f"(confidence: {classification.confidence:.2f})"
                        )

                        # Route to appropriate workflow (streaming)
                        # route_to_workflow returns AsyncIterator[str] when is_streaming=True
                        # Inject costs_metric and pre-initialized components into the
                        # classification context so downstream workflows can reuse them
                        # without re-initializing (saves ~1.5s on fallback paths).
                        classification.metadata["costs_metric"] = costs_metric
                        classification.metadata["components"] = components
                        start_time = time.time()
                        stream_result = await self.tool_classifier.route_to_workflow(
                            classification=classification,
                            request=request,
                            is_streaming=True,
                            time_metric=time_metric,
                        )
                        time_metric["classifier.route"] = time.time() - start_time

                        # Accumulate content for history only on non-RAG workflows;
                        # RAG routes through _stream_rag_pipeline which has its own hook.
                        _save_classifier_history = (
                            self.conversation_history_store is not None
                            and classification.workflow != WorkflowType.RAG
                        )
                        _classifier_accumulated: list[str] = []
                        # Tracks whether an excluded marker (OOS / guardrail violation /
                        # error) was observed at any point during the stream.  When True
                        # the entire accumulated buffer is discarded so no partial content
                        # from before the blocked marker is ever written to Redis.
                        _history_blocked = False

                        async for sse_chunk in stream_result:
                            yield sse_chunk
                            if _save_classifier_history and not _history_blocked:
                                extracted = extract_content_from_sse(sse_chunk)
                                if extracted is not None and extracted != "END":
                                    if extracted in _HISTORY_EXCLUDED_MESSAGES:
                                        # Excluded marker observed — discard any partial
                                        # content accumulated before this point and stop
                                        # accumulating for the rest of the stream.
                                        _classifier_accumulated.clear()
                                        _history_blocked = True
                                    else:
                                        _classifier_accumulated.append(extracted)

                        # Successfully completed streaming through classifier
                        logger.info(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Tool classifier streaming completed"
                        )

                        # Persist conversation history (classifier streaming, non-RAG workflows)
                        if (
                            _save_classifier_history
                            and not _history_blocked
                            and _classifier_accumulated
                        ):
                            await save_history_round(
                                self.conversation_history_store,
                                request.chatId,
                                request.message,
                                "".join(_classifier_accumulated),
                            )

                        # Log costs and timings
                        self.log_costs(costs_metric)
                        log_step_timings(time_metric, request.chatId)

                        # Budget update: use full DSPy history delta
                        _total_usage = get_lm_usage_since(initial_history_length)
                        self._update_connection_budget(
                            request.connection_id,
                            {"streaming_total": _total_usage},
                            request.environment,
                        )
                        stream_ctx.mark_completed()
                        return  # Exit after successful classifier routing

                    except Exception as classifier_error:
                        logger.error(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Tool classifier error: {classifier_error}",
                            exc_info=True,
                        )

                        if not FeatureFlags.FALLBACK_TO_RAG_ON_ERROR:
                            # Don't fallback - raise error
                            raise

                        # Fallback to RAG pipeline below
                        logger.info(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Falling back to RAG streaming due to classifier error"
                        )
                        # Continue to existing RAG streaming pipeline below
                else:
                    logger.debug(
                        f"[{request.chatId}] [{stream_ctx.stream_id}] Tool classifier disabled - using RAG streaming"
                    )

                # Execute core RAG streaming pipeline
                # NOTE: This only executes if tool classifier is disabled or fallback occurred
                async for sse_chunk in self._stream_rag_pipeline(
                    request=request,
                    components=components,
                    stream_ctx=stream_ctx,
                    costs_metric=costs_metric,
                    time_metric=time_metric,
                ):
                    yield sse_chunk

                # Pipeline completed successfully.
                # Budget update: use full DSPy history delta (covers guardrails,
                # refiner, and streaming generation across this request).
                _total_usage = get_lm_usage_since(initial_history_length)
                self._update_connection_budget(
                    request.connection_id,
                    {"streaming_total": _total_usage},
                    request.environment,
                )
                return

            except Exception as e:
                error_id = generate_error_id()
                stream_ctx.mark_error(error_id)
                log_error_with_context(
                    logger, error_id, "streaming_orchestration", request.chatId, e
                )

                yield self.format_sse(request.chatId, TECHNICAL_ISSUE_MESSAGE)
                yield self.format_sse(request.chatId, "END")

                self.log_costs(costs_metric)
                log_step_timings(time_metric, request.chatId)

                # Budget update on outer exception using full DSPy history delta.
                _total_usage = get_lm_usage_since(initial_history_length)
                self._update_connection_budget(
                    request.connection_id,
                    {"streaming_total": _total_usage},
                    request.environment,
                )

                if self.langfuse_config.langfuse_client:
                    langfuse = self.langfuse_config.langfuse_client
                    langfuse.update_current_generation(
                        metadata={
                            "error_id": error_id,
                            "error_type": type(e).__name__,
                            "streaming": True,
                            "streaming_failed": True,
                            "stream_id": stream_ctx.stream_id,
                        }
                    )
                    langfuse.flush()

    async def _stream_rag_pipeline(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        stream_ctx: StreamContext,
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
    ) -> AsyncIterator[str]:
        """
        Core RAG streaming pipeline without classifier routing.

        This method contains the RAG pipeline logic that can be called directly
        by workflows to avoid infinite recursion when the tool classifier is enabled.

        Pipeline Steps:
        1. Refine user prompt (blocking)
        2. Retrieve context chunks (blocking)
        3. Out-of-scope check (blocking)
        4. Stream through NeMo Guardrails (validation-first)

        Args:
            request: Orchestration request
            components: Initialized service components (LLM, retriever, generator, guardrails)
            stream_ctx: Stream context for tracking
            costs_metric: Dictionary to accumulate costs
            time_metric: Dictionary to accumulate timings

        Yields:
            SSE-formatted strings
        """
        streaming_start_time = datetime.now()
        detected_language = getattr(request, "_detected_language", "en")

        # STEP 1: REFINE USER PROMPT (blocking)
        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] RAG Pipeline Step 1: Refining user prompt"
        )

        start_time = time.time()
        conversation_history, conversation_summary = await get_conversation_history(
            chat_id=request.chatId,
            store=self.conversation_history_store,
            fallback=request.conversationHistory,
        )
        refined_output, refiner_usage = self._refine_user_prompt(
            llm_manager=components["llm_manager"],
            original_message=request.message,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        time_metric["prompt_refiner"] = time.time() - start_time
        costs_metric["prompt_refiner"] = refiner_usage

        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] Prompt refinement complete"
        )

        # STEP 2: RETRIEVE CONTEXT CHUNKS (blocking)
        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] RAG Pipeline Step 2: Retrieving context chunks"
        )

        try:
            start_time = time.time()
            relevant_chunks = await self._safe_retrieve_contextual_chunks(
                components["contextual_retriever"], refined_output, request
            )
            time_metric["contextual_retrieval"] = time.time() - start_time
        except (
            ContextualRetrieverInitializationError,
            ContextualRetrievalFailureError,
        ) as e:
            logger.warning(
                f"[{request.chatId}] [{stream_ctx.stream_id}] Contextual retrieval failed: {str(e)}"
            )
            logger.info(
                f"[{request.chatId}] [{stream_ctx.stream_id}] Returning out-of-scope due to retrieval failure"
            )
            localized_msg = get_localized_message(
                OUT_OF_SCOPE_MESSAGES, detected_language
            )
            yield self.format_sse(request.chatId, localized_msg)
            yield self.format_sse(request.chatId, "END")
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)
            stream_ctx.mark_completed()
            return

        if len(relevant_chunks) == 0:
            logger.info(
                f"[{request.chatId}] [{stream_ctx.stream_id}] No relevant chunks - out of scope"
            )
            localized_msg = get_localized_message(
                OUT_OF_SCOPE_MESSAGES, detected_language
            )
            yield self.format_sse(request.chatId, localized_msg)
            yield self.format_sse(request.chatId, "END")
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)
            stream_ctx.mark_completed()
            return

        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] Retrieved {len(relevant_chunks)} chunks"
        )

        # STEP 3: QUICK OUT-OF-SCOPE CHECK (blocking)
        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] RAG Pipeline Step 3: Checking if question is in scope"
        )

        start_time = time.time()
        is_out_of_scope = await components["response_generator"].check_scope_quick(
            question=refined_output.original_question,
            chunks=relevant_chunks,
            max_blocks=ResponseGenerationConstants.DEFAULT_MAX_BLOCKS,
        )
        time_metric["scope_check"] = time.time() - start_time

        if is_out_of_scope:
            logger.info(
                f"[{request.chatId}] [{stream_ctx.stream_id}] Question out of scope"
            )
            localized_msg = get_localized_message(
                OUT_OF_SCOPE_MESSAGES, detected_language
            )
            yield self.format_sse(request.chatId, localized_msg)
            yield self.format_sse(request.chatId, "END")
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)
            stream_ctx.mark_completed()
            return

        logger.info(f"[{request.chatId}] [{stream_ctx.stream_id}] Question is in scope")

        # STEP 4: STREAM THROUGH NEMO GUARDRAILS (validation-first)
        logger.info(
            f"[{request.chatId}] [{stream_ctx.stream_id}] RAG Pipeline Step 4: Starting streaming through NeMo Guardrails"
        )

        streaming_step_start = time.time()

        # Record history length before streaming
        lm = dspy.settings.lm
        history_length_before = len(lm.history) if lm and hasattr(lm, "history") else 0

        async def bot_response_generator() -> AsyncIterator[str]:
            """Generator that yields tokens from NATIVE DSPy LLM streaming."""
            async for token in stream_response_native(
                agent=components["response_generator"],
                question=refined_output.original_question,
                chunks=relevant_chunks,
                max_blocks=ResponseGenerationConstants.DEFAULT_MAX_BLOCKS,
            ):
                yield token

        # Create and store bot_generator in stream context for guaranteed cleanup
        bot_generator = bot_response_generator()
        stream_ctx.bot_generator = bot_generator

        # Wrap entire streaming logic in try/except for proper error handling
        try:
            # Track tokens and accumulated response in stream context
            accumulated_response = []  # Track the full response for production storage

            if components["guardrails_adapter"]:
                # Use NeMo's stream_with_guardrails helper method
                chunk_count = 0

                try:
                    async for validated_chunk in components[
                        "guardrails_adapter"
                    ].stream_with_guardrails(
                        user_message=refined_output.original_question,
                        bot_message_generator=bot_generator,
                    ):
                        chunk_count += 1

                        # Estimate tokens (rough approximation: 4 characters = 1 token)
                        chunk_tokens = len(validated_chunk) // 4
                        stream_ctx.token_count += chunk_tokens

                        # Accumulate response for production storage
                        accumulated_response.append(validated_chunk)

                        # Check token limit
                        if stream_ctx.token_count > StreamConfig.MAX_TOKENS_PER_STREAM:
                            logger.error(
                                f"[{request.chatId}] [{stream_ctx.stream_id}] Token limit exceeded: "
                                f"{stream_ctx.token_count} > {StreamConfig.MAX_TOKENS_PER_STREAM}"
                            )
                            yield self.format_sse(
                                request.chatId, STREAM_TOKEN_LIMIT_MESSAGE
                            )
                            yield self.format_sse(request.chatId, "END")

                            usage_info = get_lm_usage_since(history_length_before)
                            costs_metric["streaming_generation"] = usage_info
                            self.log_costs(costs_metric)
                            log_step_timings(time_metric, request.chatId)
                            stream_ctx.mark_completed()
                            return

                        # Check for guardrail violations
                        is_guardrail_error = False
                        if isinstance(validated_chunk, str):
                            blocked_phrases = GUARDRAILS_BLOCKED_PHRASES
                            chunk_lower = validated_chunk.strip().lower()
                            for phrase in blocked_phrases:
                                if (
                                    phrase.lower() in chunk_lower
                                    and len(chunk_lower) <= len(phrase.lower()) + 20
                                ):
                                    is_guardrail_error = True
                                    break

                        if is_guardrail_error:
                            logger.warning(
                                f"[{request.chatId}] [{stream_ctx.stream_id}] Guardrails violation detected"
                            )
                            yield self.format_sse(
                                request.chatId, OUTPUT_GUARDRAIL_VIOLATION_MESSAGE
                            )
                            yield self.format_sse(request.chatId, "END")

                            usage_info = get_lm_usage_since(history_length_before)
                            costs_metric["streaming_generation"] = usage_info
                            self.log_costs(costs_metric)
                            log_step_timings(time_metric, request.chatId)
                            stream_ctx.mark_completed()
                            return

                        # Yield the validated chunk to client
                        yield self.format_sse(request.chatId, validated_chunk)
                except GeneratorExit:
                    stream_ctx.mark_cancelled()
                    logger.info(
                        f"[{request.chatId}] [{stream_ctx.stream_id}] Client disconnected during guardrails streaming"
                    )
                    raise

                logger.info(
                    f"[{request.chatId}] [{stream_ctx.stream_id}] Stream completed successfully ({chunk_count} chunks)"
                )

                # Send document references before END token
                doc_references = self._extract_document_references(relevant_chunks)
                if doc_references:
                    refs_text = REFERENCES_SECTION_HEADER + "\n".join(
                        f"{i + 1}. [{ref.document_url}]({ref.document_url})"
                        for i, ref in enumerate(doc_references)
                    )
                    yield self.format_sse(request.chatId, refs_text)

                yield self.format_sse(request.chatId, "END")

            else:
                # No guardrails - stream directly
                logger.warning(
                    f"[{request.chatId}] [{stream_ctx.stream_id}] Streaming without guardrails validation"
                )
                chunk_count = 0
                async for token in bot_generator:
                    chunk_count += 1

                    token_estimate = len(token) // 4
                    stream_ctx.token_count += token_estimate
                    accumulated_response.append(token)

                    if stream_ctx.token_count > StreamConfig.MAX_TOKENS_PER_STREAM:
                        logger.error(
                            f"[{request.chatId}] [{stream_ctx.stream_id}] Token limit exceeded (no guardrails)"
                        )
                        yield self.format_sse(
                            request.chatId, STREAM_TOKEN_LIMIT_MESSAGE
                        )
                        yield self.format_sse(request.chatId, "END")
                        stream_ctx.mark_completed()
                        return

                    yield self.format_sse(request.chatId, token)

                # Send document references before END token
                doc_references = self._extract_document_references(relevant_chunks)
                if doc_references:
                    refs_text = REFERENCES_SECTION_HEADER + "\n".join(
                        f"{i + 1}. [{ref.document_url}]({ref.document_url})"
                        for i, ref in enumerate(doc_references)
                    )
                    yield self.format_sse(request.chatId, refs_text)

                yield self.format_sse(request.chatId, "END")

            # Extract usage after streaming completes. Output-rail validation runs
            # interleaved with generation in the same history window, so split the
            # two - folding them together hid a 100x guardrail cost regression.
            usage_info, guardrails_usage = get_lm_usage_since_split(
                history_length_before
            )
            costs_metric["streaming_generation"] = usage_info
            if guardrails_usage.get("num_calls", 0) > 0:
                costs_metric["output_guardrails"] = guardrails_usage

            # Record timings
            time_metric["streaming_generation"] = time.time() - streaming_step_start
            time_metric["output_guardrails"] = 0.0  # Inline during streaming

            # Calculate streaming duration
            streaming_duration = (datetime.now() - streaming_start_time).total_seconds()
            logger.info(
                f"[{request.chatId}] [{stream_ctx.stream_id}] Streaming completed in {streaming_duration:.2f}s"
            )

            # Log costs and trace
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)

            # Langfuse tracking
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client

                metadata_payload = {
                    "streaming": True,
                    "streaming_duration_seconds": streaming_duration,
                    "chunks_streamed": chunk_count,
                    "cost_breakdown": costs_metric,
                    "chat_id": request.chatId,
                    "environment": request.environment,
                    "stream_id": stream_ctx.stream_id,
                }

                try:
                    langfuse.update_current_generation(
                        metadata=metadata_payload,
                    )
                    langfuse.flush()
                except Exception as langfuse_error:
                    logger.error(
                        f"Langfuse streaming metadata update failed: {langfuse_error}",
                        exc_info=True,
                    )

            # Store inference data (for production and testing environments)
            # Set RAG data on request for unified storage method
            setattr(request, "_rag_refined_questions", refined_output.refined_questions)  # noqa: B010
            setattr(request, "_rag_ranked_chunks", relevant_chunks)  # noqa: B010
            try:
                await self.store_streaming_inference(
                    request=request,
                    final_answer="".join(accumulated_response),
                )
            except Exception as storage_error:
                logger.error(
                    f"Storage failed for chat_id: {request.chatId}, environment: {request.environment} - {str(storage_error)}"
                )

            # Persist conversation history (RAG streaming)
            if self.conversation_history_store is not None:
                _rag_bot_message = "".join(accumulated_response)
                if _rag_bot_message not in _HISTORY_EXCLUDED_MESSAGES:
                    await save_history_round(
                        self.conversation_history_store,
                        request.chatId,
                        request.message,
                        _rag_bot_message,
                    )

            # Mark stream as completed successfully
            stream_ctx.mark_completed()

        except GeneratorExit:
            # Client disconnected - mark as cancelled
            stream_ctx.mark_cancelled()
            logger.info(
                f"[{request.chatId}] [{stream_ctx.stream_id}] Client disconnected"
            )
            usage_info = get_lm_usage_since(history_length_before)
            costs_metric["streaming_generation"] = usage_info
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)

            # Update budget even on client disconnect
            self._update_connection_budget(
                request.connection_id, costs_metric, request.environment
            )
            raise
        except Exception as stream_error:
            error_id = generate_error_id()
            stream_ctx.mark_error(error_id)
            log_error_with_context(
                logger,
                error_id,
                "streaming_generation",
                request.chatId,
                stream_error,
            )
            yield self.format_sse(request.chatId, TECHNICAL_ISSUE_MESSAGE)
            yield self.format_sse(request.chatId, "END")

            usage_info = get_lm_usage_since(history_length_before)
            costs_metric["streaming_generation"] = usage_info
            self.log_costs(costs_metric)
            log_step_timings(time_metric, request.chatId)

    def format_sse(
        self,
        chat_id: str,
        content: str,
        buttons: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Format SSE message with exact specification.

        Args:
            chat_id: Chat/channel identifier
            content: Content to send (token, "END", error message, etc.)
            buttons: Optional list of choice button dicts for MCQ step responses

        Returns:
            SSE-formatted string: "data: {json}\\n\\n"
        """

        inner_payload: Dict[str, Any] = {"content": content}
        if buttons:
            inner_payload["buttons"] = buttons

        payload: Dict[str, Any] = {
            "chatId": chat_id,
            "payload": inner_payload,
            "timestamp": str(int(datetime.now().timestamp() * 1000)),
            "sentTo": [],
        }
        return f"data: {json_module.dumps(payload)}\n\n"

    @observe(name="initialize_service_components", as_type="span")
    def _initialize_service_components(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Initialize all service components and return them as a dictionary."""
        components: Dict[str, Any] = {}

        # Initialize LLM Manager
        llm_manager = self._initialize_llm_manager(
            environment=request.environment, connection_id=request.connection_id
        )
        components["llm_manager"] = llm_manager

        # Store resolved connection_id on request for downstream use (budget, inference storage)
        if llm_manager.connection_id and not request.connection_id:
            request.connection_id = llm_manager.connection_id
            logger.debug(
                f"Stored resolved vault_uuid on request: {llm_manager.connection_id}"
            )

        if request.environment in self.shared_guardrails_adapters:
            logger.info(
                f" Using shared guardrails adapter for environment='{request.environment}' "
                f"(startup-initialized, zero overhead)"
            )
            components["guardrails_adapter"] = self.shared_guardrails_adapters[
                request.environment
            ]
        else:
            logger.warning(
                f" Shared guardrails unavailable for environment='{request.environment}', "
                f"initializing per-request (slower)"
            )
            components["guardrails_adapter"] = self._safe_initialize_guardrails(
                request.environment, request.connection_id
            )

        # Initialize Contextual Retriever (cached by environment + connection_id)
        retriever_key = (request.environment, request.connection_id)
        if retriever_key in self._retriever_cache:
            components["contextual_retriever"] = self._retriever_cache[retriever_key]
            logger.info(f"Using cached ContextualRetriever for key={retriever_key}")
        else:
            with self._component_cache_lock:
                if retriever_key in self._retriever_cache:
                    components["contextual_retriever"] = self._retriever_cache[
                        retriever_key
                    ]
                else:
                    retriever = self._safe_initialize_contextual_retriever(
                        request.environment, request.connection_id
                    )
                    if retriever is not None:
                        self._retriever_cache[retriever_key] = retriever
                    components["contextual_retriever"] = retriever

        # Initialize Response Generator (fresh per request - NOT cached)
        # ResponseGeneratorAgent uses dspy.streamify() which has internal state
        # that doesn't reset between calls, causing 0-token streaming on reuse.
        # Only costs ~0.02s to create, so caching is not worth the risk.
        components["response_generator"] = self._safe_initialize_response_generator(
            components["llm_manager"]
        )

        # Log optimization status for all components
        self._log_optimization_status(components)

        return components

    def _log_optimization_status(self, components: Dict[str, Any]) -> None:
        """Log optimization status for all initialized components."""
        try:
            logger.info("=== OPTIMIZATION STATUS ===")

            self._log_guardrails_status(components)
            self._log_refiner_status(components)
            self._log_generator_status(components)

            logger.info("=== END OPTIMIZATION STATUS ===")

        except Exception as e:
            logger.warning(f"Failed to log optimization status: {str(e)}")

    def _log_guardrails_status(self, components: Dict[str, Any]) -> None:
        """Log guardrails optimization status."""
        if not components.get("guardrails_adapter"):
            logger.info(" Guardrails: Not initialized")
            return

        try:
            from src.guardrails.optimized_guardrails_loader import get_guardrails_loader

            guardrails_loader = get_guardrails_loader()
            _, metadata = guardrails_loader.get_optimized_config_path()

            if metadata.get("optimized", False):
                logger.info(
                    f" Guardrails: OPTIMIZED (version: {metadata.get('version', 'unknown')})"
                )
                metrics = metadata.get("metrics", {})
                if metrics:
                    logger.info(
                        f"  Metrics: weighted_accuracy={metrics.get('weighted_accuracy', 'N/A')}"
                    )
            else:
                logger.info(" Guardrails: BASE (no optimization)")
        except Exception as e:
            logger.warning(f" Guardrails: Status check failed - {str(e)}")

    def _log_refiner_status(self, components: Dict[str, Any]) -> None:
        """Log refiner optimization status."""
        if not hasattr(components.get("llm_manager"), "__class__"):
            logger.info(" Refiner: LLM Manager not available")
            return

        try:
            from src.prompt_refine_manager.prompt_refiner import PromptRefinerAgent

            test_refiner = PromptRefinerAgent(llm_manager=components["llm_manager"])
            refiner_info = test_refiner.get_module_info()

            if refiner_info.get("optimized", False):
                logger.info(
                    f" Refiner: OPTIMIZED (version: {refiner_info.get('version', 'unknown')})"
                )
                metrics = refiner_info.get("metrics", {})
                if metrics:
                    logger.info(
                        f"  Metrics: avg_quality={metrics.get('average_quality', 'N/A')}"
                    )
            else:
                logger.info(" Refiner: BASE (no optimization)")
        except Exception as e:
            logger.warning(f" Refiner: Status check failed - {str(e)}")

    def _log_generator_status(self, components: Dict[str, Any]) -> None:
        """Log generator optimization status."""
        if not components.get("response_generator"):
            logger.info(" Generator: Not initialized")
            return

        try:
            generator_info = components["response_generator"].get_module_info()

            if generator_info.get("optimized", False):
                logger.info(
                    f" Generator: OPTIMIZED (version: {generator_info.get('version', 'unknown')})"
                )
                metrics = generator_info.get("metrics", {})
                if metrics:
                    logger.info(
                        f"  Metrics: avg_quality={metrics.get('average_quality', 'N/A')}"
                    )
            else:
                logger.info(" Generator: BASE (no optimization)")
        except Exception as e:
            logger.warning(f" Generator: Status check failed - {str(e)}")

    @observe(name="execute_orchestration_pipeline", as_type="span")
    async def _execute_orchestration_pipeline(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
        prefix: str = "",
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Execute the main orchestration pipeline with all components.

        Args:
            request: Orchestration request
            components: Initialized service components
            costs_metric: Dictionary for cost tracking
            time_metric: Dictionary for timing tracking
            prefix: Optional prefix for timing keys (e.g., "rag" for workflow namespacing)
        """
        # Note: Query validation AND input guardrails check now happen at orchestration level
        # (in process_orchestration_request) BEFORE classifier routing for true early rejection.
        # This saves ~3.5s on blocked requests by failing fast before expensive workflow operations.

        # Step 1: Refine user prompt
        start_time = time.time()
        conversation_history, conversation_summary = await get_conversation_history(
            chat_id=request.chatId,
            store=self.conversation_history_store,
            fallback=request.conversationHistory,
        )
        refined_output, refiner_usage = self._refine_user_prompt(
            llm_manager=components["llm_manager"],
            original_message=request.message,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        timing_key = f"{prefix}.prompt_refiner" if prefix else "prompt_refiner"
        time_metric[timing_key] = time.time() - start_time
        costs_metric["prompt_refiner"] = refiner_usage

        # Step 2: Retrieve relevant chunks using contextual retrieval
        try:
            start_time = time.time()
            relevant_chunks = await self._safe_retrieve_contextual_chunks(
                components["contextual_retriever"], refined_output, request
            )
            timing_key = (
                f"{prefix}.contextual_retrieval" if prefix else "contextual_retrieval"
            )
            time_metric[timing_key] = time.time() - start_time
        except (
            ContextualRetrieverInitializationError,
            ContextualRetrievalFailureError,
        ) as e:
            logger.warning(f"Contextual retrieval failed: {str(e)}")
            return self._create_out_of_scope_response(request)

        # Handle zero chunks scenario - return out-of-scope response
        if len(relevant_chunks) == 0:
            logger.info("No relevant chunks found - returning out-of-scope response")
            return self._create_out_of_scope_response(request)

        # Step 3: Generate response
        start_time = time.time()
        generated_response = self._generate_rag_response(
            llm_manager=components["llm_manager"],
            request=request,
            refined_output=refined_output,
            relevant_chunks=relevant_chunks,
            response_generator=components["response_generator"],
            costs_metric=costs_metric,
        )
        timing_key = (
            f"{prefix}.response_generation" if prefix else "response_generation"
        )
        time_metric[timing_key] = time.time() - start_time

        # Populate retrieval_context for eval mode (DeepEval metrics need this)
        eval_mode = os.getenv("EVAL_MODE", "false").lower() == "true"
        if eval_mode and isinstance(generated_response, OrchestrationResponse):
            eval_chunks: List[Dict[str, Any]] = []
            for chunk in relevant_chunks:
                meta = chunk.get("meta", {})
                meta_dict = meta if isinstance(meta, dict) else {}
                eval_chunks.append(
                    {
                        "content": chunk.get("content", chunk.get("text", "")),
                        "metadata": {
                            "fused_score": meta_dict.get(
                                "fused_score", chunk.get("fused_score", 0)
                            ),
                            "bm25_score": meta_dict.get(
                                "bm25_score", chunk.get("bm25_score", 0)
                            ),
                            "semantic_score": meta_dict.get(
                                "semantic_score", chunk.get("semantic_score", 0)
                            ),
                        },
                    }
                )
            generated_response.retrieval_context = eval_chunks

        # Step 4: Output Guardrails Check
        # Apply guardrails to all response types for consistent safety across all environments
        start_time = time.time()
        output_guardrails_response = await self.handle_output_guardrails(
            components["guardrails_adapter"],
            generated_response,
            request,
            costs_metric,
        )
        timing_key = (
            f"{prefix}.output_guardrails_check" if prefix else "output_guardrails_check"
        )
        time_metric[timing_key] = time.time() - start_time

        # Step 5: Store inference data (for production and testing environments)
        # Only store OrchestrationResponse (has chatId), not TestOrchestrationResponse
        if request.environment in [
            PRODUCTION_DEPLOYMENT_ENVIRONMENT,
            TEST_DEPLOYMENT_ENVIRONMENT,
        ] and isinstance(output_guardrails_response, OrchestrationResponse):
            try:
                # Set RAG data on request for unified storage method
                request._rag_refined_questions = refined_output.refined_questions  # type: ignore[attr-defined]
                request._rag_ranked_chunks = relevant_chunks  # type: ignore[attr-defined]

                # Run async storage in a new event loop (sync context)
                import threading

                def _store_async() -> None:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.store_streaming_inference(
                                request=request,
                                final_answer=output_guardrails_response.content,
                            )
                        )
                        loop.close()
                    except Exception as e:
                        logger.error(f"Error in async storage thread: {str(e)}")

                storage_thread = threading.Thread(target=_store_async, daemon=True)
                storage_thread.start()
            except Exception as storage_error:
                # Log storage error but don't fail the request
                logger.error(
                    f"Storage failed for chat_id: {request.chatId}, environment: {request.environment} - {str(storage_error)}"
                )

        return output_guardrails_response

    def _safe_initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[NeMoRailsAdapter]:
        """Safely initialize guardrails adapter with error handling."""
        try:
            adapter = self._initialize_guardrails(environment, connection_id)
            logger.info("Guardrails adapter initialization successful")
            return adapter
        except Exception as guardrails_error:
            logger.warning(f"Guardrails initialization failed: {str(guardrails_error)}")
            logger.warning("Continuing without guardrails protection")
            return None

    @observe(name="safe_initialize_contextual_retriever", as_type="span")
    def _safe_initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[ContextualRetriever]:
        """Safely initialize contextual retriever with error handling."""
        try:
            retriever = self._initialize_contextual_retriever(
                environment, connection_id
            )
            logger.info("Contextual Retriever initialization successful")
            return retriever
        except Exception as retriever_error:
            logger.warning(
                f"Contextual Retriever initialization failed: {str(retriever_error)}"
            )
            logger.warning("Continuing without chunk retrieval capabilities")
            return None

    @observe(name="safe_initialize_response_generator", as_type="span")
    def _safe_initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> Optional[ResponseGeneratorAgent]:
        """Safely initialize response generator with error handling."""
        try:
            generator = self._initialize_response_generator(llm_manager)
            logger.info("Response Generator initialization successful")
            return generator
        except Exception as generator_error:
            logger.warning(
                f"Response Generator initialization failed: {str(generator_error)}"
            )
            return None

    async def handle_input_guardrails(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse, None]:
        """Check input guardrails and return blocked response if needed."""
        input_check_result = await self._check_input_guardrails_async(
            guardrails_adapter=guardrails_adapter,
            user_message=request.message,
            costs_metric=costs_metric,
        )

        if not input_check_result.allowed:
            logger.warning(f"Input blocked by guardrails: {input_check_result.reason}")

            # Get localized message based on detected language
            detected_lang = getattr(request, "_detected_language", "en")
            localized_msg = get_localized_message(
                INPUT_GUARDRAIL_VIOLATION_MESSAGES, detected_lang
            )

            if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                logger.info(
                    "Test environment detected – returning input guardrail violation message."
                )
                return TestOrchestrationResponse(
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=True,
                    content=localized_msg,
                    chunks=None,
                )
            else:
                return OrchestrationResponse(
                    chatId=request.chatId,
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=True,
                    content=localized_msg,
                )

        logger.info("Input guardrails check passed")
        return None

    def _safe_retrieve_contextual_chunks_sync(
        self,
        contextual_retriever: Optional[ContextualRetriever],
        refined_output: PromptRefinerOutput,
        request: OrchestrationRequest,
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """Synchronous wrapper for _safe_retrieve_contextual_chunks for non-streaming pipeline."""

        try:
            # Check if there's a running event loop
            try:
                asyncio.get_running_loop()
                # If we get here, there IS a running event loop; cannot use asyncio.run()
                raise ContextualRetrievalFailureError(
                    "Cannot call _safe_retrieve_contextual_chunks_sync from an async context with a running event loop. "
                    "Please use the async version _safe_retrieve_contextual_chunks instead."
                )
            except RuntimeError:
                # No running loop (get_running_loop raised RuntimeError), safe to use asyncio.run()
                pass

            return asyncio.run(
                self._safe_retrieve_contextual_chunks(
                    contextual_retriever, refined_output, request
                )
            )
        except (
            ContextualRetrieverInitializationError,
            ContextualRetrievalFailureError,
        ):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Error in synchronous contextual chunks retrieval: {str(e)}")
            raise ContextualRetrievalFailureError(
                f"Synchronous contextual retrieval wrapper failed: {str(e)}"
            ) from e

    async def _safe_retrieve_contextual_chunks(
        self,
        contextual_retriever: Optional[ContextualRetriever],
        refined_output: PromptRefinerOutput,
        request: OrchestrationRequest,
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """Safely retrieve chunks using contextual retrieval with error handling."""
        if not contextual_retriever:
            logger.info("Contextual Retriever not available, skipping chunk retrieval")
            return []

        try:
            # Ensure retriever is initialized
            if not contextual_retriever.initialized:
                initialization_success = await contextual_retriever.initialize()
                if not initialization_success:
                    logger.error("Failed to initialize contextual retriever")
                    raise ContextualRetrieverInitializationError(
                        "Contextual retriever failed to initialize"
                    )

            # Call the async method directly (DO NOT use asyncio.run())
            relevant_chunks = await contextual_retriever.retrieve_contextual_chunks(
                original_question=refined_output.original_question,
                refined_questions=refined_output.refined_questions,
                environment=request.environment,
                connection_id=request.connection_id,
            )

            logger.info(
                f"Successfully retrieved {len(relevant_chunks)} contextual chunks"
            )
            return relevant_chunks
        except ContextualRetrieverInitializationError:
            # Re-raise our custom exceptions
            raise
        except Exception as retrieval_error:
            logger.error(f"Contextual chunk retrieval failed: {str(retrieval_error)}")
            raise ContextualRetrievalFailureError(
                f"Contextual chunk retrieval failed: {str(retrieval_error)}"
            ) from retrieval_error

    async def handle_output_guardrails(
        self,
        guardrails_adapter: Optional[NeMoRailsAdapter],
        generated_response: Union[OrchestrationResponse, TestOrchestrationResponse],
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Check output guardrails and handle blocked responses for both response types."""
        # Determine if we should run guardrails (same logic for both response types)
        should_check_guardrails = (
            guardrails_adapter is not None
            and generated_response.llmServiceActive
            and not generated_response.questionOutOfLLMScope
        )

        if should_check_guardrails:
            # Type assertion: should_check_guardrails guarantees guardrails_adapter is not None
            assert guardrails_adapter is not None
            output_check_result = await self._check_output_guardrails(
                guardrails_adapter=guardrails_adapter,
                assistant_message=generated_response.content,
                costs_metric=costs_metric,
            )

            if not output_check_result.allowed:
                logger.warning(
                    f"Output blocked by guardrails: {output_check_result.reason}"
                )
                # Get localized message based on detected language
                detected_lang = getattr(request, "_detected_language", "en")
                localized_msg = get_localized_message(
                    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES, detected_lang
                )

                # Return appropriate response type based on original response type
                if isinstance(generated_response, TestOrchestrationResponse):
                    return TestOrchestrationResponse(
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=localized_msg,
                        chunks=None,
                    )
                else:
                    return OrchestrationResponse(
                        chatId=request.chatId,
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=localized_msg,
                    )

            logger.info("Output guardrails check passed")
        else:
            logger.info("Skipping output guardrails check")

        logger.info(f"Successfully generated RAG response for chatId: {request.chatId}")
        return generated_response

    def _create_error_response(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Create standardized error response with localized message."""
        # Get language from request (set during language detection)
        detected_lang = getattr(request, "_detected_language", "en")
        localized_message = get_localized_message(
            TECHNICAL_ISSUE_MESSAGES, detected_lang
        )

        return OrchestrationResponse(
            chatId=request.chatId,
            llmServiceActive=False,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=localized_message,
        )

    def _create_out_of_scope_response(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Create standardized out-of-scope response with localized message."""
        # Get language from request (set during language detection)
        detected_lang = getattr(request, "_detected_language", "en")
        localized_message = get_localized_message(OUT_OF_SCOPE_MESSAGES, detected_lang)

        return OrchestrationResponse(
            chatId=request.chatId,
            llmServiceActive=True,
            questionOutOfLLMScope=True,
            inputGuardFailed=False,
            content=localized_message,
        )

    def _extract_content_from_sse(self, sse_chunk: str) -> Optional[str]:
        """
        Extract content from an SSE-formatted chunk.

        Args:
            sse_chunk: SSE-formatted string like 'data: {"chatId": ..., "payload": {"content": "..."}}\n\n'

        Returns:
            The content string, or None if parsing fails
        """
        try:
            # SSE format: 'data: {...}\n\n'
            if not sse_chunk.startswith("data: "):
                return None
            json_str = sse_chunk[6:].strip()  # Remove 'data: ' prefix
            if not json_str:
                return None
            parsed = json_module.loads(json_str)
            return parsed.get("payload", {}).get("content")
        except (json_module.JSONDecodeError, AttributeError, TypeError):
            return None

    async def store_streaming_inference(
        self,
        request: OrchestrationRequest,
        final_answer: str,
    ) -> None:
        """
        Store streaming inference data.

        Checks for RAG data on request attributes (_rag_refined_questions, _rag_chunks).
        If not present, uses empty arrays. This enables unified storage for all
        streaming workflows (RAG, Service, Context, API Tool, etc.).

        Args:
            request: Orchestration request (may have RAG data as attributes)
            final_answer: Complete streamed response
        """
        # Only store for production and testing environments
        if request.environment not in [
            PRODUCTION_DEPLOYMENT_ENVIRONMENT,
            TEST_DEPLOYMENT_ENVIRONMENT,
        ]:
            return

        try:
            # Get RAG data from request attributes if available (set by _stream_rag_pipeline)
            refined_questions: List[str] = getattr(
                request, "_rag_refined_questions", []
            )
            ranked_chunks: List[Dict[str, Any]] = getattr(
                request, "_rag_ranked_chunks", []
            )

            # Extract embedding scores from chunks
            embedding_scores: List[float] = []
            for chunk in ranked_chunks:
                score_value = chunk.get("fused_score", chunk.get("score", 0.0))
                try:
                    if isinstance(score_value, (int, float)):
                        embedding_scores.append(float(score_value))
                    else:
                        embedding_scores.append(0.0)
                except (ValueError, TypeError):
                    embedding_scores.append(0.0)

            # Convert conversation history to list of dicts
            conversation_history_list = [
                {"role": item.authorRole, "content": item.message}
                for item in (request.conversationHistory or [])
            ]

            # Get the production store instance
            production_store = get_production_store()

            # Store the inference result (async)
            result = await production_store.store_inference_result_async(
                chat_id=request.chatId,
                user_question=request.message,
                refined_questions=refined_questions,
                conversation_history=conversation_history_list,
                ranked_chunks=ranked_chunks,
                embedding_scores=embedding_scores,
                final_answer=final_answer,
                environment=request.environment,
                vault_uuid=request.connection_id,
            )

            if result["success"]:
                logger.info(
                    f"Successfully stored streaming inference for chat_id: {request.chatId}, "
                    f"environment: {request.environment}, "
                    f"has_rag_data: {bool(refined_questions)}"
                )
            else:
                logger.warning(
                    f"Failed to store streaming inference for chat_id: {request.chatId} - "
                    f"Error: {result['error']}"
                )

        except Exception as e:
            # Log the error but don't fail the request
            logger.error(
                f"Error storing streaming inference for chat_id: {request.chatId} - {str(e)}"
            )

    def _initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> NeMoRailsAdapter:
        """
        Initialize NeMo Guardrails adapter.

        Args:
            environment: Environment context (production/testing/development)
            connection_id: Optional connection identifier

        Returns:
            NeMoRailsAdapter: Initialized guardrails adapter instance

        Raises:
            Exception: For initialization errors
        """
        try:
            logger.info(f"Initializing Guardrails for environment: {environment}")

            guardrails_adapter = NeMoRailsAdapter(
                environment=environment, connection_id=connection_id
            )

            return guardrails_adapter

        except Exception as e:
            logger.error(f"Failed to initialize Guardrails adapter: {str(e)}")
            raise

    @observe(name="check_input_guardrails", as_type="span")
    async def _check_input_guardrails_async(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        user_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """
        Check user input against guardrails and track costs (async version).

        Args:
            guardrails_adapter: The guardrails adapter instance
            user_message: The user message to check
            costs_metric: Dictionary to store cost information

        Returns:
            GuardrailCheckResult: Result of the guardrail check
        """
        logger.info("Starting input guardrails check")

        try:
            # Use async version for streaming context
            result = await guardrails_adapter.check_input_async(user_message)

            # Store guardrail costs
            costs_metric["input_guardrails"] = result.usage
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    input=user_message,
                    metadata={
                        "guardrail_type": "input",
                        "allowed": result.allowed,
                        "verdict": result.verdict,
                        "blocked_reason": result.reason if not result.allowed else None,
                        "error": result.error if result.error else None,
                    },
                )
            logger.info(
                f"Input guardrails check completed: allowed={result.allowed}, "
                f"cost=${result.usage.get('total_cost', 0):.6f}"
            )

            return result

        except Exception as e:
            logger.error(f"Input guardrails check failed: {str(e)}")
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "guardrail_type": "input",
                    }
                )
            # Return conservative result on error
            return GuardrailCheckResult(
                allowed=False,
                verdict="yes",
                content="Error during input guardrail check",
                error=str(e),
                usage={},
            )

    @observe(name="check_input_guardrails", as_type="span")
    def _check_input_guardrails(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        user_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """
        Check user input against guardrails and track costs (sync version for non-streaming).

        Args:
            guardrails_adapter: The guardrails adapter instance
            user_message: The user message to check
            costs_metric: Dictionary to store cost information

        Returns:
            GuardrailCheckResult: Result of the guardrail check
        """
        logger.info("Starting input guardrails check")

        try:
            result = guardrails_adapter.check_input(user_message)

            # Store guardrail costs
            costs_metric["input_guardrails"] = result.usage
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    input=user_message,
                    metadata={
                        "guardrail_type": "input",
                        "allowed": result.allowed,
                        "verdict": result.verdict,
                        "blocked_reason": result.reason if not result.allowed else None,
                        "error": result.error if result.error else None,
                    },
                )
            logger.info(
                f"Input guardrails check completed: allowed={result.allowed}, "
                f"cost=${result.usage.get('total_cost', 0):.6f}"
            )

            return result

        except Exception as e:
            logger.error(f"Input guardrails check failed: {str(e)}")
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "guardrail_type": "input",
                    }
                )
            # Return conservative result on error
            return GuardrailCheckResult(
                allowed=False,
                verdict="yes",
                content="Error during input guardrail check",
                error=str(e),
                usage={},
            )

    @observe(name="check_output_guardrails", as_type="span")
    async def _check_output_guardrails(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        assistant_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """
        Check assistant output against guardrails and track costs.

        Args:
            guardrails_adapter: The guardrails adapter instance
            assistant_message: The assistant message to check
            costs_metric: Dictionary to store cost information

        Returns:
            GuardrailCheckResult: Result of the guardrail check
        """
        logger.info("Starting output guardrails check")

        try:
            result = await guardrails_adapter.check_output_async(assistant_message)

            # Store guardrail costs
            costs_metric["output_guardrails"] = result.usage
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    input=assistant_message[:500],  # Truncate for readability
                    output=result.verdict,
                    metadata={
                        "guardrail_type": "output",
                        "allowed": result.allowed,
                        "verdict": result.verdict,
                        "reason": result.reason if not result.allowed else None,
                        "error": result.error if result.error else None,
                        "response_length": len(assistant_message),
                    },
                )
            logger.info(
                f"Output guardrails check completed: allowed={result.allowed}, "
                f"cost=${result.usage.get('total_cost', 0):.6f}"
            )

            return result

        except Exception as e:
            logger.error(f"Output guardrails check failed: {str(e)}")
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    metadata={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "guardrail_type": "output",
                    }
                )
            # Return conservative result on error
            return GuardrailCheckResult(
                allowed=False,
                verdict="yes",
                content="Error during output guardrail check",
                error=str(e),
                usage={},
            )

    def log_costs(self, costs_metric: Dict[str, Dict[str, Any]]) -> None:
        """
        Log cost information for tracking.

        Args:
            costs_metric: Dictionary of costs per component
        """
        try:
            if not costs_metric:
                return

            total_costs = calculate_total_costs(costs_metric)

            logger.info("LLM USAGE COSTS BREAKDOWN:")

            for component, costs in costs_metric.items():
                logger.info(
                    f"  {component:20s}: ${costs.get('total_cost', 0):.6f} "
                    f"({costs.get('num_calls', 0)} calls, "
                    f"{costs.get('total_tokens', 0)} tokens)"
                )

            logger.info(
                f"  {'TOTAL':20s}: ${total_costs['total_cost']:.6f} "
                f"({total_costs['total_calls']} calls, "
                f"{total_costs['total_tokens']} tokens)"
            )

            # Log module versions being used
            logger.info("\nMODULE VERSIONS IN USE:")
            try:
                from src.optimization.optimized_module_loader import get_module_loader
                from src.guardrails.optimized_guardrails_loader import (
                    get_guardrails_loader,
                )

                loader = get_module_loader()
                guardrails_loader = get_guardrails_loader()

                # Log refiner version (uses cache, no disk I/O)
                refiner_meta = loader.get_module_metadata("refiner")
                logger.info(
                    f"  Refiner: {refiner_meta.get('version', 'unknown')} "
                    f"({'optimized' if refiner_meta.get('optimized') else 'base'})"
                )

                # Log generator version (uses cache, no disk I/O)
                generator_meta = loader.get_module_metadata("generator")
                logger.info(
                    f"  Generator: {generator_meta.get('version', 'unknown')} "
                    f"({'optimized' if generator_meta.get('optimized') else 'base'})"
                )

                # Log guardrails version
                _, guardrails_meta = guardrails_loader.get_optimized_config_path()
                logger.info(
                    f"  Guardrails: {guardrails_meta.get('version', 'unknown')} "
                    f"({'optimized' if guardrails_meta.get('optimized') else 'base'})"
                )

            except Exception as version_error:
                logger.debug(f"Could not log module versions: {str(version_error)}")

        except Exception as e:
            logger.warning(f"Failed to log costs: {str(e)}")

    def _update_connection_budget(
        self,
        connection_id: Optional[str],
        costs_metric: Dict[str, Dict[str, Any]],
        environment: str = "development",
    ) -> None:
        """
        Update the budget for an LLM connection based on usage costs.

        Args:
            connection_id: The vault_uuid identifying the LLM connection
            costs_metric: Dictionary of costs per component
            environment: The deployment environment (production/testing/development)
        """
        try:
            budget_tracker = get_budget_tracker()

            result = budget_tracker.update_budget_from_costs(
                connection_id, costs_metric
            )

            if result.get("success"):
                if result.get("budget_exceeded"):
                    logger.warning(
                        f"Budget threshold exceeded for connection_id={connection_id}. "
                        "Connection may have been deactivated."
                    )
                else:
                    logger.debug(
                        f"Budget updated successfully for uuid={connection_id}"
                    )
            else:
                reason = result.get("reason", "unknown")
                if reason not in ["no_connection_id", "zero_or_negative_cost"]:
                    logger.warning(
                        f"Failed to update budget for connection_id={connection_id}. "
                        f"Reason: {reason}"
                    )

        except Exception as e:
            # Don't fail the orchestration if budget update fails
            logger.error(f"Error updating budget: {str(e)}")

    @observe(name="initialize_llm_manager", as_type="span")
    def _initialize_llm_manager(
        self, environment: str, connection_id: Optional[str]
    ) -> LLMManager:
        """
        Initialize LLM Manager with proper configuration.

        For production environment, resolves vault_uuid from DB if not provided.
        For testing environment, connection_id (vault_uuid) must be provided.

        Args:
            environment: Environment context (production/testing)
            connection_id: Vault UUID for the connection (required for testing, auto-resolved for production)

        Returns:
            LLMManager: Initialized LLM manager instance (with connection_id property)
        """
        try:
            logger.info(f"Initializing LLM Manager for environment: {environment}")

            resolved_connection_id = connection_id

            # Resolve vault_uuid for production if not provided
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

    @observe(name="refine_user_prompt", as_type="generation")
    def _refine_user_prompt(
        self,
        llm_manager: LLMManager,
        original_message: str,
        conversation_history: List[ConversationItem],
        conversation_summary: Optional[str] = None,
    ) -> tuple[PromptRefinerOutput, Dict[str, Any]]:
        """
        Refine user prompt using loaded LLM configuration and return usage info.

        Args:
            llm_manager: The LLM manager instance to use
            original_message: The original user message to refine
            conversation_history: Previous conversation context
            conversation_summary: Optional summary of earlier conversation rounds
                that were evicted from Redis.  When provided it is prepended to the
                DSPy history as a ``system`` turn so the refiner can use it for
                context without re-summarising via an LLM call.

        Returns:
            Tuple of (PromptRefinerOutput, usage_dict): The refined prompt output and usage info

        Raises:
            ValueError: When LLM Manager is not initialized
            ValidationError: When prompt refinement output validation fails
            Exception: For other prompt refinement failures
        """
        logger.info("Starting prompt refinement process")

        try:
            # Convert conversation history to DSPy format, optionally prepending
            # a pre-computed summary of earlier (evicted) conversation rounds.
            history: List[Dict[str, str]] = []
            if conversation_summary:
                history.append(
                    {
                        "role": "system",
                        "content": f"Summary of earlier conversation: {conversation_summary}",
                    }
                )
            for item in conversation_history:
                role = "assistant" if item.authorRole == "bot" else item.authorRole
                history.append({"role": role, "content": item.message})

            # Create prompt refiner using the same LLM manager instance
            refiner = PromptRefinerAgent(llm_manager=llm_manager)

            # Generate structured prompt refinement output with usage tracking
            refinement_result = refiner.forward_structured(
                history=history, question=original_message
            )

            # Extract usage information
            usage_info = refinement_result.get(
                "usage",
                {
                    "total_cost": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "num_calls": 0,
                },
            )

            # Validate the output schema using Pydantic
            try:
                validated_output = PromptRefinerOutput(
                    original_question=refinement_result["original_question"],
                    refined_questions=refinement_result["refined_questions"],
                )
            except Exception as validation_error:
                logger.error(
                    f"Prompt refinement output validation failed: {str(validation_error)}"
                )
                logger.error(f"Invalid refinement result: {refinement_result}")
                raise ValueError(
                    f"Prompt refinement validation failed: {str(validation_error)}"
                ) from validation_error
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                refinement_applied = (
                    original_message.strip()
                    != validated_output.original_question.strip()
                )
                langfuse.update_current_generation(
                    model=llm_manager.get_provider_info().get("model", "unknown"),
                    input=original_message,
                    usage_details={
                        "input": usage_info.get("total_prompt_tokens", 0),
                        "output": usage_info.get("total_completion_tokens", 0),
                        "total": usage_info.get("total_tokens", 0),
                    },
                    cost_details={
                        "total": usage_info.get("total_cost", 0.0),
                    },
                    metadata={
                        "num_calls": usage_info.get("num_calls", 0),
                        "num_refined_questions": len(
                            validated_output.refined_questions
                        ),
                        "refinement_applied": refinement_applied,
                        "conversation_history_length": len(history),
                    },  # type: ignore
                )
            output_json = validated_output.model_dump()
            logger.info(
                f"Prompt refinement output: {json_module.dumps(output_json, indent=2)}"
            )

            logger.info("Prompt refinement completed successfully")
            return validated_output, usage_info

        except ValueError:
            raise
        except Exception as e:
            error_id = generate_error_id()
            log_error_with_context(
                logger,
                error_id,
                "prompt_refinement",
                None,
                e,
                {"message_preview": original_message[:100]},
            )
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_generation(
                    metadata={
                        "error_id": error_id,
                        "error_type": type(e).__name__,
                        "refinement_failed": True,
                    }
                )
            raise RuntimeError(f"Prompt refinement process failed: {str(e)}") from e

    @observe(name="initialize_contextual_retriever", as_type="span")
    def _initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> ContextualRetriever:
        """
        Initialize contextual retriever for enhanced document retrieval.

        Args:
            environment: Environment for model resolution
            connection_id: Optional connection ID

        Returns:
            ContextualRetriever: Initialized contextual retriever instance
        """
        logger.info("Initializing contextual retriever")

        try:
            # Initialize with Qdrant URL - use environment variable or default
            qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

            contextual_retriever = ContextualRetriever(
                qdrant_url=qdrant_url,
                environment=environment,
                connection_id=connection_id,
                llm_service=self,  # Inject self to eliminate circular dependency
                shared_bm25=self.shared_bm25_search,  # Inject pre-warmed BM25 index
            )

            logger.info("Contextual retriever initialized successfully")
            return contextual_retriever

        except Exception as e:
            logger.error(f"Failed to initialize contextual retriever: {str(e)}")
            raise

    @observe(name="initialize_response_generator", as_type="span")
    def _initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> ResponseGeneratorAgent:
        """
        Initialize Response Generator with the provided LLM manager.

        Args:
            llm_manager: The LLM manager instance to use for response generation

        Returns:
            ResponseGeneratorAgent: Initialized response generator instance
        """
        logger.info("Initializing response generator")

        try:
            # Get custom instructions for response generation
            custom_prefix = self._get_custom_instructions_for_response_generation()

            # Set up DSPy configuration for the response generator
            with llm_manager.use_task_local():
                response_generator = ResponseGeneratorAgent(
                    custom_instructions_prefix=custom_prefix
                )

            logger.info("Response generator initialized successfully")
            return response_generator

        except Exception as e:
            logger.error(f"Failed to initialize response generator: {str(e)}")
            raise

    def _get_custom_instructions_for_response_generation(self) -> str:
        """
        Get custom prompt instructions for response generation only.

        Note: Applied only to ResponseGeneratorAgent, not PromptRefinerAgent.
        PromptRefiner focuses on query optimization for retrieval, while
        ResponseGenerator needs to follow language policy and interaction style
        for user-facing content.

        Returns:
            str: Custom instruction prefix for prepending to questions
        """
        try:
            custom_prompt = self.prompt_config_loader.get_custom_instructions()
            if custom_prompt:
                # Format for prepending to questions in ResponseGenerator
                return f"[SYSTEM INSTRUCTIONS]\n{custom_prompt}\n\n[USER QUESTION]\n"
            return ""
        except Exception as e:
            logger.error(f"Error retrieving custom instructions: {e}")
            return ""

    @staticmethod
    def _format_chunks_for_test_response(
        relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
    ) -> Optional[List[ChunkInfo]]:
        """
        Format retrieved chunks for test response.

        Args:
            relevant_chunks: List of retrieved chunks with metadata

        Returns:
            List of ChunkInfo objects with rank and content, or None if no chunks
        """
        if not relevant_chunks:
            return None

        formatted_chunks = []
        for rank, chunk in enumerate(relevant_chunks, start=1):
            # Extract text content - prefer "text" key, fallback to "content"
            chunk_text = chunk.get("text", chunk.get("content", ""))
            if isinstance(chunk_text, str) and chunk_text.strip():
                formatted_chunks.append(ChunkInfo(rank=rank, chunkRetrieved=chunk_text))

        return formatted_chunks if formatted_chunks else None

    @staticmethod
    def _extract_document_references(
        relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
    ) -> Optional[List[DocumentReference]]:
        """
        Extract unique document references from retrieved chunks.

        Args:
            relevant_chunks: List of retrieved chunks with metadata

        Returns:
            List of DocumentReference objects, or None if no chunks
        """
        if not relevant_chunks:
            return None

        seen_urls: set[str] = set()
        references: List[DocumentReference] = []

        for rank, chunk in enumerate(relevant_chunks, start=1):
            # Extract document_url - try multiple keys for robustness
            doc_url = chunk.get("document_url")
            if not doc_url:
                # Fallback to metadata
                meta = chunk.get("meta", {})
                if isinstance(meta, dict):
                    doc_url = (
                        meta.get("document_url")
                        or meta.get("source_file")
                        or meta.get("source")
                    )

            if doc_url and isinstance(doc_url, str) and doc_url.strip():
                # Only include unique URLs (deduplicate)
                if doc_url not in seen_urls:
                    seen_urls.add(doc_url)

                    # Extract score - try multiple keys, ensure it's a float
                    score_value = chunk.get("fused_score") or chunk.get("score", 0.0)
                    try:
                        if isinstance(score_value, (int, float)):
                            score = float(score_value)
                        else:
                            score = 0.0
                    except (ValueError, TypeError):
                        score = 0.0

                    references.append(
                        DocumentReference(
                            document_url=doc_url,
                            chunk_rank=rank,
                            relevance_score=round(score, 4),
                        )
                    )

        return references if references else None

    @observe(name="generate_rag_response", as_type="span")
    def _generate_rag_response(
        self,
        llm_manager: LLMManager,
        request: OrchestrationRequest,
        refined_output: PromptRefinerOutput,
        relevant_chunks: List[Dict[str, Union[str, float, Dict[str, Any]]]],
        response_generator: Optional[ResponseGeneratorAgent] = None,
        costs_metric: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """
        Generate response using retrieved chunks and ResponseGeneratorAgent only.
        No secondary LLM paths; no citations appended.
        """
        logger.info("Starting RAG response generation")

        if costs_metric is None:
            costs_metric = {}

        # If response generator is not available -> standardized technical issue
        if response_generator is None:
            logger.warning(
                "Response generator unavailable – returning technical issue message."
            )

            # Get localized message based on detected language
            detected_lang = getattr(request, "_detected_language", "en")
            localized_msg = get_localized_message(
                TECHNICAL_ISSUE_MESSAGES, detected_lang
            )

            if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                logger.info(
                    "Test environment detected – returning technical issue message."
                )
                return TestOrchestrationResponse(
                    llmServiceActive=False,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=localized_msg,
                    chunks=None,  # No chunks for technical failures
                )
            else:
                return OrchestrationResponse(
                    chatId=request.chatId,
                    llmServiceActive=False,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=TECHNICAL_ISSUE_MESSAGE,
                )

        try:
            with llm_manager.use_task_local():
                generator_result = response_generator.forward(
                    question=refined_output.original_question,
                    chunks=relevant_chunks or [],
                    max_blocks=ResponseGenerationConstants.DEFAULT_MAX_BLOCKS,
                )

            answer = (generator_result.get("answer") or "").strip()
            question_out_of_scope = bool(
                generator_result.get("questionOutOfLLMScope", False)
            )

            # Extract and store response generator costs
            generator_usage = generator_result.get(
                "usage",
                {
                    "total_cost": 0.0,
                    "total_prompt_tokens": 0,
                    "total_completion_tokens": 0,
                    "total_tokens": 0,
                    "num_calls": 0,
                },
            )
            costs_metric["response_generator"] = generator_usage
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    metadata={
                        "model": llm_manager.get_provider_info().get(
                            "model", "unknown"
                        ),
                        "num_calls": generator_usage.get("num_calls", 0),
                        "question_out_of_scope": question_out_of_scope,
                        "num_chunks_used": len(relevant_chunks)
                        if relevant_chunks
                        else 0,
                    },
                    output=answer,
                )
            if question_out_of_scope:
                logger.info(
                    "Question determined out-of-scope – sending fixed message without references."
                )

                # Get localized message based on detected language
                detected_lang = getattr(request, "_detected_language", "en")
                localized_msg = get_localized_message(
                    OUT_OF_SCOPE_MESSAGES, detected_lang
                )

                # Do NOT include references when question is out of scope
                # (data did not provide sufficient context to answer)
                if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                    logger.info(
                        "Test environment detected – returning out-of-scope message."
                    )
                    return TestOrchestrationResponse(
                        llmServiceActive=True,  # service OK; insufficient context
                        questionOutOfLLMScope=True,
                        inputGuardFailed=False,
                        content=localized_msg,
                        chunks=None,  # No chunks when question is out of scope
                    )
                else:
                    return OrchestrationResponse(
                        chatId=request.chatId,
                        llmServiceActive=True,  # service OK; insufficient context
                        questionOutOfLLMScope=True,
                        inputGuardFailed=False,
                        content=localized_msg,
                    )

            # In-scope: return the answer as-is (NO citations)
            logger.info("Returning in-scope answer without citations.")

            # Extract document references and append to content
            doc_references = self._extract_document_references(relevant_chunks)
            content_with_refs = answer
            if doc_references:
                refs_text = REFERENCES_SECTION_HEADER + "\n".join(
                    f"{i + 1}. {ref.document_url}"
                    for i, ref in enumerate(doc_references)
                )
                content_with_refs += refs_text

            if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                logger.info("Test environment detected – returning generated answer.")
                return TestOrchestrationResponse(
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=content_with_refs,
                    chunks=self._format_chunks_for_test_response(relevant_chunks),
                )
            else:
                return OrchestrationResponse(
                    chatId=request.chatId,
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=content_with_refs,
                )

        except Exception as e:
            error_id = generate_error_id()
            log_error_with_context(
                logger,
                error_id,
                "rag_response_generation",
                request.chatId,
                e,
                {"num_chunks": len(relevant_chunks) if relevant_chunks else 0},
            )
            if self.langfuse_config.langfuse_client:
                langfuse = self.langfuse_config.langfuse_client
                langfuse.update_current_span(
                    metadata={
                        "error_id": error_id,
                        "error_type": type(e).__name__,
                        "response_type": "technical_issue",
                        "refinement_failed": False,
                    }
                )
            # Standardized technical issue; no second LLM call, no citations
            # Get localized message based on detected language
            detected_lang = getattr(request, "_detected_language", "en")
            localized_msg = get_localized_message(
                TECHNICAL_ISSUE_MESSAGES, detected_lang
            )

            if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
                logger.info(
                    "Test environment detected – returning technical issue message."
                )
                return TestOrchestrationResponse(
                    llmServiceActive=False,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=localized_msg,
                    chunks=None,  # No chunks for technical failures
                )
            else:
                return OrchestrationResponse(
                    chatId=request.chatId,
                    llmServiceActive=False,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=TECHNICAL_ISSUE_MESSAGE,
                )

    # ========================================================================
    # Vector Indexer Support Methods (Isolated from RAG Pipeline)
    # ========================================================================
    @observe(name="create_embeddings_for_indexer", as_type="span")
    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Create embeddings for vector indexer using vault-driven model resolution.

        This method is completely isolated from the RAG pipeline and uses lazy
        initialization to avoid interfering with the main orchestration flow.

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
            # Lazy initialization of embedding manager
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

        This method is completely isolated from the RAG pipeline and uses lazy
        initialization to avoid interfering with the main orchestration flow.

        Args:
            request: Context generation request with document and chunk prompts

        Returns:
            Dictionary with generated context and metadata
        """
        logger.info("Generating context for chunks using Anthropic methodology")

        try:
            # Lazy initialization of context manager
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
            # Lazy initialization of embedding manager
            embedding_manager = self._get_embedding_manager()
            config_loader = self._get_config_loader()

            available_models: List[str] = embedding_manager.get_available_models(
                environment
            )

            # Get default model by resolving what would be used
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

    # ========================================================================
    # Lazy Initialization Helpers for Vector Indexer (Private Methods)
    # ========================================================================

    def _get_embedding_manager(self) -> "EmbeddingManager":
        """Lazy initialization of EmbeddingManager for vector indexer."""
        if not hasattr(self, "_embedding_manager"):
            from src.llm_orchestrator_config.embedding_manager import EmbeddingManager
            from src.llm_orchestrator_config.vault.vault_client import get_vault_client

            vault_client = get_vault_client()
            config_loader = self._get_config_loader()

            self._embedding_manager = EmbeddingManager(vault_client, config_loader)
            logger.debug("Lazy initialized EmbeddingManager for vector indexer")

        return self._embedding_manager

    def _get_context_manager(self) -> "ContextGenerationManager":
        """Lazy initialization of ContextGenerationManager for vector indexer."""
        if not hasattr(self, "_context_manager"):
            from src.llm_orchestrator_config.context_manager import (
                ContextGenerationManager,
            )
            from src.utils.connection_id_fetcher import get_connection_id_fetcher

            # Resolve production vault_uuid from DB before creating LLM manager
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
        if not hasattr(self, "_config_loader"):
            from src.llm_orchestrator_config.config.loader import ConfigurationLoader

            self._config_loader = ConfigurationLoader()
            logger.debug("Lazy initialized ConfigurationLoader for vector indexer")

        return self._config_loader

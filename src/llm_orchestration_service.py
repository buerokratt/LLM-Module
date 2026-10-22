"""LLM Orchestration Service - Business logic for LLM orchestration."""

from typing import Optional, List, Dict, Union, Any, AsyncIterator
import time
from src.loki_logger import LokiLogger
from langfuse import Langfuse, observe
import dspy

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
from src.response_generator.response_generate import ResponseGeneratorAgent
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    TECHNICAL_ISSUE_MESSAGE,
    INPUT_GUARDRAIL_VIOLATION_MESSAGE,
    PRODUCTION_DEPLOYMENT_ENVIRONMENT,
    RUUTER_PROMPT_CONFIG_ENDPOINT,
    PROMPT_CONFIG_CACHE_TTL,
    QDRANT_URL,
    LANGFUSE_URL,
)
from src.utils.error_utils import generate_error_id, log_error_with_context
from src.utils.stream_manager import stream_manager, StreamContext
from src.utils.cost_utils import calculate_total_costs, get_lm_usage_since

from src.utils.time_tracker import log_step_timings
from src.utils.prompt_config_loader import PromptConfigurationLoader
from src.utils.sse_utils import extract_content_from_sse
from src.utils.conversation_history_store import should_save_history, save_history_round
from src.guardrails import NeMoRailsAdapter, GuardrailCheckResult
from src.contextual_retrieval import ContextualRetriever
from src.contextual_retrieval.bm25_search import SmartBM25Search
from src.llm_orchestrator_config.feature_flags import FeatureFlags
from src.tool_classifier import ToolClassifier, WorkflowType
from src.tool_classifier.constants import SERVICE_STEP_PREFIXES
from src.tool_classifier.workflows.service_workflow import ServiceWorkflowExecutor
from src.vector_indexer.indexer_support import IndexerSupport
from src.orchestration.response_builders import (
    format_sse as _format_sse,
    create_error_response,
    create_out_of_scope_response,
    format_chunks_for_test_response,
    extract_document_references,
)
from src.orchestration.cost_budget import (
    log_costs as _log_costs,
    update_connection_budget,
)
from src.orchestration.inference_storage import (
    store_streaming_inference as _store_streaming_inference,
)
from src.orchestration.component_factory import ComponentFactory
from src.orchestration.guardrails_coordinator import GuardrailsCoordinator
from src.orchestration.rag_pipeline import RagPipeline
from src.orchestration.prompt_refinement import refine_user_prompt
from src.orchestration.request_preamble import RequestPreamble, build_block_response

# Re-exported for backwards compatibility — these now live in
# src/orchestration/constants.py so the extracted collaborators can share them.
from src.orchestration.constants import (  # noqa: F401
    REFERENCES_SECTION_HEADER,
    _HISTORY_EXCLUDED_MESSAGES,
)

# Initialize Loki logger for orchestration service
logger = LokiLogger(service_name="llm-orchestration-service")


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
                        host=langfuse_secrets.get("host") or LANGFUSE_URL,
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

        # Builds (and caches) the per-request components. Reads startup state
        # back off this façade, so it must be created before first use only —
        # not before shared_guardrails_adapters / shared_bm25_search are set.
        self._component_factory = ComponentFactory(self)

        # Runs NeMo input/output checks and the safe retrieval wrapper.
        self._guardrails = GuardrailsCoordinator(self.langfuse_config)

        # Core RAG flow (streaming + blocking). Calls back through this façade
        # for shared helpers so workflow/test patch points stay intact.
        self._rag_pipeline = RagPipeline(self)

        # Pre-classifier gates shared by both entry points.
        self._preamble = RequestPreamble()

        # Embeddings + context-generation support for the vector indexer.
        # Isolated collaborator, no dependency on the RAG/streaming pipeline.
        self._indexer_support = IndexerSupport()

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
        qdrant_url = QDRANT_URL
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

            # STEP 0 + 0.15: Language detection, then post-violation "why?" check
            detected_language, block = self._preamble.run_early(
                request, time_metric, is_streaming=False
            )
            if block:
                if block.log_timings:
                    log_step_timings(time_metric, request.chatId)
                return build_block_response(request, block)

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

            # STEP 0.5 + 0.6: Query validation, then connection/budget gate
            block = self._preamble.run_late(
                request, time_metric, detected_language, is_streaming=False
            )
            if block:
                return build_block_response(request, block)

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
            self._update_connection_budget(request.connection_id, costs_metric)

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
            self._update_connection_budget(request.connection_id, costs_metric)

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

        # STEP 0 + 0.15: Language detection, then post-violation "why?" check
        detected_language, block = self._preamble.run_early(
            request, time_metric, is_streaming=True
        )
        if block:
            yield self.format_sse(request.chatId, block.message)
            yield self.format_sse(request.chatId, "END")
            if block.log_timings:
                log_step_timings(time_metric, request.chatId)
            return

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

        # STEP 0.5 + 0.6: Query validation, then connection/budget gate
        block = self._preamble.run_late(
            request, time_metric, detected_language, is_streaming=True
        )
        if block:
            yield self.format_sse(request.chatId, block.message)
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

    def _stream_rag_pipeline(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        stream_ctx: StreamContext,
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
    ) -> AsyncIterator[str]:
        """Core RAG streaming pipeline. See RagPipeline.stream for details."""
        return self._rag_pipeline.stream(
            request=request,
            components=components,
            stream_ctx=stream_ctx,
            costs_metric=costs_metric,
            time_metric=time_metric,
        )

    def format_sse(
        self,
        chat_id: str,
        content: str,
        buttons: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format an SSE message. See response_builders.format_sse for details."""
        return _format_sse(chat_id, content, buttons)

    @observe(name="initialize_service_components", as_type="span")
    def _initialize_service_components(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Initialize all service components. See ComponentFactory for details."""
        components = self._component_factory.initialize_service_components(request)

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

    async def _execute_orchestration_pipeline(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
        prefix: str = "",
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Execute the blocking RAG pipeline. See RagPipeline.execute for details."""
        return await self._rag_pipeline.execute(
            request=request,
            components=components,
            costs_metric=costs_metric,
            time_metric=time_metric,
            prefix=prefix,
        )

    def _safe_initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[NeMoRailsAdapter]:
        """Safely initialize guardrails adapter. See ComponentFactory for details."""
        return self._component_factory.safe_initialize_guardrails(
            environment, connection_id
        )

    def _safe_initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> Optional[ContextualRetriever]:
        """Safely initialize contextual retriever. See ComponentFactory for details."""
        return self._component_factory.safe_initialize_contextual_retriever(
            environment, connection_id
        )

    def _safe_initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> Optional[ResponseGeneratorAgent]:
        """Safely initialize response generator. See ComponentFactory for details."""
        return self._component_factory.safe_initialize_response_generator(llm_manager)

    async def handle_input_guardrails(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse, None]:
        """Check input guardrails. See GuardrailsCoordinator for details."""
        return await self._guardrails.handle_input(
            guardrails_adapter, request, costs_metric
        )

    async def _safe_retrieve_contextual_chunks(
        self,
        contextual_retriever: Optional[ContextualRetriever],
        refined_output: PromptRefinerOutput,
        request: OrchestrationRequest,
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """Safely retrieve chunks. See GuardrailsCoordinator for details."""
        return await self._guardrails.safe_retrieve_contextual_chunks(
            contextual_retriever, refined_output, request
        )

    async def handle_output_guardrails(
        self,
        guardrails_adapter: Optional[NeMoRailsAdapter],
        generated_response: Union[OrchestrationResponse, TestOrchestrationResponse],
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Check output guardrails. See GuardrailsCoordinator for details."""
        return await self._guardrails.handle_output(
            guardrails_adapter, generated_response, request, costs_metric
        )

    def _create_error_response(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Create standardized error response. See response_builders for details."""
        return create_error_response(request)

    def _create_out_of_scope_response(
        self, request: OrchestrationRequest
    ) -> OrchestrationResponse:
        """Create standardized out-of-scope response. See response_builders for details."""
        return create_out_of_scope_response(request)

    async def store_streaming_inference(
        self,
        request: OrchestrationRequest,
        final_answer: str,
    ) -> None:
        """Store inference data. See cost_budget.store_streaming_inference for details."""
        await _store_streaming_inference(request, final_answer)

    def _initialize_guardrails(
        self, environment: str, connection_id: Optional[str]
    ) -> NeMoRailsAdapter:
        """Initialize NeMo Guardrails adapter. See ComponentFactory for details."""
        return self._component_factory.initialize_guardrails(environment, connection_id)

    async def _check_input_guardrails_async(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        user_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """Check user input against guardrails. See GuardrailsCoordinator."""
        return await self._guardrails.check_input(
            guardrails_adapter, user_message, costs_metric
        )

    async def _check_output_guardrails(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        assistant_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """Check assistant output against guardrails. See GuardrailsCoordinator."""
        return await self._guardrails.check_output(
            guardrails_adapter, assistant_message, costs_metric
        )

    def log_costs(self, costs_metric: Dict[str, Dict[str, Any]]) -> None:
        """Log cost breakdown. See cost_budget.log_costs for details."""
        _log_costs(costs_metric)

    def _update_connection_budget(
        self,
        vault_uuid: Optional[str],
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> None:
        """Update the budget for an LLM connection. See cost_budget for details."""
        update_connection_budget(vault_uuid, costs_metric)

    def _initialize_llm_manager(
        self, environment: str, connection_id: Optional[str]
    ) -> LLMManager:
        """Initialize LLM Manager. See ComponentFactory for details."""
        return self._component_factory.initialize_llm_manager(
            environment, connection_id
        )

    def _refine_user_prompt(
        self,
        llm_manager: LLMManager,
        original_message: str,
        conversation_history: List[ConversationItem],
        conversation_summary: Optional[str] = None,
    ) -> tuple[PromptRefinerOutput, Dict[str, Any]]:
        """Refine the user prompt. See prompt_refinement for details."""
        return refine_user_prompt(
            llm_manager=llm_manager,
            original_message=original_message,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
            langfuse_client=self.langfuse_config.langfuse_client,
        )

    def _initialize_contextual_retriever(
        self, environment: str, connection_id: Optional[str]
    ) -> ContextualRetriever:
        """Initialize contextual retriever. See ComponentFactory for details."""
        return self._component_factory.initialize_contextual_retriever(
            environment, connection_id
        )

    def _initialize_response_generator(
        self, llm_manager: LLMManager
    ) -> ResponseGeneratorAgent:
        """Initialize Response Generator. See ComponentFactory for details."""
        return self._component_factory.initialize_response_generator(llm_manager)

    def _get_custom_instructions_for_response_generation(self) -> str:
        """Get custom response-generation instructions. See ComponentFactory."""
        return self._component_factory.get_custom_instructions_for_response_generation()

    @staticmethod
    def _format_chunks_for_test_response(
        relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
    ) -> Optional[List[ChunkInfo]]:
        """Format retrieved chunks for test response. See response_builders for details."""
        return format_chunks_for_test_response(relevant_chunks)

    @staticmethod
    def _extract_document_references(
        relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
    ) -> Optional[List[DocumentReference]]:
        """Extract unique document references from chunks. See response_builders for details."""
        return extract_document_references(relevant_chunks)

    def _generate_rag_response(
        self,
        llm_manager: LLMManager,
        request: OrchestrationRequest,
        refined_output: PromptRefinerOutput,
        relevant_chunks: List[Dict[str, Union[str, float, Dict[str, Any]]]],
        response_generator: Optional[ResponseGeneratorAgent] = None,
        costs_metric: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Generate a RAG response. See RagPipeline.generate_response for details."""
        return self._rag_pipeline.generate_response(
            llm_manager=llm_manager,
            request=request,
            refined_output=refined_output,
            relevant_chunks=relevant_chunks,
            response_generator=response_generator,
            costs_metric=costs_metric,
        )

    # ========================================================================
    # Vector Indexer Support (delegates to IndexerSupport collaborator)
    # ========================================================================
    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Create embeddings for vector indexer. See IndexerSupport for details."""
        return self._indexer_support.create_embeddings_for_indexer(
            texts=texts,
            environment=environment,
            connection_id=connection_id,
            batch_size=batch_size,
        )

    def generate_context_for_chunks(
        self, request: ContextGenerationRequest
    ) -> Dict[str, Any]:
        """Generate context for chunks. See IndexerSupport for details."""
        return self._indexer_support.generate_context_for_chunks(request)

    def get_available_embedding_models_for_indexer(
        self, environment: str = PRODUCTION_DEPLOYMENT_ENVIRONMENT
    ) -> Dict[str, Any]:
        """Get available embedding models. See IndexerSupport for details."""
        return self._indexer_support.get_available_embedding_models_for_indexer(
            environment=environment
        )

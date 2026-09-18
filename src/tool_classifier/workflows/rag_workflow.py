"""RAG workflow executor - Layer 3: Knowledge base retrieval."""

from typing import Any, AsyncIterator, Dict, Optional, Union, cast, TYPE_CHECKING
from src.loki_logger import LokiLogger

from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationResponse,
)
from src.utils.stream_manager import StreamContext
from tool_classifier.base_workflow import BaseWorkflow

if TYPE_CHECKING:
    from llm_orchestration_service import LLMOrchestrationService

# Initialize Loki logger
logger = LokiLogger(service_name="rag-workflow")


class RAGWorkflowExecutor(BaseWorkflow):
    """
    Wrapper for existing RAG (Retrieval-Augmented Generation) workflow (Layer 3).

    This workflow handles queries that require searching the knowledge base
    and generating responses based on retrieved chunks. It uses the existing
    RAG pipeline:
    1. Prompt refinement
    2. Contextual retrieval (Qdrant + BM25)
    3. Rank fusion (RRF)
    4. Response generation
    5. Output guardrails (validation-first streaming)

    Examples of RAG queries:
    - "What are digital signatures?"
    - "How do I register a company?"
    - "Explain tax regulations"

    Implementation Status: COMPLETE
    This is a thin wrapper that delegates to existing LLMOrchestrationService methods.

    No TODO - Just wraps existing pipeline:
    - Non-streaming: Calls _execute_orchestration_pipeline()
    - Streaming: Calls existing streaming logic with NeMo guardrails

    Note: If no relevant chunks found, returns OOD response (not None)
    """

    def __init__(self, orchestration_service: "LLMOrchestrationService") -> None:
        """
        Initialize RAG workflow executor.

        Args:
            orchestration_service: Reference to LLMOrchestrationService
                                   for calling existing RAG pipeline
        """
        self.orchestration_service = orchestration_service
        logger.info("RAG workflow executor initialized (wrapper)")

    async def execute_async(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[Union[OrchestrationResponse, TestOrchestrationResponse]]:
        """
        Execute RAG workflow in non-streaming mode.

        Delegates to existing LLMOrchestrationService._execute_orchestration_pipeline()
        which handles:
        - Prompt refinement
        - Chunk retrieval (Qdrant + BM25)
        - Response generation
        - Output guardrails

        Args:
            request: Orchestration request with user query
            context: May contain pre-initialized "components" to avoid duplicate init
            time_metric: Optional timing dictionary from parent (for unified tracking)

        Returns:
            OrchestrationResponse with RAG-generated answer
            Never returns None (handles OOD internally)
        """
        logger.info(f"[{request.chatId}] Executing RAG workflow (non-streaming)")

        # Initialize components needed for RAG pipeline
        costs_metric: Dict[str, Any] = {}
        # Use parent time_metric or create new one
        if time_metric is None:
            time_metric = {}

        # Reuse components from context if available, otherwise initialize
        components = context.get("components")
        if components is None:
            components = self.orchestration_service._initialize_service_components(
                request
            )

        # Call existing RAG pipeline with "rag" prefix for namespacing
        response = await self.orchestration_service._execute_orchestration_pipeline(
            request=request,
            components=components,
            costs_metric=costs_metric,
            time_metric=time_metric,
            prefix="rag",
        )

        # Log costs (timing is logged by parent orchestration service)
        self.orchestration_service.log_costs(costs_metric)

        return response

    async def execute_streaming(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """
        Execute RAG workflow in streaming mode.

        Coroutine that returns an AsyncIterator so callers can safely use
        ``await workflow.execute_streaming(...)`` and then iterate over the
        returned stream without hitting a TypeError from awaiting an async
        generator.

        Delegates to existing streaming pipeline which handles:
        - Prompt refinement (blocking)
        - Chunk retrieval (blocking)
        - Streaming through NeMo guardrails (validation-first)
        - Real-time token validation

        The existing implementation uses NeMo's stream_with_guardrails which:
        - Buffers tokens (chunk_size=200)
        - Validates each buffer before yielding
        - Provides true validation-first streaming

        Args:
            request: Orchestration request with user query
            context: May contain pre-initialized "components" and "stream_ctx"
            time_metric: Optional timing dictionary from parent (for unified tracking)

        Returns:
            AsyncIterator yielding SSE-formatted strings
            Never returns None (handles OOD internally)
        """
        logger.info(f"[{request.chatId}] Executing RAG workflow (streaming)")

        # Initialize tracking dictionaries
        costs_metric: Dict[str, Any] = {}
        # Use parent time_metric or create new one
        if time_metric is None:
            time_metric = {}

        # Get components from context if provided, otherwise initialize
        components = context.get("components")
        if components is None:
            components = self.orchestration_service._initialize_service_components(
                request
            )

        # Get stream context from context if provided, otherwise create minimal tracking
        stream_ctx = context.get("stream_ctx")
        if stream_ctx is None:

            class MinimalStreamContext:
                """Minimal stream context for RAG workflow when called directly."""

                def __init__(self, chat_id: str) -> None:
                    self.stream_id = f"rag-{chat_id}"
                    self.token_count = 0
                    self.bot_generator = None

                def mark_completed(self) -> None:
                    # Intentionally empty: lifecycle tracking is handled by the orchestration service, not this minimal context
                    pass

                def mark_cancelled(self) -> None:
                    # Intentionally empty: lifecycle tracking is handled by the orchestration service, not this minimal context
                    pass

                def mark_error(self, error_id: str) -> None:
                    # Intentionally empty: lifecycle tracking is handled by the orchestration service, not this minimal context
                    pass

            stream_ctx = MinimalStreamContext(request.chatId)

        # Return an inner async generator so this method stays a coroutine.
        # This avoids the TypeError when callers do ``await execute_streaming(...)``.
        async def _stream() -> AsyncIterator[str]:
            async for sse_chunk in self.orchestration_service._stream_rag_pipeline(
                request=request,
                components=components,
                stream_ctx=cast(StreamContext, stream_ctx),
                costs_metric=costs_metric,
                time_metric=time_metric,
            ):
                yield sse_chunk

        return _stream()

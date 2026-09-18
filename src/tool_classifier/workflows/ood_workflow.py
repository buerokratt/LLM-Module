"""OOD workflow executor - Layer 4: Out-of-domain fallback."""

from typing import Any, AsyncIterator, Dict, Optional
from src.loki_logger import LokiLogger

from models.request_models import OrchestrationRequest, OrchestrationResponse
from tool_classifier.base_workflow import BaseWorkflow

# Initialize Loki logger
logger = LokiLogger(service_name="ood-workflow")


class OODWorkflowExecutor(BaseWorkflow):
    """
    Handles out-of-domain queries that no workflow can answer (Layer 4).

    This is the final fallback in the workflow chain. It returns a polite
    "cannot answer" message when:
    - No service matches (Layer 1 failed)
    - No context match (Layer 2 failed)
    - No relevant knowledge chunks (Layer 3 failed)

    Examples of OOD queries:
    - "What's the weather today?" (not in scope)
    - "Tell me a joke" (not government service)
    - Questions with no relevant knowledge

    """

    def __init__(self) -> None:
        """Initialize OOD workflow executor."""
        logger.info("OOD workflow executor initialized (skeleton)")

    async def execute_async(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[OrchestrationResponse]:
        """
        Execute OOD workflow in non-streaming mode.

        TODO: Implement OOD response:
        ```python
        from src.llm_orchestrator_config.llm_ochestrator_constants import (
            get_localized_message,
            OUT_OF_SCOPE_MESSAGES,
        )

        # Get detected language from request
        detected_language = getattr(request, "_detected_language", "en")

        # Get localized message
        ood_message = get_localized_message(OUT_OF_SCOPE_MESSAGES, detected_language)

        return OrchestrationResponse(
            chatId=request.chatId,
            llmServiceActive=True,
            questionOutOfLLMScope=True,  # Flag as out of scope
            inputGuardFailed=False,
            content=ood_message,
        )
        ```

        Args:
            request: Orchestration request with user query
            context: Unused (OOD doesn't need metadata)
            time_metric: Optional timing dictionary for future timing tracking

        Returns:
            OrchestrationResponse with OOD message
            Never returns None (this is final fallback)
        """
        logger.info(
            f"[{request.chatId}] OOD workflow execute_async called "
            f"(not implemented - returning None for now)"
        )

        # Implement OOD response logic here
        # For now, return None (will be implemented as simple message return)
        return None

    async def execute_streaming(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """
        Execute OOD workflow in streaming mode.

        TODO: Implement OOD streaming:
        ```python
        from src.llm_orchestrator_config.llm_ochestrator_constants import (
            get_localized_message,
            OUT_OF_SCOPE_MESSAGES,
        )

        # Get localized OOD message
        detected_language = getattr(request, "_detected_language", "en")
        ood_message = get_localized_message(OUT_OF_SCOPE_MESSAGES, detected_language)

        # Stream message for UX consistency (no guardrails needed - fixed message)
        async def stream_ood_message():
            for chunk in split_into_tokens(ood_message, chunk_size=5):
                yield self.format_sse(request.chatId, chunk)
                await asyncio.sleep(0.01)
            yield self.format_sse(request.chatId, "END")

        return stream_ood_message()
        ```

        Note: No output guardrails needed since this is a fixed, safe message.

        Args:
            request: Orchestration request with user query
            context: Unused (OOD doesn't need metadata)

        Returns:
            AsyncIterator yielding SSE strings
            Never returns None (this is final fallback)
        """
        logger.info(
            f"[{request.chatId}] OOD workflow execute_streaming called "
            f"(not implemented - returning None for now)"
        )

        # Implement OOD streaming logic here
        # For now, return None (will be implemented as simple message streaming)
        return None

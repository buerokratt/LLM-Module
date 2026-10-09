"""Guardrails coordination and safe retrieval wrappers (Groups E and F).

Extracted from LLMOrchestrationService. Runs NeMo input/output checks, records
their cost into ``costs_metric``, annotates the Langfuse span, and converts a
blocked verdict into the correct localized response type.

Also holds the safe ``ContextualRetriever`` wrapper, which shares the same
"never raise into the pipeline, degrade gracefully" contract.
"""

from typing import Any, Dict, List, Optional, Union

from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    PromptRefinerOutput,
    TestOrchestrationResponse,
)
from src.contextual_retrieval import ContextualRetriever
from src.guardrails import GuardrailCheckResult, NeMoRailsAdapter
from src.llm_orchestrator_config.exceptions import (
    ContextualRetrievalFailureError,
    ContextualRetrieverInitializationError,
)
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
    TEST_DEPLOYMENT_ENVIRONMENT,
    get_localized_message,
)

logger = LokiLogger(service_name="llm-orchestration-service")


def _conservative_block_result(
    error: Exception, direction: str
) -> GuardrailCheckResult:
    """Fail closed: on checker error, block rather than let content through."""
    return GuardrailCheckResult(
        allowed=False,
        verdict="yes",
        content=f"Error during {direction} guardrail check",
        error=str(error),
        usage={},
    )


class GuardrailsCoordinator:
    """Runs guardrail checks and converts blocked verdicts into responses."""

    def __init__(self, langfuse_config: Any) -> None:  # noqa: ANN401 — LangfuseConfig, avoids circular import
        self._langfuse_config = langfuse_config

    @property
    def _langfuse(self) -> Optional[Any]:  # noqa: ANN401 — Langfuse client
        return getattr(self._langfuse_config, "langfuse_client", None)

    def _annotate_error(self, error: Exception, direction: str) -> None:
        """Record a checker failure on the current Langfuse span."""
        langfuse = self._langfuse
        if langfuse:
            langfuse.update_current_span(
                metadata={
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "guardrail_type": direction,
                }
            )

    # ------------------------------------------------------------------
    # Raw checks
    # ------------------------------------------------------------------

    @observe(name="check_input_guardrails", as_type="span")
    async def check_input(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        user_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """Check user input against guardrails and track costs."""
        logger.info("Starting input guardrails check")

        try:
            result = await guardrails_adapter.check_input_async(user_message)

            costs_metric["input_guardrails"] = result.usage

            langfuse = self._langfuse
            if langfuse:
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
            self._annotate_error(e, "input")
            return _conservative_block_result(e, "input")

    @observe(name="check_output_guardrails", as_type="span")
    async def check_output(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        assistant_message: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> GuardrailCheckResult:
        """Check assistant output against guardrails and track costs."""
        logger.info("Starting output guardrails check")

        try:
            result = await guardrails_adapter.check_output_async(assistant_message)

            costs_metric["output_guardrails"] = result.usage

            langfuse = self._langfuse
            if langfuse:
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
            self._annotate_error(e, "output")
            return _conservative_block_result(e, "output")

    # ------------------------------------------------------------------
    # Check + build-blocked-response
    # ------------------------------------------------------------------

    async def handle_input(
        self,
        guardrails_adapter: NeMoRailsAdapter,
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse, None]:
        """Check input guardrails; return a blocked response, or None if allowed."""
        input_check_result = await self.check_input(
            guardrails_adapter=guardrails_adapter,
            user_message=request.message,
            costs_metric=costs_metric,
        )

        if not input_check_result.allowed:
            logger.warning(f"Input blocked by guardrails: {input_check_result.reason}")

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
            return OrchestrationResponse(
                chatId=request.chatId,
                llmServiceActive=True,
                questionOutOfLLMScope=False,
                inputGuardFailed=True,
                content=localized_msg,
            )

        logger.info("Input guardrails check passed")
        return None

    async def handle_output(
        self,
        guardrails_adapter: Optional[NeMoRailsAdapter],
        generated_response: Union[OrchestrationResponse, TestOrchestrationResponse],
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Check output guardrails, replacing the response if it is blocked."""
        should_check_guardrails = (
            guardrails_adapter is not None
            and generated_response.llmServiceActive
            and not generated_response.questionOutOfLLMScope
        )

        if should_check_guardrails:
            # should_check_guardrails guarantees guardrails_adapter is not None
            assert guardrails_adapter is not None
            output_check_result = await self.check_output(
                guardrails_adapter=guardrails_adapter,
                assistant_message=generated_response.content,
                costs_metric=costs_metric,
            )

            if not output_check_result.allowed:
                logger.warning(
                    f"Output blocked by guardrails: {output_check_result.reason}"
                )
                detected_lang = getattr(request, "_detected_language", "en")
                localized_msg = get_localized_message(
                    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES, detected_lang
                )

                if isinstance(generated_response, TestOrchestrationResponse):
                    return TestOrchestrationResponse(
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=localized_msg,
                        chunks=None,
                    )
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

    # ------------------------------------------------------------------
    # Safe retrieval (Group F)
    # ------------------------------------------------------------------

    async def safe_retrieve_contextual_chunks(
        self,
        contextual_retriever: Optional[ContextualRetriever],
        refined_output: PromptRefinerOutput,
        request: OrchestrationRequest,
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """Retrieve chunks via contextual retrieval, normalising failures.

        Raises:
            ContextualRetrieverInitializationError: retriever could not start.
            ContextualRetrievalFailureError: retrieval itself failed.
        """
        if not contextual_retriever:
            logger.info("Contextual Retriever not available, skipping chunk retrieval")
            return []

        try:
            if not contextual_retriever.initialized:
                initialization_success = await contextual_retriever.initialize()
                if not initialization_success:
                    logger.error("Failed to initialize contextual retriever")
                    raise ContextualRetrieverInitializationError(
                        "Contextual retriever failed to initialize"
                    )

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
            raise
        except Exception as retrieval_error:
            logger.error(f"Contextual chunk retrieval failed: {str(retrieval_error)}")
            raise ContextualRetrievalFailureError(
                f"Contextual chunk retrieval failed: {str(retrieval_error)}"
            ) from retrieval_error

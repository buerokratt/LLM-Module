"""Context workflow executor - Layer 2: Conversation history and greetings."""

from typing import Any, AsyncIterator, Dict, Optional, cast
import time
import dspy
from langfuse import observe
from src.loki_logger import LokiLogger

from src.models.conversation_history_models import ConversationHistoryState
from src.models.request_models import OrchestrationRequest, OrchestrationResponse
from src.utils.conversation_history_store import ConversationHistoryStore
from tool_classifier.base_workflow import BaseWorkflow
from tool_classifier.context_analyzer import ContextAnalyzer, ContextDetectionResult
from tool_classifier.workflows.service_workflow import LLMServiceProtocol
from src.guardrails.nemo_rails_adapter import NeMoRailsAdapter
from src.llm_orchestrator_config.llm_manager import LLMManager
from src.utils.cost_utils import get_lm_usage_since
from src.utils.language_detector import detect_language
from src.utils.observation_utils import (
    safe_observation_context,
    update_observation_safe,
)
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    GUARDRAILS_BLOCKED_PHRASES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGE,
)

# Initialize Loki logger
logger = LokiLogger(service_name="context-workflow")


class ContextWorkflowExecutor(BaseWorkflow):
    """
    Handles greetings and conversation history queries (Layer 2).

    Detects:
    - Greetings: "Hello", "Thanks", "Goodbye" (multilingual: Estonian, English)
    - History references: "What did you say earlier?", "Can you repeat that?"

    Uses LLM for semantic detection (multilingual), no regex patterns.

    Implementation Strategy:
    1. Detect language from user query
    2. Use ContextAnalyzer (LLM-based) to check if:
       - Query is a greeting -> generate friendly response
       - Query references conversation history -> extract answer
    3. If can answer -> return response
    4. Otherwise -> return None (fallback to RAG)

    Cost Tracking:
    - Tracks LLM costs for context analysis
    - Logs via orchestration_service.log_costs() (same as service/RAG workflows)
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        orchestration_service: Optional[LLMServiceProtocol] = None,
        conversation_history_store: Optional[ConversationHistoryStore] = None,
    ) -> None:
        """
        Initialize context workflow executor.

        Args:
            llm_manager: LLM manager for context analysis
            orchestration_service: Reference to LLMOrchestrationService for cost logging
            conversation_history_store: Redis-backed conversation history store;
                when provided, history is read from Redis (canonical source) rather
                than from ``request.conversationHistory``.
        """
        self.llm_manager = llm_manager
        self.orchestration_service = orchestration_service
        self.conversation_history_store = conversation_history_store
        self.context_analyzer = ContextAnalyzer(llm_manager)
        logger.info("Context workflow executor initialized")

    async def _build_history(
        self, request: OrchestrationRequest
    ) -> tuple[list[Dict[str, Any]], Optional[str]]:
        """Fetch conversation history, preferring Redis over request payload.

        When a :class:`ConversationHistoryStore` is wired in and the session has
        stored rounds, the Redis state is used as the canonical source of truth and
        the GUI-provided ``request.conversationHistory`` is ignored.  Any running
        summary stored alongside the rounds is returned as ``pre_computed_summary``
        so that downstream callers can skip an expensive LLM summarisation step.

        If the store is absent, raises, or returns no rounds the method falls back
        to the conversation history supplied in the request.

        Returns:
            A ``(history_dicts, pre_computed_summary)`` tuple where
            *history_dicts* is a list of ``{"authorRole", "message", "timestamp"}``
            dicts and *pre_computed_summary* is the Redis summary string or ``None``.
        """
        if self.conversation_history_store is not None:
            try:
                state: ConversationHistoryState = (
                    await self.conversation_history_store.get_context(request.chatId)
                )
                if state.rounds:
                    history: list[Dict[str, Any]] = []
                    for round_ in state.rounds:
                        history.append(
                            {
                                "authorRole": "user",
                                "message": round_.user_message,
                                "timestamp": str(round_.timestamp),
                            }
                        )
                        history.append(
                            {
                                "authorRole": "bot",
                                "message": round_.bot_message,
                                "timestamp": str(round_.timestamp),
                            }
                        )
                    logger.debug(
                        f"[{request.chatId}] Using Redis history: {len(state.rounds)} rounds, summary={'present' if state.summary else 'absent'}"
                    )
                    return history, state.summary
            except Exception as exc:
                logger.warning(
                    f"[{request.chatId}] Redis history fetch failed, falling back to request history: {exc}"
                )

        # Fallback: use the conversation history supplied in the request
        request_history: list[Dict[str, Any]] = [
            {
                "authorRole": item.authorRole,
                "message": item.message,
                "timestamp": item.timestamp,
            }
            for item in request.conversationHistory
        ]
        return request_history, None

    @observe(name="context_workflow_detect", as_type="generation")
    async def _detect(
        self,
        message: str,
        history: list[Dict[str, Any]],
        time_metric: Dict[str, float],
        costs_metric: Dict[str, Dict[str, Any]],
        pre_computed_summary: Optional[str] = None,
    ) -> Optional[ContextDetectionResult]:
        """Phase 1: run context detection with summary fallback.

        Checks the last 10 conversation turns first. If the query cannot be
        answered from those and the history exceeds 10 turns (or a Redis summary
        is available), falls back to a summary-based check. Returns None on error
        so the caller falls through to RAG.

        Args:
            pre_computed_summary: Running conversation summary retrieved from Redis.
                When provided, the expensive LLM summarisation step is skipped and
                this value is used directly.
        """
        try:
            start = time.time()
            (
                result,
                cost,
            ) = await self.context_analyzer.detect_context_with_summary_fallback(
                query=message,
                conversation_history=history,
                pre_computed_summary=pre_computed_summary,
            )
            time_metric["context.detection"] = time.time() - start
            costs_metric["context_detection"] = cost
            update_observation_safe(
                input_data={"query": message, "history_length": len(history)},
                output_data={
                    "is_greeting": result.is_greeting if result else None,
                    "can_answer_from_context": result.can_answer_from_context
                    if result
                    else None,
                },
                metadata={"usage": cost},
            )
            return result
        except Exception as e:
            logger.error(f"Phase 1 detection failed: {e}", exc_info=True)
            update_observation_safe(
                output_data={"error": str(e)},
                metadata={"usage": {}},
            )
            return None

    def _log_costs(self, costs_metric: Dict[str, Dict[str, Any]]) -> None:
        if self.orchestration_service:
            self.orchestration_service.log_costs(costs_metric)

    @staticmethod
    def _is_guardrail_violation(chunk: str) -> bool:
        """Return True if the chunk matches a known guardrail blocked phrase."""
        chunk_lower = chunk.strip().lower()
        return any(
            phrase.lower() in chunk_lower
            and len(chunk_lower) <= len(phrase.lower()) + 20
            for phrase in GUARDRAILS_BLOCKED_PHRASES
        )

    @observe(name="context_workflow_generate_response", as_type="generation")
    async def _generate_response_async(
        self,
        request: OrchestrationRequest,
        context_snippet: str,
        time_metric: Dict[str, float],
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Optional[OrchestrationResponse]:
        """Non-streaming: Generate response + apply output guardrails."""
        try:
            start = time.time()
            answer, cost = await self.context_analyzer.generate_context_response(
                query=request.message, context_snippet=context_snippet
            )
            time_metric["context.generation"] = time.time() - start
            costs_metric["context_response"] = cost
            update_observation_safe(
                input_data={
                    "chat_id": request.chatId,
                    "query": request.message,
                    "context_snippet_length": len(context_snippet),
                },
                output_data={"has_answer": bool(answer)},
                metadata={"usage": cost},
            )
        except Exception as e:
            logger.error(f"Phase 2 generation failed: {e}", exc_info=True)
            update_observation_safe(
                output_data={"error": str(e)},
                metadata={"usage": {}},
            )
            self._log_costs(costs_metric)
            return None

        if not answer:
            logger.warning(f"[{request.chatId}] Phase 2 empty answer — fallback to RAG")
            self._log_costs(costs_metric)
            return None

        response = OrchestrationResponse(
            chatId=request.chatId,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=answer,
        )
        if self.orchestration_service:
            try:
                components = self.orchestration_service._initialize_service_components(
                    request
                )
                response = cast(
                    OrchestrationResponse,
                    await self.orchestration_service.handle_output_guardrails(
                        guardrails_adapter=components.get("guardrails_adapter"),
                        generated_response=response,
                        request=request,
                        costs_metric=costs_metric,
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"[{request.chatId}] Output guardrails check failed: {e}"
                )
            self._log_costs(costs_metric)
        return response

    async def _stream_history_generator(
        self,
        chat_id: str,
        query: str,
        context_snippet: str,
        history_length_before: int,
        guardrails_adapter: NeMoRailsAdapter,
        costs_metric: Dict[str, Dict[str, Any]],
        request: OrchestrationRequest,
    ) -> AsyncIterator[str]:
        """Async generator: stream history answer through NeMo Guardrails."""
        bot_generator = self.context_analyzer.stream_context_response(
            query=query, context_snippet=context_snippet
        )
        orchestration_service = self.orchestration_service
        if orchestration_service is None:
            return
        accumulated_response: list[str] = []
        with safe_observation_context(
            as_type="generation",
            name="context_workflow_streaming",
            input={"query": query, "chat_id": chat_id},
        ) as _generation:
            async for validated_chunk in guardrails_adapter.stream_with_guardrails(
                user_message=query, bot_message_generator=bot_generator
            ):
                if isinstance(validated_chunk, str) and self._is_guardrail_violation(
                    validated_chunk
                ):
                    logger.warning(
                        f"[{chat_id}] Guardrails violation in context streaming"
                    )
                    yield orchestration_service.format_sse(
                        chat_id, OUTPUT_GUARDRAIL_VIOLATION_MESSAGE
                    )
                    await orchestration_service.store_streaming_inference(
                        request, OUTPUT_GUARDRAIL_VIOLATION_MESSAGE
                    )
                    yield orchestration_service.format_sse(chat_id, "END")
                    costs_metric["context_response"] = get_lm_usage_since(
                        history_length_before
                    )
                    _usage = costs_metric["context_response"]
                    try:
                        if _generation is not None:
                            _generation.update(
                                usage_details={
                                    "input": _usage.get("total_prompt_tokens", 0),
                                    "output": _usage.get("total_completion_tokens", 0),
                                    "total": _usage.get("total_tokens", 0),
                                },
                                cost_details={"total": _usage.get("total_cost", 0.0)},
                                output={"guardrail_violation": True},
                            )
                    except Exception as _e:
                        logger.debug(
                            f"Langfuse streaming observation update skipped: {_e}"
                        )
                    orchestration_service.log_costs(costs_metric)
                    return
                accumulated_response.append(validated_chunk)
                yield orchestration_service.format_sse(chat_id, validated_chunk)
            final_answer = "".join(accumulated_response)
            await orchestration_service.store_streaming_inference(request, final_answer)
            yield orchestration_service.format_sse(chat_id, "END")
            logger.info(f"[{chat_id}] Context streaming complete")
            costs_metric["context_response"] = get_lm_usage_since(history_length_before)
            _usage = costs_metric["context_response"]
            try:
                if _generation is not None:
                    _generation.update(
                        usage_details={
                            "input": _usage.get("total_prompt_tokens", 0),
                            "output": _usage.get("total_completion_tokens", 0),
                            "total": _usage.get("total_tokens", 0),
                        },
                        cost_details={"total": _usage.get("total_cost", 0.0)},
                        output={"answer_preview": final_answer[:500]},
                    )
            except Exception as _e:
                logger.debug(f"Langfuse streaming observation update skipped: {_e}")
            orchestration_service.log_costs(costs_metric)

    async def _create_history_stream(
        self,
        request: OrchestrationRequest,
        context_snippet: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Optional[AsyncIterator[str]]:
        """Set up guardrails adapter and return the history streaming generator."""
        if not self.orchestration_service:
            logger.warning(
                f"[{request.chatId}] No orchestration_service — cannot stream with guardrails"
            )
            return None
        try:
            components = self.orchestration_service._initialize_service_components(
                request
            )
            guardrails_adapter = components.get("guardrails_adapter")
        except Exception as e:
            logger.error(
                f"[{request.chatId}] Failed to initialize components: {e}",
                exc_info=True,
            )
            self._log_costs(costs_metric)
            return None

        if not isinstance(guardrails_adapter, NeMoRailsAdapter):
            logger.warning(
                f"[{request.chatId}] guardrails_adapter unavailable — cannot stream"
            )
            self._log_costs(costs_metric)
            return None

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception:
            pass

        return self._stream_history_generator(
            chat_id=request.chatId,
            query=request.message,
            context_snippet=context_snippet,
            history_length_before=history_length_before,
            guardrails_adapter=guardrails_adapter,
            costs_metric=costs_metric,
            request=request,
        )

    @observe(name="context_workflow_execute_async", as_type="span")
    async def execute_async(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[OrchestrationResponse]:
        """
        Execute context workflow in non-streaming mode (two-phase).

        Phase 1: Detect if query is a greeting or can be answered from history.
        Phase 2: Generate response (greetings: pre-built; history: LLM + guardrails).

        Returns:
            OrchestrationResponse or None to fallback to RAG
        """
        logger.info(
            f"[{request.chatId}] CONTEXT WORKFLOW (NON-STREAMING) | "
            f"Query: '{request.message[:100]}'"
        )
        costs_metric: Dict[str, Dict[str, Any]] = {}
        if time_metric is None:
            time_metric = {}

        language = detect_language(request.message)
        history, pre_computed_summary = await self._build_history(request)

        # Check if analysis is pre-computed (e.g. from classifier classify step)
        pre_computed = context.get("analysis_result")
        if (
            pre_computed is not None
            and hasattr(pre_computed, "is_greeting")
            and hasattr(pre_computed, "can_answer_from_context")
        ):
            detection_result: ContextDetectionResult = cast(
                ContextDetectionResult, pre_computed
            )
            costs_metric.setdefault(
                "context_detection",
                {"total_cost": 0.0, "total_tokens": 0, "num_calls": 0},
            )
        else:
            _detected = await self._detect(
                request.message,
                history,
                time_metric,
                costs_metric,
                pre_computed_summary,
            )
            if _detected is None:
                update_observation_safe(
                    input_data={"chat_id": request.chatId, "query": request.message},
                    output_data={"workflow_result": "fallback_to_rag"},
                    metadata={"costs": costs_metric},
                )
                self._log_costs(costs_metric)
                context["costs_dict"] = costs_metric
                return None
            detection_result = _detected

        logger.info(
            f"[{request.chatId}] Detection: greeting={detection_result.is_greeting} "
            f"can_answer={detection_result.can_answer_from_context}"
        )

        if detection_result.is_greeting:
            from src.tool_classifier.greeting_constants import get_greeting_response

            greeting = get_greeting_response(
                greeting_type=detection_result.greeting_type, language=language
            )
            self._log_costs(costs_metric)
            context["costs_dict"] = costs_metric
            update_observation_safe(
                input_data={"chat_id": request.chatId, "query": request.message},
                output_data={"workflow_result": "greeting_response"},
                metadata={"costs": costs_metric},
            )
            return OrchestrationResponse(
                chatId=request.chatId,
                llmServiceActive=True,
                questionOutOfLLMScope=False,
                inputGuardFailed=False,
                content=greeting,
            )

        if (
            detection_result.can_answer_from_context
            and detection_result.context_snippet
        ):
            context["costs_dict"] = costs_metric
            update_observation_safe(
                input_data={"chat_id": request.chatId, "query": request.message},
                output_data={"workflow_result": "context_answer_generation"},
                metadata={"costs": costs_metric},
            )
            return await self._generate_response_async(
                request, detection_result.context_snippet, time_metric, costs_metric
            )

        logger.warning(
            f"[{request.chatId}] Cannot answer from context — falling back to RAG"
        )
        update_observation_safe(
            input_data={"chat_id": request.chatId, "query": request.message},
            output_data={"workflow_result": "fallback_to_rag"},
            metadata={"costs": costs_metric},
        )
        self._log_costs(costs_metric)
        context["costs_dict"] = costs_metric
        return None

    @observe(
        name="context_workflow_execute_streaming",
        as_type="span",
        capture_output=False,
    )
    async def execute_streaming(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """
        Execute context workflow in streaming mode (two-phase).

        Phase 1: Detect context (blocking, fast — classification only).
        Phase 2: Stream answer through NeMo Guardrails (same pipeline as RAG).

        Returns:
            AsyncIterator yielding SSE strings or None to fallback to RAG
        """
        logger.info(
            f"[{request.chatId}] CONTEXT WORKFLOW (STREAMING) | "
            f"Query: '{request.message[:100]}'"
        )
        costs_metric: Dict[str, Dict[str, Any]] = {}
        if time_metric is None:
            time_metric = {}

        language = detect_language(request.message)
        history, pre_computed_summary = await self._build_history(request)

        detection_result = await self._detect(
            request.message, history, time_metric, costs_metric, pre_computed_summary
        )
        if detection_result is None:
            update_observation_safe(
                input_data={"chat_id": request.chatId, "query": request.message},
                output_data={"workflow_result": "fallback_to_rag"},
                metadata={"costs": costs_metric},
            )
            self._log_costs(costs_metric)
            return None

        logger.info(
            f"[{request.chatId}] Detection: greeting={detection_result.is_greeting} "
            f"can_answer={detection_result.can_answer_from_context}"
        )

        if detection_result.is_greeting:
            from src.tool_classifier.greeting_constants import get_greeting_response

            greeting = get_greeting_response(
                greeting_type=detection_result.greeting_type, language=language
            )
            orchestration_service = self.orchestration_service
            if orchestration_service is None:
                self._log_costs(costs_metric)
                return None
            chat_id = request.chatId

            async def _stream_greeting() -> AsyncIterator[str]:
                yield orchestration_service.format_sse(chat_id, greeting)
                await orchestration_service.store_streaming_inference(request, greeting)
                yield orchestration_service.format_sse(chat_id, "END")
                orchestration_service.log_costs(costs_metric)

            update_observation_safe(
                input_data={"chat_id": request.chatId, "query": request.message},
                output_data={"workflow_result": "greeting_stream"},
                metadata={"costs": costs_metric},
            )
            return _stream_greeting()

        if (
            detection_result.can_answer_from_context
            and detection_result.context_snippet
        ):
            update_observation_safe(
                input_data={"chat_id": request.chatId, "query": request.message},
                output_data={"workflow_result": "context_stream_generation"},
                metadata={"costs": costs_metric},
            )
            return await self._create_history_stream(
                request, detection_result.context_snippet, costs_metric
            )

        logger.warning(
            f"[{request.chatId}] Cannot answer from context — falling back to RAG"
        )
        update_observation_safe(
            input_data={"chat_id": request.chatId, "query": request.message},
            output_data={"workflow_result": "fallback_to_rag"},
            metadata={"costs": costs_metric},
        )
        self._log_costs(costs_metric)
        return None

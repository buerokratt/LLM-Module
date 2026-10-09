"""Core RAG pipeline: refine → retrieve → scope-check → generate (Group C).

Extracted from LLMOrchestrationService. Holds the streaming and non-streaming
RAG flows plus the blocking response generator.

Every shared helper is reached through the orchestration-service façade
(``self._s``) rather than imported directly. That is deliberate: workflows and
the test-suite patch these methods on the service instance
(``_refine_user_prompt``, ``_safe_retrieve_contextual_chunks``,
``handle_output_guardrails``, ...), and routing through the façade keeps those
seams intact.
"""

import asyncio
import os
import threading
import time
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import dspy
from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    PromptRefinerOutput,
    TestOrchestrationResponse,
)
from src.llm_orchestrator_config.exceptions import (
    ContextualRetrievalFailureError,
    ContextualRetrieverInitializationError,
)
from llm_orchestrator_config.llm_manager import LLMManager
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    OUT_OF_SCOPE_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGE,
    OUTPUT_GUARDRAIL_VIOLATION_PARTIAL_MESSAGES,
    PRODUCTION_DEPLOYMENT_ENVIRONMENT,
    STREAM_TOKEN_LIMIT_MESSAGE,
    TECHNICAL_ISSUE_MESSAGE,
    TECHNICAL_ISSUE_MESSAGES,
    TEST_DEPLOYMENT_ENVIRONMENT,
    get_localized_message,
    is_output_guardrail_violation,
)
from src.llm_orchestrator_config.stream_config import StreamConfig
from src.orchestration.constants import (
    _HISTORY_EXCLUDED_MESSAGES,
    REFERENCES_SECTION_HEADER,
)
from src.response_generator.response_generate import (
    ResponseGeneratorAgent,
    stream_response_native,
)
from src.utils.conversation_history_helpers import get_conversation_history
from src.utils.conversation_history_store import save_history_round
from src.utils.cost_utils import get_lm_usage_since_split
from src.utils.error_utils import generate_error_id, log_error_with_context
from src.utils.stream_manager import StreamContext
from src.utils.time_tracker import log_step_timings
from src.vector_indexer.constants import ResponseGenerationConstants

logger = LokiLogger(service_name="llm-orchestration-service")


def _format_references_block(doc_references: List[Any], *, as_links: bool) -> str:
    """Render the trailing references section.

    Args:
        doc_references: DocumentReference objects to render.
        as_links: True for the streaming paths (markdown links), False for the
            blocking path (bare URLs). This asymmetry is pre-existing behaviour.
    """
    if as_links:
        body = "\n".join(
            f"{i + 1}. [{ref.document_url}]({ref.document_url})"
            for i, ref in enumerate(doc_references)
        )
    else:
        body = "\n".join(
            f"{i + 1}. {ref.document_url}" for i, ref in enumerate(doc_references)
        )
    return REFERENCES_SECTION_HEADER + body


class RagPipeline:
    """Runs the RAG flow in both streaming and blocking modes."""

    def __init__(self, orchestration_service: Any) -> None:  # noqa: ANN401 — façade, avoids circular import
        self._s = orchestration_service

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        stream_ctx: StreamContext,
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
    ) -> AsyncIterator[str]:
        """Core RAG streaming pipeline without classifier routing.

        Callable directly by workflows to avoid infinite recursion when the
        tool classifier is enabled.

        Steps:
        1. Refine user prompt (blocking)
        2. Retrieve context chunks (blocking)
        3. Out-of-scope check (blocking)
        4. Stream through NeMo Guardrails (validation-first)

        Yields:
            SSE-formatted strings
        """
        service = self._s
        chat_id = request.chatId
        stream_id = stream_ctx.stream_id
        streaming_start_time = datetime.now()
        detected_language = getattr(request, "_detected_language", "en")

        def out_of_scope_message() -> str:
            return get_localized_message(OUT_OF_SCOPE_MESSAGES, detected_language)

        # STEP 1: REFINE USER PROMPT (blocking)
        logger.info(
            f"[{chat_id}] [{stream_id}] RAG Pipeline Step 1: Refining user prompt"
        )

        start_time = time.time()
        conversation_history, conversation_summary = await get_conversation_history(
            chat_id=chat_id,
            store=service.conversation_history_store,
            fallback=request.conversationHistory,
        )
        refined_output, refiner_usage = service._refine_user_prompt(
            llm_manager=components["llm_manager"],
            original_message=request.message,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        time_metric["prompt_refiner"] = time.time() - start_time
        costs_metric["prompt_refiner"] = refiner_usage

        logger.info(f"[{chat_id}] [{stream_id}] Prompt refinement complete")

        # STEP 2: RETRIEVE CONTEXT CHUNKS (blocking)
        logger.info(
            f"[{chat_id}] [{stream_id}] RAG Pipeline Step 2: Retrieving context chunks"
        )

        try:
            start_time = time.time()
            relevant_chunks = await service._safe_retrieve_contextual_chunks(
                components["contextual_retriever"], refined_output, request
            )
            time_metric["contextual_retrieval"] = time.time() - start_time
        except (
            ContextualRetrieverInitializationError,
            ContextualRetrievalFailureError,
        ) as e:
            logger.warning(
                f"[{chat_id}] [{stream_id}] Contextual retrieval failed: {str(e)}"
            )
            logger.info(
                f"[{chat_id}] [{stream_id}] Returning out-of-scope due to retrieval failure"
            )
            yield service.format_sse(chat_id, out_of_scope_message())
            yield service.format_sse(chat_id, "END")
            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)
            stream_ctx.mark_completed()
            return

        if len(relevant_chunks) == 0:
            logger.info(f"[{chat_id}] [{stream_id}] No relevant chunks - out of scope")
            yield service.format_sse(chat_id, out_of_scope_message())
            yield service.format_sse(chat_id, "END")
            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)
            stream_ctx.mark_completed()
            return

        logger.info(
            f"[{chat_id}] [{stream_id}] Retrieved {len(relevant_chunks)} chunks"
        )

        # STEP 3: QUICK OUT-OF-SCOPE CHECK (blocking)
        logger.info(
            f"[{chat_id}] [{stream_id}] RAG Pipeline Step 3: Checking if question is in scope"
        )

        start_time = time.time()
        is_out_of_scope = await components["response_generator"].check_scope_quick(
            question=refined_output.original_question,
            chunks=relevant_chunks,
            max_blocks=ResponseGenerationConstants.DEFAULT_MAX_BLOCKS,
        )
        time_metric["scope_check"] = time.time() - start_time

        if is_out_of_scope:
            logger.info(f"[{chat_id}] [{stream_id}] Question out of scope")
            yield service.format_sse(chat_id, out_of_scope_message())
            yield service.format_sse(chat_id, "END")
            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)
            stream_ctx.mark_completed()
            return

        logger.info(f"[{chat_id}] [{stream_id}] Question is in scope")

        # STEP 4: STREAM THROUGH NEMO GUARDRAILS (validation-first)
        logger.info(
            f"[{chat_id}] [{stream_id}] RAG Pipeline Step 4: Starting streaming through NeMo Guardrails"
        )

        streaming_step_start = time.time()

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

        # Store bot_generator in stream context for guaranteed cleanup
        bot_generator = bot_response_generator()
        stream_ctx.bot_generator = bot_generator

        def record_streaming_usage() -> None:
            # Output-rail validation runs interleaved with generation in the same
            # LM history window, so split the two — folding them together hid a
            # 100x guardrail cost regression.
            usage_info, guardrails_usage = get_lm_usage_since_split(
                history_length_before
            )
            costs_metric["streaming_generation"] = usage_info
            if guardrails_usage.get("num_calls", 0) > 0:
                costs_metric["output_guardrails"] = guardrails_usage

        try:
            accumulated_response: List[str] = []
            # Whether any real answer content has already been sent to the client.
            # Used to choose the guardrail-violation wording (start vs mid-answer).
            content_streamed = False

            if components["guardrails_adapter"]:
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
                        stream_ctx.token_count += len(validated_chunk) // 4
                        accumulated_response.append(validated_chunk)

                        if stream_ctx.token_count > StreamConfig.MAX_TOKENS_PER_STREAM:
                            logger.error(
                                f"[{chat_id}] [{stream_id}] Token limit exceeded: "
                                f"{stream_ctx.token_count} > {StreamConfig.MAX_TOKENS_PER_STREAM}"
                            )
                            yield service.format_sse(
                                chat_id, STREAM_TOKEN_LIMIT_MESSAGE
                            )
                            yield service.format_sse(chat_id, "END")

                            record_streaming_usage()
                            service.log_costs(costs_metric)
                            log_step_timings(time_metric, chat_id)
                            stream_ctx.mark_completed()
                            return

                        # Also catches NeMo's `enable_rails_exceptions` JSON payload
                        if is_output_guardrail_violation(validated_chunk):
                            logger.warning(
                                f"[{chat_id}] [{stream_id}] Guardrails violation detected"
                            )

                            if content_streamed:
                                violation_message = "\n\n" + get_localized_message(
                                    OUTPUT_GUARDRAIL_VIOLATION_PARTIAL_MESSAGES,
                                    detected_language,
                                )
                            else:
                                violation_message = OUTPUT_GUARDRAIL_VIOLATION_MESSAGE
                            yield service.format_sse(chat_id, violation_message)
                            yield service.format_sse(chat_id, "END")

                            record_streaming_usage()
                            service.log_costs(costs_metric)
                            log_step_timings(time_metric, chat_id)
                            stream_ctx.mark_completed()
                            return

                        yield service.format_sse(chat_id, validated_chunk)
                        content_streamed = True
                except GeneratorExit:
                    stream_ctx.mark_cancelled()
                    logger.info(
                        f"[{chat_id}] [{stream_id}] Client disconnected during guardrails streaming"
                    )
                    raise

                logger.info(
                    f"[{chat_id}] [{stream_id}] Stream completed successfully ({chunk_count} chunks)"
                )

                doc_references = service._extract_document_references(relevant_chunks)
                if doc_references:
                    yield service.format_sse(
                        chat_id, _format_references_block(doc_references, as_links=True)
                    )

                yield service.format_sse(chat_id, "END")

            else:
                logger.warning(
                    f"[{chat_id}] [{stream_id}] Streaming without guardrails validation"
                )
                chunk_count = 0
                async for token in bot_generator:
                    chunk_count += 1

                    stream_ctx.token_count += len(token) // 4
                    accumulated_response.append(token)

                    if stream_ctx.token_count > StreamConfig.MAX_TOKENS_PER_STREAM:
                        logger.error(
                            f"[{chat_id}] [{stream_id}] Token limit exceeded (no guardrails)"
                        )
                        yield service.format_sse(chat_id, STREAM_TOKEN_LIMIT_MESSAGE)
                        yield service.format_sse(chat_id, "END")
                        stream_ctx.mark_completed()
                        return

                    yield service.format_sse(chat_id, token)

                doc_references = service._extract_document_references(relevant_chunks)
                if doc_references:
                    yield service.format_sse(
                        chat_id, _format_references_block(doc_references, as_links=True)
                    )

                yield service.format_sse(chat_id, "END")

            record_streaming_usage()

            time_metric["streaming_generation"] = time.time() - streaming_step_start
            time_metric["output_guardrails"] = 0.0  # Inline during streaming

            streaming_duration = (datetime.now() - streaming_start_time).total_seconds()
            logger.info(
                f"[{chat_id}] [{stream_id}] Streaming completed in {streaming_duration:.2f}s"
            )

            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)

            langfuse = service.langfuse_config.langfuse_client
            if langfuse:
                try:
                    langfuse.update_current_generation(
                        metadata={
                            "streaming": True,
                            "streaming_duration_seconds": streaming_duration,
                            "chunks_streamed": chunk_count,
                            "cost_breakdown": costs_metric,
                            "chat_id": chat_id,
                            "environment": request.environment,
                            "stream_id": stream_id,
                        },
                    )
                    langfuse.flush()
                except Exception as langfuse_error:
                    logger.error(
                        f"Langfuse streaming metadata update failed: {langfuse_error}",
                        exc_info=True,
                    )

            # Store inference data (production and testing environments).
            # Set RAG data on request for the unified storage method.
            setattr(request, "_rag_refined_questions", refined_output.refined_questions)  # noqa: B010
            setattr(request, "_rag_ranked_chunks", relevant_chunks)  # noqa: B010
            final_answer = "".join(accumulated_response)
            try:
                await service.store_streaming_inference(
                    request=request,
                    final_answer=final_answer,
                )
            except Exception as storage_error:
                logger.error(
                    f"Storage failed for chat_id: {chat_id}, "
                    f"environment: {request.environment} - {str(storage_error)}"
                )

            # Persist conversation history (RAG streaming)
            if service.conversation_history_store is not None:
                if final_answer not in _HISTORY_EXCLUDED_MESSAGES:
                    await save_history_round(
                        service.conversation_history_store,
                        chat_id,
                        request.message,
                        final_answer,
                    )

            stream_ctx.mark_completed()

        except GeneratorExit:
            # Client disconnected
            stream_ctx.mark_cancelled()
            logger.info(f"[{chat_id}] [{stream_id}] Client disconnected")
            record_streaming_usage()
            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)

            service._update_connection_budget(request.connection_id, costs_metric)
            raise
        except Exception as stream_error:
            error_id = generate_error_id()
            stream_ctx.mark_error(error_id)
            log_error_with_context(
                logger, error_id, "streaming_generation", chat_id, stream_error
            )
            yield service.format_sse(chat_id, TECHNICAL_ISSUE_MESSAGE)
            yield service.format_sse(chat_id, "END")

            record_streaming_usage()
            service.log_costs(costs_metric)
            log_step_timings(time_metric, chat_id)

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    @observe(name="execute_orchestration_pipeline", as_type="span")
    async def execute(
        self,
        request: OrchestrationRequest,
        components: Dict[str, Any],
        costs_metric: Dict[str, Dict[str, Any]],
        time_metric: Dict[str, float],
        prefix: str = "",
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Execute the blocking orchestration pipeline with all components.

        Args:
            request: Orchestration request
            components: Initialized service components
            costs_metric: Dictionary for cost tracking
            time_metric: Dictionary for timing tracking
            prefix: Optional prefix for timing keys (e.g. "rag" for workflow
                namespacing)
        """
        # Query validation AND input guardrails happen at orchestration level
        # (process_orchestration_request) BEFORE classifier routing, for true
        # early rejection — saves ~3.5s on blocked requests.
        service = self._s

        def timing_key(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        # Step 1: Refine user prompt
        start_time = time.time()
        conversation_history, conversation_summary = await get_conversation_history(
            chat_id=request.chatId,
            store=service.conversation_history_store,
            fallback=request.conversationHistory,
        )
        refined_output, refiner_usage = service._refine_user_prompt(
            llm_manager=components["llm_manager"],
            original_message=request.message,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        time_metric[timing_key("prompt_refiner")] = time.time() - start_time
        costs_metric["prompt_refiner"] = refiner_usage

        # Step 2: Retrieve relevant chunks using contextual retrieval
        try:
            start_time = time.time()
            relevant_chunks = await service._safe_retrieve_contextual_chunks(
                components["contextual_retriever"], refined_output, request
            )
            time_metric[timing_key("contextual_retrieval")] = time.time() - start_time
        except (
            ContextualRetrieverInitializationError,
            ContextualRetrievalFailureError,
        ) as e:
            logger.warning(f"Contextual retrieval failed: {str(e)}")
            return service._create_out_of_scope_response(request)

        if len(relevant_chunks) == 0:
            logger.info("No relevant chunks found - returning out-of-scope response")
            return service._create_out_of_scope_response(request)

        # Step 3: Generate response
        start_time = time.time()
        generated_response = service._generate_rag_response(
            llm_manager=components["llm_manager"],
            request=request,
            refined_output=refined_output,
            relevant_chunks=relevant_chunks,
            response_generator=components["response_generator"],
            costs_metric=costs_metric,
        )
        time_metric[timing_key("response_generation")] = time.time() - start_time

        # Populate retrieval_context for eval mode (DeepEval metrics need this)
        if os.getenv("EVAL_MODE", "false").lower() == "true" and isinstance(
            generated_response, OrchestrationResponse
        ):
            generated_response.retrieval_context = _build_eval_context(relevant_chunks)

        # Step 4: Output Guardrails Check
        # Applied to all response types for consistent safety across environments
        start_time = time.time()
        output_guardrails_response = await service.handle_output_guardrails(
            components["guardrails_adapter"],
            generated_response,
            request,
            costs_metric,
        )
        time_metric[timing_key("output_guardrails_check")] = time.time() - start_time

        # Step 5: Store inference data (production and testing environments).
        # Only OrchestrationResponse has chatId, not TestOrchestrationResponse.
        if request.environment in [
            PRODUCTION_DEPLOYMENT_ENVIRONMENT,
            TEST_DEPLOYMENT_ENVIRONMENT,
        ] and isinstance(output_guardrails_response, OrchestrationResponse):
            self._store_inference_in_background(
                request=request,
                refined_output=refined_output,
                relevant_chunks=relevant_chunks,
                final_answer=output_guardrails_response.content,
            )

        return output_guardrails_response

    def _store_inference_in_background(
        self,
        request: OrchestrationRequest,
        refined_output: PromptRefinerOutput,
        relevant_chunks: List[Dict[str, Union[str, float, Dict[str, Any]]]],
        final_answer: str,
    ) -> None:
        """Fire-and-forget inference storage from a sync context.

        Runs the async store on a daemon thread with its own event loop because
        the surrounding pipeline is sync at this point. Failures are logged and
        never propagated — storage must not fail the request.
        """
        try:
            # Set RAG data on request for the unified storage method
            request._rag_refined_questions = refined_output.refined_questions  # type: ignore[attr-defined]
            request._rag_ranked_chunks = relevant_chunks  # type: ignore[attr-defined]

            def _store_async() -> None:
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(
                        self._s.store_streaming_inference(
                            request=request,
                            final_answer=final_answer,
                        )
                    )
                    loop.close()
                except Exception as e:
                    logger.error(f"Error in async storage thread: {str(e)}")

            threading.Thread(target=_store_async, daemon=True).start()
        except Exception as storage_error:
            logger.error(
                f"Storage failed for chat_id: {request.chatId}, "
                f"environment: {request.environment} - {str(storage_error)}"
            )

    # ------------------------------------------------------------------
    # Blocking response generation
    # ------------------------------------------------------------------

    @observe(name="generate_rag_response", as_type="span")
    def generate_response(
        self,
        llm_manager: LLMManager,
        request: OrchestrationRequest,
        refined_output: PromptRefinerOutput,
        relevant_chunks: List[Dict[str, Union[str, float, Dict[str, Any]]]],
        response_generator: Optional[ResponseGeneratorAgent] = None,
        costs_metric: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Generate a response from retrieved chunks via ResponseGeneratorAgent.

        No secondary LLM paths; no inline citations.
        """
        logger.info("Starting RAG response generation")

        service = self._s
        if costs_metric is None:
            costs_metric = {}

        is_test_env = request.environment == TEST_DEPLOYMENT_ENVIRONMENT
        detected_lang = getattr(request, "_detected_language", "en")

        def technical_issue_response() -> Union[
            OrchestrationResponse, TestOrchestrationResponse
        ]:
            localized_msg = get_localized_message(
                TECHNICAL_ISSUE_MESSAGES, detected_lang
            )
            if is_test_env:
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
            # NOTE: pre-existing asymmetry — the non-test branch returns the
            # non-localized constant while the test branch returns the
            # localized message.
            return OrchestrationResponse(
                chatId=request.chatId,
                llmServiceActive=False,
                questionOutOfLLMScope=False,
                inputGuardFailed=False,
                content=TECHNICAL_ISSUE_MESSAGE,
            )

        # If response generator is unavailable -> standardized technical issue
        if response_generator is None:
            logger.warning(
                "Response generator unavailable – returning technical issue message."
            )
            return technical_issue_response()

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

            langfuse = service.langfuse_config.langfuse_client
            if langfuse:
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
                localized_msg = get_localized_message(
                    OUT_OF_SCOPE_MESSAGES, detected_lang
                )

                # Do NOT include references when out of scope (insufficient context)
                if is_test_env:
                    logger.info(
                        "Test environment detected – returning out-of-scope message."
                    )
                    return TestOrchestrationResponse(
                        llmServiceActive=True,  # service OK; insufficient context
                        questionOutOfLLMScope=True,
                        inputGuardFailed=False,
                        content=localized_msg,
                        chunks=None,
                    )
                return OrchestrationResponse(
                    chatId=request.chatId,
                    llmServiceActive=True,  # service OK; insufficient context
                    questionOutOfLLMScope=True,
                    inputGuardFailed=False,
                    content=localized_msg,
                )

            logger.info("Returning in-scope answer without citations.")

            content_with_refs = answer
            doc_references = service._extract_document_references(relevant_chunks)
            if doc_references:
                content_with_refs += _format_references_block(
                    doc_references, as_links=False
                )

            if is_test_env:
                logger.info("Test environment detected – returning generated answer.")
                return TestOrchestrationResponse(
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=content_with_refs,
                    chunks=service._format_chunks_for_test_response(relevant_chunks),
                )
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
            langfuse = service.langfuse_config.langfuse_client
            if langfuse:
                langfuse.update_current_span(
                    metadata={
                        "error_id": error_id,
                        "error_type": type(e).__name__,
                        "response_type": "technical_issue",
                        "refinement_failed": False,
                    }
                )
            # Standardized technical issue; no second LLM call, no citations
            return technical_issue_response()


def _build_eval_context(
    relevant_chunks: List[Dict[str, Union[str, float, Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Build the DeepEval retrieval_context payload from ranked chunks."""
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
    return eval_chunks

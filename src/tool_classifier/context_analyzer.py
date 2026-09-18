"""Context analyzer for greeting detection and conversation history analysis."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional
import json
import dspy
import dspy.streaming
from dspy.streaming import StreamListener
from langfuse import observe
from src.utils.observation_utils import (
    safe_observation_context,
    update_observation_safe,
)
from src.loki_logger import LokiLogger
from pydantic import BaseModel, Field

from src.utils.cost_utils import get_lm_usage_since
from tool_classifier.greeting_constants import get_greeting_response

logger = LokiLogger(service_name="context-workflow")


def _get_current_model_name() -> str:
    """Best-effort model name lookup from current DSPy LM."""
    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "model"):
            model_name = lm.model
            if isinstance(model_name, str) and model_name:
                return model_name
    except Exception:
        pass
    return "unknown"


class ContextAnalysisResult(BaseModel):
    """Result of context analysis."""

    is_greeting: bool = Field(
        ..., description="Whether the query is a greeting (hello, goodbye, thanks)"
    )
    can_answer_from_context: bool = Field(
        ..., description="Whether the query can be answered from conversation history"
    )
    answer: Optional[str] = Field(
        None, description="Generated response (greeting or context-based answer)"
    )
    reasoning: str = Field(..., description="Brief explanation of the analysis")
    answered_from_summary: bool = Field(
        default=False,
        description="Whether the answer was derived from a conversation summary (older turns beyond recent 10)",
    )


class ContextAnalysisSignature(dspy.Signature):
    """Analyze user query for greeting detection and conversation history references.

    This signature instructs the LLM to:
    1. Detect greetings in multiple languages (Estonian, English)
    2. Check if query references conversation history
    3. Generate appropriate responses based on context

    Supported greeting types:
    - hello: Tere, Hello, Hi, Hei, Hey, Moi, Good morning, Good afternoon, Good evening
    - goodbye: Nägemist, Bye, Goodbye, See you, Good night
    - thanks: Tänan, Aitäh, Thank you, Thanks, Much appreciated
    - casual: Tervist, Tšau, Moikka

    IMPORTANT — Greeting + Question distinction:
    - A message is a greeting ONLY if it contains NOTHING beyond the greeting itself.
    - If the message contains a greeting AND a substantive question or request,
      set is_greeting to FALSE so the question is answered properly.

    The LLM should respond in the SAME language as the user's query.
    """

    conversation_history: str = dspy.InputField(
        desc="Recent conversation history (last 10 turns) formatted as JSON"
    )
    user_query: str = dspy.InputField(
        desc="Current user query to analyze for greetings or context references"
    )
    analysis_result: str = dspy.OutputField(
        desc='JSON object with: {"is_greeting": bool, "can_answer_from_context": bool, "answer": str|null, "reasoning": str}. '
        "For greetings, generate a friendly response in the same language. "
        "For context references, extract the answer from conversation history if available."
    )


class ConversationSummarySignature(dspy.Signature):
    """Generate a concise summary of conversation history.

    Summarize the key topics, facts, decisions, and information discussed
    in the conversation. Preserve specific details like numbers, names,
    dates, and other factual information that might be referenced later.

    The summary should be in the SAME language as the conversation.
    """

    conversation_history: str = dspy.InputField(
        desc="Conversation history formatted as JSON to summarize"
    )
    summary: str = dspy.OutputField(
        desc="Concise summary capturing key topics, facts, and information discussed. "
        "Preserve specific details (numbers, names, dates) that could be referenced later."
    )


class IncrementalSummarySignature(dspy.Signature):
    """Merge newly evicted conversation rounds into an existing summary.

    Given an existing summary (which may be empty for the first eviction) and a
    JSON-formatted list of conversation rounds that were just evicted from the
    active history window, produce an updated summary that incorporates all new
    information.

    Guidelines:
    - Preserve all factual details from the existing summary (names, numbers, dates).
    - Integrate only new, non-redundant information from the evicted rounds.
    - Keep the summary concise — omit filler, focus on actionable/memorable facts.
    - Respond in the SAME language as the conversation.
    """

    existing_summary: str = dspy.InputField(
        desc=(
            "Current conversation summary. May be an empty string if no summary "
            "exists yet (first eviction)."
        )
    )
    new_rounds: str = dspy.InputField(
        desc=(
            "JSON array of conversation rounds that were just evicted from the "
            "active history window, each with user_message, bot_message, and timestamp."
        )
    )
    updated_summary: str = dspy.OutputField(
        desc=(
            "Updated summary that merges the existing summary with the new rounds. "
            "Preserve all factual details; drop redundant information. "
            "Same language as the conversation."
        )
    )


class SummaryAnalysisSignature(dspy.Signature):
    """Analyze if a user query can be answered from a conversation summary.

    Given a summary of earlier conversation and the current user query,
    determine if the query references information from the summarized conversation.
    If yes, generate an appropriate answer based on the summary.

    The response should be in the SAME language as the user's query.
    """

    conversation_summary: str = dspy.InputField(
        desc="Summary of earlier conversation history"
    )
    user_query: str = dspy.InputField(
        desc="Current user query to check against the conversation summary"
    )
    analysis_result: str = dspy.OutputField(
        desc='JSON object with: {"can_answer_from_context": bool, "answer": str|null, "reasoning": str}. '
        "If the query references information from the summary, extract/generate the answer. "
        "If the summary does not contain relevant information, set can_answer_from_context to false."
    )


class ContextDetectionResult(BaseModel):
    """Result of Phase 1 context detection (classify only, no answer generation)."""

    is_greeting: bool = Field(..., description="Whether the query is a greeting")
    greeting_type: str = Field(
        default="hello",
        description="Type of greeting: hello, goodbye, thanks, or casual",
    )
    can_answer_from_context: bool = Field(
        ..., description="Whether the query can be answered from conversation history"
    )
    reasoning: str = Field(..., description="Brief explanation of the detection")
    answered_from_summary: bool = Field(
        default=False,
        description="Whether summary analysis was used for detection",
    )
    # Relevant context snippet extracted for use in Phase 2 generation
    context_snippet: Optional[str] = Field(
        default=None,
        description="The relevant part of history/summary to answer from, for Phase 2",
    )


class ContextDetectionSignature(dspy.Signature):
    """Detect if a user query is a greeting or can be answered from conversation history.

    Phase 1 (detection only): classify the query WITHOUT generating the answer.

    Supported greeting types:
    - hello: Tere, Hello, Hi, Hei, Hey, Moi, Good morning/afternoon/evening
    - goodbye: Nägemist, Bye, Goodbye, See you, Good night
    - thanks: Tänan, Aitäh, Thank you, Thanks, Much appreciated
    - casual: Tervist, Tšau, Moikka

    IMPORTANT — Greeting + Question distinction:
    - A message is a greeting ONLY if it contains NOTHING beyond the greeting itself
      (e.g. "Hello!", "Tere!", "Thanks!", "Aitäh!").
    - If the message contains a greeting AND a substantive question or request
      (e.g. "Hello, how to show uninterest to a policy?",
       "Tere, mis on sünnitoetus?", "Hi, what are the tax benefits?"),
      set is_greeting to FALSE. The question must be answered via RAG, not a greeting template.
    - When in doubt, prefer is_greeting=false so the user's question is answered.

    Do NOT generate the answer here — only detect and extract a relevant context snippet.
    """

    conversation_history: str = dspy.InputField(
        desc="Recent conversation history (last 10 turns) formatted as JSON"
    )
    user_query: str = dspy.InputField(desc="Current user query to classify")
    detection_result: str = dspy.OutputField(
        desc='JSON object with: {"is_greeting": bool, "greeting_type": str, "can_answer_from_context": bool, '
        '"reasoning": str, "context_snippet": str|null}. '
        'greeting_type must be one of: "hello", "goodbye", "thanks", "casual" — '
        'set it only when is_greeting is true, defaulting to "hello" otherwise. '
        "context_snippet should contain the relevant excerpt from history if can_answer_from_context is true, "
        "or null otherwise. Do NOT generate the final answer — only detect and extract. "
        "CRITICAL: is_greeting must be false when the message contains a question or request alongside the greeting."
    )


class ContextResponseGenerationSignature(dspy.Signature):
    """Generate a response to a user query based on conversation history context.

    Phase 2 (generation): given the user query and relevant context, generate a helpful answer.
    Respond in the SAME language as the user query.
    """

    context_snippet: str = dspy.InputField(
        desc="Relevant excerpt from conversation history or summary that contains the answer"
    )
    user_query: str = dspy.InputField(desc="Current user query to answer")
    answer: str = dspy.OutputField(
        desc="A helpful, natural response to the user query based on the provided context. "
        "Respond in the same language as the user query."
    )


class ContextAnalyzer:
    """
    Analyzer for greeting detection and context-based question answering.

    This class uses an LLM to intelligently detect:
    - Greetings in multiple languages (Estonian, English)
    - Questions that reference conversation history
    - Generate appropriate responses based on context

    Example Usage:
        analyzer = ContextAnalyzer(llm_manager)
        result = await analyzer.analyze_context(
            query="Tere!",
            conversation_history=[],
            language="et"
        )
        # result.is_greeting = True
        # result.answer = "Tere! Kuidas ma saan sind aidata?"
    """

    def __init__(self, llm_manager: Any) -> None:  # noqa: ANN401
        """
        Initialize the context analyzer.

        Args:
            llm_manager: LLM manager instance for making LLM calls
        """
        self.llm_manager = llm_manager
        self._module: Optional[dspy.Module] = None
        self._summary_module: Optional[dspy.Module] = None
        self._summary_analysis_module: Optional[dspy.Module] = None
        # Phase 1 & 2 modules for two-phase detection+generation flow
        self._detection_module: Optional[dspy.Module] = None
        self._response_generation_module: Optional[dspy.Module] = None
        logger.info("Context analyzer initialized")

    def _format_conversation_history(
        self, conversation_history: List[Dict[str, Any]], max_turns: int = 10
    ) -> str:
        """
        Format conversation history for LLM consumption.

        Args:
            conversation_history: List of conversation items with authorRole, message, timestamp
            max_turns: Maximum number of turns to include (default: 10)

        Returns:
            Formatted conversation history as JSON string
        """
        # Take last N turns
        recent_history = (
            conversation_history[-max_turns:] if conversation_history else []
        )

        # Format as readable JSON
        formatted_history = [
            {
                "role": item.get("authorRole", "unknown"),
                "message": item.get("message", ""),
                "timestamp": item.get("timestamp", ""),
            }
            for item in recent_history
        ]

        if not formatted_history:
            return "[]"

        return json.dumps(formatted_history, ensure_ascii=False, indent=2)

    @staticmethod
    def _merge_cost_dicts(
        cost1: Dict[str, Any], cost2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge two cost dictionaries by summing numeric values.

        Args:
            cost1: First cost dictionary
            cost2: Second cost dictionary

        Returns:
            Merged cost dictionary with summed values
        """
        return {
            "total_cost": cost1.get("total_cost", 0) + cost2.get("total_cost", 0),
            "total_tokens": cost1.get("total_tokens", 0) + cost2.get("total_tokens", 0),
            "total_prompt_tokens": cost1.get("total_prompt_tokens", 0)
            + cost2.get("total_prompt_tokens", 0),
            "total_completion_tokens": cost1.get("total_completion_tokens", 0)
            + cost2.get("total_completion_tokens", 0),
            "num_calls": cost1.get("num_calls", 0) + cost2.get("num_calls", 0),
        }

    @observe(name="context_detection_phase1", as_type="generation")
    async def detect_context(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
    ) -> tuple[ContextDetectionResult, Dict[str, Any]]:
        """
        Phase 1: Detect if query is a greeting or can be answered from history.

        Classify-only — no answer generated here. Returns a ContextDetectionResult
        with is_greeting/can_answer_from_context flags and a context_snippet for
        Phase 2 generation.

        Args:
            query: User query to classify
            conversation_history: Full conversation history

        Returns:
            Tuple of (ContextDetectionResult, cost_dict)
        """
        total_turns = len(conversation_history)
        logger.info(
            f"CONTEXT DETECTOR: Phase 1 | Query: '{query[:100]}' | "
            f"History: {total_turns} turns"
        )

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length for detection: {e}")

        formatted_history = self._format_conversation_history(conversation_history)

        self.llm_manager.ensure_global_config()
        try:
            with self.llm_manager.use_task_local():
                if self._detection_module is None:
                    self._detection_module = dspy.ChainOfThought(
                        ContextDetectionSignature
                    )
                response = self._detection_module(
                    conversation_history=formatted_history,
                    user_query=query,
                )

            try:
                detection_data = json.loads(response.detection_result)
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse detection response: {response.detection_result[:100]}"
                )
                detection_data = {
                    "is_greeting": False,
                    "can_answer_from_context": False,
                    "reasoning": "Failed to parse detection response",
                    "context_snippet": None,
                }

            result = ContextDetectionResult(
                is_greeting=detection_data.get("is_greeting", False),
                greeting_type=detection_data.get("greeting_type", "hello"),
                can_answer_from_context=detection_data.get(
                    "can_answer_from_context", False
                ),
                reasoning=detection_data.get("reasoning", "Detection completed"),
                context_snippet=detection_data.get("context_snippet"),
            )
            logger.info(
                f"DETECTION RESULT | Greeting: {result.is_greeting} | "
                f"Can Answer: {result.can_answer_from_context} | "
                f"Has snippet: {result.context_snippet is not None}"
            )

        except Exception as e:
            logger.error(f"Context detection failed: {e}", exc_info=True)
            result = ContextDetectionResult(
                is_greeting=False,
                can_answer_from_context=False,
                reasoning=f"Detection error: {str(e)}",
            )

        cost_dict = get_lm_usage_since(history_length_before)
        update_observation_safe(
            input_data={
                "query": query,
                "history_turns": total_turns,
            },
            output_data={
                "is_greeting": result.is_greeting,
                "greeting_type": result.greeting_type,
                "can_answer_from_context": result.can_answer_from_context,
                "has_context_snippet": bool(result.context_snippet),
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": cost_dict,
            },
        )
        logger.info(
            f"Detection cost | Total: ${cost_dict.get('total_cost', 0):.6f} | "
            f"Tokens: {cost_dict.get('total_tokens', 0)}"
        )
        return result, cost_dict

    @observe(name="context_detection_with_summary_fallback", as_type="chain")
    async def detect_context_with_summary_fallback(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        pre_computed_summary: Optional[str] = None,
    ) -> tuple[ContextDetectionResult, Dict[str, Any]]:
        """
        Phase 1 with summary fallback: detect if query can be answered from history.

        Implements a 3-step flow:
        1. Check the last 10 turns via detect_context().
        2. If cannot answer AND (total history > 10 turns OR a pre-computed summary
           is available from Redis):
           - Use *pre_computed_summary* directly when provided (skips the expensive
             LLM summarisation call).
           - Otherwise generate a concise summary of the older turns.
           - Check whether the query can be answered from that summary.
        3. If still cannot answer, return can_answer=False (workflow falls back to RAG).

        When the summary path succeeds, the returned ContextDetectionResult has:
        - can_answer_from_context=True
        - answered_from_summary=True
        - context_snippet set to the answer extracted from the summary, so that
          Phase 2 (stream_context_response / generate_context_response) can use it
          directly as the context for response generation.

        Args:
            query: User query to classify
            conversation_history: Full conversation history
            pre_computed_summary: Running conversation summary retrieved from Redis.
                When provided the LLM summary-generation step is skipped and this
                value is used directly.  The summary path is also attempted even
                when ``total_turns <= 10`` because evicted rounds may exist in Redis
                beyond what is currently in *conversation_history*.

        Returns:
            Tuple of (ContextDetectionResult, cost_dict)
        """
        total_turns = len(conversation_history)

        # Step 1: check the most recent 10 turns
        result, cost_dict = await self.detect_context(
            query=query, conversation_history=conversation_history
        )

        # If already answered or it's a greeting, return immediately
        if result.is_greeting or result.can_answer_from_context:
            return result, cost_dict

        # Step 2 & 3: try summary-based detection when history is long *or* Redis
        # has a pre-computed summary (which may cover evicted rounds beyond what is
        # currently in memory).
        if total_turns > 10 or pre_computed_summary is not None:
            try:
                if pre_computed_summary is not None:
                    # Redis path: reuse the incremental summary, skip LLM generation.
                    logger.info(
                        "Pre-computed summary available | "
                        "Skipping LLM summary generation, using Redis summary directly"
                    )
                    summary = pre_computed_summary
                    summary_cost: Dict[str, Any] = {
                        "total_cost": 0.0,
                        "total_tokens": 0,
                        "num_calls": 0,
                    }
                else:
                    # On-demand path: summarise older turns via LLM.
                    logger.info(
                        f"History has {total_turns} turns (> 10) | "
                        f"Cannot answer from recent 10 | Attempting summary-based detection"
                    )
                    older_history = conversation_history[:-10]
                    logger.info(f"Summarizing {len(older_history)} older turns")
                    summary, summary_cost = await self._generate_conversation_summary(
                        older_history
                    )

                cost_dict = self._merge_cost_dicts(cost_dict, summary_cost)

                if summary:
                    summary_result, analysis_cost = await self._analyze_from_summary(
                        query=query, summary=summary
                    )
                    cost_dict = self._merge_cost_dicts(cost_dict, analysis_cost)

                    if summary_result.can_answer_from_context and summary_result.answer:
                        logger.info(
                            f"DETECTION: Can answer from summary | "
                            f"Reasoning: {summary_result.reasoning}"
                        )
                        # Surface the summary-derived answer as context_snippet so
                        # Phase 2 can generate a polished response from it.
                        return ContextDetectionResult(
                            is_greeting=False,
                            can_answer_from_context=True,
                            reasoning=summary_result.reasoning,
                            context_snippet=summary_result.answer,
                            answered_from_summary=True,
                        ), cost_dict

                    logger.info(
                        "Cannot answer from summary either | Falling back to RAG"
                    )
                else:
                    logger.warning(
                        "Summary generation returned empty | Falling back to RAG"
                    )

            except Exception as e:
                logger.error(f"Summary-based detection failed: {e}", exc_info=True)
        else:
            logger.info(
                f"History has {total_turns} turns (<= 10) | "
                f"No summary needed | Falling back to RAG"
            )

        return result, cost_dict

    @staticmethod
    def _yield_in_chunks(text: str, chunk_size: int = 5) -> list[str]:
        """Split text into word-group chunks for simulated streaming."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            group = words[i : i + chunk_size]
            trailing = " " if i + chunk_size < len(words) else ""
            chunks.append(" ".join(group) + trailing)
        return chunks

    async def stream_context_response(
        self,
        query: str,
        context_snippet: str,
    ) -> AsyncIterator[str]:
        """
        Phase 2 (streaming): Stream a generated answer using DSPy native streaming.

        Creates a fresh streamify predictor per call (avoids stale StreamListener
        issues that occur when the cached predictor is reused across calls).

        Fallback chain:
        1. DSPy streamify → yield StreamResponse tokens as they arrive.
        2. If no stream tokens received but final Prediction has an answer,
           yield it in word-group chunks.
        3. If that is also empty, call generate_context_response() directly
           and yield the result in word-group chunks.

        Args:
            query: The user query to answer
            context_snippet: Relevant context extracted during Phase 1 detection

        Yields:
            Token strings as they arrive from the LLM (or simulated chunks)
        """
        with safe_observation_context(
            as_type="generation",
            name="context_response_streaming",
            input={"query": query[:500]},
        ) as _generation:
            logger.info(
                f"CONTEXT GENERATOR: Phase 2 streaming | Query: '{query[:100]}'"
            )

            self.llm_manager.ensure_global_config()
            history_length_before = 0
            try:
                lm = dspy.settings.lm
                if lm and hasattr(lm, "history"):
                    history_length_before = len(lm.history)
            except Exception as e:
                logger.warning(
                    f"Failed to get LM history length for streaming generation: {e}"
                )
            output_stream = None
            stream_started = False
            prediction_answer: Optional[str] = None
            assembled_answer = ""  # Collect all yielded tokens here
            try:
                with self.llm_manager.use_task_local():
                    # Always create a fresh StreamListener + streamified predictor so that
                    # the listener's internal state is clean for this call.
                    answer_listener = StreamListener(signature_field_name="answer")
                    stream_predictor: Any = dspy.streamify(
                        dspy.Predict(ContextResponseGenerationSignature),
                        stream_listeners=[answer_listener],
                    )
                    output_stream = stream_predictor(
                        context_snippet=context_snippet,
                        user_query=query,
                    )

                    async for chunk in output_stream:
                        if isinstance(chunk, dspy.streaming.StreamResponse):
                            if chunk.signature_field_name == "answer":
                                stream_started = True
                                assembled_answer += chunk.chunk
                                yield chunk.chunk
                        elif isinstance(chunk, dspy.Prediction):
                            logger.info(
                                "Context response streaming complete (final Prediction received)"
                            )
                            if not stream_started:
                                # Tokens didn't stream — extract answer from the Prediction
                                # directly as first fallback before leaving the LM context.
                                prediction_answer = getattr(chunk, "answer", "") or ""

            except GeneratorExit:
                raise
            except Exception as e:
                logger.error(f"Error during context response streaming: {e}")
                raise
            finally:
                if output_stream is not None:
                    try:
                        await output_stream.aclose()
                    except Exception as cleanup_error:
                        logger.debug(
                            f"Error during context stream cleanup: {cleanup_error}"
                        )

            stream_usage = get_lm_usage_since(history_length_before)

            # Determine final answer for Langfuse output
            final_answer = assembled_answer
            if not stream_started and prediction_answer:
                final_answer = prediction_answer

            try:
                _generation.update(
                    model=_get_current_model_name(),
                    usage_details={
                        "input": stream_usage.get("total_prompt_tokens", 0),
                        "output": stream_usage.get("total_completion_tokens", 0),
                        "total": stream_usage.get("total_tokens", 0),
                    },
                    cost_details={
                        "total": stream_usage.get("total_cost", 0.0),
                    },
                    input={
                        "query": query,
                        "context_snippet_preview": context_snippet[:500],
                    },
                    output={
                        "answer": final_answer,
                        "stream_started": stream_started,
                        "has_prediction_answer": bool(prediction_answer),
                    },
                    metadata={
                        "num_calls": stream_usage.get("num_calls", 0),
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update context streaming observation in Langfuse: {e}"
                )

            if stream_started:
                return

            # Fallback 1: answer was in the final Prediction but didn't stream as tokens
            if prediction_answer:
                logger.warning(
                    "Stream tokens not received — yielding answer from final Prediction in chunks."
                )
                for text_chunk in self._yield_in_chunks(prediction_answer):
                    yield text_chunk
                return

            # Fallback 2: Prediction had no answer either — call generate_context_response
            logger.warning(
                "No answer from streamify — falling back to generate_context_response."
            )
            fallback_answer, _ = await self.generate_context_response(
                query=query, context_snippet=context_snippet
            )
            if fallback_answer:
                for text_chunk in self._yield_in_chunks(fallback_answer):
                    yield text_chunk
            else:
                logger.error(
                    "All Phase 2 streaming fallbacks exhausted — empty response."
                )

    @observe(name="context_response_non_streaming", as_type="generation")
    async def generate_context_response(
        self,
        query: str,
        context_snippet: str,
    ) -> tuple[str, Dict[str, Any]]:
        """
        Phase 2 (non-streaming): Generate a complete answer from context snippet.

        Used for non-streaming mode after Phase 1 detection confirms context can answer.

        Args:
            query: The user query to answer
            context_snippet: Relevant context extracted during Phase 1 detection

        Returns:
            Tuple of (answer_text, cost_dict)
        """
        logger.info(
            f"CONTEXT GENERATOR: Phase 2 non-streaming | Query: '{query[:100]}'"
        )

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length for generation: {e}")

        self.llm_manager.ensure_global_config()
        answer = ""
        try:
            with self.llm_manager.use_task_local():
                if self._response_generation_module is None:
                    self._response_generation_module = dspy.ChainOfThought(
                        ContextResponseGenerationSignature
                    )
                response = self._response_generation_module(
                    context_snippet=context_snippet,
                    user_query=query,
                )
                answer = getattr(response, "answer", "") or ""
                logger.info(
                    f"Context response generated: {len(answer)} chars | "
                    f"Preview: '{answer[:150]}'"
                )
        except Exception as e:
            logger.error(f"Context response generation failed: {e}", exc_info=True)

        cost_dict = get_lm_usage_since(history_length_before)
        update_observation_safe(
            input_data={
                "query": query,
                "context_snippet_preview": context_snippet[:500],
            },
            output_data={
                "answer_preview": answer[:500],
                "has_answer": bool(answer),
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": cost_dict,
            },
        )
        logger.info(
            f"Generation cost | Total: ${cost_dict.get('total_cost', 0):.6f} | "
            f"Tokens: {cost_dict.get('total_tokens', 0)}"
        )
        return answer, cost_dict

    @observe(name="conversation_summary_generation", as_type="generation")
    async def _generate_conversation_summary(
        self,
        older_history: List[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate a concise summary of older conversation turns.

        Args:
            older_history: Conversation turns older than the recent 10

        Returns:
            Tuple of (summary_text, cost_dict)
        """
        logger.info(f"SUMMARY GENERATION: Summarizing {len(older_history)} older turns")

        # Track costs
        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length for summary: {e}")

        # Format older history
        formatted_history = self._format_conversation_history(
            older_history, max_turns=len(older_history)
        )

        # Initialize and run summary module within task-local LLM config
        try:
            self.llm_manager.ensure_global_config()
            with self.llm_manager.use_task_local():
                if self._summary_module is None:
                    self._summary_module = dspy.ChainOfThought(
                        ConversationSummarySignature
                    )
                response = self._summary_module(
                    conversation_history=formatted_history,
                )
                summary = response.summary
                logger.info(
                    f"Summary generated: {len(summary)} chars | "
                    f"Preview: '{summary[:150]}...'"
                )
        except Exception as e:
            logger.error(f"Summary generation failed: {e}", exc_info=True)
            summary = ""

        cost_dict = get_lm_usage_since(history_length_before)
        update_observation_safe(
            input_data={
                "older_history_turns": len(older_history),
            },
            output_data={
                "summary_preview": summary[:500],
                "summary_length": len(summary),
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": cost_dict,
            },
        )
        logger.info(
            f"Summary cost | Total: ${cost_dict.get('total_cost', 0):.6f} | "
            f"Tokens: {cost_dict.get('total_tokens', 0)}"
        )

        return summary, cost_dict

    @observe(name="summary_answerability_analysis", as_type="generation")
    async def _analyze_from_summary(
        self,
        query: str,
        summary: str,
    ) -> tuple[ContextAnalysisResult, Dict[str, Any]]:
        """
        Check if a query can be answered from a conversation summary.

        Args:
            query: User query to check
            summary: Summary of older conversation turns

        Returns:
            Tuple of (ContextAnalysisResult, cost_dict)
        """
        logger.info(
            f"SUMMARY ANALYSIS: Checking query against summary | Query: '{query[:100]}'"
        )

        # Ensure DSPy is configured and run analysis in a task-local LM context
        self.llm_manager.ensure_global_config()
        history_length_before = 0
        with self.llm_manager.use_task_local():
            # Track costs
            try:
                lm = dspy.settings.lm
                if lm and hasattr(lm, "history"):
                    history_length_before = len(lm.history)
            except Exception as e:
                logger.warning(
                    f"Failed to get LM history length for summary analysis: {e}"
                )
            # Initialize summary analysis module if needed
            if self._summary_analysis_module is None:
                self._summary_analysis_module = dspy.ChainOfThought(
                    SummaryAnalysisSignature
                )
            try:
                response = self._summary_analysis_module(
                    conversation_summary=summary,
                    user_query=query,
                )
                # Parse JSON response
                try:
                    analysis_data = json.loads(response.analysis_result)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse summary analysis response: "
                        f"{response.analysis_result[:100]}"
                    )
                    analysis_data = {
                        "can_answer_from_context": False,
                        "answer": None,
                        "reasoning": "Failed to parse summary analysis response",
                    }
                can_answer = analysis_data.get("can_answer_from_context", False)
                answer = analysis_data.get("answer")
                reasoning = analysis_data.get("reasoning", "Summary analysis completed")
                logger.debug(
                    f"Raw summary analysis parsed | "
                    f"can_answer_from_context={can_answer} | "
                    f"has_answer={answer is not None}"
                )
                # Only mark as answerable when both the LLM flag is True AND an answer exists
                can_answer_from_context = bool(can_answer and answer)
                result = ContextAnalysisResult(
                    is_greeting=False,
                    can_answer_from_context=can_answer_from_context,
                    answer=answer,
                    reasoning=reasoning,
                    answered_from_summary=can_answer_from_context,
                )
                logger.info(
                    "SUMMARY ANALYSIS RESULT | "
                    f"Can answer from summary: {can_answer} | "
                    f"Can answer from context: {can_answer_from_context} | "
                    f"Has answer: {answer is not None} | Reasoning: {reasoning}"
                )
            except Exception as e:
                logger.error(f"Summary analysis failed: {e}", exc_info=True)
                result = ContextAnalysisResult(
                    is_greeting=False,
                    can_answer_from_context=False,
                    answer=None,
                    reasoning=f"Summary analysis error: {str(e)}",
                )

        cost_dict = get_lm_usage_since(history_length_before)
        update_observation_safe(
            input_data={
                "query": query,
                "summary_preview": summary[:500],
            },
            output_data={
                "can_answer_from_context": result.can_answer_from_context,
                "answered_from_summary": result.answered_from_summary,
                "has_answer": bool(result.answer),
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": cost_dict,
            },
        )
        logger.info(
            f"Summary analysis cost | Total: ${cost_dict.get('total_cost', 0):.6f} | "
            f"Tokens: {cost_dict.get('total_tokens', 0)}"
        )

        return result, cost_dict

    @observe(name="context_analysis_orchestration", as_type="chain")
    async def analyze_context(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        language: str = "et",
    ) -> tuple[ContextAnalysisResult, Dict[str, Any]]:
        """
        Analyze if query is a greeting or can be answered from conversation history.

        Implements a 3-step flow:
        1. Analyze recent 10 turns for greetings and history-answerable queries
        2. If cannot answer and total history > 10 turns, generate a summary of older turns
        3. Check if the query can be answered from the summary
        4. If still cannot answer, return cannot-answer result (falls through to RAG)

        Args:
            query: User query to analyze
            conversation_history: List of conversation items
            language: Language code (et, en) for response generation

        Returns:
            Tuple of (ContextAnalysisResult, cost_dict)
        """
        total_turns = len(conversation_history)
        logger.info(
            f"CONTEXT ANALYZER: Starting analysis | Query: '{query[:100]}' | "
            f"History: {total_turns} turns | Language: {language}"
        )

        # STEP 1: Analyze recent 10 turns (existing behavior)
        result, cost_dict = await self._analyze_recent_history(
            query=query,
            conversation_history=conversation_history,
            language=language,
        )

        # If greeting or can answer from recent history, return immediately
        if (result.is_greeting or result.can_answer_from_context) and result.answer:
            logger.info(
                f"Answered from recent history | "
                f"Greeting: {result.is_greeting} | From context: {result.can_answer_from_context}"
            )
            return result, cost_dict

        # STEP 2 & 3: If history > 10 turns and couldn't answer from recent, try summary
        if total_turns > 10:
            logger.info(
                f"History exceeds 10 turns ({total_turns} total) | "
                f"Cannot answer from recent 10 | Attempting summary-based analysis"
            )

            # Get older turns (everything before the last 10)
            older_history = conversation_history[:-10]
            logger.info(f"Older history: {len(older_history)} turns to summarize")

            try:
                # Generate summary of older turns
                summary, summary_cost = await self._generate_conversation_summary(
                    older_history
                )
                cost_dict = self._merge_cost_dicts(cost_dict, summary_cost)

                if summary:
                    # Analyze query against summary
                    summary_result, analysis_cost = await self._analyze_from_summary(
                        query=query,
                        summary=summary,
                    )
                    cost_dict = self._merge_cost_dicts(cost_dict, analysis_cost)

                    if summary_result.can_answer_from_context and summary_result.answer:
                        logger.info(
                            f"Answered from conversation summary | "
                            f"Reasoning: {summary_result.reasoning}"
                        )
                        return summary_result, cost_dict

                    logger.info(
                        "Cannot answer from summary either | Falling back to RAG"
                    )
                else:
                    logger.warning(
                        "Summary generation returned empty | Falling back to RAG"
                    )

            except Exception as e:
                logger.error(f"Summary-based analysis failed: {e}", exc_info=True)
        else:
            logger.info(
                f"History has {total_turns} turns (<= 10) | "
                f"No summary needed | Falling back to RAG"
            )

        # Cannot answer from context at all
        logger.info(
            f"CONTEXT ANALYZER FINAL DECISION | "
            f"can_answer_from_context={result.can_answer_from_context} | "
            f"is_greeting={result.is_greeting} | "
            f"answered_from_summary={result.answered_from_summary} | "
            f"has_answer={result.answer is not None} | "
            f"action={'RESPOND' if (result.can_answer_from_context or result.is_greeting) and result.answer else 'FALLBACK_TO_RAG'}"
        )
        return result, cost_dict

    @observe(name="context_analysis_recent_history", as_type="generation")
    async def _analyze_recent_history(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        language: str = "et",
    ) -> tuple[ContextAnalysisResult, Dict[str, Any]]:
        """
        Analyze the query against the most recent conversation turns.

        This is the original analysis logic extracted into its own method.
        Checks for greetings and history-answerable queries in the last 10 turns.

        Args:
            query: User query to analyze
            conversation_history: Full conversation history (last 10 will be used)
            language: Language code for response generation

        Returns:
            Tuple of (ContextAnalysisResult, cost_dict)
        """
        logger.info("STEP 1: Analyzing recent history (last 10 turns)")

        # Track LLM history for cost calculation
        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length: {e}")

        # Format conversation history (last 10 turns)
        formatted_history = self._format_conversation_history(conversation_history)

        # Ensure LM is configured and use task-local context for DSPy operations
        self.llm_manager.ensure_global_config()
        try:
            with self.llm_manager.use_task_local():
                # Initialize DSPy module if not already done
                if self._module is None:
                    self._module = dspy.ChainOfThought(ContextAnalysisSignature)
                # Call LLM for analysis
                logger.info(
                    "Calling LLM for context analysis (greeting/history check)..."
                )
                response = self._module(
                    conversation_history=formatted_history,
                    user_query=query,
                )

            # Parse the analysis result
            analysis_json = response.analysis_result

            # Try to parse JSON response
            try:
                analysis_data = json.loads(analysis_json)
                logger.debug(
                    f"Raw LLM response parsed | "
                    f"can_answer_from_context={analysis_data.get('can_answer_from_context')} | "
                    f"is_greeting={analysis_data.get('is_greeting')} | "
                    f"has_answer={analysis_data.get('answer') is not None}"
                )
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse LLM response as JSON: {analysis_json[:100]}"
                )
                # Fallback: treat as cannot answer
                analysis_data = {
                    "is_greeting": False,
                    "can_answer_from_context": False,
                    "answer": None,
                    "reasoning": "Failed to parse LLM response",
                }

            # Create result object
            result = ContextAnalysisResult(
                is_greeting=analysis_data.get("is_greeting", False),
                can_answer_from_context=analysis_data.get(
                    "can_answer_from_context", False
                ),
                answer=analysis_data.get("answer"),
                reasoning=analysis_data.get("reasoning", "Analysis completed"),
            )

            logger.info(
                f"ANALYSIS RESULT | Greeting: {result.is_greeting} | "
                f"Can Answer from Context: {result.can_answer_from_context} | "
                f"Answer: {result.answer[:100] if result.answer else None} | "
                f"Reasoning: {result.reasoning}"
            )

            # If greeting detected but LLM didn't generate an answer, use fallback
            if result.is_greeting and result.answer is None:
                greeting_type = self._detect_greeting_type(query)
                fallback_answer = get_greeting_response(greeting_type, language)
                result = ContextAnalysisResult(
                    is_greeting=result.is_greeting,
                    can_answer_from_context=result.can_answer_from_context,
                    answer=fallback_answer,
                    reasoning=result.reasoning,
                )

        except Exception as e:
            logger.error(f"Context analysis failed: {e}", exc_info=True)
            # Fallback result
            result = ContextAnalysisResult(
                is_greeting=False,
                can_answer_from_context=False,
                answer=None,
                reasoning=f"Analysis error: {str(e)}",
            )

        # Calculate costs
        cost_dict = get_lm_usage_since(history_length_before)
        update_observation_safe(
            input_data={
                "query": query,
                "history_turns": len(conversation_history),
                "language": language,
            },
            output_data={
                "is_greeting": result.is_greeting,
                "can_answer_from_context": result.can_answer_from_context,
                "has_answer": bool(result.answer),
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": cost_dict,
            },
        )
        logger.info(
            f"Cost tracking | Total cost: ${cost_dict.get('total_cost', 0):.6f} | "
            f"Tokens: {cost_dict.get('total_tokens', 0)} | "
            f"Calls: {cost_dict.get('num_calls', 0)}"
        )

        return result, cost_dict

    def _detect_greeting_type(self, query: str) -> str:
        """
        Detect the type of greeting from the query text.

        Args:
            query: User query string

        Returns:
            Greeting type: 'thanks', 'goodbye', 'casual', or 'hello' (default)
        """
        query_lower = query.lower().strip()
        thanks_keywords = ["thank", "thanks", "tänan", "aitäh", "tänud"]
        goodbye_keywords = ["bye", "goodbye", "nägemist", "tsau", "tšau", "head aega"]
        casual_keywords = ["hei", "hey", "moi", "moikka"]
        for kw in thanks_keywords:
            if kw in query_lower:
                return "thanks"
        for kw in goodbye_keywords:
            if kw in query_lower:
                return "goodbye"
        for kw in casual_keywords:
            if kw in query_lower:
                return "casual"
        return "hello"

    def get_fallback_greeting_response(self, language: str = "et") -> str:
        """
        Get a fallback greeting response without LLM call.

        Used when LLM-based greeting detection fails but we still want
        to provide a friendly response.

        Args:
            language: Language code (et, en)

        Returns:
            Greeting message in the specified language
        """
        greetings = {
            "et": "Tere! Kuidas ma saan sind aidata?",
            "en": "Hello! How can I help you?",
        }
        return greetings.get(language, greetings["et"])

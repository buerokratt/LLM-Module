from __future__ import annotations
from typing import List, Dict, Any, Tuple, AsyncIterator, Optional
import re
import dspy
from src.loki_logger import LokiLogger
import asyncio
import dspy.streaming
from dspy.streaming import StreamListener
from langfuse import observe
from src.utils.observation_utils import (
    safe_observation_context,
    update_observation_safe,
)

from src.llm_orchestrator_config.llm_ochestrator_constants import OUT_OF_SCOPE_MESSAGE
from src.utils.cost_utils import get_lm_usage_since
from src.optimization.optimized_module_loader import get_module_loader
from src.vector_indexer.constants import ResponseGenerationConstants

# Initialize Loki logger for response generator
logger = LokiLogger(service_name="response-generator")


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


class ResponseGenerator(dspy.Signature):
    """Produce a grounded answer from the provided context ONLY.

    CRITICAL LANGUAGE RULE:
    - The answer MUST be in the SAME language as the input question
    - Estonian question → Estonian answer
    - Russian question → Russian answer
    - English question → English answer
    - Maintain the natural language flow and grammar of the detected language

    Rules:
    - Use ONLY the provided context blocks; do not invent facts.
    - If the context is insufficient, set questionOutOfLLMScope=true and say so briefly.
    - Do not include citations in the 'answer' field.
    """

    question: str = dspy.InputField(
        desc="User's question. Answer in the SAME language as this question."
    )
    context_blocks: List[str] = dspy.InputField()
    citations: List[str] = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Human-friendly answer in THE SAME LANGUAGE as the question, without citations"
    )
    questionOutOfLLMScope: bool = dspy.OutputField(
        desc="True if context is insufficient to answer"
    )


class ScopeChecker(dspy.Signature):
    """Quick check if question can be answered from context.

    LANGUAGE NOTE: This is an internal check, language doesn't matter for scope determination.

    Rules:
    - Return True ONLY if context is completely insufficient
    - Return False if context has ANY relevant information
    - Be lenient - prefer False over True
    """

    question: str = dspy.InputField()
    context_blocks: List[str] = dspy.InputField()
    out_of_scope: bool = dspy.OutputField(
        desc="True ONLY if context is completely insufficient"
    )


def build_context_and_citations(
    chunks: List[Dict[str, Any]], use_top_k: Optional[int] = None
) -> Tuple[List[str], List[str], bool]:
    """
    Turn retriever chunks -> numbered context blocks and source labels.
    Returns (blocks, labels, has_real_context).
    """
    if use_top_k is None:
        use_top_k = ResponseGenerationConstants.DEFAULT_MAX_BLOCKS
    logger.info(f"Building context from {len(chunks)} chunks (top_k={use_top_k}).")
    blocks: List[str] = []
    labels: List[str] = []
    for i, ch in enumerate(chunks[:use_top_k]):
        text = (ch.get("text") or "").strip()
        meta: Dict[str, Any] = ch.get("meta") or {}
        source_file = meta.get("source_file")
        source = meta.get("source")
        label = source_file or source or f"Chunk-{i + 1}"
        if text:
            blocks.append(f"[Context {i + 1}]\n{text}")
            labels.append(str(label))

    has_real_context = len(blocks) > 0
    if not has_real_context:
        blocks = ["[Context 1]\n(No relevant context available.)"]
        labels = ["No source"]
    logger.info(
        f"Created {len(blocks)} context blocks. Has real context: {has_real_context}."
    )
    return blocks, labels, has_real_context


def _should_flag_out_of_scope(
    answer_text: str, has_real_context: bool, require_citation_marker: bool = False
) -> bool:
    """
    Heuristics to decide out-of-scope when model output is ambiguous:
    - No real context was supplied
    - Very short or empty answer
    - (Optional) No citation markers like [1], [2] present if require_citation_marker is True
    """
    if not has_real_context:
        return True
    if not (answer_text or "").strip():
        return True
    if require_citation_marker and not re.search(r"\[\d+\]", answer_text or ""):
        return True
    return False


class ResponseGeneratorAgent(dspy.Module):
    """
    Creates a grounded, humanized answer from retrieved chunks.
    Now supports loading optimized modules from DSPy optimization process.
    Supports both streaming and non-streaming generation.
    Returns a dict: {"answer": str, "questionOutOfLLMScope": bool, "usage": dict}
    """

    def __init__(
        self,
        max_retries: int = 2,
        use_optimized: bool = True,
        custom_instructions_prefix: str = "",
    ) -> None:
        super().__init__()
        self._max_retries = max(0, int(max_retries))
        self._custom_instructions_prefix = custom_instructions_prefix

        # Attribute to cache the streamified predictor
        self._stream_predictor: Optional[Any] = None

        # Try to load optimized module
        self._optimized_metadata = {}
        if use_optimized:
            self._predictor = self._load_optimized_or_base()
        else:
            logger.info("Using base (non-optimized) generator module")
            self._predictor = dspy.Predict(ResponseGenerator)
            self._optimized_metadata = {
                "component": "generator",
                "version": "base",
                "optimized": False,
            }

        # Separate scope checker for quick pre-checks
        self._scope_checker = dspy.Predict(ScopeChecker)

    def _load_optimized_or_base(self) -> dspy.Module:
        """
        Load optimized generator module if available, otherwise use base.

        Returns:
            DSPy module (optimized or base)
        """
        try:
            loader = get_module_loader()
            optimized_module, metadata = loader.load_generator_module()

            self._optimized_metadata = metadata

            if optimized_module is not None:
                logger.info(
                    f"Loaded OPTIMIZED generator module "
                    f"(version: {metadata.get('version', 'unknown')}, "
                    f"optimizer: {metadata.get('optimizer', 'unknown')})"
                )

                metrics = metadata.get("metrics", {})
                if metrics:
                    logger.info(
                        f"  Optimization metrics: "
                        f"avg_quality={metrics.get('average_quality', 'N/A')}"
                    )

                return optimized_module
            else:
                logger.warning(
                    f"Could not load optimized generator module, using base module. "
                    f"Reason: {metadata.get('error', 'Not found')}"
                )
                return dspy.Predict(ResponseGenerator)

        except Exception as e:
            logger.error(f"Error loading optimized generator: {str(e)}")
            logger.warning("Falling back to base generator module")
            self._optimized_metadata = {
                "component": "generator",
                "version": "base",
                "optimized": False,
                "error": str(e),
            }
            return dspy.Predict(ResponseGenerator)

    def get_module_info(self) -> Dict[str, Any]:
        """Get information about the loaded module."""
        return self._optimized_metadata.copy()

    def _get_stream_predictor(self) -> Any:
        """Get or create the cached streamified predictor."""
        if self._stream_predictor is None:
            logger.info("Initializing streamify wrapper for ResponseGeneratorAgent")

            # Define a listener for the 'answer' field of the ResponseGenerator signature
            answer_listener = StreamListener(signature_field_name="answer")

            # Wrap the internal predictor
            # self._predictor is the dspy.Predict(ResponseGenerator) or optimized module
            self._stream_predictor = dspy.streamify(
                self._predictor, stream_listeners=[answer_listener]
            )
            logger.info("Streamify wrapper created and cached on agent.")

        return self._stream_predictor

    async def stream_response(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_blocks: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens directly from LLM using DSPy's native streaming.

        Args:
            question: User's question
            chunks: Retrieved context chunks
            max_blocks: Maximum number of context blocks (default: ResponseGenerationConstants.DEFAULT_MAX_BLOCKS)

        Yields:
            Token strings as they arrive from the LLM
        """
        with safe_observation_context(
            as_type="generation",
            name="response_generation_streaming",
            input={"question": question[:500], "chunks_count": len(chunks)},
        ) as _generation:
            if max_blocks is None:
                max_blocks = ResponseGenerationConstants.DEFAULT_MAX_BLOCKS

            logger.info(
                f"Starting NATIVE DSPy streaming for question with {len(chunks)} chunks"
            )

            history_length_before = 0
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)

            # Apply custom instructions while keeping the user question first, if provided
            augmented_question = question
            if self._custom_instructions_prefix:
                augmented_question = f"{question}\n\n{self._custom_instructions_prefix}"
                logger.debug(
                    f"Applied custom instructions after question for streaming ({len(self._custom_instructions_prefix)} chars)"
                )

            output_stream = None
            streamed_tokens: List[str] = []
            final_answer_text: str = ""
            try:
                # Build context
                context_blocks, citation_labels, has_real_context = (
                    build_context_and_citations(chunks, use_top_k=max_blocks)
                )

                if not has_real_context:
                    logger.warning(
                        "No real context available for streaming, yielding nothing."
                    )
                    return

                # Get the streamified predictor
                stream_predictor = self._get_stream_predictor()

                # Call the streamified predictor with augmented question
                logger.info("Calling streamified predictor with signature inputs...")
                output_stream = stream_predictor(
                    question=augmented_question,
                    context_blocks=context_blocks,
                    citations=citation_labels,
                )

                stream_started = False
                try:
                    async for chunk in output_stream:
                        # The stream yields StreamResponse objects for tokens
                        # and a final Prediction object
                        if isinstance(chunk, dspy.streaming.StreamResponse):
                            if chunk.signature_field_name == "answer":
                                stream_started = True
                                streamed_tokens.append(chunk.chunk)
                                yield chunk.chunk  # Yield the token string
                        elif isinstance(chunk, dspy.Prediction):
                            # The final prediction object is yielded last
                            logger.info(
                                "Streaming complete, final Prediction object received."
                            )
                            full_answer = getattr(chunk, "answer", "[No answer field]")
                            if isinstance(full_answer, str):
                                final_answer_text = full_answer
                            logger.debug(f"Full streamed answer: {full_answer}")
                except GeneratorExit:
                    # Generator was closed early (e.g., by guardrails violation)
                    logger.info("Stream generator closed early - cleaning up")
                    # Properly close the stream
                    if output_stream is not None:
                        try:
                            await output_stream.aclose()
                        except Exception as close_error:
                            logger.debug(
                                f"Error closing stream (expected): {close_error}"
                            )
                        output_stream = None  # Prevent double-close in finally block
                    raise

                if not stream_started:
                    logger.warning(
                        "Streaming call finished but no 'answer' tokens were received."
                    )

            except Exception as e:
                logger.error(f"Error during native DSPy streaming: {str(e)}")
                logger.exception("Full traceback:")
                raise
            finally:
                # Ensure cleanup even if exception occurs
                if output_stream is not None:
                    try:
                        await output_stream.aclose()
                    except Exception as cleanup_error:
                        logger.debug(f"Error during cleanup (aclose): {cleanup_error}")

            usage_info = get_lm_usage_since(history_length_before)
            stream_output = final_answer_text or "".join(streamed_tokens)
            if not stream_output:
                stream_output = "[stream completed with empty output]"

            try:
                _generation.update(
                    model=_get_current_model_name(),
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
                        "streaming": True,
                    },
                    output=stream_output,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update streaming generation usage in Langfuse: {e}"
                )

    @observe(name="response_scope_check", as_type="generation")
    async def check_scope_quick(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_blocks: Optional[int] = None,
    ) -> bool:
        """
        Quick async check if question is out of scope.

        Args:
            question: User's question
            chunks: Retrieved context chunks
            max_blocks: Maximum context blocks to use (default: ResponseGenerationConstants.DEFAULT_MAX_BLOCKS)

        Returns:
            True if out of scope, False if in scope
        """
        if max_blocks is None:
            max_blocks = ResponseGenerationConstants.DEFAULT_MAX_BLOCKS

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception:
            pass

        try:
            context_blocks, _, has_real_context = build_context_and_citations(
                chunks, use_top_k=max_blocks
            )

            if not has_real_context:
                usage_info = get_lm_usage_since(history_length_before)
                update_observation_safe(
                    input_data={
                        "question": question,
                        "chunks_count": len(chunks),
                        "max_blocks": max_blocks,
                    },
                    output_data={"out_of_scope": True, "reason": "no_context"},
                    metadata={
                        "model": _get_current_model_name(),
                        "usage": usage_info,
                    },
                )
                return True

            # Use DSPy to quickly check scope
            result = await asyncio.to_thread(
                self._scope_checker, question=question, context_blocks=context_blocks
            )

            out_of_scope = getattr(result, "out_of_scope", False)
            usage_info = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "question": question,
                    "chunks_count": len(chunks),
                    "max_blocks": max_blocks,
                },
                output_data={"out_of_scope": bool(out_of_scope)},
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage_info,
                },
            )
            logger.info(
                f"Quick scope check result: {'OUT OF SCOPE' if out_of_scope else 'IN SCOPE'}"
            )

            return bool(out_of_scope)

        except Exception as e:
            logger.error(f"Scope check error: {e}")
            usage_info = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "question": question,
                    "chunks_count": len(chunks),
                    "max_blocks": max_blocks,
                },
                output_data={"out_of_scope": False, "error": str(e)},
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage_info,
                },
            )
            # On error, assume in-scope to allow generation to proceed
            return False

    def _predict_once(
        self, question: str, context_blocks: List[str], citation_labels: List[str]
    ) -> dspy.Prediction:
        """Single LM call. Returns Prediction object."""
        result = self._predictor(
            question=question, context_blocks=context_blocks, citations=citation_labels
        )
        logger.info(f"LLM output - answer: {getattr(result, 'answer', '')[:200]}...")
        logger.info(
            f"LLM output - out_of_scope: {getattr(result, 'questionOutOfLLMScope', None)}"
        )
        return result

    def _validate_prediction(self, pred: dspy.Prediction) -> bool:
        """Validate that prediction has required fields with correct types."""
        try:
            answer = getattr(pred, "answer", None)
            out_of_scope = getattr(pred, "questionOutOfLLMScope", None)

            if not isinstance(answer, str):
                return False
            if not isinstance(out_of_scope, bool):
                return False
            return True
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return False

    @observe(name="response_generation_non_streaming", as_type="generation")
    def forward(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_blocks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Non-streaming forward pass for backward compatibility."""
        if max_blocks is None:
            max_blocks = ResponseGenerationConstants.DEFAULT_MAX_BLOCKS

        logger.info(f"Generating response for question: '{question}'")

        # Apply custom instructions while keeping the user question first, if provided
        augmented_question = question
        if self._custom_instructions_prefix:
            augmented_question = f"{question}\n\n{self._custom_instructions_prefix}"
            logger.debug(
                f"Applied custom instructions after question ({len(self._custom_instructions_prefix)} chars)"
            )

        lm = dspy.settings.lm
        history_length_before = len(lm.history) if lm and hasattr(lm, "history") else 0

        context_blocks, citation_labels, has_real_context = build_context_and_citations(
            chunks, use_top_k=max_blocks
        )

        pred = self._predict_once(augmented_question, context_blocks, citation_labels)
        valid = self._validate_prediction(pred)

        attempts = 0
        while not valid and attempts < self._max_retries:
            attempts += 1
            logger.warning(f"Retry attempt {attempts}/{self._max_retries}")

            pred = self._predictor(
                question=augmented_question,
                context_blocks=context_blocks,
                citations=citation_labels,
                config={"rollout_id": attempts, "temperature": 0.1},
            )
            valid = self._validate_prediction(pred)

        usage_info = get_lm_usage_since(history_length_before)

        if not valid:
            logger.warning(
                "Failed to obtain valid prediction after retries. Using fallback."
            )
            answer = getattr(pred, "answer", "")
            if not isinstance(answer, str):
                answer = str(answer) if answer else ""

            scope_flag = _should_flag_out_of_scope(answer, has_real_context)
            if not answer or scope_flag:
                answer = OUT_OF_SCOPE_MESSAGE
                scope_flag = True

            update_observation_safe(
                input_data={
                    "question": question,
                    "chunks_count": len(chunks),
                    "max_blocks": max_blocks,
                    "attempts": attempts,
                    "valid_prediction": False,
                },
                output_data={
                    "questionOutOfLLMScope": scope_flag,
                    "answer_preview": answer[:500],
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage_info,
                },
            )

            return {
                "answer": answer,
                "questionOutOfLLMScope": scope_flag,
                "usage": usage_info,
            }

        ans: str = getattr(pred, "answer", "")
        scope: bool = bool(getattr(pred, "questionOutOfLLMScope", False))

        if scope is False and _should_flag_out_of_scope(ans, has_real_context):
            logger.warning("Flipping out-of-scope to True based on heuristics.")
            scope = True

        update_observation_safe(
            input_data={
                "question": question,
                "chunks_count": len(chunks),
                "max_blocks": max_blocks,
                "attempts": attempts,
                "valid_prediction": True,
            },
            output_data={
                "questionOutOfLLMScope": scope,
                "answer_preview": ans[:500],
            },
            metadata={
                "model": _get_current_model_name(),
                "usage": usage_info,
            },
        )

        return {
            "answer": ans.strip(),
            "questionOutOfLLMScope": scope,
            "usage": usage_info,
        }


async def stream_response_native(
    agent: ResponseGeneratorAgent,
    question: str,
    chunks: List[Dict[str, Any]],
    max_blocks: int = 10,
) -> AsyncIterator[str]:
    """
    Compatibility wrapper for the new stream_response method.
    DEPRECATED: Use agent.stream_response() instead.
    This function is kept for backward compatibility.

    Args:
        agent: ResponseGeneratorAgent instance
        question: User's question
        chunks: Retrieved context chunks
        max_blocks: Maximum number of context blocks

    Yields:
        Token strings as they arrive from the LLM
    """
    async for token in agent.stream_response(question, chunks, max_blocks):
        yield token

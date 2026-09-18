"""Multi-API response formatter using DSPy — synthesises multiple API results into one answer."""

from typing import Any, AsyncIterator, Dict, List, Tuple, Union

import dspy
import dspy.streaming
from dspy.streaming import StreamListener
from langfuse import observe
from src.loki_logger import LokiLogger
from llm_orchestrator_config.llm_ochestrator_constants import get_localized_message
from src.utils.cost_utils import get_lm_usage_since
from src.utils.observation_utils import (
    safe_observation_context,
    update_observation_safe,
)
from tool_classifier.api_response_formatter import (
    APIResponseFormatterModule,
    _LANGUAGE_NAMES,
    build_params_context,
)


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


logger = LokiLogger(service_name="api-tool-calling")

_MAX_TOTAL_RESPONSE_BYTES: int = 100_000

_MULTI_FORMATTER_ERROR_MESSAGES: Dict[str, str] = {
    "et": "Vastuste kuvamine ebaõnnestus. Palun proovige uuesti.",
    "ru": "Не удалось отобразить ответы. Пожалуйста, попробуйте ещё раз.",
    "en": "I was unable to format the responses. Please try again.",
}
"""Localized fallback shown when MultiResponseFormatterModule raises an exception."""


class MultiResponseFormatterSignature(dspy.Signature):
    """Synthesise multiple API results into a single, coherent natural-language answer.

    CRITICAL LANGUAGE RULE:
    - ALWAYS write the unified_answer in the language specified by response_language.
    - IGNORE the language of any text inside api_results_block — the data may contain
      names or labels in a different language; the answer must still be in response_language.
    - IGNORE the language of user_query for output language decisions — short follow-up
      messages are unreliable indicators. Always use response_language.

    If custom_instructions is non-empty, follow those rules with HIGHEST PRIORITY —
    they override defaults (e.g. language policy, tone, formatting style).

    Rules:
    - Present each API result in its OWN separate paragraph, in the same order they
      appear in api_results_block. Separate paragraphs with a single blank line.
      Do NOT merge or blend information from different endpoints into one paragraph.
    - Each paragraph should be self-contained and answer the part of the user's query
      that corresponds to that endpoint.
    - Address EVERY result section in api_results_block. Do not silently omit any endpoint.
    - Within each paragraph, present the information naturally in prose or a short list.
      Do NOT return raw JSON or wrap content in code blocks.
    - If a result section includes a 'Parameters used:' line, acknowledge the date range or
      time period for that result in your discussion
      (e.g. 'For the period 1 Jan to 30 Jun 2026, XYZ showed...').
    - If a result is marked [EMPTY RESPONSE], politely mention that no data was available
      from that source without dwelling on it.
    - If a result contains an error field or is marked [FAILED], acknowledge the failure
      briefly and in a friendly, non-technical way, then continue with the remaining results.
    - If api_results_block contains only empty or failed results, respond with a polite
      message that no results were available.
    - If num_results is 1, a single paragraph is fine (no need to split).
    - Output must be clean text — no markdown headers (##), no code blocks (```), no raw
      JSON. The answer must be ready for direct display to the user.
    - Be concise but complete. Prioritise the most relevant information for the user's query.

    STRICT ENDING RULE — HIGHEST PRIORITY:
    The unified_answer MUST end immediately after the last data point. It is FORBIDDEN to
    append any sentence that:
    - offers to provide more details (e.g. "If you need statistics for a specific member...")
    - invites the user to ask a follow-up question (e.g. "Let me know if...", "Feel free to ask...")
    - mentions that a dataset is large or partial (e.g. "only a sample is shown here")
    - suggests the user can specify a name, party, or other filter
    The very last character of unified_answer must be part of the actual data, not a helper offer.
    """

    user_query: str = dspy.InputField(
        desc="The user's original question or request, in Estonian, Russian, or English"
    )
    api_results_block: str = dspy.InputField(
        desc=(
            "A labeled text block containing one section per API result. "
            "Each section is headed by the endpoint name and description, "
            "followed by the raw response string. "
            "May contain EMPTY RESPONSE or FAILED markers for individual results."
        )
    )
    response_language: str = dspy.InputField(
        desc=(
            "The language to write the answer in, detected from the user's first message: "
            "'English', 'Estonian', or 'Russian'. "
            "Always use this — do not infer language from api_results_block content."
        )
    )
    custom_instructions: str = dspy.InputField(
        desc=(
            "Optional system-level instructions configured by the organisation "
            "(e.g. 'Always respond in Estonian', 'Use structured format'). "
            "Empty string when no custom config is active. "
            "When non-empty, follow these rules with highest priority."
        )
    )
    num_results: str = dspy.InputField(
        desc=(
            "The total number of API results included in api_results_block, "
            "as a plain integer string (e.g. '3'). "
            "Use this to verify all results are addressed."
        )
    )

    unified_answer: str = dspy.OutputField(
        desc=(
            "One paragraph per API result, separated by blank lines, in the same order "
            "as api_results_block. Each paragraph is self-contained and covers exactly "
            "one endpoint's data. Written entirely in the language specified by "
            "response_language. No raw JSON, no code blocks, no markdown headers. "
            "MUST end after the last data point. "
            "FORBIDDEN: any closing sentence offering more help, inviting follow-up questions, "
            "mentioning that the dataset is partial, or suggesting the user specify a name/party."
        )
    )


class MultiResponseFormatterModule(dspy.Module):
    """DSPy Module that synthesises multiple API results into one natural-language answer."""

    def __init__(self, custom_instructions: str = "") -> None:
        """Initialise formatter with a direct DSPy Predict.

        Args:
            custom_instructions: Optional organisation-level prompt rules (e.g. language
                policy). Passed verbatim to the DSPy predictor on every call. Defaults
                to empty string (no custom config).
        """
        super().__init__()
        self.formatter = dspy.Predict(MultiResponseFormatterSignature)
        self._custom_instructions = custom_instructions

    @observe(name="multi_api_response_formatting_llm", as_type="generation")
    def forward(
        self,
        user_query: str,
        api_results: List[
            Tuple[str, str, Union[str, Dict[str, Any], List[Any]], Dict[str, Any]]
        ],
        detected_language: str = "en",
    ) -> str:
        """Synthesise multiple API results into a single natural-language answer.

        Args:
            user_query: The user's original question.
            api_results: A list of
                ``(endpoint_name, endpoint_description, api_response, collected_params)``
                4-tuples. ``api_response`` may be a JSON string, dict, or list.
                ``collected_params`` is used to generate a date-range
                acknowledgment inside each result section.
            detected_language: ISO language code from the agentic loop session
                ('en', 'et', 'ru'). Defaults to 'en'. This is the authoritative
                language for the answer.

        Returns:
            A clean, unified natural-language answer ready for display to the user.
        """
        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(
                f"Failed to get LM history length for multi response formatting: {e}"
            )

        try:
            results_block = self._build_results_block(api_results)
            response_language = _LANGUAGE_NAMES.get(detected_language, "English")

            result = self.formatter(
                user_query=user_query,
                api_results_block=results_block,
                response_language=response_language,
                custom_instructions=self._custom_instructions,
                num_results=str(len(api_results)),
            )
            unified_answer = result.unified_answer
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "num_results": len(api_results),
                    "response_language": response_language,
                },
                output_data={
                    "unified_answer_preview": str(unified_answer)[:500],
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": False,
                },
            )
            return unified_answer  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(
                f"MultiResponseFormatterModule.forward failed: {e}", exc_info=True
            )
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "num_results": len(api_results),
                    "detected_language": detected_language,
                },
                output_data={"error": str(e)},
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": False,
                },
            )
            safe_language = (
                detected_language
                if detected_language in _MULTI_FORMATTER_ERROR_MESSAGES
                else "en"
            )
            return get_localized_message(_MULTI_FORMATTER_ERROR_MESSAGES, safe_language)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def stream_forward_multi(
        self,
        user_query: str,
        api_results: List[
            Tuple[str, str, Union[str, Dict[str, Any], List[Any]], Dict[str, Any]]
        ],
        detected_language: str = "en",
    ) -> AsyncIterator[str]:
        """Stream unified_answer tokens using DSPy native streaming.

        API results are pre-resolved; only the LLM synthesis step is streamed.
        Yields individual token strings as they arrive from the LLM.

        Fallback chain:
        1. DSPy ``StreamResponse`` tokens (true token-by-token streaming)
        2. Final ``dspy.Prediction.unified_answer`` (if streamify yields no tokens)
        3. Blocking ``forward()`` call (if no Prediction was received)
        4. Localized error message on any exception.

        Args:
            user_query: The user's original question.
            api_results: A list of
                ``(endpoint_name, endpoint_description, api_response, collected_params)``
                4-tuples. ``api_response`` may be a JSON string, dict, or list.
            detected_language: ISO code ('en', 'et', 'ru'). Defaults to 'en'.

        Yields:
            Token strings from the LLM ``unified_answer`` field.
        """
        safe_language = (
            detected_language
            if detected_language in _MULTI_FORMATTER_ERROR_MESSAGES
            else "en"
        )
        output_stream = None

        with safe_observation_context(
            as_type="generation",
            name="multi_api_response_formatting_streaming",
            input={
                "user_query": user_query[:500],
                "num_results": len(api_results),
                "detected_language": detected_language,
            },
        ) as generation:
            history_length_before = 0
            try:
                lm = dspy.settings.lm
                if lm and hasattr(lm, "history"):
                    history_length_before = len(lm.history)
            except Exception as e:
                logger.warning(
                    f"Failed to get LM history length for multi response streaming: {e}"
                )

            try:
                results_block = self._build_results_block(api_results)
                response_language = _LANGUAGE_NAMES.get(detected_language, "English")

                # A fresh wrapper is created on every call because dspy.configure(lm=...)
                # is called per request. A cached wrapper retains a stale LM reference and
                # yields a bare dspy.Prediction instead of StreamResponse tokens.
                logger.debug(
                    "MultiResponseFormatterModule: creating fresh streamify wrapper "
                    "for unified_answer field"
                )
                listener = StreamListener(signature_field_name="unified_answer")
                stream_predictor: Any = dspy.streamify(
                    self.formatter, stream_listeners=[listener]
                )
                output_stream = stream_predictor(
                    user_query=user_query,
                    api_results_block=results_block,
                    response_language=response_language,
                    custom_instructions=self._custom_instructions,
                    num_results=str(len(api_results)),
                )

                stream_started = False
                token_count = 0
                accumulated: list[str] = []
                final_prediction: dspy.Prediction | None = None
                async for chunk in output_stream:
                    if isinstance(chunk, dspy.streaming.StreamResponse):
                        if chunk.signature_field_name == "unified_answer":
                            stream_started = True
                            token_count += 1
                            accumulated.append(chunk.chunk)
                            yield chunk.chunk
                    elif isinstance(chunk, dspy.Prediction):
                        final_prediction = chunk
                        # dspy.streamify did not stream individual tokens — yield the
                        # full answer from the final Prediction as a single frame.
                        if not stream_started:
                            answer = getattr(chunk, "unified_answer", None)
                            if answer:
                                logger.info(
                                    "MultiResponseFormatterModule.stream_forward_multi: "
                                    "no StreamResponse tokens — yielding full Prediction answer"
                                )
                                stream_started = True
                                accumulated.append(answer)
                                yield answer

                assembled_answer = "".join(accumulated)

                if stream_started and token_count > 0:
                    logger.debug(
                        f"MultiResponseFormatterModule.stream_forward_multi: "
                        f"streamed {token_count} tokens"
                    )
                    # DSPy streaming can drop the last few tokens before EOS.
                    # The final dspy.Prediction holds the authoritative complete answer.
                    # Yield any tail that wasn't delivered as StreamResponse chunks.
                    if final_prediction is not None:
                        full_answer = getattr(final_prediction, "unified_answer", None)
                        if full_answer:
                            streamed_text = assembled_answer
                            if full_answer.startswith(streamed_text) and len(
                                full_answer
                            ) > len(streamed_text):
                                tail = full_answer[len(streamed_text) :]
                                if tail.strip():
                                    logger.debug(
                                        f"MultiResponseFormatterModule.stream_forward_multi: "
                                        f"yielding {len(tail)} missing tail chars from Prediction"
                                    )
                                    assembled_answer += tail
                                    yield tail
                            elif streamed_text and not full_answer.startswith(
                                streamed_text
                            ):
                                logger.warning(
                                    "MultiResponseFormatterModule.stream_forward_multi: "
                                    "streamed output is not a prefix of final Prediction; "
                                    "skipping tail reconciliation"
                                )

                if not stream_started:
                    # Last-resort fallback: blocking forward() — covers cases where
                    # dspy.streamify yields neither StreamResponse nor Prediction.
                    logger.warning(
                        "MultiResponseFormatterModule.stream_forward_multi: "
                        "streamify produced no tokens and no Prediction — using blocking forward()"
                    )
                    result = self.forward(
                        user_query=user_query,
                        api_results=api_results,
                        detected_language=detected_language,
                    )
                    assembled_answer = result
                    yield result

                usage = get_lm_usage_since(history_length_before)
                try:
                    generation.update(
                        input={
                            "user_query": user_query,
                            "num_results": len(api_results),
                            "detected_language": detected_language,
                        },
                        output=assembled_answer,
                        usage_details={
                            "input": usage.get("total_prompt_tokens", 0),
                            "output": usage.get("total_completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        },
                        cost_details={
                            "total": usage.get("total_cost", 0.0),
                        },
                        metadata={
                            "stream_started": stream_started,
                            "chunk_count": token_count,
                            "streaming": True,
                        },
                    )
                except Exception as update_error:
                    logger.debug(
                        f"Langfuse generation update skipped for multi response streaming: {update_error}"
                    )

            except Exception as e:
                logger.error(
                    f"MultiResponseFormatterModule.stream_forward_multi failed: {e}",
                    exc_info=True,
                )
                usage = get_lm_usage_since(history_length_before)
                try:
                    generation.update(
                        input={
                            "user_query": user_query,
                            "num_results": len(api_results),
                            "detected_language": detected_language,
                        },
                        output={"error": str(e)},
                        usage_details={
                            "input": usage.get("total_prompt_tokens", 0),
                            "output": usage.get("total_completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        },
                        cost_details={
                            "total": usage.get("total_cost", 0.0),
                        },
                        metadata={"streaming": True},
                    )
                except Exception as update_error:
                    logger.debug(
                        f"Langfuse error update skipped for multi response streaming: {update_error}"
                    )
                yield get_localized_message(
                    _MULTI_FORMATTER_ERROR_MESSAGES, safe_language
                )
            finally:
                if output_stream is not None:
                    try:
                        await output_stream.aclose()
                    except Exception as cleanup_error:
                        logger.debug(f"Error during stream cleanup: {cleanup_error}")

    @staticmethod
    def _build_results_block(
        api_results: List[
            Tuple[str, str, Union[str, Dict[str, Any], List[Any]], Dict[str, Any]]
        ],
    ) -> str:
        """Serialise a list of API results into a labeled text block for the LLM.

        Each result becomes a section headed by the endpoint name and description,
        optionally followed by a ``Parameters used: ...`` line when date-typed
        values are present in ``collected_params``, then the normalised, annotated,
        and truncated response string.
        A combined size guard caps the total block at ``_MAX_TOTAL_RESPONSE_BYTES``
        to prevent LLM context overflow.

        Args:
            api_results: List of
                ``(endpoint_name, endpoint_description, api_response, collected_params)``
                4-tuples.

        Returns:
            A multiline string with one clearly delimited section per result.
        """
        if not api_results:
            return "[NO RESULTS: No API results were provided]"

        sections: List[str] = []
        total_bytes = 0

        for idx, (name, description, raw_response, collected_params) in enumerate(
            api_results, start=1
        ):
            normalized = APIResponseFormatterModule._normalize_response(raw_response)
            normalized = APIResponseFormatterModule._annotate_empty(normalized)
            normalized = APIResponseFormatterModule._truncate_if_needed(normalized)

            params_context = build_params_context(collected_params)
            params_line = (
                f"Parameters used: {params_context}\n" if params_context else ""
            )

            section = (
                f"--- Result {idx}: {name} ---\n"
                f"Description: {description}\n"
                f"{params_line}"
                f"Response:\n{normalized}\n"
            )
            section_bytes = len(section.encode("utf-8"))

            if total_bytes + section_bytes > _MAX_TOTAL_RESPONSE_BYTES:
                remaining = _MAX_TOTAL_RESPONSE_BYTES - total_bytes
                if remaining > 0:
                    suffix = (
                        "\n[NOTE: Combined results truncated due to total size limit]"
                    )
                    suffix_bytes = len(suffix.encode("utf-8"))
                    encoded = section.encode("utf-8")
                    truncated = encoded[: max(0, remaining - suffix_bytes)].decode(
                        "utf-8", errors="ignore"
                    )
                    sections.append(truncated + suffix)
                    total_bytes = _MAX_TOTAL_RESPONSE_BYTES
                break

            sections.append(section)
            total_bytes += section_bytes

        return "\n".join(sections)

"""API response formatter using DSPy — converts raw JSON API responses to natural language."""

import json
import re
from typing import Any, AsyncIterator, Dict, List, Union

import dspy
import dspy.streaming
from dspy.streaming import StreamListener
from langfuse import observe
from src.utils.observation_utils import (
    safe_observation_context,
    update_observation_safe,
)
from src.loki_logger import LokiLogger
from llm_orchestrator_config.llm_ochestrator_constants import get_localized_message
from src.utils.cost_utils import get_lm_usage_since

logger = LokiLogger(service_name="api-tool-calling")

_MAX_ITEMS: int = 500
_MAX_RESPONSE_BYTES: int = 150_000


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


class APIResponseFormatterSignature(dspy.Signature):
    """Convert a raw API JSON response into a natural-language answer for the user.

    CRITICAL LANGUAGE RULE:
    - ALWAYS write the formatted_answer in the language specified by response_language.
    - IGNORE the language of any text inside api_response — the data may contain names or
      labels in a different language; the answer must still be in response_language.
    - IGNORE the language of user_query for output language decisions — short follow-up
      messages are unreliable indicators. Always use response_language.

    If custom_instructions is non-empty, follow those rules with HIGHEST PRIORITY —
    they override defaults (e.g. language policy, tone, formatting style).

    Rules:
    - Format data in a readable way using bullet points, numbered lists, or natural prose.
      Do NOT return raw JSON or wrap content in code blocks.
    - If api_response is empty, null, or marked as [EMPTY RESPONSE], respond with a polite
      message that no results were found for the query.
    - If api_response contains an error field or error status, explain the issue to the user
      in a friendly, non-technical way.
    - If the data contains more than 50 items, summarize the key highlights rather than
      listing every item. Always mention the total count when summarizing.
    - Output must be clean text — no markdown headers (##), no code blocks (```), no raw
      JSON. The answer must be ready for direct display to the user.
    - Be concise but complete. Prioritize the most relevant information for the user's query.

    STRICT ENDING RULE — HIGHEST PRIORITY:
    The formatted_answer MUST end immediately after the last data point. It is FORBIDDEN to
    append any sentence that:
    - offers to provide more details (e.g. "If you need statistics for a specific member...")
    - invites the user to ask a follow-up question (e.g. "Let me know if...", "Feel free to ask...")
    - mentions that a dataset is large or partial (e.g. "only a sample is shown here")
    - suggests the user can specify a name, party, or other filter
    The very last character of formatted_answer must be part of the actual data, not a helper offer.
    """

    user_query: str = dspy.InputField(
        desc="The user's original question or request, in Estonian, Russian, or English"
    )
    api_response: str = dspy.InputField(
        desc=(
            "The raw JSON response from the API, as a string. "
            "May be empty, null, an error, or a large dataset."
        )
    )
    endpoint_description: str = dspy.InputField(
        desc=(
            "A short description of what the API endpoint does "
            "(e.g., 'Get public holidays for a country')"
        )
    )
    response_language: str = dspy.InputField(
        desc=(
            "The language to write the answer in, detected from the user's first message: "
            "'English', 'Estonian', or 'Russian'. "
            "Always use this — do not infer language from api_response content."
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
    query_params_context: str = dspy.InputField(
        desc=(
            "If non-empty, briefly acknowledge the time period or filter at the "
            "start of the answer (e.g. 'For the period 2026-01-01 to 2026-12-31, ...'). "
            "Empty string when no date or time filter was applied."
        )
    )

    formatted_answer: str = dspy.OutputField(
        desc=(
            "A clean, natural-language answer derived from the api_response, "
            "written entirely in the language specified by response_language. "
            "No raw JSON, no code blocks, no markdown headers. "
            "MUST end after the last data point. "
            "FORBIDDEN: any closing sentence offering more help, inviting follow-up questions, "
            "mentioning that the dataset is partial, or suggesting the user specify a name/party."
        )
    )


_LANGUAGE_NAMES: Dict[str, str] = {"en": "English", "et": "Estonian", "ru": "Russian"}

# ISO-8601 datetime pattern with optional timezone (Z or ±HH:MM).
# Anchored at both start (^) and end ($) to avoid partial matches like "2026-01-01foo".
_DATE_VALUE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?)?$"
)


def build_params_context(collected_params: Dict[str, Any]) -> str:
    """Format date/datetime valued params into a readable context string.

    Detects values that look like ISO-8601 date strings (``YYYY-MM-DD`` or
    ``YYYY-MM-DDTHH:MM:SS``) and formats them as human-readable key-value pairs.
    Returns an empty string when no date-type values are found.

    Args:
        collected_params: Dict of param names → values collected from the user.

    Returns:
        A comma-separated string such as
        ``"start date: 2026-01-01, end date: 2026-12-31"``
        or ``""`` when no date params are present.
    """
    parts: List[str] = []
    for name, value in collected_params.items():
        if isinstance(value, str) and _DATE_VALUE_RE.match(value):
            # Convert camelCase to spaced words, lower-case.
            readable = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()
            readable = readable.replace("_", " ")
            parts.append(f"{readable}: {value}")
    return ", ".join(parts)


_FORMATTER_ERROR_MESSAGES: Dict[str, str] = {
    "et": "Vastuse kuvamine ebaõnnestus. Palun proovige uuesti.",
    "ru": "Не удалось отобразить ответ. Пожалуйста, попробуйте ещё раз.",
    "en": "I was unable to format the response. Please try again.",
}
"""Localized fallback shown when APIResponseFormatterModule.forward() raises an exception."""


class APIResponseFormatterModule(dspy.Module):
    """DSPy Module that converts raw API JSON responses into natural-language answers."""

    def __init__(self, custom_instructions: str = "") -> None:
        """Initialize formatter with a direct DSPy Predict.

        Args:
            custom_instructions: Optional organisation-level prompt rules (e.g. language
                policy).  Passed verbatim to the DSPy predictor on every call.  Defaults
                to empty string (no custom config).
        """
        super().__init__()
        self.formatter = dspy.Predict(APIResponseFormatterSignature)
        self._custom_instructions = custom_instructions

    @observe(name="api_response_formatting_llm", as_type="generation")
    def forward(
        self,
        user_query: str,
        api_response: Union[str, Dict[str, Any], List[Any]],
        endpoint_description: str,
        detected_language: str = "en",
        collected_params: Dict[str, Any] | None = None,
    ) -> str:
        """Convert a raw API response to a natural-language answer.

        Args:
            user_query: The user's original question.
            api_response: The raw API response — a JSON string, dict, or list.
            endpoint_description: A short description of what the endpoint does.
            detected_language: ISO language code from the agentic loop session
                ('en', 'et', 'ru'). Defaults to 'en'. This is the authoritative
                language for the answer — the LLM will not infer it from the data.
            collected_params: Dict of param names → values collected from the user.
                Used to build a date-range acknowledgment prefix when date params
                are present. Defaults to None.

        Returns:
            A clean, natural-language answer ready for display to the user.
        """

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(
                f"Failed to get LM history length for response formatting: {e}"
            )

        collected_params = collected_params or {}

        try:
            normalized = self._normalize_response(api_response)
            normalized = self._annotate_empty(normalized)
            normalized = self._truncate_if_needed(normalized)
            response_language = _LANGUAGE_NAMES.get(detected_language, "English")
            params_context = build_params_context(collected_params)

            result = self.formatter(
                user_query=user_query,
                api_response=normalized,
                endpoint_description=endpoint_description,
                response_language=response_language,
                custom_instructions=self._custom_instructions,
                query_params_context=params_context,
            )
            formatted_answer = result.formatted_answer
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "api_response": api_response,
                    "endpoint_description": endpoint_description,
                    "response_language": response_language,
                },
                output_data={
                    "formatted_answer_preview": str(formatted_answer)[:500],
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": False,
                },
            )
            return formatted_answer  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(
                f"APIResponseFormatterModule.forward failed: {e}", exc_info=True
            )
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "endpoint_description": endpoint_description,
                    "detected_language": detected_language,
                    "api_response": api_response,
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
                if detected_language in _FORMATTER_ERROR_MESSAGES
                else "en"
            )
            return get_localized_message(_FORMATTER_ERROR_MESSAGES, safe_language)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_stream_predictor(self) -> Any:
        """Return a fresh streamified predictor for each call.

        A new wrapper is created on every invocation because ``dspy.configure(lm=...)``
        is called per request (each ``LLMManager`` instantiation reconfigures the global
        DSPy LM).  A cached wrapper retains a stale reference to the old LM patch and
        produces a bare ``dspy.Prediction`` instead of ``StreamResponse`` tokens on
        subsequent calls.  Re-creating the wrapper is cheap (no LLM I/O).
        """
        logger.debug(
            "APIResponseFormatterModule: creating fresh streamify wrapper "
            "for formatted_answer field"
        )
        listener = StreamListener(signature_field_name="formatted_answer")
        return dspy.streamify(self.formatter, stream_listeners=[listener])

    async def stream_forward(
        self,
        user_query: str,
        api_response: Union[str, Dict[str, Any], List[Any]],
        endpoint_description: str,
        detected_language: str = "en",
        collected_params: Dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream formatted_answer tokens using DSPy native streaming.
        Yields individual token strings as they arrive from the LLM.

        Fallback chain:
        1. DSPy ``StreamResponse`` tokens (true token-by-token streaming)
        2. Final ``dspy.Prediction.formatted_answer`` (if streamify yields no tokens)
        3. Blocking ``forward()`` call (if no Prediction was received)
        4. Localized error message on any exception.

        Args:
            user_query: The user's original question.
            api_response: Raw API response (dict, list, or string).
            endpoint_description: Short description of what the endpoint does.
            detected_language: ISO code ('en', 'et', 'ru'). Defaults to 'en'.
            collected_params: Dict of param names → values collected from the user.
                Used to build a date-range acknowledgment prefix. Defaults to None.

        Yields:
            Token strings from the LLM ``formatted_answer`` field.
        """
        collected_params = collected_params or {}
        safe_language = (
            detected_language
            if detected_language in _FORMATTER_ERROR_MESSAGES
            else "en"
        )
        output_stream = None

        with safe_observation_context(
            as_type="generation",
            name="api_response_formatting_streaming",
            input={
                "user_query": user_query[:500],
                "endpoint_description": endpoint_description,
                "detected_language": detected_language,
            },
        ) as generation:
            output_stream = None

            history_length_before = 0
            try:
                lm = dspy.settings.lm
                if lm and hasattr(lm, "history"):
                    history_length_before = len(lm.history)
            except Exception as e:
                logger.warning(
                    "Failed to get LM history length for response formatting streaming: "
                    f"{e}"
                )
            try:
                normalized = self._normalize_response(api_response)
                normalized = self._annotate_empty(normalized)
                normalized = self._truncate_if_needed(normalized)
                response_language = _LANGUAGE_NAMES.get(detected_language, "English")
                params_context = build_params_context(collected_params)

                stream_predictor = self._get_stream_predictor()
                output_stream = stream_predictor(
                    user_query=user_query,
                    api_response=normalized,
                    endpoint_description=endpoint_description,
                    response_language=response_language,
                    custom_instructions=self._custom_instructions,
                    query_params_context=params_context,
                )

                stream_started = False
                token_count = 0
                accumulated: list[str] = []
                final_prediction: dspy.Prediction | None = None
                async for chunk in output_stream:
                    if isinstance(chunk, dspy.streaming.StreamResponse):
                        if chunk.signature_field_name == "formatted_answer":
                            stream_started = True
                            token_count += 1
                            accumulated.append(chunk.chunk)
                            yield chunk.chunk
                    elif isinstance(chunk, dspy.Prediction):
                        final_prediction = chunk
                        # dspy.streamify did not stream individual tokens — yield the
                        # full answer from the final Prediction as a single frame.
                        if not stream_started:
                            answer = getattr(chunk, "formatted_answer", None)
                            if answer:
                                logger.info(
                                    "APIResponseFormatterModule.stream_forward: "
                                    "no StreamResponse tokens — yielding full Prediction answer"
                                )
                                stream_started = True
                                accumulated.append(answer)
                                yield answer

                assembled_answer = "".join(accumulated)

                if stream_started and token_count > 0:
                    logger.debug(
                        f"APIResponseFormatterModule.stream_forward: streamed {token_count} tokens"
                    )
                    # DSPy streaming can drop the last few tokens before EOS.
                    # The final dspy.Prediction holds the authoritative complete answer.
                    # Yield any tail that wasn't delivered as StreamResponse chunks.
                    if final_prediction is not None:
                        full_answer = getattr(
                            final_prediction, "formatted_answer", None
                        )
                        if full_answer:
                            streamed_text = assembled_answer
                            if full_answer.startswith(streamed_text) and len(
                                full_answer
                            ) > len(streamed_text):
                                tail = full_answer[len(streamed_text) :]
                                if tail.strip():
                                    logger.debug(
                                        f"APIResponseFormatterModule.stream_forward: "
                                        f"yielding {len(tail)} missing tail chars from Prediction"
                                    )
                                    assembled_answer += tail
                                    yield tail
                            elif streamed_text and not full_answer.startswith(
                                streamed_text
                            ):
                                logger.warning(
                                    "APIResponseFormatterModule.stream_forward: "
                                    "streamed output is not a prefix of final Prediction; "
                                    "skipping tail reconciliation"
                                )

                if not stream_started:
                    # Last-resort fallback: blocking forward() — covers cases where
                    # dspy.streamify yields neither StreamResponse nor Prediction.
                    logger.warning(
                        "APIResponseFormatterModule.stream_forward: "
                        "streamify produced no tokens and no Prediction — using blocking forward()"
                    )
                    result = self.forward(
                        user_query=user_query,
                        api_response=api_response,
                        endpoint_description=endpoint_description,
                        detected_language=detected_language,
                        collected_params=collected_params,
                    )
                    assembled_answer = result
                    yield result

                usage = get_lm_usage_since(history_length_before)
                try:
                    generation.update(
                        input={
                            "user_query": user_query,
                            "api_response": api_response,
                            "endpoint_description": endpoint_description,
                            "detected_language": detected_language,
                        },
                        output=assembled_answer,
                        metadata={
                            "stream_started": stream_started,
                            "chunk_count": token_count,
                            "num_calls": usage.get("num_calls", 0),
                            "streaming": True,
                        },
                        usage_details={
                            "input": usage.get("total_prompt_tokens", 0),
                            "output": usage.get("total_completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        },
                        cost_details={
                            "total": usage.get("total_cost", 0.0),
                        },
                    )
                except Exception as update_error:
                    logger.debug(
                        "Langfuse generation update skipped for response formatting "
                        f"streaming: {update_error}"
                    )

            except Exception as e:
                logger.error(
                    f"APIResponseFormatterModule.stream_forward failed: {e}",
                    exc_info=True,
                )
                usage = get_lm_usage_since(history_length_before)
                try:
                    generation.update(
                        input={
                            "user_query": user_query,
                            "api_response": api_response,
                            "endpoint_description": endpoint_description,
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
                        metadata={
                            "num_calls": usage.get("num_calls", 0),
                            "streaming": True,
                        },
                    )
                except Exception as update_error:
                    logger.debug(
                        "Langfuse error update skipped for response formatting "
                        f"streaming: {update_error}"
                    )
                yield get_localized_message(_FORMATTER_ERROR_MESSAGES, safe_language)
            finally:
                if output_stream is not None:
                    try:
                        await output_stream.aclose()
                    except Exception as cleanup_error:
                        logger.debug(f"Error during stream cleanup: {cleanup_error}")

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_response(
        api_response: Union[str, Dict[str, Any], List[Any]],
    ) -> str:
        """Convert dict or list inputs to a JSON string."""
        if isinstance(api_response, (dict, list)):
            return json.dumps(api_response, ensure_ascii=False)
        return str(api_response)

    @staticmethod
    def _annotate_empty(api_response_str: str) -> str:
        """Annotate obviously empty responses so the LLM handles them gracefully."""
        try:
            parsed = json.loads(api_response_str)
        except (json.JSONDecodeError, ValueError):
            return api_response_str

        if parsed is None or parsed == [] or parsed == {}:
            return "[EMPTY RESPONSE: The API returned no data for this query]"
        return api_response_str

    @staticmethod
    def _truncate_if_needed(api_response_str: str) -> str:
        """Truncate responses that exceed the item count or byte-size limits."""
        try:
            parsed = json.loads(api_response_str)
        except (json.JSONDecodeError, ValueError):
            parsed = None

        if isinstance(parsed, list) and len(parsed) > _MAX_ITEMS:
            total = len(parsed)
            truncated = json.dumps(parsed[:_MAX_ITEMS], ensure_ascii=False)
            api_response_str = (
                f"[NOTE: Response truncated to {_MAX_ITEMS} of {total} total items]\n"
                + truncated
            )

        encoded = api_response_str.encode("utf-8")
        if len(encoded) > _MAX_RESPONSE_BYTES:
            suffix = "\n[NOTE: Response truncated due to size limit]"
            suffix_bytes = len(suffix.encode("utf-8"))
            truncated_str = encoded[: _MAX_RESPONSE_BYTES - suffix_bytes].decode(
                "utf-8", errors="ignore"
            )
            logger.warning(
                f"API response exceeded byte limit: {len(encoded)} bytes — truncating to {len(truncated_str.encode('utf-8'))} bytes"
            )
            return truncated_str + suffix

        return api_response_str

"""API parameter extraction using DSPy."""

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

import dspy
import dspy.streaming
from dspy.streaming import StreamListener
from langfuse import observe
from src.utils.observation_utils import update_observation_safe
from src.loki_logger import LokiLogger
from src.utils.error_utils import generate_error_id

from src.utils.cost_utils import get_lm_usage_since

_TRUTHY_STRINGS = {"true", "yes", "jah", "1", "on", "õige", "да"}
_FALSY_STRINGS = {"false", "no", "ei", "0", "off", "vale", "нет"}

_MAX_HISTORY_TURNS = 5

logger = LokiLogger(service_name="api-tool-calling")

# Regex patterns to strip format hints from parameter descriptions before
# they are fed to the question-generation prompt. This prevents the LLM
# from including format instructions (e.g. "YYYY-MM-DD") in its questions.
_FORMAT_HINT_PATTERNS: List[re.Pattern[str]] = [
    # Parenthesised format hints: (YYYY-MM-DD), (ISO 8601), (2-letter code), (HH:MM:SS)
    re.compile(
        r"\s*\([^)]*(?:YYYY|MM|DD|HH|SS|ISO\s*\d*|letter|format)[^)]*\)",
        re.IGNORECASE,
    ),
    # Trailing phrases: "in format YYYY-MM-DD" or "in the format YYYY-MM-DD"
    re.compile(r"\s*,?\s*in\s+(?:the\s+)?format\s+\S+", re.IGNORECASE),
]


def strip_format_hints(description: str) -> str:
    """Remove format hints from a parameter description.

    Strips patterns such as ``(YYYY-MM-DD)``, ``(ISO 8601)``,
    ``(2-letter code)``, ``(HH:MM:SS)``, and trailing
    ``in the format YYYY-MM-DD`` phrases before the schema is passed to the
    LLM for both extraction and question generation.  This prevents format
    instructions from leaking into clarifying questions (e.g. "What date?
    (YYYY-MM-DD)").  Type coercion is handled independently by
    :meth:`~ParamExtractionModule._validate_param_type`, so the format hints
    are not needed by the extractor.
    """
    for pattern in _FORMAT_HINT_PATTERNS:
        description = pattern.sub("", description)
    return description.strip()


class ParamExtractionResult(TypedDict):
    """Return contract for ParamExtractionModule.forward()."""

    extracted_params: Dict[str, Any]
    missing_required: List[str]
    clarifying_question: str


class ParamExtractionSignature(dspy.Signature):
    """Extract API parameter values from user message and conversation history.

    CRITICAL LANGUAGE RULE:
    - Understand Estonian, English, and Russian input
    - Generate clarifying_question in the language specified by session_language
    - IGNORE the language of the current user_message for output language decisions —
      short follow-up messages ("I'm not sure", "2026-01-01") are unreliable indicators.
      Always use session_language.

    If custom_instructions is non-empty, follow those rules with HIGHEST PRIORITY —
    they override defaults (e.g. language policy, tone) for the clarifying_question output.

    Extraction rules:
    - Extract values for ALL parameters listed in params_schema that appear in user_message
      or conversation_history, regardless of whether they are already in already_collected
    - If the user explicitly provides a new or corrected value for a parameter that is
      already in already_collected, still extract the new value — it will override the old one
    - Only skip extraction for a param if the user has NOT mentioned it at all in this turn
    - Validate types: dates must be ISO 8601 (YYYY-MM-DD), integers must be whole numbers,
      numbers must be numeric, booleans must be true or false
    - SINGLE-VALUE ASSIGNMENT RULE: When the user's message contains exactly ONE value of a
      given type (e.g. one date) and MULTIPLE required parameters of the same type are still
      missing (e.g. both startDate and endDate are missing), assign that single value to the
      FIRST such missing required parameter in the order they appear in params_schema — never
      to a later one. For example, if startDate appears before endDate in params_schema and
      both are missing, a lone date like "2026-04-01" must be assigned to startDate, not endDate.

    missing_required rules:
    - List every required parameter (required=true in schema) whose value is absent
      AFTER combining already_collected with newly extracted params
    - Do NOT list optional parameters as missing

    clarifying_question rules:
    - After extraction, check whether ALL required params are now satisfied
      (i.e., present in already_collected OR just extracted).
    - If ALL required params are satisfied, return the literal string "none".
      Do NOT add any acknowledgment or thank-you — just return "none".
    - If ONE OR MORE required params are still missing, generate ONE friendly message
      that first acknowledges what was just collected (if anything) then asks for
      all remaining missing params in the same message.

    Acknowledgment rules (when params are still missing):
    - CASE A — turn_count is not '0' AND extracted_params is non-empty
      (the user was responding to a follow-up question and gave something):
        Prepend a brief natural acknowledgment of the values just provided, then
        ask for the remaining missing params.
        Use the parameter's description field to describe what was received, NOT
        the raw camelCase name (e.g. "Thank you for providing the group" not
        "Thank you for providing group").
        Keep it concise: two sentences — one acknowledgment sentence, one question sentence.
        Join the two clauses with a period and a space, never with a dash, em dash,
        or any other punctuation connector.
        VARY the acknowledgment opener each turn — do NOT repeat the same phrase.
        Use different natural openers across turns, for example:
          English:  "Got it.", "Perfect.", "Got the address.", "Noted.",
                    "Thanks for that.", "Great.", "Received."
          Estonian: "Selge.", "Hästi.", "Aitäh.", "Märgitud.", "Tubli."
          Russian:  "Принято.", "Понял.", "Отлично.", "Хорошо.", "Записал."
        Match the tone and language to session_language.
    - CASE B — turn_count is '0' (this is the user's original query):
        Do NOT add any acknowledgment or thank-you, even if params were extracted
        from the original message. Simply ask for the remaining missing params.
        The user was not responding to a question — they just asked naturally.
    - CASE C — turn_count is not '0' AND extracted_params is empty (the user did NOT answer what was asked):
        Do NOT add any acknowledgment or thank-you. Ask for the missing params
        again directly, rephrasing if helpful.
    - Use each missing parameter's description field to phrase the question naturally
      (e.g., "Which country and date would you like to use?" not "Provide countryIsoCode and startDate")
    - Never expose raw parameter names (camelCase identifiers) to the user
    - NEVER include format requirements, expected formats, format examples, or
      structural hints (such as "YYYY-MM-DD", "ISO 8601", "2-letter code",
      "in the format...") in the question — only ask WHAT information is needed,
      not HOW it should be formatted. The system handles format conversion
      internally from any natural-language input the user provides.
    - When intent_groups is a non-empty JSON array with 2 or more groups, structure
      the question with a clear natural-language separation per group — e.g.,
      "Which address or place would you like to search for, and what is the unique
      identifier of the initiative you want to check?" Use conjunctions like "and"
      to link the groups naturally into one fluent sentence.
    """

    user_message: str = dspy.InputField(
        desc="Current turn message from the user in Estonian, English, or Russian"
    )
    conversation_history: str = dspy.InputField(
        desc="Recent conversation turns formatted as 'role: message', one per line"
    )
    session_language: str = dspy.InputField(
        desc=(
            "ISO language code for the response language detected from the user's "
            "first message: 'en' (English), 'et' (Estonian), 'ru' (Russian). "
            "Always generate clarifying_question in this language."
        )
    )
    params_schema: str = dspy.InputField(
        desc='JSON array of parameter schemas: [{"name": str, "type": str, "required": bool, "description": str}]'
    )
    already_collected: str = dspy.InputField(
        desc=(
            "JSON object of parameter values collected in prior turns: {param_name: value}. "
            "Use this as context to understand what has already been provided. "
            "If the user explicitly mentions a new value for a param already here, "
            "still extract the new value — corrections are allowed."
        )
    )
    turn_count: str = dspy.InputField(
        desc=(
            "Zero-based turn index as a string. '0' means this is the user's original query "
            "(not a response to a follow-up question). '1' or higher means the user is "
            "responding to a clarifying question asked by the system."
        )
    )
    custom_instructions: str = dspy.InputField(
        desc=(
            "Optional system-level instructions configured by the organisation "
            "(e.g. 'Always respond in Estonian', 'Use formal tone'). "
            "Empty string when no custom config is active. "
            "When non-empty, follow these rules with highest priority for the clarifying_question."
        )
    )
    intent_groups: str = dspy.InputField(
        desc=(
            "Optional JSON array grouping missing required params by their owning intent/endpoint: "
            '[{"intent": "<endpoint name>", "missing_param_descriptions": ["desc1", ...]}, ...]. '
            "Empty JSON array [] when not in multi-intent mode or when no intents have "
            "missing params. "
            "When 1 group is present, use the intent name to add context to the question "
            "(e.g., 'For the electricity market prices, what is the start and end date?'). "
            "When 2+ groups are present, use this to phrase a single question "
            "that clearly separates the needs of each intent with natural conjunctions."
        )
    )

    extracted_params: str = dspy.OutputField(
        desc='Valid JSON object of newly extracted parameters only: {"param_name": value}. Empty object {} if nothing new found.'
    )
    missing_required: str = dspy.OutputField(
        desc='Valid JSON array of required parameter names still missing after extraction: ["param1", "param2"]. Empty array [] if all required params are satisfied.'
    )
    clarifying_question: str = dspy.OutputField(
        desc=(
            "One natural-language message in session_language. "
            'Return the literal string "none" if all required params are collected (no acknowledgment). '
            "If params are still missing and turn_count is '0' (original query): "
            "skip any acknowledgment and ask for the missing params directly. "
            "If params are still missing and turn_count > 0 and extracted_params is non-empty: "
            "briefly acknowledge the values just provided (using natural descriptions, not raw param names), "
            "then ask for all remaining missing params in the same message. "
            "If params are still missing and turn_count > 0 and extracted_params is empty: "
            "skip any acknowledgment and ask for the missing params directly. "
            'Never include format instructions or examples (e.g. "YYYY-MM-DD", '
            '"ISO 8601", "2-letter code") — only ask what information is needed. '
            "When intent_groups contains 1 group, prefix the question with the intent "
            "name for context (e.g. 'For the electricity market prices, what is the "
            "start and end date?'). "
            "When intent_groups contains 2+ groups, the question must use natural conjunctions "
            "(e.g. 'and') to separate the missing params of each group into one fluent sentence."
        )
    )


class ParamExtractionModule(dspy.Module):
    """DSPy Module for API parameter extraction from natural language."""

    def __init__(self, custom_instructions: str = "") -> None:
        """Initialize param extraction module with Predict (direct prediction).

        Args:
            custom_instructions: Optional organisation-level prompt rules (e.g. language
                policy).  Passed verbatim to the DSPy predictor on every call.  Defaults
                to empty string (no custom config).
        """
        super().__init__()
        self.extractor = dspy.Predict(ParamExtractionSignature)
        self._custom_instructions = custom_instructions

    @observe(name="api_param_extraction_llm", as_type="generation")
    def forward(
        self,
        user_message: str,
        params_schema: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        already_collected: Optional[Dict[str, Any]] = None,
        session_language: str = "en",
        turn_count: int = 0,
        intent_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> ParamExtractionResult:
        """
        Extract parameter values from user message and conversation history.

        Args:
            user_message: Current turn message from the user
            params_schema: List of parameter schema dicts with name, type, required, description
            conversation_history: Recent conversation messages (optional)
            already_collected: Parameter values collected in prior turns (optional)
            session_language: Language code detected on turn 0 ('en', 'et', 'ru').
                All clarifying questions will be generated in this language.
            intent_groups: Optional list of intent-group dicts, each with an
                ``"intent"`` key (endpoint name) and a
                ``"missing_param_descriptions"`` key (list of human-readable
                descriptions for the missing params of that intent).  Pass this
                when a single user query maps to multiple intents that each have
                outstanding required parameters; the extractor will then phrase
                one clarifying question that addresses every group's missing
                params using natural conjunctions.  Pass ``None`` (default) or
                an empty list when operating in single-intent mode.

        Returns:
            ParamExtractionResult with extracted_params, missing_required, clarifying_question
        """
        already_collected = already_collected or {}

        history_text = self._format_conversation_history(conversation_history)
        sanitized_schema = [
            {**p, "description": strip_format_hints(p.get("description", ""))}
            if isinstance(p, dict)
            else p
            for p in params_schema
        ]
        params_schema_json = json.dumps(sanitized_schema, ensure_ascii=False)
        already_collected_json = json.dumps(already_collected, ensure_ascii=False)
        intent_groups_json = json.dumps(intent_groups or [], ensure_ascii=False)

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(
                f"Failed to get LM history length for parameter extraction: {e}"
            )

        result = None
        _t0 = time.time()
        try:
            result = self.extractor(
                user_message=user_message,
                conversation_history=history_text,
                session_language=session_language,
                params_schema=params_schema_json,
                already_collected=already_collected_json,
                turn_count=str(turn_count),
                custom_instructions=self._custom_instructions,
                intent_groups=intent_groups_json,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.debug(
                f"ParamExtractionModule: LLM extraction complete"
                f" | event_type=param_extraction_llm_complete"
                f" turn_count={turn_count} duration_ms={_duration_ms}"
            )
            parsed = self._parse_prediction(result, params_schema, already_collected)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(parsed["missing_required"]),
                    "clarifying_question": parsed["clarifying_question"],
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "extracted_params_count": len(parsed["extracted_params"]),
                    "streaming": False,
                },
            )
            return parsed
        except json.JSONDecodeError as e:
            _raw_ep = getattr(result, "extracted_params", None)
            _raw_mr = getattr(result, "missing_required", None)
            logger.error(
                f"ParamExtractionModule: JSON parse error in forward"
                f" | event_type=param_extraction_json_error"
                f" error_id={generate_error_id()}"
                f" exc_type={type(e).__name__} exc_msg={e!r}"
                f" raw_extracted_params={_raw_ep!r}"
                f" raw_missing_required={_raw_mr!r}"
            )
            fallback = self._safe_defaults(params_schema, already_collected)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(fallback["missing_required"]),
                    "clarifying_question": fallback["clarifying_question"],
                    "error": f"JSON parse error: {e}",
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": False,
                },
            )
            return fallback

        except Exception as e:
            logger.error(
                f"ParamExtractionModule: forward failed"
                f" | event_type=param_extraction_failed"
                f" error_id={generate_error_id()}"
                f" exc_type={type(e).__name__} exc_msg={e!r}"
            )
            fallback = self._safe_defaults(params_schema, already_collected)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(fallback["missing_required"]),
                    "clarifying_question": fallback["clarifying_question"],
                    "error": str(e),
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": False,
                },
            )
            return fallback

    def _get_stream_predictor(self) -> Any:
        """Return a fresh streamified predictor for each call.

        See :meth:`~api_response_formatter.APIResponseFormatterModule._get_stream_predictor`
        for the rationale — ``dspy.configure(lm=...)`` is called per request so any cached
        wrapper becomes stale.  Re-creating is cheap (no LLM I/O).
        """
        logger.debug(
            "ParamExtractionModule: creating fresh streamify wrapper "
            "for clarifying_question field"
        )
        listener = StreamListener(signature_field_name="clarifying_question")
        return dspy.streamify(self.extractor, stream_listeners=[listener])

    @observe(name="api_param_extraction_streaming", as_type="generation")
    async def stream_forward(
        self,
        user_message: str,
        params_schema: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        already_collected: Optional[Dict[str, Any]] = None,
        session_language: str = "en",
        turn_count: int = 0,
        intent_groups: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[str], ParamExtractionResult]:
        """Stream clarifying_question tokens while returning the full extraction result.

        Uses the same DSPy streamify pattern as
        :meth:`~api_response_formatter.APIResponseFormatterModule.stream_forward`.
        Collects ``clarifying_question`` tokens as they arrive from the LLM and
        parses ``extracted_params`` / ``missing_required`` from the final
        ``dspy.Prediction``.

        If all required params are already collected the question will be ``"none"``
        and the returned token list will be empty.

        Args:
            user_message: Current turn message from the user.
            params_schema: Parameter schema list.
            conversation_history: Recent conversation messages (optional).
            already_collected: Parameter values from prior turns (optional).
            session_language: Language code (``'en'``, ``'et'``, ``'ru'``).
            intent_groups: Optional list of intent-group dicts, each with an
                ``"intent"`` key (endpoint name) and a
                ``"missing_param_descriptions"`` key (list of human-readable
                descriptions for the missing params of that intent).  Pass this
                when a single user query maps to multiple intents that each have
                outstanding required parameters; the extractor will then phrase
                one clarifying question that addresses every group's missing
                params using natural conjunctions, streamed token by token.
                Pass ``None`` (default) or an empty list for single-intent mode.

        Returns:
            Tuple of ``(question_tokens, extraction_result)``.
            ``question_tokens`` is empty when no clarifying question is needed.
        """
        already_collected = already_collected or {}

        history_text = self._format_conversation_history(conversation_history)
        sanitized_schema = [
            {**p, "description": strip_format_hints(p.get("description", ""))}
            if isinstance(p, dict)
            else p
            for p in params_schema
        ]
        params_schema_json = json.dumps(sanitized_schema, ensure_ascii=False)
        already_collected_json = json.dumps(already_collected, ensure_ascii=False)
        intent_groups_json = json.dumps(intent_groups or [], ensure_ascii=False)

        output_stream = None

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(
                "Failed to get LM history length for parameter extraction "
                f"streaming: {e}"
            )

        try:
            stream_predictor = self._get_stream_predictor()
            output_stream = stream_predictor(
                user_message=user_message,
                conversation_history=history_text,
                session_language=session_language,
                params_schema=params_schema_json,
                already_collected=already_collected_json,
                turn_count=str(turn_count),
                custom_instructions=self._custom_instructions,
                intent_groups=intent_groups_json,
            )

            tokens: List[str] = []
            prediction: Any = None

            async for chunk in output_stream:
                if isinstance(chunk, dspy.streaming.StreamResponse):
                    if chunk.signature_field_name == "clarifying_question":
                        tokens.append(chunk.chunk)
                elif isinstance(chunk, dspy.Prediction):
                    prediction = chunk

            if prediction is None:
                logger.warning(
                    "ParamExtractionModule: no Prediction received"
                    " | event_type=stream_no_prediction_received"
                )
                result = await asyncio.to_thread(
                    self.forward,
                    user_message,
                    params_schema,
                    conversation_history,
                    already_collected,
                    session_language,
                    turn_count,
                    intent_groups,
                )
                fallback_token = result["clarifying_question"]
                return (
                    [fallback_token] if fallback_token not in ("", "none") else [],
                    result,
                )

            result = self._parse_prediction(
                prediction, params_schema, already_collected
            )

            # Clear tokens when no question is needed (all params satisfied)
            if result["clarifying_question"] in ("", "none"):
                tokens = []

            if tokens:
                logger.debug(
                    f"ParamExtractionModule: stream tokens collected"
                    f" | event_type=stream_tokens_collected token_count={len(tokens)}"
                )

            # Join all streamed token chunks into the complete assembled question for langfuse logging.
            assembled_question = (
                "".join(tokens) if tokens else result["clarifying_question"]
            )

            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(result["missing_required"]),
                    "clarifying_question": assembled_question,
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "extracted_params_count": len(result["extracted_params"]),
                    "streaming": True,
                    "question_token_count": len(tokens),
                },
            )

            return tokens, result

        except json.JSONDecodeError as e:
            logger.error(
                f"ParamExtractionModule: JSON parse error in stream_forward"
                f" | event_type=stream_json_error"
                f" error_id={generate_error_id()}"
                f" exc_type={type(e).__name__} exc_msg={e!r}"
            )
            fallback = self._safe_defaults(params_schema, already_collected)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(fallback["missing_required"]),
                    "clarifying_question": fallback["clarifying_question"],
                    "error": f"JSON parse error: {e}",
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": True,
                },
            )
            return [], fallback

        except Exception as e:
            logger.error(
                f"ParamExtractionModule: stream_forward failed"
                f" | event_type=stream_extraction_failed"
                f" error_id={generate_error_id()}"
                f" exc_type={type(e).__name__} exc_msg={e!r}"
            )
            fallback = self._safe_defaults(params_schema, already_collected)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_message": user_message,
                    "params_schema_count": len(params_schema),
                },
                output_data={
                    "missing_required_count": len(fallback["missing_required"]),
                    "clarifying_question": fallback["clarifying_question"],
                    "error": str(e),
                },
                metadata={
                    "usage": usage,
                    "num_calls": usage.get("num_calls", 0),
                    "streaming": True,
                },
            )
            return [], fallback

        finally:
            if output_stream is not None:
                try:
                    await output_stream.aclose()
                except Exception as e:
                    logger.debug(
                        f"ParamExtractionModule: stream cleanup error"
                        f" | event_type=stream_cleanup_error"
                        f" exc_type={type(e).__name__} exc_msg={e!r}"
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_param_type(
        self, value: object, param_type: str
    ) -> tuple[bool, object]:
        """
        Validate and coerce a raw extracted value to the expected param type.

        Args:
            value: Raw value from LLM output
            param_type: Expected type from param schema

        Returns:
            (is_valid, coerced_value) — coerced_value equals value when is_valid is False
        """
        if value is None:
            return False, value

        str_value = str(value).strip()

        if param_type == "string":
            return True, str_value

        if param_type == "date":
            try:
                # Accept ISO 8601 date strings; datetime.fromisoformat handles YYYY-MM-DD
                parsed = datetime.fromisoformat(str_value)
                return True, parsed.date().isoformat()
            except (ValueError, TypeError):
                return False, value

        if param_type == "datetime":
            try:
                # Accept ISO 8601 datetime strings with or without timezone suffix.
                # Normalise to UTC zone suffix (Z) if none is present.
                clean = str_value.replace("Z", "+00:00")
                parsed_dt = datetime.fromisoformat(clean)
                # Convert timezone-aware datetimes to UTC before formatting.
                # Naive datetimes are assumed to already be UTC.
                if parsed_dt.tzinfo is not None:
                    parsed_dt = parsed_dt.astimezone(timezone.utc)
                # Re-serialise in the exact format required by the external APIs:
                # YYYY-MM-DDTHH:MM:SSZ (no microseconds, UTC Z suffix)
                return True, parsed_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                return False, value

        if param_type == "integer":
            try:
                # int() already rejects float strings like "3.5" with ValueError
                int_val = int(str_value)
                return True, int_val
            except (ValueError, TypeError):
                return False, value

        if param_type == "number":
            try:
                return True, float(str_value)
            except (ValueError, TypeError):
                return False, value

        if param_type == "boolean":
            lower = str_value.lower()
            if lower in _TRUTHY_STRINGS:
                return True, True
            if lower in _FALSY_STRINGS:
                return True, False
            return False, value

        # Unknown type — accept as string to avoid silent data loss
        logger.warning(
            f"ParamExtractionModule: unknown param type"
            f" | event_type=param_type_unknown param_type={param_type!r}"
        )
        return True, str_value

    def _format_conversation_history(
        self, conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """
        Format the last N conversation turns as plain text for the LLM prompt.

        Args:
            conversation_history: List of message dicts with authorRole and message keys

        Returns:
            Newline-separated string of "role: message" or "(No conversation history)"
        """
        if not conversation_history:
            return "(No conversation history)"

        history_lines: List[str] = []
        for msg in conversation_history[-_MAX_HISTORY_TURNS:]:
            role = msg.get("authorRole", "unknown")
            content = msg.get("message", "")
            if content:
                history_lines.append(f"{role}: {content}")

        return (
            "\n".join(history_lines) if history_lines else "(No conversation history)"
        )

    def _safe_defaults(
        self,
        params_schema: List[Dict[str, Any]],
        already_collected: Dict[str, Any],
    ) -> ParamExtractionResult:
        """
        Return safe default result when the LLM call or JSON parsing fails.

        All required params not already collected are put in missing_required.
        """
        missing_required = [
            p["name"]
            for p in params_schema
            if isinstance(p, dict)
            and p.get("required", False)
            and p["name"] not in already_collected
        ]
        return ParamExtractionResult(
            extracted_params={},
            missing_required=missing_required,
            clarifying_question="none" if not missing_required else "",
        )

    def _parse_prediction(
        self,
        result: dspy.Prediction,
        params_schema: List[Dict[str, Any]],
        already_collected: Dict[str, Any],
    ) -> ParamExtractionResult:
        """Parse a raw DSPy Prediction into a validated ParamExtractionResult.

        Called by both :meth:`forward` (blocking) and :meth:`stream_forward` (streaming)
        so JSON parsing and type-validation logic lives in one place.

        Raises:
            json.JSONDecodeError: If ``extracted_params`` or ``missing_required`` is not
                valid JSON.
            ValueError: If the parsed JSON has the wrong container type.
        """
        # Parse extracted_params
        extracted_raw = json.loads(result.extracted_params)
        if not isinstance(extracted_raw, dict):
            raise ValueError("extracted_params is not a JSON object")

        # Parse missing_required
        missing_raw = json.loads(result.missing_required)
        if not isinstance(missing_raw, list):
            raise ValueError("missing_required is not a JSON array")

        clarifying_question = (result.clarifying_question or "").strip()

        # Validate extracted param types against schema
        schema_map: Dict[str, Dict[str, Any]] = {
            p["name"]: p for p in params_schema if isinstance(p, dict)
        }
        validated_params: Dict[str, Any] = {}
        type_invalid_params: List[str] = []

        for param_name, raw_value in extracted_raw.items():
            schema_entry = schema_map.get(param_name)
            if schema_entry is None:
                # Param not in schema — skip silently
                continue
            param_type = schema_entry.get("type", "string")
            is_valid, coerced = self._validate_param_type(raw_value, param_type)
            if is_valid:
                validated_params[param_name] = coerced
            else:
                logger.warning(
                    f"ParamExtractionModule: param value type mismatch"
                    f" | event_type=param_value_type_mismatch"
                    f" param_name={param_name!r} expected_type={param_type!r}"
                    f" raw_value={raw_value!r}"
                )
                type_invalid_params.append(param_name)

        # Re-derive missing required params after type validation.
        # validated_params (current turn) takes precedence over already_collected
        # so that explicit user corrections override prior values.
        all_collected = {**already_collected, **validated_params}
        missing_required: List[str] = [
            p["name"]
            for p in params_schema
            if isinstance(p, dict)
            and p.get("required", False)
            and p["name"] not in all_collected
        ]

        # Add type-invalid required params back to missing list
        for param_name in type_invalid_params:
            schema_entry = schema_map.get(param_name)
            if (
                schema_entry is not None
                and schema_entry.get("required", False)
                and param_name not in missing_required
            ):
                missing_required.append(param_name)

        # Normalise clarifying_question: override with "none" when nothing is missing
        if not missing_required:
            clarifying_question = "none"
        elif clarifying_question.lower() == "none":
            # LLM incorrectly returned "none" despite missing params — reset to empty
            # string so callers receive a reliable signal that a follow-up is needed.
            logger.warning(
                f"ParamExtractionModule: LLM returned 'none' despite missing params: {missing_required}"
            )
            clarifying_question = ""

        return ParamExtractionResult(
            extracted_params=validated_params,
            missing_required=missing_required,
            clarifying_question=clarifying_question,
        )

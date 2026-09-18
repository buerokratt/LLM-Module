"""Follow-up detection using DSPy — classifies whether a user query is a follow-up to a previous ATC API call."""

import json
import time
from typing import Any, Dict, List, TypedDict

import dspy
from src.loki_logger import LokiLogger
from src.utils.error_utils import generate_error_id

from .param_extractor import strip_format_hints

logger = LokiLogger(service_name="api-tool-calling")

_VALID_FOLLOW_UP_TYPES = {"param_update", "response_question", "new_intent"}


def _validate_updated_params(
    updated_params: Dict[str, Any],
    params_schema: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate and filter updated_params against the schema.

    Removes unexpected keys not present in the schema and logs warnings for them.
    This prevents untrusted LLM-generated parameters from leaking downstream.

    Args:
        updated_params: Parameter dict from LLM (untrusted)
        params_schema: List of parameter schema dicts with name, type, etc.

    Returns:
        Filtered dict containing only schema-valid parameter keys
    """
    # Build lookup: param_name -> param_schema
    schema_lookup = {
        p["name"]: p for p in params_schema if isinstance(p, dict) and "name" in p
    }

    validated = {}
    for key, value in updated_params.items():
        if key not in schema_lookup:
            logger.warning(
                f"FollowUpDetectorModule: unexpected param dropped | event_type=unexpected_param_dropped param_name={key!r}"
            )
        else:
            validated[key] = value

    return validated


class FollowUpDetectionResult(TypedDict):
    """Return contract for FollowUpDetectorModule.forward()."""

    follow_up_type: str
    updated_params: Dict[str, Any]


class FollowUpDetectionSignature(dspy.Signature):
    """Classify whether a user query is a follow-up to a previous API call.

    Determine the relationship between the current user query and the previous query
    that triggered an ATC (API Tool Call) API call.

    CLASSIFICATION RULES:

    param_update — The user is modifying or refining the SAME intent with different parameter values.
        Examples:
        - Previous: "Show public holidays in 2026" → Current: "What about 2025 instead?" → param_update
          (same holiday lookup intent, only the year changed)
        - Previous: "Convert EUR to USD" → Current: "What about GBP to JPY?" → param_update
          (same currency conversion intent, only currencies changed)
        - Previous: "Weather in Tallinn" → Current: "And in Tartu?" → param_update
          (same weather intent, only city changed)

    response_question — The user is asking a question ABOUT the data/results already returned.
        Examples:
        - Previous: "List public holidays in 2026" → Response showed 12 holidays → Current: "Which of those falls on a Monday?" → response_question
          (asking about the already-returned data, no new API call needed)
        - Previous: "Show electricity prices" → Response showed prices → Current: "Why is the price so high in January?" → response_question
          (analysing the returned results, not requesting new data)
        - Previous: "Get train schedule from Tallinn to Tartu" → Response showed times → Current: "How long does the first journey take?" → response_question
          (asking about details within the returned results)

    new_intent — The user is starting a completely different request, unrelated to the previous API call.
        Examples:
        - Previous: "Show public holidays in 2026" → Current: "How do I apply for a driving licence?" → new_intent
          (completely different topic)
        - Previous: "Convert EUR to USD" → Current: "What is the weather in Riga?" → new_intent
          (unrelated request)
        - Previous: "Get train schedules" → Current: "Tell me about Estonian history" → new_intent
          (no connection to the previous API call)

    CRITICAL LANGUAGE RULE:
    - Understand Estonian, Russian, and English queries equally
    - Short follow-ups like "ja 2025?" (Estonian: "and 2025?") or "а в 2025?" (Russian: "and in 2025?") are param_update
    - Language of the query does NOT affect classification — only intent relationship matters

    OUTPUT RULES:
    - follow_up_type MUST be exactly one of: "param_update", "response_question", "new_intent"
    - updated_params MUST be a valid JSON object (can be empty {})
    - For param_update: populate updated_params with the changed parameter values extracted from user_query
    - For response_question or new_intent: return updated_params as empty {}
    """

    user_query: str = dspy.InputField(
        desc="Current user query in Estonian, English, or Russian"
    )
    previous_query: str = dspy.InputField(
        desc="The user query that triggered the previous ATC API call"
    )
    previous_params: str = dspy.InputField(
        desc="JSON object of parameter values used in the previous API call: {param_name: value}"
    )
    params_schema: str = dspy.InputField(
        desc='JSON array of parameter schemas for the previous API: [{"name": str, "type": str, "required": bool, "description": str}]'
    )

    follow_up_type: str = dspy.OutputField(
        desc='Classification result — MUST be exactly one of: "param_update", "response_question", "new_intent"'
    )
    updated_params: str = dspy.OutputField(
        desc="Valid JSON object of updated parameter values extracted from user_query. Non-empty only for param_update. Empty object {} for response_question and new_intent."
    )


class FollowUpDetectorModule(dspy.Module):
    """DSPy Module for follow-up query classification."""

    def __init__(self) -> None:
        """Initialize follow-up detector module with Predict (direct prediction)."""
        super().__init__()
        self.detector = dspy.Predict(FollowUpDetectionSignature)

    def forward(
        self,
        user_query: str,
        previous_query: str,
        previous_params: Dict[str, Any],
        params_schema: List[Dict[str, Any]],
    ) -> FollowUpDetectionResult:
        """
        Classify whether the user query is a follow-up to a previous API call.

        Args:
            user_query: Current user query
            previous_query: The query that triggered the previous ATC API call
            previous_params: Parameter values used in the previous API call
            params_schema: Parameter schemas for the previous API

        Returns:
            FollowUpDetectionResult with follow_up_type and updated_params
        """
        _safe_fallback: FollowUpDetectionResult = {
            "follow_up_type": "new_intent",
            "updated_params": {},
        }

        previous_params_json = json.dumps(previous_params, ensure_ascii=False)
        sanitized_schema = [
            {**p, "description": strip_format_hints(p.get("description", ""))}
            if isinstance(p, dict)
            else p
            for p in params_schema
        ]
        params_schema_json = json.dumps(sanitized_schema, ensure_ascii=False)

        result = None
        _t0 = time.time()
        try:
            result = self.detector(
                user_query=user_query,
                previous_query=previous_query,
                previous_params=previous_params_json,
                params_schema=params_schema_json,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)

            # Parse and validate follow_up_type
            follow_up_type = result.follow_up_type.strip().strip("'\"")
            if follow_up_type not in _VALID_FOLLOW_UP_TYPES:
                logger.warning(
                    f"FollowUpDetectorModule: invalid follow_up_type | event_type=follow_up_type_invalid received={follow_up_type!r}"
                )
                follow_up_type = "new_intent"

            # Parse updated_params JSON only for param_update; force {} for other types
            updated_params: Dict[str, Any] = {}
            if follow_up_type == "param_update":
                raw_updated_params = result.updated_params
                try:
                    # Sanitize: strip whitespace and outer quotes (both single and double)
                    sanitized_params = raw_updated_params.strip().strip("'\"")
                    updated_params = json.loads(sanitized_params)
                    if not isinstance(updated_params, dict):
                        logger.warning(
                            f"FollowUpDetectorModule: updated_params not a dict | event_type=updated_params_not_dict received_type={type(updated_params).__name__}"
                        )
                        updated_params = {}
                except json.JSONDecodeError as exc:
                    _err_id = generate_error_id()
                    logger.error(
                        f"FollowUpDetectorModule: JSON parse error in updated_params | event_type=updated_params_json_error error_id={_err_id} error={exc}"
                    )
                    updated_params = {}

            # Enforce OUTPUT RULES: updated_params must be {} unless follow_up_type is param_update.
            # This prevents unintended params from leaking downstream even if the LLM returns them.
            if follow_up_type != "param_update":
                updated_params = {}
            else:
                # Validate against schema to drop unexpected/injected parameters
                updated_params = _validate_updated_params(updated_params, params_schema)

            logger.debug(
                f"FollowUpDetectorModule: detection complete | event_type=follow_up_detection_complete follow_up_type={follow_up_type} updated_params_count={len(updated_params)} duration_ms={_duration_ms}"
            )

            return FollowUpDetectionResult(
                follow_up_type=follow_up_type,
                updated_params=updated_params,
            )

        except Exception as exc:
            _err_id = generate_error_id()
            logger.error(
                f"FollowUpDetectorModule: detection failed | event_type=follow_up_detection_failed error_id={_err_id} error={exc}"
            )
            return _safe_fallback

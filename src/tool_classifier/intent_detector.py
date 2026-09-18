"""Service intent detection using DSPy."""

import json
from typing import Any, Dict, List, Optional

import dspy
from langfuse import observe
from src.loki_logger import LokiLogger

from src.utils.cost_utils import get_lm_usage_since
from src.utils.observation_utils import update_observation_safe


# Initialize Loki logger
logger = LokiLogger(service_name="intent-detector")


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


class ServiceIntentDetector(dspy.Signature):
    """Detect which service matches user intent and extract entities.

    CRITICAL LANGUAGE RULE:
    - Understand Estonian, Russian, and English queries
    - Extract entities in their original form from the query

    Rules:
    - Match user query against available services
    - Extract required entity values from the query
    - Return valid JSON format strictly
    - If no service matches well (confidence < 0.7), return null for matched_service_id
    - Be conservative - only match when confident
    - Prioritize services whose examples closely match the user query
    """

    user_query: str = dspy.InputField(
        desc="User's question/request in Estonian, Russian, or English"
    )
    available_services: str = dspy.InputField(
        desc="JSON string of available services with id, name, description, entities, examples"
    )
    conversation_context: str = dspy.InputField(
        desc="Recent conversation history for context (optional, may be empty)"
    )

    intent_result: str = dspy.OutputField(
        desc='Valid JSON only: {"matched_service_id": "id_string" or null, "confidence": 0.0-1.0, "entities": {}, "reasoning": "brief explanation"}'
    )


class IntentDetectionModule(dspy.Module):
    """DSPy Module for service intent detection."""

    def __init__(self) -> None:
        """Initialize intent detection module with Predict (direct prediction)."""
        super().__init__()
        self.detector = dspy.Predict(ServiceIntentDetector)

    @observe(name="service_intent_detection_llm", as_type="generation")
    def forward(
        self,
        user_query: str,
        services: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Detect service intent using LLM via DSPy.

        Args:
            user_query: User's query
            services: List of service dicts with serviceId, name, description, entities, examples
            conversation_history: Recent messages (optional)

        Returns:
            Parsed intent result dict with matched_service_id, confidence, entities, reasoning
        """
        # Format services for prompt (keep it concise)
        services_formatted = []
        for s in services:
            service_entry = {
                "service_id": s.get("serviceId", s.get("service_id")),
                "name": s.get("name", "Unknown"),
                "description": s.get("description", ""),
                "required_entities": s.get("entities", []),
                "examples": s.get("examples", [])[:3],  # Top 3 examples
            }
            services_formatted.append(service_entry)

        services_json = json.dumps(services_formatted, ensure_ascii=False, indent=2)

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length for intent detection: {e}")

        # Format conversation history
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-3:]:  # Last 3 turns
                role = msg.get("authorRole", "unknown")
                content = msg.get("message", "")
                if content:
                    history_lines.append(f"{role}: {content}")
            history_text = "\n".join(history_lines) if history_lines else "(Empty)"
        else:
            history_text = "(No conversation history)"

        # Call DSPy detector with ChainOfThought
        result = None
        try:
            result = self.detector(
                user_query=user_query,
                available_services=services_json,
                conversation_context=history_text,
            )

            # Parse JSON response
            intent_data = json.loads(result.intent_result)

            # Validate structure
            if not isinstance(intent_data, dict):
                raise ValueError("Intent result is not a dictionary")

            # Ensure required keys exist
            intent_data.setdefault("matched_service_id", None)
            intent_data.setdefault("confidence", 0.0)
            intent_data.setdefault("entities", {})
            intent_data.setdefault("reasoning", "")

            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "services_count": len(services),
                },
                output_data={
                    "matched_service_id": intent_data.get("matched_service_id"),
                    "confidence": intent_data.get("confidence", 0.0),
                    "entities": intent_data.get("entities", {}),
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                },
            )

            return intent_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent JSON: {e}")
            if result:
                logger.error(f"Raw response: {result.intent_result}")
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "services_count": len(services),
                },
                output_data={
                    "matched_service_id": None,
                    "confidence": 0.0,
                    "error": f"JSON parse error: {e}",
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                },
            )
            return {
                "matched_service_id": None,
                "confidence": 0.0,
                "entities": {},
                "reasoning": f"JSON parse error: {e}",
            }
        except Exception as e:
            logger.error(f"Intent detection forward failed: {e}", exc_info=True)
            usage = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "user_query": user_query,
                    "services_count": len(services),
                },
                output_data={
                    "matched_service_id": None,
                    "confidence": 0.0,
                    "error": f"Detection error: {e}",
                },
                metadata={
                    "model": _get_current_model_name(),
                    "usage": usage,
                },
            )
            return {
                "matched_service_id": None,
                "confidence": 0.0,
                "entities": {},
                "reasoning": f"Detection error: {e}",
            }

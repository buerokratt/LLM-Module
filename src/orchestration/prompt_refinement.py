"""DSPy prompt refinement (Group G).

Extracted from LLMOrchestrationService. Turns the raw user message plus
conversation context into a ``PromptRefinerOutput`` (original question + N
refined variations) and reports usage/cost to Langfuse.

Takes the Langfuse client explicitly rather than a service back-reference —
this is a stateless step with no other dependency on the façade.
"""

import json
from typing import Any, Dict, List, Optional

from langfuse import observe

from src.loki_logger import LokiLogger
from models.request_models import ConversationItem, PromptRefinerOutput
from llm_orchestrator_config.llm_manager import LLMManager
from prompt_refine_manager.prompt_refiner import PromptRefinerAgent
from src.utils.error_utils import generate_error_id, log_error_with_context

logger = LokiLogger(service_name="llm-orchestration-service")

_DEFAULT_USAGE: Dict[str, Any] = {
    "total_cost": 0.0,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "num_calls": 0,
}


def _build_dspy_history(
    conversation_history: List[ConversationItem],
    conversation_summary: Optional[str],
) -> List[Dict[str, str]]:
    """Convert conversation history to DSPy chat format.

    A pre-computed summary of earlier (evicted) rounds is prepended as a
    ``system`` turn so the refiner gets that context without spending an
    extra LLM call to re-summarise.
    """
    history: List[Dict[str, str]] = []

    if conversation_summary:
        history.append(
            {
                "role": "system",
                "content": f"Summary of earlier conversation: {conversation_summary}",
            }
        )

    for item in conversation_history:
        role = "assistant" if item.authorRole == "bot" else item.authorRole
        history.append({"role": role, "content": item.message})

    return history


@observe(name="refine_user_prompt", as_type="generation")
def refine_user_prompt(
    llm_manager: LLMManager,
    original_message: str,
    conversation_history: List[ConversationItem],
    conversation_summary: Optional[str] = None,
    langfuse_client: Optional[Any] = None,  # noqa: ANN401 — Langfuse client
) -> tuple[PromptRefinerOutput, Dict[str, Any]]:
    """Refine the user prompt and return the output plus usage info.

    Args:
        llm_manager: The LLM manager instance to use
        original_message: The original user message to refine
        conversation_history: Previous conversation context
        conversation_summary: Optional summary of earlier conversation rounds
            that were evicted from Redis.
        langfuse_client: Optional Langfuse client for generation tracking.

    Returns:
        Tuple of (PromptRefinerOutput, usage_dict).

    Raises:
        ValueError: When the refiner output fails schema validation.
        RuntimeError: For any other prompt refinement failure.
    """
    logger.info("Starting prompt refinement process")

    try:
        history = _build_dspy_history(conversation_history, conversation_summary)

        # Create prompt refiner using the same LLM manager instance
        refiner = PromptRefinerAgent(llm_manager=llm_manager)

        refinement_result = refiner.forward_structured(
            history=history, question=original_message
        )

        usage_info = refinement_result.get("usage", dict(_DEFAULT_USAGE))

        # Validate the output schema using Pydantic
        try:
            validated_output = PromptRefinerOutput(
                original_question=refinement_result["original_question"],
                refined_questions=refinement_result["refined_questions"],
            )
        except Exception as validation_error:
            logger.error(
                f"Prompt refinement output validation failed: {str(validation_error)}"
            )
            logger.error(f"Invalid refinement result: {refinement_result}")
            raise ValueError(
                f"Prompt refinement validation failed: {str(validation_error)}"
            ) from validation_error

        if langfuse_client:
            refinement_applied = (
                original_message.strip() != validated_output.original_question.strip()
            )
            langfuse_client.update_current_generation(
                model=llm_manager.get_provider_info().get("model", "unknown"),
                input=original_message,
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
                    "num_refined_questions": len(validated_output.refined_questions),
                    "refinement_applied": refinement_applied,
                    "conversation_history_length": len(history),
                },  # type: ignore
            )

        logger.info(
            f"Prompt refinement output: "
            f"{json.dumps(validated_output.model_dump(), indent=2)}"
        )
        logger.info("Prompt refinement completed successfully")
        return validated_output, usage_info

    except ValueError:
        raise
    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(
            logger,
            error_id,
            "prompt_refinement",
            None,
            e,
            {"message_preview": original_message[:100]},
        )
        if langfuse_client:
            langfuse_client.update_current_generation(
                metadata={
                    "error_id": error_id,
                    "error_type": type(e).__name__,
                    "refinement_failed": True,
                }
            )
        raise RuntimeError(f"Prompt refinement process failed: {str(e)}") from e

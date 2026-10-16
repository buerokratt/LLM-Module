"""Cost calculation utilities for LLM usage tracking."""

import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple
from src.loki_logger import LokiLogger
import dspy

# Initialize Loki logger for cost tracking
logger = LokiLogger(service_name="cost-utils")


class RequestUsageTracker(Protocol):
    def get_total_tokens(self) -> Dict[str, Dict[str, Any]]: ...


def _to_float(value: str | int | float | bytes | bytearray | None) -> float:
    """Best-effort float conversion for cost values."""
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@contextmanager
def track_request_usage() -> Iterator[RequestUsageTracker]:
    """Track DSPy usage for one request and tolerate cross-context stream close."""
    usage_context = dspy.track_usage()
    tracker = usage_context.__enter__()

    try:
        yield tracker
    except BaseException as request_error:
        try:
            suppress_error = usage_context.__exit__(*sys.exc_info())
        except ValueError as close_error:
            if "different Context" not in str(close_error):
                raise
            logger.debug("DSPy usage context closed from a different context")
            raise request_error.with_traceback(
                request_error.__traceback__
            ) from close_error
        if not suppress_error:
            raise
    else:
        try:
            usage_context.__exit__(None, None, None)
        except ValueError as close_error:
            if "different Context" not in str(close_error):
                raise
            logger.debug("DSPy usage context closed from a different context")


def _token_count(value: int | float | str | bytes | bytearray | None) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _price_model_usage(model_name: str, usage: Dict[str, Any]) -> float:
    prompt_tokens = _token_count(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion_tokens = _token_count(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    if prompt_tokens == 0 and completion_tokens == 0:
        return 0.0

    try:
        from litellm.cost_calculator import cost_per_token

        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return _to_float(input_cost) + _to_float(output_cost)
    except Exception as e:
        logger.debug(f"Cost lookup failed for model '{model_name}': {e}")
        return 0.0


def _is_configured_deployment(model_name: str, deployment_name: str) -> bool:
    return model_name == deployment_name or model_name.endswith(f"/{deployment_name}")


def get_request_usage_summary(
    usage_tracker: RequestUsageTracker, model_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Summarize request tracker usage and price each model independently."""
    summary: Dict[str, Any] = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_calls": 0,
        "models": {},
    }

    try:
        usage_by_model = usage_tracker.get_total_tokens()
        usage_entries = getattr(usage_tracker, "usage_data", {})
        model_info = model_info or {}
        deployment_name = model_info.get("deployment_name")
        fallback_model = model_info.get("model")

        for model_name, usage in usage_by_model.items():
            if not isinstance(usage, dict):
                continue

            prompt_tokens = _token_count(
                usage.get("prompt_tokens", usage.get("input_tokens"))
            )
            completion_tokens = _token_count(
                usage.get("completion_tokens", usage.get("output_tokens"))
            )
            total_tokens = _token_count(usage.get("total_tokens"))
            if total_tokens == 0:
                total_tokens = prompt_tokens + completion_tokens

            calls = len(usage_entries.get(model_name, []))
            if calls == 0 and (prompt_tokens or completion_tokens):
                calls = 1

            total_cost = _price_model_usage(model_name, usage)
            if (
                total_cost == 0.0
                and isinstance(deployment_name, str)
                and deployment_name
                and isinstance(fallback_model, str)
                and fallback_model
                and _is_configured_deployment(model_name, deployment_name)
            ):
                total_cost = _price_model_usage(fallback_model, usage)

            if total_tokens > 0 and total_cost == 0.0:
                logger.warning(
                    f"Unable to price {total_tokens} tokens for model '{model_name}'"
                )

            summary["models"][model_name] = {
                "total_cost": total_cost,
                "total_prompt_tokens": prompt_tokens,
                "total_completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "num_calls": calls,
            }
            summary["total_cost"] += total_cost
            summary["total_prompt_tokens"] += prompt_tokens
            summary["total_completion_tokens"] += completion_tokens
            summary["total_tokens"] += total_tokens
            summary["total_calls"] += calls
    except Exception as e:
        logger.error(f"Error summarizing request usage: {e}")

    return summary


def _extract_cost_with_fallback(
    item: Dict[str, Any], usage: Dict[str, Any]
) -> Tuple[float, str]:
    """Extract cost from history entry with streaming-safe fallbacks."""
    # Primary source used by DSPy for non-streaming calls.
    item_cost = _to_float(item.get("cost"))
    if item_cost > 0.0:
        return item_cost, "item.cost"

    # Some providers put cost directly into usage for streaming responses.
    usage_cost = _to_float(usage.get("cost")) if isinstance(usage, dict) else 0.0
    if usage_cost > 0.0:
        return usage_cost, "usage.cost"

    # Final fallback: estimate from model + tokens when available.
    prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
    completion_tokens = (
        int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
    )
    model_name = item.get("model")

    if not model_name or (prompt_tokens == 0 and completion_tokens == 0):
        return 0.0, "missing_model_or_tokens"

    try:
        from litellm.cost_calculator import cost_per_token

        input_cost, output_cost = cost_per_token(
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        estimated_cost = input_cost + output_cost
        return _to_float(estimated_cost), "litellm.cost_per_token"
    except Exception as e:
        logger.debug(f"Cost fallback failed for model '{model_name}': {e}")
        return 0.0, "fallback_error"


def extract_cost_from_lm_history(lm_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract cost and usage information from LM history.

    Args:
        lm_history: List of LM history items from dspy.LM.history

    Returns:
        Dictionary containing:
            - total_cost: Total cost in dollars
            - total_prompt_tokens: Total input tokens
            - total_completion_tokens: Total output tokens
            - total_tokens: Total tokens used
            - num_calls: Number of LM calls
    """
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    num_calls = 0

    try:
        for item in lm_history:
            num_calls += 1

            # Extract usage information
            usage = item.get("usage", {})

            # Extract cost with fallback path for streaming entries.
            entry_cost, _ = _extract_cost_with_fallback(item, usage)
            total_cost += entry_cost

            if usage:
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                total_tokens += usage.get("total_tokens", 0)

    except Exception as e:
        logger.error(f"Error extracting cost from LM history: {str(e)}")

    return {
        "total_cost": round(total_cost, 6),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "num_calls": num_calls,
    }


def calculate_total_costs(component_costs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate total costs across all components.

    Args:
        component_costs: Dictionary mapping component names to their cost dictionaries

    Returns:
        Dictionary containing aggregate totals
    """
    total = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_calls": 0,
    }

    try:
        for costs in component_costs.values():
            total["total_cost"] += costs.get("total_cost", 0.0)
            total["total_prompt_tokens"] += costs.get("total_prompt_tokens", 0)
            total["total_completion_tokens"] += costs.get("total_completion_tokens", 0)
            total["total_tokens"] += costs.get("total_tokens", 0)
            total["total_calls"] += costs.get("num_calls", 0)

        total["total_cost"] = round(total["total_cost"], 6)

    except Exception as e:
        logger.error(f"Error calculating total costs: {str(e)}")

    return total


def get_lm_usage_since(history_length_before: int) -> Dict[str, Any]:
    """
    Extract usage information from LM history since a specific point.

    Args:
        history_length_before: The history length to measure from

    Returns:
        Dictionary containing usage statistics
    """
    usage_info = get_default_usage_dict()

    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "history"):
            new_history = lm.history[history_length_before:]
            usage_info = extract_cost_from_lm_history(new_history)
    except Exception as e:
        logger.warning(f"Failed to extract usage info: {str(e)}")

    return usage_info


# Every guardrails self-check prompt opens with this phrase (see the
# self_check_input / self_check_output tasks in src/guardrails/rails_config.yaml).
# Matching on it lets us bill guardrail traffic separately from generation, which
# otherwise hides inside the same LM history window.
_GUARDRAIL_PROMPT_MARKER = "you are tasked with evaluating if a"


def _history_entry_text(item: Dict[str, Any]) -> str:
    """Best-effort extraction of the prompt text from an LM history entry."""
    prompt = item.get("prompt")
    if isinstance(prompt, str):
        return prompt

    messages = item.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts)

    return ""


def _is_guardrail_entry(item: Dict[str, Any]) -> bool:
    """Whether an LM history entry is a guardrails self-check call."""
    return _GUARDRAIL_PROMPT_MARKER in _history_entry_text(item).lower()


def get_lm_usage_since_split(
    history_length_before: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract usage since a point, split into generation and guardrails buckets.

    During streaming, output-rail validation calls are interleaved with the
    generation call in the same LM history window. Reporting them as one figure
    makes runaway guardrail spend invisible, so we attribute each entry by its
    prompt.

    Args:
        history_length_before: The history length to measure from

    Returns:
        ``(generation_usage, guardrails_usage)``
    """
    generation_usage = get_default_usage_dict()
    guardrails_usage = get_default_usage_dict()

    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "history"):
            new_history = lm.history[history_length_before:]
            guardrail_entries = [i for i in new_history if _is_guardrail_entry(i)]
            generation_entries = [i for i in new_history if not _is_guardrail_entry(i)]
            generation_usage = extract_cost_from_lm_history(generation_entries)
            guardrails_usage = extract_cost_from_lm_history(guardrail_entries)
    except Exception as e:
        logger.warning(f"Failed to split usage info: {str(e)}")

    return generation_usage, guardrails_usage


def get_default_usage_dict() -> Dict[str, Any]:
    """
    Return a default usage dictionary with zero values.

    Returns:
        Dictionary with default usage values
    """
    return {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "num_calls": 0,
    }

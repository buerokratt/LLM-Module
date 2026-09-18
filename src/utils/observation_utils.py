"""Langfuse observation utilities with graceful degradation.

Two canonical tracing patterns are used throughout this codebase. Always use
the helpers defined here rather than calling ``get_client()`` directly.

Pattern A — Non-streaming (single LLM call wrapped with ``@observe``):
    Use ``@observe(name="...", as_type="generation")`` on the method, then call
    ``update_observation_safe()`` at the end to attach input/output/cost data.

    NOTE: Use ``as_type="generation"`` (not ``"chain"``) for any method that makes
    exactly one LLM call. ``"chain"`` extends the async observation context longer
    than a single LLM call, which can cause DSPy history entries from *this* step
    to bleed into the baseline capture of the *next* step, inflating cost reports.
    Reserve ``"chain"`` for orchestration methods that call multiple
    ``@observe``-decorated children (Langfuse auto-aggregates their costs).

Pattern B — Streaming (``async def`` that ``yield``s tokens):
    Do NOT put ``@observe`` on async generators — the decorator interferes with
    async iteration. Use ``safe_observation_context()`` as a context manager
    instead, then call ``generation.update()`` once after streaming finishes.

Pattern C — Orchestration calling multiple observed children:
    Use ``@observe(name="...", as_type="chain")``. No manual update needed;
    Langfuse auto-aggregates child generations.

Pattern D — Pipeline step with no LLM call:
    Use ``@observe(name="...", as_type="span")``. Optionally call
    ``get_client().update_current_span(metadata={...})`` for extra context.
"""

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Dict, Optional

from langfuse import get_client
from loguru import logger


def safe_observation_context(**kwargs: Any) -> AbstractContextManager[Any]:
    """Return a Langfuse generation/span context manager with a no-op fallback.

    Use this for **streaming** paths (Pattern B) where ``@observe`` cannot be
    used on an async generator.  Wraps
    ``get_client().start_as_current_observation()``; returns ``nullcontext()``
    when Langfuse is unavailable and streaming continues uninterrupted.

    The object bound via ``as`` will be ``None`` on fallback — all call sites
    already guard ``.update()`` with try/except.

    Args:
        **kwargs: Forwarded to ``start_as_current_observation()``
                  (e.g. ``as_type``, ``name``, ``input``).
    """
    try:
        return get_client().start_as_current_observation(**kwargs)
    except Exception as e:
        logger.debug(f"Langfuse observation unavailable, using no-op context: {e}")
        return nullcontext()


def update_observation_safe(
    *,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach input/output/cost data to the active Langfuse span or generation.

    Use this for **non-streaming** paths (Pattern A) inside ``@observe``-decorated
    methods.  Silently no-ops on any failure so tracing never blocks a response.

    Dispatches to ``update_current_generation()`` when ``metadata["usage"]`` is a
    dict (i.e. an LLM call was made), otherwise to ``update_current_span()``.
    The method must be decorated with ``@observe(as_type="generation")`` for
    ``update_current_generation()`` to have an active target.

    Args:
        input_data: Dict to set as the observation ``input``.
        output_data: Dict to set as the observation ``output``.
        metadata: Dict that may contain ``"model"`` (str) and ``"usage"`` (dict
                  with ``total_prompt_tokens``, ``total_completion_tokens``,
                  ``total_tokens``, ``total_cost``).  All other keys are forwarded
                  as Langfuse ``metadata``.
    """
    try:
        payload: Dict[str, Any] = {}
        if input_data is not None:
            payload["input"] = input_data
        if output_data is not None:
            payload["output"] = output_data

        model_name = None
        usage = None
        metadata_payload: Dict[str, Any] = {}
        if metadata is not None:
            metadata_payload = dict(metadata)
            model_name = metadata_payload.pop("model", None)
            usage = metadata_payload.pop("usage", None)

        if metadata_payload:
            payload["metadata"] = metadata_payload

        if isinstance(usage, dict):
            get_client().update_current_generation(
                model=model_name
                if isinstance(model_name, str) and model_name
                else None,
                usage_details={
                    "input": usage.get("total_prompt_tokens", 0),
                    "output": usage.get("total_completion_tokens", 0),
                    "total": usage.get("total_tokens", 0),
                },
                cost_details={
                    "total": usage.get("total_cost", 0.0),
                },
                **payload,
            )
        else:
            get_client().update_current_span(**payload)
    except Exception as e:
        logger.debug(f"Langfuse observation update skipped: {e}")

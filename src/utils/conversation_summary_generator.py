"""Factory for creating incremental conversation summarizer callables."""

from __future__ import annotations

import json
from typing import Any, Protocol

import dspy
from src.loki_logger import LokiLogger

from src.models.conversation_history_models import ConversationRound
from tool_classifier.context_analyzer import IncrementalSummarySignature

logger = LokiLogger(service_name="context-workflow")


class SummarizerCallable(Protocol):
    """Protocol for incremental summary callables injected into the history store."""

    async def __call__(
        self,
        existing_summary: str | None,
        evicted_rounds: list[ConversationRound],
    ) -> str:
        """Merge *evicted_rounds* into *existing_summary* and return the result.

        Args:
            existing_summary: The current summary string, or None / empty string
                when no summary exists yet.
            evicted_rounds: The rounds that were just trimmed from the active
                history window.

        Returns:
            The updated summary string, or an empty string on failure.
        """
        ...


def _format_rounds_as_json(rounds: list[ConversationRound]) -> str:
    """Serialise *rounds* to a compact JSON string suitable for LLM prompts."""
    return json.dumps(
        [r.model_dump() for r in rounds],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def create_incremental_summarizer(llm_manager: Any) -> SummarizerCallable:  # noqa: ANN401
    """Return an async callable that merges evicted rounds into a running summary.

    The returned callable is safe to use as a fire-and-forget background task.
    Any exception is caught and logged; the caller always receives either a
    non-empty updated summary or an empty string (graceful degradation).

    Args:
        llm_manager: The application-wide LLM manager instance.

    Returns:
        An async callable matching the ``SummarizerCallable`` protocol.
    """
    _module: dspy.Module | None = None

    async def _summarize(
        existing_summary: str | None,
        evicted_rounds: list[ConversationRound],
    ) -> str:
        nonlocal _module
        try:
            rounds_json = _format_rounds_as_json(evicted_rounds)
            current_summary = existing_summary or ""

            llm_manager.ensure_global_config()
            with llm_manager.use_task_local():
                if _module is None:
                    _module = dspy.ChainOfThought(IncrementalSummarySignature)
                response = _module(
                    existing_summary=current_summary,
                    new_rounds=rounds_json,
                )

            updated: str = response.updated_summary
            if not updated or not updated.strip():
                logger.warning(
                    "[IncrementalSummarizer] LLM returned empty summary; "
                    "keeping existing summary unchanged."
                )
                return ""
            return updated.strip()

        except Exception as exc:
            logger.error(
                f"[IncrementalSummarizer] Failed to generate incremental summary: {exc}"
            )
            return ""

    return _summarize  # type: ignore[return-value]

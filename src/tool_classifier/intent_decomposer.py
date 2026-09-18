"""Intent Decomposer — detects multi-intent queries and produces focused sub-queries."""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, cast

import dspy
from src.loki_logger import LokiLogger

from src.utils.cost_utils import get_lm_usage_since
from src.utils.observation_utils import safe_observation_context
from tool_classifier.constants import MULTI_API_MAX_ENDPOINTS

logger = LokiLogger(service_name="api-tool-calling")


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


@dataclass
class DecompositionResult:
    """Result returned by IntentDecomposerModule.decompose().

    Attributes:
        mode: ``"single"`` if the query has one intent, ``"parallel"`` if
            it contains multiple independent intents each requiring a
            separate API call.
        sub_queries: Focused sub-queries for each detected intent.
            Empty list when ``mode="single"``.
            Length is capped at ``MULTI_API_MAX_ENDPOINTS``.
    """

    mode: str
    sub_queries: list[str] = field(default_factory=list)

    def set_lm_usage(self, *args: object, **kwargs: object) -> None:
        """No-op stub for DSPy's internal Langfuse callback compatibility."""


class IntentDecompositionSignature(dspy.Signature):
    """Determine whether a user query contains a single intent or multiple
    independent intents that each require a separate API call.

    Rules:
    - Return mode="single" if the query has one clear intent
    - Return mode="single" if you are uncertain — never force parallel
    - Return mode="parallel" ONLY if the query CLEARLY asks for 2 or 3
      completely distinct, independently answerable things
    - Each sub_query must be self-contained — answerable without the others
    - Each sub_query must be phrased using domain-specific, descriptive
      language that matches how the service would be described (e.g.,
      "address lookup and location search" not "find an address for me",
      "vehicle tax calculation" not "calculate my tax"). This is critical
      because sub_queries are used for semantic vector search.
    - sub_queries must be a valid JSON list of strings, e.g. ["q1", "q2"]
    - When mode="single", sub_queries MUST be an empty JSON list: []
    - Understands Estonian, English, and Russian queries
    - Examples of multi-intent: "holidays in Estonia AND weather in Tallinn"
    - Examples of single-intent: "renew my ID card", "book an appointment"
    """

    user_query: str = dspy.InputField(
        desc="User's full natural language query in Estonian, English, or Russian"
    )
    mode: str = dspy.OutputField(desc='Either "single" or "parallel"')
    sub_queries: str = dspy.OutputField(
        desc=(
            'JSON list of focused sub-queries when mode="parallel", '
            'or empty list [] when mode="single". '
            "Each sub-query must be self-contained, independently answerable, "
            "and phrased with domain-specific descriptive language suitable for "
            "semantic search (e.g. 'address lookup and location search' rather "
            "than 'find an address for me')."
        )
    )


class IntentDecomposerModule(dspy.Module):
    """DSPy module that detects whether a user query contains multiple independent
    intents and decomposes it into focused sub-queries for parallel API search.

    This module is invoked only when the gate search returns a medium-confidence
    result (cosine score in the ambiguous band), indicating the query embedding
    may be diluted by multiple intents.

    The ``decompose()`` coroutine wraps the synchronous DSPy forward call in a
    thread pool so it does not block the asyncio event loop.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(IntentDecompositionSignature)

    def forward(
        self,
        user_query: str,
    ) -> DecompositionResult:
        """Run intent decomposition synchronously (called from a thread pool).

        Args:
            user_query: The user's full natural language query.

        Returns:
            DecompositionResult with mode and sub_queries.
            Returns mode="single" with empty sub_queries on any LLM or
            parse failure — conservative fallback to the existing single path.
        """
        try:
            prediction = self.predictor(user_query=user_query)
            raw_mode = prediction.mode.strip().lower()
            raw_sub_queries = prediction.sub_queries.strip()

            if raw_mode not in ("single", "parallel"):
                logger.warning(
                    f"IntentDecomposer: unexpected mode={raw_mode!r} — "
                    f"falling back to single"
                )
                return DecompositionResult(mode="single")

            if raw_mode == "single":
                return DecompositionResult(mode="single")

            # mode = "parallel" — parse and validate sub_queries JSON
            sub_queries = _parse_sub_queries(raw_sub_queries)
            if len(sub_queries) < 2:
                logger.warning(
                    f"IntentDecomposer: mode=parallel but only "
                    f"{len(sub_queries)} sub-query parsed — falling back to single"
                )
                return DecompositionResult(mode="single")

            # Cap at MULTI_API_MAX_ENDPOINTS
            if len(sub_queries) > MULTI_API_MAX_ENDPOINTS:
                logger.info(
                    f"IntentDecomposer: {len(sub_queries)} sub-queries exceed cap "
                    f"({MULTI_API_MAX_ENDPOINTS}) — truncating"
                )
                sub_queries = sub_queries[:MULTI_API_MAX_ENDPOINTS]

            logger.info(
                f"IntentDecomposer: mode=parallel, "
                f"{len(sub_queries)} sub-queries: {sub_queries}"
            )
            return DecompositionResult(mode="parallel", sub_queries=sub_queries)

        except Exception as exc:
            logger.error(
                f"IntentDecomposer: decomposition failed: {exc} — "
                f"falling back to single",
                exc_info=True,
            )
            return DecompositionResult(mode="single")

    async def decompose(self, user_query: str) -> DecompositionResult:
        """Async wrapper — runs forward() in a thread pool.

        Keeps the asyncio event loop unblocked while the synchronous DSPy
        LLM call executes.

        Args:
            user_query: The user's full natural language query.

        Returns:
            DecompositionResult with mode and sub_queries.
        """
        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(
                f"Failed to get LM history length for intent decomposition: {e}"
            )

        with safe_observation_context(
            name="intent_decomposition_llm",
            as_type="generation",
            input={"user_query": user_query},
        ) as generation:
            result = cast(
                DecompositionResult, await asyncio.to_thread(self, user_query)
            )

            # Update Langfuse observation with output and usage
            try:
                if generation is not None:
                    usage = get_lm_usage_since(history_length_before)
                    generation.update(
                        model=_get_current_model_name(),
                        output={"mode": result.mode, "sub_queries": result.sub_queries},
                        usage_details={
                            "input": usage.get("total_prompt_tokens", 0),
                            "output": usage.get("total_completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        },
                        cost_details={"total": usage.get("total_cost", 0.0)},
                    )
            except Exception as e:
                logger.debug(f"Langfuse generation update skipped: {e}")

        return result


def _parse_sub_queries(raw: str) -> list[str]:
    """Parse a JSON list of sub-query strings from LLM output.

    Handles common LLM output variations: markdown code fences, extra
    whitespace, and non-list JSON values. Returns an empty list on any
    parse failure.

    Args:
        raw: Raw string output from the LLM (expected to be a JSON list).

    Returns:
        List of non-empty sub-query strings, or [] on parse error.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        # Drop opening fence (and optional language tag) and closing fence
        inner = [line for line in lines[1:] if line.strip() != "```"]
        cleaned = "\n".join(inner).strip()

    try:
        parsed: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"IntentDecomposer: could not parse sub_queries JSON: {raw!r}")
        return []

    if not isinstance(parsed, list):
        logger.warning(
            f"IntentDecomposer: sub_queries is not a list: {type(parsed).__name__}"
        )
        return []

    return [str(item).strip() for item in parsed if str(item).strip()]

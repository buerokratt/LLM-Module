"""Detect a bare "why?" follow-up to a guardrail-violation message.

Blocked exchanges are intentionally never persisted to Redis conversation
history (see ``_HISTORY_EXCLUDED_MESSAGES`` in ``llm_orchestration_service.py``),
so the pipeline itself has no memory of a prior block. Instead of adding
server-side state, this checks the client-resent ``conversationHistory`` on
the request — the same payload already used elsewhere as the history fallback
(see ``conversation_history_helpers.get_conversation_history``) — so if the
client's last displayed bot message was a violation string, a bare "why?"
reply can be answered with that same message.
"""

from typing import NamedTuple, Optional

from models.request_models import OrchestrationRequest
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
)

# Localized violation strings, kept separate so a repeated message can be
# reported back under the same guard type (input vs. output) as the
# original block.
_INPUT_VIOLATION_MESSAGES: frozenset[str] = frozenset(
    INPUT_GUARDRAIL_VIOLATION_MESSAGES.values()
)
_OUTPUT_VIOLATION_MESSAGES: frozenset[str] = frozenset(
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES.values()
)

# Exact-match triggers (after lowercasing and stripping trailing punctuation)
# for a bare "why?" style follow-up, in the three supported languages.
_WHY_FOLLOWUP_TRIGGERS: frozenset[str] = frozenset(
    {
        "why",
        "why not",
        "miks",
        "miks mitte",
        "почему",
        "почему нет",
    }
)


class RepeatedViolation(NamedTuple):
    """A prior violation message and which guard type produced it."""

    message: str
    is_input_violation: bool


def get_repeated_violation_message(
    request: OrchestrationRequest,
) -> Optional[RepeatedViolation]:
    """Return the prior violation if this turn is a bare "why?" reply to it.

    Returns None when the current message isn't a "why?"-style follow-up, or
    when the last item in ``request.conversationHistory`` wasn't a guardrail
    violation message.
    """
    normalized = request.message.strip().lower().rstrip("?!.,")
    if normalized not in _WHY_FOLLOWUP_TRIGGERS:
        return None
    if not request.conversationHistory:
        return None
    last_item = request.conversationHistory[-1]
    if last_item.authorRole != "bot":
        return None
    if last_item.message in _INPUT_VIOLATION_MESSAGES:
        return RepeatedViolation(last_item.message, is_input_violation=True)
    if last_item.message in _OUTPUT_VIOLATION_MESSAGES:
        return RepeatedViolation(last_item.message, is_input_violation=False)
    return None

"""Pydantic models for conversation history state."""

import time
from typing import Optional

from pydantic import BaseModel, Field


class ConversationRound(BaseModel):
    """A single user+bot exchange in a conversation.

    Stored as part of ``ConversationHistoryState`` in Redis, keyed by chat_id.
    """

    user_message: str = Field(..., description="The user's message text for this round")
    bot_message: str = Field(..., description="The bot's response text for this round")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of when the round was recorded",
    )


class ConversationHistoryState(BaseModel):
    """Persisted conversation history for a chat session.

    Keyed by chat_id in Redis with a sliding 30-minute TTL.
    Retains up to the most recent 10 rounds; older rounds are trimmed.
    An optional summary field holds a condensed representation of rounds
    that have been evicted. When a summarizer is injected into the store,
    it is automatically generated and persisted as rounds are evicted.
    If no summarizer is provided, the summary must be managed by the caller.
    """

    chat_id: str = Field(..., description="Unique conversation identifier")
    rounds: list[ConversationRound] = Field(
        default_factory=list,
        description="Ordered list of conversation rounds (newest last), capped at 10",
    )
    summary: Optional[str] = Field(
        default=None,
        description=(
            "Optional condensed summary of earlier conversation turns that have been "
            "evicted from the rounds list. Generated and stored by the caller."
        ),
    )

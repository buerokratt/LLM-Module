"""Pydantic models for API tool session state."""

from typing import Any, Literal
from pydantic import BaseModel, Field
import time


class LastCallContext(BaseModel):
    """Stores the result of the most recent successful ATC API call for a chat session.

    Used by the L2 cache tier to enable follow-up query handling:
    param inheritance, response questions, and partial param updates.

    For multi-intent sessions, ``ATCCacheStore.set_l2()`` stores a
    ``list[LastCallContext]`` keyed by ``atc:last:{chat_id}``. Single-intent
    calls store a single-element list for consistency. The follow-up detector
    always searches this list by ``api_name``.
    """

    api_name: str = Field(..., description="Endpoint name that was called")
    endpoint: dict[str, Any] = Field(
        ..., description="Full endpoint definition as stored in Qdrant payload"
    )
    collected_params: dict[str, Any] = Field(
        ..., description="Parameter values that were passed to the API call"
    )
    raw_response: Any = Field(
        ...,
        description="Parsed API JSON response; Any to accept either list or dict",
    )
    original_query: str = Field(
        ..., description="User's first-turn query that triggered this API call"
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp of when the call was made (for TTL/staleness checks)",
    )


class EndpointSessionState(BaseModel):
    """Per-endpoint parameter collection state for parallel API tool execution.

    One instance per endpoint in a parallel-mode session. Tracks what has been
    collected for that specific endpoint and whether all required params are ready
    (``completed=True``). Note: ``completed`` signals param-readiness only — API
    execution is gated on ``AgenticLoopStatus.COMPLETED`` at the loop level.
    """

    endpoint: dict[str, Any] = Field(
        ...,
        description="The API endpoint definition (name, url, method, params, etc.)",
    )
    collected_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters collected for this endpoint so far",
    )
    completed: bool = Field(
        default=False,
        description=(
            "True once all required params for this endpoint have been collected. "
            "This flag signals param-readiness only — it does NOT indicate that the "
            "API has been called. Actual API execution is triggered by "
            "AgenticLoopStatus.COMPLETED at the loop level, after all endpoints "
            "reach this state."
        ),
    )


class APIToolSession(BaseModel):
    """Persisted session state for the API Tool Calling agentic loop.

    Keyed by chat_id in Redis with a sliding 30-minute TTL.
    """

    chat_id: str = Field(..., description="Unique conversation identifier")
    state: str = Field(
        ...,
        description="Current state of the agentic loop (e.g. 'collecting_params', 'ready', 'completed')",
    )
    selected_endpoint: dict[str, Any] | None = Field(
        default=None,
        description="The API endpoint selected for this conversation",
    )
    collected_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters collected from the user so far",
    )
    turn_count: int = Field(
        default=0,
        ge=0,
        description="Number of turns elapsed in the agentic loop",
    )
    max_turns: int = Field(
        default=5,
        ge=1,
        description="Maximum turns allowed before the session is abandoned",
    )
    awaiting_continuation: bool = Field(
        default=False,
        description=(
            "True when the loop has reached the continuation threshold and is waiting "
            "for the user to decide whether to continue collecting parameters or exit "
            "to the RAG workflow."
        ),
    )
    detected_language: str = Field(
        default="en",
        description=(
            "Language detected from the user's first message ('en', 'et', 'ru'). "
            "Persisted so all subsequent clarifying questions use the same language, "
            "even when follow-up messages are too short to reliably re-detect."
        ),
    )
    original_query: str = Field(
        default="",
        description=(
            "The user's first message that triggered this session. "
            "Preserved across turns so the response formatter always receives the "
            "full original intent, not just the last short follow-up message."
        ),
    )

    # ── Parallel mode fields (all optional/defaulted; empty = single mode) ────

    execution_mode: Literal["single", "parallel"] = Field(
        default="single",
        description=(
            "'single' or 'parallel' — drives which code path manages this session. "
            "Single-mode sessions leave the three fields below at their defaults."
        ),
    )
    parallel_endpoints: list[EndpointSessionState] = Field(
        default_factory=list,
        description=(
            "Per-endpoint state list populated when execution_mode='parallel'. "
            "Each entry tracks collected params and completion for one endpoint. "
            "Always empty in single mode."
        ),
    )
    active_endpoint_index: int = Field(
        default=0,
        ge=0,
        description=(
            "Index into parallel_endpoints of the endpoint currently being collected. "
            "Only relevant when execution_mode='parallel'; always 0 in single mode."
        ),
    )

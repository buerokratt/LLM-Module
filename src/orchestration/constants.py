"""Shared orchestration constants.

Kept in their own module so both the orchestration façade and the extracted
pipeline collaborators can use them without a circular import.
"""

from src.llm_orchestrator_config.llm_ochestrator_constants import (
    BUDGET_EXCEEDED_MESSAGES,
    CONNECTION_INACTIVE_MESSAGES,
    ENTITY_EXTRACTION_FAILED_MESSAGES,
    INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    INSUFFICIENT_CONTEXT_MESSAGES,
    NO_CONTEXT_AVAILABLE_MESSAGES,
    OUT_OF_SCOPE_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
    QUERY_VALIDATION_FAILED_MESSAGES,
    SERVICE_EXECUTION_ERROR_MESSAGES,
    SERVICE_NOT_FOUND_MESSAGES,
    SERVICE_TIMEOUT_ERROR_MESSAGES,
    SERVICE_VALIDATION_FAILED_MESSAGES,
    STREAM_TOKEN_LIMIT_MESSAGE,
    TECHNICAL_ISSUE_MESSAGES,
)

REFERENCES_SECTION_HEADER = "\n\n**References:**\n"

# Set of content strings that must NOT be persisted in conversation history.
# Covers all multilingual error / OOS / guardrail-violation messages so that
# failed or blocked exchanges are never written to Redis.
_HISTORY_EXCLUDED_MESSAGES: frozenset[str] = frozenset(
    {
        *OUT_OF_SCOPE_MESSAGES.values(),
        *TECHNICAL_ISSUE_MESSAGES.values(),
        *INPUT_GUARDRAIL_VIOLATION_MESSAGES.values(),
        *OUTPUT_GUARDRAIL_VIOLATION_MESSAGES.values(),
        *QUERY_VALIDATION_FAILED_MESSAGES.values(),
        *CONNECTION_INACTIVE_MESSAGES.values(),
        *BUDGET_EXCEEDED_MESSAGES.values(),
        *SERVICE_NOT_FOUND_MESSAGES.values(),
        *SERVICE_VALIDATION_FAILED_MESSAGES.values(),
        *SERVICE_TIMEOUT_ERROR_MESSAGES.values(),
        *SERVICE_EXECUTION_ERROR_MESSAGES.values(),
        *ENTITY_EXTRACTION_FAILED_MESSAGES.values(),
        *INSUFFICIENT_CONTEXT_MESSAGES.values(),
        *NO_CONTEXT_AVAILABLE_MESSAGES.values(),
        STREAM_TOKEN_LIMIT_MESSAGE,
    }
)

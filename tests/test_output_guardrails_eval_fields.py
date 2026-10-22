"""When the output guardrail blocks an answer, the replacement response must keep
the EVAL_MODE-only ``tool_calls`` so evaluation can score tool selection
separately from the guardrail outcome. Production is unaffected (field is None).

``LLMOrchestrationService.handle_output_guardrails`` delegates to
``GuardrailsCoordinator.handle_output``; the coordinator is tested directly.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from models.request_models import OrchestrationRequest, OrchestrationResponse
from src.guardrails import GuardrailCheckResult
from src.orchestration.guardrails_coordinator import GuardrailsCoordinator


def _coordinator(allowed: bool) -> GuardrailsCoordinator:
    coord = GuardrailsCoordinator(langfuse_config=None)
    coord.check_output = AsyncMock(  # type: ignore[method-assign]
        return_value=GuardrailCheckResult(
            allowed=allowed, verdict="safe" if allowed else "unsafe", reason="test"
        )
    )
    return coord


def _request() -> OrchestrationRequest:
    return OrchestrationRequest(
        chatId="chat-1",
        message="Viru tn 4, Tallinn?",
        authorId="u",
        conversationHistory=[],
        url="https://example.com",
        environment="testing",
        connection_id="evalconnection-1",
    )


_TOOL_CALLS = [
    {
        "name": "search_address",
        "method": "GET",
        "url": "https://inaadress.maaamet.ee/inaadress/gazetteer",
        "input_parameters": {"address": "Viru tn 4, Tallinn"},
        "success": True,
        "status_code": 200,
        "cache_source": None,
    }
]


def _response(tool_calls) -> OrchestrationResponse:
    return OrchestrationResponse(
        chatId="chat-1",
        llmServiceActive=True,
        questionOutOfLLMScope=False,
        inputGuardFailed=False,
        content="Viru tn 4, Tallinn asub Harju maakonnas.",
        tool_calls=tool_calls,
    )


@pytest.mark.asyncio
async def test_blocked_response_keeps_tool_calls() -> None:
    coord = _coordinator(allowed=False)
    original = _response(_TOOL_CALLS)

    result = await coord.handle_output(MagicMock(), original, _request(), {})

    assert isinstance(result, OrchestrationResponse)
    assert result is not original
    assert result.content != original.content  # replaced by the violation message
    assert result.tool_calls == _TOOL_CALLS


@pytest.mark.asyncio
async def test_blocked_response_without_tool_calls_stays_none() -> None:
    """Production shape: no eval field leaks into the JSON."""
    coord = _coordinator(allowed=False)

    result = await coord.handle_output(MagicMock(), _response(None), _request(), {})

    assert result.tool_calls is None
    assert "tool_calls" not in result.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_allowed_response_is_returned_unchanged() -> None:
    coord = _coordinator(allowed=True)
    original = _response(_TOOL_CALLS)

    result = await coord.handle_output(MagicMock(), original, _request(), {})

    assert result is original

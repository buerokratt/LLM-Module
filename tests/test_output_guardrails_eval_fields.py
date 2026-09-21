"""When the output guardrail blocks an answer, the replacement response must keep
the EVAL_MODE-only ``tool_calls`` so evaluation can score tool selection
separately from the guardrail outcome. Production is unaffected (field is None).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from guardrails.nemo_rails_adapter import GuardrailCheckResult
from models.request_models import OrchestrationRequest, OrchestrationResponse
from src.llm_orchestration_service import LLMOrchestrationService


def _service(allowed: bool) -> LLMOrchestrationService:
    svc: LLMOrchestrationService = object.__new__(LLMOrchestrationService)
    svc._check_output_guardrails = AsyncMock(  # type: ignore[method-assign]
        return_value=GuardrailCheckResult(
            allowed=allowed, verdict="safe" if allowed else "unsafe", reason="test"
        )
    )
    return svc


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
    svc = _service(allowed=False)
    original = _response(_TOOL_CALLS)

    result = await svc.handle_output_guardrails(MagicMock(), original, _request(), {})

    assert isinstance(result, OrchestrationResponse)
    assert result is not original
    assert result.content != original.content  # replaced by the violation message
    assert result.tool_calls == _TOOL_CALLS


@pytest.mark.asyncio
async def test_blocked_response_without_tool_calls_stays_none() -> None:
    """Production shape: no eval field leaks into the JSON."""
    svc = _service(allowed=False)

    result = await svc.handle_output_guardrails(
        MagicMock(), _response(None), _request(), {}
    )

    assert result.tool_calls is None
    assert "tool_calls" not in result.model_dump(exclude_none=True)


@pytest.mark.asyncio
async def test_allowed_response_is_returned_unchanged() -> None:
    svc = _service(allowed=True)
    original = _response(_TOOL_CALLS)

    result = await svc.handle_output_guardrails(MagicMock(), original, _request(), {})

    assert result is original

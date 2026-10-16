import asyncio
import contextvars
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import dspy
import pytest
from litellm import cost_calculator

from src.guardrails.dspy_nemo_adapter import DSPyNeMoLLM
from src.llm_orchestration_service import LLMOrchestrationService
from src.utils import cost_utils


def _add_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    dspy.settings.usage_tracker.add_usage(
        model,
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@pytest.mark.asyncio
async def test_parallel_requests_keep_usage_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_utils,
        "_price_model_usage",
        lambda _model, usage: usage["prompt_tokens"] + usage["completion_tokens"],
    )

    async def run_request(model: str, calls: int) -> dict:
        with cost_utils.track_request_usage() as tracker:
            for _ in range(calls):
                await asyncio.sleep(0)
                _add_usage(model, 2, 1)
        return cost_utils.get_request_usage_summary(tracker)

    first, second = await asyncio.gather(
        run_request("openai/model-a", 3),
        run_request("openai/model-b", 5),
    )

    assert first["total_calls"] == 3
    assert first["total_tokens"] == 9
    assert set(first["models"]) == {"openai/model-a"}
    assert second["total_calls"] == 5
    assert second["total_tokens"] == 15
    assert set(second["models"]) == {"openai/model-b"}
    assert first["total_calls"] + second["total_calls"] == 8


def test_request_tracker_is_independent_of_lm_history_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_utils,
        "_price_model_usage",
        lambda _model, usage: usage["prompt_tokens"] + usage["completion_tokens"],
    )
    history = deque(maxlen=5)

    with dspy.context(lm=SimpleNamespace(history=history)):
        with cost_utils.track_request_usage() as tracker:
            for _ in range(12):
                history.append({"usage": "bounded"})
                _add_usage("openai/model", 1, 1)

    summary = cost_utils.get_request_usage_summary(tracker)

    assert len(history) == 5
    assert summary["total_tokens"] == 24
    assert summary["total_calls"] == 12


def test_usage_from_an_alternate_model_is_kept_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cost_utils,
        "_price_model_usage",
        lambda _model, usage: usage["prompt_tokens"] + usage["completion_tokens"],
    )

    with cost_utils.track_request_usage() as tracker:
        _add_usage("azure/default-deployment", 10, 2)
        with dspy.context(lm=SimpleNamespace(model="openai/other-model")):
            _add_usage("openai/other-model", 4, 1)

    summary = cost_utils.get_request_usage_summary(tracker)

    assert summary["total_calls"] == 2
    assert summary["total_tokens"] == 17
    assert summary["models"]["azure/default-deployment"]["total_tokens"] == 12
    assert summary["models"]["openai/other-model"]["total_tokens"] == 5


def test_unknown_azure_deployment_uses_connection_model_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def price(model: str, prompt_tokens: int, completion_tokens: int) -> tuple:
        calls.append(model)
        if model == "azure/gpt-4o-deployment":
            raise ValueError("deployment is not in the LiteLLM price map")
        return (0.02, 0.03)

    monkeypatch.setattr(cost_calculator, "cost_per_token", price)
    with cost_utils.track_request_usage() as tracker:
        _add_usage("azure/gpt-4o-deployment", 100, 50)

    summary = cost_utils.get_request_usage_summary(
        tracker,
        {"deployment_name": "gpt-4o-deployment", "model": "gpt-4o"},
    )

    assert calls == ["azure/gpt-4o-deployment", "gpt-4o"]
    assert summary["models"]["azure/gpt-4o-deployment"]["total_cost"] == pytest.approx(
        0.05
    )


def test_unpriced_tokens_emit_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    warning = Mock()
    monkeypatch.setattr(cost_utils.logger, "warning", warning)
    monkeypatch.setattr(cost_calculator, "cost_per_token", lambda **_kwargs: (0, 0))

    with cost_utils.track_request_usage() as tracker:
        _add_usage("openai/unknown-model", 8, 2)

    summary = cost_utils.get_request_usage_summary(tracker)

    assert summary["total_tokens"] == 10
    assert summary["total_cost"] == 0
    warning.assert_called_once()


def test_request_usage_scope_can_close_in_another_context() -> None:
    usage_scope = cost_utils.track_request_usage()
    request_context = contextvars.copy_context()
    tracker = request_context.run(usage_scope.__enter__)

    request_context.copy().run(usage_scope.__exit__, None, None, None)

    assert tracker.get_total_tokens() == {}


@pytest.mark.asyncio
async def test_nemo_executor_propagates_request_usage_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MockLM:
        def __call__(self, _prompt: str, **_kwargs):
            def stream():
                _add_usage("openai/mock-model", 7, 3)
                yield "token"

            return stream()

    monkeypatch.setattr(DSPyNeMoLLM, "_get_dspy_lm", lambda _self: MockLM())
    monkeypatch.setattr(
        cost_utils,
        "_price_model_usage",
        lambda _model, usage: usage["prompt_tokens"] + usage["completion_tokens"],
    )
    adapter = DSPyNeMoLLM()

    with cost_utils.track_request_usage() as tracker:
        chunks = [chunk async for chunk in adapter._astream("prompt")]

    summary = cost_utils.get_request_usage_summary(tracker)

    assert [chunk.text for chunk in chunks] == ["token"]
    assert summary["total_tokens"] == 10
    assert summary["total_calls"] == 1


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        chatId="chat-1",
        authorId="author-1",
        environment="testing",
        connection_id="connection-1",
        message="question",
    )


def _service() -> LLMOrchestrationService:
    service = object.__new__(LLMOrchestrationService)
    service.langfuse_config = SimpleNamespace(langfuse_client=None)
    service._update_connection_budget = Mock()
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "input_block", "error"])
async def test_non_streaming_request_updates_budget_once(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setattr(cost_utils, "_price_model_usage", lambda _model, _usage: 0.25)
    service = _service()

    async def process(_request, _costs_metric, _usage_context):
        _add_usage("openai/model", 10, 5)
        if outcome == "error":
            raise RuntimeError("request failed")
        return outcome

    service._process_orchestration_request = AsyncMock(side_effect=process)

    if outcome == "error":
        with pytest.raises(RuntimeError, match="request failed"):
            await service.process_orchestration_request(_request())
    else:
        assert await service.process_orchestration_request(_request()) == outcome

    service._update_connection_budget.assert_called_once_with(
        "connection-1", {"request_total": {"total_cost": 0.25}}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", ["success", "input_block", "error", "disconnect"])
async def test_streaming_request_updates_budget_once(
    monkeypatch: pytest.MonkeyPatch, exit_path: str
) -> None:
    monkeypatch.setattr(cost_utils, "_price_model_usage", lambda _model, _usage: 0.25)
    service = _service()
    inner_closed = False

    async def stream(_request, _costs_metric, _usage_context):
        nonlocal inner_closed
        try:
            _add_usage("openai/model", 10, 5)
            yield "blocked" if exit_path == "input_block" else "partial"
            if exit_path == "error":
                raise RuntimeError("stream failed")
        finally:
            inner_closed = True

    service._stream_orchestration_response = stream
    response_stream = service.stream_orchestration_response(_request())
    first_chunk = await anext(response_stream)

    if exit_path == "disconnect":

        async def close_stream() -> None:
            await response_stream.aclose()

        await asyncio.create_task(close_stream())
        assert first_chunk == "partial"
    elif exit_path == "error":
        with pytest.raises(RuntimeError, match="stream failed"):
            await anext(response_stream)
    else:
        with pytest.raises(StopAsyncIteration):
            await anext(response_stream)
        expected_chunk = "blocked" if exit_path == "input_block" else "partial"
        assert first_chunk == expected_chunk

    assert inner_closed
    service._update_connection_budget.assert_called_once_with(
        "connection-1", {"request_total": {"total_cost": 0.25}}
    )

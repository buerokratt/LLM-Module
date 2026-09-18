"""Unit tests for ServiceWorkflowExecutor direct step executor methods.

Tests execute_direct_step() (non-streaming) and
execute_direct_step_streaming() (SSE) which handle #service button payloads
for multi-step MCQ flows.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.request_models import OrchestrationRequest
from tool_classifier.workflows.service_workflow import ServiceWorkflowExecutor


def _make_request(
    message: str = "#service, /POST/services/active/mcq_step_1",
    chat_id: str = "test-chat-123",
    author_id: str = "user-456",
) -> OrchestrationRequest:
    """Build a minimal OrchestrationRequest for testing."""
    return OrchestrationRequest(
        chatId=chat_id,
        authorId=author_id,
        message=message,
        url="https://test.example.com",
        environment="production",
        connection_id=None,
        conversationHistory=[],
    )


def _make_executor(
    orchestration_service: Any = None,
) -> ServiceWorkflowExecutor:
    """Build a ServiceWorkflowExecutor with a stubbed LLM manager."""
    return ServiceWorkflowExecutor(
        llm_manager=MagicMock(),
        orchestration_service=orchestration_service,
    )


SAMPLE_CONTENT = "Which year was your passport issued?"
SAMPLE_BUTTONS: List[Dict[str, Any]] = [
    {
        "title": "2023",
        "payload": "#service, /POST/services/active/mcq_year_2023",
    },
    {
        "title": "2024",
        "payload": "#service, /POST/services/active/mcq_year_2024",
    },
]


class TestExecuteDirectStep:
    """Tests for execute_direct_step (non-streaming)."""

    @pytest.mark.asyncio
    async def test_returns_response_with_content_and_buttons(self) -> None:
        """Valid prefix + successful endpoint → full OrchestrationResponse."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": SAMPLE_CONTENT, "buttons": SAMPLE_BUTTONS}
        )

        result = await executor.execute_direct_step(_make_request())

        assert result is not None
        assert result.content == SAMPLE_CONTENT
        assert result.buttons is not None
        assert len(result.buttons) == 2
        assert result.buttons[0].title == "2023"
        assert (
            result.buttons[0].payload == "#service, /POST/services/active/mcq_year_2023"
        )
        assert result.buttons[1].title == "2024"
        assert (
            result.buttons[1].payload == "#service, /POST/services/active/mcq_year_2024"
        )

    @pytest.mark.asyncio
    async def test_endpoint_returns_none(self) -> None:
        """Endpoint failure → method returns None."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(return_value=None)

        result = await executor.execute_direct_step(_make_request())

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_prefix_returns_none(self) -> None:
        """Unparseable message → returns None without calling the endpoint."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock()

        result = await executor.execute_direct_step(
            _make_request(message="Hello, I need help")
        )

        assert result is None
        executor._call_service_endpoint.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_buttons_sets_none(self) -> None:
        """No buttons in response → buttons field is None, not empty list."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "Final answer.", "buttons": []}
        )

        result = await executor.execute_direct_step(_make_request())

        assert result is not None
        assert result.buttons is None

    @pytest.mark.asyncio
    async def test_entities_array_is_empty(self) -> None:
        """Direct steps pass an empty entities_array to the endpoint."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "ok", "buttons": []}
        )

        await executor.execute_direct_step(_make_request())

        call_kwargs = executor._call_service_endpoint.call_args
        assert call_kwargs.kwargs["entities_array"] == []

    @pytest.mark.asyncio
    async def test_time_metric_populated(self) -> None:
        """time_metric['service.direct_step'] is set after the call."""
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "ok", "buttons": []}
        )
        time_metric: Dict[str, float] = {}

        await executor.execute_direct_step(_make_request(), time_metric=time_metric)

        assert "service.direct_step" in time_metric
        assert time_metric["service.direct_step"] >= 0

    @pytest.mark.asyncio
    async def test_buttons_without_required_keys_filtered(self) -> None:
        """Buttons missing 'title' or 'payload' are silently dropped."""
        bad_buttons = [
            {"title": "Good", "payload": "#service, /POST/ok"},
            {"title": "No payload field"},
            {"payload": "#service, /POST/no_title"},
        ]
        executor = _make_executor()
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "text", "buttons": bad_buttons}
        )

        result = await executor.execute_direct_step(_make_request())

        assert result is not None
        assert result.buttons is not None
        assert len(result.buttons) == 1
        assert result.buttons[0].title == "Good"


class TestExecuteDirectStepStreaming:
    """Tests for execute_direct_step_streaming (SSE)."""

    @pytest.mark.asyncio
    async def test_yields_content_and_end(self) -> None:
        """Valid prefix → yields exactly 2 SSE chunks (content, END)."""
        mock_sse = MagicMock()
        mock_sse.store_streaming_inference = AsyncMock()
        mock_sse.format_sse = MagicMock(side_effect=["sse_content", "sse_end"])

        executor = _make_executor(orchestration_service=mock_sse)
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": SAMPLE_CONTENT, "buttons": SAMPLE_BUTTONS}
        )

        stream = await executor.execute_direct_step_streaming(_make_request())

        assert stream is not None
        chunks = [chunk async for chunk in stream]
        assert chunks == ["sse_content", "sse_end"]

    @pytest.mark.asyncio
    async def test_format_sse_called_with_buttons(self) -> None:
        """format_sse receives content and buttons on first call, 'END' on second."""
        mock_sse = MagicMock()
        mock_sse.store_streaming_inference = AsyncMock()
        mock_sse.format_sse = MagicMock(return_value="data: ...\n\n")

        executor = _make_executor(orchestration_service=mock_sse)
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": SAMPLE_CONTENT, "buttons": SAMPLE_BUTTONS}
        )

        stream = await executor.execute_direct_step_streaming(_make_request())
        assert stream is not None
        _ = [chunk async for chunk in stream]

        calls = mock_sse.format_sse.call_args_list
        assert len(calls) == 2
        # First call: content + buttons
        assert calls[0].args == ("test-chat-123", SAMPLE_CONTENT, SAMPLE_BUTTONS)
        # Second call: END marker
        assert calls[1].args == ("test-chat-123", "END")

    @pytest.mark.asyncio
    async def test_endpoint_returns_none(self) -> None:
        """Endpoint failure → returns None (no stream)."""
        mock_sse = MagicMock()
        executor = _make_executor(orchestration_service=mock_sse)
        executor._call_service_endpoint = AsyncMock(return_value=None)

        result = await executor.execute_direct_step_streaming(_make_request())

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_prefix_returns_none(self) -> None:
        """Unparseable message → returns None without calling the endpoint."""
        mock_sse = MagicMock()
        executor = _make_executor(orchestration_service=mock_sse)
        executor._call_service_endpoint = AsyncMock()

        result = await executor.execute_direct_step_streaming(
            _make_request(message="just a normal question")
        )

        assert result is None
        executor._call_service_endpoint.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_orchestration_service_raises(self) -> None:
        """Missing orchestration_service → RuntimeError."""
        executor = _make_executor(orchestration_service=None)
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "ok", "buttons": []}
        )

        with pytest.raises(RuntimeError, match="not initialized for streaming"):
            await executor.execute_direct_step_streaming(_make_request())

    @pytest.mark.asyncio
    async def test_time_metric_populated(self) -> None:
        """time_metric['service.direct_step'] is set in streaming path."""
        mock_sse = MagicMock()
        mock_sse.format_sse = MagicMock(return_value="data: ...\n\n")

        executor = _make_executor(orchestration_service=mock_sse)
        executor._call_service_endpoint = AsyncMock(
            return_value={"content": "ok", "buttons": []}
        )
        time_metric: Dict[str, float] = {}

        await executor.execute_direct_step_streaming(
            _make_request(), time_metric=time_metric
        )

        assert "service.direct_step" in time_metric
        assert time_metric["service.direct_step"] >= 0

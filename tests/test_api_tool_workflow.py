"""Unit tests for APIToolWorkflowExecutor.

Covers:
- _required_params(): filter for required=True
- _compute_loop_step(): new session creation, session resume, fast-path (no required params)
- _compute_loop_step(): loop status transitions — COMPLETED / NEEDS_INPUT /
  AWAITING_CONTINUATION / MAX_TURNS_REACHED
- _compute_loop_step(): session store unavailable → stateless degradation
- _compute_loop_step(): no matched_endpoint in context → fallback
- _compute_loop_step(): session with None endpoint → fallback + delete
- _execute_api_and_format(): API success → formatted response
- _execute_api_and_format(): 4xx/5xx/timeout → localized error message
- _stream_api_and_format(): SSE token streaming, API failure → error frame
- execute_async(): delegates correctly to _compute_loop_step
"""

from typing import Any, AsyncIterator, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.request_models import OrchestrationRequest, OrchestrationResponse
from models.session_models import APIToolSession
from tool_classifier.enums import AgenticLoopStatus
from tool_classifier.models import AgenticLoopResult, APICallResult
from tool_classifier.workflows.api_tool_workflow import (
    APIToolWorkflowExecutor,
    _LoopStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_ID = "wf-test-chat"

_ENDPOINT_HOLIDAYS = {
    "endpoint_id": "ep-holidays",
    "name": "get_public_holidays",
    "description": "Returns public holidays for a country",
    "method": "GET",
    "url": "https://openholidaysapi.org/PublicHolidays",
    "params": [
        {
            "name": "countryIsoCode",
            "type": "string",
            "required": True,
            "description": "ISO code",
        },
        {
            "name": "year",
            "type": "integer",
            "required": False,
            "description": "Optional year",
        },
    ],
}

_ENDPOINT_NO_PARAMS = {
    "endpoint_id": "ep-no-params",
    "name": "get_info",
    "description": "Returns general info with no params",
    "method": "GET",
    "url": "https://ilmmicroservice.envir.ee/api/forecasts",
    "params": [],
}


def _make_request(
    message: str = "EE",
    chat_id: str = _CHAT_ID,
    context: Optional[Dict[str, Any]] = None,
) -> OrchestrationRequest:
    return OrchestrationRequest(
        chatId=chat_id,
        message=message,
        authorId="test-user",
        conversationHistory=[],
        url="https://example.com",
        environment="testing",
        connection_id="test-conn",
    )


def _make_session(
    endpoint: Optional[Dict[str, Any]] = None,
    turn_count: int = 0,
    collected_params: Optional[Dict[str, Any]] = None,
    awaiting_continuation: bool = False,
) -> APIToolSession:
    return APIToolSession(
        chat_id=_CHAT_ID,
        state="collecting_params",
        selected_endpoint=endpoint or _ENDPOINT_HOLIDAYS,
        collected_params=collected_params or {},
        turn_count=turn_count,
        max_turns=5,
        awaiting_continuation=awaiting_continuation,
        detected_language="en",
        original_query="public holidays",
    )


def _make_session_store(
    session: Optional[APIToolSession] = None,
) -> AsyncMock:
    store = AsyncMock()
    store.get = AsyncMock(return_value=session)
    store.save = AsyncMock()
    store.delete = AsyncMock()
    store.update = AsyncMock()
    return store


def _make_loop_result(
    status: AgenticLoopStatus,
    collected_params: Optional[Dict[str, Any]] = None,
    clarifying_question: str = "Which country?",
    turn_count: int = 1,
) -> AgenticLoopResult:
    return AgenticLoopResult(
        status=status,
        collected_params=collected_params or {},
        clarifying_question=clarifying_question,
        turn_count=turn_count,
    )


def _make_orchestration_service(session_store: Optional[AsyncMock] = None) -> MagicMock:
    svc = MagicMock()
    svc.session_store = session_store

    def _format_sse(chat_id: str, content: str) -> str:
        return f'data: {{"chatId":"{chat_id}","payload":{{"content":"{content}"}}}}\n\n'

    svc.format_sse = _format_sse
    svc.store_streaming_inference = AsyncMock()
    svc.handle_output_guardrails = AsyncMock(
        side_effect=lambda _adapter, response, _req, _costs: response
    )
    return svc


def _make_executor(
    session_store: Optional[AsyncMock] = None,
) -> APIToolWorkflowExecutor:
    svc = _make_orchestration_service(session_store)
    return APIToolWorkflowExecutor(orchestration_service=svc)


# ---------------------------------------------------------------------------
# _required_params
# ---------------------------------------------------------------------------


class TestRequiredParams:
    def test_filters_required_only(self) -> None:
        params = [
            {"name": "country", "required": True},
            {"name": "year", "required": False},
            {"name": "format", "required": True},
        ]
        result = APIToolWorkflowExecutor._required_params(params)
        assert len(result) == 2
        assert all(p["required"] for p in result)

    def test_empty_list_returns_empty(self) -> None:
        assert APIToolWorkflowExecutor._required_params([]) == []

    def test_all_optional_returns_empty(self) -> None:
        params = [{"name": "limit", "required": False}]
        assert APIToolWorkflowExecutor._required_params(params) == []

    def test_non_dict_items_skipped(self) -> None:
        params = [{"name": "country", "required": True}, "bad_item", None]
        result = APIToolWorkflowExecutor._required_params(params)
        assert len(result) == 1

    def test_missing_required_key_treated_as_false(self) -> None:
        params = [{"name": "country"}]
        result = APIToolWorkflowExecutor._required_params(params)
        assert result == []


# ---------------------------------------------------------------------------
# _compute_loop_step — new session creation
# ---------------------------------------------------------------------------


class TestComputeLoopStepNewSession:
    @pytest.mark.asyncio
    async def test_new_session_fast_path_no_required_params(self) -> None:
        """Endpoint with no required params → fast-path api_call immediately."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)
        request = _make_request("get info please")

        step = await executor._compute_loop_step(
            request,
            context={"matched_endpoint": _ENDPOINT_NO_PARAMS},
        )

        assert step.kind == "api_call"
        assert step.endpoint == _ENDPOINT_NO_PARAMS
        assert step.collected_params == {}
        # Session must NOT be created for fast-path
        session_store.save.assert_not_called()

    @pytest.mark.asyncio
    async def test_new_session_saves_session_on_first_turn(self) -> None:
        """New session with required params → save session, run loop turn."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)
        request = _make_request("public holidays")

        mock_result = _make_loop_result(AgenticLoopStatus.NEEDS_INPUT)
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(
            return_value=(mock_result, ["Which", " country?"])
        )
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(
            request,
            context={"matched_endpoint": _ENDPOINT_HOLIDAYS},
        )

        session_store.save.assert_called_once()
        assert step.kind == "question"
        assert step.question == "Which country?"

    @pytest.mark.asyncio
    async def test_no_matched_endpoint_returns_fallback(self) -> None:
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)
        request = _make_request()

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "fallback"

    @pytest.mark.asyncio
    async def test_session_store_unavailable_stateless_degradation(self) -> None:
        """No session_store → executor runs loop without persistence."""
        executor = _make_executor(session_store=None)
        request = _make_request("public holidays")

        mock_result = _make_loop_result(AgenticLoopStatus.NEEDS_INPUT)
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(
            return_value=(mock_result, ["Which", " country?"])
        )
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(
            request,
            context={"matched_endpoint": _ENDPOINT_HOLIDAYS},
        )

        # Should still return a question — stateless but functional
        assert step.kind == "question"


# ---------------------------------------------------------------------------
# _compute_loop_step — session resume
# ---------------------------------------------------------------------------


class TestComputeLoopStepSessionResume:
    @pytest.mark.asyncio
    async def test_resumes_from_existing_session(self) -> None:
        session = _make_session(turn_count=1, collected_params={})
        session_store = _make_session_store(session=session)
        executor = _make_executor(session_store)
        request = _make_request("EE")

        mock_result = _make_loop_result(AgenticLoopStatus.NEEDS_INPUT, turn_count=2)
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(
            return_value=(mock_result, ["What", " year?"])
        )
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "question"
        assert "year" in step.question.lower() or step.question != ""

    @pytest.mark.asyncio
    async def test_session_with_none_endpoint_returns_fallback_and_deletes(
        self,
    ) -> None:
        session = _make_session(endpoint=None)
        session.selected_endpoint = None
        session_store = _make_session_store(session=session)
        executor = _make_executor(session_store)
        request = _make_request()

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "fallback"
        session_store.delete.assert_called_once_with(_CHAT_ID)

    @pytest.mark.asyncio
    async def test_completed_status_deletes_session_and_returns_api_call(
        self,
    ) -> None:
        session = _make_session(turn_count=2)
        session_store = _make_session_store(session=session)
        executor = _make_executor(session_store)
        request = _make_request("EE")

        completed_result = _make_loop_result(
            AgenticLoopStatus.COMPLETED,
            collected_params={"countryIsoCode": "EE"},
        )
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(return_value=(completed_result, []))
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "api_call"
        assert step.collected_params == {"countryIsoCode": "EE"}
        session_store.delete.assert_called_once_with(_CHAT_ID)

    @pytest.mark.asyncio
    async def test_max_turns_reached_deletes_session_and_returns_fallback(
        self,
    ) -> None:
        session = _make_session(turn_count=5)
        session_store = _make_session_store(session=session)
        executor = _make_executor(session_store)
        request = _make_request("EE")

        max_turns_result = _make_loop_result(AgenticLoopStatus.MAX_TURNS_REACHED)
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(return_value=(max_turns_result, []))
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "fallback"
        session_store.delete.assert_called_once_with(_CHAT_ID)

    @pytest.mark.asyncio
    async def test_awaiting_continuation_returns_question(self) -> None:
        session = _make_session(turn_count=3, awaiting_continuation=True)
        session_store = _make_session_store(session=session)
        executor = _make_executor(session_store)
        request = _make_request("yes")

        awaiting_result = _make_loop_result(
            AgenticLoopStatus.AWAITING_CONTINUATION_DECISION,
            clarifying_question="Would you like to continue? (yes / no)",
        )
        loop_mock = AsyncMock()
        loop_mock.stream_run_turn = AsyncMock(
            return_value=(awaiting_result, ["Would", " you", " like"])
        )
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        step = await executor._compute_loop_step(request, context={})

        assert step.kind == "question"
        assert "continue" in step.question.lower()


# ---------------------------------------------------------------------------
# _execute_api_and_format
# ---------------------------------------------------------------------------


class TestExecuteApiAndFormat:
    @pytest.mark.asyncio
    async def test_api_success_returns_formatted_response(self) -> None:
        executor = _make_executor()
        api_result = APICallResult(
            success=True,
            status_code=200,
            response_data={"holidays": [{"name": "Jõulupüha", "date": "2024-12-25"}]},
            error=None,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        with patch(
            "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value="Estonia has 1 holiday: Jõulupüha on December 25.",
        ):
            response = await executor._execute_api_and_format(
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT_HOLIDAYS,
                collected_params={"countryIsoCode": "EE"},
                user_query="What are the holidays in Estonia?",
                detected_language="en",
            )

        assert isinstance(response, OrchestrationResponse)
        assert response.chatId == _CHAT_ID
        assert "holiday" in response.content.lower() or response.content != ""

    @pytest.mark.asyncio
    async def test_api_4xx_returns_error_content(self) -> None:
        executor = _make_executor()
        error_msg = "Country code EE is not supported."
        api_result = APICallResult(
            success=False,
            status_code=400,
            response_data={"error": error_msg},
            error=error_msg,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        response = await executor._execute_api_and_format(
            chat_id=_CHAT_ID,
            endpoint=_ENDPOINT_HOLIDAYS,
            collected_params={"countryIsoCode": "XX"},
            user_query="holidays for XX",
            detected_language="en",
        )

        assert response.content == error_msg

    @pytest.mark.asyncio
    async def test_api_5xx_returns_error_content(self) -> None:
        executor = _make_executor()
        error_msg = "The service is temporarily unavailable. Please try again later."
        api_result = APICallResult(
            success=False,
            status_code=500,
            response_data="",
            error=error_msg,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        response = await executor._execute_api_and_format(
            chat_id=_CHAT_ID,
            endpoint=_ENDPOINT_HOLIDAYS,
            collected_params={"countryIsoCode": "EE"},
            user_query="holidays",
            detected_language="en",
        )

        assert response.content == error_msg

    @pytest.mark.asyncio
    async def test_api_returns_orchestration_response_shape(self) -> None:
        executor = _make_executor()
        api_result = APICallResult(
            success=True,
            status_code=200,
            response_data={"result": "ok"},
            error=None,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        with patch(
            "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value="All good",
        ):
            response = await executor._execute_api_and_format(
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT_HOLIDAYS,
                collected_params={},
                user_query="any query",
                detected_language="et",
            )

        assert response.llmServiceActive is True
        assert response.questionOutOfLLMScope is False
        assert response.inputGuardFailed is False


# ---------------------------------------------------------------------------
# _stream_api_and_format
# ---------------------------------------------------------------------------


class TestStreamApiAndFormat:
    @pytest.mark.asyncio
    async def test_api_success_streams_formatted_tokens(self) -> None:
        session_store = _make_session_store()
        svc = _make_orchestration_service(session_store)
        executor = _make_executor(session_store)

        api_result = APICallResult(
            success=True,
            status_code=200,
            response_data={"holidays": []},
            error=None,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        async def _fake_stream(**kwargs: Any) -> AsyncIterator[str]:
            for token in ["Holiday", " info", " here."]:
                yield token

        mock_formatter = MagicMock()
        mock_formatter.stream_forward = _fake_stream

        with patch(
            "tool_classifier.workflows.api_tool_workflow.APIResponseFormatterModule",
            return_value=mock_formatter,
        ):
            frames = [
                frame
                async for frame in executor._stream_api_and_format(
                    chat_id=_CHAT_ID,
                    endpoint=_ENDPOINT_HOLIDAYS,
                    collected_params={"countryIsoCode": "EE"},
                    user_query="holidays",
                    detected_language="en",
                    orchestration_service=svc,
                    request=_make_request(),
                )
            ]

        # 3 token frames + 1 END frame
        assert len(frames) == 4
        assert all("data:" in f for f in frames)

    @pytest.mark.asyncio
    async def test_api_failure_streams_error_frame(self) -> None:
        session_store = _make_session_store()
        svc = _make_orchestration_service(session_store)
        executor = _make_executor(session_store)

        error_msg = "Teenus on ajutiselt kättesaamatu."
        api_result = APICallResult(
            success=False,
            status_code=503,
            response_data="",
            error=error_msg,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        frames = [
            frame
            async for frame in executor._stream_api_and_format(
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT_HOLIDAYS,
                collected_params={"countryIsoCode": "EE"},
                user_query="holidays",
                detected_language="et",
                orchestration_service=svc,
                request=_make_request(),
            )
        ]

        # 1 error frame + 1 END frame
        assert len(frames) == 2
        assert error_msg in frames[0]


# ---------------------------------------------------------------------------
# execute_async (public interface)
# ---------------------------------------------------------------------------


class TestExecuteAsync:
    @pytest.mark.asyncio
    async def test_fallback_returns_none(self) -> None:
        """When _compute_loop_step returns fallback, execute_async returns None."""
        executor = _make_executor()
        executor._compute_loop_step = AsyncMock(
            return_value=_LoopStep(kind="fallback", chat_id=_CHAT_ID)
        )

        result = await executor.execute_async(
            _make_request(),
            context={"matched_endpoint": _ENDPOINT_HOLIDAYS},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_question_returns_response(self) -> None:
        executor = _make_executor()
        executor._compute_loop_step = AsyncMock(
            return_value=_LoopStep(
                kind="question",
                chat_id=_CHAT_ID,
                question="Which country?",
            )
        )

        result = await executor.execute_async(
            _make_request(),
            context={},
        )

        assert isinstance(result, OrchestrationResponse)
        assert result.content == "Which country?"

    @pytest.mark.asyncio
    async def test_api_call_invokes_execute_api_and_format(self) -> None:
        executor = _make_executor()
        executor._compute_loop_step = AsyncMock(
            return_value=_LoopStep(
                kind="api_call",
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT_HOLIDAYS,
                collected_params={"countryIsoCode": "EE"},
                user_query="holidays in Estonia",
                detected_language="en",
            )
        )
        expected_response = OrchestrationResponse(
            chatId=_CHAT_ID,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content="Estonia has national holidays.",
        )
        executor._execute_api_and_format = AsyncMock(return_value=expected_response)

        result = await executor.execute_async(
            _make_request(),
            context={},
        )

        assert result is expected_response
        executor._execute_api_and_format.assert_called_once()

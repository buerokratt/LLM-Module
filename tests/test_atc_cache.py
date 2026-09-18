"""Unit tests for the ATC Response Cache integration — T-6 (tests 6–13 of the spec).

Tests 1–5 (L1 hit, L1 miss, param normalisation, L2 round-trip, L2 invalidate) are
covered by ``test_atc_cache_store.py`` and are not duplicated here.

This file exercises the eight cross-cutting scenarios:

  A) Workflow write: ``cacheable=False`` endpoint → no L1 write
  B) ``_compute_loop_step``: L1 hit → ``cached_response`` step
  C) ``_compute_loop_step``: L2 routing — four FollowUpDetector outcomes
  D) ``ToolClassifier``: intent switch → ``invalidate_l2``
  E) Multi-intent write: one ``LastCallContext`` per succeeded endpoint
"""

import asyncio
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.request_models import OrchestrationRequest
from models.session_models import (
    APIToolSession,
    EndpointSessionState,
    LastCallContext,
)
from tool_classifier.classifier import ToolClassifier
from tool_classifier.enums import AgenticLoopStatus, WorkflowType
from tool_classifier.models import (
    AgenticLoopResult,
    APICallResult,
    MultiAPICallResult,
)
from tool_classifier.workflows.api_tool_workflow import APIToolWorkflowExecutor
from utils.atc_cache_store import ATCCacheStore

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_CHAT_ID = "atc-cache-test-001"
_API_NAME = "get_national_holidays"
_RESPONSE: Dict[str, Any] = {
    "holidays": [{"date": "2026-02-24", "name": "Independence Day"}]
}
_PARAMS: Dict[str, Any] = {"year": 2026, "country": "EE"}

# Endpoint with two required params (year + country)
_ENDPOINT: Dict[str, Any] = {
    "endpoint_id": "ep-holidays",
    "name": _API_NAME,
    "description": "Returns public holidays for a country and year",
    "method": "GET",
    "url": "https://openholidaysapi.org/PublicHolidays",
    "params": [
        {"name": "year", "type": "integer", "required": True, "description": "Year"},
        {
            "name": "country",
            "type": "string",
            "required": True,
            "description": "Country ISO",
        },
    ],
    "cacheable": True,
    "cache_ttl_seconds": None,
}

_ENDPOINT_NON_CACHEABLE: Dict[str, Any] = {
    **_ENDPOINT,
    "name": "get_doc_status",
    "cacheable": False,
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_last_call_ctx(
    api_name: str = _API_NAME,
    collected_params: Optional[Dict[str, Any]] = None,
) -> LastCallContext:
    return LastCallContext(
        api_name=api_name,
        endpoint=_ENDPOINT,
        collected_params=collected_params if collected_params is not None else _PARAMS,
        raw_response=_RESPONSE,
        original_query="What are the public holidays in Estonia in 2026?",
        timestamp=1748000000.0,
    )


def _make_request(
    message: str = "public holidays",
    chat_id: str = _CHAT_ID,
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


def _make_session_store(
    session: Optional[APIToolSession] = None,
) -> AsyncMock:
    store = AsyncMock()
    store.get = AsyncMock(return_value=session)
    store.save = AsyncMock()
    store.delete = AsyncMock()
    store.update = AsyncMock()
    return store


def _make_orchestration_service(
    session_store: Optional[AsyncMock] = None,
) -> MagicMock:
    svc = MagicMock()
    svc.session_store = session_store
    # None prevents _get_custom_instructions from calling asyncio.to_thread, which
    # would otherwise consume the mocked return value before the FUD call.
    svc.prompt_config_loader = None

    def _format_sse(chat_id: str, content: str, **kwargs: Any) -> str:
        return f'data: {{"chatId":"{chat_id}","payload":{{"content":"{content}"}}}}\n\n'

    svc.format_sse = _format_sse
    return svc


def _make_executor(
    session_store: Optional[AsyncMock] = None,
) -> APIToolWorkflowExecutor:
    return APIToolWorkflowExecutor(
        orchestration_service=_make_orchestration_service(session_store)
    )


def _make_loop_result(
    status: AgenticLoopStatus,
    collected_params: Optional[Dict[str, Any]] = None,
) -> AgenticLoopResult:
    return AgenticLoopResult(
        status=status,
        collected_params=collected_params or {},
        clarifying_question="Which year?",
        turn_count=1,
    )


def _make_loop_mock() -> AsyncMock:
    """Return a mock AgenticLoop that reports NEEDS_INPUT with a question."""
    loop_mock = AsyncMock()
    loop_mock.stream_run_turn = AsyncMock(
        return_value=(
            _make_loop_result(AgenticLoopStatus.NEEDS_INPUT),
            ["Which", " year?"],
        )
    )
    return loop_mock


# ─────────────────────────────────────────────────────────────────────────────
# Group A — Workflow write path
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheableFlag:
    @pytest.mark.asyncio
    async def test_l1_skipped_when_not_cacheable(self) -> None:
        """cacheable=False endpoint → no L1 write after a successful API call."""
        executor = _make_executor()
        api_result = APICallResult(
            success=True,
            status_code=200,
            response_data=_RESPONSE,
            error=None,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        mock_cache_store = AsyncMock()
        mock_cache_store.set_l1 = AsyncMock()
        mock_cache_store.set_l2 = AsyncMock()

        with (
            patch(
                "tool_classifier.workflows.api_tool_workflow.ATCCacheStore",
                return_value=mock_cache_store,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value="Formatted response.",
            ),
        ):
            await executor._execute_api_and_format(
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT_NON_CACHEABLE,
                collected_params=_PARAMS,
                user_query="document status?",
                detected_language="en",
            )
            # Flush any background tasks that might have been scheduled
            await asyncio.sleep(0)

        # Neither L1 nor L2 may be written for non-cacheable endpoints
        mock_cache_store.set_l1.assert_not_called()
        mock_cache_store.set_l2.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Group B — _compute_loop_step: L1 cache hit
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeLoopStepL1Hit:
    @pytest.mark.asyncio
    async def test_cache_hit_in_compute_loop_step(self) -> None:
        """L1 hit on a new (session-less) query → _LoopStep(kind='cached_response')."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)

        with (
            patch.object(
                ATCCacheStore,
                "get_l1",
                new=AsyncMock(return_value=_RESPONSE),
            ),
            patch.object(ATCCacheStore, "get_l2", new=AsyncMock(return_value=None)),
            patch.object(ATCCacheStore, "set_l1", new=AsyncMock()),
            patch.object(ATCCacheStore, "set_l2", new=AsyncMock()),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request(),
                context={"matched_endpoint": _ENDPOINT},
            )

        assert step.kind == "cached_response"
        assert step.cache_source == "L1"
        assert step.cached_raw_response == _RESPONSE


# ─────────────────────────────────────────────────────────────────────────────
# Group C — _compute_loop_step: L2 follow-up routing
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeLoopStepL2Routing:
    """L1 miss + L2 hit: four outcomes based on FollowUpDetectorModule result.

    All four tests share:
    - No active session (session_store.get returns None)
    - L1 miss  (get_l1 returns None)
    - L2 hit   (get_l2 returns [ctx] where ctx.api_name matches endpoint name)
    - asyncio.to_thread is patched to return the desired FUD result dict directly
    """

    @pytest.mark.asyncio
    async def test_follow_up_detector_response_question(self) -> None:
        """FUD → 'response_question': returns cached_response from L2 raw_response."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)
        ctx = _make_last_call_ctx(collected_params=_PARAMS)

        fud_result = {"follow_up_type": "response_question", "updated_params": {}}

        with (
            patch.object(ATCCacheStore, "get_l1", new=AsyncMock(return_value=None)),
            patch.object(ATCCacheStore, "get_l2", new=AsyncMock(return_value=[ctx])),
            patch.object(ATCCacheStore, "set_l1", new=AsyncMock()),
            patch.object(ATCCacheStore, "set_l2", new=AsyncMock()),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=fud_result,
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request("Which of those holidays falls on a Monday?"),
                context={"matched_endpoint": _ENDPOINT},
            )

        assert step.kind == "cached_response"
        assert step.cache_source == "L2"
        assert step.cached_raw_response == ctx.raw_response

    @pytest.mark.asyncio
    async def test_follow_up_detector_param_update_complete(self) -> None:
        """FUD → 'param_update'; all required params in merged result → api_call step.

        Previous call had only 'country'; FUD supplies 'year' → merged is complete.
        _param_hash is NOT patched so the real inequality check runs correctly.
        """
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)

        # Prior call was missing 'year'; the new query supplies it
        ctx = _make_last_call_ctx(collected_params={"country": "EE"})
        fud_result = {
            "follow_up_type": "param_update",
            "updated_params": {"year": 2025},
        }

        with (
            patch.object(ATCCacheStore, "get_l1", new=AsyncMock(return_value=None)),
            patch.object(ATCCacheStore, "get_l2", new=AsyncMock(return_value=[ctx])),
            patch.object(ATCCacheStore, "set_l1", new=AsyncMock()),
            patch.object(ATCCacheStore, "set_l2", new=AsyncMock()),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=fud_result,
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request("What about 2025 instead?"),
                context={"matched_endpoint": _ENDPOINT},
            )

        assert step.kind == "api_call"
        assert step.collected_params == {"country": "EE", "year": 2025}

    @pytest.mark.asyncio
    async def test_follow_up_detector_param_update_partial(self) -> None:
        """FUD → 'param_update'; merged params still incomplete → seeded_params set.

        Previous call only has 'year'; no updated params → 'country' still missing.
        The agentic loop is mocked so the test can verify the step kind.
        """
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)

        ctx = _make_last_call_ctx(collected_params={"year": 2026})
        # Empty update → merged = {"year": 2026}, still missing "country"
        fud_result = {"follow_up_type": "param_update", "updated_params": {}}

        loop_mock = _make_loop_mock()
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        context: Dict[str, Any] = {"matched_endpoint": _ENDPOINT}

        with (
            patch.object(ATCCacheStore, "get_l1", new=AsyncMock(return_value=None)),
            patch.object(ATCCacheStore, "get_l2", new=AsyncMock(return_value=[ctx])),
            patch.object(ATCCacheStore, "set_l1", new=AsyncMock()),
            patch.object(ATCCacheStore, "set_l2", new=AsyncMock()),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=fud_result,
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request("Show me the holidays"),
                context=context,
            )

        # seeded_params must be injected into the mutable context dict
        assert "seeded_params" in context
        assert context["seeded_params"] == {"year": 2026}
        # Normal agentic loop ran after seeding → returns a question
        assert step.kind == "question"

    @pytest.mark.asyncio
    async def test_follow_up_detector_new_intent(self) -> None:
        """FUD → 'new_intent': falls through to normal loop; seeded_params NOT set."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)

        ctx = _make_last_call_ctx(collected_params=_PARAMS)
        fud_result = {"follow_up_type": "new_intent", "updated_params": {}}

        loop_mock = _make_loop_mock()
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        context: Dict[str, Any] = {"matched_endpoint": _ENDPOINT}

        with (
            patch.object(ATCCacheStore, "get_l1", new=AsyncMock(return_value=None)),
            patch.object(ATCCacheStore, "get_l2", new=AsyncMock(return_value=[ctx])),
            patch.object(ATCCacheStore, "set_l1", new=AsyncMock()),
            patch.object(ATCCacheStore, "set_l2", new=AsyncMock()),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value=fud_result,
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request("How do I apply for a driving licence?"),
                context=context,
            )

        # new_intent path must NOT seed params
        assert "seeded_params" not in context
        # Normal agentic loop took over → question returned
        assert step.kind == "question"


# ─────────────────────────────────────────────────────────────────────────────
# Group D — ToolClassifier: intent switch → invalidate_l2
# ─────────────────────────────────────────────────────────────────────────────

_ENDPOINT_HOLIDAYS_CLS: Dict[str, Any] = {
    "endpoint_id": "ep-holidays",
    "name": "get_public_holidays",
    "description": "Returns public holidays",
    "method": "GET",
    "url": "https://openholidaysapi.org/PublicHolidays",
    "params": [
        {
            "name": "countryIsoCode",
            "type": "string",
            "required": True,
            "description": "ISO",
        }
    ],
    "cosine_score": 0.68,
    "rrf_score": 0.01,
    "confidence": "high",
}

_ENDPOINT_WEATHER_CLS: Dict[str, Any] = {
    "endpoint_id": "ep-weather",
    "name": "get_weather",
    "description": "Current weather for a city",
    "method": "GET",
    "url": "https://publicapi.envir.ee/v1/weather",
    "params": [],
    "cosine_score": 0.65,
    "rrf_score": 0.009,
    "confidence": "high",
}


def _make_api_tool_search_result(endpoint: Dict[str, Any]) -> MagicMock:
    """Build a mock APIToolSearchResult from an endpoint dict."""
    result = MagicMock()
    result.endpoint_id = endpoint["endpoint_id"]
    result.name = endpoint["name"]
    result.description = endpoint["description"]
    result.method = endpoint["method"]
    result.url = endpoint["url"]
    result.params = endpoint.get("params", [])
    result.cosine_score = endpoint.get("cosine_score", 0.65)
    result.rrf_score = endpoint.get("rrf_score", 0.01)
    result.confidence = endpoint.get("confidence", "high")
    result.to_dict.return_value = endpoint
    return result


def _make_classifier_instance(
    session_store: Optional[AsyncMock] = None,
) -> ToolClassifier:
    """Instantiate ToolClassifier with all external I/O patched away."""
    svc = MagicMock()
    svc.create_embeddings_for_indexer.return_value = {"embeddings": [[0.1] * 10]}
    svc.session_store = session_store

    with patch("tool_classifier.classifier.httpx.AsyncClient"):
        return ToolClassifier(
            llm_manager=MagicMock(),
            orchestration_service=svc,
        )


class TestIntentSwitchInvalidatesL2:
    @pytest.mark.asyncio
    async def test_intent_switch_invalidates_l2(self) -> None:
        """Intent switch during session resume → ATCCacheStore.invalidate_l2 called."""
        # Active session for endpoint A (holidays)
        active_session = APIToolSession(
            chat_id=_CHAT_ID,
            state="collecting_params",
            selected_endpoint=_ENDPOINT_HOLIDAYS_CLS,
            collected_params={},
            turn_count=1,
            max_turns=5,
            awaiting_continuation=False,
            detected_language="en",
            original_query="holiday query",
        )
        session_store = _make_session_store(session=active_session)

        classifier = _make_classifier_instance(session_store)
        # New query matches endpoint B (weather) — a different endpoint
        classifier.api_tool_searcher.search = AsyncMock(
            return_value=[_make_api_tool_search_result(_ENDPOINT_WEATHER_CLS)]
        )

        mock_cache_store = AsyncMock()
        mock_cache_store.invalidate_l2 = AsyncMock()

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags"
                ".API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.ATC_RESPONSE_CACHE_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.ATCCacheStore",
                return_value=mock_cache_store,
            ),
        ):
            result = await classifier.classify(
                query="What is the weather in Tallinn?",
                conversation_history=[],
                language="en",
                request=_make_request("What is the weather in Tallinn?"),
            )

        # Both L2 invalidation and session delete must fire on intent switch
        mock_cache_store.invalidate_l2.assert_called_once_with(_CHAT_ID)
        session_store.delete.assert_called_once_with(_CHAT_ID)
        assert result.workflow == WorkflowType.API_TOOL_CALLING
        assert result.metadata.get("matched_endpoint", {}).get("name") == "get_weather"


# ─────────────────────────────────────────────────────────────────────────────
# Group E — Multi-intent write: one LastCallContext per succeeded endpoint
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiIntentCacheWrite:
    @pytest.mark.asyncio
    async def test_multi_intent_writes_multiple_l2_entries(self) -> None:
        """_execute_multi_api_and_format stores one LastCallContext per succeeded endpoint."""
        executor = _make_executor()

        ep1: Dict[str, Any] = {
            "name": "get_national_holidays",
            "description": "Returns holidays",
            "method": "GET",
            "url": "https://example.com/holidays",
            "cacheable": True,
            "cache_ttl_seconds": None,
        }
        ep2: Dict[str, Any] = {
            "name": "get_electricity_prices",
            "description": "Returns electricity prices",
            "method": "GET",
            "url": "https://example.com/electricity",
            "cacheable": True,
            "cache_ttl_seconds": None,
        }

        parallel_endpoints = [
            EndpointSessionState(
                endpoint=ep1, collected_params={"country": "EE"}, completed=True
            ),
            EndpointSessionState(
                endpoint=ep2, collected_params={"date": "2026-05-28"}, completed=True
            ),
        ]

        multi_result = MultiAPICallResult(
            results=[
                APICallResult(
                    success=True, status_code=200, response_data={"h": 1}, error=None
                ),
                APICallResult(
                    success=True, status_code=200, response_data={"p": 2}, error=None
                ),
            ],
            endpoints=[ep1, ep2],
        )

        mock_multi_caller = AsyncMock()
        mock_multi_caller.call_all = AsyncMock(return_value=multi_result)

        mock_cache_store = AsyncMock()
        mock_cache_store.set_l1 = AsyncMock()
        mock_cache_store.set_l2 = AsyncMock()

        with (
            patch(
                "tool_classifier.workflows.api_tool_workflow.MultiAPICaller",
                return_value=mock_multi_caller,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.ATCCacheStore",
                return_value=mock_cache_store,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value="Multi-endpoint formatted response.",
            ),
        ):
            await executor._execute_multi_api_and_format(
                chat_id=_CHAT_ID,
                parallel_endpoints=parallel_endpoints,
                user_query="Show me holidays and electricity prices",
                detected_language="en",
            )
            # Flush the background _write_multi_cache() asyncio.create_task
            await asyncio.sleep(0)

        # set_l2 must be called once with both endpoint contexts
        mock_cache_store.set_l2.assert_called_once()
        _call_args = mock_cache_store.set_l2.call_args
        contexts: list[LastCallContext] = _call_args[0][1]
        assert len(contexts) == 2
        api_names = {c.api_name for c in contexts}
        assert api_names == {"get_national_holidays", "get_electricity_prices"}


# ─────────────────────────────────────────────────────────────────────────────
# Group F — Kill-switch: ATC_RESPONSE_CACHE_ENABLED=false disables all caching
# ─────────────────────────────────────────────────────────────────────────────


class TestCacheKillSwitch:
    """Verify that ATC_RESPONSE_CACHE_ENABLED=false disables all cache operations."""

    @pytest.mark.asyncio
    async def test_kill_switch_disables_l1_check_in_execute_api(self) -> None:
        """With cache disabled, _execute_api_and_format does NOT check L1."""
        executor = _make_executor()
        api_result = APICallResult(
            success=True,
            status_code=200,
            response_data=_RESPONSE,
            error=None,
        )
        executor._api_caller.call = AsyncMock(return_value=api_result)

        mock_cache_store = AsyncMock()
        mock_cache_store.get_l1 = AsyncMock()  # Should NOT be called
        mock_cache_store.set_l1 = AsyncMock()  # Should NOT be called
        mock_cache_store.set_l2 = AsyncMock()  # Should NOT be called

        with (
            patch(
                "tool_classifier.workflows.api_tool_workflow.ATCCacheStore",
                return_value=mock_cache_store,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value="Formatted response.",
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                False,  # KILL SWITCH: disabled
            ),
        ):
            await executor._execute_api_and_format(
                chat_id=_CHAT_ID,
                endpoint=_ENDPOINT,
                collected_params=_PARAMS,
                user_query="What are the public holidays?",
                detected_language="en",
            )
            # Flush any background tasks
            await asyncio.sleep(0)

        # All cache methods must be skipped when kill-switch is off
        mock_cache_store.get_l1.assert_not_called()
        mock_cache_store.set_l1.assert_not_called()
        mock_cache_store.set_l2.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_disables_multi_cache_write(self) -> None:
        """With cache disabled, _execute_multi_api_and_format background write is skipped."""
        executor = _make_executor()

        ep1: Dict[str, Any] = {
            "name": "get_national_holidays",
            "description": "Returns holidays",
            "method": "GET",
            "url": "https://example.com/holidays",
            "cacheable": True,
            "cache_ttl_seconds": None,
        }
        ep2: Dict[str, Any] = {
            "name": "get_electricity_prices",
            "description": "Returns electricity prices",
            "method": "GET",
            "url": "https://example.com/electricity",
            "cacheable": True,
            "cache_ttl_seconds": None,
        }

        parallel_endpoints = [
            EndpointSessionState(
                endpoint=ep1, collected_params={"country": "EE"}, completed=True
            ),
            EndpointSessionState(
                endpoint=ep2, collected_params={"date": "2026-05-28"}, completed=True
            ),
        ]

        multi_result = MultiAPICallResult(
            results=[
                APICallResult(
                    success=True, status_code=200, response_data={"h": 1}, error=None
                ),
                APICallResult(
                    success=True, status_code=200, response_data={"p": 2}, error=None
                ),
            ],
            endpoints=[ep1, ep2],
        )

        mock_multi_caller = AsyncMock()
        mock_multi_caller.call_all = AsyncMock(return_value=multi_result)

        mock_cache_store = AsyncMock()
        mock_cache_store.set_l1 = AsyncMock()  # Should NOT be called
        mock_cache_store.set_l2 = AsyncMock()  # Should NOT be called

        with (
            patch(
                "tool_classifier.workflows.api_tool_workflow.MultiAPICaller",
                return_value=mock_multi_caller,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.ATCCacheStore",
                return_value=mock_cache_store,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value="Multi-endpoint formatted response.",
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                False,  # KILL SWITCH: disabled
            ),
        ):
            await executor._execute_multi_api_and_format(
                chat_id=_CHAT_ID,
                parallel_endpoints=parallel_endpoints,
                user_query="Show me holidays and electricity prices",
                detected_language="en",
            )
            # Flush any background tasks
            await asyncio.sleep(0)

        # No cache writes should occur when kill-switch is off
        mock_cache_store.set_l1.assert_not_called()
        mock_cache_store.set_l2.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_disables_cache_checks_in_compute_loop_step(
        self,
    ) -> None:
        """With cache disabled, _compute_loop_step does NOT check L1 or L2."""
        session_store = _make_session_store(session=None)
        executor = _make_executor(session_store)

        mock_cache_store = AsyncMock()
        mock_cache_store.get_l1 = AsyncMock()  # Should NOT be called
        mock_cache_store.get_l2 = AsyncMock()  # Should NOT be called
        mock_cache_store.set_l1 = AsyncMock()  # Should NOT be called
        mock_cache_store.set_l2 = AsyncMock()  # Should NOT be called

        # Create a loop mock to simulate normal agentic loop behavior
        loop_mock = _make_loop_mock()
        executor._build_agentic_loop = MagicMock(return_value=loop_mock)

        with (
            patch(
                "tool_classifier.workflows.api_tool_workflow.ATCCacheStore",
                return_value=mock_cache_store,
            ),
            patch(
                "tool_classifier.workflows.api_tool_workflow.FeatureFlags"
                ".ATC_RESPONSE_CACHE_ENABLED",
                False,  # KILL SWITCH: disabled
            ),
        ):
            step = await executor._compute_loop_step(
                _make_request(),
                context={"matched_endpoint": _ENDPOINT},
            )

        # With cache disabled, agentic loop should run normally (not cached_response)
        assert step.kind == "question"  # Normal loop behavior

        # No cache checks should have been performed
        mock_cache_store.get_l1.assert_not_called()
        mock_cache_store.get_l2.assert_not_called()
        # set_l1 and set_l2 should also not be called during loop execution
        mock_cache_store.set_l1.assert_not_called()
        mock_cache_store.set_l2.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_disables_invalidate_l2_on_intent_switch(self) -> None:
        """With cache disabled, intent switch does NOT call invalidate_l2."""
        # Active session for endpoint A (holidays)
        active_session = APIToolSession(
            chat_id=_CHAT_ID,
            state="collecting_params",
            selected_endpoint=_ENDPOINT_HOLIDAYS_CLS,
            collected_params={},
            turn_count=1,
            max_turns=5,
            awaiting_continuation=False,
            detected_language="en",
            original_query="holiday query",
        )
        session_store = _make_session_store(session=active_session)

        classifier = _make_classifier_instance(session_store)
        # New query matches endpoint B (weather) — a different endpoint
        classifier.api_tool_searcher.search = AsyncMock(
            return_value=[_make_api_tool_search_result(_ENDPOINT_WEATHER_CLS)]
        )

        mock_cache_store = AsyncMock()
        mock_cache_store.invalidate_l2 = AsyncMock()  # Should NOT be called

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags"
                ".API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.ATC_RESPONSE_CACHE_ENABLED",
                False,  # KILL SWITCH: disabled
            ),
            patch(
                "tool_classifier.classifier.ATCCacheStore",
                return_value=mock_cache_store,
            ),
        ):
            result = await classifier.classify(
                query="What is the weather in Tallinn?",
                conversation_history=[],
                language="en",
                request=_make_request("What is the weather in Tallinn?"),
            )

        # With cache disabled, invalidate_l2 should NOT be called
        mock_cache_store.invalidate_l2.assert_not_called()
        # But session delete should still occur (independent of cache)
        session_store.delete.assert_called_once_with(_CHAT_ID)
        assert result.workflow == WorkflowType.API_TOOL_CALLING
        assert result.metadata.get("matched_endpoint", {}).get("name") == "get_weather"

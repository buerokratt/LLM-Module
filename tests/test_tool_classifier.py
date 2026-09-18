"""Unit tests for ToolClassifier.classify() and helper methods.

Covers:
- Session resume shortcut (active session → API_TOOL_CALLING directly)
- Intent switch detection (different endpoint → delete old + create new)
- SERVICE_WORKFLOW_ENABLED=False flag → skips standard service search
- Embedding failure → graceful fallback to CONTEXT
- API tool match → API_TOOL_CALLING result
- No match anywhere → CONTEXT/RAG fallback
- _try_api_tool_classification(): precomputed embedding reuse, empty results → None
- _get_query_embedding(): success, exception → None
- Qdrant timeout during classification → fallback
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.session_models import APIToolSession
from models.request_models import OrchestrationRequest
from tool_classifier.classifier import ToolClassifier
from tool_classifier.enums import WorkflowType
from tool_classifier.intent_decomposer import IntentDecomposerModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_ID = "classifier-test-chat"

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
            "description": "ISO",
        }
    ],
    "cosine_score": 0.68,
    "rrf_score": 0.01,
    "confidence": "high",
}

_ENDPOINT_WEATHER = {
    "endpoint_id": "ep-weather",
    "name": "get_weather",
    "description": "Current weather for a city",
    "method": "GET",
    "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
    "params": [],
    "cosine_score": 0.62,
    "rrf_score": 0.009,
    "confidence": "high",
}


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


def _make_orchestration_service(
    embedding: Optional[List[float]] = None,
    session_store: Optional[Any] = None,
) -> MagicMock:
    svc = MagicMock()
    svc.create_embeddings_for_indexer.return_value = {
        "embeddings": [embedding or ([0.1] * 10)]
    }
    svc.session_store = session_store
    return svc


def _make_classifier(
    orchestration_service: Optional[MagicMock] = None,
) -> ToolClassifier:
    """Instantiate ToolClassifier with a patched httpx client."""
    if orchestration_service is None:
        orchestration_service = _make_orchestration_service()

    with patch("tool_classifier.classifier.httpx.AsyncClient"):
        return ToolClassifier(
            llm_manager=MagicMock(),
            orchestration_service=orchestration_service,
        )


def _make_api_tool_search_result(endpoint: Dict[str, Any]) -> MagicMock:
    """Return a MagicMock that looks like an APIToolSearchResult."""
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


def _make_active_session(endpoint: Dict[str, Any]) -> APIToolSession:
    return APIToolSession(
        chat_id=_CHAT_ID,
        state="collecting_params",
        selected_endpoint=endpoint,
        collected_params={},
        turn_count=1,
        max_turns=5,
        awaiting_continuation=False,
        detected_language="en",
        original_query="holiday query",
    )


# ---------------------------------------------------------------------------
# _get_query_embedding
# ---------------------------------------------------------------------------


class TestGetQueryEmbedding:
    def test_returns_embedding_on_success(self) -> None:
        svc = MagicMock()
        svc.create_embeddings_for_indexer.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        classifier = _make_classifier(_make_orchestration_service())
        classifier.orchestration_service = svc

        result = classifier._get_query_embedding("holiday")

        assert result == [0.1, 0.2, 0.3]

    def test_returns_none_on_exception(self) -> None:
        svc = MagicMock()
        svc.create_embeddings_for_indexer.side_effect = RuntimeError("API down")
        classifier = _make_classifier()
        classifier.orchestration_service = svc

        result = classifier._get_query_embedding("holiday")

        assert result is None

    def test_returns_none_when_no_embeddings_returned(self) -> None:
        svc = MagicMock()
        svc.create_embeddings_for_indexer.return_value = {"embeddings": []}
        classifier = _make_classifier()
        classifier.orchestration_service = svc

        result = classifier._get_query_embedding("holiday")

        assert result is None

    def test_returns_none_when_service_is_none(self) -> None:
        classifier = _make_classifier()
        classifier.orchestration_service = None

        result = classifier._get_query_embedding("holiday")

        assert result is None


# ---------------------------------------------------------------------------
# _try_api_tool_classification
# ---------------------------------------------------------------------------


class TestTryApiToolClassification:
    @pytest.mark.asyncio
    async def test_returns_classification_when_match_found(self) -> None:
        classifier = _make_classifier()
        mock_result = _make_api_tool_search_result(_ENDPOINT_HOLIDAYS)
        classifier.api_tool_searcher.search = AsyncMock(return_value=[mock_result])

        request = _make_request("public holidays in Estonia")
        result = await classifier._try_api_tool_classification(
            "public holidays", request
        )

        assert result is not None
        assert result.workflow == WorkflowType.API_TOOL_CALLING
        assert result.metadata.get("matched_endpoint") is not None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self) -> None:
        classifier = _make_classifier()
        classifier.api_tool_searcher.search = AsyncMock(return_value=[])

        result = await classifier._try_api_tool_classification("some query", None)

        assert result is None

    @pytest.mark.asyncio
    async def test_reuses_precomputed_embedding(self) -> None:
        classifier = _make_classifier()
        mock_result = _make_api_tool_search_result(_ENDPOINT_HOLIDAYS)
        classifier.api_tool_searcher.search = AsyncMock(return_value=[mock_result])

        precomputed = [0.5] * 10
        await classifier._try_api_tool_classification(
            "public holidays",
            None,
            precomputed_embedding=precomputed,
        )

        call_kwargs = classifier.api_tool_searcher.search.call_args.kwargs
        assert call_kwargs.get("precomputed_embedding") == precomputed

    @pytest.mark.asyncio
    async def test_returns_none_on_search_exception(self) -> None:
        classifier = _make_classifier()
        classifier.api_tool_searcher.search = AsyncMock(
            side_effect=RuntimeError("Qdrant unavailable")
        )

        result = await classifier._try_api_tool_classification("holiday query", None)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_feature_flag_disabled(self) -> None:
        classifier = _make_classifier()
        classifier.api_tool_searcher.search = AsyncMock(
            return_value=[_make_api_tool_search_result(_ENDPOINT_HOLIDAYS)]
        )

        with patch(
            "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
            False,
        ):
            result = await classifier._try_api_tool_classification("holiday", None)

        assert result is None
        classifier.api_tool_searcher.search.assert_not_called()


# ---------------------------------------------------------------------------
# classify() — session resume shortcut
# ---------------------------------------------------------------------------


class TestClassifySessionResumeShortcut:
    @pytest.mark.asyncio
    async def test_active_session_returns_api_tool_calling(self) -> None:
        """When active session exists and query is not a different endpoint → resume."""
        session = _make_active_session(_ENDPOINT_HOLIDAYS)
        session_store = AsyncMock()
        session_store.get = AsyncMock(return_value=session)

        svc = _make_orchestration_service(session_store=session_store)
        classifier = _make_classifier(svc)
        # api_tool_searcher returns same endpoint → no intent switch
        classifier.api_tool_searcher.search = AsyncMock(
            return_value=[_make_api_tool_search_result(_ENDPOINT_HOLIDAYS)]
        )

        request = _make_request("EE")
        with patch(
            "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
            True,
        ):
            result = await classifier.classify(
                query="EE",
                language="en",
                request=request,
            )

        assert result.workflow == WorkflowType.API_TOOL_CALLING
        assert result.metadata.get("reason") == "active_session_resume"

    @pytest.mark.asyncio
    async def test_no_active_session_does_not_short_circuit(self) -> None:
        """No existing session → proceeds with normal classification."""
        session_store = AsyncMock()
        session_store.get = AsyncMock(return_value=None)

        svc = _make_orchestration_service(session_store=session_store)
        svc.create_embeddings_for_indexer.side_effect = RuntimeError("no embedding")
        classifier = _make_classifier(svc)

        request = _make_request("some query")
        with patch(
            "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
            True,
        ):
            result = await classifier.classify(
                query="some query",
                language="en",
                request=request,
            )

        # Embedding failure → CONTEXT fallback (normal path)
        assert result.workflow == WorkflowType.CONTEXT


# ---------------------------------------------------------------------------
# classify() — intent switch detection
# ---------------------------------------------------------------------------


class TestClassifyIntentSwitch:
    @pytest.mark.asyncio
    async def test_different_endpoint_match_deletes_old_session(self) -> None:
        """When new query matches a DIFFERENT endpoint → delete session, return new."""
        session = _make_active_session(_ENDPOINT_HOLIDAYS)
        session_store = AsyncMock()
        session_store.get = AsyncMock(return_value=session)
        session_store.delete = AsyncMock()

        svc = _make_orchestration_service(session_store=session_store)
        classifier = _make_classifier(svc)

        # New query matches WEATHER endpoint (different from active HOLIDAYS session)
        weather_result = _make_api_tool_search_result(_ENDPOINT_WEATHER)
        classifier.api_tool_searcher.search = AsyncMock(return_value=[weather_result])

        request = _make_request("What is the weather in Tallinn?")
        with patch(
            "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
            True,
        ):
            result = await classifier.classify(
                query="What is the weather in Tallinn?",
                language="en",
                request=request,
            )

        session_store.delete.assert_called_once_with(_CHAT_ID)
        assert result.workflow == WorkflowType.API_TOOL_CALLING
        assert result.metadata.get("matched_endpoint", {}).get("name") == "get_weather"


# ---------------------------------------------------------------------------
# classify() — SERVICE_WORKFLOW_ENABLED = False
# ---------------------------------------------------------------------------


class TestClassifyServiceWorkflowDisabled:
    @pytest.mark.asyncio
    async def test_api_tool_match_when_service_disabled(self) -> None:
        """SERVICE_WORKFLOW_ENABLED=False + API tool match → API_TOOL_CALLING."""
        svc = _make_orchestration_service(session_store=None)
        classifier = _make_classifier(svc)
        mock_result = _make_api_tool_search_result(_ENDPOINT_HOLIDAYS)
        classifier.api_tool_searcher.search = AsyncMock(return_value=[mock_result])

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
                False,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
        ):
            result = await classifier.classify(
                query="public holidays",
                language="en",
                request=_make_request("public holidays"),
            )

        assert result.workflow == WorkflowType.API_TOOL_CALLING

    @pytest.mark.asyncio
    async def test_context_fallback_when_service_disabled_and_no_api_match(
        self,
    ) -> None:
        """SERVICE_WORKFLOW_ENABLED=False + no API match → CONTEXT."""
        svc = _make_orchestration_service(session_store=None)
        classifier = _make_classifier(svc)
        classifier.api_tool_searcher.search = AsyncMock(return_value=[])

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
                False,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
        ):
            result = await classifier.classify(
                query="tell me a joke",
                language="en",
                request=_make_request("tell me a joke"),
            )

        assert result.workflow == WorkflowType.CONTEXT
        assert result.metadata.get("reason") == "service_workflow_disabled"


# ---------------------------------------------------------------------------
# classify() — embedding failure
# ---------------------------------------------------------------------------


class TestClassifyEmbeddingFailure:
    @pytest.mark.asyncio
    async def test_embedding_failure_falls_back_to_context(self) -> None:
        svc = _make_orchestration_service(session_store=None)
        svc.create_embeddings_for_indexer.side_effect = RuntimeError("embedding failed")
        classifier = _make_classifier(svc)

        with patch(
            "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
            True,
        ):
            result = await classifier.classify(
                query="public holidays",
                language="en",
            )

        assert result.workflow == WorkflowType.CONTEXT
        assert result.metadata.get("reason") == "embedding_generation_failed"


# ---------------------------------------------------------------------------
# classify() — Qdrant unavailable / timeout
# ---------------------------------------------------------------------------


class TestClassifyQdrantTimeout:
    @pytest.mark.asyncio
    async def test_qdrant_timeout_falls_back_to_context(self) -> None:
        import httpx

        svc = _make_orchestration_service(session_store=None)
        svc.create_embeddings_for_indexer.return_value = {"embeddings": [[0.1] * 10]}
        classifier = _make_classifier(svc)

        # Both dense and hybrid calls timeout
        classifier._qdrant_client.post = AsyncMock(
            side_effect=httpx.ReadTimeout("timeout")
        )
        classifier._qdrant_client.get = AsyncMock(
            side_effect=httpx.ReadTimeout("timeout")
        )

        with patch(
            "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
            True,
        ):
            # Also mock api_tool_searcher to return empty (so it falls all the way to CONTEXT)
            classifier.api_tool_searcher.search = AsyncMock(return_value=[])
            result = await classifier.classify(
                query="public holidays",
                language="en",
            )

        assert result.workflow == WorkflowType.CONTEXT


# ---------------------------------------------------------------------------
# IntentDecomposerModule
# ---------------------------------------------------------------------------


class TestIntentDecomposer:
    """Unit tests for IntentDecomposerModule.forward() and .decompose().

    The DSPy predictor is replaced with a MagicMock so no real LLM is called.
    """

    def _make_module_with_prediction(
        self,
        mode: str,
        sub_queries: str,
    ) -> IntentDecomposerModule:
        module = IntentDecomposerModule()
        mock_pred = MagicMock()
        mock_pred.mode = mode
        mock_pred.sub_queries = sub_queries
        module.predictor = MagicMock(return_value=mock_pred)
        return module

    def test_forward_single_mode_returns_single(self) -> None:
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction("single", "[]")
        result = module.forward("What are the public holidays in Estonia?")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "single"
        assert result.sub_queries == []

    def test_forward_parallel_mode_returns_sub_queries(self) -> None:
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction(
            "parallel",
            '["public holidays in Estonia", "weather in Tallinn"]',
        )
        result = module.forward(
            "What are the public holidays in Estonia AND the weather in Tallinn?"
        )

        assert isinstance(result, DecompositionResult)
        assert result.mode == "parallel"
        assert len(result.sub_queries) == 2
        assert "public holidays in Estonia" in result.sub_queries

    def test_forward_caps_sub_queries_at_max_endpoints(self) -> None:
        """sub_queries exceeding MULTI_API_MAX_ENDPOINTS are truncated."""
        from tool_classifier.intent_decomposer import (
            DecompositionResult,
            IntentDecomposerModule,
        )
        from tool_classifier.constants import MULTI_API_MAX_ENDPOINTS

        module = IntentDecomposerModule()
        # Build a prediction with more sub-queries than the cap allows
        over_cap = ["query " + str(i) for i in range(MULTI_API_MAX_ENDPOINTS + 2)]
        import json

        mock_pred = MagicMock()
        mock_pred.mode = "parallel"
        mock_pred.sub_queries = json.dumps(over_cap)
        module.predictor = MagicMock(return_value=mock_pred)

        result = module.forward("many intents query")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "parallel"
        assert len(result.sub_queries) == MULTI_API_MAX_ENDPOINTS

    def test_forward_unexpected_mode_falls_back_to_single(self) -> None:
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction("unknown_value", "[]")
        result = module.forward("some query")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "single"
        assert result.sub_queries == []

    def test_forward_parallel_with_fewer_than_2_sub_queries_falls_back(self) -> None:
        """mode=parallel but only 1 sub-query parsed → conservative fallback to single."""
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction("parallel", '["only one query"]')
        result = module.forward("something")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "single"

    def test_forward_predictor_exception_falls_back_to_single(self) -> None:
        """Any exception from the DSPy predictor → conservative single fallback."""
        from tool_classifier.intent_decomposer import (
            DecompositionResult,
            IntentDecomposerModule,
        )

        module = IntentDecomposerModule()
        module.predictor = MagicMock(side_effect=RuntimeError("LLM unavailable"))

        result = module.forward("multi intent query")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "single"

    def test_forward_parallel_with_invalid_json_falls_back_to_single(self) -> None:
        """Invalid JSON in sub_queries → falls back to mode=single."""
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction("parallel", "not valid json")
        result = module.forward("holidays and weather")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "single"

    def test_forward_markdown_fenced_json_parsed_correctly(self) -> None:
        """sub_queries wrapped in markdown code fences are unwrapped before JSON parse."""
        from tool_classifier.intent_decomposer import DecompositionResult

        fenced = '```json\n["query A", "query B"]\n```'
        module = self._make_module_with_prediction("parallel", fenced)
        result = module.forward("query")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "parallel"
        assert result.sub_queries == ["query A", "query B"]

    @pytest.mark.asyncio
    async def test_decompose_async_wraps_forward(self) -> None:
        """.decompose() is the async wrapper — result matches .forward() output."""
        from tool_classifier.intent_decomposer import DecompositionResult

        module = self._make_module_with_prediction("parallel", '["sub A", "sub B"]')

        with patch(
            "tool_classifier.intent_decomposer.asyncio.to_thread",
            new_callable=AsyncMock,
        ) as mock_thread:
            mock_thread.return_value = DecompositionResult(
                mode="parallel", sub_queries=["sub A", "sub B"]
            )
            result = await module.decompose("two intents")

        assert isinstance(result, DecompositionResult)
        assert result.mode == "parallel"
        assert result.sub_queries == ["sub A", "sub B"]


# ---------------------------------------------------------------------------
# classify() — MULTI_INTENT_ENABLED feature flag toggle
# ---------------------------------------------------------------------------


class TestClassifyMultiIntentFeatureFlag:
    """Verify the MULTI_INTENT_ENABLED flag gates the parallel decomposition path."""

    @pytest.mark.asyncio
    async def test_multi_intent_disabled_suppresses_hint_result(self) -> None:
        """When MULTI_INTENT_ENABLED=False a multi_intent_hint result is suppressed
        and the classifier falls through to CONTEXT/RAG."""
        from tool_classifier.api_semantic_searcher import APIToolSearchResult

        svc = _make_orchestration_service(session_store=None)
        classifier = _make_classifier(svc)

        # Build a result with multi_intent_hint=True (disambiguator rejected all)
        hint_result = MagicMock(spec=APIToolSearchResult)
        hint_result.endpoint_id = "ep-holidays"
        hint_result.name = "get_public_holidays"
        hint_result.description = "Returns public holidays"
        hint_result.method = "GET"
        hint_result.url = "https://openholidaysapi.org/PublicHolidays"
        hint_result.params = []
        hint_result.cosine_score = 0.55
        hint_result.rrf_score = 0.01
        hint_result.confidence = "medium"
        hint_result.llm_validated = False
        hint_result.multi_intent_hint = True
        hint_result.to_dict.return_value = {"endpoint_id": "ep-holidays"}

        classifier.api_tool_searcher.search = AsyncMock(return_value=[hint_result])

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.MULTI_INTENT_ENABLED",
                False,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
                False,
            ),
        ):
            result = await classifier.classify(
                query="public holidays AND weather",
                conversation_history=[],
                language="en",
                request=_make_request("public holidays AND weather"),
            )

        assert result.workflow == WorkflowType.CONTEXT

    @pytest.mark.asyncio
    async def test_multi_intent_enabled_triggers_decomposer_on_ambiguous_band(
        self,
    ) -> None:
        """MULTI_INTENT_ENABLED=True + cosine in ambiguous band → IntentDecomposer runs."""
        from tool_classifier.api_semantic_searcher import APIToolSearchResult
        from tool_classifier.intent_decomposer import DecompositionResult

        svc = _make_orchestration_service(session_store=None)
        classifier = _make_classifier(svc)

        # A result in the ambiguous band (not llm_validated, not multi_intent_hint)
        ambiguous_result = MagicMock(spec=APIToolSearchResult)
        ambiguous_result.endpoint_id = "ep-holidays"
        ambiguous_result.name = "get_public_holidays"
        ambiguous_result.description = "Returns public holidays"
        ambiguous_result.method = "GET"
        ambiguous_result.url = "https://openholidaysapi.org/PublicHolidays"
        ambiguous_result.params = []
        ambiguous_result.cosine_score = 0.50  # in [0.40, 0.60) band
        ambiguous_result.rrf_score = 0.01
        ambiguous_result.confidence = "medium"
        ambiguous_result.llm_validated = False
        ambiguous_result.multi_intent_hint = False
        ambiguous_result.to_dict.return_value = {
            "endpoint_id": "ep-holidays",
            "name": "get_public_holidays",
            "description": "Returns public holidays",
            "method": "GET",
            "url": "https://openholidaysapi.org/PublicHolidays",
            "params": [],
            "cosine_score": 0.50,
            "rrf_score": 0.01,
            "confidence": "medium",
        }

        classifier.api_tool_searcher.search = AsyncMock(return_value=[ambiguous_result])

        # IntentDecomposer returns single → falls through to single-endpoint path
        decomposer_result = DecompositionResult(mode="single", sub_queries=[])
        classifier.intent_decomposer.decompose = AsyncMock(
            return_value=decomposer_result
        )

        with (
            patch(
                "tool_classifier.classifier.FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.MULTI_INTENT_ENABLED",
                True,
            ),
            patch(
                "tool_classifier.classifier.FeatureFlags.SERVICE_WORKFLOW_ENABLED",
                False,
            ),
        ):
            result = await classifier.classify(
                query="public holidays AND weather",
                conversation_history=[],
                language="en",
                request=_make_request("public holidays AND weather"),
            )

        # Decomposer was consulted
        classifier.intent_decomposer.decompose.assert_awaited_once()
        # Single mode → normal API_TOOL_CALLING result
        assert result.workflow == WorkflowType.API_TOOL_CALLING

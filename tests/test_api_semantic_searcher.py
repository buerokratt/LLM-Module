"""Unit tests for APISemanticSearcher and EndpointDisambiguatorModule.

Covers:
- search(): high-confidence shortcut (cosine >= 0.60, gap >= 0.05)
- search(): single medium-confidence with large gap → direct return
- search(): multiple medium → LLM disambiguation
- search(): below threshold → empty list
- search(): precomputed embedding reuse (no extra embedding call)
- _dense_search(): deduplication by endpoint_id, sorted by cosine
- _dense_search(): empty Qdrant response → []
- _hybrid_search(): RRF fusion, fallback to dense when hybrid empty
- EndpointDisambiguatorModule.forward(): valid winner, "none" → None, unknown id, DSPy exception
"""

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tool_classifier.api_semantic_searcher import (
    APISemanticSearcher,
    DisambiguationResult,
    EndpointDisambiguatorModule,
)
from tool_classifier.constants import (
    API_TOOL_HIGH_CONFIDENCE_THRESHOLD,
    API_TOOL_MIN_THRESHOLD,
    API_TOOL_SCORE_GAP_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EP_HOLIDAYS = {
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
        }
    ],
}

_EP_WEATHER = {
    "endpoint_id": "ep-weather",
    "name": "get_weather",
    "description": "Current weather for a city",
    "method": "GET",
    "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
    "params": [],
}


def _make_qdrant_dense_response(
    points: List[Dict[str, Any]],
    status_code: int = 200,
) -> MagicMock:
    """Build a mock httpx response for Qdrant dense search."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"result": {"points": points}}
    mock_resp.text = ""
    return mock_resp


def _make_qdrant_hybrid_response(
    points: List[Dict[str, Any]],
    status_code: int = 200,
) -> MagicMock:
    """Build a mock httpx response for Qdrant hybrid search."""
    return _make_qdrant_dense_response(points, status_code)


def _make_count_response(count: int = 10, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"result": {"points_count": count}}
    return mock_resp


def _point(endpoint: Dict[str, Any], score: float) -> Dict[str, Any]:
    return {"payload": {**endpoint}, "score": score}


def _make_embedding_service(embedding: Optional[List[float]] = None) -> MagicMock:
    svc = MagicMock()
    svc.create_embeddings_for_indexer.return_value = {
        "embeddings": [embedding or ([0.1] * 10)]
    }
    return svc


def _make_searcher(
    qdrant_client: AsyncMock,
    embedding: Optional[List[float]] = None,
    disambiguator: Optional[MagicMock] = None,
) -> APISemanticSearcher:
    svc = _make_embedding_service(embedding)
    return APISemanticSearcher(
        embedding_service=svc,
        qdrant_client=qdrant_client,
        disambiguator=disambiguator,
    )


def _make_async_qdrant_client(
    dense_response: MagicMock,
    hybrid_response: Optional[MagicMock] = None,
    count_response: Optional[MagicMock] = None,
) -> AsyncMock:
    """Return an AsyncMock pretending to be an httpx.AsyncClient."""
    client = AsyncMock()
    client.post = AsyncMock(
        side_effect=[
            dense_response,
            hybrid_response or dense_response,
        ]
    )
    # Hybrid search calls GET for collection info first
    count_resp = count_response or _make_count_response(10)
    client.get = AsyncMock(return_value=count_resp)
    return client


# ---------------------------------------------------------------------------
# EndpointDisambiguatorModule
# ---------------------------------------------------------------------------


class TestEndpointDisambiguatorModule:
    def _make_module(self, best_endpoint_id: str) -> EndpointDisambiguatorModule:
        module = EndpointDisambiguatorModule()
        mock_result = MagicMock()
        mock_result.best_endpoint_id = best_endpoint_id
        module.predictor = MagicMock(return_value=mock_result)
        return module

    def test_returns_winning_endpoint_id(self) -> None:
        module = self._make_module("ep-holidays")
        candidates = [
            {
                "endpoint_id": "ep-holidays",
                "name": "get_public_holidays",
                "description": "Returns public holidays",
                "cosine_score": 0.52,
            },
            {
                "endpoint_id": "ep-weather",
                "name": "get_weather",
                "description": "Returns weather",
                "cosine_score": 0.48,
            },
        ]
        result = module.forward("When is the next holiday?", candidates)
        assert result.winner_id == "ep-holidays"

    def test_returns_none_when_predictor_returns_none_string(self) -> None:
        module = self._make_module("none")
        candidates = [
            {
                "endpoint_id": "ep-holidays",
                "name": "get_public_holidays",
                "description": "Returns public holidays",
                "cosine_score": 0.45,
            },
        ]
        result = module.forward("Tell me a joke", candidates)
        assert result.winner_id is None

    def test_returns_none_when_predictor_returns_none_uppercase(self) -> None:
        module = self._make_module("NONE")
        candidates = [
            {
                "endpoint_id": "ep-holidays",
                "name": "get_public_holidays",
                "description": "Returns public holidays",
                "cosine_score": 0.45,
            },
        ]
        result = module.forward("Tell me a joke", candidates)
        assert result.winner_id is None

    def test_returns_none_on_dspy_exception(self) -> None:
        module = EndpointDisambiguatorModule()
        module.predictor = MagicMock(side_effect=RuntimeError("DSPy LM unavailable"))
        candidates = [
            {
                "endpoint_id": "ep-holidays",
                "name": "get_public_holidays",
                "description": "Returns public holidays",
                "cosine_score": 0.50,
            },
        ]
        result = module.forward("Which holidays?", candidates)
        assert result.winner_id is None

    def test_strips_whitespace_from_endpoint_id(self) -> None:
        module = self._make_module("  ep-holidays  ")
        candidates = [
            {
                "endpoint_id": "ep-holidays",
                "name": "get_public_holidays",
                "description": "Returns public holidays",
                "cosine_score": 0.52,
            },
        ]
        result = module.forward("Holidays?", candidates)
        assert result.winner_id == "ep-holidays"


# ---------------------------------------------------------------------------
# APISemanticSearcher._dense_search
# ---------------------------------------------------------------------------


class TestDenseSearch:
    @pytest.mark.asyncio
    async def test_returns_deduplicated_by_endpoint_id(self) -> None:
        """Multiple points for the same endpoint_id should collapse to best score."""
        points = [
            _point({**_EP_HOLIDAYS}, 0.75),
            _point({**_EP_HOLIDAYS}, 0.65),  # lower score, same endpoint
            _point({**_EP_WEATHER}, 0.55),
        ]
        dense_resp = _make_qdrant_dense_response(points)
        # Supply a second response for the hybrid call that isn't used in _dense_search
        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher._dense_search([0.1] * 10, top_k=5)

        assert len(results) == 2
        holidays = next(r for r in results if r["endpoint_id"] == "ep-holidays")
        assert holidays["cosine_score"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_results_sorted_by_cosine_descending(self) -> None:
        points = [
            _point({**_EP_WEATHER}, 0.50),
            _point({**_EP_HOLIDAYS}, 0.80),
        ]
        dense_resp = _make_qdrant_dense_response(points)
        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher._dense_search([0.1] * 10, top_k=5)

        assert results[0]["cosine_score"] >= results[1]["cosine_score"]

    @pytest.mark.asyncio
    async def test_empty_qdrant_response_returns_empty_list(self) -> None:
        dense_resp = _make_qdrant_dense_response([])
        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher._dense_search([0.1] * 10, top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_non_200_response_returns_empty_list(self) -> None:
        dense_resp = _make_qdrant_dense_response([], status_code=503)
        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher._dense_search([0.1] * 10, top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_list(self) -> None:
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.ReadTimeout("timeout"))

        searcher = _make_searcher(client)
        results = await searcher._dense_search([0.1] * 10, top_k=5)

        assert results == []


# ---------------------------------------------------------------------------
# APISemanticSearcher._hybrid_search
# ---------------------------------------------------------------------------


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_returns_rrf_fused_results(self) -> None:
        from tool_classifier.sparse_encoder import SparseVector

        hybrid_points = [
            {**_point({**_EP_HOLIDAYS}, 0.012)},  # RRF score
        ]
        count_resp = _make_count_response(5)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(return_value=hybrid_resp)

        searcher = _make_searcher(client)
        sparse = SparseVector(indices=[1, 2], values=[1.0, 2.0])
        results = await searcher._hybrid_search([0.1] * 10, sparse, top_k=5)

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_empty_collection_returns_empty_list(self) -> None:
        from tool_classifier.sparse_encoder import SparseVector

        count_resp = _make_count_response(0)
        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock()  # should not be called

        searcher = _make_searcher(client)
        sparse = SparseVector(indices=[1], values=[1.0])
        results = await searcher._hybrid_search([0.1] * 10, sparse, top_k=5)

        assert results == []


# ---------------------------------------------------------------------------
# APISemanticSearcher.search — confidence routing
# ---------------------------------------------------------------------------


class TestSearchHighConfidence:
    @pytest.mark.asyncio
    async def test_high_confidence_returns_single_result_without_disambiguation(
        self,
    ) -> None:
        """cosine >= 0.60 and gap >= 0.05 → high confidence, no LLM needed."""
        high_score = API_TOOL_HIGH_CONFIDENCE_THRESHOLD + 0.05  # e.g. 0.65
        low_score = high_score - API_TOOL_SCORE_GAP_THRESHOLD - 0.01  # e.g. 0.59

        dense_points = [
            _point({**_EP_HOLIDAYS}, high_score),
            _point({**_EP_WEATHER}, low_score),
        ]
        hybrid_points = [
            _point({**_EP_HOLIDAYS, "rrf_score": 0.015}, 0.015),
        ]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        mock_disambiguator = MagicMock()
        searcher = _make_searcher(client, disambiguator=mock_disambiguator)

        results = await searcher.search("public holidays in Estonia")

        assert len(results) == 1
        assert results[0].endpoint_id == "ep-holidays"
        assert results[0].confidence == "high"
        # Disambiguator must NOT be called for high-confidence matches
        mock_disambiguator.assert_not_called()

    @pytest.mark.asyncio
    async def test_high_confidence_sets_correct_scores(self) -> None:
        high_score = 0.72
        low_score = 0.45

        dense_points = [
            _point({**_EP_HOLIDAYS}, high_score),
            _point({**_EP_WEATHER}, low_score),
        ]
        hybrid_points = [_point({**_EP_HOLIDAYS}, 0.014)]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        searcher = _make_searcher(client)
        results = await searcher.search("Estonian public holidays")

        assert results[0].cosine_score == pytest.approx(high_score)


class TestSearchMediumConfidence:
    @pytest.mark.asyncio
    async def test_single_medium_with_large_gap_returns_directly(self) -> None:
        """Single result with medium cosine and large enough gap → no disambiguation."""
        cosine = API_TOOL_MIN_THRESHOLD + 0.10  # e.g. 0.50
        # No second result → gap = cosine - 0.0 = cosine (>= 0.05)

        dense_points = [_point({**_EP_HOLIDAYS}, cosine)]
        hybrid_points = [_point({**_EP_HOLIDAYS}, 0.012)]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        mock_disambiguator = MagicMock()
        searcher = _make_searcher(client, disambiguator=mock_disambiguator)

        results = await searcher.search("holidays")

        assert len(results) == 1
        assert results[0].confidence == "medium"
        mock_disambiguator.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_medium_triggers_disambiguation(self) -> None:
        """Two medium-confidence results → LLM disambiguation called."""
        cos_a = API_TOOL_MIN_THRESHOLD + 0.08  # 0.48
        cos_b = API_TOOL_MIN_THRESHOLD + 0.02  # 0.42

        dense_points = [
            _point({**_EP_HOLIDAYS}, cos_a),
            _point({**_EP_WEATHER}, cos_b),
        ]
        hybrid_points = [
            _point({**_EP_HOLIDAYS}, 0.012),
            _point({**_EP_WEATHER}, 0.010),
        ]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        # Inject our disambiguator — searcher calls self._disambiguator(query, candidates)
        # which in turn calls forward() via __call__
        async_disambiguator = MagicMock()
        async_disambiguator.forward = MagicMock(
            return_value=DisambiguationResult(winner_id="ep-holidays")
        )

        searcher = _make_searcher(client, disambiguator=async_disambiguator)

        # Patch asyncio.to_thread so it runs the callable synchronously in tests
        with patch(
            "tool_classifier.api_semantic_searcher.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=DisambiguationResult(winner_id="ep-holidays"),
        ):
            results = await searcher.search("something ambiguous")

        assert len(results) == 1
        assert results[0].endpoint_id == "ep-holidays"

    @pytest.mark.asyncio
    async def test_disambiguation_rejects_all_returns_empty(self) -> None:
        """Disambiguator returns None → no match."""
        cos_a = API_TOOL_MIN_THRESHOLD + 0.05
        cos_b = API_TOOL_MIN_THRESHOLD + 0.01

        dense_points = [
            _point({**_EP_HOLIDAYS}, cos_a),
            _point({**_EP_WEATHER}, cos_b),
        ]
        hybrid_points = [
            _point({**_EP_HOLIDAYS}, 0.011),
            _point({**_EP_WEATHER}, 0.009),
        ]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        searcher = _make_searcher(client)

        with patch(
            "tool_classifier.api_semantic_searcher.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=None,  # Disambiguator says "none"
        ):
            results = await searcher.search("tell me a joke")

        assert results == []

    @pytest.mark.asyncio
    async def test_disambiguation_rejects_all_multi_candidates_returns_top_with_hint(
        self,
    ) -> None:
        """Disambiguator returns None for >1 medium candidates → top candidate returned
        with multi_intent_hint=True and llm_validated=False so IntentDecomposer gate
        can run in the classifier."""
        cos_a = API_TOOL_MIN_THRESHOLD + 0.08  # higher cosine → becomes 'top'
        cos_b = API_TOOL_MIN_THRESHOLD + 0.02

        dense_points = [
            _point({**_EP_HOLIDAYS}, cos_a),
            _point({**_EP_WEATHER}, cos_b),
        ]
        hybrid_points = [
            _point({**_EP_HOLIDAYS}, 0.012),
            _point({**_EP_WEATHER}, 0.009),
        ]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        searcher = _make_searcher(client)

        # asyncio.to_thread is called twice:
        #   1st call → _get_query_embedding → must return a valid embedding vector
        #   2nd call → disambiguator.forward  → must return None ("none" response)
        precomputed = [0.1] * 10
        _call_count = 0

        async def _to_thread_side_effect(fn: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal _call_count
            _call_count += 1
            if _call_count == 1:
                return precomputed  # embedding call
            return DisambiguationResult(
                winner_id=None
            )  # disambiguator call → rejects all candidates

        with patch(
            "tool_classifier.api_semantic_searcher.asyncio.to_thread",
            side_effect=_to_thread_side_effect,
        ):
            results = await searcher.search("holidays AND weather")

        # Must return exactly one result — the top cosine candidate
        assert len(results) == 1
        top = results[0]
        # Top candidate by cosine score is ep-holidays
        assert top.endpoint_id == "ep-holidays"
        # NOT llm_validated — disambiguator explicitly rejected it
        assert top.llm_validated is False
        # multi_intent_hint signals the classifier to try IntentDecomposer
        assert top.multi_intent_hint is True


class TestSearchBelowThreshold:
    @pytest.mark.asyncio
    async def test_below_min_threshold_returns_empty(self) -> None:
        """cosine < API_TOOL_MIN_THRESHOLD → no match."""
        low_score = API_TOOL_MIN_THRESHOLD - 0.05  # e.g. 0.35

        dense_points = [_point({**_EP_HOLIDAYS}, low_score)]
        dense_resp = _make_qdrant_dense_response(dense_points)

        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher.search("completely unrelated query")

        assert results == []

    @pytest.mark.asyncio
    async def test_no_dense_results_returns_empty(self) -> None:
        dense_resp = _make_qdrant_dense_response([])
        client = AsyncMock()
        client.post = AsyncMock(return_value=dense_resp)

        searcher = _make_searcher(client)
        results = await searcher.search("some query")

        assert results == []

    @pytest.mark.asyncio
    async def test_embedding_failure_returns_empty(self) -> None:
        client = AsyncMock()
        svc = MagicMock()
        svc.create_embeddings_for_indexer.side_effect = RuntimeError("embedding failed")

        searcher = APISemanticSearcher(
            embedding_service=svc,
            qdrant_client=client,
        )
        results = await searcher.search("public holidays")

        assert results == []


class TestPrecomputedEmbedding:
    @pytest.mark.asyncio
    async def test_precomputed_embedding_skips_embedding_call(self) -> None:
        """When precomputed_embedding is supplied, no embedding API call is made."""
        precomputed = [0.5] * 10
        score = API_TOOL_HIGH_CONFIDENCE_THRESHOLD + 0.05

        dense_points = [_point({**_EP_HOLIDAYS}, score)]
        hybrid_points = [_point({**_EP_HOLIDAYS}, 0.015)]

        dense_resp = _make_qdrant_dense_response(dense_points)
        hybrid_resp = _make_qdrant_hybrid_response(hybrid_points)
        count_resp = _make_count_response(10)

        client = AsyncMock()
        client.get = AsyncMock(return_value=count_resp)
        client.post = AsyncMock(side_effect=[dense_resp, hybrid_resp])

        svc = MagicMock()
        searcher = APISemanticSearcher(
            embedding_service=svc,
            qdrant_client=client,
        )

        results = await searcher.search(
            "public holidays",
            precomputed_embedding=precomputed,
        )

        assert len(results) == 1
        # Embedding service must not be called
        svc.create_embeddings_for_indexer.assert_not_called()

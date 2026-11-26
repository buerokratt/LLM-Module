"""Unit tests for issue #589: test and production connections query one shared
knowledge base.

Covers:
- /orchestrate/test accepts the connection's vault_uuid (Ruuter sends a UUID;
  typed as int every test-connection query was rejected with a 422)
- provider detection never falls back to a collection with another vector size
- the shared BM25 index: kept when the knowledge base is empty at startup,
  shared by test and production retrievers, deduplicated across collections,
  emptied when the knowledge base is, consistent while a refresh runs
- querying through both connections at the same time: a transient Qdrant
  error on one health check, and both retrievals running in parallel with
  their own connection's embeddings
"""

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.contextual_retrieval.bm25_search import SmartBM25Search
from src.contextual_retrieval.config import ContextualRetrievalConfig
from src.contextual_retrieval.contextual_retriever import ContextualRetriever
from src.contextual_retrieval.provider_detection import DynamicProviderDetection
from src.models.request_models import TestOrchestrationRequest

AZURE = "contextual_chunks_azure"
AWS = "contextual_chunks_aws"
VAULT_UUID = "5f0c9a52-3a3e-4f0e-9d55-8f1d4b1a2c3d"


# ---------------------------------------------------------------------------
# /orchestrate/test request model
# ---------------------------------------------------------------------------


def test_test_request_accepts_the_connections_vault_uuid() -> None:
    request = TestOrchestrationRequest.model_validate(
        {"message": "hi", "environment": "testing", "connectionId": VAULT_UUID}
    )
    assert request.connectionId == VAULT_UUID


def test_test_request_connection_id_is_optional() -> None:
    request = TestOrchestrationRequest.model_validate(
        {"message": "hi", "environment": "production"}
    )
    assert request.connectionId is None


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def _detector(
    points: Dict[str, int],
    model: Optional[str],
    failures: Optional[Dict[str, List[Union[int, Exception]]]] = None,
) -> DynamicProviderDetection:
    """A detector whose Qdrant has ``points`` per collection and whose
    connection resolves to embedding ``model`` (None: cannot be resolved).

    ``failures`` lists, per collection, what its next health checks get before
    Qdrant answers normally: a status code (e.g. 408) or an exception to raise.
    """
    pending = {name: list(items) for name, items in (failures or {}).items()}

    def handler(request: httpx.Request) -> httpx.Response:
        collection = request.url.path.rsplit("/", 1)[-1]
        if pending.get(collection):
            failure = pending[collection].pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, json={"status": "error"})
        return httpx.Response(
            200, json={"result": {"points_count": points.get(collection, 0)}}
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    detector = DynamicProviderDetection(
        "http://qdrant:6333", ContextualRetrievalConfig()
    )
    detector._http_client_manager = SimpleNamespace(
        get_client=AsyncMock(return_value=client)
    )  # type: ignore[assignment]
    detector._get_default_embedding_model = MagicMock(return_value=model)  # type: ignore[method-assign]
    return detector


class TestProviderDetection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("environment", ["production", "testing"])
    async def test_each_connection_searches_the_collection_of_its_model(
        self, environment: str
    ) -> None:
        detector = _detector({AZURE: 10}, "azure_openai/text-embedding-3-large")

        collections = await detector.detect_optimal_collections(environment, VAULT_UUID)

        assert collections == [AZURE]
        detector._get_default_embedding_model.assert_called_once_with(
            environment, VAULT_UUID
        )

    @pytest.mark.asyncio
    async def test_no_fallback_to_a_collection_of_another_vector_size(self) -> None:
        # Connection on AWS (1024), knowledge base built on Azure (3072).
        detector = _detector({AZURE: 10}, "aws_bedrock/amazon.titan-embed-text-v2:0")

        assert await detector.detect_optimal_collections("testing", VAULT_UUID) == []

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_searches_nothing(self) -> None:
        detector = _detector({}, "azure_openai/text-embedding-3-large")

        assert await detector.detect_optimal_collections("production") == []

    @pytest.mark.asyncio
    async def test_unresolved_model_searches_only_filled_collections(self) -> None:
        detector = _detector({AWS: 4}, None)

        assert await detector.detect_optimal_collections("production") == [AWS]

    @pytest.mark.asyncio
    async def test_unresolved_model_and_empty_knowledge_base_searches_nothing(
        self,
    ) -> None:
        detector = _detector({}, None)

        assert await detector.detect_optimal_collections("production") == []

    # A Qdrant hiccup on the health check must not read as "knowledge base
    # empty": under concurrent load Qdrant answered 408 or dropped the pooled
    # connection, and the query got an out-of-scope answer.

    @pytest.fixture(autouse=True)
    def _no_retry_delay(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.contextual_retrieval.provider_detection.HttpClientConstants."
            "HEALTH_CHECK_RETRY_DELAY",
            0,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure", [408, 503, httpx.ReadError(""), httpx.ConnectError("")]
    )
    @pytest.mark.parametrize("environment", ["production", "testing"])
    async def test_transient_health_check_failure_is_retried(
        self, environment: str, failure: Union[int, Exception]
    ) -> None:
        detector = _detector(
            {AZURE: 537}, "azure_openai/text-embedding-3-large", {AZURE: [failure]}
        )

        assert await detector.detect_optimal_collections(environment, VAULT_UUID) == [
            AZURE
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("environment", ["production", "testing"])
    async def test_collection_that_cannot_be_checked_is_still_searched(
        self, environment: str
    ) -> None:
        detector = _detector(
            {AZURE: 537},
            "azure_openai/text-embedding-3-large",
            {AZURE: [408, httpx.ReadError("")]},
        )

        assert await detector.detect_optimal_collections(environment, VAULT_UUID) == [
            AZURE
        ]

    @pytest.mark.asyncio
    async def test_unchecked_collection_never_adds_one_of_another_vector_size(
        self,
    ) -> None:
        # Connection on AWS, its collection missing (404); the Azure check
        # failing must not make it searchable.
        detector = _detector(
            {AZURE: 537},
            "aws_bedrock/amazon.titan-embed-text-v2:0",
            {AWS: [404, 404], AZURE: [408, 408]},
        )

        assert await detector.detect_optimal_collections("testing", VAULT_UUID) == []

    @pytest.mark.asyncio
    async def test_missing_collection_is_not_retried(self) -> None:
        # A retry would reach the 537 points behind the 404.
        detector = _detector(
            {AZURE: 537}, "azure_openai/text-embedding-3-large", {AZURE: [404]}
        )

        assert await detector._collection_points_count(AZURE) == 0


# ---------------------------------------------------------------------------
# Shared BM25 index
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str) -> Dict[str, Any]:
    return {"chunk_id": chunk_id, "original_content": text, "contextual_content": ""}


# BM25 gives a term no weight when it is in half or more of the corpus, so
# tiny test corpora get unrelated chunks around the ones a query should find.
FILLER = [
    _chunk(f"filler_chunk_{i:03d}", f"unrelated topic number {i}") for i in range(6)
]


def _bm25(
    chunks: List[Dict[str, Any]], stats: Optional[Dict[str, Any]] = None
) -> SmartBM25Search:
    bm25 = SmartBM25Search("http://qdrant:6333", ContextualRetrievalConfig())
    bm25._fetch_all_contextual_chunks = AsyncMock(return_value=chunks)  # type: ignore[method-assign]
    bm25._get_collection_stats = AsyncMock(  # type: ignore[method-assign]
        return_value=stats or {AZURE: {"points_count": len(chunks)}}
    )
    return bm25


class TestSharedBM25:
    @pytest.mark.asyncio
    async def test_same_chunk_in_both_collections_is_indexed_once(self) -> None:
        # Mid re-index: the chunk exists in the old and the new collection.
        bm25 = _bm25(
            [
                _chunk("doc_chunk_000", "parking permit application"),
                _chunk("doc_chunk_000", "parking permit application"),
                _chunk("doc_chunk_001", "opening hours of the office"),
                *FILLER,
            ]
        )

        assert await bm25.initialize_index()
        assert len(bm25.chunk_mapping) == 2 + len(FILLER)

        results = await bm25.search_bm25("parking permit", [], limit=5)
        assert [r["chunk_id"] for r in results] == ["doc_chunk_000"]

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_drops_the_old_index_and_records_stats(
        self,
    ) -> None:
        bm25 = _bm25([_chunk("a_chunk_000", "parking permit")])
        assert await bm25.initialize_index()

        bm25._fetch_all_contextual_chunks = AsyncMock(return_value=[])  # type: ignore[method-assign]
        empty_stats = {AZURE: {"points_count": 0}}
        bm25._get_collection_stats = AsyncMock(return_value=empty_stats)  # type: ignore[method-assign]

        assert not await bm25.initialize_index()
        assert bm25.bm25_index is None and bm25.chunk_mapping == {}
        # Recorded, so the refresh fires on the first change, not on every query.
        assert bm25.last_collection_stats == empty_stats
        assert not await bm25._should_refresh_index()

    @pytest.mark.asyncio
    async def test_empty_index_fills_itself_once_chunks_are_indexed(self) -> None:
        bm25 = _bm25([], stats={AZURE: {"points_count": 0}})
        assert not await bm25.initialize_index()
        assert await bm25.search_bm25("parking", [], limit=5) == []

        # The indexer stores chunks; the next query notices and refreshes.
        bm25._fetch_all_contextual_chunks = AsyncMock(  # type: ignore[method-assign]
            return_value=[_chunk("a_chunk_000", "parking permit"), *FILLER]
        )
        bm25._get_collection_stats = AsyncMock(  # type: ignore[method-assign]
            return_value={AZURE: {"points_count": 1 + len(FILLER)}}
        )
        await bm25.search_bm25("parking", [], limit=5)
        await asyncio.gather(*bm25._background_tasks)

        results = await bm25.search_bm25("parking", [], limit=5)
        assert [r["chunk_id"] for r in results] == ["a_chunk_000"]

    @pytest.mark.asyncio
    async def test_test_and_production_query_concurrently_during_a_refresh(
        self,
    ) -> None:
        """Both connections share one index; queries during a sync-triggered
        refresh always see a consistent index (old or new, never mixed)."""
        old = [_chunk(f"old_chunk_{i:03d}", f"parking permit {i}") for i in range(30)]
        new = [_chunk(f"new_chunk_{i:03d}", f"parking permit {i}") for i in range(3)]
        bm25 = _bm25(old)
        assert await bm25.initialize_index()

        bm25._fetch_all_contextual_chunks = AsyncMock(return_value=new)  # type: ignore[method-assign]
        bm25._get_collection_stats = AsyncMock(  # type: ignore[method-assign]
            return_value={AZURE: {"points_count": len(new)}}
        )

        async def query() -> List[str]:
            results = await bm25.search_bm25("parking permit", [], limit=40)
            return [r["chunk_id"] for r in results]

        rounds = await asyncio.gather(
            *(query() for _ in range(20)), bm25._background_refresh_index()
        )

        for chunk_ids in rounds[:-1]:
            assert chunk_ids, "a query got no results while the index was refreshed"
            prefixes = {chunk_id.split("_")[0] for chunk_id in chunk_ids}
            assert len(prefixes) == 1, f"mixed old/new index: {chunk_ids}"
        assert all(cid.startswith("new_") for cid in await query())


@pytest.mark.asyncio
async def test_prewarm_keeps_the_shared_index_when_knowledge_base_is_empty() -> None:
    from src import llm_orchestration_service as service_module

    empty = MagicMock()
    empty.initialize_index = AsyncMock(return_value=False)
    service = SimpleNamespace(shared_bm25_search=None)

    with patch.object(service_module, "SmartBM25Search", return_value=empty):
        await service_module.LLMOrchestrationService._prewarm_shared_bm25(service)  # type: ignore[arg-type]

    assert service.shared_bm25_search is empty


def test_test_and_production_retrievers_share_one_bm25_index() -> None:
    from src.orchestration import component_factory as factory_module

    shared = MagicMock()
    factory = object.__new__(factory_module.ComponentFactory)
    factory._service = SimpleNamespace(shared_bm25_search=shared)

    with (
        patch("src.contextual_retrieval.contextual_retriever.DynamicProviderDetection"),
        patch("src.contextual_retrieval.contextual_retriever.QdrantContextualSearch"),
    ):
        production = factory.initialize_contextual_retriever("production", None)
        testing = factory.initialize_contextual_retriever("testing", VAULT_UUID)

    assert production.bm25_search is shared
    assert testing.bm25_search is shared
    assert testing.connection_id == VAULT_UUID


# ---------------------------------------------------------------------------
# Querying through both connections at the same time
# ---------------------------------------------------------------------------

PRODUCTION_UUID = "0b7e2c1a-9d4f-4e8a-b6c3-1f2e3d4c5b6a"


@pytest.mark.asyncio
async def test_both_connections_find_the_shared_collection_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production and test detect collections at the same time through one
    client, and Qdrant answers one of the health checks with 408 (as it did
    under concurrent load): both still search the shared collection."""
    monkeypatch.setattr(
        "src.contextual_retrieval.provider_detection.HttpClientConstants."
        "HEALTH_CHECK_RETRY_DELAY",
        0,
    )
    detector = _detector(
        {AZURE: 537}, "azure_openai/text-embedding-3-large", {AZURE: [408]}
    )

    production, testing = await asyncio.gather(
        detector.detect_optimal_collections("production", PRODUCTION_UUID),
        detector.detect_optimal_collections("testing", VAULT_UUID),
    )

    assert production == [AZURE]
    assert testing == [AZURE]


def _retriever(environment: str, connection_id: str, embed: Any) -> ContextualRetriever:
    """A retriever whose embedding call is ``embed`` and whose Qdrant search
    returns one chunk per query, tagged with the connection."""
    retriever = object.__new__(ContextualRetriever)
    retriever.environment = environment
    retriever.connection_id = connection_id
    retriever._llm_service = MagicMock()
    retriever._session_llm_service = None
    retriever.qdrant_search = SimpleNamespace(get_embeddings_for_queries_batch=embed)  # type: ignore[assignment]
    retriever._search_single_query_with_embedding = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda query, index, embedding, collections, limit: [
            {"chunk_id": f"{environment}_{index}", "score": 0.9}
        ]
    )
    return retriever


@pytest.mark.asyncio
async def test_both_connections_retrieve_at_the_same_time() -> None:
    """Production and test query the knowledge base concurrently, each with its
    own connection's embeddings, without one waiting for the other.

    The embedding call is synchronous (an HTTP call to the provider). Both calls
    must be in flight at once to pass the barrier: if either ran on the event
    loop it would block the other request, the barrier would time out and the
    retrieval would come back empty instead of hanging the test.
    """
    barrier = threading.Barrier(2, timeout=5)
    calls: List[tuple] = []

    def embed(
        queries: List[str], llm_service: Any, environment: str, connection_id: str
    ) -> List[List[float]]:
        calls.append((environment, connection_id))
        barrier.wait()
        return [[0.1, 0.2, 0.3] for _ in queries]

    production = _retriever("production", PRODUCTION_UUID, embed)
    testing = _retriever("testing", VAULT_UUID, embed)
    queries = ["what is digidoc?", "What is DigiDoc used for?"]

    production_chunks, testing_chunks = await asyncio.gather(
        production._execute_batch_query_searches(
            queries, [AZURE], 40, "production", PRODUCTION_UUID
        ),
        testing._execute_batch_query_searches(
            queries, [AZURE], 40, "testing", VAULT_UUID
        ),
    )

    assert not barrier.broken, "the two connections' embeddings did not overlap"
    assert sorted(calls) == [
        ("production", PRODUCTION_UUID),
        ("testing", VAULT_UUID),
    ]
    assert [c["chunk_id"] for c in production_chunks] == [
        "production_0",
        "production_1",
    ]
    assert [c["chunk_id"] for c in testing_chunks] == ["testing_0", "testing_1"]

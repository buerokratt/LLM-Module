"""Unit tests for issue #805: index with whichever LLM connection exists.

Covers:
- resolve_indexing_connection(): production only / testing only / both
  (production wins) / neither / Resql error / malformed rows
- main(): no connection -> exit 4 without processing; a resolved connection is
  written to the indexer config before processing; Resql error -> exit 1
- _process_single_document(): a document with dropped chunks keeps its stored
  chunks but fails, so it is not marked processed and is retried next sync

Resql is replaced by an httpx.MockTransport; nothing touches the network.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# The indexer imports its sibling packages top-level (e.g. `diff_identifier`),
# as cron-manager's PYTHONPATH includes /app/src/vector_indexer. Mirror that.
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "src" / "vector_indexer")
)

from vector_indexer import main_indexer  # noqa: E402
from vector_indexer.main_indexer import (  # noqa: E402
    EXIT_NO_LLM_CONNECTION,
    ConnectionLookupError,
    IndexingConnection,
    resolve_indexing_connection,
)

RESQL_URL = "http://resql:8082/rag-search"


def _row(vault_uuid: Optional[str], embedding_model: str = "text-embedding-3-large"):
    return [{"vaultUuid": vault_uuid, "embeddingModel": embedding_model}]


@pytest.fixture
def resql(monkeypatch: pytest.MonkeyPatch):
    """Route the resolver's httpx client to a fake Resql.

    Returns a function that sets the per-environment responses and gives back
    the list of Resql endpoints that were called, in order.
    """

    def configure(
        production: Any = (),
        testing: Any = (),
        status_code: int = 200,
    ) -> List[str]:
        calls: List[str] = []
        bodies: Dict[str, Any] = {
            "get-production-connection": list(production)
            if isinstance(production, tuple)
            else production,
            "get-testing-connection": list(testing)
            if isinstance(testing, tuple)
            else testing,
        }

        def handler(request: httpx.Request) -> httpx.Response:
            endpoint = request.url.path.rsplit("/", 1)[-1]
            calls.append(endpoint)
            return httpx.Response(status_code, json=bodies[endpoint])

        real_client = httpx.AsyncClient
        monkeypatch.setattr(
            main_indexer.httpx,
            "AsyncClient",
            lambda **kwargs: real_client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )
        return calls

    return configure


# ---------------------------------------------------------------------------
# resolve_indexing_connection
# ---------------------------------------------------------------------------


class TestResolveIndexingConnection:
    @pytest.mark.asyncio
    async def test_production_only_uses_production(self, resql) -> None:
        calls = resql(production=_row("prod-uuid"))

        connection = await resolve_indexing_connection(RESQL_URL)

        assert connection == IndexingConnection(
            "production", "prod-uuid", "text-embedding-3-large"
        )
        assert calls == ["get-production-connection"]

    @pytest.mark.asyncio
    async def test_testing_only_uses_testing(self, resql) -> None:
        calls = resql(testing=_row("test-uuid"))

        connection = await resolve_indexing_connection(RESQL_URL)

        assert connection is not None
        assert connection.environment == "testing"
        assert connection.vault_uuid == "test-uuid"
        assert calls == ["get-production-connection", "get-testing-connection"]

    @pytest.mark.asyncio
    async def test_both_prefers_production(self, resql) -> None:
        calls = resql(production=_row("prod-uuid"), testing=_row("test-uuid"))

        connection = await resolve_indexing_connection(RESQL_URL)

        assert connection is not None
        assert connection.environment == "production"
        assert connection.vault_uuid == "prod-uuid"
        # Testing is never queried once production is found.
        assert calls == ["get-production-connection"]

    @pytest.mark.asyncio
    async def test_neither_returns_none(self, resql) -> None:
        resql()

        assert await resolve_indexing_connection(RESQL_URL) is None

    @pytest.mark.asyncio
    async def test_wrapped_response_is_accepted(self, resql) -> None:
        resql(production={"response": _row("prod-uuid")})

        connection = await resolve_indexing_connection(RESQL_URL)

        assert connection is not None
        assert connection.vault_uuid == "prod-uuid"

    @pytest.mark.asyncio
    async def test_row_without_vault_uuid_falls_through(self, resql) -> None:
        resql(production=_row(None), testing=_row("test-uuid"))

        connection = await resolve_indexing_connection(RESQL_URL)

        assert connection is not None
        assert connection.environment == "testing"

    @pytest.mark.asyncio
    async def test_resql_error_raises_lookup_error(self, resql) -> None:
        resql(status_code=500)

        with pytest.raises(ConnectionLookupError, match="production"):
            await resolve_indexing_connection(RESQL_URL)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _fake_indexer(failed_documents: int = 0) -> MagicMock:
    """A VectorIndexer stand-in with a real-looking config object."""
    indexer = MagicMock()
    indexer.config = SimpleNamespace(
        resql_url=RESQL_URL, environment="production", connection_id=None
    )
    indexer.run_health_check = AsyncMock(return_value=True)
    indexer.process_all_documents = AsyncMock(
        return_value=SimpleNamespace(documents_failed=failed_documents)
    )
    indexer.cleanup = AsyncMock()
    return indexer


async def _run_main(indexer: MagicMock, resolver: AsyncMock) -> int:
    with (
        patch.object(main_indexer, "VectorIndexer", return_value=indexer),
        patch.object(main_indexer, "resolve_indexing_connection", resolver),
        patch.object(main_indexer, "_acquire_indexer_lock", return_value=MagicMock()),
        patch.object(main_indexer, "fcntl", MagicMock()),
        patch("sys.argv", ["main_indexer.py", "--signed-url", "http://x/data.zip"]),
    ):
        return await main_indexer.main()


class TestMainConnectionHandling:
    @pytest.mark.asyncio
    async def test_no_connection_exits_4_without_processing(self) -> None:
        indexer = _fake_indexer()

        exit_code = await _run_main(indexer, AsyncMock(return_value=None))

        assert exit_code == EXIT_NO_LLM_CONNECTION == 4
        indexer.process_all_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_testing_connection_is_used_for_processing(self) -> None:
        indexer = _fake_indexer()
        seen: Dict[str, Any] = {}

        async def capture_config() -> SimpleNamespace:
            seen["environment"] = indexer.config.environment
            seen["connection_id"] = indexer.config.connection_id
            return SimpleNamespace(documents_failed=0)

        indexer.process_all_documents = AsyncMock(side_effect=capture_config)
        resolver = AsyncMock(
            return_value=IndexingConnection("testing", "test-uuid", "model")
        )

        exit_code = await _run_main(indexer, resolver)

        assert exit_code == 0
        assert seen == {"environment": "testing", "connection_id": "test-uuid"}

    @pytest.mark.asyncio
    async def test_resql_error_exits_1_without_processing(self) -> None:
        indexer = _fake_indexer()

        exit_code = await _run_main(
            indexer, AsyncMock(side_effect=ConnectionLookupError("down"))
        )

        assert exit_code == 1
        indexer.process_all_documents.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_documents_still_exit_2(self) -> None:
        indexer = _fake_indexer(failed_documents=1)
        resolver = AsyncMock(
            return_value=IndexingConnection("production", "prod-uuid", "model")
        )

        assert await _run_main(indexer, resolver) == 2


# ---------------------------------------------------------------------------
# _process_single_document: dropped chunks
# ---------------------------------------------------------------------------


def _document_processor(chunks: List[Any], failed_chunks: int) -> SimpleNamespace:
    """Bare stand-in for VectorIndexer ``self`` used by _process_single_document."""
    document = SimpleNamespace(document_hash="doc-hash")
    return SimpleNamespace(
        document_loader=SimpleNamespace(load_document=lambda _info: document),
        contextual_processor=SimpleNamespace(
            process_document=AsyncMock(return_value=(chunks, failed_chunks))
        ),
        error_logger=MagicMock(),
    )


class TestDroppedChunks:
    @pytest.mark.asyncio
    async def test_dropped_chunk_fails_document_but_keeps_stored_chunks(
        self,
    ) -> None:
        import asyncio

        stub = _document_processor(chunks=["c0", "c2"], failed_chunks=1)
        qdrant = SimpleNamespace(store_chunks=AsyncMock())

        with pytest.raises(RuntimeError, match="1 of 3 chunks failed"):
            await main_indexer.VectorIndexer._process_single_document(
                stub,  # type: ignore[arg-type]
                SimpleNamespace(document_hash="doc-hash"),  # type: ignore[arg-type]
                qdrant,  # type: ignore[arg-type]
                asyncio.Semaphore(1),
            )

        qdrant.store_chunks.assert_awaited_once_with(["c0", "c2"])

    @pytest.mark.asyncio
    async def test_complete_document_succeeds(self) -> None:
        import asyncio

        stub = _document_processor(chunks=["c0", "c1"], failed_chunks=0)
        qdrant = SimpleNamespace(store_chunks=AsyncMock())

        result = await main_indexer.VectorIndexer._process_single_document(
            stub,  # type: ignore[arg-type]
            SimpleNamespace(document_hash="doc-hash"),  # type: ignore[arg-type]
            qdrant,  # type: ignore[arg-type]
            asyncio.Semaphore(1),
        )

        assert result == (2, "doc-hash", 0)

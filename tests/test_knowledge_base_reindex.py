"""Unit tests for issue #588: re-index the knowledge base when the indexing
connection's embedding model differs from the one that built it.

Covers:
- QdrantManager: reading which collections hold embeddings (and with which
  model), counting a document's complete chunks, pruning stale chunks
- plan_reindex(): Azure -> AWS and AWS -> Azure need a re-index; same platform,
  empty knowledge base or unknown platform do not
- VectorIndexer._run_full_reindex(): resume skips finished documents, a failure
  keeps the old embeddings, success removes them and stale chunks, the test
  embedding gate (credentials not ready, wrong vector size)
- main(): a re-index plan reaches process_all_documents

Qdrant is an httpx.MockTransport or a fake; nothing touches the network.
"""

import asyncio
import io
import json
import os
import signal
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Like cron-manager's PYTHONPATH (for `diff_identifier`); appended so that
# src/vector_indexer/models.py does not shadow the src/models package.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "vector_indexer"))

from vector_indexer import dataset_download, main_indexer  # noqa: E402
from vector_indexer.main_indexer import (  # noqa: E402
    IndexingConnection,
    ReindexPlan,
    plan_reindex,
)
from vector_indexer.models import DocumentInfo, ProcessingStats  # noqa: E402
from vector_indexer.qdrant_manager import (  # noqa: E402
    QdrantManager,
    QdrantOperationError,
    StoredEmbeddings,
)

AZURE_COLLECTION = "contextual_chunks_azure"
AWS_COLLECTION = "contextual_chunks_aws"
AZURE_STORED = StoredEmbeddings(
    AZURE_COLLECTION, "azure", "text-embedding-3-large", 3072, 120
)
AWS_STORED = StoredEmbeddings(
    AWS_COLLECTION, "aws", "amazon.titan-embed-text-v2:0", 1024, 80
)


def _connection(platform: str) -> IndexingConnection:
    model = {
        "azure": "text-embedding-3-large",
        "aws": "amazon.titan-embed-text-v2:0",
    }.get(platform, "unknown-model")
    return IndexingConnection("production", "prod-uuid", model, platform)


# ---------------------------------------------------------------------------
# QdrantManager (HTTP faked)
# ---------------------------------------------------------------------------


def _qdrant(handler) -> QdrantManager:
    manager = QdrantManager(SimpleNamespace(qdrant_url="http://qdrant:6333"))  # type: ignore[arg-type]
    manager.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return manager


class TestGetStoredEmbeddings:
    @pytest.mark.asyncio
    async def test_reports_filled_collections_with_their_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            collection = request.url.path.split("/")[2]
            if collection == AWS_COLLECTION:
                return httpx.Response(404)  # never created
            if request.url.path.endswith("/count"):
                return httpx.Response(200, json={"result": {"count": 7}})
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "payload": {
                                    "embedding_model": "azure_openai/text-embedding-3-large",
                                    "vector_dimensions": 3072,
                                }
                            }
                        ]
                    }
                },
            )

        stored = await _qdrant(handler).get_stored_embeddings()

        assert stored == [
            StoredEmbeddings(
                AZURE_COLLECTION, "azure", "text-embedding-3-large", 3072, 7
            )
        ]

    @pytest.mark.asyncio
    async def test_empty_collections_are_not_reported(self) -> None:
        stored = await _qdrant(
            lambda _r: httpx.Response(200, json={"result": {"count": 0}})
        ).get_stored_embeddings()

        assert stored == []

    @pytest.mark.asyncio
    async def test_qdrant_error_raises(self) -> None:
        with pytest.raises(QdrantOperationError):
            await _qdrant(lambda _r: httpx.Response(500)).get_stored_embeddings()


class TestCountDocumentChunks:
    @staticmethod
    def _points(indexes: List[int], total: int) -> Dict[str, Any]:
        return {
            "result": {
                "points": [
                    {"payload": {"chunk_index": i, "total_chunks": total}}
                    for i in indexes
                ]
            }
        }

    @pytest.mark.asyncio
    async def test_complete_document_returns_its_chunk_count(self) -> None:
        manager = _qdrant(
            lambda _r: httpx.Response(200, json=self._points([0, 1, 2], 3))
        )

        assert await manager.count_document_chunks(AWS_COLLECTION, "h") == 3

    @pytest.mark.asyncio
    async def test_partial_document_is_not_complete(self) -> None:
        manager = _qdrant(lambda _r: httpx.Response(200, json=self._points([0, 2], 3)))

        assert await manager.count_document_chunks(AWS_COLLECTION, "h") is None

    @pytest.mark.asyncio
    async def test_missing_document_is_not_complete(self) -> None:
        manager = _qdrant(lambda _r: httpx.Response(200, json=self._points([], 0)))

        assert await manager.count_document_chunks(AWS_COLLECTION, "h") is None


class TestDeleteChunksNotIn:
    @pytest.mark.asyncio
    async def test_deletes_chunks_of_documents_outside_the_dataset(self) -> None:
        bodies: List[Dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "ok"})

        await _qdrant(handler).delete_chunks_not_in(AWS_COLLECTION, {"b", "a"})

        assert bodies == [
            {
                "filter": {
                    "must_not": [{"key": "document_hash", "match": {"any": ["a", "b"]}}]
                }
            }
        ]

    @pytest.mark.asyncio
    async def test_empty_keep_set_never_deletes_everything(self) -> None:
        handler = MagicMock()

        await _qdrant(handler).delete_chunks_not_in(AWS_COLLECTION, set())

        handler.assert_not_called()


# ---------------------------------------------------------------------------
# plan_reindex
# ---------------------------------------------------------------------------


def _stored(*stored: StoredEmbeddings) -> SimpleNamespace:
    return SimpleNamespace(get_stored_embeddings=AsyncMock(return_value=list(stored)))


class TestPlanReindex:
    @pytest.mark.asyncio
    async def test_azure_to_aws_requires_reindex(self) -> None:
        plan = await plan_reindex(_stored(AZURE_STORED), _connection("aws"))  # type: ignore[arg-type]

        assert plan is not None
        assert plan.target_collection == AWS_COLLECTION
        assert plan.old_collections == [AZURE_COLLECTION]
        assert "text-embedding-3-large" in plan.describe()
        assert "amazon.titan-embed-text-v2:0" in plan.describe()

    @pytest.mark.asyncio
    async def test_aws_to_azure_requires_reindex(self) -> None:
        plan = await plan_reindex(_stored(AWS_STORED), _connection("azure"))  # type: ignore[arg-type]

        assert plan is not None
        assert plan.target_collection == AZURE_COLLECTION
        assert plan.old_collections == [AWS_COLLECTION]

    @pytest.mark.asyncio
    async def test_interrupted_reindex_is_planned_again(self) -> None:
        # Both collections filled: the old one was not emptied yet.
        plan = await plan_reindex(
            _stored(AZURE_STORED, AWS_STORED),
            _connection("aws"),  # type: ignore[arg-type]
        )

        assert plan is not None
        assert plan.old_collections == [AZURE_COLLECTION]

    @pytest.mark.asyncio
    async def test_same_platform_needs_no_reindex(self) -> None:
        assert await plan_reindex(_stored(AZURE_STORED), _connection("azure")) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_empty_knowledge_base_needs_no_reindex(self) -> None:
        assert await plan_reindex(_stored(), _connection("aws")) is None  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_unknown_platform_is_not_reindexed(self) -> None:
        qdrant = _stored(AZURE_STORED)

        assert await plan_reindex(qdrant, _connection("openai")) is None  # type: ignore[arg-type]
        qdrant.get_stored_embeddings.assert_not_awaited()


# ---------------------------------------------------------------------------
# VectorIndexer._run_full_reindex
# ---------------------------------------------------------------------------

AZURE_TO_AWS = ReindexPlan(
    AWS_COLLECTION, "amazon.titan-embed-text-v2:0 (aws)", [AZURE_STORED]
)


def _doc(document_hash: str) -> DocumentInfo:
    return DocumentInfo(
        document_hash=document_hash,
        cleaned_txt_path=f"/data/{document_hash}/cleaned.txt",
        source_meta_path=f"/data/{document_hash}/cleaned.meta.json",
        dataset_collection="agency",
    )


class FakeQdrant:
    collections_config = {
        AZURE_COLLECTION: {"vector_size": 3072},
        AWS_COLLECTION: {"vector_size": 1024},
    }

    def __init__(self, complete: Optional[Dict[str, int]] = None) -> None:
        self.complete = complete or {}
        self.ensure_collections_exist = AsyncMock()
        self.clear_collection = AsyncMock()
        self.delete_chunks_not_in = AsyncMock()

    async def __aenter__(self) -> "FakeQdrant":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def count_document_chunks(self, _collection: str, document_hash: str):
        return self.complete.get(document_hash)


def _indexer(
    documents: List[DocumentInfo],
    failing: Set[str] = frozenset(),  # type: ignore[assignment]
    embedding_dimensions: int = 1024,
) -> Any:
    indexer = object.__new__(main_indexer.VectorIndexer)
    indexer.config = SimpleNamespace(max_concurrent_documents=2)
    indexer.document_loader = SimpleNamespace(discover_all_documents=lambda: documents)
    indexer.stats = ProcessingStats()
    indexer.error_logger = MagicMock()
    indexer.api_client = SimpleNamespace(
        create_embeddings_batch=AsyncMock(
            return_value={"embeddings": [[0.1] * embedding_dimensions]}
        )
    )
    indexer.processed: List[str] = []

    async def process_single(doc_info: DocumentInfo, _qdrant: Any, _semaphore: Any):
        indexer.processed.append(doc_info.document_hash)
        if doc_info.document_hash in failing:
            raise RuntimeError("context generation failed")
        return (4, doc_info.document_hash, 0)

    indexer._process_single_document = process_single
    indexer._record_full_reindex_metadata = AsyncMock()
    indexer._cleanup_datasets = MagicMock()
    return indexer


async def _reindex(indexer: Any, qdrant: FakeQdrant, plan: ReindexPlan = AZURE_TO_AWS):
    with patch.object(main_indexer, "QdrantManager", return_value=qdrant):
        return await indexer._run_full_reindex(plan)


class TestRunFullReindex:
    @pytest.mark.asyncio
    async def test_success_reindexes_all_and_removes_old_embeddings(self) -> None:
        indexer = _indexer([_doc("a"), _doc("b")])
        qdrant = FakeQdrant()

        stats = await _reindex(indexer, qdrant)

        assert sorted(indexer.processed) == ["a", "b"]
        assert stats.documents_failed == 0
        assert stats.documents_processed == 2
        qdrant.clear_collection.assert_awaited_once_with(AZURE_COLLECTION)
        qdrant.delete_chunks_not_in.assert_awaited_once_with(AWS_COLLECTION, {"a", "b"})
        indexer._record_full_reindex_metadata.assert_awaited_once()
        documents, chunks_info = indexer._record_full_reindex_metadata.await_args.args
        assert [d.document_hash for d in documents] == ["a", "b"]
        assert chunks_info == {"a": {"chunk_count": 4}, "b": {"chunk_count": 4}}

    @pytest.mark.asyncio
    async def test_resume_skips_documents_already_reindexed(self) -> None:
        indexer = _indexer([_doc("a"), _doc("b"), _doc("c")])
        qdrant = FakeQdrant(complete={"a": 5})

        stats = await _reindex(indexer, qdrant)

        assert sorted(indexer.processed) == ["b", "c"]
        assert stats.documents_processed == 3
        _, chunks_info = indexer._record_full_reindex_metadata.await_args.args
        assert chunks_info["a"] == {"chunk_count": 5}
        qdrant.clear_collection.assert_awaited_once_with(AZURE_COLLECTION)

    @pytest.mark.asyncio
    async def test_failure_keeps_old_embeddings_for_the_retry(self) -> None:
        indexer = _indexer([_doc("a"), _doc("b")], failing={"b"})
        qdrant = FakeQdrant()

        stats = await _reindex(indexer, qdrant)

        assert stats.documents_failed == 1  # main() exits 2: hash not saved
        qdrant.clear_collection.assert_not_awaited()
        qdrant.delete_chunks_not_in.assert_not_awaited()
        indexer._record_full_reindex_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_dataset_keeps_old_embeddings(self) -> None:
        indexer = _indexer([])
        qdrant = FakeQdrant()

        await _reindex(indexer, qdrant)

        qdrant.clear_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_with_wrong_vector_size_aborts_before_processing(
        self,
    ) -> None:
        indexer = _indexer([_doc("a")], embedding_dimensions=1536)
        qdrant = FakeQdrant()

        with pytest.raises(RuntimeError, match="1536 dimensions"):
            await _reindex(indexer, qdrant)

        assert indexer.processed == []
        qdrant.clear_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_waits_for_credentials_that_are_still_being_stored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_indexer, "REINDEX_EMBEDDING_READY_DELAY_SECONDS", 0)
        indexer = _indexer([_doc("a")])
        indexer.api_client.create_embeddings_batch = AsyncMock(
            side_effect=[
                RuntimeError("secret not found"),
                {"embeddings": [[0.1] * 1024]},
            ]
        )

        await _reindex(indexer, FakeQdrant())

        assert indexer.processed == ["a"]

    @pytest.mark.asyncio
    async def test_no_working_embedding_fails_without_touching_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_indexer, "REINDEX_EMBEDDING_READY_DELAY_SECONDS", 0)
        indexer = _indexer([_doc("a")])
        indexer.api_client.create_embeddings_batch = AsyncMock(
            side_effect=RuntimeError("secret not found")
        )
        qdrant = FakeQdrant()

        with pytest.raises(RuntimeError, match="no working embedding"):
            await _reindex(indexer, qdrant)

        assert indexer.processed == []
        qdrant.clear_collection.assert_not_awaited()


# ---------------------------------------------------------------------------
# main(): the plan reaches process_all_documents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_passes_the_reindex_plan_to_processing() -> None:
    indexer = MagicMock()
    indexer.config = SimpleNamespace(
        resql_url="http://resql", environment="production", connection_id=None
    )
    indexer.run_health_check = AsyncMock(return_value=True)
    indexer.process_all_documents = AsyncMock(
        return_value=SimpleNamespace(documents_failed=0)
    )
    indexer.cleanup = AsyncMock()
    qdrant = FakeQdrant()

    with (
        patch.object(main_indexer, "VectorIndexer", return_value=indexer),
        patch.object(
            main_indexer,
            "resolve_indexing_connection",
            AsyncMock(return_value=_connection("aws")),
        ),
        patch.object(main_indexer, "QdrantManager", return_value=qdrant),
        patch.object(
            main_indexer, "plan_reindex", AsyncMock(return_value=AZURE_TO_AWS)
        ),
        patch.object(main_indexer, "_acquire_indexer_lock", return_value=MagicMock()),
        patch.object(main_indexer, "fcntl", MagicMock()),
        patch("sys.argv", ["main_indexer.py", "--signed-url", "http://x/data.zip"]),
    ):
        exit_code = await main_indexer.main()

    assert exit_code == 0
    indexer.process_all_documents.assert_awaited_once_with(reindex=AZURE_TO_AWS)


# ---------------------------------------------------------------------------
# Logging: start, end, failure and interruption name the old and new model
# ---------------------------------------------------------------------------

OLD_MODEL = "text-embedding-3-large"
NEW_MODEL = "amazon.titan-embed-text-v2:0"


def _logged(mock_logger: MagicMock, marker: str) -> List[str]:
    """Messages containing ``KB re-index <marker>``, at any level."""
    messages = [
        str(call.args[0])
        for method in (mock_logger.info, mock_logger.warning, mock_logger.error)
        for call in method.call_args_list
    ]
    return [m for m in messages if f"KB re-index {marker}" in m]


def _names_both_models(message: str) -> bool:
    return OLD_MODEL in message and NEW_MODEL in message


class TestReindexLogging:
    @pytest.mark.asyncio
    async def test_start_and_completion_name_both_models(self) -> None:
        with patch.object(main_indexer, "logger") as log:
            await _reindex(_indexer([_doc("a"), _doc("b")]), FakeQdrant())

        started, completed = _logged(log, "STARTED"), _logged(log, "COMPLETED")
        assert len(started) == 1 and _names_both_models(started[0])
        assert len(completed) == 1 and _names_both_models(completed[0])

    @pytest.mark.asyncio
    async def test_incomplete_run_names_both_models(self) -> None:
        with patch.object(main_indexer, "logger") as log:
            await _reindex(
                _indexer([_doc("a"), _doc("b")], failing={"b"}), FakeQdrant()
            )

        incomplete = _logged(log, "INCOMPLETE")
        assert len(incomplete) == 1 and _names_both_models(incomplete[0])
        assert "1 of 2 documents failed" in incomplete[0]
        assert _logged(log, "COMPLETED") == []

    @pytest.mark.asyncio
    async def test_failure_names_both_models(self) -> None:
        with patch.object(main_indexer, "logger") as log:
            with pytest.raises(RuntimeError):
                await _reindex(
                    _indexer([_doc("a")], embedding_dimensions=1536), FakeQdrant()
                )

        failed = _logged(log, "FAILED")
        assert len(failed) == 1 and _names_both_models(failed[0])

    @pytest.mark.asyncio
    async def test_interruption_names_both_models_and_keeps_old_embeddings(
        self,
    ) -> None:
        qdrant = FakeQdrant()
        qdrant.count_document_chunks = AsyncMock(side_effect=asyncio.CancelledError())

        with patch.object(main_indexer, "logger") as log:
            with pytest.raises(asyncio.CancelledError):
                await _reindex(_indexer([_doc("a")]), qdrant)

        interrupted = _logged(log, "INTERRUPTED")
        assert len(interrupted) == 1 and _names_both_models(interrupted[0])
        assert _logged(log, "FAILED") == []
        qdrant.clear_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resumed_run_is_logged_with_both_models(self) -> None:
        with patch.object(main_indexer, "logger") as log:
            await _reindex(
                _indexer([_doc("a"), _doc("b"), _doc("c")]),
                FakeQdrant(complete={"a": 5}),
            )

        resumed = _logged(log, "RESUMED")
        assert len(resumed) == 1 and _names_both_models(resumed[0])
        assert "1 of 3 documents already re-indexed" in resumed[0]

    @pytest.mark.asyncio
    async def test_fresh_run_is_not_logged_as_resumed(self) -> None:
        with patch.object(main_indexer, "logger") as log:
            await _reindex(_indexer([_doc("a")]), FakeQdrant())

        assert _logged(log, "RESUMED") == []

    @pytest.mark.asyncio
    async def test_dataset_download_failure_names_both_models(self) -> None:
        indexer = _indexer([_doc("a")])
        indexer.signed_url = "http://s3/data.zip?sig=1"
        indexer.config = SimpleNamespace(
            max_concurrent_documents=2, dataset_base_path="datasets"
        )
        indexer.api_client.close = AsyncMock()

        with (
            patch.object(main_indexer, "logger") as log,
            patch.object(
                main_indexer,
                "download_and_extract_dataset",
                side_effect=RuntimeError("403 Forbidden"),
            ),
        ):
            with pytest.raises(RuntimeError):
                await indexer.process_all_documents(reindex=AZURE_TO_AWS)

        failed = _logged(log, "FAILED")
        assert len(failed) == 1 and _names_both_models(failed[0])
        assert "403 Forbidden" in failed[0]


@pytest.mark.asyncio
async def test_sigterm_interrupts_the_run_instead_of_killing_it() -> None:
    """SIGTERM cancels the run: main() logs it and returns 130 (#588)."""
    indexer = MagicMock()
    indexer.config = SimpleNamespace(
        resql_url="http://resql", environment="production", connection_id=None
    )
    indexer.run_health_check = AsyncMock(return_value=True)
    indexer.cleanup = AsyncMock()
    reached_end = False

    async def process_all_documents(reindex: Any) -> Any:
        nonlocal reached_end
        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(5)
        reached_end = True

    indexer.process_all_documents = process_all_documents
    loop = asyncio.get_running_loop()
    try:
        with (
            patch.object(main_indexer, "VectorIndexer", return_value=indexer),
            patch.object(
                main_indexer,
                "resolve_indexing_connection",
                AsyncMock(return_value=_connection("aws")),
            ),
            patch.object(main_indexer, "QdrantManager", return_value=FakeQdrant()),
            patch.object(
                main_indexer, "plan_reindex", AsyncMock(return_value=AZURE_TO_AWS)
            ),
            patch.object(
                main_indexer, "_acquire_indexer_lock", return_value=MagicMock()
            ),
            patch.object(main_indexer, "fcntl", MagicMock()),
            patch.object(main_indexer, "logger") as log,
            patch("sys.argv", ["main_indexer.py", "--signed-url", "http://x/data.zip"]),
        ):
            exit_code = await main_indexer.main()
    finally:
        loop.remove_signal_handler(signal.SIGTERM)

    assert exit_code == 130
    assert not reached_end
    indexer.cleanup.assert_awaited_once()
    assert any("interrupted" in str(c.args[0]) for c in log.warning.call_args_list)


# ---------------------------------------------------------------------------
# Dataset download: a retry indexes exactly the new dataset
# ---------------------------------------------------------------------------


def _zip_bytes(files: Dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


def _download_response(content: bytes) -> MagicMock:
    response = MagicMock()
    response.iter_content.return_value = [content]
    return response


class TestDatasetDownload:
    def test_leftovers_of_an_unfinished_run_are_removed_before_extracting(
        self, tmp_path: Path
    ) -> None:
        leftover = tmp_path / "removed-collection" / "doc" / "cleaned.txt"
        leftover.parent.mkdir(parents=True)
        leftover.write_text("content CKB no longer has")
        (tmp_path / "stray.txt").write_text("x")
        archive = _zip_bytes({"agency/doc1/cleaned.txt": "current content"})

        with patch.object(
            dataset_download.requests,
            "get",
            return_value=_download_response(archive),
        ):
            path, count = dataset_download.download_and_extract_dataset(
                "http://s3/data.zip?sig=1", datasets_path=tmp_path
            )

        assert path == str(tmp_path) and count == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["agency"]
        assert (tmp_path / "agency" / "doc1" / "cleaned.txt").read_text() == (
            "current content"
        )

    def test_failed_download_leaves_the_folder_untouched(self, tmp_path: Path) -> None:
        kept = tmp_path / "agency" / "doc1" / "cleaned.txt"
        kept.parent.mkdir(parents=True)
        kept.write_text("previous content")

        with patch.object(
            dataset_download.requests,
            "get",
            side_effect=dataset_download.requests.ConnectionError("unreachable"),
        ):
            with pytest.raises(dataset_download.requests.ConnectionError):
                dataset_download.download_and_extract_dataset(
                    "http://s3/data.zip?sig=1", datasets_path=tmp_path
                )

        assert kept.read_text() == "previous content"


def test_indexer_lock_is_one_for_the_whole_knowledge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main_indexer, "INDEXER_LOCK_DIR", tmp_path)
    flock = MagicMock()
    monkeypatch.setattr(main_indexer.fcntl, "flock", flock)

    first = main_indexer._acquire_indexer_lock("https://s3/data.zip?sig=1")
    second = main_indexer._acquire_indexer_lock("https://s3/data.zip?sig=2")

    try:
        # Different presigned URLs, same lock file
        assert first is not None and second is not None
        assert Path(first.name) == Path(second.name)
        assert Path(first.name).name == f"{main_indexer.KNOWLEDGE_BASE_LOCK_KEY}.lock"
    finally:
        for handle in (first, second):
            if handle:
                handle.close()

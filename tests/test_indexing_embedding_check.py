"""Issue #587: the indexer checks the embedding model really behind the
indexing connection before it indexes anything.

On Azure the deployment name decides the model, not the dropdown: a deployment
of text-embedding-ada-002 (1536 dimensions) behind a connection set to
text-embedding-3-large (3072) must not get as far as embedding documents. The
run fails before any document is processed, so the agency hash is not saved and
the next sync retries once the connection is fixed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Like cron-manager's PYTHONPATH (for `diff_identifier`); appended so that
# src/vector_indexer/models.py does not shadow the src/models package.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "vector_indexer"))

from vector_indexer import main_indexer  # noqa: E402
from vector_indexer.models import ProcessingStats  # noqa: E402


class FakeQdrant:
    collections_config = {
        "contextual_chunks_azure": {"vector_size": 3072},
        "contextual_chunks_aws": {"vector_size": 1024},
    }

    def __init__(self, *_args: Any) -> None:
        self.ensure_collections_exist = AsyncMock()

    async def __aenter__(self) -> "FakeQdrant":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _indexer(platform: Optional[str], dimensions: int) -> Any:
    indexer = object.__new__(main_indexer.VectorIndexer)
    indexer.config = SimpleNamespace(max_concurrent_documents=2, dataset_base_path="/tmp/ds")
    indexer.signed_url = None
    indexer.indexing_platform = platform
    indexer.stats = ProcessingStats()
    indexer.error_logger = MagicMock()
    indexer.api_client = SimpleNamespace(
        close=AsyncMock(),
        create_embeddings_batch=AsyncMock(
            return_value={"embeddings": [[0.1] * dimensions], "model_used": "x"}
        ),
    )
    indexer._filter_documents_by_paths = lambda paths: [
        SimpleNamespace(document_hash="doc-a", cleaned_txt_path=p) for p in paths
    ]
    indexer._process_documents = AsyncMock(return_value=([], {}))
    indexer._cleanup_datasets = MagicMock()
    indexer._log_final_summary = MagicMock()
    return indexer


async def _process(indexer: Any, changed: List[str]) -> Any:
    diff = SimpleNamespace(
        total_files_scanned=1,
        previously_processed_count=0,
        new_files=changed,
        modified_files=[],
        deleted_files=[],
        unchanged_files=[],
        is_first_run=True,
        chunks_to_delete={},
    )
    detector = MagicMock()
    detector.get_changed_files = AsyncMock(return_value=diff)
    detector.mark_files_processed = AsyncMock()
    with (
        patch.object(main_indexer, "QdrantManager", FakeQdrant),
        patch.object(main_indexer, "DiffDetector", return_value=detector),
        patch.object(main_indexer, "create_diff_config", return_value=None),
    ):
        return await indexer.process_all_documents()


@pytest.mark.asyncio
async def test_wrong_azure_deployment_aborts_before_any_document_is_processed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_indexer, "REINDEX_EMBEDDING_READY_DELAY_SECONDS", 0)
    indexer = _indexer("azure", dimensions=1536)  # an ada-002 deployment

    with pytest.raises(RuntimeError, match="1536 dimensions.*needs 3072"):
        await _process(indexer, ["/data/doc-a/cleaned.txt"])

    indexer._process_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_matching_model_is_indexed() -> None:
    indexer = _indexer("aws", dimensions=1024)

    await _process(indexer, ["/data/doc-a/cleaned.txt"])

    indexer.api_client.create_embeddings_batch.assert_awaited_once()
    indexer._process_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_nothing_to_index_makes_no_test_embedding() -> None:
    indexer = _indexer("azure", dimensions=3072)

    await _process(indexer, [])

    indexer.api_client.create_embeddings_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_platform_skips_the_check() -> None:
    indexer = _indexer(None, dimensions=1536)

    await _process(indexer, ["/data/doc-a/cleaned.txt"])

    indexer.api_client.create_embeddings_batch.assert_not_awaited()
    indexer._process_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_passes_the_connection_platform_to_the_indexer() -> None:
    indexer = MagicMock()
    indexer.config = SimpleNamespace(
        resql_url="http://resql", environment="production", connection_id=None
    )
    indexer.run_health_check = AsyncMock(return_value=True)
    indexer.cleanup = AsyncMock()
    seen: dict = {}

    async def process_all_documents(reindex: Any) -> Any:
        seen["platform"] = indexer.indexing_platform
        return SimpleNamespace(documents_failed=0)

    indexer.process_all_documents = process_all_documents
    connection = main_indexer.IndexingConnection(
        "testing", "test-uuid", "amazon.titan-embed-text-v2:0", "aws"
    )
    with (
        patch.object(main_indexer, "VectorIndexer", return_value=indexer),
        patch.object(
            main_indexer, "resolve_indexing_connection", AsyncMock(return_value=connection)
        ),
        patch.object(main_indexer, "QdrantManager", FakeQdrant),
        patch.object(main_indexer, "plan_reindex", AsyncMock(return_value=None)),
        patch.object(main_indexer, "_acquire_indexer_lock", return_value=MagicMock()),
        patch.object(main_indexer, "fcntl", MagicMock()),
        patch("sys.argv", ["main_indexer.py"]),
    ):
        assert await main_indexer.main() == 0

    assert seen == {"platform": "aws"}

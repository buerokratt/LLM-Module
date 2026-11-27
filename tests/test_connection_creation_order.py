"""Issue #586: both creation orders, test -> production and production -> test.

Each test walks through the whole sequence an operator goes through:

1. The first connection is created (the embedding check allows it).
2. The hourly sync runs the indexer: it indexes with that connection.
3. The second connection is created (allowed, no re-index required).
4. The next sync: the indexer picks production, finds the knowledge base already
   built with the same model (no re-index plan, same collection, so no second
   knowledge base) and only processes changed files - none, so nothing is
   embedded again.

The real decision code runs: resolve_indexing_connection (against a fake Resql
over httpx.MockTransport), plan_reindex, the indexer's diff path,
check_embedding_compatibility and reindex_required_for (what the data/update
sync asks via /knowledge-base/status). Qdrant, the diff detector and the LLM
calls are fakes; nothing touches the network.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Like cron-manager's PYTHONPATH (for `diff_identifier`); appended so that
# src/vector_indexer/models.py does not shadow the src/models package.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "vector_indexer"))

from src.llm_orchestrator_config.knowledge_base_lock import (  # noqa: E402
    LockedModel,
    OtherConnection,
    check_embedding_compatibility,
    reindex_required_for,
)
from vector_indexer import main_indexer  # noqa: E402
from vector_indexer.models import ProcessingStats  # noqa: E402
from vector_indexer.qdrant_manager import StoredEmbeddings  # noqa: E402

RESQL_URL = "http://resql:8082/rag-search"
AZURE_MODEL = "text-embedding-3-large"
AWS_MODEL = "amazon.titan-embed-text-v2:0"
AZURE_COLLECTION = "contextual_chunks_azure"


class Environment:
    """The LLM connections (Resql) and the knowledge base (Qdrant) of one setup."""

    def __init__(self) -> None:
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.stored: List[StoredEmbeddings] = []
        self.indexer_runs: List[Dict[str, Any]] = []

    # -- operator actions ---------------------------------------------------

    def create_connection(
        self, environment: str, platform: str = "azure", model: str = AZURE_MODEL
    ):
        """Run the embedding check like the GUI and add.yml do; save if allowed."""
        result = check_embedding_compatibility(
            platform=platform,
            embedding_model=model,
            detected_vector_dimensions=3072 if platform == "azure" else 1024,
            locked_models=self.locked_models(),
            other_connections=self.other_connections(),
            environment=environment,
        )
        if result.allowed:
            self.connections[environment] = {
                "id": len(self.connections) + 1,
                "connectionName": f"{environment} connection",
                "vaultUuid": f"{environment}-uuid",
                "embeddingPlatform": platform,
                "embeddingModel": model,
            }
        return result

    # -- what the services read --------------------------------------------

    def locked_models(self) -> List[LockedModel]:
        return [
            LockedModel(
                s.platform, s.embedding_model, s.vector_dimensions, s.point_count
            )
            for s in self.stored
        ]

    def other_connections(self) -> List[OtherConnection]:
        return [
            OtherConnection(
                row["id"],
                row["connectionName"],
                environment,
                row["embeddingPlatform"],
                row["embeddingModel"],
            )
            for environment, row in self.connections.items()
        ]

    def status_requires_reindex(self) -> bool:
        """What /knowledge-base/status answers the data/update sync."""
        indexing = self.connections.get("production") or self.connections.get("testing")
        return bool(
            indexing
            and reindex_required_for(
                indexing["embeddingPlatform"], self.locked_models()
            )
        )

    # -- the hourly sync's indexer run --------------------------------------

    async def run_indexer(self, changed_files: List[str]) -> int:
        """Run main_indexer.main() against this setup.

        ``changed_files`` is what the diff detector reports as new or modified.
        Indexed documents land in the collection of the indexing connection.
        """
        env = self
        run: Dict[str, Any] = {"processed": []}
        self.indexer_runs.append(run)

        def resql(request: httpx.Request) -> httpx.Response:
            environment = request.url.path.rsplit("/", 1)[-1].split("-")[1]
            row = env.connections.get(environment)
            return httpx.Response(200, json=[row] if row else [])

        class FakeQdrant:
            collections_config = {
                AZURE_COLLECTION: {"vector_size": 3072},
                "contextual_chunks_aws": {"vector_size": 1024},
            }

            def __init__(self, *_args: Any) -> None:
                self.ensure_collections_exist = AsyncMock()

            async def __aenter__(self) -> "FakeQdrant":
                return self

            async def __aexit__(self, *_args: Any) -> None:
                return None

            async def get_stored_embeddings(self) -> List[StoredEmbeddings]:
                return list(env.stored)

        diff = SimpleNamespace(
            total_files_scanned=3,
            previously_processed_count=3 - len(changed_files),
            new_files=changed_files,
            modified_files=[],
            deleted_files=[],
            unchanged_files=[],
            is_first_run=not env.stored,
            chunks_to_delete={},
        )
        diff_detector = MagicMock()
        diff_detector.get_changed_files = AsyncMock(return_value=diff)
        diff_detector.mark_files_processed = AsyncMock()

        indexer = object.__new__(main_indexer.VectorIndexer)
        indexer.config = SimpleNamespace(
            resql_url=RESQL_URL,
            environment="production",
            connection_id=None,
            max_concurrent_documents=2,
            dataset_base_path="/tmp/datasets",
        )
        indexer.signed_url = None
        indexer.indexing_platform = None
        indexer.stats = ProcessingStats()
        indexer.error_logger = MagicMock()
        indexer.api_client = SimpleNamespace(
            close=AsyncMock(),
            create_embeddings_batch=AsyncMock(
                side_effect=lambda _texts: {
                    "embeddings": [
                        [0.1] * (3072 if indexer.indexing_platform == "azure" else 1024)
                    ]
                }
            ),
        )
        indexer.run_health_check = AsyncMock(return_value=True)
        indexer.cleanup = AsyncMock()
        indexer._cleanup_datasets = MagicMock()
        indexer._log_final_summary = MagicMock()
        indexer._filter_documents_by_paths = lambda paths: [
            SimpleNamespace(document_hash=Path(p).parent.name, cleaned_txt_path=p)
            for p in paths
        ]

        async def process_documents(documents: List[Any], _qdrant: Any):
            run["processed"] = [doc.document_hash for doc in documents]
            platform = indexer.indexing_platform
            model = AZURE_MODEL if platform == "azure" else AWS_MODEL
            collection = f"contextual_chunks_{platform}"
            existing = [s for s in env.stored if s.collection == collection]
            count = (existing[0].point_count if existing else 0) + 4 * len(documents)
            env.stored = [s for s in env.stored if s.collection != collection] + [
                StoredEmbeddings(
                    collection,
                    platform,
                    model,
                    3072 if platform == "azure" else 1024,
                    count,
                )
            ]
            indexer.stats.documents_processed += len(documents)
            return documents, {}

        indexer._process_documents = process_documents

        real_client = httpx.AsyncClient
        with (
            patch.object(main_indexer, "VectorIndexer", return_value=indexer),
            patch.object(
                main_indexer.httpx,
                "AsyncClient",
                lambda **kwargs: real_client(
                    transport=httpx.MockTransport(resql), **kwargs
                ),
            ),
            patch.object(main_indexer, "QdrantManager", FakeQdrant),
            patch.object(main_indexer, "DiffDetector", return_value=diff_detector),
            patch.object(main_indexer, "create_diff_config", return_value=None),
            patch.object(
                main_indexer, "_acquire_indexer_lock", return_value=MagicMock()
            ),
            patch.object(main_indexer, "fcntl", MagicMock()),
            patch("sys.argv", ["main_indexer.py"]),
        ):
            reindex_plans: List[Optional[Any]] = []
            real_plan_reindex = main_indexer.plan_reindex

            async def spy_plan_reindex(*args: Any) -> Any:
                plan = await real_plan_reindex(*args)
                reindex_plans.append(plan)
                return plan

            with patch.object(main_indexer, "plan_reindex", spy_plan_reindex):
                exit_code = await main_indexer.main()

        run["exit_code"] = exit_code
        run["reindex_plan"] = reindex_plans[0] if reindex_plans else None
        # The connection main() chose for this run (all LLM calls use it)
        run["environment"] = indexer.config.environment if reindex_plans else None
        run["connection_id"] = indexer.config.connection_id
        return exit_code


DATASET = [
    "/data/doc-a/cleaned.txt",
    "/data/doc-b/cleaned.txt",
    "/data/doc-c/cleaned.txt",
]


def _assert_one_knowledge_base(env: Environment, chunks: int) -> None:
    """One filled collection only, with the documents indexed exactly once."""
    assert [(s.collection, s.embedding_model, s.point_count) for s in env.stored] == [
        (AZURE_COLLECTION, AZURE_MODEL, chunks)
    ]


@pytest.mark.asyncio
async def test_order_test_then_production() -> None:
    env = Environment()

    # 1. Only a test connection exists: it may be created, and the sync indexes with it.
    assert env.create_connection("testing").allowed
    assert await env.run_indexer(changed_files=DATASET) == 0
    first = env.indexer_runs[-1]
    assert first["environment"] == "testing"
    assert first["connection_id"] == "testing-uuid"
    assert first["processed"] == ["doc-a", "doc-b", "doc-c"]
    _assert_one_knowledge_base(env, chunks=12)

    # 2. The production connection with the same embedding model: allowed and no re-index.
    result = env.create_connection("production")
    assert result.allowed
    assert not result.reindex_required
    assert result.mismatched_connections == []
    assert not env.status_requires_reindex()  # the sync will not force a run

    # 3. Next run: production is used, nothing is re-indexed from scratch.
    assert await env.run_indexer(changed_files=[]) == 0
    second = env.indexer_runs[-1]
    assert second["environment"] == "production"
    assert second["reindex_plan"] is None
    assert second["processed"] == []
    _assert_one_knowledge_base(env, chunks=12)

    # 4. A later CKB change is indexed with production into the same knowledge base.
    assert await env.run_indexer(changed_files=["/data/doc-d/cleaned.txt"]) == 0
    third = env.indexer_runs[-1]
    assert third["environment"] == "production"
    assert third["connection_id"] == "production-uuid"
    assert third["processed"] == ["doc-d"]
    _assert_one_knowledge_base(env, chunks=16)


@pytest.mark.asyncio
async def test_order_production_then_test() -> None:
    env = Environment()

    # 1. Only a production connection exists: indexing works as before.
    assert env.create_connection("production").allowed
    assert await env.run_indexer(changed_files=DATASET) == 0
    first = env.indexer_runs[-1]
    assert first["environment"] == "production"
    assert first["processed"] == ["doc-a", "doc-b", "doc-c"]
    _assert_one_knowledge_base(env, chunks=12)

    # 2. A test connection with the same embedding model: allowed and no re-index.
    result = env.create_connection("testing")
    assert result.allowed
    assert not result.reindex_required
    assert not env.status_requires_reindex()

    # 3. Next run: still production, nothing re-indexed, still one knowledge base.
    assert await env.run_indexer(changed_files=[]) == 0
    second = env.indexer_runs[-1]
    assert (second["environment"], second["connection_id"]) == (
        "production",
        "production-uuid",
    )
    assert second["reindex_plan"] is None
    assert second["processed"] == []
    _assert_one_knowledge_base(env, chunks=12)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["testing", "production"])
async def test_second_connection_on_another_embedding_platform(first: str) -> None:
    """Issue #587: a test connection cannot bring a second embedding model;
    production can, and then re-indexes the knowledge base (#588)."""
    env = Environment()
    assert env.create_connection(first).allowed
    await env.run_indexer(changed_files=DATASET)

    if first == "testing":
        # Production may switch: the knowledge base is then re-indexed (#588).
        result = env.create_connection("production", "aws", AWS_MODEL)
        assert result.allowed and result.reindex_required
    else:
        result = env.create_connection("testing", "aws", AWS_MODEL)
        assert not result.allowed
        assert result.required_model == AZURE_MODEL
        assert result.locked_by == "knowledge_base"
        assert "testing" not in env.connections


@pytest.mark.asyncio
async def test_no_connection_yet_is_skipped_and_indexed_once_one_exists() -> None:
    env = Environment()

    # Exit 4: vector_indexer_pipeline.sh does not save the agency hash.
    assert (
        await env.run_indexer(changed_files=DATASET)
        == main_indexer.EXIT_NO_LLM_CONNECTION
    )
    assert env.stored == []

    assert env.create_connection("testing").allowed
    assert await env.run_indexer(changed_files=DATASET) == 0
    _assert_one_knowledge_base(env, chunks=12)

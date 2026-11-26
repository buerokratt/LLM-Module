"""Integration tests for issues #588 and #589 against the real stack.

#588 (re-index when the production embedding model changes):
- /knowledge-base/status reports a re-index when production's embedding
  platform differs from the knowledge base (Azure -> AWS and AWS -> Azure),
  and none when it matches
- /knowledge-base/check-embedding lets production switch model (with
  reindex_required) and still blocks a testing connection
- the indexer plans the re-index from real Qdrant, and after the switch-over
  (old collection emptied, stale chunks pruned) nothing is left to re-index

#589 (test and production connections share one knowledge base):
- /orchestrate/test accepts the vault_uuid Ruuter sends (was a 422)
- deleting one connection leaves the knowledge base intact; the other one
  becomes the indexing connection

These need Qdrant, Postgres, Resql and the orchestration service, but no LLM
credentials: the knowledge base is seeded with synthetic points.
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

AZURE = "contextual_chunks_azure"
AWS = "contextual_chunks_aws"
PLATFORMS: Dict[str, Dict[str, Any]] = {
    "azure": {
        "collection": AZURE,
        "model": "text-embedding-3-large",
        "stored_model": "azure_openai/text-embedding-3-large",
        "dims": 3072,
    },
    "aws": {
        "collection": AWS,
        "model": "amazon.titan-embed-text-v2:0",
        "stored_model": "aws_bedrock/amazon.titan-embed-text-v2:0",
        "dims": 1024,
    },
}
QDRANT_URL = "http://localhost:6333"


def _ensure_collection(client: QdrantClient, name: str, dims: int) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            name, vectors_config=VectorParams(size=dims, distance=Distance.COSINE)
        )


@pytest.fixture
def empty_knowledge_base(qdrant_client: QdrantClient) -> Iterator[Callable[..., None]]:
    """Empty both collections for the test, then put the original points back.

    Returns ``seed(platform, document_hashes)``, which stores one synthetic
    chunk per document the way the indexer does (payload model and size).
    """
    backup: Dict[str, List[Any]] = {}
    for spec in PLATFORMS.values():
        name = spec["collection"]
        if qdrant_client.collection_exists(name):
            points, offset = [], None
            while True:
                batch, offset = qdrant_client.scroll(
                    name, limit=256, offset=offset, with_vectors=True, with_payload=True
                )
                points.extend(batch)
                if offset is None:
                    break
            backup[name] = points
            qdrant_client.delete_collection(name)
        _ensure_collection(qdrant_client, name, spec["dims"])

    def seed(platform: str, document_hashes: List[str]) -> None:
        spec = PLATFORMS[platform]
        qdrant_client.upsert(
            spec["collection"],
            points=[
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc}_chunk_000")),
                    vector=[0.01] * spec["dims"],
                    payload={
                        "chunk_id": f"{doc}_chunk_000",
                        "document_hash": doc,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "original_content": f"content of {doc}",
                        "contextual_content": f"content of {doc}",
                        "embedding_model": spec["stored_model"],
                        "vector_dimensions": spec["dims"],
                    },
                )
                for doc in document_hashes
            ],
            wait=True,
        )

    try:
        yield seed
    finally:
        for spec in PLATFORMS.values():
            name = spec["collection"]
            qdrant_client.delete_collection(name)
            _ensure_collection(qdrant_client, name, spec["dims"])
            if backup.get(name):
                qdrant_client.upsert(
                    name,
                    points=[
                        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
                        for p in backup[name]
                    ],
                    wait=True,
                )


@pytest.fixture
def llm_connections(postgres_client) -> Iterator[Callable[..., Dict[str, Any]]]:
    """Only the connections a test adds are active; the DB is restored after.

    Returns ``add(environment, platform)`` -> {"id", "vault_uuid"}.
    """
    cursor = postgres_client.cursor()
    cursor.execute(
        "SELECT id FROM llm_connections WHERE connection_status = 'active' "
        "AND environment IN ('production', 'testing')"
    )
    previously_active = [row[0] for row in cursor.fetchall()]
    if previously_active:
        cursor.execute(
            "UPDATE llm_connections SET connection_status = 'inactive' WHERE id = ANY(%s)",
            (previously_active,),
        )
    postgres_client.commit()
    inserted: List[int] = []

    def add(environment: str, platform: str) -> Dict[str, Any]:
        cursor.execute(
            "INSERT INTO llm_connections (connection_name, environment, llm_platform, "
            "llm_model, embedding_platform, embedding_model, monthly_budget, "
            "connection_status) VALUES (%s, %s, 'azure', 'gpt-4o-mini', %s, %s, 100, "
            "'active') RETURNING id, vault_uuid",
            (
                f"kb-sharing-{environment}-{platform}",
                environment,
                platform,
                PLATFORMS[platform]["model"],
            ),
        )
        row_id, vault_uuid = cursor.fetchone()
        postgres_client.commit()
        inserted.append(row_id)
        return {"id": row_id, "vault_uuid": str(vault_uuid)}

    try:
        yield add
    finally:
        if inserted:
            cursor.execute(
                "DELETE FROM llm_connections WHERE id = ANY(%s)", (inserted,)
            )
        if previously_active:
            cursor.execute(
                "UPDATE llm_connections SET connection_status = 'active' WHERE id = ANY(%s)",
                (previously_active,),
            )
        postgres_client.commit()


def _status(orchestration_client) -> Dict[str, Any]:
    response = orchestration_client.get(
        f"{orchestration_client.base_url}/knowledge-base/status", timeout=30
    )
    assert response.status_code == 200, response.text
    return response.json()


def _indexer_modules():
    sys.path.append(str(Path(__file__).parent.parent.parent / "src" / "vector_indexer"))
    from vector_indexer import main_indexer, qdrant_manager

    return main_indexer, qdrant_manager


# ---------------------------------------------------------------------------
# #588
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("old, new", [("azure", "aws"), ("aws", "azure")])
def test_status_requires_reindex_when_production_platform_differs(
    orchestration_client, empty_knowledge_base, llm_connections, old: str, new: str
) -> None:
    empty_knowledge_base(old, ["doc-a", "doc-b"])
    llm_connections("production", new)

    status = _status(orchestration_client)

    assert status["reindex_required"] is True
    assert status["indexing_environment"] == "production"
    assert status["indexing_platform"] == new
    assert [m["platform"] for m in status["locked_models"]] == [old]


def test_status_up_to_date_when_platform_matches(
    orchestration_client, empty_knowledge_base, llm_connections
) -> None:
    empty_knowledge_base("azure", ["doc-a"])
    llm_connections("testing", "azure")
    llm_connections("production", "azure")

    assert _status(orchestration_client)["reindex_required"] is False


def test_status_uses_testing_when_there_is_no_production(
    orchestration_client, empty_knowledge_base, llm_connections
) -> None:
    llm_connections("testing", "aws")

    status = _status(orchestration_client)

    assert status["indexing_environment"] == "testing"
    assert status["reindex_required"] is False  # empty knowledge base


@pytest.mark.parametrize(
    "environment, allowed, reindex",
    [("production", True, True), ("testing", False, False)],
)
def test_check_embedding_lets_only_production_switch_model(
    orchestration_client,
    empty_knowledge_base,
    llm_connections,
    environment: str,
    allowed: bool,
    reindex: bool,
) -> None:
    empty_knowledge_base("azure", ["doc-a"])

    response = orchestration_client.post(
        f"{orchestration_client.base_url}/knowledge-base/check-embedding",
        json={
            "embedding_platform": "aws",
            "embedding_model": PLATFORMS["aws"]["model"],
            "deployment_environment": environment,
        },
        timeout=60,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["allowed"] is allowed
    assert body["reindex_required"] is reindex


@pytest.mark.parametrize("old, new", [("azure", "aws"), ("aws", "azure")])
def test_reindex_plan_and_switch_over_on_real_qdrant(
    orchestration_client, empty_knowledge_base, llm_connections, old: str, new: str
) -> None:
    main_indexer, qdrant_manager_module = _indexer_modules()
    empty_knowledge_base(old, ["doc-a", "doc-b"])
    connection = main_indexer.IndexingConnection(
        "production", "uuid", PLATFORMS[new]["model"], new
    )
    llm_connections("production", new)

    async def run() -> None:
        config = type("Config", (), {"qdrant_url": QDRANT_URL})()
        async with qdrant_manager_module.QdrantManager(config) as manager:
            plan = await main_indexer.plan_reindex(manager, connection)
            assert plan is not None
            assert plan.target_collection == PLATFORMS[new]["collection"]
            assert plan.old_collections == [PLATFORMS[old]["collection"]]

            # Re-index done: doc-a re-embedded, plus a stale chunk of a
            # document no longer in the dataset from an earlier partial run.
            empty_knowledge_base(new, ["doc-a", "doc-gone"])
            assert (
                await manager.count_document_chunks(plan.target_collection, "doc-a")
                == 1
            )

            for collection in plan.old_collections:
                await manager.clear_collection(collection)
            await manager.delete_chunks_not_in(plan.target_collection, {"doc-a"})

            stored = await manager.get_stored_embeddings()
            assert [(s.platform, s.point_count) for s in stored] == [(new, 1)]
            assert await main_indexer.plan_reindex(manager, connection) is None

    asyncio.run(run())

    assert _status(orchestration_client)["reindex_required"] is False


# ---------------------------------------------------------------------------
# #589
# ---------------------------------------------------------------------------


def test_orchestrate_test_accepts_the_vault_uuid(orchestration_client) -> None:
    response = orchestration_client.post(
        f"{orchestration_client.base_url}/orchestrate/test",
        json={
            "message": "What are the opening hours?",
            "environment": "testing",
            "connectionId": str(uuid.uuid4()),
        },
        timeout=120,
    )

    # Unknown connection: may fail later on, but the request itself is valid.
    assert response.status_code != 422, response.text


def test_deleting_a_connection_keeps_the_knowledge_base(
    orchestration_client,
    ruuter_private_client,
    qdrant_client: QdrantClient,
    empty_knowledge_base,
    llm_connections,
) -> None:
    empty_knowledge_base("azure", ["doc-a", "doc-b"])
    llm_connections("testing", "azure")
    production = llm_connections("production", "azure")

    response = ruuter_private_client.post(
        f"{ruuter_private_client.base_url}/rag-search/llm-connections/delete",
        json={"connection_id": production["id"]},
        headers={"Cookie": "customJwtCookie=test-session-token"},
        timeout=60,
    )
    assert response.status_code == 200, response.text

    assert qdrant_client.count(AZURE, exact=True).count == 2
    status = _status(orchestration_client)
    assert status["indexing_environment"] == "testing"
    assert status["reindex_required"] is False

"""
Integration tests for the vector indexing pipeline.

These tests verify the full flow:
1. Upload document to MinIO
2. Generate presigned URL
3. Run VectorIndexer
4. Verify embeddings in Qdrant
"""

import asyncio
import tempfile
from pathlib import Path
from datetime import timedelta
import json
import sys

import pytest
from minio import Minio
from qdrant_client import QdrantClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestIndexingPipeline:
    """Test the complete indexing pipeline from MinIO to Qdrant."""

    def test_minio_connection(self, minio_client: Minio):
        """Verify MinIO is accessible."""
        # List buckets to verify connection
        buckets = minio_client.list_buckets()
        assert buckets is not None

    def test_qdrant_connection(self, qdrant_client: QdrantClient):
        """Verify Qdrant is accessible."""
        # Get collections to verify connection
        collections = qdrant_client.get_collections()
        assert collections is not None

    def test_create_and_upload_document(self, minio_client: Minio, test_bucket: str):
        """Test document upload to MinIO."""
        # Verify bucket was created
        assert minio_client.bucket_exists(test_bucket)

        # Create and upload a simple test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name

        try:
            minio_client.fput_object(test_bucket, "test.txt", temp_path)

            # Verify object exists
            stat = minio_client.stat_object(test_bucket, "test.txt")
            assert stat is not None
            assert stat.size > 0
        finally:
            Path(temp_path).unlink()

    def test_presigned_url_generation(self, minio_client: Minio, test_document):
        """Test presigned URL generation."""
        bucket_name, object_prefix, _ = test_document

        # Generate presigned URL
        url = minio_client.presigned_get_object(
            bucket_name, f"{object_prefix}/cleaned.txt", expires=timedelta(hours=1)
        )

        assert url is not None
        assert "localhost:9000" in url
        assert bucket_name in url

    def test_document_structure(self, minio_client: Minio, test_document):
        """Verify test document has correct structure."""
        bucket_name, object_prefix, local_path = test_document

        # Check local files exist
        cleaned_file = local_path / "cleaned.txt"
        meta_file = local_path / "source.meta.json"

        assert cleaned_file.exists()
        assert meta_file.exists()

        # Verify content
        content = cleaned_file.read_text()
        assert "RAG" in content
        assert "integration testing" in content

        # Verify metadata
        meta = json.loads(meta_file.read_text())
        assert meta["source"] == "integration_test"
        assert "title" in meta


class TestQdrantOperations:
    """Test Qdrant-specific operations."""

    def test_collection_operations(self, qdrant_client: QdrantClient):
        """Test creating and querying collections."""
        from qdrant_client.models import Distance, VectorParams

        test_collection = "test_integration_collection"

        try:
            # Create collection
            qdrant_client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

            # Verify collection exists
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert test_collection in collection_names

            # Get collection info
            info = qdrant_client.get_collection(test_collection)
            assert info.config.params.vectors.size == 1536

        finally:
            # Cleanup
            try:
                qdrant_client.delete_collection(test_collection)
            except Exception:
                pass

    def test_point_operations(self, qdrant_client: QdrantClient):
        """Test inserting and querying points."""
        from qdrant_client.models import Distance, VectorParams, PointStruct

        test_collection = "test_points_collection"

        try:
            # Create collection
            qdrant_client.create_collection(
                collection_name=test_collection,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE),
            )

            # Insert points
            points = [
                PointStruct(
                    id=1,
                    vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"document_hash": "test123", "text": "test chunk"},
                ),
                PointStruct(
                    id=2,
                    vector=[0.2, 0.3, 0.4, 0.5],
                    payload={"document_hash": "test123", "text": "another chunk"},
                ),
            ]

            qdrant_client.upsert(collection_name=test_collection, points=points)

            # Query by filter
            results = qdrant_client.scroll(
                collection_name=test_collection,
                scroll_filter={
                    "must": [{"key": "document_hash", "match": {"value": "test123"}}]
                },
                limit=10,
            )

            assert len(results[0]) == 2

        finally:
            # Cleanup
            try:
                qdrant_client.delete_collection(test_collection)
            except Exception:
                pass


# Resql as exposed by docker-compose-test.yml (resql: 8082:8082).
RESQL_TEST_URL = "http://localhost:8082/rag-search"


@pytest.fixture
def isolated_llm_connections(postgres_client):
    """Give a test an empty set of active connections, then restore the DB.

    Other integration tests share the database and may leave active
    connections behind, so this deactivates them for the duration of the test
    and returns a helper that inserts a connection row (as the GUI's add flow
    would, minus the Vault secrets, which the indexer's resolver never reads).
    """
    cursor = postgres_client.cursor()
    cursor.execute(
        "SELECT id FROM llm_connections WHERE connection_status = 'active' "
        "AND environment IN ('production', 'testing')"
    )
    previously_active = [row[0] for row in cursor.fetchall()]
    if previously_active:
        cursor.execute(
            "UPDATE llm_connections SET connection_status = 'inactive' "
            "WHERE id = ANY(%s)",
            (previously_active,),
        )
    postgres_client.commit()

    inserted: list = []

    def add_connection(environment: str) -> str:
        cursor.execute(
            "INSERT INTO llm_connections (connection_name, environment, "
            "llm_platform, llm_model, embedding_platform, embedding_model, "
            "monthly_budget, connection_status) "
            "VALUES (%s, %s, 'azure', 'gpt-4o-mini', 'azure', "
            "'text-embedding-3-large', 100, 'active') "
            "RETURNING id, vault_uuid",
            (f"issue-805-{environment}", environment),
        )
        row_id, vault_uuid = cursor.fetchone()
        postgres_client.commit()
        inserted.append(row_id)
        return str(vault_uuid)

    try:
        yield add_connection
    finally:
        if inserted:
            cursor.execute(
                "DELETE FROM llm_connections WHERE id = ANY(%s)", (inserted,)
            )
        if previously_active:
            cursor.execute(
                "UPDATE llm_connections SET connection_status = 'active' "
                "WHERE id = ANY(%s)",
                (previously_active,),
            )
        postgres_client.commit()


class TestIndexingConnectionSelection:
    """Issue #805: the indexer picks production, else testing, via real Resql.

    The second connection only changes which connection future runs use; it
    does not trigger indexing (that happens only when the CKB hash changes),
    and re-indexing is prevented by the diff detector and deterministic point
    IDs, which these tests do not need an LLM to show.
    """

    @staticmethod
    def _resolve():
        sys.path.insert(
            0, str(Path(__file__).parent.parent.parent / "src" / "vector_indexer")
        )
        from vector_indexer.main_indexer import resolve_indexing_connection

        return asyncio.run(resolve_indexing_connection(RESQL_TEST_URL))

    def test_no_connection_resolves_none(self, isolated_llm_connections):
        assert self._resolve() is None

    def test_order_test_then_production(self, isolated_llm_connections):
        testing_uuid = isolated_llm_connections("testing")

        first = self._resolve()
        assert first is not None
        assert (first.environment, first.vault_uuid) == ("testing", testing_uuid)

        production_uuid = isolated_llm_connections("production")

        second = self._resolve()
        assert second is not None
        assert (second.environment, second.vault_uuid) == (
            "production",
            production_uuid,
        )

    def test_order_production_then_test(self, isolated_llm_connections):
        production_uuid = isolated_llm_connections("production")

        first = self._resolve()
        assert first is not None
        assert (first.environment, first.vault_uuid) == (
            "production",
            production_uuid,
        )

        isolated_llm_connections("testing")

        second = self._resolve()
        assert second is not None
        assert (second.environment, second.vault_uuid) == (
            "production",
            production_uuid,
        )

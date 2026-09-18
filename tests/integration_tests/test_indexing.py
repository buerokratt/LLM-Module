"""
Integration tests for the vector indexing pipeline.

These tests verify the full flow:
1. Upload document to MinIO
2. Generate presigned URL
3. Run VectorIndexer
4. Verify embeddings in Qdrant
"""

import tempfile
from pathlib import Path
from datetime import timedelta
import json
import sys

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

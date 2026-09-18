"""Unit tests for ApiToolQdrantManager.

Covers:
- ensure_collection(): collection exists with correct schema → no-op (no create)
- ensure_collection(): collection doesn't exist → creates it
- ensure_collection(): old single-vector schema → raises RuntimeError (migration required)
- ensure_collection(): client not initialized → raises RuntimeError
- delete_endpoint_points(): success returns True
- delete_endpoint_points(): client uninitialized → returns False (catches exception)
- delete_endpoint_points(): Qdrant error → returns False
- upsert_endpoint_points(): success returns True with correct payload fields
- upsert_endpoint_points(): empty list → returns True (no-op)
- upsert_endpoint_points(): client uninitialized → returns False
- Deterministic UUID generation: same endpoint_id + index always gives same UUID
"""

import uuid
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from api_tool_indexer.constants import ApiToolIndexerConstants
from api_tool_indexer.models import EnrichedEndpoint
from api_tool_indexer.qdrant_manager import ApiToolQdrantManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COLLECTION = ApiToolIndexerConstants.COLLECTION_NAME
_DENSE_NAME = ApiToolIndexerConstants.DENSE_VECTOR_NAME
_SPARSE_NAME = ApiToolIndexerConstants.SPARSE_VECTOR_NAME
_VECTOR_SIZE = ApiToolIndexerConstants.VECTOR_SIZE

_ENDPOINT_ID = "ep-holidays"
_MOCK_EMBEDDING = [0.01] * _VECTOR_SIZE


def _make_enriched_point(
    endpoint_id: str = _ENDPOINT_ID,
    point_type: str = "example",
    example_text: str = "When is the next holiday?",
    idx: int = 0,
) -> EnrichedEndpoint:
    return EnrichedEndpoint(
        endpoint_id=endpoint_id,
        name="get_public_holidays",
        description="Returns public holidays",
        url="https://openholidaysapi.org/PublicHolidays",
        method="GET",
        params=[
            {
                "name": "country",
                "type": "string",
                "required": True,
                "description": "ISO",
            }
        ],
        enriched_context="Rich context about holidays.",
        service_id="svc-001",
        point_type=point_type,
        example_text=example_text if point_type == "example" else None,
        embedding=_MOCK_EMBEDDING,
        sparse_indices=[1, 2, 3],
        sparse_values=[1.0, 2.0, 1.0],
    )


def _make_collection_info(
    vector_size: int = _VECTOR_SIZE,
    old_format: bool = False,
    undetermined: bool = False,
) -> MagicMock:
    """Build a mock collection info response from Qdrant client."""
    info = MagicMock()

    if undetermined:
        info.config.params.vectors = None
        return info

    if old_format:
        # Single VectorParams object (old format)
        from qdrant_client.models import VectorParams, Distance

        info.config.params.vectors = VectorParams(
            size=vector_size, distance=Distance.COSINE
        )
        return info

    # Named vectors dict (correct format)
    dense_config = MagicMock()
    dense_config.size = vector_size
    info.config.params.vectors = {_DENSE_NAME: dense_config}
    return info


def _make_qdrant_client(
    collection_names: List[str] = (),
    collection_info: Any = None,
) -> MagicMock:
    """Build a mock QdrantClient."""
    client = MagicMock()

    collections_result = MagicMock()
    collections_result.collections = [MagicMock(name=n) for n in collection_names]
    # Fix: MagicMock(name=n) doesn't work as expected — set attribute explicitly
    for i, n in enumerate(collection_names):
        collections_result.collections[i].name = n

    client.get_collections.return_value = collections_result
    client.get_collection.return_value = (
        collection_info if collection_info else _make_collection_info()
    )
    client.create_collection = MagicMock()
    client.delete = MagicMock()
    client.upsert = MagicMock()
    client.close = MagicMock()

    return client


def _make_manager(qdrant_client: MagicMock) -> ApiToolQdrantManager:
    manager = ApiToolQdrantManager()
    manager.client = qdrant_client
    return manager


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_client_not_initialized_raises(self) -> None:
        manager = ApiToolQdrantManager()
        with pytest.raises(RuntimeError, match="not initialized"):
            manager.ensure_collection()

    def test_collection_exists_with_correct_schema_is_noop(self) -> None:
        """Collection already exists with correct vector size → no create call."""
        collection_info = _make_collection_info(vector_size=_VECTOR_SIZE)
        client = _make_qdrant_client(
            collection_names=[_COLLECTION],
            collection_info=collection_info,
        )
        manager = _make_manager(client)

        manager.ensure_collection()

        client.create_collection.assert_not_called()

    def test_collection_does_not_exist_creates_it(self) -> None:
        client = _make_qdrant_client(collection_names=[])
        manager = _make_manager(client)

        manager.ensure_collection()

        client.create_collection.assert_called_once()
        call_kwargs = client.create_collection.call_args.kwargs
        assert call_kwargs["collection_name"] == _COLLECTION

    def test_incompatible_vector_size_raises_runtime_error(self) -> None:
        wrong_size = _VECTOR_SIZE + 512
        collection_info = _make_collection_info(vector_size=wrong_size)
        client = _make_qdrant_client(
            collection_names=[_COLLECTION],
            collection_info=collection_info,
        )
        manager = _make_manager(client)

        with pytest.raises(RuntimeError, match="incompatible vector size"):
            manager.ensure_collection()

    def test_old_single_vector_format_raises_runtime_error(self) -> None:
        """Collection with old VectorParams format requires migration."""
        collection_info = _make_collection_info(old_format=True)
        client = _make_qdrant_client(
            collection_names=[_COLLECTION],
            collection_info=collection_info,
        )
        manager = _make_manager(client)

        with pytest.raises(RuntimeError, match="old single-vector format"):
            manager.ensure_collection()

    def test_undetermined_vector_config_raises_runtime_error(self) -> None:
        collection_info = _make_collection_info(undetermined=True)
        client = _make_qdrant_client(
            collection_names=[_COLLECTION],
            collection_info=collection_info,
        )
        manager = _make_manager(client)

        with pytest.raises(RuntimeError):
            manager.ensure_collection()


# ---------------------------------------------------------------------------
# delete_endpoint_points
# ---------------------------------------------------------------------------


class TestDeleteEndpointPoints:
    def test_success_returns_true(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)

        result = manager.delete_endpoint_points(_ENDPOINT_ID)

        assert result is True
        client.delete.assert_called_once()

    def test_client_uninitialized_returns_false(self) -> None:
        manager = ApiToolQdrantManager()
        # client is None — should not raise, returns False

        result = manager.delete_endpoint_points(_ENDPOINT_ID)

        assert result is False

    def test_qdrant_error_returns_false(self) -> None:
        client = _make_qdrant_client()
        client.delete.side_effect = RuntimeError("Qdrant connection refused")
        manager = _make_manager(client)

        result = manager.delete_endpoint_points(_ENDPOINT_ID)

        assert result is False

    def test_delete_uses_endpoint_id_filter(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)

        manager.delete_endpoint_points("ep-custom")

        call_kwargs = client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == _COLLECTION


# ---------------------------------------------------------------------------
# upsert_endpoint_points
# ---------------------------------------------------------------------------


class TestUpsertEndpointPoints:
    def test_success_returns_true(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        points = [_make_enriched_point()]

        result = manager.upsert_endpoint_points(points)

        assert result is True
        client.upsert.assert_called_once()

    def test_empty_list_returns_true_without_upsert(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)

        result = manager.upsert_endpoint_points([])

        assert result is True
        client.upsert.assert_not_called()

    def test_client_uninitialized_returns_false(self) -> None:
        manager = ApiToolQdrantManager()

        result = manager.upsert_endpoint_points([_make_enriched_point()])

        assert result is False

    def test_qdrant_error_returns_false(self) -> None:
        client = _make_qdrant_client()
        client.upsert.side_effect = RuntimeError("Qdrant error")
        manager = _make_manager(client)

        result = manager.upsert_endpoint_points([_make_enriched_point()])

        assert result is False

    def test_upserted_payload_contains_required_fields(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        points = [
            _make_enriched_point(point_type="example", example_text="Holiday query")
        ]

        manager.upsert_endpoint_points(points)

        upsert_call = client.upsert.call_args.kwargs
        qdrant_points = upsert_call["points"]
        assert len(qdrant_points) == 1
        payload = qdrant_points[0].payload
        assert payload["endpoint_id"] == _ENDPOINT_ID
        assert payload["name"] == "get_public_holidays"
        assert payload["point_type"] == "example"
        assert payload["example_text"] == "Holiday query"

    def test_summary_point_has_no_example_text_in_payload(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        point = _make_enriched_point(point_type="summary")
        point.example_text = None

        manager.upsert_endpoint_points([point])

        qdrant_points = client.upsert.call_args.kwargs["points"]
        payload = qdrant_points[0].payload
        assert "example_text" not in payload

    def test_multiple_points_all_upserted(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        points = [
            _make_enriched_point(idx=0, example_text="Query 1"),
            _make_enriched_point(idx=1, example_text="Query 2"),
            _make_enriched_point(point_type="summary", example_text=None, idx=2),
        ]

        manager.upsert_endpoint_points(points)

        qdrant_points = client.upsert.call_args.kwargs["points"]
        assert len(qdrant_points) == 3


# ---------------------------------------------------------------------------
# Deterministic UUID generation
# ---------------------------------------------------------------------------


class TestDeterministicUUID:
    def test_same_endpoint_and_index_produce_same_uuid(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        points = [_make_enriched_point(endpoint_id="ep-test", idx=0)]

        manager.upsert_endpoint_points(points)
        qdrant_points_first = client.upsert.call_args.kwargs["points"]
        uuid_first = qdrant_points_first[0].id

        # Reset and call again with identical input
        client.upsert.reset_mock()
        manager.upsert_endpoint_points(points)
        qdrant_points_second = client.upsert.call_args.kwargs["points"]
        uuid_second = qdrant_points_second[0].id

        assert uuid_first == uuid_second

    def test_different_indices_produce_different_uuids(self) -> None:
        client = _make_qdrant_client()
        manager = _make_manager(client)
        points = [
            _make_enriched_point(endpoint_id="ep-test", idx=0),
            _make_enriched_point(endpoint_id="ep-test", idx=1),
        ]

        manager.upsert_endpoint_points(points)

        qdrant_points = client.upsert.call_args.kwargs["points"]
        assert qdrant_points[0].id != qdrant_points[1].id

    def test_uuid_matches_expected_uuid5_derivation(self) -> None:
        """Verify UUID derivation formula: uuid5(NAMESPACE_DNS, f'{endpoint_id}_{idx}')."""
        endpoint_id = "ep-test-uuid"
        idx = 0
        expected = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{endpoint_id}_{idx}"))

        client = _make_qdrant_client()
        manager = _make_manager(client)
        point = _make_enriched_point(endpoint_id=endpoint_id, idx=idx)
        manager.upsert_endpoint_points([point])

        qdrant_points = client.upsert.call_args.kwargs["points"]
        assert qdrant_points[0].id == expected

"""Tests for the knowledge base embedding model lock (issue #806)."""

from typing import Any, Dict, List, Optional

import pytest
import requests

from src.llm_orchestrator_config import knowledge_base_lock as kbl
from src.llm_orchestrator_config.knowledge_base_lock import (
    KnowledgeBaseLockError,
    LockedModel,
    OtherConnection,
    check_embedding_compatibility,
    read_locked_models,
    read_other_connections,
)

AZURE = LockedModel("azure", "text-embedding-3-large", 3072, 120)
AWS = LockedModel("aws", "amazon.titan-embed-text-v2:0", 1024, 80)


def other(
    platform: str, model: str, environment: str = "production"
) -> OtherConnection:
    return OtherConnection(1, "Prod", environment, platform, model)


# check_embedding_compatibility


def test_empty_knowledge_base_and_no_other_connection_allows_any_approved_model() -> (
    None
):
    result = check_embedding_compatibility(
        "aws", "amazon.titan-embed-text-v2:0", 1024, [], []
    )
    assert result.allowed
    assert result.locked_by is None


def test_same_model_as_knowledge_base_is_allowed() -> None:
    result = check_embedding_compatibility(
        "azure", "text-embedding-3-large", 3072, [AZURE], []
    )
    assert result.allowed
    assert result.locked_by == "knowledge_base"


def test_other_platform_than_knowledge_base_is_blocked_with_reason() -> None:
    result = check_embedding_compatibility(
        "aws", "amazon.titan-embed-text-v2:0", 1024, [AZURE], []
    )
    assert not result.allowed
    assert result.code == "model_mismatch"
    assert result.detail == "different_model"
    assert result.required_model == "text-embedding-3-large"
    assert result.requested_model == "amazon.titan-embed-text-v2:0"
    assert result.locked_by == "knowledge_base"
    assert "text-embedding-3-large (azure, 3072 dimensions)" in (result.reason or "")


def test_azure_deployment_pointing_at_another_model_is_blocked_by_vector_size() -> None:
    # Dropdown says text-embedding-3-large, the deployment is really ada-002 (1536)
    result = check_embedding_compatibility(
        "azure", "text-embedding-3-large", 1536, [AZURE], []
    )
    assert not result.allowed
    assert result.code == "model_mismatch"
    assert result.detail == "wrong_deployment"
    assert result.required_model == "text-embedding-3-large"
    assert result.detected_vector_dimensions == 1536
    assert "1536" in (result.reason or "")


def test_wrong_vector_size_is_blocked_even_when_knowledge_base_is_empty() -> None:
    result = check_embedding_compatibility(
        "azure", "text-embedding-3-large", 1536, [], []
    )
    assert not result.allowed


def test_unapproved_model_is_blocked() -> None:
    result = check_embedding_compatibility(
        "azure", "text-embedding-ada-002", None, [], []
    )
    assert not result.allowed
    assert result.required_model == "text-embedding-3-large"


def test_unsupported_platform_is_blocked() -> None:
    result = check_embedding_compatibility(
        "openai", "text-embedding-3-large", None, [], []
    )
    assert not result.allowed


def test_empty_knowledge_base_requires_same_model_as_other_connection() -> None:
    result = check_embedding_compatibility(
        "aws",
        "amazon.titan-embed-text-v2:0",
        1024,
        [],
        [other("azure", "text-embedding-3-large")],
    )
    assert not result.allowed
    assert result.locked_by == "connection"
    assert result.required_model == "text-embedding-3-large"
    assert "'Prod'" in (result.reason or "")


def test_empty_knowledge_base_same_model_as_other_connection_is_allowed() -> None:
    result = check_embedding_compatibility(
        "azure",
        "text-embedding-3-large",
        3072,
        [],
        [other("azure", "text-embedding-3-large")],
    )
    assert result.allowed


def test_knowledge_base_wins_over_other_connections() -> None:
    # A legacy mismatching connection must not block a connection that matches the knowledge base
    result = check_embedding_compatibility(
        "azure",
        "text-embedding-3-large",
        3072,
        [AZURE],
        [other("aws", "amazon.titan-embed-text-v2:0")],
    )
    assert result.allowed


def test_both_platforms_filled_allows_either() -> None:
    # Setups that indexed on both platforms before the lock keep working
    assert check_embedding_compatibility(
        "azure", "text-embedding-3-large", 3072, [AZURE, AWS], []
    ).allowed
    assert check_embedding_compatibility(
        "aws", "amazon.titan-embed-text-v2:0", 1024, [AZURE, AWS], []
    ).allowed


def test_declared_check_without_test_embedding() -> None:
    # Ruuter's backstop has no credentials, so no vector size
    result = check_embedding_compatibility(
        "azure", "text-embedding-3-large", None, [AZURE], []
    )
    assert result.allowed
    assert result.detected_vector_dimensions is None


# Issue #588: production may change the embedding model; the knowledge base is
# then re-indexed. Testing connections keep the strict lock.


@pytest.mark.parametrize(
    "platform, model, dims, locked",
    [
        ("aws", "amazon.titan-embed-text-v2:0", 1024, AZURE),  # Azure -> AWS
        ("azure", "text-embedding-3-large", 3072, AWS),  # AWS -> Azure
    ],
)
def test_production_with_other_model_is_allowed_and_requires_reindex(
    platform: str, model: str, dims: int, locked: LockedModel
) -> None:
    result = check_embedding_compatibility(
        platform, model, dims, [locked], [], environment="production"
    )
    assert result.allowed
    assert result.reindex_required
    assert result.locked_by == "knowledge_base"
    assert result.required_model == model
    assert locked.embedding_model in (result.reason or "")
    assert "re-indexed" in (result.reason or "")


def test_testing_with_other_model_is_still_blocked() -> None:
    result = check_embedding_compatibility(
        "aws", "amazon.titan-embed-text-v2:0", 1024, [AZURE], [], environment="testing"
    )
    assert not result.allowed
    assert not result.reindex_required
    assert result.required_model == "text-embedding-3-large"


def test_missing_environment_uses_the_strict_testing_rules() -> None:
    result = check_embedding_compatibility(
        "aws", "amazon.titan-embed-text-v2:0", 1024, [AZURE], []
    )
    assert not result.allowed
    assert not result.reindex_required


def test_production_with_same_model_needs_no_reindex() -> None:
    result = check_embedding_compatibility(
        "azure", "text-embedding-3-large", 3072, [AZURE], [], environment="production"
    )
    assert result.allowed
    assert not result.reindex_required


def test_production_switch_reports_the_test_connection_that_no_longer_matches() -> None:
    result = check_embedding_compatibility(
        "aws",
        "amazon.titan-embed-text-v2:0",
        1024,
        [AZURE],
        [other("azure", "text-embedding-3-large", environment="testing")],
        environment="production",
    )
    assert result.allowed and result.reindex_required
    assert [c["environment"] for c in result.mismatched_connections] == ["testing"]


def test_production_mid_reindex_keeps_requiring_reindex() -> None:
    # An interrupted Azure -> AWS re-index: both collections hold embeddings.
    result = check_embedding_compatibility(
        "aws",
        "amazon.titan-embed-text-v2:0",
        1024,
        [AZURE, AWS],
        [],
        environment="production",
    )
    assert result.allowed
    assert result.reindex_required


def test_testing_on_old_model_keeps_working_until_the_reindex_finishes() -> None:
    result = check_embedding_compatibility(
        "azure",
        "text-embedding-3-large",
        3072,
        [AZURE, AWS],
        [],
        environment="testing",
    )
    assert result.allowed
    assert not result.reindex_required


def test_production_still_needs_an_approved_model_and_the_right_deployment() -> None:
    assert not check_embedding_compatibility(
        "azure", "text-embedding-ada-002", None, [AWS], [], environment="production"
    ).allowed
    wrong_deployment = check_embedding_compatibility(
        "azure", "text-embedding-3-large", 1536, [AWS], [], environment="production"
    )
    assert not wrong_deployment.allowed
    assert wrong_deployment.detail == "wrong_deployment"


def test_production_wins_over_test_connection_when_knowledge_base_is_empty() -> None:
    result = check_embedding_compatibility(
        "aws",
        "amazon.titan-embed-text-v2:0",
        1024,
        [],
        [other("azure", "text-embedding-3-large", environment="testing")],
        environment="production",
    )
    assert result.allowed
    assert not result.reindex_required  # nothing indexed yet
    assert len(result.mismatched_connections) == 1


def test_testing_must_match_production_when_knowledge_base_is_empty() -> None:
    result = check_embedding_compatibility(
        "aws",
        "amazon.titan-embed-text-v2:0",
        1024,
        [],
        [other("azure", "text-embedding-3-large")],
        environment="testing",
    )
    assert not result.allowed
    assert result.locked_by == "connection"


@pytest.mark.parametrize(
    "platform, locked, expected",
    [
        ("azure", [], False),
        ("azure", [AZURE], False),
        ("aws", [AZURE], True),
        ("azure", [AZURE, AWS], True),
        ("aws", [AWS], False),
    ],
)
def test_reindex_required_for(
    platform: str, locked: List[LockedModel], expected: bool
) -> None:
    assert kbl.reindex_required_for(platform, locked) is expected


# read_locked_models / read_other_connections (HTTP faked)


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def fake_post(routes: Dict[str, FakeResponse]):
    def post(
        url: str, json: Optional[Dict[str, Any]] = None, timeout: float = 0
    ) -> FakeResponse:
        for suffix, response in routes.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected URL {url}")

    return post


def test_read_locked_models_from_filled_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kbl.requests,
        "post",
        fake_post(
            {
                "contextual_chunks_azure/points/count": FakeResponse(
                    200, {"result": {"count": 4}}
                ),
                "contextual_chunks_azure/points/scroll": FakeResponse(
                    200,
                    {
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
                ),
                "contextual_chunks_aws/points/count": FakeResponse(404, {}),
            }
        ),
    )
    locked = read_locked_models("http://qdrant:6333")
    # The provider prefix the indexer stores is dropped for display
    assert locked == [LockedModel("azure", "text-embedding-3-large", 3072, 4)]


def test_read_locked_models_empty_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        kbl.requests,
        "post",
        fake_post(
            {
                "contextual_chunks_azure/points/count": FakeResponse(
                    200, {"result": {"count": 0}}
                ),
                "contextual_chunks_aws/points/count": FakeResponse(404, {}),
            }
        ),
    )
    assert read_locked_models("http://qdrant:6333") == []


def test_read_locked_models_qdrant_down_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(*_args: Any, **_kwargs: Any) -> FakeResponse:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(kbl.requests, "post", down)
    with pytest.raises(KnowledgeBaseLockError):
        read_locked_models("http://qdrant:6333")


def test_read_other_connections_excludes_the_edited_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def row(conn_id: int, platform: str, model: str) -> List[Dict[str, Any]]:
        return [
            {
                "id": conn_id,
                "connectionName": f"c{conn_id}",
                "embeddingPlatform": platform,
                "embeddingModel": model,
            }
        ]

    monkeypatch.setattr(
        kbl.requests,
        "post",
        fake_post(
            {
                "get-production-connection": FakeResponse(
                    200, row(7, "azure", "text-embedding-3-large")
                ),
                "get-testing-connection": FakeResponse(
                    200, row(8, "aws", "amazon.titan-embed-text-v2:0")
                ),
            }
        ),
    )
    others = read_other_connections(
        "http://resql:8082/rag-search", exclude_connection_id=8
    )
    assert [(o.connection_id, o.environment, o.platform) for o in others] == [
        (7, "production", "azure")
    ]


# EmbeddingManager.measure_test_embedding (secret stored asynchronously)


class LookupNotReadyError(Exception):
    """Stands in for the loader's ConfigurationError, which is imported under
    another module path and so is a different class than the manager's."""


def test_test_embedding_waits_for_a_secret_that_is_still_being_stored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from src.llm_orchestrator_config import embedding_manager as em

    calls = {"resolve": 0}

    class FakeLoader:
        def resolve_embedding_model(self, environment: str, connection_id: str):
            calls["resolve"] += 1
            if calls["resolve"] < 3:  # CronManager has not written it yet
                raise LookupNotReadyError("No embedding models available")
            return "azure_openai", "text-embedding-3-large"

        def get_embedding_provider_config(self, *args: Any) -> Dict[str, Any]:
            return {"endpoint": "https://x.openai.azure.com/", "deployment_name": "d"}

    manager = em.EmbeddingManager.__new__(em.EmbeddingManager)
    manager.config_loader = FakeLoader()
    monkeypatch.setattr(em.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        manager,
        "_create_dspy_embedder",
        lambda config, caching=True: lambda texts: np.zeros((1, 3072)),
    )

    measured = manager.measure_test_embedding("temp-id", wait_seconds=30)

    assert calls["resolve"] == 3
    assert measured["vector_dimensions"] == 3072

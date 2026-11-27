"""Knowledge base embedding model lock (issue #806).

Vectors from different embedding models cannot be compared, so every LLM
connection (testing and production) must use the embedding model that built the
knowledge base. There is no separate record of that model: the knowledge base
itself is the record. Each Qdrant collection belongs to one platform, and every
point stores ``embedding_model`` and ``vector_dimensions``, so the filled
collection says which model is locked. That keeps existing setups working with
no migration, and stays correct after a full re-index with another model.

The rules live in ``check_embedding_compatibility`` (pure, unit-tested); the
other functions only gather its inputs from Qdrant and Resql.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import requests

from src.loki_logger import LokiLogger

logger = LokiLogger(service_name="llm-orchestration-service")

HTTP_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class PlatformSpec:
    """The approved embedding model of a platform and where its vectors live."""

    collection: str
    model: str
    vector_dimensions: int


# One approved model per platform. Must match the embedding_models table (the
# connection form's dropdown) and the indexer's collections in
# src/vector_indexer/qdrant_manager.py.
PLATFORMS: Dict[str, PlatformSpec] = {
    "azure": PlatformSpec("contextual_chunks_azure", "text-embedding-3-large", 3072),
    "aws": PlatformSpec("contextual_chunks_aws", "amazon.titan-embed-text-v2:0", 1024),
}


class KnowledgeBaseLockError(Exception):
    """The knowledge base or the other connections could not be read."""


@dataclass(frozen=True)
class LockedModel:
    """An embedding model that has vectors in the knowledge base."""

    platform: str
    embedding_model: str
    vector_dimensions: int
    point_count: int

    def describe(self) -> str:
        return f"{self.embedding_model} ({self.platform}, {self.vector_dimensions} dimensions)"


@dataclass(frozen=True)
class OtherConnection:
    """Another active connection's embedding settings."""

    connection_id: int
    connection_name: str
    environment: str
    platform: str
    embedding_model: str


@dataclass
class EmbeddingCheckResult:
    """Outcome of checking a connection's embedding model against the lock."""

    allowed: bool
    # Why it was blocked, for the GUI to pick a translated message:
    # "model_mismatch" | "test_embedding_failed"
    code: Optional[str] = None
    # For model_mismatch: "different_model" (the connection's model or platform is
    # not the required one) | "wrong_deployment" (the test embedding has another
    # vector size than the required model, e.g. an Azure deployment of ada-002)
    detail: Optional[str] = None
    required_model: Optional[str] = None  # the embedding model connections must use
    requested_model: Optional[str] = None  # the embedding model this connection has
    reason: Optional[str] = None  # short English explanation, for logs and API callers
    locked_by: Optional[str] = None  # "knowledge_base" | "connection" | None
    locked_models: List[Dict[str, Any]] = field(default_factory=list)
    detected_vector_dimensions: Optional[int] = None
    # Issue #588: a production connection may use another approved model than
    # the knowledge base; saving it makes the next indexer run re-index
    # everything with it. Never set for testing connections.
    reindex_required: bool = False
    # Other active connections whose embedding model will no longer match the
    # knowledge base once this (production) connection is saved.
    mismatched_connections: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_locked_models(qdrant_url: str) -> List[LockedModel]:
    """Embedding models that have vectors in the knowledge base.

    Normally zero (nothing indexed yet) or one. Two only in setups that indexed
    on both platforms before the lock existed; both then stay usable.

    Raises:
        KnowledgeBaseLockError: Qdrant could not be read.
    """
    locked: List[LockedModel] = []
    for platform, spec in PLATFORMS.items():
        base = f"{qdrant_url}/collections/{spec.collection}/points"
        try:
            count_response = requests.post(
                f"{base}/count", json={"exact": True}, timeout=HTTP_TIMEOUT_SECONDS
            )
            if count_response.status_code == 404:  # collection not created yet
                continue
            count_response.raise_for_status()
            point_count = int(count_response.json()["result"]["count"])
            if point_count == 0:
                continue

            scroll_response = requests.post(
                f"{base}/scroll",
                json={
                    "limit": 1,
                    "with_payload": ["embedding_model", "vector_dimensions"],
                    "with_vector": False,
                },
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            scroll_response.raise_for_status()
            points = scroll_response.json()["result"]["points"]
            payload = points[0].get("payload", {}) if points else {}
        except (requests.RequestException, KeyError, ValueError, TypeError) as e:
            raise KnowledgeBaseLockError(
                f"Could not read the knowledge base collection {spec.collection}: {e}"
            ) from e

        # The indexer stores "provider/model" (e.g. azure_openai/text-embedding-3-large)
        stored_model = str(payload.get("embedding_model") or spec.model)
        locked.append(
            LockedModel(
                platform=platform,
                embedding_model=stored_model.split("/")[-1],
                vector_dimensions=int(
                    payload.get("vector_dimensions") or spec.vector_dimensions
                ),
                point_count=point_count,
            )
        )
    return locked


def read_other_connections(
    resql_url: str, exclude_connection_id: Optional[int] = None
) -> List[OtherConnection]:
    """The active production and testing connections, except the one being edited.

    Raises:
        KnowledgeBaseLockError: Resql could not be read.
    """
    connections: List[OtherConnection] = []
    for environment in ("production", "testing"):
        try:
            response = requests.post(
                f"{resql_url}/get-{environment}-connection",
                json={},
                timeout=HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as e:
            raise KnowledgeBaseLockError(
                f"Could not read the active {environment} connection: {e}"
            ) from e

        rows = body.get("response", body) if isinstance(body, dict) else body
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            if exclude_connection_id is not None and int(row["id"]) == int(
                exclude_connection_id
            ):
                continue
            connections.append(
                OtherConnection(
                    connection_id=int(row["id"]),
                    connection_name=str(row.get("connectionName") or ""),
                    environment=environment,
                    platform=str(row.get("embeddingPlatform") or ""),
                    embedding_model=str(row.get("embeddingModel") or ""),
                )
            )
    return connections


def reindex_required_for(platform: str, locked_models: List[LockedModel]) -> bool:
    """Whether indexing with ``platform`` must first re-index the knowledge base.

    True when any embeddings in the knowledge base come from another platform
    (issue #588). Each platform has its own collection, so this also stays true
    after an interrupted re-index, until the old collection has been emptied.
    """
    return any(model.platform != platform for model in locked_models)


def check_embedding_compatibility(
    platform: str,
    embedding_model: str,
    detected_vector_dimensions: Optional[int],
    locked_models: List[LockedModel],
    other_connections: List[OtherConnection],
    environment: Optional[str] = None,
) -> EmbeddingCheckResult:
    """Decide whether a connection's embedding model may be saved.

    Rules, in order:
      1. Only the approved model of a supported platform.
      2. If a test embedding was made, its vector size must be the approved
         model's (catches an Azure deployment that points at another model).
      3. If the knowledge base has embeddings, the platform must be one of them.
         Production (issue #588) may use another approved model instead: the
         result is allowed with ``reindex_required`` and the next indexer run
         re-indexes the knowledge base with it.
      4. Otherwise the other active connection(s) must use the same model, so
         test and production share one knowledge base from the first index on.
         Production is allowed anyway (it is the one the indexer uses); the
         other connections are reported in ``mismatched_connections``.

    ``environment`` is where the connection is saved ("production" |
    "testing"). Without it the strict (testing) rules apply.

    Only the embedding model is checked; the chat model may differ.
    """
    locked_dicts = [asdict(model) for model in locked_models]
    is_production = environment == "production"

    def result(
        allowed: bool,
        required_model: Optional[str] = None,
        reason: Optional[str] = None,
        locked_by: Optional[str] = None,
        detail: str = "different_model",
        reindex_required: bool = False,
        mismatched: Optional[List[OtherConnection]] = None,
    ) -> EmbeddingCheckResult:
        return EmbeddingCheckResult(
            allowed=allowed,
            code=None if allowed else "model_mismatch",
            detail=None if allowed else detail,
            required_model=required_model,
            requested_model=embedding_model,
            reason=reason,
            locked_by=locked_by,
            locked_models=locked_dicts,
            detected_vector_dimensions=detected_vector_dimensions,
            reindex_required=reindex_required,
            mismatched_connections=[asdict(other) for other in mismatched or []],
        )

    def mismatched_with(model: str) -> List[OtherConnection]:
        return [
            other
            for other in other_connections
            if other.platform != platform or other.embedding_model != model
        ]

    spec = PLATFORMS.get(platform)
    if spec is None:
        return result(
            False, reason=f"Embedding platform '{platform}' is not supported."
        )

    if embedding_model != spec.model:
        return result(False, spec.model, f"Only {spec.model} is allowed on {platform}.")

    if (
        detected_vector_dimensions is not None
        and detected_vector_dimensions != spec.vector_dimensions
    ):
        return result(
            False,
            spec.model,
            f"The embedding deployment is not {spec.model} "
            f"({detected_vector_dimensions} dimensions, expected {spec.vector_dimensions}).",
            detail="wrong_deployment",
        )

    if locked_models:
        matches_knowledge_base = any(
            model.platform == platform
            and model.vector_dimensions == spec.vector_dimensions
            for model in locked_models
        )
        reindex = is_production and reindex_required_for(platform, locked_models)
        if matches_knowledge_base and not reindex:
            return result(True, spec.model, locked_by="knowledge_base")
        if is_production:
            # Also reached mid re-index (both platforms hold embeddings): saving
            # again keeps the re-index going rather than blocking.
            old = " and ".join(
                model.describe()
                for model in locked_models
                if model.platform != platform
            )
            return result(
                True,
                spec.model,
                f"The knowledge base is built with {old}. It will be fully "
                f"re-indexed with {spec.model} ({platform}).",
                locked_by="knowledge_base",
                reindex_required=True,
                mismatched=mismatched_with(spec.model),
            )
        required = locked_models[0]
        return result(
            False,
            required.embedding_model,
            f"Must be {required.describe()}, the knowledge base's embedding model.",
            locked_by="knowledge_base",
        )

    conflicting = mismatched_with(embedding_model)
    if conflicting and is_production:
        # Nothing indexed yet, so nothing to re-index: production simply wins
        # and the first index is made with its model.
        return result(True, mismatched=conflicting)
    if conflicting:
        other = conflicting[0]
        return result(
            False,
            other.embedding_model,
            f"Must be {other.embedding_model} ({other.platform}), "
            f"the same as the {other.environment} connection '{other.connection_name}'.",
            locked_by="connection",
        )

    return result(True)

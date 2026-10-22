"""Unit tests for the semantic corroboration gate in ContextualRetriever.

Background
----------
BM25 has no relevance floor: it returns top-N by keyword score however weak the
match. Semantic search does have one (``search.score_threshold``). So when
semantic returns nothing while BM25 returns results, those BM25 survivors are
lexical coincidence rather than relevance.

Reproduction this guards against: sending ``"explain it?"`` as the first message
of a fresh chat produced 0 semantic + 40 BM25 results, which rank fusion turned
into 5 chunks. That defeated the zero-chunk out-of-scope gate downstream, and the
generator answered a question the user never asked.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.contextual_retrieval.contextual_retriever import ContextualRetriever


def _semantic_chunk(chunk_id: str, score: float = 0.72) -> Dict[str, Any]:
    """A result shaped like one returned by semantic search."""
    return {
        "chunk_id": chunk_id,
        "id": chunk_id,
        "score": score,
        "content": f"semantic content for {chunk_id}",
        "original_content": f"semantic content for {chunk_id}",
        "metadata": {"source_file": "semantic.md"},
    }


def _bm25_chunk(chunk_id: str, score: float = 47.09) -> Dict[str, Any]:
    """A result shaped like one returned by BM25 search."""
    return {
        "chunk_id": chunk_id,
        "id": chunk_id,
        "bm25_score": score,
        "content": f"bm25 content for {chunk_id}",
        "original_content": f"bm25 content for {chunk_id}",
        "metadata": {"source_file": "bm25.md"},
    }


@pytest.fixture
def retriever() -> ContextualRetriever:
    """A retriever with external dependencies stubbed out.

    Only ``_semantic_search`` and ``_bm25_search`` are replaced per-test; rank
    fusion runs for real so the gate is exercised against genuine RRF output.
    """
    with (
        patch(
            "src.contextual_retrieval.contextual_retriever.DynamicProviderDetection"
        ) as mock_provider,
        patch("src.contextual_retrieval.contextual_retriever.QdrantContextualSearch"),
        patch("src.contextual_retrieval.contextual_retriever.SmartBM25Search"),
    ):
        mock_provider.return_value.detect_optimal_collections = AsyncMock(
            return_value=["contextual_chunks_azure"]
        )
        instance = ContextualRetriever(
            qdrant_url="http://qdrant:6333",
            environment="production",
            connection_id="test-connection",
        )

    instance.initialized = True
    instance._clear_session_cache = MagicMock()
    return instance


async def _retrieve(
    retriever: ContextualRetriever,
    semantic: List[Dict[str, Any]],
    bm25: List[Dict[str, Any]],
    semantic_raises: bool = False,
) -> List[Dict[str, Any]]:
    """Drive retrieve_contextual_chunks with controlled search results."""
    if semantic_raises:
        retriever._semantic_search = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("embedder unavailable")
        )
    else:
        retriever._semantic_search = AsyncMock(return_value=semantic)  # type: ignore[method-assign]
    retriever._bm25_search = AsyncMock(return_value=bm25)  # type: ignore[method-assign]

    return await retriever.retrieve_contextual_chunks(
        original_question="explain it?",
        refined_questions=["Can you provide a detailed explanation?"],
    )


class TestSemanticCorroborationGate:
    """The gate itself: semantic-empty + BM25-populated must yield no chunks."""

    @pytest.mark.asyncio
    async def test_bm25_only_results_are_discarded(
        self, retriever: ContextualRetriever
    ) -> None:
        """The bug reproduction: 0 semantic, many BM25 -> empty result."""
        assert retriever.config.search.require_semantic_corroboration is True

        results = await _retrieve(
            retriever,
            semantic=[],
            bm25=[_bm25_chunk(f"bm25_{i}") for i in range(40)],
        )

        assert results == [], (
            "BM25-only results must be discarded when semantic search found "
            "nothing above threshold - they are lexical noise, not relevance"
        )

    @pytest.mark.asyncio
    async def test_semantic_failure_falls_back_to_bm25(
        self, retriever: ContextualRetriever
    ) -> None:
        """An embedding outage must degrade to BM25, not refuse everything.

        Both 'semantic found nothing' and 'semantic errored' produce an empty
        list. Only the first is a relevance verdict; the second is an
        infrastructure problem and must not suppress the BM25 fallback.
        """
        results = await _retrieve(
            retriever,
            semantic=[],
            bm25=[_bm25_chunk(f"bm25_{i}") for i in range(40)],
            semantic_raises=True,
        )

        assert len(results) > 0, (
            "When semantic search FAILS the gate must not fire - otherwise an "
            "embedding outage silently turns every query out-of-scope"
        )

    @pytest.mark.asyncio
    async def test_semantic_results_present_keeps_bm25_only_chunks(
        self, retriever: ContextualRetriever
    ) -> None:
        """Normal hybrid retrieval is unaffected; BM25 still contributes recall."""
        results = await _retrieve(
            retriever,
            semantic=[_semantic_chunk(f"sem_{i}") for i in range(3)],
            bm25=[_bm25_chunk(f"bm25_{i}") for i in range(10)],
        )

        assert len(results) > 0
        # _format_results_for_compatibility nests the id under "meta"
        returned_ids = {
            str(r.get("meta", {}).get("chunk_id", ""))  # type: ignore[union-attr]
            for r in results
        }
        assert any(cid.startswith("bm25_") for cid in returned_ids), (
            f"BM25-only chunks must still be returned when semantic found "
            f"results; got {sorted(returned_ids)}"
        )
        assert any(cid.startswith("sem_") for cid in returned_ids), (
            f"Semantic chunks must be returned; got {sorted(returned_ids)}"
        )

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(
        self, retriever: ContextualRetriever
    ) -> None:
        """Unchanged behaviour when neither retriever found anything."""
        results = await _retrieve(retriever, semantic=[], bm25=[])
        assert results == []

    @pytest.mark.asyncio
    async def test_gate_can_be_disabled(self, retriever: ContextualRetriever) -> None:
        """Turning the flag off restores the pre-fix behaviour."""
        retriever.config.search.require_semantic_corroboration = False

        results = await _retrieve(
            retriever,
            semantic=[],
            bm25=[_bm25_chunk(f"bm25_{i}") for i in range(40)],
        )

        assert len(results) > 0, (
            "With require_semantic_corroboration disabled, BM25-only results "
            "should pass through as they did before the fix"
        )


class TestConfiguration:
    """The flag is wired through constants, model default and YAML loader."""

    def test_default_is_enabled(self) -> None:
        from src.contextual_retrieval.config import SearchConfig

        assert SearchConfig().require_semantic_corroboration is True

    def test_constant_exists(self) -> None:
        from src.contextual_retrieval.constants import SearchConstants

        assert SearchConstants.DEFAULT_REQUIRE_SEMANTIC_CORROBORATION is True

    def test_yaml_override_is_respected(self, tmp_path: Any) -> None:
        """A config file can turn the gate off without a code change."""
        from src.contextual_retrieval.config import ConfigLoader

        config_file = tmp_path / "retrieval.yaml"
        config_file.write_text(
            "contextual_retrieval:\n"
            "  search:\n"
            "    require_semantic_corroboration: false\n",
            encoding="utf-8",
        )

        config = ConfigLoader.load_config(str(config_file))
        assert config.search.require_semantic_corroboration is False

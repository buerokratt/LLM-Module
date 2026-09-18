"""API Tool Semantic Searcher — hybrid search against api_tool_collection."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, cast

import dspy
import httpx
from src.utils.observation_utils import safe_observation_context
from src.loki_logger import LokiLogger

from tool_classifier.constants import (
    API_TOOL_COLLECTION,
    API_TOOL_HIGH_CONFIDENCE_THRESHOLD,
    API_TOOL_MIN_THRESHOLD,
    API_TOOL_SCORE_GAP_THRESHOLD,
    API_TOOL_SEARCH_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
)
from tool_classifier.sparse_encoder import compute_sparse_vector
from tool_classifier.sparse_encoder import SparseVector
from src.utils.cost_utils import get_lm_usage_since


def _get_current_model_name() -> str:
    """Best-effort model name lookup from current DSPy LM."""
    try:
        lm = dspy.settings.lm
        if lm and hasattr(lm, "model"):
            model_name = lm.model
            if isinstance(model_name, str) and model_name:
                return model_name
    except Exception:
        pass
    return "unknown"


logger = LokiLogger(service_name="api-tool-calling")


class EmbeddingServiceProtocol(Protocol):
    """Protocol for any service that can generate text embeddings."""

    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[str, Any]: ...


class APIToolSearchResult:
    """Result from API Tool semantic search."""

    def __init__(
        self,
        endpoint_id: str,
        name: str,
        description: str,
        method: str,
        url: str,
        params: List[Dict],
        cosine_score: float,
        rrf_score: float,
        confidence: str,
        llm_validated: bool = False,
        multi_intent_hint: bool = False,
    ) -> None:
        self.endpoint_id = endpoint_id
        self.name = name
        self.description = description
        self.method = method
        self.url = url
        self.params = params
        self.cosine_score = (
            cosine_score  # Real dense cosine similarity (used for thresholds)
        )
        self.rrf_score = rrf_score  # Hybrid RRF fusion score (used for ranking)
        self.confidence = confidence  # "high", "medium", "none"
        self.llm_validated = (
            llm_validated  # True when disambiguator confirmed this match
        )
        self.multi_intent_hint = (
            multi_intent_hint  # True when disambiguator rejected all multi-candidates;
            # this result is only valid when MULTI_INTENT_ENABLED=True
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "url": self.url,
            "params": self.params,
            "cosine_score": round(self.cosine_score, 4),
            "rrf_score": round(self.rrf_score, 6),
            "confidence": self.confidence,
            "llm_validated": self.llm_validated,
        }


class EndpointDisambiguationSignature(dspy.Signature):
    """Determine which API endpoint best matches a user query, or none.

    Rules:
    - Analyze the user query against the candidate endpoints carefully
    - Return the endpoint_id of the best match if one clearly addresses the query
    - Return exactly "none" if no endpoint clearly fits — do not guess
    - Be conservative — only match when confident
    - Understand Estonian, Russian, and English queries
    """

    user_query: str = dspy.InputField(
        desc="User's question or request in Estonian, Russian, or English"
    )
    candidates: str = dspy.InputField(
        desc="JSON list of candidate endpoints: [{endpoint_id, name, description, cosine_score}]"
    )

    best_endpoint_id: str = dspy.OutputField(
        desc='The endpoint_id of the best match, or exactly "none" if no endpoint clearly fits'
    )


@dataclass
class DisambiguationResult:
    """Wrapper for disambiguation result to satisfy DSPy's Langfuse callback."""

    winner_id: Optional[str]

    def set_lm_usage(self, *args: object, **kwargs: object) -> None:
        """No-op stub for DSPy's internal Langfuse callback compatibility."""


class EndpointDisambiguatorModule(dspy.Module):
    """DSPy Module for resolving ambiguous API endpoint candidates via LLM.

    Called when multiple endpoints score in the medium-confidence range and
    a clear winner cannot be determined from cosine scores alone.
    """

    def __init__(self) -> None:
        """Initialize with a direct DSPy predictor."""
        super().__init__()
        self.predictor = dspy.Predict(EndpointDisambiguationSignature)

    def forward(
        self,
        user_query: str,
        candidates: List[Dict[str, Any]],
    ) -> DisambiguationResult:
        """Pick the best matching endpoint_id from candidates, or return None.

        Args:
            user_query: The user's natural language query.
            candidates: List of candidate dicts with endpoint_id, name,
                description, and cosine_score.

        Returns:
            DisambiguationResult with winner_id (endpoint_id string or None).
        """
        candidates_payload = [
            {
                "endpoint_id": c["endpoint_id"],
                "name": c["name"],
                "description": c["description"],
                "cosine_score": round(c["cosine_score"], 4),
            }
            for c in candidates
        ]
        candidates_json = json.dumps(candidates_payload, ensure_ascii=False, indent=2)

        try:
            result = self.predictor(
                user_query=user_query,
                candidates=candidates_json,
            )
            winner = result.best_endpoint_id.strip()
            if winner.lower() == "none":
                return DisambiguationResult(winner_id=None)
            return DisambiguationResult(winner_id=winner)
        except Exception as e:
            logger.error(
                f"EndpointDisambiguatorModule: Disambiguation failed: {e}",
                exc_info=True,
            )
            return DisambiguationResult(winner_id=None)


class APISemanticSearcher:
    """Semantic searcher for API Tool endpoints stored in api_tool_collection.

    Usage:
        searcher = APISemanticSearcher(
            qdrant_client=shared_httpx_client,
            embedding_service=orchestration_service,
        )
        results = await searcher.search("What are national holidays in Estonia?")
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        qdrant_client: Optional[httpx.AsyncClient] = None,
        disambiguator: Optional[EndpointDisambiguatorModule] = None,
    ) -> None:
        """Initialize the API semantic searcher.

        Args:
            embedding_service: Service that generates dense embeddings.
            qdrant_client: Optional shared httpx client. If None, creates its own.
            disambiguator: Optional DSPy disambiguation module. If None, a default
                instance is created. Inject a custom instance for testing.
        """
        self.embedding_service = embedding_service
        self._disambiguator = (
            disambiguator
            if disambiguator is not None
            else EndpointDisambiguatorModule()
        )
        self._owns_client = qdrant_client is None

        if qdrant_client is not None:
            self._qdrant_client = qdrant_client
        else:
            self._qdrant_client = httpx.AsyncClient(
                base_url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                timeout=QDRANT_TIMEOUT,
                limits=httpx.Limits(
                    max_connections=10,
                    max_keepalive_connections=5,
                ),
            )

    async def aclose(self) -> None:
        """Close the httpx client if we own it."""
        if self._owns_client:
            await self._qdrant_client.aclose()

    async def search(
        self,
        query: str,
        environment: str = "production",
        connection_id: Optional[str] = None,
        top_k: int = API_TOOL_SEARCH_TOP_K,
        precomputed_embedding: Optional[List[float]] = None,
        min_cosine_override: Optional[float] = None,
    ) -> List[APIToolSearchResult]:
        """Search api_tool_collection for the best matching API endpoints.

        Uses a two-step approach:
        1. Dense search → get real cosine similarity scores
        2. Hybrid search (dense + sparse + RRF) → get best-ranked matches

        The effective minimum cosine threshold is determined by ``min_cosine_override``
        when provided, otherwise ``API_TOOL_MIN_THRESHOLD`` is used. Confidence levels
        are always evaluated against the same effective minimum:

        - "high":   cosine >= API_TOOL_HIGH_CONFIDENCE_THRESHOLD AND score gap is large
        - "medium": cosine >= effective minimum threshold but ambiguous
        - "none":   cosine < effective minimum threshold (no match)

        Args:
            query: Natural language user query.
            environment: Environment for embedding model resolution.
            connection_id: Optional connection ID for embedding service.
            top_k: Maximum number of results to return.
            precomputed_embedding: Dense vector already computed upstream (e.g. by
                the service classifier). When provided the embedding step is skipped
                entirely, saving one embedding API call per request.
            min_cosine_override: Optional cosine similarity threshold that replaces
                ``API_TOOL_MIN_THRESHOLD`` for this call. Pass a lower value to
                broaden matching (e.g. during multi-intent re-classification) or a
                higher value to tighten it. When None, ``API_TOOL_MIN_THRESHOLD``
                applies unchanged.

        Returns:
            List containing exactly one APIToolSearchResult (the resolved best match),
            or an empty list if no suitable API tool endpoint was found.
            Never returns more than one result — ambiguous medium-confidence candidates
            are resolved via LLM disambiguation before returning.
        """
        # Step 1: Reuse caller's embedding if provided, otherwise generate a new one
        if precomputed_embedding is not None:
            logger.debug(
                "APISemanticSearcher: reusing precomputed query embedding (no extra API call)"
            )
            query_embedding = precomputed_embedding
        else:
            # loop is not blocked — this allows parallel sub-query searches
            query_embedding = await asyncio.to_thread(
                self._get_query_embedding, query, environment, connection_id
            )
        if query_embedding is None:
            logger.error("APISemanticSearcher: Failed to generate query embedding")
            return []

        # Step 2: Dense search → real cosine scores for relevance check
        dense_results = await self._dense_search(query_embedding, top_k=top_k)
        if not dense_results:
            logger.info("APISemanticSearcher: No results from dense search")
            return []

        top_cosine = dense_results[0]["cosine_score"]
        second_cosine = (
            dense_results[1]["cosine_score"] if len(dense_results) > 1 else 0.0
        )
        cosine_gap = top_cosine - second_cosine

        effective_min = (
            min_cosine_override
            if min_cosine_override is not None
            else API_TOOL_MIN_THRESHOLD
        )

        logger.info(f"APISemanticSearcher: query={query!r}")
        logger.info(
            f"APISemanticSearcher: dense top={dense_results[0]['name']} "
            f"(cosine={top_cosine:.4f}), gap={cosine_gap:.4f}"
        )

        # Below minimum threshold → no match
        if top_cosine < effective_min:
            logger.info(
                f"APISemanticSearcher: cosine {top_cosine:.4f} < "
                f"threshold {effective_min} — no API tool match"
            )
            return []

        # Step 3: Hybrid search → best-ranked results using dense + sparse + RRF
        query_sparse = compute_sparse_vector(query)
        hybrid_results = await self._hybrid_search(
            query_embedding, query_sparse, top_k=top_k
        )

        # Fall back to dense results if hybrid returns nothing
        if not hybrid_results:
            hybrid_results = dense_results

        # Build a lookup from endpoint_id → real cosine score from dense results
        dense_cosine_map = {r["endpoint_id"]: r["cosine_score"] for r in dense_results}

        # Step 4: Annotate each result with confidence level
        results: List[APIToolSearchResult] = []
        for i, point in enumerate(hybrid_results):
            endpoint_id = point.get("endpoint_id", "")

            # Prefer cosine from dense search; if this hybrid result was not in the
            # dense top-N set, fall back to the cosine carried on the hybrid result
            # itself. Skip entirely if no actual cosine score is available.
            point_cosine = dense_cosine_map.get(endpoint_id)
            if point_cosine is None:
                point_cosine = point.get("cosine_score")
            if point_cosine is None:
                continue  # Skip results that do not have an actual cosine score
            point_rrf = point.get("rrf_score", 0.0)

            # Compute gap relative to this candidate: its cosine vs the best other
            # dense cosine. This is correct even when hybrid re-ranks the top result.
            next_best_cosine = next(
                (
                    r["cosine_score"]
                    for r in dense_results
                    if r["endpoint_id"] != endpoint_id
                ),
                0.0,
            )
            effective_gap = point_cosine - next_best_cosine

            if (
                point_cosine >= API_TOOL_HIGH_CONFIDENCE_THRESHOLD
                and point_cosine >= effective_min
                and effective_gap >= API_TOOL_SCORE_GAP_THRESHOLD
                and i == 0
            ):
                confidence = "high"
            elif point_cosine >= effective_min:
                confidence = "medium"
            else:
                continue  # Skip results below threshold

            results.append(
                APIToolSearchResult(
                    endpoint_id=endpoint_id,
                    name=point.get("name", ""),
                    description=point.get("description", ""),
                    method=point.get("method", "GET"),
                    url=point.get("url", ""),
                    params=point.get("params", []),
                    cosine_score=point_cosine,
                    rrf_score=point_rrf,
                    confidence=confidence,
                )
            )

        # Step 5: Resolve to exactly one result
        high_results = [r for r in results if r.confidence == "high"]
        if high_results:
            logger.info(
                f"APISemanticSearcher: high-confidence match → {high_results[0].name!r} "
                f"(cosine={high_results[0].cosine_score:.4f})"
            )
            return [high_results[0]]

        medium_results = [r for r in results if r.confidence == "medium"]
        if not medium_results:
            logger.info(
                "APISemanticSearcher: no results above threshold — no API tool match"
            )
            return []

        # Single medium result — only return directly if the gap is large enough
        # (gap < SCORE_GAP_THRESHOLD means runner-up was close, LLM should validate)
        if len(medium_results) == 1 and cosine_gap >= API_TOOL_SCORE_GAP_THRESHOLD:
            logger.info(
                f"APISemanticSearcher: single medium-confidence match (gap={cosine_gap:.4f}) → "
                f"{medium_results[0].name!r} (cosine={medium_results[0].cosine_score:.4f})"
            )
            return [medium_results[0]]

        # Multiple ambiguous candidates, OR single candidate with small gap — LLM validates
        if len(medium_results) == 1:
            logger.info(
                f"APISemanticSearcher: single medium result but gap={cosine_gap:.4f} < "
                f"{API_TOOL_SCORE_GAP_THRESHOLD} — sending to LLM for validation"
            )
        winner_id = await self._disambiguate(query, medium_results)
        if winner_id is None:
            if len(medium_results) > 1:
                # Disambiguator rejected all candidates on a multi-candidate query.
                # This happens when the query spans multiple independent intents and
                # no single endpoint fully answers it. Return the top cosine-scored
                # candidate WITHOUT llm_validated so the classifier's IntentDecomposer
                # gate can run and detect the parallel path.
                top = max(medium_results, key=lambda r: r.cosine_score)
                logger.info(
                    f"APISemanticSearcher: disambiguator rejected all "
                    f"{len(medium_results)} candidates (possible multi-intent) — "
                    f"returning top candidate {top.name!r} "
                    f"(cosine={top.cosine_score:.4f}) for IntentDecomposer"
                )
                top.multi_intent_hint = True
                return [top]
            logger.info(
                "APISemanticSearcher: disambiguator rejected sole candidate — "
                "no API tool match"
            )
            return []

        winner = next((r for r in medium_results if r.endpoint_id == winner_id), None)
        if winner is None:
            logger.warning(
                f"APISemanticSearcher: disambiguator returned unknown "
                f"endpoint_id={winner_id!r} — no API tool match"
            )
            return []

        winner.llm_validated = True
        logger.info(
            f"APISemanticSearcher: disambiguated winner → {winner.name!r} "
            f"(cosine={winner.cosine_score:.4f}, llm_validated=True)"
        )
        return [winner]

    async def _disambiguate(
        self,
        query: str,
        candidates: List[APIToolSearchResult],
    ) -> Optional[str]:
        """Invoke LLM disambiguation on ambiguous medium-confidence candidates.

        Args:
            query: The original user query.
            candidates: Medium-confidence APIToolSearchResult items to choose between.

        Returns:
            The endpoint_id of the winner, or None if the LLM rejects all candidates.
        """
        candidate_dicts = [
            {
                "endpoint_id": r.endpoint_id,
                "name": r.name,
                "description": r.description,
                "cosine_score": r.cosine_score,
            }
            for r in candidates
        ]
        logger.info(
            f"APISemanticSearcher: disambiguating {len(candidates)} candidates "
            f"for query: {query!r}"
        )

        history_length_before = 0
        try:
            lm = dspy.settings.lm
            if lm and hasattr(lm, "history"):
                history_length_before = len(lm.history)
        except Exception as e:
            logger.warning(f"Failed to get LM history length for disambiguation: {e}")

        with safe_observation_context(
            name="api_endpoint_disambiguation_llm",
            as_type="generation",
            input={"user_query": query, "candidates_count": len(candidates)},
        ) as generation:
            # Run the synchronous DSPy LLM call in a thread pool so it does not
            # block the asyncio event loop while waiting for the LLM response.
            disambiguation_result = cast(
                DisambiguationResult,
                await asyncio.to_thread(
                    self._disambiguator,
                    user_query=query,
                    candidates=candidate_dicts,
                ),
            )
            winner_id = disambiguation_result.winner_id

            # Update Langfuse observation with output and usage
            try:
                if generation is not None:
                    usage = get_lm_usage_since(history_length_before)
                    generation.update(
                        model=_get_current_model_name(),
                        output={"winner": winner_id},
                        usage_details={
                            "input": usage.get("total_prompt_tokens", 0),
                            "output": usage.get("total_completion_tokens", 0),
                            "total": usage.get("total_tokens", 0),
                        },
                        cost_details={"total": usage.get("total_cost", 0.0)},
                    )
            except Exception as e:
                logger.debug(f"Langfuse generation update skipped: {e}")

        if winner_id:
            logger.info(
                f"APISemanticSearcher: disambiguator picked endpoint_id={winner_id!r}"
            )
        else:
            logger.info("APISemanticSearcher: disambiguator rejected all candidates")
        return winner_id

    def _get_query_embedding(
        self,
        query: str,
        environment: str,
        connection_id: Optional[str],
    ) -> Optional[List[float]]:
        """Generate dense embedding for the query."""
        try:
            result = self.embedding_service.create_embeddings_for_indexer(
                texts=[query],
                environment=environment,
                connection_id=connection_id,
                batch_size=1,
            )
            embeddings = result.get("embeddings", [])
            if embeddings:
                return embeddings[0]
            logger.error("APISemanticSearcher: No embedding returned")
            return None
        except Exception as e:
            logger.error(f"APISemanticSearcher: Embedding generation failed: {e}")
            return None

    async def _dense_search(
        self,
        dense_vector: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Dense-only search on api_tool_collection for real cosine scores.

        Returns deduplicated results by endpoint_id, sorted by cosine score.
        """
        try:
            search_payload = {
                "query": dense_vector,
                "using": "dense",
                "limit": top_k * 2,
                "with_payload": True,
            }

            response = await self._qdrant_client.post(
                f"/collections/{API_TOOL_COLLECTION}/points/query",
                json=search_payload,
            )

            if response.status_code != 200:
                logger.error(
                    f"APISemanticSearcher: Dense search failed "
                    f"HTTP {response.status_code} — {response.text}"
                )
                return []

            points = response.json().get("result", {}).get("points", [])
            if not points:
                return []

            # Deduplicate by endpoint_id, keep best cosine score
            endpoint_results: Dict[str, Dict[str, Any]] = {}
            for point in points:
                payload = point.get("payload", {})
                score = float(point.get("score", 0))
                endpoint_id = payload.get("endpoint_id", "unknown")

                if endpoint_id not in endpoint_results or score > endpoint_results[
                    endpoint_id
                ].get("cosine_score", 0):
                    endpoint_results[endpoint_id] = {
                        "endpoint_id": endpoint_id,
                        "name": payload.get("name", ""),
                        "description": payload.get("description", ""),
                        "method": payload.get("method", "GET"),
                        "url": payload.get("url", ""),
                        "params": payload.get("params", []),
                        "cosine_score": score,
                    }

            return sorted(
                endpoint_results.values(),
                key=lambda x: x["cosine_score"],
                reverse=True,
            )

        except httpx.TimeoutException:
            logger.error(
                f"APISemanticSearcher: Dense search timeout after {QDRANT_TIMEOUT}s"
            )
            return []
        except Exception as e:
            logger.error(
                f"APISemanticSearcher: Dense search failed: {e}", exc_info=True
            )
            return []

    async def _hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: SparseVector,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Hybrid search using dense + sparse + RRF fusion.

        Sends both vectors in a single Qdrant prefetch query.
        Returns deduplicated results by endpoint_id, sorted by RRF score.
        """
        try:
            # Verify collection is non-empty before searching.
            # This check is only an optimization: if it fails, continue with the
            # actual search request instead of failing closed.
            try:
                collection_info = await self._qdrant_client.get(
                    f"/collections/{API_TOOL_COLLECTION}"
                )
                if collection_info.status_code == 200:
                    points_count = (
                        collection_info.json().get("result", {}).get("points_count", 0)
                    )
                    if points_count == 0:
                        logger.info("APISemanticSearcher: api_tool_collection is empty")
                        return []
                else:
                    logger.warning(
                        f"APISemanticSearcher: Could not verify collection: "
                        f"HTTP {collection_info.status_code}; continuing with search"
                    )
            except Exception as e:
                logger.warning(
                    f"APISemanticSearcher: Collection verification failed: {e}; "
                    f"continuing with search"
                )

            # Build prefetch + RRF payload
            search_payload: Dict[str, Any] = {
                "prefetch": [
                    {
                        "query": dense_vector,
                        "using": "dense",
                        "limit": top_k * 2,
                    },
                ],
                "query": {"fusion": "rrf"},
                "limit": top_k,
                "with_payload": True,
            }

            # Add sparse prefetch only if non-empty
            if not sparse_vector.is_empty():
                search_payload["prefetch"].append(
                    {
                        "query": sparse_vector.to_dict(),
                        "using": "sparse",
                        "limit": top_k * 2,
                    }
                )

            response = await self._qdrant_client.post(
                f"/collections/{API_TOOL_COLLECTION}/points/query",
                json=search_payload,
            )

            if response.status_code != 200:
                logger.error(
                    f"APISemanticSearcher: Hybrid search failed "
                    f"HTTP {response.status_code} — {response.text}"
                )
                return []

            points = response.json().get("result", {}).get("points", [])
            if not points:
                return []

            # Deduplicate by endpoint_id, keep best RRF score
            endpoint_results: Dict[str, Dict[str, Any]] = {}
            for point in points:
                payload = point.get("payload", {})
                score = float(point.get("score", 0))
                endpoint_id = payload.get("endpoint_id", "unknown")

                if endpoint_id not in endpoint_results or score > endpoint_results[
                    endpoint_id
                ].get("rrf_score", 0):
                    endpoint_results[endpoint_id] = {
                        "endpoint_id": endpoint_id,
                        "name": payload.get("name", ""),
                        "description": payload.get("description", ""),
                        "method": payload.get("method", "GET"),
                        "url": payload.get("url", ""),
                        "params": payload.get("params", []),
                        "rrf_score": score,
                        # cosine_score patched in search() from dense results
                    }

            sorted_results = sorted(
                endpoint_results.values(),
                key=lambda x: x["rrf_score"],
                reverse=True,
            )

            logger.info(
                f"APISemanticSearcher: hybrid returned {len(sorted_results)} unique endpoints"
            )
            for i, r in enumerate(sorted_results[:3]):
                logger.debug(
                    f"  Rank {i + 1}: {r['name']} "
                    f"(endpoint_id={r['endpoint_id']}, rrf={r['rrf_score']:.6f})"
                )

            return sorted_results

        except httpx.TimeoutException:
            logger.error(
                f"APISemanticSearcher: Hybrid search timeout after {QDRANT_TIMEOUT}s"
            )
            return []
        except Exception as e:
            logger.error(
                f"APISemanticSearcher: Hybrid search failed: {e}", exc_info=True
            )
            return []

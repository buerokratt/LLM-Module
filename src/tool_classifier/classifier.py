"""Main tool classifier for workflow routing with hybrid search classification."""

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    overload,
    TYPE_CHECKING,
)
import httpx

import asyncio

from src.loki_logger import LokiLogger
from llm_orchestrator_config.llm_manager import LLMManager
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationResponse,
)
from tool_classifier.base_workflow import BaseWorkflow
from tool_classifier.enums import (
    WorkflowType,
    WORKFLOW_DISPLAY_NAMES,
    WORKFLOW_LAYER_ORDER,
    ExecutionMode,
)
from tool_classifier.models import ClassificationResult
from tool_classifier.constants import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    QDRANT_TIMEOUT,
    HYBRID_SEARCH_TOP_K,
    DENSE_SEARCH_TOP_K,
    DENSE_MIN_THRESHOLD,
    DENSE_HIGH_CONFIDENCE_THRESHOLD,
    DENSE_SCORE_GAP_THRESHOLD,
    API_TOOL_MIN_THRESHOLD,
    API_TOOL_HIGH_CONFIDENCE_THRESHOLD,
    API_TOOL_INTENT_SWITCH_THRESHOLD,
)

from tool_classifier.sparse_encoder import SparseVector, compute_sparse_vector
from tool_classifier.api_semantic_searcher import (
    APISemanticSearcher,
    APIToolSearchResult,
)
from tool_classifier.intent_decomposer import IntentDecomposerModule

from tool_classifier.workflows import (
    APIToolWorkflowExecutor,
    ServiceWorkflowExecutor,
    ContextWorkflowExecutor,
    RAGWorkflowExecutor,
    OODWorkflowExecutor,
)
from llm_orchestrator_config.feature_flags import FeatureFlags
from utils.atc_cache_store import ATCCacheStore

# Initialize Loki logger
logger = LokiLogger(service_name="tool-classifier")

if TYPE_CHECKING:
    from llm_orchestration_service import LLMOrchestrationService


class ToolClassifier:
    """
    Main classifier that determines which workflow should handle user queries.

    Uses a two-step search approach for classification:
    1. Dense-only search → real cosine similarity scores for relevance check
    2. Hybrid search (dense + sparse + RRF) → best service identification

    Routing decisions:
    - High-confidence service match → SERVICE workflow (skip discovery + intent detection)
    - Ambiguous match → SERVICE workflow with LLM confirmation
    - No match → CONTEXT/RAG workflow (skip SERVICE entirely)

    Implements a layer-wise filtering approach:
    Layer 1: Service Workflow → External API calls
    Layer 2: Context Workflow → Conversation history/greetings
    Layer 3: RAG Workflow → Knowledge base retrieval
    Layer 4: OOD Workflow → Out-of-domain fallback
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        orchestration_service: "LLMOrchestrationService",
    ) -> None:
        """
        Initialize tool classifier with required dependencies.

        Args:
            llm_manager: LLM manager for making LLM calls (intent detection, context check)
            orchestration_service: Reference to main orchestration service (for RAG workflow)
        """
        self.llm_manager = llm_manager
        self.orchestration_service = orchestration_service

        # Shared httpx client for Qdrant queries (connection pooling)
        self._qdrant_base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
        self._qdrant_client = httpx.AsyncClient(
            base_url=self._qdrant_base_url,
            timeout=QDRANT_TIMEOUT,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )

        # Initialize workflow executors
        self.api_tool_workflow = APIToolWorkflowExecutor(
            orchestration_service=orchestration_service,
        )
        self.service_workflow = ServiceWorkflowExecutor(
            llm_manager=llm_manager,
            orchestration_service=orchestration_service,
        )
        self.context_workflow = ContextWorkflowExecutor(
            llm_manager=llm_manager,
            orchestration_service=orchestration_service,
            conversation_history_store=getattr(
                orchestration_service, "conversation_history_store", None
            ),
        )
        self.rag_workflow = RAGWorkflowExecutor(
            orchestration_service=orchestration_service,
        )
        self.ood_workflow = OODWorkflowExecutor()

        # API tool semantic searcher - reuses the shared Qdrant client
        self.api_tool_searcher = APISemanticSearcher(
            embedding_service=orchestration_service,
            qdrant_client=self._qdrant_client,
        )

        # Intent decomposer
        self.intent_decomposer = IntentDecomposerModule()

        logger.info(
            "Tool classifier initialized with hybrid search classification "
            f"(Qdrant: {self._qdrant_base_url})"
        )

    async def aclose(self) -> None:
        """Close the shared httpx client and release connection pool resources.

        Must be awaited during application shutdown to avoid connection leaks.
        """
        await self._qdrant_client.aclose()
        logger.debug("ToolClassifier Qdrant httpx client closed")

    async def classify(
        self,
        query: str,
        language: str,
        request: Optional[OrchestrationRequest] = None,
    ) -> ClassificationResult:
        """
        Classify a user query using a two-step search approach.

        Step 1: Dense-only search → cosine similarity for relevance check
        Step 2: Hybrid search (dense + sparse + RRF) → service identification

        Routing:
        - cosine < DENSE_MIN_THRESHOLD AND no ATC match → CONTEXT/RAG
        - cosine ≥ HIGH_CONFIDENCE + large gap → SERVICE (no LLM needed)
        - ATC match found (when SERVICE misses) → API_TOOL_CALLING
        - else → SERVICE with LLM confirmation

        Args:
            query: User's query string
            language: Detected language code (e.g., 'en', 'et')
            request: Original orchestration request (needed for ATC search
                which requires environment and connection_id for embedding).

        Returns:
            ClassificationResult indicating which workflow to use
        """
        logger.info(f"Classifying query: {query[:100]}...")

        try:
            # Pre-classification: if an API tool session already exists for this
            # chat_id, the user is responding to a param-collection question.
            # Short-circuit directly to API_TOOL_CALLING — no need to re-classify.
            if FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED and request is not None:
                session_store = getattr(
                    self.orchestration_service, "session_store", None
                )
                if session_store is not None:
                    existing_session = await session_store.get(request.chatId)
                    if existing_session is not None:
                        endpoint_name = (
                            existing_session.selected_endpoint.get("name")
                            if existing_session.selected_endpoint
                            else "unknown"
                        )

                        # Before resuming, check if the user's new message is a
                        # strong match for a DIFFERENT endpoint (intent switch).
                        # If so, abandon the old session and start fresh rather
                        # than treating the new query as a param-collection reply.
                        new_api_match = await self._try_api_tool_classification(
                            query,
                            request,
                            min_cosine_override=API_TOOL_INTENT_SWITCH_THRESHOLD,
                        )
                        if (
                            new_api_match is not None
                            and new_api_match.metadata.get("matched_endpoint", {}).get(
                                "name"
                            )
                            != endpoint_name
                        ):
                            logger.info(
                                f"[{request.chatId}] Intent switch detected: "
                                f"active session={endpoint_name!r}, "
                                f"new match={new_api_match.metadata.get('matched_endpoint', {}).get('name')!r} "
                                f"— abandoning old session"
                            )
                            await session_store.delete(request.chatId)
                            if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED:
                                await ATCCacheStore().invalidate_l2(request.chatId)
                                logger.info(
                                    f"[{request.chatId}] ATC cache: L2 invalidated on intent switch"
                                )
                            return new_api_match

                        logger.info(
                            f"[{request.chatId}] Active API tool session found "
                            f"(endpoint={endpoint_name!r}) "
                            f"— short-circuiting to API_TOOL_CALLING"
                        )
                        return ClassificationResult(
                            workflow=WorkflowType.API_TOOL_CALLING,
                            confidence=1.0,
                            metadata={
                                "reason": "active_session_resume",
                                "matched_endpoint": existing_session.selected_endpoint,
                            },
                            reasoning="Resuming active API tool parameter-collection session",
                        )

            if not FeatureFlags.SERVICE_WORKFLOW_ENABLED:
                logger.info(
                    "SERVICE_WORKFLOW_ENABLED=false - skipping standard service search"
                )
                api_tool_result = await self._try_api_tool_classification(
                    query, request
                )
                if api_tool_result:
                    return api_tool_result
                logger.info("No API tool match either — routing to CONTEXT/RAG")
                return ClassificationResult(
                    workflow=WorkflowType.CONTEXT,
                    confidence=1.0,
                    metadata={"reason": "service_workflow_disabled"},
                    reasoning="Service workflow disabled, no ATC match - fallback to Context/RAG",
                )

            # Step 1: Generate dense embedding for query
            query_embedding = self._get_query_embedding(query)
            if query_embedding is None:
                logger.warning(
                    "Failed to generate query embedding, falling back to CONTEXT/RAG"
                )
                return ClassificationResult(
                    workflow=WorkflowType.CONTEXT,
                    confidence=1.0,
                    metadata={"reason": "embedding_generation_failed"},
                    reasoning="Could not generate embedding - skip to Context/RAG",
                )

            # Step 2: Dense-only search → get actual cosine similarity scores
            dense_results = await self._dense_search(
                dense_vector=query_embedding,
                top_k=DENSE_SEARCH_TOP_K,
            )

            if not dense_results:
                logger.info(
                    "No dense search results from intent_collections - trying API tools"
                )
                api_tool_result = await self._try_api_tool_classification(
                    query, request, precomputed_embedding=query_embedding
                )
                if api_tool_result:
                    return api_tool_result
                logger.info("No API tool match either — routing to CONTEXT/RAG")
                return ClassificationResult(
                    workflow=WorkflowType.CONTEXT,
                    confidence=1.0,
                    metadata={"reason": "no_service_match"},
                    reasoning="No services matched the query (dense search empty)",
                )

            top_cosine = dense_results[0].get("cosine_score", 0.0)
            top_service_name = dense_results[0].get("name", "unknown")
            second_cosine = (
                dense_results[1].get("cosine_score", 0.0)
                if len(dense_results) > 1
                else 0.0
            )
            cosine_gap = top_cosine - second_cosine

            logger.info(
                f"Dense search: top={top_service_name} "
                f"(cosine={top_cosine:.4f}), "
                f"second={dense_results[1].get('name', 'none') if len(dense_results) > 1 else 'none'} "
                f"(cosine={second_cosine:.4f}), "
                f"gap={cosine_gap:.4f}"
            )

            # Decision: Is this a service query at all?
            if top_cosine < DENSE_MIN_THRESHOLD:
                logger.info(
                    f"Low service relevance (cosine={top_cosine:.4f} < {DENSE_MIN_THRESHOLD}) "
                    f"— trying API tools before falling to CONTEXT/RAG"
                )
                api_tool_result = await self._try_api_tool_classification(
                    query, request, precomputed_embedding=query_embedding
                )
                if api_tool_result:
                    return api_tool_result
                logger.info("No API tool match — routing to CONTEXT/RAG")
                return ClassificationResult(
                    workflow=WorkflowType.CONTEXT,
                    confidence=1.0,
                    metadata={
                        "reason": "below_dense_threshold",
                        "top_cosine": top_cosine,
                        "top_service": top_service_name,
                    },
                    reasoning=(
                        f"Dense cosine {top_cosine:.4f} below threshold "
                        f"{DENSE_MIN_THRESHOLD} - skip to Context/RAG"
                    ),
                )

            # Step 3: Hybrid search → identify best service using RRF
            query_sparse = compute_sparse_vector(query)
            hybrid_results = await self._hybrid_search(
                dense_vector=query_embedding,
                sparse_vector=query_sparse,
                top_k=HYBRID_SEARCH_TOP_K,
            )

            # Use hybrid results for service identification, dense scores for confidence
            if not hybrid_results:
                # Dense matched but hybrid didn't — use dense results
                hybrid_results = dense_results

            top_result = hybrid_results[0]
            top_service_id = top_result.get("service_id", "unknown")
            top_service_name_hybrid = top_result.get("name", "unknown")

            logger.info(
                f"Hybrid search: best service={top_service_name_hybrid} "
                f"(service_id={top_service_id})"
            )

            # High confidence: cosine is high AND clear gap to second result
            if (
                top_cosine >= DENSE_HIGH_CONFIDENCE_THRESHOLD
                and cosine_gap >= DENSE_SCORE_GAP_THRESHOLD
            ):
                logger.info(
                    f"HIGH-CONFIDENCE match: {top_service_name_hybrid} "
                    f"(cosine={top_cosine:.4f}, gap={cosine_gap:.4f})"
                )
                return ClassificationResult(
                    workflow=WorkflowType.SERVICE,
                    confidence=min(top_cosine, 1.0),
                    metadata={
                        "matched_service_id": top_service_id,
                        "matched_service_name": top_service_name_hybrid,
                        "cosine_score": top_cosine,
                        "cosine_gap": cosine_gap,
                        "needs_llm_confirmation": False,
                        "top_results": hybrid_results[:3],
                    },
                    reasoning=(
                        f"High-confidence match: {top_service_name_hybrid} "
                        f"(cosine={top_cosine:.4f}, gap={cosine_gap:.4f})"
                    ),
                )

            # Medium confidence: above min threshold but ambiguous
            logger.info(
                f"AMBIGUOUS match: {top_service_name_hybrid} "
                f"(cosine={top_cosine:.4f}, gap={cosine_gap:.4f}) - needs LLM confirmation"
            )
            return ClassificationResult(
                workflow=WorkflowType.SERVICE,
                confidence=0.5,
                metadata={
                    "matched_service_id": top_service_id,
                    "matched_service_name": top_service_name_hybrid,
                    "cosine_score": top_cosine,
                    "cosine_gap": cosine_gap,
                    "needs_llm_confirmation": True,
                    "top_results": hybrid_results[:3],
                },
                reasoning=(
                    f"Ambiguous match: {top_service_name_hybrid} "
                    f"(cosine={top_cosine:.4f}) - LLM confirmation needed"
                ),
            )

        except Exception as e:
            logger.error(f"Hybrid classification failed: {e}", exc_info=True)
            return ClassificationResult(
                workflow=WorkflowType.CONTEXT,
                confidence=1.0,
                metadata={"reason": "classification_error", "error": str(e)},
                reasoning=f"Classification error - falling back to Context/RAG: {e}",
            )

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate dense embedding for a query using the orchestration service.

        Args:
            query: Query text to embed

        Returns:
            List of floats representing the dense embedding, or None on failure
        """
        try:
            if not self.orchestration_service:
                logger.error("Orchestration service not available for embedding")
                return None

            result = self.orchestration_service.create_embeddings_for_indexer(
                texts=[query],
                environment="production",
                batch_size=1,
            )

            embeddings = result.get("embeddings", [])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]

            logger.error("No embedding returned for query")
            return None

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    async def _dense_search(
        self,
        dense_vector: List[float],
        top_k: int = DENSE_SEARCH_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Execute dense-only search on Qdrant to get actual cosine similarity scores.

        This is used as a pre-filter: the cosine scores tell us HOW RELEVANT
        the top results actually are, unlike RRF scores which are purely rank-based.

        Args:
            dense_vector: Dense embedding vector (3072-dim)
            top_k: Number of results to return

        Returns:
            List of result dicts with service metadata and cosine_score,
            deduplicated by service_id (best score per service)
        """
        try:
            search_payload = {
                "query": dense_vector,
                "using": "dense",
                "limit": top_k * 2,  # Get more to allow dedup by service
                "with_payload": True,
            }

            response = await self._qdrant_client.post(
                f"/collections/{QDRANT_COLLECTION}/points/query",
                json=search_payload,
            )

            if response.status_code != 200:
                logger.error(
                    f"Qdrant dense search failed: HTTP {response.status_code} - "
                    f"{response.text}"
                )
                return []

            search_results = response.json()
            points = search_results.get("result", {}).get("points", [])

            if not points:
                logger.info("No results from dense search")
                return []

            # Deduplicate by service_id (keep best cosine score per service)
            service_results: Dict[str, Dict[str, Any]] = {}
            for point in points:
                payload = point.get("payload", {})
                score = float(point.get("score", 0))
                service_id = payload.get("service_id", "unknown")

                if service_id not in service_results or score > service_results[
                    service_id
                ].get("cosine_score", 0):
                    service_results[service_id] = {
                        "service_id": service_id,
                        "name": payload.get("name", ""),
                        "description": payload.get("description", ""),
                        "examples": payload.get("examples", []),
                        "entities": payload.get("entities", []),
                        "context": payload.get("context", ""),
                        "point_type": payload.get("point_type", "unknown"),
                        "example_text": payload.get("example_text"),
                        "cosine_score": score,
                    }

            # Sort by cosine score descending
            sorted_results = sorted(
                service_results.values(),
                key=lambda x: x["cosine_score"],
                reverse=True,
            )

            logger.info(
                f"Dense search found {len(sorted_results)} unique services "
                f"(top cosine: {sorted_results[0]['cosine_score']:.4f})"
            )

            return sorted_results

        except httpx.TimeoutException:
            logger.error(f"Qdrant dense search timeout after {QDRANT_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)
            return []

    async def _hybrid_search(
        self,
        dense_vector: List[float],
        sparse_vector: SparseVector,
        top_k: int = HYBRID_SEARCH_TOP_K,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid search on Qdrant using prefetch + RRF fusion.

        Sends both dense and sparse vectors in a single Qdrant query,
        using the prefetch API for parallel retrieval and RRF for fusion.

        Args:
            dense_vector: Dense embedding vector (3072-dim)
            sparse_vector: SparseVector with indices and values
            top_k: Number of results to return

        Returns:
            List of result dicts with service metadata and rrf_score
        """
        try:
            # Check if collection exists and has data
            try:
                collection_info = await self._qdrant_client.get(
                    f"/collections/{QDRANT_COLLECTION}"
                )
                if collection_info.status_code == 200:
                    info = collection_info.json()
                    points_count = info.get("result", {}).get("points_count", 0)
                    if points_count == 0:
                        logger.info("Intent collection is empty - no services indexed")
                        return []
                else:
                    logger.warning(
                        f"Could not verify collection: HTTP {collection_info.status_code}"
                    )
                    return []
            except Exception as e:
                logger.warning(f"Could not verify intent collection: {e}")
                return []

            # Build hybrid search payload with prefetch + RRF
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

            # Add sparse prefetch only if sparse vector is non-empty
            if not sparse_vector.is_empty():
                search_payload["prefetch"].append(
                    {
                        "query": sparse_vector.to_dict(),
                        "using": "sparse",
                        "limit": top_k * 2,
                    }
                )

            response = await self._qdrant_client.post(
                f"/collections/{QDRANT_COLLECTION}/points/query",
                json=search_payload,
            )

            if response.status_code != 200:
                logger.error(
                    f"Qdrant hybrid search failed: HTTP {response.status_code} - "
                    f"{response.text}"
                )
                return []

            search_results = response.json()
            points = search_results.get("result", {}).get("points", [])

            if not points:
                logger.info("No results from hybrid search")
                return []

            # Parse and deduplicate results (group by service_id, keep best score)
            service_results: Dict[str, Dict[str, Any]] = {}
            for point in points:
                payload = point.get("payload", {})
                score = float(point.get("score", 0))
                service_id = payload.get("service_id", "unknown")

                if service_id not in service_results or score > service_results[
                    service_id
                ].get("rrf_score", 0):
                    service_results[service_id] = {
                        "service_id": service_id,
                        "name": payload.get("name", ""),
                        "description": payload.get("description", ""),
                        "examples": payload.get("examples", []),
                        "entities": payload.get("entities", []),
                        "context": payload.get("context", ""),
                        "point_type": payload.get("point_type", "unknown"),
                        "example_text": payload.get("example_text"),
                        "rrf_score": score,
                    }

            # Sort by RRF score descending
            sorted_results = sorted(
                service_results.values(),
                key=lambda x: x["rrf_score"],
                reverse=True,
            )

            logger.info(
                f"Hybrid search found {len(sorted_results)} unique services "
                f"from {len(points)} points"
            )

            for i, r in enumerate(sorted_results[:3]):
                logger.debug(
                    f"  Rank {i + 1}: {r['name']} "
                    f"(service_id={r['service_id']}, "
                    f"rrf_score={r['rrf_score']:.6f}, "
                    f"type={r['point_type']})"
                )

            return sorted_results

        except httpx.TimeoutException:
            logger.error(f"Qdrant hybrid search timeout after {QDRANT_TIMEOUT}s")
            return []
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}", exc_info=True)
            return []

    @overload
    async def route_to_workflow(
        self,
        classification: ClassificationResult,
        request: OrchestrationRequest,
        is_streaming: Literal[False] = False,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]: ...

    @overload
    async def route_to_workflow(
        self,
        classification: ClassificationResult,
        request: OrchestrationRequest,
        is_streaming: Literal[True],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> AsyncIterator[str]: ...

    async def route_to_workflow(
        self,
        classification: ClassificationResult,
        request: OrchestrationRequest,
        is_streaming: bool = False,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse, AsyncIterator[str]]:
        """
        Route request to appropriate workflow based on classification.

        Implements fallback chain: If a workflow returns None, tries the next layer.
        This ensures queries always get handled, even if primary workflow fails.

        Args:
            classification: Classification result from classify()
            request: Original orchestration request
            is_streaming: Whether to use streaming mode (for /orchestrate/stream)
            time_metric: Optional timing dictionary for workflow step tracking

        Returns:
            OrchestrationResponse for non-streaming mode
            AsyncIterator[str] for streaming mode

        Fallback Chain:
            SERVICE → CONTEXT → RAG → OOD
            Each layer returns None if it cannot handle, triggering next layer.
        """
        chat_id = request.chatId
        workflow_name = WORKFLOW_DISPLAY_NAMES.get(
            classification.workflow, classification.workflow.value
        )

        logger.info(
            f"[{chat_id}] Routing to {workflow_name} "
            f"(streaming: {is_streaming}, confidence: {classification.confidence:.2f})"
        )

        # Get the workflow executor
        workflow = self._get_workflow_executor(classification.workflow)

        if is_streaming:
            # STREAMING MODE: For /orchestrate/stream endpoint
            # Return the async iterator directly
            return self._execute_with_fallback_streaming(
                workflow=workflow,
                request=request,
                context=classification.metadata,
                start_layer=classification.workflow,
                time_metric=time_metric,
            )
        else:
            # NON-STREAMING MODE: For /orchestrate and /orchestrate/test endpoints
            return await self._execute_with_fallback_async(
                workflow=workflow,
                request=request,
                context=classification.metadata,
                start_layer=classification.workflow,
                time_metric=time_metric,
            )

    def _get_workflow_executor(self, workflow_type: WorkflowType) -> BaseWorkflow:
        """Get workflow executor instance for given workflow type."""
        workflow_map = {
            WorkflowType.SERVICE: self.service_workflow,
            WorkflowType.API_TOOL_CALLING: self.api_tool_workflow,
            WorkflowType.CONTEXT: self.context_workflow,
            WorkflowType.RAG: self.rag_workflow,
            WorkflowType.OOD: self.ood_workflow,
        }
        return workflow_map[workflow_type]

    def _is_workflow_enabled(self, workflow_type: WorkflowType) -> bool:
        """Return True if the given workflow type is enabled via feature flags.

        RAG and OOD are always enabled (they are the safety net fallbacks).
        """
        flag_map = {
            WorkflowType.SERVICE: FeatureFlags.SERVICE_WORKFLOW_ENABLED,
            WorkflowType.API_TOOL_CALLING: FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED,
            WorkflowType.CONTEXT: FeatureFlags.CONTEXT_WORKFLOW_ENABLED,
            WorkflowType.RAG: True,
            WorkflowType.OOD: True,
        }
        return flag_map.get(workflow_type, True)

    async def _try_api_tool_classification(
        self,
        query: str,
        request: Optional[OrchestrationRequest] = None,
        precomputed_embedding: Optional[List[float]] = None,
        min_cosine_override: Optional[float] = None,
    ) -> Optional[ClassificationResult]:
        """Search api_tool_collection and return a ClassificationResult if a match is found.

        Called when intent_collections search yields no usable service match.

        Args:
            query: User's query string.
            request: Orchestration request (provides environment + connection_id).
                When None, defaults to production environment.
            precomputed_embedding: Dense embedding vector already computed for this
                query by the service search step. When provided, the ATC searcher
                reuses it instead of making a second embedding API call.
            min_cosine_override: Optional cosine similarity threshold that replaces
                the default ATC minimum score configured in the searcher. When
                provided, only endpoints whose cosine score meets or exceeds this
                value are considered a match. Use a lower value to broaden matching
                (e.g. during multi-intent re-classification) or a higher value to
                tighten it. When None, the searcher's default threshold applies.

        Returns:
            ClassificationResult with API_TOOL_CALLING workflow if a match is found,
            or None if no endpoint matched.
        """
        if not FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED:
            logger.info("API_TOOL_CALLING_WORKFLOW_ENABLED=false — skipping ATC search")
            return None

        environment = request.environment if request else "production"
        connection_id = request.connection_id if request else None

        try:
            results = await self.api_tool_searcher.search(
                query=query,
                environment=environment,
                connection_id=connection_id,
                precomputed_embedding=precomputed_embedding,
                min_cosine_override=min_cosine_override,
            )
            if not results:
                return None

            matched = results[0]
            logger.info(
                f"API tool match: {matched.name!r} "
                f"(confidence={matched.confidence}, cosine={matched.cosine_score:.4f})"
            )

            # Suppress multi-intent hint results when the feature is disabled.
            # The searcher returns a hint result (multi_intent_hint=True) when the
            # disambiguator rejected all multi-candidates — it is only meaningful when
            # MULTI_INTENT_ENABLED=True.
            if matched.multi_intent_hint and not FeatureFlags.MULTI_INTENT_ENABLED:
                logger.info(
                    f"ATC: {matched.name!r} is a multi-intent hint but "
                    f"MULTI_INTENT_ENABLED=False — falling through to CONTEXT/RAG"
                )
                return None

            # ── Score-band gate: try multi-intent decomposition ───────────
            # A score between the min and high-confidence thresholds may indicate
            # a diluted embedding caused by multiple intents in one query.
            # Scores at or above the high-confidence threshold are normally clear
            # single matches — UNLESS the disambiguator already ran and rejected
            # all candidates (multi_intent_hint=True). In that case the score
            # landed just above the threshold only because two close intents
            # pushed the embedding upward together; IntentDecomposer must still run.
            in_ambiguous_band = (
                API_TOOL_MIN_THRESHOLD
                <= matched.cosine_score
                < API_TOOL_HIGH_CONFIDENCE_THRESHOLD
            )
            if (
                (in_ambiguous_band or matched.multi_intent_hint)
                and FeatureFlags.MULTI_INTENT_ENABLED
                and not matched.llm_validated
            ):
                reason = (
                    "multi_intent_hint (disambiguator rejected all candidates)"
                    if matched.multi_intent_hint
                    else f"cosine={matched.cosine_score:.4f} in ambiguous band "
                    f"[{API_TOOL_MIN_THRESHOLD}, {API_TOOL_HIGH_CONFIDENCE_THRESHOLD})"
                )
                logger.info(f"ATC: {reason} — running IntentDecomposer")
                decomposition = await self.intent_decomposer.decompose(query)

                if decomposition.mode == ExecutionMode.PARALLEL:
                    parallel_result = await self._try_parallel_api_tool_classification(
                        sub_queries=decomposition.sub_queries,
                        environment=environment,
                        connection_id=connection_id,
                        original_matched=matched,
                    )
                    if parallel_result is not None:
                        return parallel_result
                    # Fewer than 2 endpoints matched — fall through to single path

            # ── Single-endpoint path (unchanged) ─────────────────────────
            return ClassificationResult(
                workflow=WorkflowType.API_TOOL_CALLING,
                confidence=matched.cosine_score,
                metadata={
                    "matched_endpoint": matched.to_dict(),
                    "execution_mode": ExecutionMode.SINGLE,
                },
                reasoning=(
                    f"API tool match: {matched.name} "
                    f"(cosine={matched.cosine_score:.4f}, confidence={matched.confidence})"
                ),
            )

        except Exception as e:
            logger.error(f"API tool classification failed: {e}", exc_info=True)
        return None

    async def _try_parallel_api_tool_classification(
        self,
        sub_queries: list[str],
        environment: str,
        connection_id: Optional[str],
        original_matched: APIToolSearchResult,
    ) -> Optional[ClassificationResult]:
        """Run parallel endpoint searches for each sub-query from IntentDecomposer.

        Searches all sub-queries concurrently. Returns a ClassificationResult with
        execution_mode="parallel" if at least 2 distinct endpoints are matched, or
        None to signal the caller to fall back to the single-endpoint path.

        Args:
            sub_queries: Focused sub-queries from IntentDecomposer (2-3 items).
            environment: LLM environment from the original request.
            connection_id: Connection ID from the original request.
            original_matched: The gate search result (used as fallback reference).

        Returns:
            ClassificationResult with parallel metadata, or None if <2 matched.
        """

        async def search_one(sub_query: str) -> Optional[APIToolSearchResult]:
            try:
                sub_results = await self.api_tool_searcher.search(
                    query=sub_query,
                    environment=environment,
                    connection_id=connection_id,
                )
                return sub_results[0] if sub_results else None
            except Exception as exc:
                logger.warning(
                    f"ATC parallel sub-search failed for {sub_query!r}: {exc}"
                )
                return None

        sub_results = await asyncio.gather(*[search_one(q) for q in sub_queries])

        # Deduplicate by endpoint name — keep first occurrence
        seen: set[str] = set()
        matched_endpoints: list[dict[str, Any]] = []
        for result in sub_results:
            if result is None:
                continue
            name: str = result.name
            if name not in seen:
                seen.add(name)
                matched_endpoints.append(result.to_dict())

        if len(matched_endpoints) < 2:
            logger.info(
                f"ATC parallel: only {len(matched_endpoints)} distinct endpoint(s) matched "
                f"— falling back to single path"
            )
            return None

        logger.info(
            f"ATC parallel: {len(matched_endpoints)} distinct endpoints matched: "
            f"{[e.get('name') for e in matched_endpoints]}"
        )
        return ClassificationResult(
            workflow=WorkflowType.API_TOOL_CALLING,
            confidence=original_matched.cosine_score,
            metadata={
                "execution_mode": ExecutionMode.PARALLEL,
                "matched_endpoints": matched_endpoints,
            },
            reasoning=(
                f"Multi-intent parallel match: "
                f"{[e.get('name') for e in matched_endpoints]}"
            ),
        )

    async def _execute_with_fallback_async(
        self,
        workflow: BaseWorkflow,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        start_layer: WorkflowType,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """
        Execute workflow with fallback to subsequent layers (non-streaming).

        Implementation:
        1. Try primary workflow
        2. If returns None, try next layer in WORKFLOW_LAYER_ORDER
        3. Continue until workflow returns non-None result
        4. OOD workflow always returns result (never None)

        Args:
            workflow: Primary workflow executor
            request: Orchestration request
            context: Workflow context/metadata
            start_layer: Starting workflow type
            time_metric: Optional timing dictionary for tracking
        """
        chat_id = request.chatId
        workflow_name = WORKFLOW_DISPLAY_NAMES.get(start_layer, start_layer.value)

        logger.info(f"[{chat_id}] Executing {workflow_name} (non-streaming)")

        try:
            if self._is_workflow_enabled(start_layer):
                result = await workflow.execute_async(request, context, time_metric)

                if result is not None:
                    logger.info(f"[{chat_id}] {workflow_name} handled successfully")
                    return result

                # Implement layer-wise fallback chain
                logger.info(
                    f"[{chat_id}] {workflow_name} returned None, "
                    f"trying next layer in fallback chain"
                )
            else:
                logger.info(
                    f"[{chat_id}] {workflow_name} is disabled via feature flag, "
                    f"trying next layer in fallback chain"
                )

            # Get the layer order starting from current layer

            current_index = WORKFLOW_LAYER_ORDER.index(start_layer)
            remaining_layers = WORKFLOW_LAYER_ORDER[current_index + 1 :]

            # Try each subsequent layer in order
            for next_layer in remaining_layers:
                if not self._is_workflow_enabled(next_layer):
                    next_name = WORKFLOW_DISPLAY_NAMES.get(next_layer, next_layer.value)
                    logger.info(f"[{chat_id}] Skipping disabled workflow: {next_name}")
                    continue

                next_workflow = self._get_workflow_executor(next_layer)
                next_name = WORKFLOW_DISPLAY_NAMES.get(next_layer, next_layer.value)

                logger.info(
                    f"[{chat_id}] Falling back to {next_name} "
                    f"(Layer {WORKFLOW_LAYER_ORDER.index(next_layer) + 1})"
                )

                result = await next_workflow.execute_async(request, {}, time_metric)

                if result is not None:
                    logger.info(f"[{chat_id}] {next_name} handled successfully")
                    return result

                logger.info(f"[{chat_id}] {next_name} returned None, continuing...")

            # This should never happen since RAG/OOD should always return result
            raise RuntimeError("All workflows returned None (unexpected)")

        except Exception as e:
            logger.error(f"[{chat_id}] Error executing {workflow_name}: {e}")
            # Fallback to RAG on error
            logger.info(f"[{chat_id}] Falling back to RAG due to error")
            rag_result = await self.rag_workflow.execute_async(request, {}, time_metric)
            if rag_result is not None:
                return rag_result
            else:
                raise RuntimeError("RAG workflow returned None unexpectedly") from e

    async def _execute_with_fallback_streaming(
        self,
        workflow: BaseWorkflow,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        start_layer: WorkflowType,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> AsyncIterator[str]:
        """
        Execute workflow with fallback to subsequent layers (streaming).

        Implementation:
        1. Try primary workflow
        2. If returns None, try next layer in WORKFLOW_LAYER_ORDER
        3. Stream from the first workflow that returns non-None
        4. OOD workflow always returns result (never None)

        Args:
            workflow: Primary workflow executor
            request: Orchestration request
            context: Workflow context/metadata
            start_layer: Starting workflow type
            time_metric: Optional timing dictionary for tracking
        """
        chat_id = request.chatId
        workflow_name = WORKFLOW_DISPLAY_NAMES.get(start_layer, start_layer.value)

        logger.info(f"[{chat_id}] Executing {workflow_name} (streaming)")

        try:
            if self._is_workflow_enabled(start_layer):
                result = await workflow.execute_streaming(request, context, time_metric)

                if result is not None:
                    logger.info(f"[{chat_id}] {workflow_name} streaming started")
                    async for chunk in result:
                        yield chunk
                    return

                # Implement layer-wise fallback chain for streaming
                logger.info(
                    f"[{chat_id}] {workflow_name} returned None, "
                    f"trying next layer in fallback chain"
                )
            else:
                logger.info(
                    f"[{chat_id}] {workflow_name} is disabled via feature flag, "
                    f"trying next layer in fallback chain"
                )

            # Get the layer order starting from current layer

            current_index = WORKFLOW_LAYER_ORDER.index(start_layer)
            remaining_layers = WORKFLOW_LAYER_ORDER[current_index + 1 :]

            # Try each subsequent layer in order
            for next_layer in remaining_layers:
                if not self._is_workflow_enabled(next_layer):
                    next_name = WORKFLOW_DISPLAY_NAMES.get(next_layer, next_layer.value)
                    logger.info(f"[{chat_id}] Skipping disabled workflow: {next_name}")
                    continue

                next_workflow = self._get_workflow_executor(next_layer)
                next_name = WORKFLOW_DISPLAY_NAMES.get(next_layer, next_layer.value)

                layer_number = WORKFLOW_LAYER_ORDER.index(next_layer) + 1
                logger.info(
                    f"[{chat_id}] Falling back to {next_name} streaming "
                    f"(Layer {layer_number})"
                )

                result = await next_workflow.execute_streaming(
                    request, context, time_metric
                )

                if result is not None:
                    logger.info(f"[{chat_id}] {next_name} streaming started")
                    async for chunk in result:
                        yield chunk
                    return

                logger.info(f"[{chat_id}] {next_name} returned None, continuing...")

            # This should never happen
            raise RuntimeError("All workflows returned None in streaming (unexpected)")

        except Exception as e:
            logger.error(f"[{chat_id}] Error executing {workflow_name} streaming: {e}")
            # Fallback to RAG on error
            logger.info(f"[{chat_id}] Falling back to RAG streaming due to error")
            streaming_result = await self.rag_workflow.execute_streaming(
                request, context, time_metric
            )
            if streaming_result is not None:
                async for chunk in streaming_result:
                    yield chunk
            else:
                raise RuntimeError("RAG workflow returned None unexpectedly") from e

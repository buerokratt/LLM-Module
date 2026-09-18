"""
Main Contextual Retriever

Orchestrates the full Anthropic Contextual Retrieval pipeline:
- Dynamic provider detection for collection selection
- Semantic search on contextual embeddings
- BM25 lexical search on contextual content
- Dynamic score fusion using RRF

Achieves 49% improvement in retrieval accuracy.
"""

from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from src.loki_logger import LokiLogger
import asyncio
import time
from langfuse import observe

from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Type checking import to avoid circular dependency at runtime
if TYPE_CHECKING:
    from src.llm_orchestration_service import LLMOrchestrationService
from contextual_retrieval.provider_detection import DynamicProviderDetection
from contextual_retrieval.qdrant_search import QdrantContextualSearch

from contextual_retrieval.bm25_search import SmartBM25Search
from contextual_retrieval.rank_fusion import DynamicRankFusion

# Initialize Loki logger
logger = LokiLogger(service_name="contextual-retriever")


class ContextualRetriever:
    """
    Main contextual retrieval orchestrator implementing Anthropic methodology.

    This replaces the commented HybridRetriever in LLMOrchestrationService with
    enhanced contextual retrieval capabilities.
    """

    def __init__(
        self,
        qdrant_url: str,
        environment: str = "production",
        connection_id: Optional[str] = None,
        config_path: Optional[str] = None,
        llm_service: Optional["LLMOrchestrationService"] = None,
        shared_bm25: Optional[SmartBM25Search] = None,
    ) -> None:
        """
        Initialize contextual retriever.

        Args:
            qdrant_url: Qdrant server URL
            environment: Environment for model resolution
            connection_id: Optional connection ID
            config_path: Optional config file path
            llm_service: Optional LLM service instance (prevents circular dependency)
            shared_bm25: Optional pre-warmed SmartBM25Search singleton.  When
                provided the retriever skips the expensive index-build step during
                initialize() and reuses the already-ready index, eliminating the
                cold-start latency on the first query.
        """
        self.qdrant_url = qdrant_url
        self.environment = environment
        self.connection_id = connection_id

        # Store injected LLM service (for dependency injection)
        self._llm_service = llm_service

        # Load configuration
        self.config = (
            ConfigLoader.load_config(config_path)
            if config_path
            else ContextualRetrievalConfig()
        )

        # Initialize components with configuration
        self.provider_detection = DynamicProviderDetection(qdrant_url, self.config)
        self.qdrant_search = QdrantContextualSearch(qdrant_url, self.config)
        # Use the injected pre-warmed singleton when available; create a fresh
        # instance only as a fallback (avoids duplicate Qdrant scroll on startup).
        self.bm25_search: SmartBM25Search = (
            shared_bm25
            if shared_bm25 is not None
            else SmartBM25Search(qdrant_url, self.config)
        )
        self._bm25_is_shared: bool = shared_bm25 is not None
        self.rank_fusion = DynamicRankFusion(self.config)

        # State
        self.initialized = False

        # Connection pooling - cached per retrieval session
        self._session_llm_service = None

        # Embedding batching configuration
        self.enable_embedding_batching = True

    async def initialize(self) -> bool:
        """Initialize the retriever components."""
        try:
            logger.info("Initializing Contextual Retriever...")

            # If received a pre-warmed shared BM25 index, reuse it directly.
            # This is the normal startup path and adds zero latency to the first query.
            if self._bm25_is_shared and self.bm25_search.bm25_index is not None:
                logger.info(
                    "Using pre-warmed shared BM25 index - skipping BM25 build "
                    f"({len(self.bm25_search.chunk_mapping)} chunks ready)"
                )
            else:
                # No shared index available - build it now (fallback path).
                bm25_success = await self.bm25_search.initialize_index()
                if not bm25_success:
                    logger.warning("BM25 initialization failed - will skip BM25 search")

            self.initialized = True
            logger.info("Contextual Retriever initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Contextual Retriever: {e}")
            return False

    def _get_session_llm_service(self) -> "LLMOrchestrationService":
        """
        Get cached LLM service for current retrieval session.
        Uses injected service if available, creates new instance as fallback.
        """
        if self._session_llm_service is None:
            if self._llm_service is not None:
                # Use injected service (eliminates circular dependency)
                logger.debug("Using injected LLM service for session")
                self._session_llm_service = self._llm_service
            else:
                # No fallback - enforce dependency injection pattern
                raise RuntimeError(
                    "LLM service not injected. ContextualRetriever requires "
                    "LLMOrchestrationService to be provided via dependency injection. "
                    "Pass llm_service parameter during initialization."
                )

        return self._session_llm_service

    def _clear_session_cache(self) -> None:
        """Clear cached connections at end of retrieval session."""
        if self._session_llm_service is not None:
            logger.debug("Clearing session LLM service cache")
            self._session_llm_service = None

    @observe(name="retrieve_contextual_chunks", as_type="retriever")
    async def retrieve_contextual_chunks(
        self,
        original_question: str,
        refined_questions: List[str],
        environment: Optional[str] = None,
        connection_id: Optional[str] = None,
        # Use configuration defaults
        topk_semantic: Optional[int] = None,
        topk_bm25: Optional[int] = None,
        final_top_n: Optional[int] = None,
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """
        Retrieve contextual chunks using Anthropic methodology.

        This method signature matches the commented _retrieve_relevant_chunks method
        to ensure seamless integration.

        Args:
            original_question: Original user question
            refined_questions: Refined questions from prompt refinement
            environment: Override environment
            connection_id: Override connection ID
            topk_semantic: Top K semantic results
            topk_bm25: Top K BM25 results
            final_top_n: Final number of results

        Returns:
            List of contextual chunks with scores and metadata
        """
        if not self.initialized:
            logger.error("Contextual Retriever not initialized")
            return []

        # Apply configuration defaults
        topk_semantic = topk_semantic or self.config.search.topk_semantic
        topk_bm25 = topk_bm25 or self.config.search.topk_bm25
        final_top_n = final_top_n or self.config.search.final_top_n

        start_time = time.time()

        try:
            # Use provided environment or fallback to instance default
            env = environment or self.environment
            conn_id = connection_id or self.connection_id

            logger.info(
                f"Starting contextual retrieval for query: {original_question[:100]}..."
            )

            # Step 1: Dynamic provider detection
            collections = await self.provider_detection.detect_optimal_collections(
                env, conn_id
            )

            if not collections:
                logger.warning("No collections available for search")
                return []

            # Step 2: Execute multi-query searches in parallel for enhanced coverage
            semantic_results: List[Dict[str, Any]] = []
            bm25_results: List[Dict[str, Any]] = []

            if self.config.enable_parallel_search:
                semantic_task = self._semantic_search(
                    original_question,
                    refined_questions,
                    collections,
                    topk_semantic,
                    env,
                    conn_id,
                )
                bm25_task = self._bm25_search(
                    original_question, refined_questions, topk_bm25
                )

                search_results = await asyncio.gather(
                    semantic_task, bm25_task, return_exceptions=True
                )

                # Handle exceptions and assign results with proper type narrowing
                semantic_result = search_results[0]
                if isinstance(semantic_result, BaseException):
                    logger.error(f"Semantic search failed: {semantic_result}")
                    semantic_results = []
                else:
                    semantic_results = semantic_result

                bm25_result = search_results[1]
                if isinstance(bm25_result, BaseException):
                    logger.error(f"BM25 search failed: {bm25_result}")
                    bm25_results = []
                else:
                    bm25_results = bm25_result
            else:
                # Sequential execution
                semantic_results = await self._semantic_search(
                    original_question,
                    refined_questions,
                    collections,
                    topk_semantic,
                    env,
                    conn_id,
                )
                bm25_results = await self._bm25_search(
                    original_question, refined_questions, topk_bm25
                )

            # Step 4: Fuse results using dynamic RRF
            fused_results = self.rank_fusion.fuse_results(
                semantic_results, bm25_results, final_top_n
            )

            # Step 5: Convert to expected format for compatibility
            formatted_results = self._format_results_for_compatibility(fused_results)

            retrieval_time = time.time() - start_time
            logger.info(
                f"Contextual retrieval completed in {retrieval_time:.2f}s: "
                f"{len(semantic_results)} semantic + {len(bm25_results)} BM25 → "
                f"{len(formatted_results)} final chunks"
            )

            # Log fusion statistics
            fusion_stats = self.rank_fusion.calculate_fusion_stats(fused_results)
            logger.debug(f"Fusion stats: {fusion_stats}")

            return formatted_results

        except Exception as e:
            logger.error(f"Contextual retrieval failed: {e}")
            return []
        finally:
            # Clear session cache to free resources after retrieval
            self._clear_session_cache()

    async def _semantic_search(
        self,
        original_question: str,
        refined_questions: List[str],
        collections: List[str],
        limit: int,
        environment: str,
        connection_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Execute multi-query semantic search with parallel embedding generation.

        Implements Option 1: Parallel execution of semantic searches for all queries
        (original + refined) to match BM25's comprehensive query coverage.
        """
        try:
            all_queries = [original_question] + refined_questions
            logger.info(
                f"Starting multi-query semantic search with {len(all_queries)} queries"
            )

            # Generate embeddings and execute searches for all queries
            all_results = await self._execute_multi_query_searches(
                all_queries, collections, limit, environment, connection_id
            )

            # Deduplicate results by chunk_id while preserving best scores
            deduplicated_results = self._deduplicate_semantic_results(all_results)

            logger.info(
                f"Multi-query semantic search: {len(all_results)} total → {len(deduplicated_results)} unique chunks"
            )

            return deduplicated_results

        except Exception as e:
            logger.error(f"Multi-query semantic search failed: {e}")
            return []

    async def _execute_multi_query_searches(
        self,
        queries: List[str],
        collections: List[str],
        limit: int,
        environment: str,
        connection_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Execute semantic searches for multiple queries with optional batching."""
        if self.enable_embedding_batching and len(queries) > 1:
            return await self._execute_batch_query_searches(
                queries, collections, limit, environment, connection_id
            )
        else:
            return await self._execute_sequential_query_searches(
                queries, collections, limit, environment, connection_id
            )

    async def _execute_batch_query_searches(
        self,
        queries: List[str],
        collections: List[str],
        limit: int,
        environment: str,
        connection_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Execute semantic searches using batch embedding generation."""
        try:
            logger.info(f"Starting batch embedding for {len(queries)} queries")

            # Step 1: Generate all embeddings in a single batch
            llm_service = self._get_session_llm_service()
            batch_embeddings = self.qdrant_search.get_embeddings_for_queries_batch(
                queries, llm_service, environment, connection_id
            )

            if not batch_embeddings:
                logger.warning(
                    "Batch embedding failed, falling back to sequential processing"
                )
                return await self._execute_sequential_query_searches(
                    queries, collections, limit, environment, connection_id
                )

            logger.info(
                f"Successfully generated {len(batch_embeddings)} batch embeddings"
            )

            # Step 2: Execute searches with pre-computed embeddings in parallel
            search_tasks = [
                self._search_single_query_with_embedding(
                    query, i, embedding, collections, limit
                )
                for i, (query, embedding) in enumerate(
                    zip(queries, batch_embeddings, strict=True)
                )
            ]

            # Execute all searches in parallel
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Collect successful results
            all_results: List[Dict[str, Any]] = []
            successful_searches = 0

            for i, result in enumerate(search_results):
                if isinstance(result, Exception):
                    logger.warning(f"Batch search failed for query {i + 1}: {result}")
                    continue

                if result and isinstance(result, list):
                    successful_searches += 1
                    all_results.extend(result)

            logger.info(
                f"Completed {successful_searches}/{len(queries)} batch semantic searches, {len(all_results)} total results"
            )
            return all_results

        except Exception as e:
            logger.error(
                f"Batch query processing failed: {e}, falling back to sequential"
            )
            return await self._execute_sequential_query_searches(
                queries, collections, limit, environment, connection_id
            )

    async def _execute_sequential_query_searches(
        self,
        queries: List[str],
        collections: List[str],
        limit: int,
        environment: str,
        connection_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Execute semantic searches for multiple queries sequentially (fallback method)."""
        all_results: List[Dict[str, Any]] = []
        successful_searches = 0

        for i, query in enumerate(queries):
            results = await self._search_single_query(
                query, i, collections, limit, environment, connection_id
            )
            if results:
                successful_searches += 1
                all_results.extend(results)

        logger.info(
            f"Completed {successful_searches}/{len(queries)} sequential semantic searches, {len(all_results)} total results"
        )
        return all_results

    async def _search_single_query(
        self,
        query: str,
        query_index: int,
        collections: List[str],
        limit: int,
        environment: str,
        connection_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Execute semantic search for a single query."""
        try:
            # Generate embedding for this query using cached service
            llm_service = self._get_session_llm_service()
            embedding = self.qdrant_search.get_embedding_for_query_with_service(
                query, llm_service, environment, connection_id
            )

            if embedding is None:
                logger.warning(f"Failed to get embedding for query {query_index + 1}")
                return []

            # Execute semantic search
            results = await self.qdrant_search.search_contextual_embeddings(
                embedding, collections, limit
            )

            if results:
                # Add query context to each result for debugging
                for chunk in results:
                    chunk["source_query"] = (
                        query[:100] + "..." if len(query) > 100 else query
                    )
                    chunk["query_type"] = (
                        "original" if query_index == 0 else f"refined_{query_index}"
                    )
                return results

            return []

        except Exception as e:
            logger.warning(f"Search failed for query {query_index + 1}: {e}")
            return []

    async def _search_single_query_with_embedding(
        self,
        query: str,
        query_index: int,
        embedding: List[float],
        collections: List[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute semantic search for a single query with pre-computed embedding."""
        try:
            logger.debug(
                f"Starting search for query {query_index + 1} with pre-computed embedding"
            )

            results = await self.qdrant_search.search_contextual_embeddings_direct(
                embedding, collections, limit
            )

            if results:
                # Add query context to each result for debugging
                for chunk in results:
                    chunk["source_query"] = (
                        query[:100] + "..." if len(query) > 100 else query
                    )
                    chunk["query_type"] = (
                        "original" if query_index == 0 else f"refined_{query_index}"
                    )
                return results

            return []

        except Exception as e:
            logger.error(f"Query {query_index + 1} search with embedding failed: {e}")
            return []

    def _deduplicate_semantic_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate semantic search results by chunk_id, keeping the highest scoring version.
        """
        seen_chunks: Dict[str, Dict[str, Any]] = {}

        for result in results:
            chunk_id = result.get("chunk_id", result.get("id", "unknown"))
            score = result.get("score", 0)

            if chunk_id not in seen_chunks or score > seen_chunks[chunk_id].get(
                "score", 0
            ):
                seen_chunks[chunk_id] = result

        # Sort by score descending
        deduplicated = list(seen_chunks.values())
        deduplicated.sort(key=lambda x: x.get("score", 0), reverse=True)

        return deduplicated

    async def _bm25_search(
        self, query: str, refined_queries: List[str], limit: int
    ) -> List[Dict[str, Any]]:
        """Execute BM25 search with error handling."""
        try:
            return await self.bm25_search.search_bm25(query, refined_queries, limit)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def _format_results_for_compatibility(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Union[str, float, Dict[str, Any]]]]:
        """
        Format results to match the expected format for ResponseGeneratorAgent.

        ResponseGenerator expects: {"text": content, "meta": metadata}
        """
        formatted: List[Dict[str, Union[str, float, Dict[str, Any]]]] = []

        for i, result in enumerate(results):
            # Extract content - prefer contextual_content over original_content
            content_text = str(
                result.get("contextual_content", result.get("original_content", ""))
            )

            # Create metadata structure expected by ResponseGenerator
            metadata = {
                "source_file": str(result.get("document_url", "")),
                "source": str(result.get("document_url", "")),
                "chunk_id": str(result.get("chunk_id", result.get("id", f"chunk_{i}"))),
                "retrieval_type": "contextual",
                "primary_source": str(result.get("primary_source", "unknown")),
                "semantic_score": float(result.get("normalized_score", 0)),
                "bm25_score": float(result.get("normalized_bm25_score", 0)),
                "fused_score": float(result.get("fused_score", 0)),
                **result.get("metadata", {}),  # Include original metadata
            }

            # Create format expected by ResponseGeneratorAgent
            formatted_chunk: Dict[str, Union[str, float, Dict[str, Any]]] = {
                # Core fields expected by response generator
                "text": content_text,  # This is the key field ResponseGenerator looks for
                "meta": metadata,  # This is where ResponseGenerator gets source info
                # Legacy compatibility fields (for other components that might use them)
                "id": str(result.get("chunk_id", result.get("id", f"chunk_{i}"))),
                "score": float(result.get("fused_score", result.get("score", 0))),
                "content": content_text,
                "document_url": str(result.get("document_url", "")),
                "retrieval_type": "contextual",
            }

            formatted.append(formatted_chunk)

        return formatted

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all retrieval components."""
        health_status: Dict[str, Any] = {
            "initialized": self.initialized,
            "provider_detection": False,
            "qdrant_search": False,
            "bm25_search": False,
            "collections": {},
        }

        try:
            # Check provider detection
            collections = await self.provider_detection.detect_optimal_collections(
                self.environment, self.connection_id
            )
            health_status["provider_detection"] = len(collections) > 0

            # Check collection stats
            stats = await self.provider_detection.get_collection_stats()
            health_status["collections"] = stats

            # Check BM25 index
            health_status["bm25_search"] = self.bm25_search.bm25_index is not None

            # Check Qdrant connectivity
            health_status["qdrant_search"] = len(collections) > 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status["error"] = str(e)

        return health_status

    async def close(self) -> None:
        """Clean up resources."""
        try:
            await self.provider_detection.close()
            await self.qdrant_search.close()
            await self.bm25_search.close()
            logger.info("Contextual Retriever closed successfully")
        except Exception as e:
            logger.error(f"Error closing Contextual Retriever: {e}")

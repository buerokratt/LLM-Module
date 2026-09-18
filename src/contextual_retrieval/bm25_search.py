"""
In-Memory BM25 Search using rank-bm25

Implements fast lexical search on contextual content with smart refresh
when collection data changes.
"""

from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from src.loki_logger import LokiLogger
from rank_bm25 import BM25Okapi
import re
import asyncio

from contextual_retrieval.contextual_retrieval_api_client import get_http_client_manager
from contextual_retrieval.error_handler import SecureErrorHandler
from contextual_retrieval.constants import (
    HttpStatusConstants,
    ErrorContextConstants,
    LoggingConstants,
    SearchConstants,
)
from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Initialize Loki logger
logger = LokiLogger(service_name="bm25-search")

if TYPE_CHECKING:
    from contextual_retrieval.contextual_retrieval_api_client import HTTPClientManager


class SmartBM25Search:
    """In-memory BM25 search with smart refresh capabilities."""

    def __init__(
        self, qdrant_url: str, config: Optional["ContextualRetrievalConfig"] = None
    ) -> None:
        self.qdrant_url = qdrant_url
        self._config = config if config is not None else ConfigLoader.load_config()
        self._http_client_manager = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunk_mapping: Dict[int, Dict[str, Any]] = {}
        self.last_collection_stats: Dict[str, Any] = {}
        self.tokenizer_pattern = re.compile(r"\w+")  # Simple word tokenizer
        # Background refresh state - prevents blocking queries during index rebuild
        self._refresh_in_progress: bool = False
        self._refresh_lock: asyncio.Lock = asyncio.Lock()
        # Strong references to background tasks to prevent premature GC
        self._background_tasks: Set[asyncio.Task[None]] = set()

    async def _get_http_client_manager(self) -> "HTTPClientManager":
        """Get the HTTP client manager instance."""
        if self._http_client_manager is None:
            self._http_client_manager = await get_http_client_manager()
        return self._http_client_manager

    async def initialize_index(self) -> bool:
        """Build initial BM25 index from existing contextual collections."""
        try:
            logger.info("Building BM25 index from contextual collections...")

            # Fetch all contextual chunks from both collections
            all_chunks = await self._fetch_all_contextual_chunks()

            if not all_chunks:
                logger.warning("No chunks found for BM25 index")
                return False

            # Build corpus for BM25
            corpus: List[List[str]] = []
            self.chunk_mapping = {}

            for i, chunk in enumerate(all_chunks):
                # Combine contextual and original content for better matching
                contextual_content = chunk.get("contextual_content", "")
                original_content = chunk.get("original_content", "")

                # Prioritize contextual content but include original for completeness
                combined_content = f"{contextual_content} {original_content}"

                # Tokenize content
                tokenized = self._tokenize_text(combined_content)
                corpus.append(tokenized)

                # Store chunk mapping with index
                self.chunk_mapping[i] = chunk

            # Create BM25 index
            self.bm25_index = BM25Okapi(corpus)

            # Store collection stats for smart refresh
            self.last_collection_stats = await self._get_collection_stats()

            logger.info(f"BM25 index built with {len(corpus)} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize BM25 index: {e}")
            return False

    async def search_bm25(
        self, query: str, refined_queries: List[str], limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search BM25 index with automatic refresh check.

        Args:
            query: Original query
            refined_queries: List of refined queries from prompt refinement
            limit: Maximum results to return (uses config default if None)

        Returns:
            List of chunks with BM25 scores
        """
        # Use configuration default if not specified
        if limit is None:
            limit = self._config.search.topk_bm25

        try:
            # Check if index needs refresh (non-blocking: schedule background rebuild,
            # current query continues with the existing index to avoid latency).
            if await self._should_refresh_index():
                # Avoid scheduling multiple concurrent refresh tasks; coalesce while a
                # refresh is already in progress.
                if not self._refresh_in_progress:
                    logger.info(
                        "Collection data changed - scheduling background BM25 refresh "
                        "(current query uses existing index)"
                    )
                    task = asyncio.create_task(self._background_refresh_index())
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                else:
                    logger.debug(
                        "BM25 refresh already in progress; skipping scheduling of a "
                        "new background refresh task"
                    )

            if not self.bm25_index:
                logger.error("BM25 index not initialized")
                return []

            # Combine original and refined queries for comprehensive search
            all_queries = [query] + refined_queries
            combined_query = " ".join(all_queries)

            # Tokenize query
            tokenized_query = self._tokenize_text(combined_query)

            if not tokenized_query:
                logger.warning("Empty tokenized query")
                return []

            # Get BM25 scores
            scores = self.bm25_index.get_scores(tokenized_query)

            # Get top results (handle numpy array types)
            top_indices = scores.argsort()[-limit:][::-1]

            results: List[Dict[str, Any]] = []
            for idx in top_indices:  # Iterate over numpy array
                idx_int = int(idx)  # Convert numpy index to int
                score = float(scores[idx_int])
                if score > 0:  # Only positive scores
                    chunk = self.chunk_mapping[idx_int].copy()
                    chunk["bm25_score"] = score
                    chunk["score"] = score  # Standard score field
                    chunk["search_type"] = "bm25"
                    results.append(chunk)

            logger.info(f"BM25 search found {len(results)} chunks")

            # Detailed results at DEBUG level (filters based on log level config)
            logger.debug("=== BM25 SEARCH RESULTS BREAKDOWN ===")
            for i, chunk in enumerate(results[:10]):  # Show top 10 results
                content_preview = (
                    (chunk.get("original_content", "")[:150] + "...")
                    if len(chunk.get("original_content", "")) > 150
                    else chunk.get("original_content", "")
                )
                logger.debug(
                    f"  Rank {i + 1}: BM25_score={chunk['score']:.4f}, id={chunk.get('chunk_id', 'unknown')}"
                )
                logger.debug(f"           content: '{content_preview}'")
            logger.debug("=== END BM25 SEARCH RESULTS ===")

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    async def _background_refresh_index(self) -> None:
        """
        Rebuild the BM25 index in the background without blocking in-flight queries.

        Uses a lock to ensure only one rebuild runs at a time.  If a rebuild is
        already in progress when a second collection-change is detected, the
        duplicate request is silently discarded — the in-progress rebuild will
        capture the latest data anyway.
        """
        if self._refresh_in_progress:
            logger.debug("BM25 background refresh already running - skipping duplicate")
            return
        async with self._refresh_lock:
            if self._refresh_in_progress:
                return
            self._refresh_in_progress = True
        try:
            logger.info("Starting background BM25 index refresh...")
            await self.initialize_index()
            logger.info("Background BM25 index refresh complete")
        except Exception as e:
            logger.error(f"Background BM25 refresh failed: {e}")
        finally:
            self._refresh_in_progress = False

    async def _fetch_all_contextual_chunks(self) -> List[Dict[str, Any]]:
        """Fetch all chunks from contextual collections."""
        all_chunks: List[Dict[str, Any]] = []
        collections = ["contextual_chunks_azure", "contextual_chunks_aws"]

        for collection_name in collections:
            try:
                # Use scroll to get all points from collection
                chunks = await self._scroll_collection(collection_name)
                all_chunks.extend(chunks)
                logger.info(f"Fetched {len(chunks)} chunks from {collection_name}")

            except Exception as e:
                logger.warning(f"Failed to fetch chunks from {collection_name}: {e}")

        logger.info(f"Total chunks fetched for BM25 index: {len(all_chunks)}")
        return all_chunks

    async def _scroll_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Scroll through all points in a collection with pagination."""
        chunks: List[Dict[str, Any]] = []
        next_page_offset = None
        batch_count = 0

        try:
            client_manager = await self._get_http_client_manager()
            client = await client_manager.get_client()

            scroll_url = (
                f"{self.qdrant_url}/collections/{collection_name}/points/scroll"
            )

            # Pagination loop to fetch all chunks
            while True:
                scroll_payload = {
                    "limit": SearchConstants.DEFAULT_SCROLL_BATCH_SIZE,
                    "with_payload": True,
                    "with_vector": False,
                }

                # Add offset for continuation
                if next_page_offset is not None:
                    scroll_payload["offset"] = next_page_offset

                response = await client.post(scroll_url, json=scroll_payload)

                if response.status_code != HttpStatusConstants.OK:
                    SecureErrorHandler.log_secure_error(
                        error=Exception(
                            f"Failed to scroll collection with status {response.status_code}"
                        ),
                        context=ErrorContextConstants.PROVIDER_DETECTION,
                        request_url=scroll_url,
                        level=LoggingConstants.WARNING,
                    )
                    return chunks  # Return what we have so far

                result = response.json()
                points = result.get("result", {}).get("points", [])
                next_page_offset = result.get("result", {}).get("next_page_offset")

                # Add chunks from this batch
                for point in points:
                    payload = point.get("payload", {})
                    chunks.append(payload)

                batch_count += 1
                logger.debug(
                    f"Fetched batch {batch_count} with {len(points)} points from {collection_name}"
                )

                # Exit conditions: no more points or no next page offset
                if not points or next_page_offset is None:
                    break

            logger.debug(
                f"Completed scrolling {collection_name}: {len(chunks)} total chunks in {batch_count} batches"
            )
            return chunks

        except Exception as e:
            SecureErrorHandler.log_secure_error(
                error=e,
                context="bm25_collection_scroll",
                request_url=f"{self.qdrant_url}/collections/{collection_name}",
                level="error",
            )
            return []

    async def _should_refresh_index(self) -> bool:
        """Smart refresh: only when collection data changes."""
        try:
            current_stats = await self._get_collection_stats()

            # Compare with last known stats
            if current_stats != self.last_collection_stats:
                logger.info("Collection data changed - refresh needed")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to check refresh status: {e}")
            return False

    async def _get_collection_stats(self) -> Dict[str, Any]:
        """Get current statistics for all contextual collections."""
        stats: Dict[str, Any] = {}
        collections = ["contextual_chunks_azure", "contextual_chunks_aws"]

        for collection_name in collections:
            try:
                client_manager = await self._get_http_client_manager()
                client = await client_manager.get_client()
                response = await client.get(
                    f"{self.qdrant_url}/collections/{collection_name}"
                )

                if response.status_code == HttpStatusConstants.OK:
                    collection_info = response.json()
                    stats[collection_name] = {
                        "points_count": collection_info.get("result", {}).get(
                            "points_count", 0
                        ),
                        "status": collection_info.get("result", {}).get(
                            "status", "unknown"
                        ),
                    }
                else:
                    stats[collection_name] = {
                        "points_count": 0,
                        "status": "unavailable",
                    }

            except Exception as e:
                logger.warning(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = {"points_count": 0, "status": "error"}

        return stats

    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        if not text:
            return []

        # Convert to lowercase and extract words
        tokens = self.tokenizer_pattern.findall(text.lower())
        return tokens

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client_manager:
            await self._http_client_manager.close()

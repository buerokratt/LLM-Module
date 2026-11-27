"""
In-Memory BM25 Search using rank-bm25

Implements fast lexical search on contextual content with smart refresh
when collection data changes.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from src.loki_logger import LokiLogger
from rank_bm25 import BM25Okapi
import re
import asyncio
import time

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


class CollectionReadError(Exception):
    """A collection could not be read completely; an index built from it would
    silently miss chunks."""


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
        # time.monotonic() of the last index build attempt (refresh throttle)
        self._last_build_started: Optional[float] = None

    async def _get_http_client_manager(self) -> "HTTPClientManager":
        """Get the HTTP client manager instance."""
        if self._http_client_manager is None:
            self._http_client_manager = await get_http_client_manager()
        return self._http_client_manager

    async def initialize_index(self) -> bool:
        """Build the BM25 index from the contextual collections.

        The index is built in a worker thread and swapped in afterwards, so
        queries keep being served while it is built. If a collection cannot be
        read completely, the current index and its recorded collection stats
        are kept: a partial index must not be recorded as up to date.
        """
        self._last_build_started = time.monotonic()
        try:
            logger.info("Building BM25 index from contextual collections...")

            # Record the stats before reading, so chunks stored while the
            # collections are read count as a change for the next refresh.
            stats_before = await self._get_collection_stats()

            # Fetch all contextual chunks from both collections
            all_chunks = self._deduplicate_chunks(
                await self._fetch_all_contextual_chunks()
            )

            if not all_chunks:
                # An empty knowledge base is a valid state (nothing indexed yet,
                # or emptied for a re-index). Drop any old index so removed
                # chunks are not served, and record the stats so the smart
                # refresh fires once chunks appear instead of on every query.
                logger.warning("No chunks found for BM25 index")
                self.bm25_index = None
                self.chunk_mapping = {}
                self.last_collection_stats = stats_before
                return False

            # Tokenising every chunk and building BM25Okapi is CPU work: on the
            # event loop it froze every request for the whole build (review
            # finding H3). Build aside in a worker thread, then swap the index
            # and mapping in together, so a query never sees a new mapping with
            # the old index.
            bm25_index, chunk_mapping = await asyncio.to_thread(
                self._build_index, all_chunks
            )
            self.bm25_index, self.chunk_mapping = bm25_index, chunk_mapping

            # Store collection stats for smart refresh
            self.last_collection_stats = stats_before

            logger.info(f"BM25 index built with {len(chunk_mapping)} documents")
            return True

        except Exception as e:
            # The previous index (if any) keeps serving; the stats are not
            # updated, so the next refresh check tries again.
            logger.error(f"Failed to initialize BM25 index (previous index kept): {e}")
            return False

    def _build_index(
        self, chunks: List[Dict[str, Any]]
    ) -> Tuple[BM25Okapi, Dict[int, Dict[str, Any]]]:
        """Tokenise the chunks and build the BM25 index (runs in a worker thread)."""
        corpus: List[List[str]] = []
        chunk_mapping: Dict[int, Dict[str, Any]] = {}
        for i, chunk in enumerate(chunks):
            # Combine contextual and original content for better matching:
            # contextual content first, original for completeness
            combined_content = (
                f"{chunk.get('contextual_content', '')} "
                f"{chunk.get('original_content', '')}"
            )
            corpus.append(self._tokenize_text(combined_content))
            chunk_mapping[i] = chunk
        return BM25Okapi(corpus), chunk_mapping

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

            # One consistent snapshot for this query, even if a refresh swaps in
            # a new index meanwhile.
            bm25_index, chunk_mapping = self.bm25_index, self.chunk_mapping
            if not bm25_index:
                logger.warning(
                    "BM25 index is empty (knowledge base not indexed yet) - "
                    "skipping BM25 search"
                )
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
            scores = bm25_index.get_scores(tokenized_query)

            # Get top results (handle numpy array types)
            top_indices = scores.argsort()[-limit:][::-1]

            results: List[Dict[str, Any]] = []
            for idx in top_indices:  # Iterate over numpy array
                idx_int = int(idx)  # Convert numpy index to int
                score = float(scores[idx_int])
                if score > 0:  # Only positive scores
                    chunk = chunk_mapping[idx_int].copy()
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

    @staticmethod
    def _deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep one chunk per chunk_id.

        Chunk ids are deterministic (document hash + index), so while the
        knowledge base is re-indexed onto another embedding platform the same
        chunk exists in both collections. Duplicates would take BM25 top-k slots.
        Chunks without a chunk_id are kept as they are.
        """
        seen: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
            unique.append(chunk)
        if len(unique) < len(chunks):
            logger.info(
                f"BM25 index: skipped {len(chunks) - len(unique)} duplicate chunks "
                "(same chunk in more than one collection)"
            )
        return unique

    async def _fetch_all_contextual_chunks(self) -> List[Dict[str, Any]]:
        """Fetch all chunks from contextual collections.

        Raises:
            CollectionReadError: A collection could not be read completely. Not
                swallowed: an index without that collection's chunks would be
                recorded as up to date (review finding H3).
        """
        all_chunks: List[Dict[str, Any]] = []
        collections = ["contextual_chunks_azure", "contextual_chunks_aws"]

        for collection_name in collections:
            # Use scroll to get all points from collection
            chunks = await self._scroll_collection(collection_name)
            all_chunks.extend(chunks)
            logger.info(f"Fetched {len(chunks)} chunks from {collection_name}")

        logger.info(f"Total chunks fetched for BM25 index: {len(all_chunks)}")
        return all_chunks

    async def _scroll_collection(self, collection_name: str) -> List[Dict[str, Any]]:
        """Scroll through all points in a collection with pagination.

        Returns:
            Every chunk payload; empty when the collection does not exist.

        Raises:
            CollectionReadError: Qdrant failed part-way; the chunks read so far
                are discarded rather than returned as if complete.
        """
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

                if (
                    response.status_code == HttpStatusConstants.NOT_FOUND
                    and next_page_offset is None
                ):
                    # Collection not created yet (e.g. no chunks on this platform)
                    return []

                if response.status_code != HttpStatusConstants.OK:
                    SecureErrorHandler.log_secure_error(
                        error=Exception(
                            f"Failed to scroll collection with status {response.status_code}"
                        ),
                        context=ErrorContextConstants.PROVIDER_DETECTION,
                        request_url=scroll_url,
                        level=LoggingConstants.WARNING,
                    )
                    raise CollectionReadError(
                        f"Scrolling {collection_name} failed with status "
                        f"{response.status_code} after {len(chunks)} chunks"
                    )

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

        except CollectionReadError:
            raise
        except Exception as e:
            SecureErrorHandler.log_secure_error(
                error=e,
                context="bm25_collection_scroll",
                request_url=f"{self.qdrant_url}/collections/{collection_name}",
                level="error",
            )
            raise CollectionReadError(
                f"Scrolling {collection_name} failed after {len(chunks)} chunks: {e}"
            ) from e

    async def _should_refresh_index(self) -> bool:
        """Smart refresh: only when the number of chunks changes.

        Only point counts are compared: a collection's status flips between
        green and yellow while Qdrant optimises, which triggered rebuilds
        without any change in content. While an index exists, rebuilds are at
        most every BM25_MIN_REFRESH_INTERVAL_SECONDS (the hourly sync changes
        the count after nearly every document); an empty index refreshes on the
        first change so new content is searchable right away.
        """
        try:
            if (
                self.bm25_index is not None
                and self._last_build_started is not None
                and time.monotonic() - self._last_build_started
                < SearchConstants.BM25_MIN_REFRESH_INTERVAL_SECONDS
            ):
                return False

            current_stats = await self._get_collection_stats()
            if any(
                stats.get("points_count") is None for stats in current_stats.values()
            ):
                return False  # Qdrant could not be read; nothing to compare

            # Compare with last known stats
            if current_stats != self.last_collection_stats:
                logger.info("Collection data changed - refresh needed")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to check refresh status: {e}")
            return False

    async def _get_collection_stats(self) -> Dict[str, Any]:
        """Point count per contextual collection: 0 when the collection does not
        exist, None when Qdrant could not be read."""
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
                        )
                        or 0,
                    }
                elif response.status_code == HttpStatusConstants.NOT_FOUND:
                    stats[collection_name] = {"points_count": 0}
                else:
                    stats[collection_name] = {"points_count": None}

            except Exception as e:
                logger.warning(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = {"points_count": None}

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

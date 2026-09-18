"""
Dynamic Score Fusion for Contextual Retrieval

Combines semantic and BM25 search results using Reciprocal Rank Fusion (RRF)
without hardcoded weights, adapting dynamically to result distributions.
"""

from typing import List, Dict, Any, Optional
from src.loki_logger import LokiLogger

from contextual_retrieval.constants import QueryTypeConstants
from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Initialize Loki logger
logger = LokiLogger(service_name="rank-fusion")


class DynamicRankFusion:
    """Dynamic score fusion without hardcoded collection weights."""

    def __init__(self, config: Optional["ContextualRetrievalConfig"] = None) -> None:
        """
        Initialize rank fusion with configuration.

        Args:
            config: Configuration object (loads default if None)
        """
        self._config = config if config is not None else ConfigLoader.load_config()
        self.rrf_k = self._config.rank_fusion.rrf_k

    def fuse_results(
        self,
        semantic_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        final_top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse semantic and BM25 results using dynamic RRF.

        Args:
            semantic_results: Results from semantic search
            bm25_results: Results from BM25 search
            final_top_n: Number of final results to return (uses config default if None)

        Returns:
            Fused and ranked results
        """
        # Use configuration default if not specified
        if final_top_n is None:
            final_top_n = self._config.search.final_top_n

        try:
            logger.info(
                f"Fusing {len(semantic_results)} semantic + {len(bm25_results)} BM25 results"
            )

            # Normalize scores for fair comparison
            semantic_normalized = self._normalize_scores(semantic_results, "score")
            bm25_normalized = self._normalize_scores(bm25_results, "bm25_score")

            # Apply Reciprocal Rank Fusion
            fused_results = self._reciprocal_rank_fusion(
                semantic_normalized, bm25_normalized
            )

            # Sort by fused score and return top N
            fused_results.sort(key=lambda x: x.get("fused_score", 0), reverse=True)
            final_results = fused_results[:final_top_n]

            logger.info(f"Fusion completed: {len(final_results)} final results")

            # Detailed results at DEBUG level (filters based on log level config)
            logger.debug("=== RANK FUSION FINAL RESULTS ===")
            for i, chunk in enumerate(final_results):
                content_preview_len = self._config.rank_fusion.content_preview_length
                content_preview = (
                    (chunk.get("original_content", "")[:content_preview_len] + "...")
                    if len(chunk.get("original_content", "")) > content_preview_len
                    else chunk.get("original_content", "")
                )
                sem_score = chunk.get("semantic_score", 0)
                bm25_score = chunk.get("bm25_score", 0)
                fused_score = chunk.get("fused_score", 0)
                search_type = chunk.get("search_type", QueryTypeConstants.UNKNOWN)
                logger.debug(
                    f"  Final Rank {i + 1}: fused_score={fused_score:.4f}, semantic={sem_score:.4f}, bm25={bm25_score:.4f}, type={search_type}"
                )
                logger.debug(
                    f"                  id={chunk.get('chunk_id', QueryTypeConstants.UNKNOWN)}, content: '{content_preview}'"
                )
            logger.debug("=== END RANK FUSION RESULTS ===")

            return final_results

        except Exception as e:
            logger.error(f"Score fusion failed: {e}")
            # Fallback: return semantic results if available
            if semantic_results:
                return semantic_results[:final_top_n]
            return bm25_results[:final_top_n]

    def _normalize_scores(
        self, results: List[Dict[str, Any]], score_field: str
    ) -> List[Dict[str, Any]]:
        """
        Normalize scores to 0-1 range for fair fusion.

        Args:
            results: List of search results
            score_field: Field containing the score

        Returns:
            Results with normalized scores
        """
        if not results:
            return []

        # Extract scores
        scores = [r.get(score_field, 0) for r in results]

        if not scores or all(s == 0 for s in scores):
            return results

        # Min-max normalization
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            # All scores are the same
            for result in results:
                result["normalized_" + score_field] = 1.0
        else:
            for i, result in enumerate(results):
                original_score = scores[i]
                normalized = (original_score - min_score) / score_range
                result["normalized_" + score_field] = normalized

        return results

    def _reciprocal_rank_fusion(
        self, semantic_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply Reciprocal Rank Fusion algorithm.

        RRF Score = sum(1 / (k + rank)) for each search system
        where k is a constant (typically 60) and rank starts from 1
        """
        # Create mapping of chunk_id to results for deduplication
        chunk_scores: Dict[str, Dict[str, Any]] = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result.get("chunk_id", result.get("id", f"semantic_{rank}"))

            rrf_score = 1.0 / (self.rrf_k + rank)

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": result,
                    "semantic_rrf": rrf_score,
                    "bm25_rrf": 0.0,
                    "semantic_rank": rank,
                    "bm25_rank": None,
                }
            else:
                chunk_scores[chunk_id]["semantic_rrf"] = rrf_score
                chunk_scores[chunk_id]["semantic_rank"] = rank

        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.get("chunk_id", result.get("id", f"bm25_{rank}"))

            rrf_score = 1.0 / (self.rrf_k + rank)

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {
                    "chunk": result,
                    "semantic_rrf": 0.0,
                    "bm25_rrf": rrf_score,
                    "semantic_rank": None,
                    "bm25_rank": rank,
                }
            else:
                chunk_scores[chunk_id]["bm25_rrf"] = rrf_score
                chunk_scores[chunk_id]["bm25_rank"] = rank

        # Calculate final fused scores
        fused_results: List[Dict[str, Any]] = []
        for data in chunk_scores.values():
            chunk = data["chunk"].copy()

            # Calculate fused RRF score
            fused_score = float(data["semantic_rrf"]) + float(data["bm25_rrf"])

            # Add fusion metadata
            chunk["fused_score"] = fused_score
            chunk["semantic_rrf_score"] = data["semantic_rrf"]
            chunk["bm25_rrf_score"] = data["bm25_rrf"]
            chunk["semantic_rank"] = data["semantic_rank"]
            chunk["bm25_rank"] = data["bm25_rank"]

            # Determine primary source
            if data["semantic_rrf"] > data["bm25_rrf"]:
                chunk["primary_source"] = "semantic"
            elif data["bm25_rrf"] > data["semantic_rrf"]:
                chunk["primary_source"] = "bm25"
            else:
                chunk["primary_source"] = "hybrid"

            fused_results.append(chunk)

        logger.debug(f"RRF fusion produced {len(fused_results)} unique chunks")
        return fused_results

    def calculate_fusion_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics about the fusion process."""
        if not results:
            return {}

        semantic_only = sum(
            1 for r in results if r.get("semantic_rank") and not r.get("bm25_rank")
        )
        bm25_only = sum(
            1 for r in results if r.get("bm25_rank") and not r.get("semantic_rank")
        )
        both_sources = sum(
            1 for r in results if r.get("semantic_rank") and r.get("bm25_rank")
        )

        avg_fused_score = sum(r.get("fused_score", 0) for r in results) / len(results)

        return {
            "total_results": len(results),
            "semantic_only": semantic_only,
            "bm25_only": bm25_only,
            "both_sources": both_sources,
            "average_fused_score": avg_fused_score,
            "fusion_coverage": both_sources / len(results) if results else 0,
        }

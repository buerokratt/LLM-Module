"""
Contextual Retrieval Configuration

Centralized configuration for all contextual retrieval components including
HTTP client, search parameters, collections, and performance settings.
"""

from pydantic import BaseModel, Field
from typing import List
import yaml
from pathlib import Path
from src.loki_logger import LokiLogger

from contextual_retrieval.constants import (
    HttpClientConstants,
    SearchConstants,
    CollectionConstants,
    BM25Constants,
)

# Initialize Loki logger
logger = LokiLogger(service_name="contextual-retrieval-config")


class HttpClientConfig(BaseModel):
    """HTTP client configuration."""

    # Service resilience / Circuit breaker
    failure_threshold: int = Field(
        default=HttpClientConstants.DEFAULT_FAILURE_THRESHOLD,
        description="Circuit breaker failure threshold",
    )
    recovery_timeout: float = Field(
        default=HttpClientConstants.DEFAULT_RECOVERY_TIMEOUT,
        description="Circuit breaker recovery timeout (seconds)",
    )

    # Timeouts
    read_timeout: float = Field(
        default=HttpClientConstants.DEFAULT_READ_TIMEOUT,
        description="Default read timeout",
    )
    connect_timeout: float = Field(
        default=HttpClientConstants.DEFAULT_CONNECT_TIMEOUT,
        description="Connection timeout",
    )
    write_timeout: float = Field(
        default=HttpClientConstants.DEFAULT_WRITE_TIMEOUT, description="Write timeout"
    )
    pool_timeout: float = Field(
        default=HttpClientConstants.DEFAULT_POOL_TIMEOUT, description="Pool timeout"
    )

    # Connection pooling
    max_connections: int = Field(
        default=HttpClientConstants.DEFAULT_MAX_CONNECTIONS,
        description="Total connection pool size",
    )
    max_keepalive_connections: int = Field(
        default=HttpClientConstants.DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        description="Persistent connections",
    )
    keepalive_expiry: float = Field(
        default=HttpClientConstants.DEFAULT_KEEPALIVE_EXPIRY,
        description="Connection reuse duration",
    )

    # Retry logic
    max_retries: int = Field(
        default=HttpClientConstants.DEFAULT_MAX_RETRIES,
        description="Maximum retry attempts",
    )
    retry_delay: float = Field(
        default=HttpClientConstants.DEFAULT_RETRY_DELAY,
        description="Initial delay between retries",
    )
    backoff_factor: float = Field(
        default=HttpClientConstants.DEFAULT_BACKOFF_FACTOR,
        description="Exponential backoff multiplier",
    )


class CollectionConfig(BaseModel):
    """Collection configuration."""

    auto_detect_provider: bool = Field(
        default=CollectionConstants.DEFAULT_AUTO_DETECT_PROVIDER,
        description="Auto-detect optimal collections",
    )
    search_timeout_seconds: int = Field(
        default=SearchConstants.DEFAULT_SEARCH_TIMEOUT, description="Search timeout"
    )

    # Collection names
    azure_collection: str = Field(
        default=CollectionConstants.AZURE_COLLECTION,
        description="Azure collection name",
    )
    aws_collection: str = Field(
        default=CollectionConstants.AWS_COLLECTION, description="AWS collection name"
    )

    # Provider detection keywords
    azure_keywords: List[str] = Field(
        default=CollectionConstants.AZURE_KEYWORDS,
        description="Azure provider keywords",
    )
    aws_keywords: List[str] = Field(
        default=CollectionConstants.AWS_KEYWORDS, description="AWS provider keywords"
    )


class SearchConfig(BaseModel):
    """Search configuration."""

    topk_semantic: int = Field(
        default=SearchConstants.DEFAULT_TOPK_SEMANTIC,
        description="Top K semantic search results",
    )
    topk_bm25: int = Field(
        default=SearchConstants.DEFAULT_TOPK_BM25,
        description="Top K BM25 search results",
    )
    final_top_n: int = Field(
        default=SearchConstants.DEFAULT_FINAL_TOP_N,
        description="Final chunks returned to LLM",
    )
    score_threshold: float = Field(
        default=SearchConstants.DEFAULT_SCORE_THRESHOLD,
        description="Minimum score threshold",
    )


class BM25Config(BaseModel):
    """BM25 configuration."""

    library: str = Field(
        default=BM25Constants.DEFAULT_LIBRARY, description="BM25 implementation"
    )
    refresh_strategy: str = Field(
        default=BM25Constants.DEFAULT_REFRESH_STRATEGY,
        description="Index refresh strategy",
    )
    max_refresh_interval_seconds: int = Field(
        default=BM25Constants.DEFAULT_MAX_REFRESH_INTERVAL,
        description="Max refresh interval",
    )


class RankFusionConfig(BaseModel):
    """Rank fusion configuration."""

    rrf_k: int = Field(
        default=SearchConstants.DEFAULT_RRF_K,
        description="Reciprocal Rank Fusion constant",
    )
    content_preview_length: int = Field(
        default=SearchConstants.CONTENT_PREVIEW_LENGTH,
        description="Content preview truncation length",
    )


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    enable_parallel_search: bool = Field(
        default=True, description="Run semantic + BM25 in parallel"
    )
    enable_dynamic_scoring: bool = Field(
        default=True, description="Enable dynamic scoring"
    )
    batch_size: int = Field(
        default=SearchConstants.DEFAULT_BATCH_SIZE,
        description="Default batch size for operations",
    )


class ContextualRetrievalConfig(BaseModel):
    """Configuration for contextual retrieval system."""

    # Configuration sections
    search: SearchConfig = Field(
        default_factory=SearchConfig, description="Search configuration"
    )
    http_client: HttpClientConfig = Field(
        default_factory=HttpClientConfig, description="HTTP client configuration"
    )
    collections: CollectionConfig = Field(
        default_factory=CollectionConfig, description="Collection configuration"
    )
    bm25: BM25Config = Field(
        default_factory=BM25Config, description="BM25 configuration"
    )
    rank_fusion: RankFusionConfig = Field(
        default_factory=RankFusionConfig, description="Rank fusion configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig, description="Performance configuration"
    )

    # Legacy properties for backward compatibility
    @property
    def topk_semantic(self) -> int:
        return self.search.topk_semantic

    @property
    def topk_bm25(self) -> int:
        return self.search.topk_bm25

    @property
    def final_top_n(self) -> int:
        return self.search.final_top_n

    @property
    def auto_detect_provider(self) -> bool:
        return self.collections.auto_detect_provider

    @property
    def search_timeout_seconds(self) -> int:
        return self.collections.search_timeout_seconds

    @property
    def bm25_library(self) -> str:
        return self.bm25.library

    @property
    def refresh_strategy(self) -> str:
        return self.bm25.refresh_strategy

    @property
    def enable_parallel_search(self) -> bool:
        return self.performance.enable_parallel_search

    @property
    def max_refresh_interval_seconds(self) -> int:
        return self.bm25.max_refresh_interval_seconds


class ConfigLoader:
    """Load contextual retrieval configuration from YAML file."""

    @staticmethod
    def load_config(
        config_path: str = "src/contextual_retrieval/config/contextual_retrieval_config.yaml",
    ) -> ContextualRetrievalConfig:
        """Load configuration from YAML file."""

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(
                f"Contextual retrieval config {config_path} not found, using defaults"
            )
            return ContextualRetrievalConfig()

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)

            # Extract contextual_retrieval section
            retrieval_config = yaml_config.get("contextual_retrieval", {})

            # Load search configuration
            search_config_data = retrieval_config.get("search", {})
            search_config = SearchConfig(
                topk_semantic=search_config_data.get(
                    "topk_semantic", SearchConstants.DEFAULT_TOPK_SEMANTIC
                ),
                topk_bm25=search_config_data.get(
                    "topk_bm25", SearchConstants.DEFAULT_TOPK_BM25
                ),
                final_top_n=search_config_data.get(
                    "final_top_n", SearchConstants.DEFAULT_FINAL_TOP_N
                ),
                score_threshold=search_config_data.get(
                    "score_threshold", SearchConstants.DEFAULT_SCORE_THRESHOLD
                ),
            )

            # Load HTTP client configuration
            http_client_config_data = retrieval_config.get("http_client", {})
            http_client_config = HttpClientConfig(
                failure_threshold=http_client_config_data.get(
                    "failure_threshold", HttpClientConstants.DEFAULT_FAILURE_THRESHOLD
                ),
                recovery_timeout=http_client_config_data.get(
                    "recovery_timeout", HttpClientConstants.DEFAULT_RECOVERY_TIMEOUT
                ),
                read_timeout=http_client_config_data.get(
                    "read_timeout", HttpClientConstants.DEFAULT_READ_TIMEOUT
                ),
                connect_timeout=http_client_config_data.get(
                    "connect_timeout", HttpClientConstants.DEFAULT_CONNECT_TIMEOUT
                ),
                write_timeout=http_client_config_data.get(
                    "write_timeout", HttpClientConstants.DEFAULT_WRITE_TIMEOUT
                ),
                pool_timeout=http_client_config_data.get(
                    "pool_timeout", HttpClientConstants.DEFAULT_POOL_TIMEOUT
                ),
                max_connections=http_client_config_data.get(
                    "max_connections", HttpClientConstants.DEFAULT_MAX_CONNECTIONS
                ),
                max_keepalive_connections=http_client_config_data.get(
                    "max_keepalive_connections",
                    HttpClientConstants.DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
                ),
                keepalive_expiry=http_client_config_data.get(
                    "keepalive_expiry", HttpClientConstants.DEFAULT_KEEPALIVE_EXPIRY
                ),
                max_retries=http_client_config_data.get(
                    "max_retries", HttpClientConstants.DEFAULT_MAX_RETRIES
                ),
                retry_delay=http_client_config_data.get(
                    "retry_delay", HttpClientConstants.DEFAULT_RETRY_DELAY
                ),
                backoff_factor=http_client_config_data.get(
                    "backoff_factor", HttpClientConstants.DEFAULT_BACKOFF_FACTOR
                ),
            )

            # Load collections configuration
            collections_config_data = retrieval_config.get("collections", {})
            collections_config = CollectionConfig(
                auto_detect_provider=collections_config_data.get(
                    "auto_detect_provider",
                    CollectionConstants.DEFAULT_AUTO_DETECT_PROVIDER,
                ),
                search_timeout_seconds=collections_config_data.get(
                    "search_timeout_seconds", SearchConstants.DEFAULT_SEARCH_TIMEOUT
                ),
                azure_collection=collections_config_data.get(
                    "azure_collection", CollectionConstants.AZURE_COLLECTION
                ),
                aws_collection=collections_config_data.get(
                    "aws_collection", CollectionConstants.AWS_COLLECTION
                ),
                azure_keywords=collections_config_data.get(
                    "azure_keywords", CollectionConstants.AZURE_KEYWORDS
                ),
                aws_keywords=collections_config_data.get(
                    "aws_keywords", CollectionConstants.AWS_KEYWORDS
                ),
            )

            # Load BM25 configuration
            bm25_config_data = retrieval_config.get("bm25", {})
            bm25_config = BM25Config(
                library=bm25_config_data.get("library", BM25Constants.DEFAULT_LIBRARY),
                refresh_strategy=bm25_config_data.get(
                    "refresh_strategy", BM25Constants.DEFAULT_REFRESH_STRATEGY
                ),
                max_refresh_interval_seconds=bm25_config_data.get(
                    "max_refresh_interval_seconds",
                    BM25Constants.DEFAULT_MAX_REFRESH_INTERVAL,
                ),
            )

            # Load rank fusion configuration
            rank_fusion_config_data = retrieval_config.get("rank_fusion", {})
            rank_fusion_config = RankFusionConfig(
                rrf_k=rank_fusion_config_data.get(
                    "rrf_k", SearchConstants.DEFAULT_RRF_K
                ),
                content_preview_length=rank_fusion_config_data.get(
                    "content_preview_length", SearchConstants.CONTENT_PREVIEW_LENGTH
                ),
            )

            # Load performance configuration
            performance_config_data = retrieval_config.get("performance", {})
            performance_config = PerformanceConfig(
                enable_parallel_search=performance_config_data.get(
                    "enable_parallel_search", True
                ),
                enable_dynamic_scoring=performance_config_data.get(
                    "enable_dynamic_scoring", True
                ),
                batch_size=performance_config_data.get(
                    "batch_size", SearchConstants.DEFAULT_BATCH_SIZE
                ),
            )

            return ContextualRetrievalConfig(
                search=search_config,
                http_client=http_client_config,
                collections=collections_config,
                bm25=bm25_config,
                rank_fusion=rank_fusion_config,
                performance=performance_config,
            )

        except Exception as e:
            logger.error(
                f"Failed to load contextual retrieval config {config_path}: {e}"
            )
            return ContextualRetrievalConfig()

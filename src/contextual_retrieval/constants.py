"""
Constants for Contextual Retrieval System

Centralized constants for HTTP client, search operations, collections,
and other configurable values across the contextual retrieval system.
"""

from vector_indexer.constants import ResponseGenerationConstants


class HttpClientConstants:
    """HTTP client configuration constants."""

    # Circuit breaker / Service resilience
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RECOVERY_TIMEOUT = 60.0

    # Timeouts in seconds
    DEFAULT_READ_TIMEOUT = 30.0
    DEFAULT_CONNECT_TIMEOUT = 10.0
    DEFAULT_WRITE_TIMEOUT = 10.0
    DEFAULT_POOL_TIMEOUT = 60.0

    # Connection pooling
    DEFAULT_MAX_CONNECTIONS = 100
    DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
    DEFAULT_KEEPALIVE_EXPIRY = 30.0

    # Retry logic
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0
    DEFAULT_BACKOFF_FACTOR = 2.0

    # Transport settings
    DEFAULT_TRANSPORT_RETRIES = 0  # Handle retries at application level
    USE_HTTP2 = False  # Use HTTP/1.1 for better Qdrant compatibility
    FOLLOW_REDIRECTS = True


class SearchConstants:
    """Search configuration constants."""

    # Default search parameters
    DEFAULT_TOPK_SEMANTIC = 40
    DEFAULT_TOPK_BM25 = 40
    # Final top-N chunks returned after RRF fusion.
    DEFAULT_FINAL_TOP_N = ResponseGenerationConstants.DEFAULT_MAX_BLOCKS
    DEFAULT_SEARCH_TIMEOUT = 2

    # Score and quality thresholds
    DEFAULT_SCORE_THRESHOLD = 0.4  # Lowered from 0.5 for better semantic diversity
    DEFAULT_BATCH_SIZE = 1

    # Relevance gating
    DEFAULT_REQUIRE_SEMANTIC_CORROBORATION = True
    """Discard BM25-only result sets when semantic search found nothing above
    score_threshold. BM25 has no relevance floor - it returns top-N by keyword
    score however weak the match - so without this a contentless query returns
    lexical noise instead of an out-of-scope response."""

    # Rank fusion
    DEFAULT_RRF_K = 35  # Lowered from 60 for better score differentiation
    CONTENT_PREVIEW_LENGTH = 150

    # Normalization
    MIN_NORMALIZED_SCORE = 0.0
    MAX_NORMALIZED_SCORE = 1.0

    # BM25 indexing
    DEFAULT_SCROLL_BATCH_SIZE = 100  # Batch size for scrolling through collections


class CollectionConstants:
    """Collection and provider constants."""

    # Collection names
    AZURE_COLLECTION = "contextual_chunks_azure"
    AWS_COLLECTION = "contextual_chunks_aws"
    ALL_COLLECTIONS = [AZURE_COLLECTION, AWS_COLLECTION]

    # Provider detection keywords
    AZURE_KEYWORDS = ["azure", "text-embedding", "ada-002"]
    AWS_KEYWORDS = ["titan", "amazon", "aws", "bedrock"]

    # Default settings
    DEFAULT_AUTO_DETECT_PROVIDER = True


class HttpStatusConstants:
    """HTTP status code constants."""

    # Success codes
    OK = 200

    # Error ranges
    CLIENT_ERROR_START = 400
    CLIENT_ERROR_END = 500
    SERVER_ERROR_START = 500

    # Retry logic status codes
    SUCCESS_THRESHOLD = 400  # < 400 considered success
    RETRY_THRESHOLD = 500  # >= 500 can be retried


class CircuitBreakerConstants:
    """Circuit breaker state constants."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    # Valid states list for validation
    VALID_STATES = [CLOSED, OPEN, HALF_OPEN]


class ErrorContextConstants:
    """Error context constants for secure logging."""

    # Circuit breaker contexts
    CIRCUIT_BREAKER = "circuit_breaker"
    CIRCUIT_BREAKER_BLOCKED = "circuit_breaker_blocked"
    CIRCUIT_BREAKER_REQUEST = "circuit_breaker_request"

    # HTTP client contexts
    HTTP_CLIENT_CREATION = "http_client_creation"
    HTTP_CLIENT_CLEANUP = "http_client_cleanup"
    HTTP_CLIENT_HEALTH_CHECK = "http_client_health_check"

    # Retry contexts
    HTTP_RETRY_ATTEMPT = "http_retry_attempt"
    HTTP_RETRY_EXHAUSTED = "http_retry_exhausted"
    HTTP_RETRY_CLIENT_ERROR = "http_retry_client_error"

    # Provider contexts
    PROVIDER_HEALTH_CHECK = "provider_health_check"
    PROVIDER_DETECTION = "provider_detection"


class BM25Constants:
    """BM25 configuration constants."""

    DEFAULT_LIBRARY = "rank-bm25"
    DEFAULT_REFRESH_STRATEGY = "smart"
    DEFAULT_MAX_REFRESH_INTERVAL = 3600  # 1 hour


class QueryTypeConstants:
    """Query type constants for search tracking."""

    ORIGINAL = "original"
    REFINED_PREFIX = "refined_"
    UNKNOWN = "unknown"

    # Search types
    SEMANTIC = "semantic"
    BM25 = "bm25"
    HYBRID = "hybrid"


class ConfigKeyConstants:
    """Configuration file key constants."""

    # Main sections
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"
    SEARCH = "search"
    COLLECTIONS = "collections"
    BM25 = "bm25"
    HTTP_CLIENT = "http_client"
    RANK_FUSION = "rank_fusion"
    PERFORMANCE = "performance"

    # Search config keys
    TOPK_SEMANTIC = "topk_semantic"
    TOPK_BM25 = "topk_bm25"
    FINAL_TOP_N = "final_top_n"
    SEARCH_TIMEOUT_SECONDS = "search_timeout_seconds"
    SCORE_THRESHOLD = "score_threshold"

    # Collection config keys
    AUTO_DETECT_PROVIDER = "auto_detect_provider"
    AZURE_COLLECTION_KEY = "azure_collection"
    AWS_COLLECTION_KEY = "aws_collection"
    AZURE_KEYWORDS_KEY = "azure_keywords"
    AWS_KEYWORDS_KEY = "aws_keywords"

    # BM25 config keys
    LIBRARY = "library"
    REFRESH_STRATEGY = "refresh_strategy"
    MAX_REFRESH_INTERVAL_SECONDS = "max_refresh_interval_seconds"

    # Performance config keys
    ENABLE_PARALLEL_SEARCH = "enable_parallel_search"
    ENABLE_DYNAMIC_SCORING = "enable_dynamic_scoring"


class LoggingConstants:
    """Logging configuration constants."""

    # Log levels
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    # Log message templates
    CIRCUIT_BREAKER_OPENED_MSG = "Circuit breaker opened after {failure_count} failures"
    REQUEST_RETRY_MSG = (
        "Request failed, retrying in {delay}s (attempt {attempt}/{max_attempts})"
    )
    REQUEST_SUCCESS_MSG = "Request succeeded on attempt {attempt}"

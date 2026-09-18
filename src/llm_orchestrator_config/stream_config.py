"""Stream configuration for timeouts and size limits."""


class StreamConfig:
    """Hardcoded configuration for streaming limits and timeouts."""

    # Timeout Configuration
    MAX_STREAM_DURATION_SECONDS: int = 300  # 5 minutes, total wall clock
    # Measured between chunks, not cumulatively: a long answer that keeps
    # producing tokens is never cut off, however long it takes in total.
    IDLE_TIMEOUT_SECONDS: int = 60  # 1 minute with no output at all
    # How often to emit an SSE comment frame while the stream is quiet, so
    # intermediate proxies see traffic and hold the connection open.
    HEARTBEAT_INTERVAL_SECONDS: int = 15

    # Size Limits
    MAX_MESSAGE_LENGTH: int = 10000  # Maximum characters in message
    MAX_PAYLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB

    # Token Limits (reuse existing tracking from response_generator)
    MAX_TOKENS_PER_STREAM: int = 4000  # Maximum tokens to generate

    # Concurrency Limits
    MAX_CONCURRENT_STREAMS: int = 100  # System-wide concurrent stream limit
    MAX_STREAMS_PER_USER: int = 5  # Per-user concurrent stream limit

    # Rate Limiting Configuration
    RATE_LIMIT_ENABLED: bool = True  # Enable/disable rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 20  # Max requests per user per minute
    RATE_LIMIT_TOKENS_PER_MINUTE: int = 40_000  # Max tokens per user per minute
    RATE_LIMIT_CLEANUP_INTERVAL: int = 300  # Cleanup old entries every 5 minutes
    RATE_LIMIT_TOKEN_WINDOW_SECONDS: int = 60  # Sliding window size for token tracking

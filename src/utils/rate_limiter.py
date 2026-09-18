"""Rate limiter for streaming endpoints with sliding window algorithms."""

import time
from collections import defaultdict, deque
from typing import Dict, Deque, Optional, Any
from threading import Lock
from pydantic import BaseModel, Field, ConfigDict

from src.loki_logger import LokiLogger
from src.llm_orchestrator_config.stream_config import StreamConfig

# Initialize Loki logger
logger = LokiLogger(service_name="rate-limiter")


class RateLimitResult(BaseModel):
    """Result of rate limit check."""

    model_config = ConfigDict(frozen=True)  # Make immutable like dataclass

    allowed: bool
    retry_after: Optional[int] = Field(
        default=None, description="Seconds to wait before retrying"
    )
    limit_type: Optional[str] = Field(
        default=None, description="'requests' or 'tokens'"
    )
    current_usage: Optional[int] = Field(
        default=None, description="Current usage count"
    )
    limit: Optional[int] = Field(default=None, description="Maximum allowed limit")


class RateLimiter:
    """
    In-memory rate limiter using sliding windows for both requests and tokens.

    Features:
    - Sliding window for request rate limiting (e.g., 10 requests per minute)
    - Sliding window for token rate limiting (e.g., 40,000 tokens per minute)
    - Per-user tracking with authorId
    - Automatic cleanup of old entries to prevent memory leaks
    - Thread-safe operations

    Usage:
        rate_limiter = RateLimiter(
            requests_per_minute=10,
            tokens_per_minute=40_000,
        )

        result = rate_limiter.check_rate_limit(
            author_id="user-123",
            estimated_tokens=50
        )

        if not result.allowed:
            # Return 429 with retry_after
            pass
    """

    def __init__(
        self,
        requests_per_minute: int = StreamConfig.RATE_LIMIT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = StreamConfig.RATE_LIMIT_TOKENS_PER_MINUTE,
        cleanup_interval: int = StreamConfig.RATE_LIMIT_CLEANUP_INTERVAL,
        token_window_seconds: int = StreamConfig.RATE_LIMIT_TOKEN_WINDOW_SECONDS,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per user per minute (sliding window)
            tokens_per_minute: Maximum tokens per user per minute (sliding window)
            cleanup_interval: Seconds between automatic cleanup of old entries
            token_window_seconds: Sliding window size in seconds for token tracking
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.cleanup_interval = cleanup_interval
        self.token_window_seconds = token_window_seconds
        # Scale the per-minute limit to the actual window size so the
        # sliding-window comparison is consistent regardless of window length.
        self.tokens_per_window = int(tokens_per_minute * token_window_seconds / 60)

        # Sliding window: Track request timestamps per user
        self._request_history: Dict[str, Deque[float]] = defaultdict(deque)

        # Sliding window: Track token usage per user
        self._token_history: Dict[str, Deque[tuple[float, int]]] = defaultdict(deque)

        # Thread safety
        self._lock = Lock()

        # Cleanup tracking
        self._last_cleanup = time.time()

        logger.info(
            f"RateLimiter initialized - "
            f"requests_per_minute: {requests_per_minute}, "
            f"tokens_per_minute: {tokens_per_minute}"
        )

    def check_rate_limit(
        self,
        author_id: str,
        estimated_tokens: int = 0,
    ) -> RateLimitResult:
        """
        Check if request is allowed under rate limits.

        Args:
            author_id: User identifier for rate limiting
            estimated_tokens: Estimated tokens for this request (for token bucket)

        Returns:
            RateLimitResult with allowed status and retry information
        """
        with self._lock:
            current_time = time.time()

            # Periodic cleanup to prevent memory leaks
            if current_time - self._last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries(current_time)

            # Check 1: Sliding window (requests per minute)
            request_result = self._check_request_limit(author_id, current_time)
            if not request_result.allowed:
                return request_result

            # Check 2: Sliding window (tokens per minute)
            if estimated_tokens > 0:
                token_result = self._check_token_limit(
                    author_id, estimated_tokens, current_time
                )
                if not token_result.allowed:
                    return token_result

            # Both checks passed - record the request
            self._record_request(author_id, current_time, estimated_tokens)

            return RateLimitResult(allowed=True)

    def _check_request_limit(
        self,
        author_id: str,
        current_time: float,
    ) -> RateLimitResult:
        """
        Check sliding window request limit.

        Args:
            author_id: User identifier
            current_time: Current timestamp

        Returns:
            RateLimitResult for request limit check
        """
        request_history = self._request_history[author_id]
        window_start = current_time - 60  # 60 seconds = 1 minute

        # Remove requests outside the sliding window
        while request_history and request_history[0] < window_start:
            request_history.popleft()

        # Check if limit exceeded
        current_requests = len(request_history)
        if current_requests >= self.requests_per_minute:
            # Calculate retry_after based on oldest request in window
            oldest_request = request_history[0]
            retry_after = int(oldest_request + 60 - current_time) + 1

            logger.warning(
                f"Rate limit exceeded for {author_id} - "
                f"requests: {current_requests}/{self.requests_per_minute} "
                f"(retry after {retry_after}s)"
            )

            return RateLimitResult(
                allowed=False,
                retry_after=retry_after,
                limit_type="requests",
                current_usage=current_requests,
                limit=self.requests_per_minute,
            )

        return RateLimitResult(allowed=True)

    def _check_token_limit(
        self,
        author_id: str,
        estimated_tokens: int,
        current_time: float,
    ) -> RateLimitResult:
        """
        Check sliding window token limit.

        Sliding window algorithm:
        - Track cumulative tokens consumed within the window
        - Reject if adding estimated tokens would exceed the limit

        Args:
            author_id: User identifier
            estimated_tokens: Tokens needed for this request
            current_time: Current timestamp

        Returns:
            RateLimitResult for token limit check
        """
        token_history = self._token_history[author_id]
        window_start = current_time - self.token_window_seconds

        # Remove entries outside the sliding window
        while token_history and token_history[0][0] < window_start:
            token_history.popleft()

        # Sum tokens consumed in the current window
        current_token_usage = sum(tokens for _, tokens in token_history)

        # Check if adding this request would exceed the scaled window limit
        if current_token_usage + estimated_tokens > self.tokens_per_window:
            # Calculate retry_after based on oldest entry in window
            if token_history:
                oldest_timestamp = token_history[0][0]
                retry_after = (
                    int(oldest_timestamp + self.token_window_seconds - current_time) + 1
                )
            else:
                retry_after = 1

            logger.warning(
                f"Token rate limit exceeded for {author_id} - "
                f"needed: {estimated_tokens}, "
                f"current_usage: {current_token_usage}/{self.tokens_per_window} "
                f"(window: {self.token_window_seconds}s, "
                f"rate: {self.tokens_per_minute}/min, "
                f"retry after {retry_after}s)"
            )

            return RateLimitResult(
                allowed=False,
                retry_after=retry_after,
                limit_type="tokens",
                current_usage=current_token_usage,
                limit=self.tokens_per_window,
            )

        return RateLimitResult(allowed=True)

    def _record_request(
        self,
        author_id: str,
        current_time: float,
        tokens_consumed: int,
    ) -> None:
        """
        Record a successful request.

        Args:
            author_id: User identifier
            current_time: Current timestamp
            tokens_consumed: Tokens consumed by this request
        """
        # Record request timestamp for sliding window
        self._request_history[author_id].append(current_time)

        # Record token usage for sliding window
        if tokens_consumed > 0:
            self._token_history[author_id].append((current_time, tokens_consumed))

    def _cleanup_old_entries(self, current_time: float) -> None:
        """
        Clean up old entries to prevent memory leaks.

        Args:
            current_time: Current timestamp
        """
        logger.debug("Running rate limiter cleanup...")

        # Clean up request history (remove entries older than 1 minute)
        window_start = current_time - 60
        users_to_remove: list[str] = []

        for author_id, request_history in self._request_history.items():
            # Remove old requests
            while request_history and request_history[0] < window_start:
                request_history.popleft()

            # Remove empty histories
            if not request_history:
                users_to_remove.append(author_id)

        for author_id in users_to_remove:
            del self._request_history[author_id]

        # Clean up token history (remove entries outside window + inactive users)
        token_window_start = current_time - self.token_window_seconds
        token_users_to_remove: list[str] = []

        for author_id, token_history in self._token_history.items():
            while token_history and token_history[0][0] < token_window_start:
                token_history.popleft()
            if not token_history:
                token_users_to_remove.append(author_id)

        for author_id in token_users_to_remove:
            del self._token_history[author_id]

        self._last_cleanup = current_time

        if users_to_remove or token_users_to_remove:
            logger.debug(
                f"Cleaned up {len(users_to_remove)} request histories and "
                f"{len(token_users_to_remove)} token histories"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with stats about current usage
        """
        with self._lock:
            return {
                "total_users_tracked": len(self._request_history),
                "total_token_histories": len(self._token_history),
                "requests_per_minute_limit": self.requests_per_minute,
                "tokens_per_minute_limit": self.tokens_per_minute,
                "last_cleanup": self._last_cleanup,
            }

    def reset_user(self, author_id: str) -> None:
        """
        Reset rate limits for a specific user (useful for testing).

        Args:
            author_id: User identifier to reset
        """
        with self._lock:
            if author_id in self._request_history:
                del self._request_history[author_id]
            if author_id in self._token_history:
                del self._token_history[author_id]

            logger.info(f"Reset rate limits for user: {author_id}")

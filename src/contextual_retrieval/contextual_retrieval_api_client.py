"""
HTTP Client Manager for Contextual Retrieval

Centralized HTTP client management with proper connection pooling,
lifecycle management, and resource cleanup for all contextual retrieval components.
"""

import asyncio
from typing import Optional, Dict, Any
import httpx
from src.loki_logger import LokiLogger
import time

from contextual_retrieval.error_handler import SecureErrorHandler
from contextual_retrieval.constants import (
    HttpClientConstants,
    HttpStatusConstants,
    CircuitBreakerConstants,
    ErrorContextConstants,
    LoggingConstants,
)
from contextual_retrieval.config import ConfigLoader, ContextualRetrievalConfig

# Initialize Loki logger
logger = LokiLogger(service_name="contextual-retrieval-api-client")


class ServiceResilienceManager:
    """Service resilience manager with circuit breaker functionality for HTTP requests."""

    def __init__(self, config: Optional["ContextualRetrievalConfig"] = None) -> None:
        # Load configuration if not provided
        if config is None:
            config = ConfigLoader.load_config()

        self.failure_threshold = config.http_client.failure_threshold
        self.recovery_timeout = config.http_client.recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerConstants.CLOSED

    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerConstants.CLOSED:
            return True
        elif self.state == CircuitBreakerConstants.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitBreakerConstants.HALF_OPEN
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record successful request."""
        self.failure_count = 0
        self.state = CircuitBreakerConstants.CLOSED

    def record_failure(self) -> None:
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerConstants.OPEN
            SecureErrorHandler.log_secure_error(
                error=Exception(
                    LoggingConstants.CIRCUIT_BREAKER_OPENED_MSG.format(
                        failure_count=self.failure_count
                    )
                ),
                context=ErrorContextConstants.CIRCUIT_BREAKER,
                level=LoggingConstants.WARNING,
            )


class HTTPClientManager:
    """
    Centralized HTTP client manager for contextual retrieval components.

    Provides shared HTTP client with proper connection pooling, timeout management,
    and guaranteed resource cleanup. Thread-safe and designed for concurrent usage.
    """

    _instance: Optional["HTTPClientManager"] = None
    _lock = asyncio.Lock()

    def __init__(self, config: Optional["ContextualRetrievalConfig"] = None) -> None:
        """Initialize HTTP client manager."""
        # Load configuration if not provided
        self._config = config if config is not None else ConfigLoader.load_config()

        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
        self._is_closed = False
        self._circuit_breaker = ServiceResilienceManager(self._config)

    @classmethod
    async def get_instance(cls) -> "HTTPClientManager":
        """Get singleton instance of HTTP client manager."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = HTTPClientManager()
        return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton instance (for cleanup/testing purposes)."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None

    async def get_client(
        self, timeout_seconds: Optional[float] = None
    ) -> httpx.AsyncClient:
        """
        Get shared HTTP client with proper connection pooling.

        Args:
            timeout_seconds: Request timeout in seconds (uses config default if None)

        Returns:
            Configured httpx.AsyncClient instance

        Raises:
            RuntimeError: If client manager has been closed
        """
        # Use configured timeout if not specified
        if timeout_seconds is None:
            timeout_seconds = self._config.http_client.read_timeout
        if self._is_closed:
            raise RuntimeError("HTTP Client Manager has been closed")

        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    try:
                        logger.debug(
                            "Creating shared HTTP client with connection pooling"
                        )
                        self._client = httpx.AsyncClient(
                            timeout=httpx.Timeout(
                                connect=self._config.http_client.connect_timeout,
                                read=timeout_seconds,
                                write=self._config.http_client.write_timeout,
                                pool=self._config.http_client.pool_timeout,
                            ),
                            limits=httpx.Limits(
                                max_connections=self._config.http_client.max_connections,
                                max_keepalive_connections=self._config.http_client.max_keepalive_connections,
                                keepalive_expiry=self._config.http_client.keepalive_expiry,
                            ),
                            # Connection pooling settings
                            http2=HttpClientConstants.USE_HTTP2,
                            follow_redirects=HttpClientConstants.FOLLOW_REDIRECTS,
                            # Retry configuration for resilience
                            transport=httpx.AsyncHTTPTransport(
                                retries=HttpClientConstants.DEFAULT_TRANSPORT_RETRIES
                            ),
                        )
                        logger.info(
                            "HTTP client manager initialized with connection pooling"
                        )
                    except Exception as e:
                        SecureErrorHandler.log_secure_error(
                            error=e,
                            context=ErrorContextConstants.HTTP_CLIENT_CREATION,
                            level=LoggingConstants.ERROR,
                        )
                        raise RuntimeError(
                            SecureErrorHandler.sanitize_error_message(
                                e, "HTTP client initialization"
                            )
                        ) from e

        return self._client

    async def close(self) -> None:
        """
        Close HTTP client and cleanup resources.

        This method is idempotent and can be called multiple times safely.
        """
        if self._is_closed:
            return

        async with self._client_lock:
            if self._client is not None:
                try:
                    logger.debug("Closing shared HTTP client")
                    await self._client.aclose()
                    self._client = None
                    logger.info("HTTP client manager closed successfully")
                except Exception as e:
                    SecureErrorHandler.log_secure_error(
                        error=e,
                        context=ErrorContextConstants.HTTP_CLIENT_CLEANUP,
                        level=LoggingConstants.WARNING,
                    )
                    # Still mark as closed even if cleanup failed
                    self._client = None

            self._is_closed = True

    def health_check(self) -> bool:
        """
        Perform health check on HTTP client.

        Returns:
            True if client is healthy, False otherwise
        """
        try:
            if self._is_closed or self._client is None:
                return False

            # Check circuit breaker state
            if not self._circuit_breaker.can_execute():
                return False

            # Basic client state check
            return not self._client.is_closed

        except Exception as e:
            SecureErrorHandler.log_secure_error(
                error=e,
                context=ErrorContextConstants.HTTP_CLIENT_HEALTH_CHECK,
                level=LoggingConstants.WARNING,
            )
            return False

    async def execute_with_circuit_breaker(
        self, method: str, url: str, **kwargs: Any
    ) -> Optional[httpx.Response]:
        """
        Execute HTTP request with circuit breaker protection and retries.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request parameters

        Returns:
            Response if successful, None if circuit breaker is open or all retries failed
        """
        if not self._circuit_breaker.can_execute():
            SecureErrorHandler.log_secure_error(
                error=Exception(f"Circuit breaker is {self._circuit_breaker.state}"),
                context=ErrorContextConstants.CIRCUIT_BREAKER_BLOCKED,
                request_url=url,
                level=LoggingConstants.WARNING,
            )
            return None

        try:
            client = await self.get_client()
            response = await retry_http_request(client, method, url, **kwargs)

            if (
                response
                and response.status_code < HttpStatusConstants.SERVER_ERROR_START
            ):
                self._circuit_breaker.record_success()
            else:
                self._circuit_breaker.record_failure()

            return response

        except Exception as e:
            self._circuit_breaker.record_failure()
            SecureErrorHandler.log_secure_error(
                error=e,
                context=ErrorContextConstants.CIRCUIT_BREAKER_REQUEST,
                request_url=url,
                level=LoggingConstants.ERROR,
            )
            return None

    @property
    def is_closed(self) -> bool:
        """Check if client manager is closed."""
        return self._is_closed

    # Context Manager Protocol
    async def __aenter__(self) -> "HTTPClientManager":
        """
        Async context manager entry.

        Returns:
            Self for use within the context
        """
        # Ensure client is initialized
        await self.get_client()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """
        Async context manager exit with guaranteed cleanup.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        await self.close()

    @property
    def client_stats(self) -> Dict[str, Any]:
        """Get client connection statistics."""
        if self._client is None or self._is_closed:
            return {"status": "closed", "active_connections": 0}

        try:
            # Basic client information
            stats: Dict[str, Any] = {
                "status": "active",
                "is_closed": self._client.is_closed,
            }

            return stats

        except Exception as e:
            logger.debug(f"Could not get client stats: {e}")
            return {"status": "active", "stats_unavailable": True}


# Global instance for easy access
_global_manager: Optional[HTTPClientManager] = None


async def get_http_client_manager() -> HTTPClientManager:
    """
    Get global HTTP client manager instance.

    Convenience function for accessing the shared HTTP client manager.

    Returns:
        HTTPClientManager instance
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = await HTTPClientManager.get_instance()
    return _global_manager


async def get_managed_http_client_session() -> HTTPClientManager:
    """
    Get HTTP client manager as a context manager for session-based usage.

    Example:
        async with get_managed_http_client_session() as manager:
            client = await manager.get_client()
            response = await client.get("http://example.com")

    Returns:
        HTTPClientManager: Instance ready for context manager usage
    """
    return await HTTPClientManager.get_instance()


async def retry_http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: Optional[int] = None,
    retry_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    config: Optional["ContextualRetrievalConfig"] = None,
    **kwargs: Any,
) -> Optional[httpx.Response]:
    """
    Execute HTTP request with retry logic and secure error handling.

    Args:
        client: HTTP client to use
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts (uses config default if None)
        retry_delay: Initial delay between retries in seconds (uses config default if None)
        backoff_factor: Multiplier for retry delay after each attempt (uses config default if None)
        config: Configuration object (loads default if None)
        **kwargs: Additional arguments for the HTTP request

    Returns:
        Response object if successful, None if all retries failed
    """
    # Load configuration if not provided
    if config is None:
        config = ConfigLoader.load_config()

    # Use configuration defaults if parameters not specified
    if max_retries is None:
        max_retries = config.http_client.max_retries
    if retry_delay is None:
        retry_delay = config.http_client.retry_delay
    if backoff_factor is None:
        backoff_factor = config.http_client.backoff_factor

    last_error = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)

            # Consider 2xx and 3xx as success
            if response.status_code < HttpStatusConstants.SUCCESS_THRESHOLD:
                if attempt > 0:
                    logger.info(
                        LoggingConstants.REQUEST_SUCCESS_MSG.format(attempt=attempt + 1)
                    )
                return response

            # 4xx errors usually shouldn't be retried (client errors)
            if (
                HttpStatusConstants.CLIENT_ERROR_START
                <= response.status_code
                < HttpStatusConstants.CLIENT_ERROR_END
            ):
                SecureErrorHandler.log_secure_error(
                    error=httpx.HTTPStatusError(
                        f"Client error {response.status_code}",
                        request=response.request,
                        response=response,
                    ),
                    context=ErrorContextConstants.HTTP_RETRY_CLIENT_ERROR,
                    request_url=url,
                    request_headers=kwargs.get("headers"),
                    level=LoggingConstants.WARNING,
                )
                return response  # Don't retry client errors

            # 5xx errors can be retried (server errors)
            last_error = httpx.HTTPStatusError(
                f"Server error {response.status_code}",
                request=response.request,
                response=response,
            )

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            last_error = e
        except Exception as e:
            last_error = e

        # Log retry attempt
        if attempt < max_retries:
            SecureErrorHandler.log_secure_error(
                error=last_error,
                context=ErrorContextConstants.HTTP_RETRY_ATTEMPT,
                request_url=url,
                level=LoggingConstants.DEBUG,
            )
            logger.debug(
                LoggingConstants.REQUEST_RETRY_MSG.format(
                    delay=current_delay,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                )
            )

            # Wait before retry with exponential backoff
            await asyncio.sleep(current_delay)
            current_delay *= backoff_factor

    # All retries exhausted
    if last_error:
        SecureErrorHandler.log_secure_error(
            error=last_error,
            context=ErrorContextConstants.HTTP_RETRY_EXHAUSTED,
            request_url=url,
            request_headers=kwargs.get("headers"),
            level=LoggingConstants.ERROR,
        )

    return None


async def cleanup_http_client_manager() -> None:
    """
    Cleanup global HTTP client manager.

    Should be called during application shutdown to ensure proper resource cleanup.
    """
    global _global_manager
    if _global_manager is not None:
        await HTTPClientManager.reset_instance()
        _global_manager = None

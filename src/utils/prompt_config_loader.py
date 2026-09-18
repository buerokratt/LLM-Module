"""
Prompt configuration loader with HTTP client, caching, and retry logic.
"""

import requests
from typing import Optional, Dict, Any
import time
import threading
from enum import Enum
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="prompt-config-loader")


class PromptConfigLoadError(Exception):
    """Raised when all retry attempts to load prompt configuration fail."""

    pass


class RefreshStatus(Enum):
    """Status of a refresh operation."""

    SUCCESS = "success"  # Configuration loaded successfully
    NOT_FOUND = "not_found"  # Configuration absent in database
    FETCH_FAILED = "fetch_failed"  # Network/HTTP/upstream errors


class PromptConfigurationLoader:
    """
    Loads custom prompt configurations from Ruuter endpoint.

    Features:
    - HTTP-based loading via Ruuter
    - 5-minute TTL cache (configurable)
    - 3-attempt retry with exponential backoff
    - Thread-safe caching
    - Graceful degradation with stale cache fallback
    """

    def __init__(
        self,
        ruuter_endpoint: str,
        cache_ttl_seconds: int = 300,
        max_retries: int = 3,
        timeout_seconds: int = 10,
    ) -> None:
        """
        Initialize prompt configuration loader.

        Args:
            ruuter_endpoint: Full URL to Ruuter endpoint
            cache_ttl_seconds: Cache TTL in seconds (default: 300 = 5 minutes)
            max_retries: Maximum retry attempts on failure (default: 3)
            timeout_seconds: HTTP request timeout (default: 10)
        """
        self.ruuter_endpoint = ruuter_endpoint
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Cache storage
        self._cached_prompt: Optional[str] = None
        self._cache_timestamp: Optional[float] = None
        self._cache_lock = threading.Lock()
        self._cache_condition = threading.Condition(self._cache_lock)
        self._fetch_in_progress = False

        # Statistics for monitoring
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_failures = 0
        self._last_error: Optional[str] = None

        logger.info(
            f"PromptConfigurationLoader initialized: "
            f"endpoint={ruuter_endpoint}, ttl={cache_ttl_seconds}s, retries={max_retries}"
        )

    def get_custom_instructions(self) -> str:
        """
        Get custom prompt configuration (cached or fresh).

        Uses fine-grained locking with thundering herd prevention:
        - Quick cache check under lock
        - Release lock during slow network I/O
        - Only one thread fetches, others wait
        - Re-acquire lock to update cache

        Returns:
            str: Custom instruction text, or empty string if unavailable
        """
        # Step 1: Quick cache check under lock
        with self._cache_condition:
            # Check cache validity
            if self._is_cache_valid():
                self._cache_hits += 1
                logger.debug(
                    f"Prompt config cache HIT "
                    f"(age: {self._get_cache_age():.1f}s, "
                    f"hits: {self._cache_hits}, misses: {self._cache_misses})"
                )
                return self._cached_prompt or ""

            # Cache miss/expired
            self._cache_misses += 1
            logger.info(
                f"Prompt config cache MISS - loading from Ruuter "
                f"(cache age: {self._get_cache_age():.1f}s)"
            )

            # Thundering herd prevention: if another thread is fetching, wait
            while self._fetch_in_progress:
                logger.debug("Another thread is fetching, waiting...")
                self._cache_condition.wait()  # Release lock and wait
                # After waking up, check if cache was updated
                if self._is_cache_valid():
                    logger.debug("Cache updated by another thread")
                    return self._cached_prompt or ""

            # We're the first one, mark fetch in progress
            self._fetch_in_progress = True

        # Step 2: Fetch WITHOUT holding lock (allows concurrent cache reads)
        prompt_text = None
        fetch_error = None
        try:
            prompt_text = self._load_from_ruuter_with_retry()

        except PromptConfigLoadError as e:
            fetch_error = e
            logger.error(f"Failed to fetch prompt configuration after retries: {e}")

        except Exception as e:
            fetch_error = e
            logger.error(f"Unexpected error loading prompt configuration: {e}")

        # Step 3: Update cache and notify waiters (lock re-acquired)
        with self._cache_condition:
            try:
                if prompt_text:
                    # Success - update cache
                    self._cached_prompt = prompt_text
                    self._cache_timestamp = time.time()
                    self._last_error = None
                    logger.info(
                        f"Prompt configuration loaded successfully "
                        f"({len(prompt_text)} chars)"
                    )
                    return prompt_text

                elif prompt_text is None and fetch_error is None:
                    # Prompt field not found - cache empty result to avoid repeated loads
                    logger.warning(
                        "Prompt field not found in database; caching empty result"
                    )
                    self._cached_prompt = ""
                    self._cache_timestamp = time.time()
                    self._last_error = None
                    return ""

                else:
                    # Fetch failed - handle error
                    self._load_failures += 1
                    self._last_error = str(fetch_error)
                    logger.error(
                        f"Failed to fetch prompt configuration "
                        f"(total failures: {self._load_failures})"
                    )
                    # Fallback to stale cache or empty string
                    if self._cached_prompt:
                        logger.warning(
                            f"Using stale cache due to fetch failure (age: {self._get_cache_age():.1f}s)"
                        )
                    return self._cached_prompt or ""

            finally:
                # Always clear in-progress flag and notify waiting threads
                self._fetch_in_progress = False
                self._cache_condition.notify_all()  # Wake up all waiting threads

    def _is_cache_valid(self) -> bool:
        """Check if cache is within TTL window."""
        if self._cached_prompt is None or self._cache_timestamp is None:
            return False

        age = time.time() - self._cache_timestamp
        return age < self.cache_ttl_seconds

    def _get_cache_age(self) -> float:
        """Get cache age in seconds."""
        if self._cache_timestamp is None:
            return float("inf")
        return time.time() - self._cache_timestamp

    def _load_from_ruuter_with_retry(self) -> Optional[str]:
        """
        Load configuration from Ruuter with exponential backoff retry.

        Retry strategy:
        - Attempt 1: 0s wait
        - Attempt 2: 1s wait
        - Attempt 3: 2s wait

        Returns:
            Optional[str]: Prompt text if found, None if configuration is empty/not found

        Raises:
            PromptConfigLoadError: If all retry attempts fail due to HTTP/network errors
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    f"Calling Ruuter endpoint "
                    f"(attempt {attempt}/{self.max_retries}): {self.ruuter_endpoint}"
                )

                response = requests.post(
                    self.ruuter_endpoint,
                    json={},  # Empty POST body
                    timeout=self.timeout_seconds,
                    headers={"Content-Type": "application/json"},
                )

                # Check HTTP status
                if response.status_code == 200:
                    data = response.json()

                    # Handle response format - Ruuter wraps response in 'response' key
                    prompt = ""

                    # Unwrap Ruuter's response wrapper if present
                    if isinstance(data, dict) and "response" in data:
                        logger.debug("Unwrapping 'response' key")
                        data = data["response"]

                    # Now extract prompt from the unwrapped data
                    prompt = None  # None means field doesn't exist (not found)

                    if isinstance(data, list) and len(data) > 0:
                        # Array format: [{"id": 1, "prompt": "..."}]
                        first_elem_keys = (
                            list(data[0].keys()) if isinstance(data[0], dict) else []
                        )
                        logger.debug(
                            f"Extracting from list, first element keys: {first_elem_keys}"
                        )
                        # Check if 'prompt' KEY exists (distinguish from missing field)
                        if isinstance(data[0], dict) and "prompt" in data[0]:
                            prompt = data[0].get("prompt", "").strip()
                    elif isinstance(data, dict):
                        # Dict format: {"id": 1, "prompt": "..."}
                        logger.debug(f"Extracting from dict, keys: {list(data.keys())}")
                        # Check if 'prompt' KEY exists (distinguish from missing field)
                        if "prompt" in data:
                            prompt = data.get("prompt", "").strip()
                    else:
                        logger.warning(
                            f"Unexpected data type: {type(data).__name__}, structure not recognized"
                        )

                    # Distinguish between: exists and empty ("") vs doesn't exist (None)
                    if prompt is not None:
                        # Field exists - valid state (even if empty)
                        logger.debug(
                            f"Loaded prompt on attempt {attempt} ({len(prompt)} chars)"
                        )
                        return prompt  # Return actual value (may be empty string)
                    else:
                        # Field doesn't exist - configuration not found
                        logger.warning(
                            f"Prompt field not found or missing (attempt {attempt})"
                        )
                        return None

                else:
                    logger.warning(
                        f"HTTP {response.status_code} on attempt {attempt}: "
                        f"{response.text[:200]}"
                    )

            except requests.exceptions.Timeout:
                logger.warning(
                    f"Request timeout on attempt {attempt} "
                    f"(timeout: {self.timeout_seconds}s)"
                )

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt}: {str(e)[:100]}")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error on attempt {attempt}: {str(e)[:100]}")

            except (ValueError, KeyError) as e:
                logger.error(f"Invalid response format on attempt {attempt}: {e}")

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {e}")

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries:
                wait_time = 2 ** (attempt - 1)  # 1s, 2s
                logger.debug(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        # All retries failed - raise exception to distinguish from "not found"
        error_msg = (
            f"All {self.max_retries} attempts failed to load prompt configuration"
        )
        logger.error(error_msg)
        raise PromptConfigLoadError(error_msg)

    def force_refresh(self) -> bool:
        """
        Force immediate cache refresh.

        Returns:
            bool: True if fresh data was successfully loaded, False otherwise
        """
        status_dict = self.force_refresh_with_status()
        return status_dict["status"] == RefreshStatus.SUCCESS

    def force_refresh_with_status(self) -> Dict[str, Any]:
        """
        Force immediate cache refresh and return detailed status.

        Returns:
            Dict with keys:
                - status: RefreshStatus enum value
                - message: Human-readable message
                - error: Error message (if status != SUCCESS)
        """
        logger.info("Forcing prompt configuration cache refresh")

        # Track state before refresh
        had_cached_value = self._cached_prompt is not None

        with self._cache_condition:
            # Invalidate both timestamp and cached value so that a failed refresh
            # cannot fall back to a stale prompt and be misreported as success.
            self._cache_timestamp = None
            self._cached_prompt = None

        # Attempt fresh load
        prompt_text = None
        fetch_error = None
        try:
            prompt_text = self._load_from_ruuter_with_retry()

        except PromptConfigLoadError as e:
            fetch_error = e
            logger.error(f"Failed to fetch prompt configuration after retries: {e}")

        except Exception as e:
            fetch_error = e
            logger.error(f"Unexpected error loading prompt configuration: {e}")

        # Determine status and update cache
        with self._cache_condition:
            if prompt_text is not None:
                # Success - field exists
                self._cached_prompt = prompt_text
                self._cache_timestamp = time.time()
                self._last_error = None
                logger.info(
                    f"Prompt configuration refreshed successfully ({len(prompt_text)} chars)"
                )
                return {
                    "status": RefreshStatus.SUCCESS,
                    "message": "Prompt configuration refreshed successfully",
                    "length": len(prompt_text),
                }

            elif prompt_text is None and fetch_error is None:
                # No configuration found (prompt field missing in response from Ruuter)
                self._cached_prompt = ""
                self._cache_timestamp = time.time()
                self._last_error = None
                logger.warning("Prompt field not found in database")
                return {
                    "status": RefreshStatus.NOT_FOUND,
                    "message": "Prompt field not found in database",
                    "error": None,
                }

            else:
                # Fetch failed (network/HTTP/timeout errors)
                self._load_failures += 1
                self._last_error = str(fetch_error)
                logger.error(f"Failed to fetch prompt configuration: {fetch_error}")

                # Do NOT cache empty result on failure - let next call retry
                # Only keep stale cache if it existed before
                if had_cached_value:
                    logger.warning("Keeping stale cache due to fetch failure")

                return {
                    "status": RefreshStatus.FETCH_FAILED,
                    "message": "Failed to refresh configuration due to upstream service error",
                    "error": str(fetch_error),
                    "had_stale_cache": had_cached_value,
                }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._cache_condition:
            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "load_failures": self._load_failures,
                "cache_age_seconds": (
                    round(self._get_cache_age(), 2) if self._is_cache_valid() else None
                ),
                "has_cached_value": self._cached_prompt is not None,
                "cache_valid": self._is_cache_valid(),
                "cached_prompt_length": (
                    len(self._cached_prompt) if self._cached_prompt else 0
                ),
                "last_error": self._last_error,
                "ruuter_endpoint": self.ruuter_endpoint,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "fetch_in_progress": self._fetch_in_progress,
            }

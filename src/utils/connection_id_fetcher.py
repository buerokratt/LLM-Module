"""
Connection ID utility for fetching LLM connection IDs by environment.

This module provides functionality to fetch LLM connection IDs for different
environments (production, testing) that can be reused across services.
"""

import asyncio
import threading
from typing import Optional, Dict, Any
from src.loki_logger import LokiLogger
import requests
import aiohttp

from src.llm_orchestrator_config.llm_ochestrator_constants import RAG_SEARCH_RESQL

# Initialize Loki logger
logger = LokiLogger(service_name="connection-id-fetcher")


class ConnectionIdFetcher:
    """
    Service for fetching LLM connection IDs by environment.

    This is a reusable utility that can be used by both budget tracker
    and production store services.
    """

    def __init__(self) -> None:
        """Initialize the connection ID fetcher with endpoints."""
        # Use Resql directly for consistent performance
        self.resql_base = RAG_SEARCH_RESQL
        self.timeout = 5  # seconds

        # Cache connection IDs and vault UUIDs to avoid repeated requests
        self._connection_cache: Dict[str, int | str] = {}
        # Thread-safe lock for cache access
        self._cache_lock = threading.Lock()

    def _extract_connection_id_from_response(
        self, data: dict[str, Any] | list[Any]
    ) -> Optional[int]:
        """
        Extract connection ID from API response data.

        Args:
            data: The JSON response data (dict or list)

        Returns:
            The connection ID as integer, or None if not found
        """
        # Handle different response formats
        if isinstance(data, dict):
            # Check if it's wrapped in response key
            response_data: Any = data.get("response", data)
        else:
            response_data = data

        connection_id: Any = None
        if isinstance(response_data, list):
            # Array format: [{"id": 1, ...}]
            if len(response_data) > 0 and isinstance(response_data[0], dict):
                connection_id = response_data[0].get("id")
        elif isinstance(response_data, dict):
            # Object format: {"id": 1, ...}
            connection_id = response_data.get("id")

        if connection_id is not None:
            try:
                return int(connection_id)
            except (ValueError, TypeError):
                logger.warning(f"Invalid connection ID format: {connection_id}")
                return None

        return None

    def fetch_connection_id_sync(self, environment: str) -> Optional[int]:
        """
        Synchronously fetch the LLM connection ID for specified environment.

        Args:
            environment: The deployment environment ("production" or "testing")

        Returns:
            The connection ID (integer) or None if unavailable
        """
        # Return cached value if available
        cache_key = f"{environment}_connection_id"

        # Thread-safe cache check
        with self._cache_lock:
            if cache_key in self._connection_cache:
                cached_value = int(self._connection_cache[cache_key])
                logger.debug(
                    f"Using cached connection_id for {environment}: {cached_value}"
                )
                return cached_value

        try:
            logger.debug(f"Fetching {environment} connection ID from Resql (sync)...")

            # Use Resql endpoint for getting connection by environment
            endpoint = f"{self.resql_base}/get-{environment}-connection"

            response = requests.post(endpoint, json={}, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                connection_id = self._extract_connection_id_from_response(data)

                if connection_id is not None:
                    # Cache the connection ID (thread-safe)
                    with self._cache_lock:
                        self._connection_cache[cache_key] = connection_id
                    logger.info(
                        f"{environment.capitalize()} connection_id fetched: {connection_id}"
                    )
                    return connection_id
                else:
                    logger.warning(f"No {environment} connection ID found in response")
                    return None
            else:
                logger.error(
                    f"Failed to fetch {environment} connection. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching {environment} connection ID")
            return None

        except Exception as e:
            logger.error(f"Error fetching {environment} connection ID: {str(e)}")
            return None

    async def fetch_connection_id_async(self, environment: str) -> Optional[int]:
        """
        Asynchronously fetch the LLM connection ID for specified environment.

        Args:
            environment: The deployment environment ("production" or "testing")

        Returns:
            The connection ID (integer) or None if unavailable
        """
        # Return cached value if available
        cache_key = f"{environment}_connection_id"

        # Thread-safe cache check
        with self._cache_lock:
            if cache_key in self._connection_cache:
                cached_value = int(self._connection_cache[cache_key])
                logger.debug(
                    f"Using cached connection_id for {environment}: {cached_value}"
                )
                return cached_value

        try:
            logger.debug(f"Fetching {environment} connection ID from Resql (async)...")

            # Use Resql endpoint for getting connection by environment
            endpoint = f"{self.resql_base}/get-{environment}-connection"

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        connection_id = self._extract_connection_id_from_response(data)

                        if connection_id is not None:
                            # Cache the connection ID (thread-safe)
                            with self._cache_lock:
                                self._connection_cache[cache_key] = connection_id
                            logger.info(
                                f"{environment.capitalize()} connection_id fetched: {connection_id}"
                            )
                            return connection_id
                        else:
                            logger.warning(
                                f"No {environment} connection ID found in response"
                            )
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Failed to fetch {environment} connection. "
                            f"Status: {response.status}, Response: {error_text}"
                        )
                        return None

        except asyncio.TimeoutError:
            logger.error(f"Timeout while fetching {environment} connection ID")
            return None
        except aiohttp.ClientError as e:
            logger.error(
                f"Client error while fetching {environment} connection ID: {str(e)}"
            )
            return None
        except Exception as e:
            logger.error(f"Error fetching {environment} connection ID: {str(e)}")
            return None

    def clear_cache(self, environment: Optional[str] = None) -> None:
        """
        Clear the connection ID cache.

        Args:
            environment: Specific environment to clear, or None to clear all
        """
        with self._cache_lock:
            if environment:
                cache_key = f"{environment}_connection_id"
                vault_key = f"{environment}_vault_uuid"
                if cache_key in self._connection_cache:
                    del self._connection_cache[cache_key]
                if vault_key in self._connection_cache:
                    del self._connection_cache[vault_key]
                    logger.debug(f"Cleared cache for {environment}")
            else:
                self._connection_cache.clear()
                logger.debug("Cleared all connection_id cache")

    def fetch_vault_uuid_sync(self, environment: str) -> Optional[str]:
        """
        Synchronously fetch the vault_uuid for specified environment.

        Args:
            environment: The deployment environment ("production" or "testing")

        Returns:
            The vault_uuid (string) or None if unavailable
        """
        cache_key = f"{environment}_vault_uuid"

        with self._cache_lock:
            if cache_key in self._connection_cache:
                cached_value = self._connection_cache[cache_key]
                logger.debug(
                    f"Using cached vault_uuid for {environment}: {cached_value}"
                )
                return str(cached_value)

        try:
            logger.debug(f"Fetching {environment} vault_uuid from Resql (sync)...")

            endpoint = f"{self.resql_base}/get-{environment}-connection"
            response = requests.post(endpoint, json={}, timeout=self.timeout)

            if response.status_code == 200:
                data = response.json()
                vault_uuid = self._extract_vault_uuid_from_response(data)

                if vault_uuid is not None:
                    with self._cache_lock:
                        self._connection_cache[cache_key] = vault_uuid
                    logger.info(
                        f"{environment.capitalize()} vault_uuid fetched: {vault_uuid}"
                    )
                    return vault_uuid
                else:
                    logger.warning(f"No {environment} vault_uuid found in response")
                    return None
            else:
                logger.error(
                    f"Failed to fetch {environment} vault_uuid. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching {environment} vault_uuid")
            return None
        except Exception as e:
            logger.error(f"Error fetching {environment} vault_uuid: {str(e)}")
            return None

    def _extract_vault_uuid_from_response(
        self, data: dict[str, Any] | list[Any]
    ) -> Optional[str]:
        """
        Extract vault_uuid from API response data.

        Args:
            data: The JSON response data (dict or list)

        Returns:
            The vault_uuid as string, or None if not found
        """
        if isinstance(data, dict):
            response_data: Any = data.get("response", data)
        else:
            response_data = data

        if isinstance(response_data, list):
            if len(response_data) > 0 and isinstance(response_data[0], dict):
                vault_uuid = response_data[0].get("vaultUuid")
                return str(vault_uuid) if vault_uuid else None
        elif isinstance(response_data, dict):
            vault_uuid = response_data.get("vaultUuid")
            return str(vault_uuid) if vault_uuid else None

        return None


# Singleton instance for reuse across modules
_connection_id_fetcher: Optional[ConnectionIdFetcher] = None


def get_connection_id_fetcher() -> ConnectionIdFetcher:
    """
    Get the singleton connection ID fetcher instance.

    Returns:
        ConnectionIdFetcher instance
    """
    global _connection_id_fetcher
    if _connection_id_fetcher is None:
        _connection_id_fetcher = ConnectionIdFetcher()
    return _connection_id_fetcher

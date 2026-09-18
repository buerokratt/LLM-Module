"""Redis async connection manager for session store."""

import os
from typing import Any, Optional

import redis.asyncio as aioredis
from src.loki_logger import LokiLogger

logger = LokiLogger(service_name="redis_client")


_redis_client: Optional[aioredis.Redis] = None  # type: ignore[type-arg]


def _is_tls_enabled() -> bool:
    return os.getenv("REDIS_TLS_ENABLED", "false").lower() == "true"


def _build_redis_url() -> str:
    """Build Redis URL from environment variables."""
    host = os.getenv("REDIS_HOST", "redis")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_AUTH", "")
    db = os.getenv("REDIS_SESSION_DB", "1")
    scheme = "rediss" if _is_tls_enabled() else "redis"

    if password:
        return f"{scheme}://:{password}@{host}:{port}/{db}"
    return f"{scheme}://{host}:{port}/{db}"


def _build_tls_kwargs() -> dict[str, Any]:
    """Return SSL keyword arguments for ``from_url()`` when TLS is enabled."""
    if not _is_tls_enabled():
        return {}

    kwargs: dict[str, Any] = {"ssl_cert_reqs": "required"}

    ca = os.getenv("REDIS_TLS_CA")
    if ca:
        kwargs["ssl_ca_certs"] = ca

    cert = os.getenv("REDIS_TLS_CERT")
    if cert:
        kwargs["ssl_certfile"] = cert

    key = os.getenv("REDIS_TLS_KEY")
    if key:
        kwargs["ssl_keyfile"] = key

    return kwargs


async def init_redis_client() -> aioredis.Redis:
    """Initialize the singleton async Redis client.

    Uses db=1 (REDIS_SESSION_DB) to isolate session data from Langfuse (db=0).
    Should be called once during FastAPI lifespan startup.
    """
    global _redis_client

    url = _build_redis_url()
    tls_kwargs = _build_tls_kwargs()
    _redis_client = aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        **tls_kwargs,
    )

    # Verify connectivity
    await _redis_client.ping()
    logger.info(
        f"Redis session store connected (db={os.getenv('REDIS_SESSION_DB', '1')})"
    )
    return _redis_client


async def close_redis_client() -> None:
    """Close the Redis client connection pool gracefully."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
        logger.info("Redis session store connection closed")


def get_redis_client() -> Optional[aioredis.Redis]:
    """Return the initialized Redis client, or None if not initialized."""
    return _redis_client


async def check_redis_health() -> str:
    """Check Redis connectivity for the health endpoint.

    Returns:
        "connected" if Redis is reachable, "disconnected" otherwise.
    """
    client = get_redis_client()
    if client is None:
        return "not_configured"
    try:
        await client.ping()
        return "connected"
    except Exception:
        return "disconnected"

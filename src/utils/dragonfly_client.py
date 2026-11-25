"""Async connection manager for the Dragonfly session store.

Settings come from the DRAGONFLY_* environment variables. Dragonfly speaks the
Redis wire protocol, so the connection uses the redis-py client library; this is
the only app module that imports it. The stores get the client and WatchError
from here.
"""

import os
from typing import Any, Optional

from redis import WatchError as WatchError  # re-exported for the stores
from redis.asyncio import Redis as DragonflyClient
from src.loki_logger import LokiLogger

logger = LokiLogger(service_name="dragonfly_client")


_dragonfly_client: Optional[DragonflyClient] = None  # type: ignore[type-arg]


def _is_tls_enabled() -> bool:
    return os.getenv("DRAGONFLY_TLS_ENABLED", "false").lower() == "true"


def _build_server_kwargs() -> dict[str, Any]:
    """Host, port, password and db for the client, from the DRAGONFLY_* variables.

    Passed as separate arguments rather than a connection URL, so a password
    containing characters such as ``/``, ``#`` or ``%`` is used as it is.
    Empty values fall back to the defaults.
    """
    return {
        "host": os.getenv("DRAGONFLY_HOST") or "dragonfly",
        "port": int(os.getenv("DRAGONFLY_PORT") or "6379"),
        "password": os.getenv("DRAGONFLY_PASSWORD") or None,
        "db": int(os.getenv("DRAGONFLY_SESSION_DB") or "1"),
    }


def _build_tls_kwargs() -> dict[str, Any]:
    """Return the TLS arguments for the client when TLS is enabled."""
    if not _is_tls_enabled():
        return {}

    kwargs: dict[str, Any] = {"ssl": True, "ssl_cert_reqs": "required"}

    ca = os.getenv("DRAGONFLY_TLS_CA")
    if ca:
        kwargs["ssl_ca_certs"] = ca

    cert = os.getenv("DRAGONFLY_TLS_CERT")
    if cert:
        kwargs["ssl_certfile"] = cert

    key = os.getenv("DRAGONFLY_TLS_KEY")
    if key:
        kwargs["ssl_keyfile"] = key

    return kwargs


def dragonfly_key(key: str) -> str:
    """Prefix a store key with DRAGONFLY_KEY_PREFIX (empty by default).

    Used when the app shares db 0 with Langfuse (Dragonfly cluster mode),
    so app keys stay separate from Langfuse's keys.
    """
    return f"{os.getenv('DRAGONFLY_KEY_PREFIX', '')}{key}"


_DEFAULT_CONNECT_TIMEOUT = 2.0
_DEFAULT_SOCKET_TIMEOUT = 3.0
_HEALTH_CHECK_INTERVAL = 30
_DEFAULT_MAX_CONNECTIONS = 100


def _env_number(name: str, default: float) -> float:
    """Read a numeric setting from the environment, falling back to the default."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}; using default {default}")
        return default


def _build_connection_kwargs() -> dict[str, Any]:
    """Timeouts, retry and pool settings for the client.

    Short timeouts keep chat requests fast when the store hangs:
    every store catches the error and fails open.
    """
    return {
        "socket_connect_timeout": _env_number(
            "DRAGONFLY_CONNECT_TIMEOUT", _DEFAULT_CONNECT_TIMEOUT
        ),
        "socket_timeout": _env_number(
            "DRAGONFLY_SOCKET_TIMEOUT", _DEFAULT_SOCKET_TIMEOUT
        ),
        "health_check_interval": _HEALTH_CHECK_INTERVAL,
        "retry_on_timeout": True,
        "max_connections": int(
            _env_number("DRAGONFLY_MAX_CONNECTIONS", _DEFAULT_MAX_CONNECTIONS)
        ),
    }


async def init_dragonfly_client() -> DragonflyClient:
    """Initialize the singleton async Dragonfly client.

    Uses db=1 (DRAGONFLY_SESSION_DB) to isolate session data from Langfuse (db=0).
    Should be called once during FastAPI lifespan startup.
    """
    global _dragonfly_client

    _dragonfly_client = DragonflyClient(
        **_build_server_kwargs(),
        encoding="utf-8",
        decode_responses=True,
        **_build_connection_kwargs(),
        **_build_tls_kwargs(),
    )

    # Verify connectivity
    await _dragonfly_client.ping()
    logger.info(f"Dragonfly connected (db={os.getenv('DRAGONFLY_SESSION_DB') or '1'})")
    return _dragonfly_client


async def close_dragonfly_client() -> None:
    """Close the Dragonfly client connection pool gracefully."""
    global _dragonfly_client

    if _dragonfly_client is not None:
        await _dragonfly_client.aclose()
        _dragonfly_client = None
        logger.info("Dragonfly connection closed")


def get_dragonfly_client() -> Optional[DragonflyClient]:
    """Return the initialized Dragonfly client, or None if not initialized."""
    return _dragonfly_client


async def check_dragonfly_health() -> str:
    """Check Dragonfly connectivity for the health endpoint.

    Returns:
        "connected" if Dragonfly is reachable, "disconnected" otherwise.
    """
    client = get_dragonfly_client()
    if client is None:
        return "not_configured"
    try:
        await client.ping()
        return "connected"
    except Exception:
        return "disconnected"

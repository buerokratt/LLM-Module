"""Unit tests for src/utils/redis_client.py: URL, TLS, connection settings, health."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils import redis_client

_ENV_VARS = (
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_AUTH",
    "REDIS_SESSION_DB",
    "REDIS_TLS_ENABLED",
    "REDIS_TLS_CA",
    "REDIS_TLS_CERT",
    "REDIS_TLS_KEY",
    "REDIS_CONNECT_TIMEOUT",
    "REDIS_SOCKET_TIMEOUT",
    "REDIS_MAX_CONNECTIONS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test with none of the Redis settings set."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _fake_client(ping: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.ping = ping
    return client


class TestBuildRedisUrl:
    def test_without_password(self):
        assert redis_client._build_redis_url() == "redis://redis:6379/1"

    def test_with_password(self, monkeypatch):
        monkeypatch.setenv("REDIS_AUTH", "secret")
        assert redis_client._build_redis_url() == "redis://:secret@redis:6379/1"

    def test_tls_uses_rediss_scheme(self, monkeypatch):
        monkeypatch.setenv("REDIS_TLS_ENABLED", "true")
        assert redis_client._build_redis_url().startswith("rediss://")

    def test_empty_session_db_falls_back_to_db_1(self, monkeypatch):
        monkeypatch.setenv("REDIS_SESSION_DB", "")
        assert redis_client._build_redis_url().endswith("/1")

    def test_explicit_session_db(self, monkeypatch):
        monkeypatch.setenv("REDIS_SESSION_DB", "2")
        assert redis_client._build_redis_url().endswith("/2")


class TestBuildTlsKwargs:
    def test_disabled_returns_empty(self):
        assert redis_client._build_tls_kwargs() == {}

    def test_enabled_with_cert_files(self, monkeypatch):
        monkeypatch.setenv("REDIS_TLS_ENABLED", "true")
        monkeypatch.setenv("REDIS_TLS_CA", "/certs/ca.crt")
        monkeypatch.setenv("REDIS_TLS_CERT", "/certs/redis.crt")
        monkeypatch.setenv("REDIS_TLS_KEY", "/certs/redis.key")
        assert redis_client._build_tls_kwargs() == {
            "ssl_cert_reqs": "required",
            "ssl_ca_certs": "/certs/ca.crt",
            "ssl_certfile": "/certs/redis.crt",
            "ssl_keyfile": "/certs/redis.key",
        }


class TestBuildConnectionKwargs:
    def test_defaults(self):
        assert redis_client._build_connection_kwargs() == {
            "socket_connect_timeout": 2.0,
            "socket_timeout": 3.0,
            "health_check_interval": 30,
            "retry_on_timeout": True,
            "max_connections": 100,
        }

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("REDIS_CONNECT_TIMEOUT", "5")
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "0.5")
        monkeypatch.setenv("REDIS_MAX_CONNECTIONS", "20")
        kwargs = redis_client._build_connection_kwargs()
        assert kwargs["socket_connect_timeout"] == 5.0
        assert kwargs["socket_timeout"] == 0.5
        assert kwargs["max_connections"] == 20

    def test_invalid_value_uses_default(self, monkeypatch):
        monkeypatch.setenv("REDIS_SOCKET_TIMEOUT", "abc")
        assert redis_client._build_connection_kwargs()["socket_timeout"] == 3.0


class TestInitRedisClient:
    @pytest.mark.asyncio
    async def test_passes_connection_settings_to_from_url(self, monkeypatch):
        client = _fake_client(AsyncMock(return_value=True))
        monkeypatch.setattr(redis_client, "_redis_client", None)
        with patch.object(
            redis_client.aioredis, "from_url", return_value=client
        ) as from_url:
            await redis_client.init_redis_client()

        kwargs = from_url.call_args.kwargs
        assert kwargs["socket_connect_timeout"] == 2.0
        assert kwargs["socket_timeout"] == 3.0
        assert kwargs["retry_on_timeout"] is True
        assert kwargs["max_connections"] == 100
        assert redis_client.get_redis_client() is client


class TestCheckRedisHealth:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.setattr(redis_client, "_redis_client", None)
        assert await redis_client.check_redis_health() == "not_configured"

    @pytest.mark.asyncio
    async def test_connected(self, monkeypatch):
        client = _fake_client(AsyncMock(return_value=True))
        monkeypatch.setattr(redis_client, "_redis_client", client)
        assert await redis_client.check_redis_health() == "connected"

    @pytest.mark.asyncio
    async def test_disconnected(self, monkeypatch):
        client = _fake_client(AsyncMock(side_effect=ConnectionError("down")))
        monkeypatch.setattr(redis_client, "_redis_client", client)
        assert await redis_client.check_redis_health() == "disconnected"

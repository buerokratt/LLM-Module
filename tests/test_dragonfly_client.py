"""Unit tests for src/utils/dragonfly_client.py: server, TLS and connection settings, health."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils import dragonfly_client

_ENV_VARS = (
    "DRAGONFLY_HOST",
    "DRAGONFLY_PORT",
    "DRAGONFLY_PASSWORD",
    "DRAGONFLY_SESSION_DB",
    "DRAGONFLY_TLS_ENABLED",
    "DRAGONFLY_TLS_CA",
    "DRAGONFLY_TLS_CERT",
    "DRAGONFLY_TLS_KEY",
    "DRAGONFLY_CONNECT_TIMEOUT",
    "DRAGONFLY_SOCKET_TIMEOUT",
    "DRAGONFLY_MAX_CONNECTIONS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start every test with none of the Dragonfly settings set."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _fake_client(ping: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.ping = ping
    return client


class TestBuildServerKwargs:
    def test_defaults(self):
        assert dragonfly_client._build_server_kwargs() == {
            "host": "dragonfly",
            "port": 6379,
            "password": None,
            "db": 1,
        }

    def test_env_values(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_HOST", "df.internal")
        monkeypatch.setenv("DRAGONFLY_PORT", "6380")
        monkeypatch.setenv("DRAGONFLY_PASSWORD", "secret")
        monkeypatch.setenv("DRAGONFLY_SESSION_DB", "2")
        assert dragonfly_client._build_server_kwargs() == {
            "host": "df.internal",
            "port": 6380,
            "password": "secret",
            "db": 2,
        }

    def test_password_with_special_characters_is_kept(self, monkeypatch):
        # These characters broke the old connection URL
        monkeypatch.setenv("DRAGONFLY_PASSWORD", "ab/cd#ef%41?g@h")
        assert dragonfly_client._build_server_kwargs()["password"] == (
            "ab/cd#ef%41?g@h"
        )

    def test_empty_values_fall_back_to_defaults(self, monkeypatch):
        # docker compose passes unset .env values through as empty strings
        for name in (
            "DRAGONFLY_HOST",
            "DRAGONFLY_PORT",
            "DRAGONFLY_PASSWORD",
            "DRAGONFLY_SESSION_DB",
        ):
            monkeypatch.setenv(name, "")
        assert dragonfly_client._build_server_kwargs() == {
            "host": "dragonfly",
            "port": 6379,
            "password": None,
            "db": 1,
        }


class TestBuildTlsKwargs:
    def test_disabled_returns_empty(self):
        assert dragonfly_client._build_tls_kwargs() == {}

    def test_enabled_with_cert_files(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_TLS_ENABLED", "true")
        monkeypatch.setenv("DRAGONFLY_TLS_CA", "/certs/ca.crt")
        monkeypatch.setenv("DRAGONFLY_TLS_CERT", "/certs/dragonfly.crt")
        monkeypatch.setenv("DRAGONFLY_TLS_KEY", "/certs/dragonfly.key")
        assert dragonfly_client._build_tls_kwargs() == {
            "ssl": True,
            "ssl_cert_reqs": "required",
            "ssl_ca_certs": "/certs/ca.crt",
            "ssl_certfile": "/certs/dragonfly.crt",
            "ssl_keyfile": "/certs/dragonfly.key",
        }


class TestBuildConnectionKwargs:
    def test_defaults(self):
        assert dragonfly_client._build_connection_kwargs() == {
            "socket_connect_timeout": 2.0,
            "socket_timeout": 3.0,
            "health_check_interval": 30,
            "retry_on_timeout": True,
            "max_connections": 100,
        }

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_CONNECT_TIMEOUT", "5")
        monkeypatch.setenv("DRAGONFLY_SOCKET_TIMEOUT", "0.5")
        monkeypatch.setenv("DRAGONFLY_MAX_CONNECTIONS", "20")
        kwargs = dragonfly_client._build_connection_kwargs()
        assert kwargs["socket_connect_timeout"] == 5.0
        assert kwargs["socket_timeout"] == 0.5
        assert kwargs["max_connections"] == 20

    def test_invalid_value_uses_default(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_SOCKET_TIMEOUT", "abc")
        assert dragonfly_client._build_connection_kwargs()["socket_timeout"] == 3.0


class TestInitDragonflyClient:
    @pytest.mark.asyncio
    async def test_passes_settings_to_client(self, monkeypatch):
        client = _fake_client(AsyncMock(return_value=True))
        monkeypatch.setattr(dragonfly_client, "_dragonfly_client", None)
        with patch.object(
            dragonfly_client, "DragonflyClient", return_value=client
        ) as client_class:
            await dragonfly_client.init_dragonfly_client()

        kwargs = client_class.call_args.kwargs
        assert kwargs["host"] == "dragonfly"
        assert kwargs["port"] == 6379
        assert kwargs["db"] == 1
        assert kwargs["decode_responses"] is True
        assert kwargs["socket_connect_timeout"] == 2.0
        assert kwargs["socket_timeout"] == 3.0
        assert kwargs["retry_on_timeout"] is True
        assert kwargs["max_connections"] == 100
        assert dragonfly_client.get_dragonfly_client() is client


class TestRealClientConfiguration:
    """Build a real client (it connects lazily, so no server is needed)."""

    @pytest.mark.asyncio
    async def test_password_reaches_the_connection_unchanged(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_PASSWORD", "ab/cd#ef%41")
        client = dragonfly_client.DragonflyClient(
            **dragonfly_client._build_server_kwargs()
        )
        try:
            settings = client.connection_pool.connection_kwargs
            assert settings["password"] == "ab/cd#ef%41"
            assert settings["host"] == "dragonfly"
            assert settings["db"] == 1
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_tls_enabled_uses_ssl_connection(self, monkeypatch):
        monkeypatch.setenv("DRAGONFLY_TLS_ENABLED", "true")
        client = dragonfly_client.DragonflyClient(
            **dragonfly_client._build_server_kwargs(),
            **dragonfly_client._build_tls_kwargs(),
        )
        try:
            assert client.connection_pool.connection_class.__name__ == ("SSLConnection")
        finally:
            await client.aclose()


class TestCheckDragonflyHealth:
    @pytest.mark.asyncio
    async def test_not_configured(self, monkeypatch):
        monkeypatch.setattr(dragonfly_client, "_dragonfly_client", None)
        assert await dragonfly_client.check_dragonfly_health() == "not_configured"

    @pytest.mark.asyncio
    async def test_connected(self, monkeypatch):
        client = _fake_client(AsyncMock(return_value=True))
        monkeypatch.setattr(dragonfly_client, "_dragonfly_client", client)
        assert await dragonfly_client.check_dragonfly_health() == "connected"

    @pytest.mark.asyncio
    async def test_disconnected(self, monkeypatch):
        client = _fake_client(AsyncMock(side_effect=ConnectionError("down")))
        monkeypatch.setattr(dragonfly_client, "_dragonfly_client", client)
        assert await dragonfly_client.check_dragonfly_health() == "disconnected"

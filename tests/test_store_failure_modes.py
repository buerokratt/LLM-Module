"""Fail-open tests: every store degrades gracefully when Dragonfly fails.

Each store method must return its safe fallback (``None``, ``[]``, ``False`` or a
no-op) instead of raising, so a chat request keeps working without the store.
Covers a timeout, a lost connection, and a full store rejecting writes (OOM).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

# Load the tool_classifier package before utils.atc_cache_store: the two import
# each other, and this order avoids the circular import (as test_atc_cache_store.py does).
import tool_classifier  # noqa: F401

import src.utils.conversation_history_store as history_module
import utils.api_tool_session_store as session_module
import utils.atc_cache_store as atc_module
from models.session_models import APIToolSession, LastCallContext
from src.models.conversation_history_models import ConversationRound

CHAT_ID = "chat-failure-test"

STORE_ERRORS = [
    pytest.param(TimeoutError("Timeout reading from socket"), id="timeout"),
    pytest.param(ConnectionError("Connection refused"), id="connection"),
    pytest.param(
        ResponseError("OOM command not allowed when used memory > 'maxmemory'"),
        id="oom",
    ),
]


def _failing_client(error: Exception) -> MagicMock:
    """A client whose every command (direct or in a pipeline) raises ``error``."""
    client = MagicMock()
    for command in ("get", "set", "delete", "exists", "expire"):
        setattr(client, command, AsyncMock(side_effect=error))

    pipe = MagicMock()
    pipe.watch = AsyncMock(side_effect=error)
    pipe.get = AsyncMock(side_effect=error)
    pipe.unwatch = AsyncMock()
    pipe.execute = AsyncMock(side_effect=error)

    pipeline_cm = MagicMock()
    pipeline_cm.__aenter__.return_value = pipe
    pipeline_cm.__aexit__.return_value = False
    client.pipeline = MagicMock(return_value=pipeline_cm)
    return client


def _make_session() -> APIToolSession:
    return APIToolSession(
        chat_id=CHAT_ID,
        state="collecting_params",
        selected_endpoint={"url": "https://example.com", "method": "GET"},
        collected_params={"countryIsoCode": "EE"},
        turn_count=1,
        max_turns=5,
    )


def _make_last_call() -> LastCallContext:
    return LastCallContext(
        api_name="get_national_holidays",
        endpoint={"name": "get_national_holidays", "url": "https://example.com"},
        collected_params={"country": "EE"},
        raw_response={"holidays": []},
        original_query="Public holidays in Estonia?",
        timestamp=1748000000.0,
    )


@pytest.mark.parametrize("error", STORE_ERRORS)
class TestSessionStoreFailOpen:
    @pytest.mark.asyncio
    async def test_get_returns_none(self, error):
        with patch.object(
            session_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            assert await session_module.APIToolSessionStore().get(CHAT_ID) is None

    @pytest.mark.asyncio
    async def test_save_does_not_raise(self, error):
        with patch.object(
            session_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await session_module.APIToolSessionStore().save(_make_session())

    @pytest.mark.asyncio
    async def test_update_returns_none(self, error):
        with patch.object(
            session_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            store = session_module.APIToolSessionStore()
            assert await store.update(CHAT_ID, turn_count=2) is None

    @pytest.mark.asyncio
    async def test_delete_does_not_raise(self, error):
        with patch.object(
            session_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await session_module.APIToolSessionStore().delete(CHAT_ID)

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, error):
        with patch.object(
            session_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            assert await session_module.APIToolSessionStore().exists(CHAT_ID) is False


@pytest.mark.parametrize("error", STORE_ERRORS)
class TestConversationHistoryStoreFailOpen:
    @pytest.mark.asyncio
    async def test_get_history_returns_empty(self, error):
        with patch.object(
            history_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            store = history_module.ConversationHistoryStore()
            assert await store.get_history(CHAT_ID) == []

    @pytest.mark.asyncio
    async def test_get_summary_returns_none(self, error):
        with patch.object(
            history_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            store = history_module.ConversationHistoryStore()
            assert await store.get_summary(CHAT_ID) is None

    @pytest.mark.asyncio
    async def test_save_round_does_not_raise(self, error):
        with patch.object(
            history_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            round_ = ConversationRound(user_message="Hello", bot_message="Hi there!")
            await history_module.ConversationHistoryStore().save_round(CHAT_ID, round_)

    @pytest.mark.asyncio
    async def test_save_summary_does_not_raise(self, error):
        with patch.object(
            history_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await history_module.ConversationHistoryStore().save_summary(
                CHAT_ID, "summary"
            )


@pytest.mark.parametrize("error", STORE_ERRORS)
class TestATCCacheStoreFailOpen:
    @pytest.mark.asyncio
    async def test_get_l1_is_a_cache_miss(self, error):
        with patch.object(
            atc_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            store = atc_module.ATCCacheStore()
            assert await store.get_l1(CHAT_ID, "api", {"a": 1}) is None

    @pytest.mark.asyncio
    async def test_set_l1_does_not_raise(self, error):
        with patch.object(
            atc_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await atc_module.ATCCacheStore().set_l1(
                CHAT_ID, "api", {"a": 1}, {"ok": True}
            )

    @pytest.mark.asyncio
    async def test_get_l2_returns_none(self, error):
        with patch.object(
            atc_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            assert await atc_module.ATCCacheStore().get_l2(CHAT_ID) is None

    @pytest.mark.asyncio
    async def test_set_l2_does_not_raise(self, error):
        with patch.object(
            atc_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await atc_module.ATCCacheStore().set_l2(CHAT_ID, [_make_last_call()])

    @pytest.mark.asyncio
    async def test_invalidate_l2_does_not_raise(self, error):
        with patch.object(
            atc_module, "get_dragonfly_client", return_value=_failing_client(error)
        ):
            await atc_module.ATCCacheStore().invalidate_l2(CHAT_ID)

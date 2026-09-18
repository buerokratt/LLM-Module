"""Unit tests for src.utils.conversation_history_helpers.get_conversation_history."""

from unittest.mock import AsyncMock

import pytest

from src.models.conversation_history_models import (
    ConversationHistoryState,
    ConversationRound,
)
from models.request_models import ConversationItem
from src.utils.conversation_history_helpers import get_conversation_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(role: str = "user", msg: str = "hello") -> ConversationItem:
    return ConversationItem(
        authorRole=role, message=msg, timestamp="2024-01-01T00:00:00"
    )  # type: ignore[arg-type]


def _make_round(
    user: str = "What is the rate?",
    bot: str = "The rate is 20%.",
    ts: float = 1_700_000_000.0,
) -> ConversationRound:
    return ConversationRound(user_message=user, bot_message=bot, timestamp=ts)


# ---------------------------------------------------------------------------
# Redis available with rounds
# ---------------------------------------------------------------------------


class TestGetConversationHistoryRedisRounds:
    @pytest.mark.asyncio
    async def test_returns_redis_rounds_as_conversation_items(self) -> None:
        """Two ConversationItems (user + bot) per stored round."""
        round_ = _make_round()
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        fallback = [_make_item()]
        history, summary = await get_conversation_history("c1", store, fallback)

        assert len(history) == 2
        assert history[0].authorRole == "user"
        assert history[0].message == round_.user_message
        assert history[1].authorRole == "bot"
        assert history[1].message == round_.bot_message
        assert summary is None

    @pytest.mark.asyncio
    async def test_ignores_fallback_when_redis_has_rounds(self) -> None:
        """Fallback list is not returned when Redis has valid rounds."""
        round_ = _make_round()
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        fallback = [_make_item(msg="fallback-only message")]
        history, _ = await get_conversation_history("c1", store, fallback)

        messages = [item.message for item in history]
        assert "fallback-only message" not in messages

    @pytest.mark.asyncio
    async def test_returns_summary_alongside_rounds(self) -> None:
        """Redis summary is returned as the second tuple element."""
        round_ = _make_round()
        state = ConversationHistoryState(
            chat_id="c1",
            rounds=[round_],
            summary="Earlier we discussed tax rates.",
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        _, summary = await get_conversation_history("c1", store, [])

        assert summary == "Earlier we discussed tax rates."

    @pytest.mark.asyncio
    async def test_multiple_rounds_expand_to_all_items(self) -> None:
        """N rounds → 2*N ConversationItems in order."""
        rounds = [
            _make_round(user="q1", bot="a1"),
            _make_round(user="q2", bot="a2"),
            _make_round(user="q3", bot="a3"),
        ]
        state = ConversationHistoryState(chat_id="c1", rounds=rounds, summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        history, _ = await get_conversation_history("c1", store, [])

        assert len(history) == 6
        assert history[0].message == "q1"
        assert history[1].message == "a1"
        assert history[4].message == "q3"
        assert history[5].message == "a3"

    @pytest.mark.asyncio
    async def test_timestamp_is_str_of_round_timestamp(self) -> None:
        """Timestamp on returned items is the string form of the round's float timestamp."""
        round_ = _make_round(ts=1_700_000_123.456)
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        history, _ = await get_conversation_history("c1", store, [])

        assert history[0].timestamp == str(round_.timestamp)
        assert history[1].timestamp == str(round_.timestamp)


# ---------------------------------------------------------------------------
# Redis empty → fallback
# ---------------------------------------------------------------------------


class TestGetConversationHistoryRedisEmpty:
    @pytest.mark.asyncio
    async def test_falls_back_when_redis_returns_no_rounds(self) -> None:
        """Empty rounds list → fallback returned with summary=None."""
        state = ConversationHistoryState(
            chat_id="c1", rounds=[], summary="stale summary"
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        fallback = [_make_item(msg="from request")]
        history, summary = await get_conversation_history("c1", store, fallback)

        assert len(history) == 1
        assert history[0].message == "from request"
        assert summary is None  # summary not returned when rounds are empty


# ---------------------------------------------------------------------------
# Redis unavailable → graceful degradation
# ---------------------------------------------------------------------------


class TestGetConversationHistoryRedisUnavailable:
    @pytest.mark.asyncio
    async def test_falls_back_when_get_context_raises(self) -> None:
        """Any exception from get_context → fallback with summary=None, no propagation."""
        store = AsyncMock()
        store.get_context = AsyncMock(side_effect=RuntimeError("Redis down"))

        fallback = [_make_item(msg="fallback on error")]
        history, summary = await get_conversation_history("c1", store, fallback)

        assert len(history) == 1
        assert history[0].message == "fallback on error"
        assert summary is None

    @pytest.mark.asyncio
    async def test_falls_back_when_store_is_none(self) -> None:
        """When store=None, fallback is returned immediately without calling Redis."""
        fallback = [_make_item(role="bot", msg="bot message")]
        history, summary = await get_conversation_history("c1", None, fallback)

        assert len(history) == 1
        assert history[0].authorRole == "bot"
        assert summary is None

    @pytest.mark.asyncio
    async def test_does_not_raise_on_connection_error(self) -> None:
        """ConnectionError from Redis is caught and does not propagate."""
        store = AsyncMock()
        store.get_context = AsyncMock(side_effect=ConnectionError("refused"))

        history, summary = await get_conversation_history("c1", store, [])

        assert history == []
        assert summary is None

    @pytest.mark.asyncio
    async def test_get_context_called_with_correct_chat_id(self) -> None:
        """The helper passes chat_id to store.get_context."""
        state = ConversationHistoryState(chat_id="my-chat", rounds=[], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        await get_conversation_history("my-chat", store, [])

        store.get_context.assert_awaited_once_with("my-chat")


# ---------------------------------------------------------------------------
# Conversion correctness
# ---------------------------------------------------------------------------


class TestGetConversationHistoryConversionCorrectness:
    @pytest.mark.asyncio
    async def test_returned_items_are_conversation_item_instances(self) -> None:
        """All returned history items must be ConversationItem Pydantic objects."""
        round_ = _make_round()
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        history, _ = await get_conversation_history("c1", store, [])

        for item in history:
            assert isinstance(item, ConversationItem)

    @pytest.mark.asyncio
    async def test_user_role_is_literal_user(self) -> None:
        """User item authorRole must be the literal string 'user'."""
        round_ = _make_round()
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        history, _ = await get_conversation_history("c1", store, [])

        assert history[0].authorRole == "user"

    @pytest.mark.asyncio
    async def test_bot_role_is_literal_bot(self) -> None:
        """Bot item authorRole must be the literal string 'bot'."""
        round_ = _make_round()
        state = ConversationHistoryState(chat_id="c1", rounds=[round_], summary=None)
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        history, _ = await get_conversation_history("c1", store, [])

        assert history[1].authorRole == "bot"

    @pytest.mark.asyncio
    async def test_fallback_items_returned_unchanged(self) -> None:
        """Fallback items are returned as-is (same objects, same content)."""
        fallback = [
            _make_item(role="user", msg="user says"),
            _make_item(role="bot", msg="bot replies"),
        ]
        history, _ = await get_conversation_history("c1", None, fallback)

        assert history is fallback

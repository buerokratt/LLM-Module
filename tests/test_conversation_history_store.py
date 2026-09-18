"""Unit tests for ConversationHistoryStore and conversation history models."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from redis import WatchError

from src.models.conversation_history_models import (
    ConversationHistoryState,
    ConversationRound,
)
from src.utils.conversation_history_store import (
    ConversationHistoryStore,
    _HISTORY_TTL_SECONDS,
    _MAX_ROUNDS,
    _history_key,
    _summary_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_round(**kwargs) -> ConversationRound:
    defaults = {
        "user_message": "Hello",
        "bot_message": "Hi there!",
    }
    defaults.update(kwargs)
    return ConversationRound(**defaults)


def _make_redis_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.expire = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    return mock


def _make_pipe_mock() -> AsyncMock:
    pipe = AsyncMock()
    pipe.watch = AsyncMock()
    pipe.unwatch = AsyncMock()
    pipe.get = AsyncMock(return_value=None)
    pipe.multi = MagicMock()
    pipe.set = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=[True, True])
    pipe.__aenter__ = AsyncMock(return_value=pipe)
    pipe.__aexit__ = AsyncMock(return_value=False)
    return pipe


# ---------------------------------------------------------------------------
# ConversationRound model tests
# ---------------------------------------------------------------------------


class TestConversationRound:
    def test_required_fields(self):
        r = _make_round()
        assert r.user_message == "Hello"
        assert r.bot_message == "Hi there!"
        assert isinstance(r.timestamp, float)

    def test_timestamp_defaults_to_current_time(self):
        before = time.time()
        r = _make_round()
        after = time.time()
        assert before <= r.timestamp <= after

    def test_explicit_timestamp(self):
        r = ConversationRound(user_message="u", bot_message="b", timestamp=12345.0)
        assert r.timestamp == 12345.0

    def test_user_message_required(self):
        with pytest.raises(ValidationError):
            ConversationRound(bot_message="b")  # type: ignore[call-arg]

    def test_bot_message_required(self):
        with pytest.raises(ValidationError):
            ConversationRound(user_message="u")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self):
        r = _make_round(user_message="What is the weather?", bot_message="It is sunny.")
        restored = ConversationRound.model_validate_json(r.model_dump_json())
        assert restored == r


# ---------------------------------------------------------------------------
# ConversationHistoryState model tests
# ---------------------------------------------------------------------------


class TestConversationHistoryState:
    def test_defaults(self):
        state = ConversationHistoryState(chat_id="chat-1")
        assert state.rounds == []
        assert state.summary is None

    def test_chat_id_required(self):
        with pytest.raises(ValidationError):
            ConversationHistoryState()  # type: ignore[call-arg]

    def test_serialization_roundtrip(self):
        state = ConversationHistoryState(
            chat_id="chat-2",
            rounds=[_make_round(), _make_round(user_message="q2", bot_message="a2")],
            summary="User asked about weather and holidays.",
        )
        restored = ConversationHistoryState.model_validate_json(state.model_dump_json())
        assert restored == state
        assert len(restored.rounds) == 2
        assert restored.summary == "User asked about weather and holidays."


# ---------------------------------------------------------------------------
# Key helper tests
# ---------------------------------------------------------------------------


class TestKeyHelpers:
    def test_history_key(self):
        assert _history_key("abc") == "conv:abc"

    def test_summary_key(self):
        assert _summary_key("abc") == "conv:summary:abc"


# ---------------------------------------------------------------------------
# ConversationHistoryStore.save_round
# ---------------------------------------------------------------------------


class TestSaveRound:
    @pytest.mark.asyncio
    async def test_save_round_appends_and_sets_ttl(self):
        store = ConversationHistoryStore()
        round_ = _make_round()

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=None)  # no existing rounds

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-1", round_)

        pipe.watch.assert_awaited_once_with(_history_key("chat-1"))
        pipe.multi.assert_called_once()
        pipe.set.assert_called_once()
        set_call = pipe.set.call_args
        assert set_call[0][0] == _history_key("chat-1")
        assert set_call[1]["ex"] == _HISTORY_TTL_SECONDS
        # Verify stored JSON contains the round
        stored = json.loads(set_call[0][1])
        assert len(stored) == 1
        assert stored[0]["user_message"] == round_.user_message

    @pytest.mark.asyncio
    async def test_save_round_appends_to_existing_rounds(self):
        store = ConversationHistoryStore()
        existing = [
            _make_round(user_message=f"q{i}", bot_message=f"a{i}").model_dump()
            for i in range(3)
        ]
        new_round = _make_round(user_message="q_new", bot_message="a_new")

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=json.dumps(existing))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-2", new_round)

        stored_json = pipe.set.call_args[0][1]
        stored = json.loads(stored_json)
        assert len(stored) == 4
        assert stored[-1]["user_message"] == "q_new"

    @pytest.mark.asyncio
    async def test_save_round_trims_to_max_rounds(self):
        store = ConversationHistoryStore()
        # Start with exactly MAX_ROUNDS rounds
        existing = [
            _make_round(user_message=f"q{i}", bot_message=f"a{i}").model_dump()
            for i in range(_MAX_ROUNDS)
        ]
        new_round = _make_round(user_message="q_overflow", bot_message="a_overflow")

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=json.dumps(existing))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-3", new_round)

        stored_json = pipe.set.call_args[0][1]
        stored = json.loads(stored_json)
        assert len(stored) == _MAX_ROUNDS
        # Oldest round was evicted; newest is last
        assert stored[-1]["user_message"] == "q_overflow"
        assert stored[0]["user_message"] == "q1"

    @pytest.mark.asyncio
    async def test_save_round_resets_summary_ttl(self):
        store = ConversationHistoryStore()
        round_ = _make_round()

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=None)

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-4", round_)

        pipe.expire.assert_called_once_with(
            _summary_key("chat-4"), _HISTORY_TTL_SECONDS
        )

    @pytest.mark.asyncio
    async def test_save_round_skips_when_redis_unavailable(self):
        store = ConversationHistoryStore()
        with patch(
            "src.utils.conversation_history_store.get_redis_client", return_value=None
        ):
            # Should not raise
            await store.save_round("chat-x", _make_round())

    @pytest.mark.asyncio
    async def test_save_round_retries_on_watch_error(self):
        store = ConversationHistoryStore()
        round_ = _make_round()

        call_count = 0

        async def fake_execute():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise WatchError("conflict")
            return [True, True]

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=None)
        pipe.execute = fake_execute

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-5", round_)

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_save_round_exhausts_retries_gracefully(self):
        store = ConversationHistoryStore()
        round_ = _make_round()

        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=None)
        pipe.execute = AsyncMock(side_effect=WatchError("always conflicts"))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            # Should not raise after exhausting retries
            await store.save_round("chat-6", round_)

    @pytest.mark.asyncio
    async def test_save_round_graceful_on_unexpected_error(self):
        store = ConversationHistoryStore()

        pipe = _make_pipe_mock()
        pipe.watch = AsyncMock(side_effect=RuntimeError("boom"))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-7", _make_round())


# ---------------------------------------------------------------------------
# ConversationHistoryStore.get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_key_missing(self):
        store = ConversationHistoryStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=None)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_history("missing-chat")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_deserialized_rounds(self):
        store = ConversationHistoryStore()
        rounds = [
            _make_round(user_message="q1", bot_message="a1"),
            _make_round(user_message="q2", bot_message="a2"),
        ]
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(
            return_value=json.dumps([r.model_dump() for r in rounds])
        )

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_history("chat-1")

        assert len(result) == 2
        assert result[0].user_message == "q1"
        assert result[1].user_message == "q2"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_redis_unavailable(self):
        store = ConversationHistoryStore()
        with patch(
            "src.utils.conversation_history_store.get_redis_client", return_value=None
        ):
            result = await store.get_history("any-chat")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_error(self):
        store = ConversationHistoryStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=RuntimeError("connection lost"))

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_history("chat-err")

        assert result == []


# ---------------------------------------------------------------------------
# ConversationHistoryStore.get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    @pytest.mark.asyncio
    async def test_returns_none_when_key_missing(self):
        store = ConversationHistoryStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=None)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_summary("chat-1")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_summary_string(self):
        store = ConversationHistoryStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value="User asked about holidays.")

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_summary("chat-1")

        assert result == "User asked about holidays."

    @pytest.mark.asyncio
    async def test_returns_none_when_redis_unavailable(self):
        store = ConversationHistoryStore()
        with patch(
            "src.utils.conversation_history_store.get_redis_client", return_value=None
        ):
            result = await store.get_summary("any-chat")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self):
        store = ConversationHistoryStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            result = await store.get_summary("chat-err")

        assert result is None


# ---------------------------------------------------------------------------
# ConversationHistoryStore.save_summary
# ---------------------------------------------------------------------------


class TestSaveSummary:
    @pytest.mark.asyncio
    async def test_save_summary_sets_key_with_ttl(self):
        store = ConversationHistoryStore()

        pipe = _make_pipe_mock()
        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_summary("chat-1", "Some summary text.")

        pipe.set.assert_called_once()
        set_call = pipe.set.call_args
        assert set_call[0][0] == _summary_key("chat-1")
        assert set_call[0][1] == "Some summary text."
        assert set_call[1]["ex"] == _HISTORY_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_save_summary_resets_history_key_ttl(self):
        store = ConversationHistoryStore()

        pipe = _make_pipe_mock()
        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_summary("chat-2", "Summary.")

        pipe.expire.assert_called_once_with(
            _history_key("chat-2"), _HISTORY_TTL_SECONDS
        )

    @pytest.mark.asyncio
    async def test_save_summary_skips_when_redis_unavailable(self):
        store = ConversationHistoryStore()
        with patch(
            "src.utils.conversation_history_store.get_redis_client", return_value=None
        ):
            await store.save_summary("chat-x", "text")

    @pytest.mark.asyncio
    async def test_save_summary_graceful_on_error(self):
        store = ConversationHistoryStore()
        pipe = _make_pipe_mock()
        pipe.execute = AsyncMock(side_effect=RuntimeError("io error"))
        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_summary("chat-err", "text")


# ---------------------------------------------------------------------------
# ConversationHistoryStore.get_context
# ---------------------------------------------------------------------------


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_combines_history_and_summary(self):
        store = ConversationHistoryStore()
        rounds = [_make_round(user_message="hello", bot_message="hi")]
        summary_text = "Earlier the user greeted the bot."

        async def _fake_get_history(chat_id: str):
            return rounds

        async def _fake_get_summary(chat_id: str):
            return summary_text

        with (
            patch.object(store, "get_history", side_effect=_fake_get_history),
            patch.object(store, "get_summary", side_effect=_fake_get_summary),
        ):
            result = await store.get_context("chat-1")

        assert isinstance(result, ConversationHistoryState)
        assert result.chat_id == "chat-1"
        assert result.rounds == rounds
        assert result.summary == summary_text

    @pytest.mark.asyncio
    async def test_get_context_returns_empty_defaults_when_redis_unavailable(self):
        store = ConversationHistoryStore()
        with patch(
            "src.utils.conversation_history_store.get_redis_client", return_value=None
        ):
            result = await store.get_context("chat-gone")

        assert isinstance(result, ConversationHistoryState)
        assert result.chat_id == "chat-gone"
        assert result.rounds == []
        assert result.summary is None

    @pytest.mark.asyncio
    async def test_get_context_fetches_concurrently(self):
        """Both sub-calls must run; verify gather behaviour by checking both are awaited."""
        store = ConversationHistoryStore()
        history_called = False
        summary_called = False

        async def _hist(chat_id: str):
            nonlocal history_called
            history_called = True
            return []

        async def _summ(chat_id: str):
            nonlocal summary_called
            summary_called = True
            return None

        with (
            patch.object(store, "get_history", side_effect=_hist),
            patch.object(store, "get_summary", side_effect=_summ),
        ):
            await store.get_context("chat-concurrent")

        assert history_called
        assert summary_called


# ---------------------------------------------------------------------------
# Incremental summary: save_round eviction triggering
# ---------------------------------------------------------------------------


class TestSaveRoundIncrementalSummary:
    """Tests for the fire-and-forget summarizer integration in save_round."""

    @staticmethod
    def _make_pipe_with_existing(rounds_data: list[dict]) -> AsyncMock:
        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=json.dumps(rounds_data))
        return pipe

    @pytest.mark.asyncio
    async def test_eviction_triggers_background_task(self):
        """When len(rounds) exceeds _MAX_ROUNDS, summarizer is called with the
        evicted round(s) and the existing summary."""
        received_summary: list[str | None] = []
        received_evicted: list[list] = []

        async def mock_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            received_summary.append(existing_summary)
            received_evicted.append(list(evicted_rounds))
            return "merged summary"

        store = ConversationHistoryStore(summarizer=mock_summarizer)

        # Pre-populate with exactly _MAX_ROUNDS rounds so the new one triggers eviction.
        existing = [
            _make_round(user_message=f"q{i}", bot_message=f"a{i}").model_dump()
            for i in range(_MAX_ROUNDS)
        ]
        pipe = self._make_pipe_with_existing(existing)
        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        new_round = _make_round(user_message="q_new", bot_message="a_new")

        with (
            patch(
                "src.utils.conversation_history_store.get_redis_client",
                return_value=redis_mock,
            ),
            patch.object(store, "_get_summary_lock", return_value=asyncio.Lock()),
            patch.object(store, "get_summary", AsyncMock(return_value="old summary")),
            patch.object(store, "save_summary", AsyncMock()),
        ):
            await store.save_round("chat-evict", new_round)
            # Drain the event loop so the background task completes.
            await asyncio.gather(*store._pending_tasks, return_exceptions=True)

        assert len(received_evicted) == 1
        assert len(received_evicted[0]) == 1
        assert received_evicted[0][0].user_message == "q0"
        assert received_summary[0] == "old summary"

    @pytest.mark.asyncio
    async def test_no_eviction_does_not_trigger_summarizer(self):
        """When rounds stay within _MAX_ROUNDS, the summarizer is never called."""
        called = False

        async def mock_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            nonlocal called
            called = True
            return "should not be called"

        store = ConversationHistoryStore(summarizer=mock_summarizer)

        # Start with fewer than _MAX_ROUNDS rounds.
        existing = [
            _make_round(user_message=f"q{i}", bot_message=f"a{i}").model_dump()
            for i in range(_MAX_ROUNDS - 2)
        ]
        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=json.dumps(existing))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-no-evict", _make_round())

        assert not called
        assert len(store._pending_tasks) == 0

    @pytest.mark.asyncio
    async def test_no_summarizer_no_task_on_eviction(self):
        """Store without a summarizer still trims correctly; no tasks scheduled."""
        store = ConversationHistoryStore(summarizer=None)

        existing = [
            _make_round(user_message=f"q{i}", bot_message=f"a{i}").model_dump()
            for i in range(_MAX_ROUNDS)
        ]
        pipe = _make_pipe_mock()
        pipe.get = AsyncMock(return_value=json.dumps(existing))

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe)

        with patch(
            "src.utils.conversation_history_store.get_redis_client",
            return_value=redis_mock,
        ):
            await store.save_round("chat-no-summarizer", _make_round())

        assert len(store._pending_tasks) == 0

        # Verify that trimming still happened.
        stored_json = pipe.set.call_args[0][1]
        stored = json.loads(stored_json)
        assert len(stored) == _MAX_ROUNDS


# ---------------------------------------------------------------------------
# Incremental summary: _run_incremental_summary
# ---------------------------------------------------------------------------


class TestRunIncrementalSummary:
    """Tests for the _run_incremental_summary private method."""

    @pytest.mark.asyncio
    async def test_calls_save_summary_with_merged_result(self):
        """Happy path: summarizer returns non-empty string → save_summary called."""

        async def mock_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            return "merged: " + (existing_summary or "") + " + new info"

        store = ConversationHistoryStore(summarizer=mock_summarizer)
        evicted = [_make_round(user_message="old q", bot_message="old a")]

        with (
            patch.object(store, "_get_summary_lock", return_value=asyncio.Lock()),
            patch.object(store, "get_summary", AsyncMock(return_value="prior summary")),
            patch.object(store, "save_summary", AsyncMock()) as mock_save,
        ):
            await store._run_incremental_summary("chat-1", evicted)

        mock_save.assert_awaited_once_with("chat-1", "merged: prior summary + new info")

    @pytest.mark.asyncio
    async def test_skips_save_when_summarizer_returns_empty(self):
        """If summarizer returns empty string, save_summary must NOT be called."""

        async def mock_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            return ""

        store = ConversationHistoryStore(summarizer=mock_summarizer)
        evicted = [_make_round()]

        with (
            patch.object(store, "_get_summary_lock", return_value=asyncio.Lock()),
            patch.object(store, "get_summary", AsyncMock(return_value=None)),
            patch.object(store, "save_summary", AsyncMock()) as mock_save,
        ):
            await store._run_incremental_summary("chat-2", evicted)

        mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summarizer_exception_does_not_propagate(self):
        """A failing summarizer must be caught; the method returns cleanly."""

        async def exploding_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            raise RuntimeError("LLM unavailable")

        store = ConversationHistoryStore(summarizer=exploding_summarizer)
        evicted = [_make_round()]

        with (
            patch.object(store, "_get_summary_lock", return_value=asyncio.Lock()),
            patch.object(store, "get_summary", AsyncMock(return_value=None)),
            patch.object(store, "save_summary", AsyncMock()) as mock_save,
        ):
            # Must not raise.
            await store._run_incremental_summary("chat-3", evicted)

        mock_save.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_passes_none_existing_summary_when_no_summary_stored(self):
        """When get_summary returns None, summarizer receives None as first arg."""
        received: list[str | None] = []

        async def capture_summarizer(
            existing_summary: str | None,
            evicted_rounds: list[ConversationRound],
        ) -> str:
            received.append(existing_summary)
            return "new summary"

        store = ConversationHistoryStore(summarizer=capture_summarizer)
        evicted = [_make_round()]

        with (
            patch.object(store, "_get_summary_lock", return_value=asyncio.Lock()),
            patch.object(store, "get_summary", AsyncMock(return_value=None)),
            patch.object(store, "save_summary", AsyncMock()),
        ):
            await store._run_incremental_summary("chat-4", evicted)

        assert received == [None]

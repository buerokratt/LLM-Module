"""Unit tests for APIToolSessionStore and session models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from models.session_models import APIToolSession, EndpointSessionState
from utils.api_tool_session_store import (
    APIToolSessionStore,
    _key,
    require_session_store,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**kwargs) -> APIToolSession:
    defaults = {
        "chat_id": "test-chat-1",
        "state": "collecting_params",
        "selected_endpoint": {"url": "https://example.com", "method": "GET"},
        "collected_params": {"countryIsoCode": "EE"},
        "turn_count": 1,
        "max_turns": 5,
    }
    defaults.update(kwargs)
    return APIToolSession(**defaults)


def _make_redis_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.exists = AsyncMock(return_value=0)
    mock.ping = AsyncMock(return_value=True)
    return mock


# ---------------------------------------------------------------------------
# Session model tests
# ---------------------------------------------------------------------------


class TestEndpointSessionState:
    def test_defaults(self):
        ep = EndpointSessionState(
            endpoint={"name": "get_holidays", "url": "https://example.com"}
        )
        assert ep.collected_params == {}
        assert ep.completed is False

    def test_mark_completed(self):
        ep = EndpointSessionState(
            endpoint={"name": "get_holidays"},
            collected_params={"year": "2026"},
            completed=True,
        )
        assert ep.completed is True
        assert ep.collected_params == {"year": "2026"}

    def test_serialization_roundtrip(self):
        ep = EndpointSessionState(
            endpoint={"name": "get_electricity_prices", "url": "https://example.com"},
            collected_params={"region": "EE"},
        )
        restored = EndpointSessionState.model_validate_json(ep.model_dump_json())
        assert restored == ep

    def test_endpoint_is_required(self):
        with pytest.raises(ValidationError):
            EndpointSessionState()  # type: ignore[call-arg]


class TestAPIToolSession:
    def test_defaults(self):
        session = APIToolSession(chat_id="abc", state="collecting_params")
        assert session.collected_params == {}
        assert session.turn_count == 0
        assert session.max_turns == 5
        assert session.selected_endpoint is None
        # Phase 2 parallel-mode defaults
        assert session.execution_mode == "single"
        assert session.parallel_endpoints == []
        assert session.active_endpoint_index == 0

    def test_serialization_roundtrip(self):
        session = _make_session()
        json_str = session.model_dump_json()
        restored = APIToolSession.model_validate_json(json_str)
        assert restored == session

    def test_turn_count_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            APIToolSession(chat_id="x", state="s", turn_count=-1)

    def test_max_turns_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            APIToolSession(chat_id="x", state="s", max_turns=0)

    # ── Phase 2: parallel-mode fields ────────────────────────────────────

    def test_parallel_session_stores_endpoint_states(self):
        ep1 = EndpointSessionState(endpoint={"name": "get_holidays"})
        ep2 = EndpointSessionState(endpoint={"name": "get_electricity_prices"})
        session = APIToolSession(
            chat_id="chat-p1",
            state="collecting_params",
            execution_mode="parallel",
            parallel_endpoints=[ep1, ep2],
        )
        assert session.execution_mode == "parallel"
        assert len(session.parallel_endpoints) == 2
        assert session.parallel_endpoints[0].endpoint["name"] == "get_holidays"
        assert (
            session.parallel_endpoints[1].endpoint["name"] == "get_electricity_prices"
        )

    def test_active_endpoint_index_defaults_to_zero(self):
        session = APIToolSession(
            chat_id="chat-p2",
            state="collecting_params",
            execution_mode="parallel",
            parallel_endpoints=[EndpointSessionState(endpoint={"name": "ep1"})],
        )
        assert session.active_endpoint_index == 0

    def test_active_endpoint_index_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            APIToolSession(
                chat_id="x",
                state="s",
                active_endpoint_index=-1,
            )

    def test_parallel_session_serialization_roundtrip(self):
        ep1 = EndpointSessionState(
            endpoint={"name": "get_holidays", "url": "https://example.com"},
            collected_params={"year": "2026"},
        )
        ep2 = EndpointSessionState(
            endpoint={"name": "get_electricity_prices", "url": "https://example.com"},
            completed=True,
        )
        session = APIToolSession(
            chat_id="chat-parallel",
            state="collecting_params",
            execution_mode="parallel",
            parallel_endpoints=[ep1, ep2],
            active_endpoint_index=1,
        )
        restored = APIToolSession.model_validate_json(session.model_dump_json())
        assert restored == session
        assert restored.parallel_endpoints[1].completed is True
        assert restored.active_endpoint_index == 1

    def test_backward_compat_session_without_parallel_fields(self):
        """Sessions serialised before Phase 2 (no parallel fields) load cleanly."""
        legacy_json = (
            '{"chat_id":"legacy","state":"collecting_params",'
            '"selected_endpoint":null,"collected_params":{},'
            '"turn_count":0,"max_turns":5,"awaiting_continuation":false,'
            '"detected_language":"en","original_query":""}'
        )
        session = APIToolSession.model_validate_json(legacy_json)
        assert session.execution_mode == "single"
        assert session.parallel_endpoints == []
        assert session.active_endpoint_index == 0


# ---------------------------------------------------------------------------
# APIToolSessionStore.get
# ---------------------------------------------------------------------------


class TestSessionStoreGet:
    @pytest.mark.asyncio
    async def test_get_returns_none_when_key_missing(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=None)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.get("missing-chat")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_session_when_key_exists(self):
        store = APIToolSessionStore()
        session = _make_session()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=session.model_dump_json())

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.get(session.chat_id)

        assert result is not None
        assert result.chat_id == session.chat_id
        assert result.state == session.state

    @pytest.mark.asyncio
    async def test_get_returns_none_when_redis_unavailable(self):
        store = APIToolSessionStore()
        with patch("utils.api_tool_session_store.get_redis_client", return_value=None):
            result = await store.get("any-chat")

        assert result is None


# ---------------------------------------------------------------------------
# APIToolSessionStore.save
# ---------------------------------------------------------------------------


class TestSessionStoreSave:
    @pytest.mark.asyncio
    async def test_save_calls_redis_set_with_correct_key_and_ttl(self):
        store = APIToolSessionStore()
        session = _make_session()
        redis_mock = _make_redis_mock()

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            await store.save(session)

        redis_mock.set.assert_awaited_once()
        call_args = redis_mock.set.call_args
        assert call_args[0][0] == _key(session.chat_id)
        assert call_args[1]["ex"] == 1800

    @pytest.mark.asyncio
    async def test_save_skips_when_redis_unavailable(self):
        store = APIToolSessionStore()
        session = _make_session()

        with patch("utils.api_tool_session_store.get_redis_client", return_value=None):
            # Should not raise
            await store.save(session)


# ---------------------------------------------------------------------------
# APIToolSessionStore.update (partial merge)
# ---------------------------------------------------------------------------


class TestSessionStoreUpdate:
    @pytest.mark.asyncio
    async def test_update_rejects_unknown_fields(self):
        store = APIToolSessionStore()

        with pytest.raises(ValueError, match="Unknown session fields"):
            await store.update("chat-1", sate="ready")  # typo: sate != state

    @pytest.mark.asyncio
    async def test_update_merges_fields_and_resets_ttl(self):
        store = APIToolSessionStore()
        original = _make_session(turn_count=1, collected_params={"a": "1"})

        pipe_mock = AsyncMock()
        pipe_mock.get = AsyncMock(return_value=original.model_dump_json())
        pipe_mock.watch = AsyncMock()
        pipe_mock.unwatch = AsyncMock()
        pipe_mock.multi = MagicMock()
        pipe_mock.set = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[True])
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.update(
                original.chat_id,
                turn_count=2,
                collected_params={"a": "1", "b": "2"},
            )

        assert result is not None
        assert result.turn_count == 2
        assert result.collected_params == {"a": "1", "b": "2"}
        # Unchanged field preserved
        assert result.state == original.state

    @pytest.mark.asyncio
    async def test_update_returns_none_when_session_missing(self):
        store = APIToolSessionStore()

        pipe_mock = AsyncMock()
        pipe_mock.get = AsyncMock(return_value=None)
        pipe_mock.watch = AsyncMock()
        pipe_mock.unwatch = AsyncMock()
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.update("ghost-chat", turn_count=3)

        assert result is None

    @pytest.mark.asyncio
    async def test_update_resets_ttl(self):
        store = APIToolSessionStore()
        session = _make_session()

        pipe_mock = AsyncMock()
        pipe_mock.get = AsyncMock(return_value=session.model_dump_json())
        pipe_mock.watch = AsyncMock()
        pipe_mock.unwatch = AsyncMock()
        pipe_mock.multi = MagicMock()
        pipe_mock.set = MagicMock()
        pipe_mock.execute = AsyncMock(return_value=[True])
        pipe_mock.__aenter__ = AsyncMock(return_value=pipe_mock)
        pipe_mock.__aexit__ = AsyncMock(return_value=False)

        redis_mock = _make_redis_mock()
        redis_mock.pipeline = MagicMock(return_value=pipe_mock)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            await store.update(session.chat_id, state="ready")

        # pipe.set() should have been called with ex=1800
        pipe_mock.set.assert_called_once()
        set_call = pipe_mock.set.call_args
        assert set_call[1]["ex"] == 1800


# ---------------------------------------------------------------------------
# APIToolSessionStore.delete
# ---------------------------------------------------------------------------


class TestSessionStoreDelete:
    @pytest.mark.asyncio
    async def test_delete_calls_redis_delete(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            await store.delete("chat-to-delete")

        redis_mock.delete.assert_awaited_once_with(_key("chat-to-delete"))

    @pytest.mark.asyncio
    async def test_delete_skips_when_redis_unavailable(self):
        store = APIToolSessionStore()
        with patch("utils.api_tool_session_store.get_redis_client", return_value=None):
            await store.delete("any-chat")  # Should not raise


# ---------------------------------------------------------------------------
# APIToolSessionStore.exists
# ---------------------------------------------------------------------------


class TestSessionStoreExists:
    @pytest.mark.asyncio
    async def test_exists_returns_true_when_key_present(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.exists = AsyncMock(return_value=1)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.exists("chat-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_key_absent(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.exists = AsyncMock(return_value=0)

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.exists("chat-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_when_redis_unavailable(self):
        store = APIToolSessionStore()
        with patch("utils.api_tool_session_store.get_redis_client", return_value=None):
            result = await store.exists("any-chat")

        assert result is False


# ---------------------------------------------------------------------------
# Graceful error handling
# ---------------------------------------------------------------------------


class TestSessionStoreErrorHandling:
    @pytest.mark.asyncio
    async def test_get_returns_none_on_redis_error(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.get("chat-xyz")

        assert result is None

    @pytest.mark.asyncio
    async def test_save_does_not_raise_on_redis_error(self):
        store = APIToolSessionStore()
        session = _make_session()
        redis_mock = _make_redis_mock()
        redis_mock.set = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            await store.save(session)  # Should not raise

    @pytest.mark.asyncio
    async def test_delete_does_not_raise_on_redis_error(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.delete = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            await store.delete("chat-xyz")  # Should not raise

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_redis_error(self):
        store = APIToolSessionStore()
        redis_mock = _make_redis_mock()
        redis_mock.exists = AsyncMock(side_effect=ConnectionError("timeout"))

        with patch(
            "utils.api_tool_session_store.get_redis_client", return_value=redis_mock
        ):
            result = await store.exists("chat-xyz")

        assert result is False


# ---------------------------------------------------------------------------
# require_session_store dependency
# ---------------------------------------------------------------------------


class TestRequireSessionStore:
    def test_returns_store_when_available(self):
        store = APIToolSessionStore()
        mock_request = MagicMock()
        mock_request.app.state.session_store = store
        mock_request.url.path = "/api-tool/invoke"

        result = require_session_store(mock_request)
        assert result is store

    def test_raises_503_when_store_is_none(self):
        from fastapi import HTTPException

        mock_request = MagicMock()
        mock_request.app.state.session_store = None
        mock_request.url.path = "/api-tool/invoke"

        with pytest.raises(HTTPException) as exc_info:
            require_session_store(mock_request)

        assert exc_info.value.status_code == 503

    def test_raises_503_when_state_attr_missing(self):
        from fastapi import HTTPException

        mock_request = MagicMock(spec=["app", "url"])
        mock_request.app = MagicMock(spec=["state"])
        mock_request.app.state = MagicMock(spec=[])  # no session_store attr
        mock_request.url.path = "/api-tool/invoke"

        with pytest.raises(HTTPException) as exc_info:
            require_session_store(mock_request)

        assert exc_info.value.status_code == 503

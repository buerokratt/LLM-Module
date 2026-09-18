"""Unit tests for ATCCacheStore — T-2."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from models.session_models import LastCallContext
from tool_classifier.constants import (
    ATC_CACHE_DEFAULT_TTL_SECONDS,
    ATC_CACHE_KEY_PREFIX,
    ATC_LAST_CALL_KEY_PREFIX,
    ATC_LAST_CALL_TTL_SECONDS,
)
from utils.atc_cache_store import ATCCacheStore

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CHAT_ID = "chat-test-001"
API_NAME = "get_national_holidays"
PARAMS = {"country": "EE", "year": 2026}
RESPONSE = {"holidays": [{"date": "2026-02-24", "name": "Independence Day"}]}


def _make_redis_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    return mock


def _make_last_call_ctx(api_name: str = API_NAME) -> LastCallContext:
    return LastCallContext(
        api_name=api_name,
        endpoint={"name": api_name, "url": "https://example.com", "method": "GET"},
        collected_params=PARAMS,
        raw_response=RESPONSE,
        original_query="What are the public holidays in Estonia in 2026?",
        timestamp=1748000000.0,
    )


def _fake_redis_store() -> tuple[dict, AsyncMock]:
    """Return a (store_dict, redis_mock) pair backed by an in-memory dict."""
    store: dict = {}

    async def fake_set(key, value, ex=None):
        store[key] = value

    async def fake_get(key):
        return store.get(key)

    async def fake_delete(key):
        store.pop(key, None)

    mock = AsyncMock()
    mock.get = AsyncMock(side_effect=fake_get)
    mock.set = AsyncMock(side_effect=fake_set)
    mock.delete = AsyncMock(side_effect=fake_delete)
    return store, mock


# ---------------------------------------------------------------------------
# _normalise_params
# ---------------------------------------------------------------------------


class TestNormaliseParams:
    def test_strips_whitespace_from_strings(self):
        result = ATCCacheStore._normalise_params({"country": "  EE  "})
        assert result["country"] == "ee"

    def test_casts_numeric_string_to_int(self):
        result = ATCCacheStore._normalise_params({"year": "2026"})
        assert result["year"] == 2026
        assert isinstance(result["year"], int)

    def test_lowercases_alpha_only_strings(self):
        result = ATCCacheStore._normalise_params({"method": "GET"})
        assert result["method"] == "get"

    def test_leaves_mixed_strings_as_stripped_only(self):
        result = ATCCacheStore._normalise_params({"query": "  hello world  "})
        assert result["query"] == "hello world"

    def test_leaves_non_string_values_unchanged(self):
        result = ATCCacheStore._normalise_params({"year": 2026, "active": True})
        assert result["year"] == 2026
        assert result["active"] is True

    def test_mixed_param_types(self):
        result = ATCCacheStore._normalise_params(
            {"country": "  EE  ", "year": "2026", "method": "GET", "count": 5}
        )
        assert result == {"country": "ee", "year": 2026, "method": "get", "count": 5}


# ---------------------------------------------------------------------------
# _param_hash
# ---------------------------------------------------------------------------


class TestParamHash:
    def test_string_and_int_year_produce_same_hash(self):
        """Core normalisation requirement: "2026" and 2026 must hash identically."""
        h_str = ATCCacheStore._param_hash({"year": "2026", "country": "EE"})
        h_int = ATCCacheStore._param_hash({"year": 2026, "country": "EE"})
        assert h_str == h_int

    def test_hash_is_16_hex_chars(self):
        h = ATCCacheStore._param_hash({"year": 2026})
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_params_produce_different_hash(self):
        h1 = ATCCacheStore._param_hash({"year": 2026, "country": "EE"})
        h2 = ATCCacheStore._param_hash({"year": 2025, "country": "EE"})
        assert h1 != h2

    def test_key_order_does_not_affect_hash(self):
        h1 = ATCCacheStore._param_hash({"country": "EE", "year": 2026})
        h2 = ATCCacheStore._param_hash({"year": 2026, "country": "EE"})
        assert h1 == h2

    def test_whitespace_is_normalised_before_hashing(self):
        h1 = ATCCacheStore._param_hash({"country": "EE"})
        h2 = ATCCacheStore._param_hash({"country": "  EE  "})
        assert h1 == h2

    def test_alpha_case_is_normalised_before_hashing(self):
        h1 = ATCCacheStore._param_hash({"method": "get"})
        h2 = ATCCacheStore._param_hash({"method": "GET"})
        assert h1 == h2


# ---------------------------------------------------------------------------
# Key format
# ---------------------------------------------------------------------------


class TestKeyFormat:
    def test_l1_key_includes_all_components(self):
        expected_hash = ATCCacheStore._param_hash(PARAMS)
        key = ATCCacheStore._l1_key(CHAT_ID, API_NAME, PARAMS)
        assert key == f"{ATC_CACHE_KEY_PREFIX}:{CHAT_ID}:{API_NAME}:{expected_hash}"

    def test_l2_key_format(self):
        key = ATCCacheStore._l2_key(CHAT_ID)
        assert key == f"{ATC_LAST_CALL_KEY_PREFIX}:{CHAT_ID}"

    def test_l1_and_l2_keys_have_different_prefixes(self):
        l1 = ATCCacheStore._l1_key(CHAT_ID, API_NAME, PARAMS)
        l2 = ATCCacheStore._l2_key(CHAT_ID)
        assert not l1.startswith(ATC_LAST_CALL_KEY_PREFIX)
        assert not l2.startswith(ATC_CACHE_KEY_PREFIX)


# ---------------------------------------------------------------------------
# get_l1
# ---------------------------------------------------------------------------


class TestGetL1:
    @pytest.mark.asyncio
    async def test_returns_deserialised_value_on_hit(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=json.dumps(RESPONSE))

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result == RESPONSE

    @pytest.mark.asyncio
    async def test_returns_none_on_miss(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(return_value=None)

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_redis_unavailable(self):
        store = ATCCacheStore()
        with patch("utils.atc_cache_store.get_redis_client", return_value=None):
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_redis_exception(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=RuntimeError("connection lost"))

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result is None

    @pytest.mark.asyncio
    async def test_different_params_return_none(self):
        """Different params hash to a different key — must not return the stored entry."""
        store = ATCCacheStore()
        stored_key = ATCCacheStore._l1_key(CHAT_ID, API_NAME, PARAMS)
        different_params = {"country": "LV", "year": 2026}

        async def selective_get(key):
            return json.dumps(RESPONSE) if key == stored_key else None

        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=selective_get)

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l1(CHAT_ID, API_NAME, different_params)

        assert result is None


# ---------------------------------------------------------------------------
# set_l1
# ---------------------------------------------------------------------------


class TestSetL1:
    @pytest.mark.asyncio
    async def test_calls_redis_set_with_correct_key_and_value(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        expected_key = ATCCacheStore._l1_key(CHAT_ID, API_NAME, PARAMS)

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE)

        redis_mock.set.assert_called_once_with(
            expected_key,
            json.dumps(RESPONSE),
            ex=ATC_CACHE_DEFAULT_TTL_SECONDS,
        )

    @pytest.mark.asyncio
    async def test_uses_default_ttl_when_not_specified(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE)

        assert redis_mock.set.call_args[1]["ex"] == ATC_CACHE_DEFAULT_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_uses_custom_ttl_when_provided(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE, ttl=120)

        assert redis_mock.set.call_args[1]["ex"] == 120

    @pytest.mark.asyncio
    async def test_no_op_when_redis_unavailable(self):
        store = ATCCacheStore()
        with patch("utils.atc_cache_store.get_redis_client", return_value=None):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE)  # must not raise

    @pytest.mark.asyncio
    async def test_no_op_on_redis_exception(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.set = AsyncMock(side_effect=RuntimeError("write failed"))

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE)  # must not raise


# ---------------------------------------------------------------------------
# L1 round-trip
# ---------------------------------------------------------------------------


class TestL1RoundTrip:
    @pytest.mark.asyncio
    async def test_set_then_get_returns_same_response(self):
        store = ATCCacheStore()
        _, redis_mock = _fake_redis_store()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, RESPONSE)
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result == RESPONSE

    @pytest.mark.asyncio
    async def test_string_year_hits_entry_stored_with_int_year(self):
        """Normalisation: set with int year, get with string year — must still hit."""
        store = ATCCacheStore()
        _, redis_mock = _fake_redis_store()
        params_int = {"country": "EE", "year": 2026}
        params_str = {"country": "EE", "year": "2026"}

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, params_int, RESPONSE)
            result = await store.get_l1(CHAT_ID, API_NAME, params_str)

        assert result == RESPONSE

    @pytest.mark.asyncio
    async def test_list_response_survives_round_trip(self):
        """raw_response may be a list — must serialise/deserialise cleanly."""
        store = ATCCacheStore()
        _, redis_mock = _fake_redis_store()
        list_response = [{"date": "2026-02-24"}, {"date": "2026-06-23"}]

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l1(CHAT_ID, API_NAME, PARAMS, list_response)
            result = await store.get_l1(CHAT_ID, API_NAME, PARAMS)

        assert result == list_response


# ---------------------------------------------------------------------------
# get_l2 / set_l2
# ---------------------------------------------------------------------------


class TestSetAndGetL2:
    @pytest.mark.asyncio
    async def test_round_trip_returns_correct_context_list(self):
        store = ATCCacheStore()
        ctx = _make_last_call_ctx()
        _, redis_mock = _fake_redis_store()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx])
            result = await store.get_l2(CHAT_ID)

        assert result is not None
        assert len(result) == 1
        assert result[0].api_name == API_NAME
        assert result[0].collected_params == PARAMS
        assert result[0].raw_response == RESPONSE
        assert result[0].original_query == ctx.original_query

    @pytest.mark.asyncio
    async def test_multi_intent_stores_all_entries(self):
        """set_l2 with two contexts — get_l2 returns both."""
        store = ATCCacheStore()
        ctx1 = _make_last_call_ctx("get_national_holidays")
        ctx2 = _make_last_call_ctx("get_electricity_prices")
        _, redis_mock = _fake_redis_store()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx1, ctx2])
            result = await store.get_l2(CHAT_ID)

        assert result is not None
        assert len(result) == 2
        assert {r.api_name for r in result} == {
            "get_national_holidays",
            "get_electricity_prices",
        }

    @pytest.mark.asyncio
    async def test_set_l2_uses_correct_ttl(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        ctx = _make_last_call_ctx()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx])

        assert redis_mock.set.call_args[1]["ex"] == ATC_LAST_CALL_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_set_l2_writes_to_correct_key(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        ctx = _make_last_call_ctx()
        expected_key = ATCCacheStore._l2_key(CHAT_ID)

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx])

        assert redis_mock.set.call_args[0][0] == expected_key

    @pytest.mark.asyncio
    async def test_get_l2_returns_none_on_miss(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l2(CHAT_ID)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_l2_returns_none_when_redis_unavailable(self):
        store = ATCCacheStore()
        with patch("utils.atc_cache_store.get_redis_client", return_value=None):
            result = await store.get_l2(CHAT_ID)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_l2_returns_none_on_exception(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.get = AsyncMock(side_effect=RuntimeError("connection reset"))

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            result = await store.get_l2(CHAT_ID)

        assert result is None

    @pytest.mark.asyncio
    async def test_set_l2_no_op_when_redis_unavailable(self):
        store = ATCCacheStore()
        ctx = _make_last_call_ctx()
        with patch("utils.atc_cache_store.get_redis_client", return_value=None):
            await store.set_l2(CHAT_ID, [ctx])  # must not raise

    @pytest.mark.asyncio
    async def test_set_l2_no_op_on_redis_exception(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.set = AsyncMock(side_effect=RuntimeError("write error"))
        ctx = _make_last_call_ctx()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx])  # must not raise


# ---------------------------------------------------------------------------
# invalidate_l2
# ---------------------------------------------------------------------------


class TestInvalidateL2:
    @pytest.mark.asyncio
    async def test_calls_delete_with_correct_l2_key(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        expected_key = ATCCacheStore._l2_key(CHAT_ID)

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.invalidate_l2(CHAT_ID)

        redis_mock.delete.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_get_l2_returns_none_after_invalidate(self):
        store = ATCCacheStore()
        _, redis_mock = _fake_redis_store()
        ctx = _make_last_call_ctx()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.set_l2(CHAT_ID, [ctx])
            await store.invalidate_l2(CHAT_ID)
            result = await store.get_l2(CHAT_ID)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_only_deletes_l2_key_not_l1(self):
        """invalidate_l2 must call delete exactly once with the L2 prefix."""
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.invalidate_l2(CHAT_ID)

        deleted_key: str = redis_mock.delete.call_args[0][0]
        assert deleted_key.startswith(ATC_LAST_CALL_KEY_PREFIX)
        assert not deleted_key.startswith(ATC_CACHE_KEY_PREFIX)

    @pytest.mark.asyncio
    async def test_no_op_when_redis_unavailable(self):
        store = ATCCacheStore()
        with patch("utils.atc_cache_store.get_redis_client", return_value=None):
            await store.invalidate_l2(CHAT_ID)  # must not raise

    @pytest.mark.asyncio
    async def test_no_op_on_redis_exception(self):
        store = ATCCacheStore()
        redis_mock = _make_redis_mock()
        redis_mock.delete = AsyncMock(side_effect=RuntimeError("gone"))

        with patch("utils.atc_cache_store.get_redis_client", return_value=redis_mock):
            await store.invalidate_l2(CHAT_ID)  # must not raise

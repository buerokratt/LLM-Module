"""Unit tests for the APICaller and CircuitBreaker modules."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tool_classifier.api_caller import APICaller, CircuitBreaker
from tool_classifier.constants import (
    CB_STATE_CLOSED,
    CB_STATE_HALF_OPEN,
    CB_STATE_OPEN,
    CIRCUIT_BREAKER_OPEN_MESSAGES,
    SERVICE_TIMEOUT_MESSAGES,
    SERVICE_UNAVAILABLE_MESSAGES,
)
from tool_classifier.models import APICallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_URL = "https://openholidaysapi.org/PublicHolidays"
_URL_B = "http://api.other.com/resource"
_GET_PARAMS = {"country": "EE", "year": "2024"}
_POST_PARAMS = {"firstName": "Test", "lastName": "User"}


def _make_response(
    status_code: int,
    json_data: dict | None = None,
    text_data: str = "",
) -> MagicMock:
    """Build a mock httpx.Response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    if json_data is not None:
        mock_response.json.return_value = json_data
    else:
        mock_response.json.side_effect = json.JSONDecodeError("No JSON", "", 0)
    mock_response.text = text_data
    return mock_response


def _make_client(
    response: MagicMock | None = None,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Build a mock httpx.AsyncClient usable as an async context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    if side_effect is not None:
        mock_client.get.side_effect = side_effect
        mock_client.post.side_effect = side_effect
    elif response is not None:
        mock_client.get.return_value = response
        mock_client.post.return_value = response
    return mock_client


# ---------------------------------------------------------------------------
# GET request behaviour
# ---------------------------------------------------------------------------


class TestGetRequest:
    @pytest.mark.asyncio
    async def test_get_sends_params_as_query_parameters(self) -> None:
        response = _make_response(200, json_data={"holidays": []})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", _GET_PARAMS)

        mock_client.get.assert_called_once_with(_URL, params=_GET_PARAMS)
        mock_client.post.assert_not_called()
        assert result.success is True
        assert result.status_code == 200
        assert result.response_data == {"holidays": []}
        assert result.error is None

    @pytest.mark.asyncio
    async def test_get_method_case_insensitive(self) -> None:
        response = _make_response(200, json_data={"ok": True})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "get", _GET_PARAMS)

        mock_client.get.assert_called_once()
        assert result.success is True


# ---------------------------------------------------------------------------
# POST request behaviour
# ---------------------------------------------------------------------------


class TestPostRequest:
    @pytest.mark.asyncio
    async def test_post_sends_params_as_json_body(self) -> None:
        response = _make_response(200, json_data={"created": True})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "POST", _POST_PARAMS)

        mock_client.post.assert_called_once_with(_URL, json=_POST_PARAMS)
        mock_client.get.assert_not_called()
        assert result.success is True
        assert result.response_data == {"created": True}

    @pytest.mark.asyncio
    async def test_post_method_case_insensitive(self) -> None:
        response = _make_response(201, json_data={"id": 42})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "post", _POST_PARAMS)

        mock_client.post.assert_called_once()
        assert result.success is True
        assert result.status_code == 201


# ---------------------------------------------------------------------------
# Successful response handling
# ---------------------------------------------------------------------------


class TestSuccessResponse:
    @pytest.mark.asyncio
    async def test_non_json_response_returns_raw_text(self) -> None:
        response = _make_response(200, text_data="plain text response")
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {})

        assert result.success is True
        assert result.response_data == "plain text response"

    @pytest.mark.asyncio
    async def test_uses_custom_timeout_over_instance_default(self) -> None:
        response = _make_response(200, json_data={})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ) as mock_class:
            await APICaller(timeout=5).call(_URL, "GET", {}, timeout=30)

        mock_class.assert_called_once_with(timeout=30, follow_redirects=True)

    @pytest.mark.asyncio
    async def test_uses_instance_default_timeout_when_no_override(self) -> None:
        response = _make_response(200, json_data={})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ) as mock_class:
            await APICaller(timeout=15).call(_URL, "GET", {})

        mock_class.assert_called_once_with(timeout=15, follow_redirects=True)

    @pytest.mark.asyncio
    async def test_empty_params_dict_accepted(self) -> None:
        response = _make_response(200, json_data={"result": "ok"})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {})

        assert result.success is True


# ---------------------------------------------------------------------------
# 4xx client errors
# ---------------------------------------------------------------------------


class TestClientErrors:
    @pytest.mark.asyncio
    async def test_400_returns_error_with_body(self) -> None:
        error_body = {"error": "Invalid date format", "code": "INVALID_PARAM"}
        response = _make_response(400, json_data=error_body)
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "POST", {"date": "bad"})

        assert result.success is False
        assert result.status_code == 400
        assert result.is_client_error is True
        assert result.is_server_error is False
        assert result.response_data == error_body
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_404_with_plain_text_body(self) -> None:
        response = _make_response(404, text_data="Not Found")
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {"id": "999"})

        assert result.success is False
        assert result.status_code == 404
        assert result.is_client_error is True
        assert result.response_data == "Not Found"
        assert result.error is not None  # localized message for 404

    @pytest.mark.asyncio
    async def test_422_with_plain_text_body(self) -> None:
        response = _make_response(422, text_data="Unprocessable Entity")
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "POST", {})

        assert result.success is False
        assert result.status_code == 422
        assert result.response_data == "Unprocessable Entity"

    @pytest.mark.asyncio
    async def test_4xx_does_not_trip_circuit_breaker(self) -> None:
        """4xx client errors must not increment the circuit breaker failure count."""
        caller = APICaller(failure_threshold=3)
        response = _make_response(400, json_data={"error": "bad param"})
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            for _ in range(5):
                await caller.call(_URL, "POST", {})

        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_CLOSED


# ---------------------------------------------------------------------------
# 5xx server errors
# ---------------------------------------------------------------------------


class TestServerErrors:
    @pytest.mark.asyncio
    async def test_500_returns_friendly_message_default_language(self) -> None:
        response = _make_response(500)
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {})

        assert result.success is False
        assert result.status_code == 500
        assert result.is_server_error is True
        assert result.is_client_error is False
        assert result.error == SERVICE_UNAVAILABLE_MESSAGES["et"]

    @pytest.mark.asyncio
    async def test_503_returns_localized_message(self) -> None:
        response = _make_response(503)
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {}, language="en")

        assert result.success is False
        assert result.error == SERVICE_UNAVAILABLE_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_5xx_trips_circuit_breaker(self) -> None:
        caller = APICaller(failure_threshold=3)
        response = _make_response(500)
        mock_client = _make_client(response)

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            for _ in range(3):
                await caller.call(_URL, "GET", {})

        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN


# ---------------------------------------------------------------------------
# Timeout errors
# ---------------------------------------------------------------------------


class TestTimeoutErrors:
    @pytest.mark.asyncio
    async def test_read_timeout_returns_friendly_message(self) -> None:
        mock_client = _make_client(side_effect=httpx.ReadTimeout("timed out"))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {})

        assert result.success is False
        assert result.status_code == 0
        assert result.error == SERVICE_TIMEOUT_MESSAGES["et"]

    @pytest.mark.asyncio
    async def test_connect_timeout_returns_localized_message(self) -> None:
        mock_client = _make_client(side_effect=httpx.ConnectTimeout("timeout"))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {}, language="en")

        assert result.error == SERVICE_TIMEOUT_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_timeout_trips_circuit_breaker(self) -> None:
        caller = APICaller(failure_threshold=2)
        mock_client = _make_client(side_effect=httpx.ReadTimeout("timeout"))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            await caller.call(_URL, "GET", {})
            await caller.call(_URL, "GET", {})

        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN


# ---------------------------------------------------------------------------
# Network errors
# ---------------------------------------------------------------------------


class TestNetworkErrors:
    @pytest.mark.asyncio
    async def test_connect_error_returns_friendly_message(self) -> None:
        mock_client = _make_client(side_effect=httpx.ConnectError("connection refused"))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            result = await APICaller().call(_URL, "GET", {})

        assert result.success is False
        assert result.status_code == 0
        assert result.error == SERVICE_TIMEOUT_MESSAGES["et"]

    @pytest.mark.asyncio
    async def test_network_error_trips_circuit_breaker(self) -> None:
        caller = APICaller(failure_threshold=2)
        mock_client = _make_client(side_effect=httpx.ConnectError("refused"))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ):
            await caller.call(_URL, "GET", {})
            await caller.call(_URL, "GET", {})

        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN


# ---------------------------------------------------------------------------
# CircuitBreaker unit tests
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_open_after_threshold_failures(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

    def test_open_breaker_blocks_execution(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure(_URL)
        assert cb.can_execute(_URL) is False

    def test_transitions_to_half_open_after_cooldown(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=30.0)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

        future_time = time.time() + 31
        with patch("tool_classifier.api_caller.time.time", return_value=future_time):
            assert cb.can_execute(_URL) is True
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

    def test_half_open_to_closed_on_success(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        cb.record_failure(_URL)
        cb.can_execute(_URL)  # Cooldown elapsed → HALF_OPEN
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

        cb.record_success(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED

    def test_half_open_to_open_on_failure(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        cb.record_failure(_URL)
        cb.can_execute(_URL)  # Cooldown elapsed → HALF_OPEN
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

    def test_independent_breakers_per_url(self) -> None:
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=60.0)
        cb.record_failure(_URL)

        assert cb.get_state(_URL) == CB_STATE_OPEN
        assert cb.get_state(_URL_B) == CB_STATE_CLOSED

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure(_URL)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED

        cb.record_success(_URL)

        # The failure count was reset; two more failures should NOT open the breaker
        cb.record_failure(_URL)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_without_making_http_call(self) -> None:
        """Verify no HTTP request is issued when the circuit breaker is OPEN."""
        caller = APICaller(failure_threshold=1)
        mock_client = _make_client(_make_response(500))

        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=mock_client
        ) as mock_class:
            await caller.call(_URL, "GET", {})  # Trip the breaker
            assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN
            mock_class.reset_mock()

            result = await caller.call(_URL, "GET", {})  # Should be rejected

        mock_class.assert_not_called()
        assert result.success is False
        assert result.error == CIRCUIT_BREAKER_OPEN_MESSAGES["et"]

    @pytest.mark.asyncio
    async def test_independent_breakers_per_url_on_full_caller(self) -> None:
        caller = APICaller(failure_threshold=1)

        # Fail url_a once → open its breaker
        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient",
            return_value=_make_client(_make_response(500)),
        ):
            await caller.call(_URL, "GET", {})
        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN

        # url_b should still work
        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient",
            return_value=_make_client(_make_response(200, json_data={"data": "ok"})),
        ):
            result_b = await caller.call(_URL_B, "GET", {})
        assert result_b.success is True
        assert caller._circuit_breaker.get_state(_URL_B) == CB_STATE_CLOSED


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_invalid_method_put_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            await APICaller().call(_URL, "PUT", {})

    @pytest.mark.asyncio
    async def test_invalid_method_delete_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported HTTP method"):
            await APICaller().call(_URL, "DELETE", {})

    def test_api_call_result_is_client_error_true_for_4xx(self) -> None:
        result = APICallResult(
            success=False, status_code=404, response_data="", error="Not found"
        )
        assert result.is_client_error is True
        assert result.is_server_error is False

    def test_api_call_result_is_server_error_true_for_5xx(self) -> None:
        result = APICallResult(
            success=False, status_code=503, response_data="", error="error"
        )
        assert result.is_server_error is True
        assert result.is_client_error is False

    def test_api_call_result_neither_client_nor_server_error_on_success(self) -> None:
        result = APICallResult(
            success=True, status_code=200, response_data={"ok": True}, error=None
        )
        assert result.is_client_error is False
        assert result.is_server_error is False

    def test_api_call_result_status_code_zero_is_not_client_or_server_error(
        self,
    ) -> None:
        result = APICallResult(
            success=False, status_code=0, response_data="", error="timeout"
        )
        assert result.is_client_error is False
        assert result.is_server_error is False


# ---------------------------------------------------------------------------
# CircuitBreaker edge cases
# ---------------------------------------------------------------------------


class TestCircuitBreakerEdgeCases:
    def test_exact_threshold_crossing_opens_on_third_failure(self) -> None:
        """failure_threshold=3 → CLOSED after 2 failures, OPEN on the 3rd."""
        cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED  # 1st failure → still CLOSED

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED  # 2nd failure → still CLOSED

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN  # 3rd failure → OPEN

    def test_half_open_recovery_closes_breaker(self) -> None:
        """HALF_OPEN → record_success() → CLOSED with failure count reset."""
        cb = CircuitBreaker(failure_threshold=2, cooldown_seconds=0.0)

        cb.record_failure(_URL)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

        # Cooldown elapsed (0.0 s) → probe allowed → HALF_OPEN
        cb.can_execute(_URL)
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

        cb.record_success(_URL)
        assert cb.get_state(_URL) == CB_STATE_CLOSED
        # Verify state is fully reset
        assert cb._get_state(_URL).failure_count == 0

    def test_half_open_re_failure_reopens_breaker(self) -> None:
        """HALF_OPEN → record_failure() → OPEN again."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

        cb.can_execute(_URL)  # Cooldown elapsed → HALF_OPEN
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

    def test_cooldown_timing_keeps_open_until_elapsed(self) -> None:
        """OPEN breaker stays OPEN when cooldown has NOT yet elapsed."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=999.0)
        cb.record_failure(_URL)
        assert cb.get_state(_URL) == CB_STATE_OPEN

        # Time has NOT advanced → still OPEN
        assert cb.can_execute(_URL) is False
        assert cb.get_state(_URL) == CB_STATE_OPEN

    def test_half_open_probe_in_flight_blocks_concurrent_callers(self) -> None:
        """Only one probe is allowed through HALF_OPEN; second call is blocked."""
        cb = CircuitBreaker(failure_threshold=1, cooldown_seconds=0.0)
        cb.record_failure(_URL)

        # First probe allowed → HALF_OPEN with probe_in_flight=True
        first = cb.can_execute(_URL)
        assert first is True
        assert cb.get_state(_URL) == CB_STATE_HALF_OPEN

        # Second call blocked while probe in flight
        second = cb.can_execute(_URL)
        assert second is False

    @pytest.mark.asyncio
    async def test_full_caller_recovery_after_cooldown(self) -> None:
        """APICaller: fail → OPEN → cooldown → probe succeeds → CLOSED."""
        caller = APICaller(failure_threshold=1, cooldown_seconds=0.0)
        fail_client = _make_client(_make_response(500))
        success_client = _make_client(_make_response(200, json_data={"ok": True}))

        # Trip the breaker
        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=fail_client
        ):
            await caller.call(_URL, "GET", {})
        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_OPEN

        # After 0-second cooldown, probe is allowed
        with patch(
            "tool_classifier.api_caller.httpx.AsyncClient", return_value=success_client
        ):
            result = await caller.call(_URL, "GET", {})

        assert result.success is True
        assert caller._circuit_breaker.get_state(_URL) == CB_STATE_CLOSED

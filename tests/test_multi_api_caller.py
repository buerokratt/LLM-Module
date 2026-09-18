"""Unit tests for MultiAPICaller — parallel batch API execution."""

import asyncio

import pytest

from tool_classifier.api_caller import APICaller
from tool_classifier.constants import (
    CIRCUIT_BREAKER_OPEN_MESSAGES,
    MULTI_API_BATCH_TIMEOUT,
    MULTI_API_PARTIAL_FAILURE_MESSAGES,
    SERVICE_TIMEOUT_MESSAGES,
    SERVICE_UNAVAILABLE_MESSAGES,
)
from tool_classifier.models import APICallResult
from tool_classifier.multi_api_caller import MultiAPICaller


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_WEATHER_URL = "https://publicapi.envir.ee/v1/combinedWeatherData"
_HOLIDAYS_URL = "https://openholidaysapi.org/PublicHolidays"

_EP_A = {"url": _WEATHER_URL, "method": "GET", "call_params": {"station": "Tallinn"}}
_EP_B = {
    "url": _HOLIDAYS_URL,
    "method": "GET",
    "call_params": {
        "countryIsoCode": "EE",
        "validFrom": "2026-01-01",
        "validTo": "2026-12-31",
    },
}
_EP_C = {"url": _WEATHER_URL, "method": "GET", "call_params": {}}

_WEATHER_PAYLOAD = {
    "observations": {
        "station": [
            {
                "name": "Tallinn",
                "phenomenon": "Cloudy",
                "airtemperature": 12.5,
                "windspeed": 4.2,
                "relativehumidity": 78,
            }
        ]
    }
}
_HOLIDAYS_PAYLOAD = [
    {
        "startDate": "2026-01-01",
        "endDate": "2026-01-01",
        "type": "Public",
        "name": [{"language": "ET", "text": "Uusaasta"}],
        "nationwide": True,
    },
    {
        "startDate": "2026-02-24",
        "endDate": "2026-02-24",
        "type": "Public",
        "name": [{"language": "ET", "text": "Eesti Vabariigi aastapäev"}],
        "nationwide": True,
    },
]


def _ok(url: str = _WEATHER_URL) -> APICallResult:
    payload: object = _HOLIDAYS_PAYLOAD if url == _HOLIDAYS_URL else _WEATHER_PAYLOAD
    return APICallResult(
        success=True, status_code=200, response_data=payload, error=None
    )


def _fail(status_code: int = 500, language: str = "en") -> APICallResult:
    return APICallResult(
        success=False,
        status_code=status_code,
        response_data="",
        error=SERVICE_UNAVAILABLE_MESSAGES[language],
    )


def _timeout_result(language: str = "en") -> APICallResult:
    return APICallResult(
        success=False,
        status_code=0,
        response_data="",
        error=SERVICE_TIMEOUT_MESSAGES[language],
    )


def _cb_open_result(language: str = "en") -> APICallResult:
    return APICallResult(
        success=False,
        status_code=0,
        response_data="",
        error=CIRCUIT_BREAKER_OPEN_MESSAGES[language],
    )


def _make_caller_with_results(results_by_url: dict[str, APICallResult]) -> APICaller:
    """Return an APICaller whose .call() returns the mapped result per URL."""
    caller = APICaller()

    async def _call(
        url: str, method: str, params: dict, language: str = "et"
    ) -> APICallResult:  # noqa: ANN001
        return results_by_url[url]

    caller.call = _call  # type: ignore[method-assign]
    return caller


# ---------------------------------------------------------------------------
# All calls succeed
# ---------------------------------------------------------------------------


class TestAllSucceed:
    @pytest.mark.asyncio
    async def test_all_succeeded_true(self) -> None:
        caller = _make_caller_with_results(
            {_EP_A["url"]: _ok(_EP_A["url"]), _EP_B["url"]: _ok(_EP_B["url"])}
        )
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert result.all_succeeded is True
        assert len(result.results) == 2
        assert len(result.endpoints) == 2
        assert result.results[0].success is True
        assert result.results[1].success is True

    @pytest.mark.asyncio
    async def test_order_preserved(self) -> None:
        ok_a = APICallResult(
            success=True, status_code=200, response_data=_WEATHER_PAYLOAD, error=None
        )
        ok_b = APICallResult(
            success=True, status_code=200, response_data=_HOLIDAYS_PAYLOAD, error=None
        )
        caller = _make_caller_with_results({_EP_A["url"]: ok_a, _EP_B["url"]: ok_b})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert result.results[0].response_data == _WEATHER_PAYLOAD
        assert result.results[1].response_data == _HOLIDAYS_PAYLOAD

    @pytest.mark.asyncio
    async def test_successful_results_property(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _ok(), _EP_B["url"]: _ok()})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert len(result.successful_results) == 2
        assert len(result.failed_results) == 0


# ---------------------------------------------------------------------------
# Partial failure (one call fails)
# ---------------------------------------------------------------------------


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_one_fails_all_succeeded_false(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _ok(), _EP_B["url"]: _fail()})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert result.all_succeeded is False
        assert result.results[0].success is True
        assert result.results[1].success is False

    @pytest.mark.asyncio
    async def test_successful_and_failed_result_helpers(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _ok(), _EP_B["url"]: _fail()})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert len(result.successful_results) == 1
        assert len(result.failed_results) == 1
        ep_ok, res_ok = result.successful_results[0]
        assert ep_ok == _EP_A
        assert res_ok.success is True
        ep_fail, res_fail = result.failed_results[0]
        assert ep_fail == _EP_B
        assert res_fail.success is False

    @pytest.mark.asyncio
    async def test_5xx_error_kept_in_results(self) -> None:
        caller = _make_caller_with_results(
            {
                _EP_A["url"]: _ok(),
                _EP_B["url"]: _fail(500),
                _EP_C["url"]: _ok(),
            }
        )
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B, _EP_C], language="en")

        assert len(result.results) == 3
        assert result.results[0].success is True
        assert result.results[1].success is False
        assert result.results[1].status_code == 500
        assert result.results[2].success is True


# ---------------------------------------------------------------------------
# Individual call timeout
# ---------------------------------------------------------------------------


class TestIndividualTimeout:
    @pytest.mark.asyncio
    async def test_individual_timeout_produces_failure_result(self) -> None:
        caller = _make_caller_with_results(
            {_EP_A["url"]: _ok(), _EP_B["url"]: _timeout_result()}
        )
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert result.all_succeeded is False
        assert result.results[1].success is False
        assert result.results[1].status_code == 0


# ---------------------------------------------------------------------------
# Circuit breaker open for one URL
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpen:
    @pytest.mark.asyncio
    async def test_cb_open_url_gets_failure_others_succeed(self) -> None:
        caller = _make_caller_with_results(
            {_EP_A["url"]: _ok(), _EP_B["url"]: _cb_open_result()}
        )
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert result.all_succeeded is False
        assert result.results[0].success is True
        assert result.results[1].success is False
        assert result.results[1].status_code == 0

    @pytest.mark.asyncio
    async def test_cb_open_message_correct_language_et(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _cb_open_result("et")})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A], language="et")

        assert result.results[0].error == CIRCUIT_BREAKER_OPEN_MESSAGES["et"]


# ---------------------------------------------------------------------------
# Batch-level timeout
# ---------------------------------------------------------------------------


class TestBatchTimeout:
    @pytest.mark.asyncio
    async def test_batch_timeout_cancels_pending_returns_partial(self) -> None:
        """Batch timeout fires; completed tasks are preserved, pending get failure result."""

        async def _slow_call(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            await asyncio.sleep(10)  # never finishes in the test
            return _ok(url)

        async def _fast_call(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            return _ok(url)

        caller = APICaller()

        async def _dispatched(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            if url == _EP_A["url"]:
                return await _fast_call(url, method, params, language)
            return await _slow_call(url, method, params, language)

        caller.call = _dispatched  # type: ignore[method-assign]

        multi = MultiAPICaller(caller, batch_timeout=1)  # 1-second batch timeout
        result = await multi.call_all([_EP_A, _EP_B], language="en")

        assert len(result.results) == 2
        # EP_A completed fast; EP_B was cancelled → failure
        assert result.results[0].success is True
        assert result.results[1].success is False
        assert result.results[1].status_code == 0
        assert result.results[1].error == MULTI_API_PARTIAL_FAILURE_MESSAGES["en"]

    @pytest.mark.asyncio
    async def test_batch_timeout_all_cancelled(self) -> None:
        """When both tasks are slow, both get failure results."""

        async def _slow(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            await asyncio.sleep(10)
            return _ok(url)

        caller = APICaller()
        caller.call = _slow  # type: ignore[method-assign]

        multi = MultiAPICaller(caller, batch_timeout=1)
        result = await multi.call_all([_EP_A, _EP_B], language="ru")

        assert len(result.results) == 2
        assert not any(r.success for r in result.results)
        for r in result.results:
            assert r.error == MULTI_API_PARTIAL_FAILURE_MESSAGES["ru"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_endpoints_returns_empty_result(self) -> None:
        caller = APICaller()
        multi = MultiAPICaller(caller)
        result = await multi.call_all([], language="en")

        assert result.results == []
        assert result.endpoints == []
        assert result.all_succeeded is True
        assert result.successful_results == []
        assert result.failed_results == []

    @pytest.mark.asyncio
    async def test_single_endpoint_success(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _ok()})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A], language="en")

        assert len(result.results) == 1
        assert result.all_succeeded is True

    @pytest.mark.asyncio
    async def test_single_endpoint_failure(self) -> None:
        caller = _make_caller_with_results({_EP_A["url"]: _fail()})
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A], language="en")

        assert len(result.results) == 1
        assert result.all_succeeded is False

    @pytest.mark.asyncio
    async def test_exception_in_gather_converted_to_failure(self) -> None:
        """A BaseException raised inside a task is caught and converted to APICallResult(success=False)."""

        async def _raises(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            raise RuntimeError("unexpected internal error")

        caller = APICaller()
        caller.call = _raises  # type: ignore[method-assign]

        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A], language="en")

        assert len(result.results) == 1
        assert result.results[0].success is False
        assert result.results[0].status_code == 0
        assert result.results[0].error == MULTI_API_PARTIAL_FAILURE_MESSAGES["en"]


# ---------------------------------------------------------------------------
# Per-URL circuit breaker isolation
# ---------------------------------------------------------------------------


class TestCircuitBreakerIsolation:
    @pytest.mark.asyncio
    async def test_cb_open_for_one_url_does_not_affect_others(self) -> None:
        """Three endpoints; CB open only for B — A and C still execute normally."""
        caller = _make_caller_with_results(
            {
                _EP_A["url"]: _ok(),
                _EP_B["url"]: _cb_open_result(),
                _EP_C["url"]: _ok(),
            }
        )
        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A, _EP_B, _EP_C], language="en")

        assert result.results[0].success is True
        assert result.results[1].success is False
        assert result.results[2].success is True
        assert len(result.failed_results) == 1
        assert len(result.successful_results) == 2


# ---------------------------------------------------------------------------
# Multilingual error messages
# ---------------------------------------------------------------------------


class TestMultilingualMessages:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["et", "ru", "en"])
    async def test_partial_failure_message_correct_language(
        self, language: str
    ) -> None:
        """Batch timeout fills pending slots with the right language's message."""

        async def _slow(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            await asyncio.sleep(10)
            return _ok(url)

        caller = APICaller()
        caller.call = _slow  # type: ignore[method-assign]

        multi = MultiAPICaller(caller, batch_timeout=1)
        result = await multi.call_all([_EP_A], language=language)

        assert result.results[0].error == MULTI_API_PARTIAL_FAILURE_MESSAGES[language]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["et", "ru", "en"])
    async def test_exception_message_correct_language(self, language: str) -> None:
        async def _raises(
            url: str, method: str, params: dict, language: str = "et"
        ) -> APICallResult:
            raise ValueError("boom")

        caller = APICaller()
        caller.call = _raises  # type: ignore[method-assign]

        multi = MultiAPICaller(caller)
        result = await multi.call_all([_EP_A], language=language)

        assert result.results[0].error == MULTI_API_PARTIAL_FAILURE_MESSAGES[language]

    @pytest.mark.asyncio
    async def test_default_batch_timeout_constant_used(self) -> None:
        multi = MultiAPICaller(APICaller())
        assert multi._batch_timeout == MULTI_API_BATCH_TIMEOUT

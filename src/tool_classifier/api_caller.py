"""API Caller module for executing external HTTP requests with circuit breaker protection."""

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from src.loki_logger import LokiLogger
from src.utils.error_utils import generate_error_id

from llm_orchestrator_config.llm_ochestrator_constants import get_localized_message
from tool_classifier.constants import (
    API_CALL_TIMEOUT,
    CB_STATE_CLOSED,
    CB_STATE_HALF_OPEN,
    CB_STATE_OPEN,
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_OPEN_MESSAGES,
    CLIENT_ERROR_MESSAGES,
    REDIRECT_NOT_FOLLOWED_MESSAGES,
    SERVICE_TIMEOUT_MESSAGES,
    SERVICE_UNAVAILABLE_MESSAGES,
)
from tool_classifier.models import APICallResult

logger = LokiLogger(service_name="api-tool-calling")


@dataclass
class _BreakerState:
    """Internal per-URL circuit breaker state."""

    state: str = CB_STATE_CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    probe_in_flight: bool = False


class CircuitBreaker:
    """
    Per-URL circuit breaker that prevents repeated calls to a failing external API.

    State machine transitions:
        CLOSED    → OPEN:      after ``failure_threshold`` consecutive server/network failures.
        OPEN      → HALF_OPEN: once ``cooldown_seconds`` have elapsed since the last failure.
        HALF_OPEN → CLOSED:    on the next successful probe call.
        HALF_OPEN → OPEN:      on the next failed probe call.

    Each URL maintains its own independent breaker so that one failing API does not
    prevent calls to other URLs.

    4xx (client error) responses do **not** count as failures — they indicate bad
    input rather than a server outage and should trigger agentic loop re-prompting
    instead of circuit protection.
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        cooldown_seconds: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._breakers: dict[str, _BreakerState] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, url: str) -> _BreakerState:
        """Return the breaker state for *url*, creating it on first access."""
        if url not in self._breakers:
            self._breakers[url] = _BreakerState()
        return self._breakers[url]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def can_execute(self, url: str) -> bool:
        """Return True if a request to *url* is currently allowed."""
        breaker = self._get_state(url)
        if breaker.state == CB_STATE_CLOSED:
            return True
        if breaker.state == CB_STATE_OPEN:
            if time.time() - breaker.last_failure_time >= self._cooldown_seconds:
                breaker.state = CB_STATE_HALF_OPEN
                breaker.probe_in_flight = True
                logger.info(
                    f"CircuitBreaker: circuit half-open | event_type=circuit_breaker_half_open url={url}"
                )
                return True
            return False
        # HALF_OPEN: allow exactly one probe request through; gate subsequent
        # concurrent callers until record_success() / record_failure() resolves it.
        if breaker.probe_in_flight:
            return False
        breaker.probe_in_flight = True
        return True

    def record_success(self, url: str) -> None:
        """Reset the breaker for *url* to CLOSED after a successful call."""
        breaker = self._get_state(url)
        if breaker.state != CB_STATE_CLOSED:
            logger.info(
                f"CircuitBreaker: circuit recovered | event_type=circuit_breaker_recovered url={url} failure_count={breaker.failure_count}"
            )
        breaker.state = CB_STATE_CLOSED
        breaker.failure_count = 0
        breaker.probe_in_flight = False

    def record_failure(self, url: str) -> None:
        """Record a server/network failure for *url*.

        Opens the circuit breaker if the failure threshold is reached.
        """
        breaker = self._get_state(url)
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        breaker.probe_in_flight = False
        if breaker.failure_count >= self._failure_threshold:
            if breaker.state != CB_STATE_OPEN:
                logger.warning(
                    f"CircuitBreaker: circuit opened | event_type=circuit_breaker_opened url={url} failure_count={breaker.failure_count} threshold={self._failure_threshold}"
                )
            breaker.state = CB_STATE_OPEN

    def get_state(self, url: str) -> str:
        """Return the current circuit breaker state string for *url*."""
        return self._get_state(url).state


class APICaller:
    """
    Executes external HTTP requests with collected parameters and circuit breaker protection.

    - GET requests map *params* to URL query parameters.
    - POST requests map *params* to the JSON request body.

    A per-URL :class:`CircuitBreaker` prevents hammering a failing API: after
    ``failure_threshold`` consecutive server-side or network errors the circuit
    opens and all subsequent calls for that URL are rejected immediately until
    ``cooldown_seconds`` have elapsed.

    4xx responses do **not** count towards the failure threshold because they
    typically indicate bad user input that should be corrected via the agentic
    loop rather than a temporary server outage.
    """

    def __init__(
        self,
        timeout: int = API_CALL_TIMEOUT,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        cooldown_seconds: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    ) -> None:
        self._default_timeout = timeout
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
        )

    async def call(
        self,
        url: str,
        method: str,
        params: dict[str, Any],
        timeout: int | None = None,
        language: str = "et",
    ) -> APICallResult:
        """Execute an HTTP request and return the structured result.

        Args:
            url: Full URL of the external API endpoint.
            method: HTTP method — must be ``"GET"`` or ``"POST"`` (case-insensitive).
            params: Parameters to send. GET → query string; POST → JSON body.
            timeout: Per-call timeout override in seconds. Defaults to the instance
                default (``API_CALL_TIMEOUT``).
            language: BCP-47 language code for user-facing error messages
                (``"et"``, ``"ru"``, ``"en"``). Defaults to Estonian (``"et"``).

        Returns:
            :class:`~tool_classifier.models.APICallResult` describing the outcome.

        Raises:
            ValueError: If *method* is not ``"GET"`` or ``"POST"``.
        """
        method_upper = method.upper()
        if method_upper not in ("GET", "POST"):
            raise ValueError(
                f"Unsupported HTTP method: {method!r}. Only GET and POST are allowed."
            )

        if not self._circuit_breaker.can_execute(url):
            logger.warning(
                f"APICaller: api call rejected | event_type=api_call_rejected_circuit_open url={url} method={method_upper}"
            )
            return APICallResult(
                success=False,
                status_code=0,
                response_data="",
                error=get_localized_message(CIRCUIT_BREAKER_OPEN_MESSAGES, language),
            )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        logger.debug(
            f"APICaller: api call started | event_type=api_call_started url={url} method={method_upper} timeout={effective_timeout}"
        )
        _t0 = time.time()
        try:
            async with httpx.AsyncClient(
                timeout=effective_timeout, follow_redirects=True
            ) as client:
                if method_upper == "POST":
                    response = await client.post(url, json=params)
                else:
                    response = await client.get(url, params=params)
                result = self._handle_response(response, url, language)
                _duration_ms = round((time.time() - _t0) * 1000, 1)
                if result.success:
                    logger.info(
                        f"APICaller: api call success | event_type=api_call_success url={url} method={method_upper} status_code={result.status_code} duration_ms={_duration_ms}"
                    )
                return result

        except httpx.TimeoutException as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            _error_id = generate_error_id()
            logger.error(
                f"APICaller: api call timeout | event_type=api_call_timeout url={url} method={method_upper} error_id={_error_id} duration_ms={_duration_ms} exc={exc!r}"
            )
            self._circuit_breaker.record_failure(url)
            return APICallResult(
                success=False,
                status_code=0,
                response_data="",
                error=get_localized_message(SERVICE_TIMEOUT_MESSAGES, language),
            )

        except httpx.RequestError as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            _error_id = generate_error_id()
            logger.error(
                f"APICaller: api call network error | event_type=api_call_network_error url={url} method={method_upper} error_id={_error_id} duration_ms={_duration_ms} exc={exc!r}"
            )
            self._circuit_breaker.record_failure(url)
            return APICallResult(
                success=False,
                status_code=0,
                response_data="",
                error=get_localized_message(SERVICE_TIMEOUT_MESSAGES, language),
            )

    def _handle_response(
        self,
        response: httpx.Response,
        url: str,
        language: str,
    ) -> APICallResult:
        """Parse an HTTP response into an :class:`~tool_classifier.models.APICallResult`."""
        status_code = response.status_code

        if 200 <= status_code < 300:
            self._circuit_breaker.record_success(url)
            return APICallResult(
                success=True,
                status_code=status_code,
                response_data=self._parse_response_body(response),
                error=None,
            )

        if 300 <= status_code < 400:
            # Redirect not followed (e.g. redirect limit exceeded before this point).
            # Not a server fault — do NOT trip the circuit breaker.
            location = response.headers.get("location", "")
            logger.warning(
                f"APICaller: api call redirect | event_type=api_call_redirect url={url} status_code={status_code} location={location!r}"
            )
            base_msg = get_localized_message(REDIRECT_NOT_FOLLOWED_MESSAGES, language)
            error_msg = base_msg.format(
                status_code=status_code, location=location or "unknown"
            )
            return APICallResult(
                success=False,
                status_code=status_code,
                response_data="",
                error=error_msg,
            )

        if 400 <= status_code < 500:
            # Client error — preserve the full raw body in response_data for
            # potential future agentic loop re-prompting, but surface a localized
            # friendly message to the user via the error field.
            # 4xx does NOT trip the circuit breaker.
            error_body = self._parse_response_body(response)
            raw_msg = error_body if isinstance(error_body, str) else str(error_body)
            logger.warning(
                f"APICaller: api call client error | event_type=api_call_client_error url={url} status_code={status_code} error_preview={raw_msg[:200]!r}"
            )
            return APICallResult(
                success=False,
                status_code=status_code,
                response_data=error_body,
                error=get_localized_message(CLIENT_ERROR_MESSAGES, language),
            )

        # 5xx — server is misbehaving; trip the circuit breaker.
        logger.error(
            f"APICaller: api call server error | event_type=api_call_server_error url={url} status_code={status_code} error_id={generate_error_id()}"
        )
        self._circuit_breaker.record_failure(url)
        return APICallResult(
            success=False,
            status_code=status_code,
            response_data="",
            error=get_localized_message(SERVICE_UNAVAILABLE_MESSAGES, language),
        )

    @staticmethod
    def _parse_response_body(
        response: httpx.Response,
    ) -> dict[str, object] | list[object] | str:
        """Attempt to parse the response body as JSON; fall back to raw text."""
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text

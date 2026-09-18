"""MultiAPICaller — concurrent batch execution of multiple API endpoints."""

import asyncio
import time
from typing import Any

from src.loki_logger import LokiLogger
from src.utils.error_utils import generate_error_id

from tool_classifier.api_caller import APICaller
from tool_classifier.constants import (
    MULTI_API_BATCH_TIMEOUT,
    MULTI_API_PARTIAL_FAILURE_MESSAGES,
)
from tool_classifier.models import APICallResult, MultiAPICallResult
from llm_orchestrator_config.llm_ochestrator_constants import get_localized_message

logger = LokiLogger(service_name="api-tool-calling")


class MultiAPICaller:
    """
    Executes multiple API endpoint calls concurrently using :func:`asyncio.gather`.

    Reuses the caller's :class:`~tool_classifier.api_caller.APICaller` instance so
    that per-URL circuit breaker state is shared across single and batch invocations.

    A batch-level timeout caps total wall-clock time.  When the timeout fires, any
    still-pending tasks are cancelled and their slots are filled with a failure result,
    ensuring the caller always receives a fully populated :class:`MultiAPICallResult`.

    Args:
        api_caller: Shared :class:`~tool_classifier.api_caller.APICaller` instance.
            Circuit breaker state is inherited from this instance.
        batch_timeout: Maximum seconds to wait for the entire batch. Defaults to
            :data:`~tool_classifier.constants.MULTI_API_BATCH_TIMEOUT`.
    """

    def __init__(
        self,
        api_caller: APICaller,
        batch_timeout: int = MULTI_API_BATCH_TIMEOUT,
    ) -> None:
        self._api_caller = api_caller
        self._batch_timeout = batch_timeout

    async def call_all(
        self,
        endpoints: list[dict[str, Any]],
        language: str = "et",
    ) -> MultiAPICallResult:
        """Execute all *endpoints* concurrently and return a consolidated result.

        Each endpoint dict must contain at minimum:
            - ``"url"`` (str): Full URL for the request.
            - ``"method"`` (str): HTTP method — ``"GET"`` or ``"POST"``.
            - ``"call_params"`` (dict): Query / body parameters to send.
              This key is intentionally distinct from the ``"params"`` key on
              endpoint schema dicts (which holds a *list* of parameter
              descriptors used during parameter collection), to prevent the
              schema list from being forwarded to the HTTP call.

        Results are returned in the same order as *endpoints*.  Failed tasks
        (exceptions or batch timeout) produce ``APICallResult(success=False, …)``.

        Args:
            endpoints: List of endpoint descriptor dicts.
            language: BCP-47 language code for user-facing error messages.

        Returns:
            :class:`~tool_classifier.models.MultiAPICallResult` with one result per
            endpoint.
        """
        if not endpoints:
            return MultiAPICallResult(results=[], endpoints=[])

        logger.info(
            f"MultiAPICaller: batch started | event_type=multi_api_batch_started"
            f" endpoint_count={len(endpoints)} batch_timeout={self._batch_timeout}"
        )
        _t0 = time.time()

        tasks: list[asyncio.Task[APICallResult]] = [
            asyncio.create_task(
                self._api_caller.call(
                    url=ep["url"],
                    method=ep["method"],
                    params=ep.get("call_params", {}),
                    language=language,
                ),
                name=f"multi_api_{i}",
            )
            for i, ep in enumerate(endpoints)
        ]

        results: list[APICallResult]
        try:
            raw = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self._batch_timeout,
            )
            results = [self._coerce_result(item, language) for item in raw]

        except asyncio.TimeoutError:
            pending = [t for t in tasks if not t.done()]
            logger.warning(
                f"MultiAPICaller: batch timeout | event_type=multi_api_batch_timeout"
                f" pending_count={len(pending)} batch_timeout={self._batch_timeout}"
            )
            for task in pending:
                task.cancel()
            # Await cancelled tasks so their coroutines can complete cleanup
            # (e.g. close HTTP connections) before we return.
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            results = []
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    outcome: APICallResult | BaseException = (
                        exc if exc is not None else task.result()
                    )
                    results.append(self._coerce_result(outcome, language))
                else:
                    results.append(
                        APICallResult(
                            success=False,
                            status_code=0,
                            response_data="",
                            error=get_localized_message(
                                MULTI_API_PARTIAL_FAILURE_MESSAGES, language
                            ),
                        )
                    )

        had_failure = any(not r.success for r in results)
        if had_failure:
            logger.warning(
                f"MultiAPICaller: partial failure | event_type=multi_api_partial_failure"
                f" failed_count={sum(not r.success for r in results)} total_count={len(results)}"
            )

        _duration_ms = round((time.time() - _t0) * 1000, 1)
        logger.info(
            f"MultiAPICaller: batch completed | event_type=multi_api_batch_completed"
            f" endpoint_count={len(endpoints)}"
            f" success_count={sum(r.success for r in results)}"
            f" failed_count={sum(not r.success for r in results)}"
            f" duration_ms={_duration_ms}"
        )

        return MultiAPICallResult(results=results, endpoints=endpoints)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_result(
        self,
        item: APICallResult | BaseException | None,
        language: str,
    ) -> APICallResult:
        """Convert a raw gather() output item to an :class:`APICallResult`.

        :func:`asyncio.gather` with ``return_exceptions=True`` returns either the
        coroutine's return value or the exception that was raised.  We identify
        exceptions by checking against :class:`BaseException` rather than the
        positive type, which avoids issues with Python's dual import paths
        (``tool_classifier.models`` vs ``src.tool_classifier.models``).
        """
        if not isinstance(item, BaseException):
            # Assume any non-exception value is already an APICallResult.
            return item  # type: ignore[return-value]
        error_id = generate_error_id()
        exc_type = type(item).__name__
        exc_msg = str(item)
        logger.error(
            f"MultiAPICaller: task exception | event_type=multi_api_task_exception"
            f" error_id={error_id} exc_type={exc_type} exc_msg={exc_msg!r}"
        )
        return APICallResult(
            success=False,
            status_code=0,
            response_data="",
            error=get_localized_message(MULTI_API_PARTIAL_FAILURE_MESSAGES, language),
        )

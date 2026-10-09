"""Shared request preamble for both orchestration entry points (Step 8).

``process_orchestration_request`` (blocking) and ``stream_orchestration_response``
(SSE) ran ~80 lines of near-identical gate-keeping before reaching the
classifier. That logic lives here once.

The two entry points differ irreducibly in *how* they emit an early exit — one
returns a response model, the other yields SSE frames. So these helpers decide
**whether** to stop and **what to say**, returning a :class:`PreambleBlock`
verdict; each entry point renders it in its own idiom.

The gates do not form one contiguous run: the ``#service`` prefix check sits
between them and genuinely differs per entry point (``execute_direct_step`` vs
``execute_direct_step_streaming``). Hence two methods, called around it:

    lang, block = preamble.run_early(request, time_metric, is_streaming=...)
    if block: <emit and stop>
    # ... #service prefix branch, inline per entry point ...
    block = preamble.run_late(request, time_metric, lang, is_streaming=...)
    if block: <emit and stop>
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from src.loki_logger import LokiLogger
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationResponse,
)
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    QUERY_VALIDATION_FAILED_MESSAGES,
    TEST_DEPLOYMENT_ENVIRONMENT,
    get_localized_message,
)
from src.utils.guardrail_followup import get_repeated_violation_message
from src.utils.language_detector import detect_language, get_language_name
from src.utils.query_validator import validate_query_basic

logger = LokiLogger(service_name="llm-orchestration-service")


@dataclass(frozen=True)
class PreambleBlock:
    """A decision to stop before the classifier, plus what to tell the user.

    Attributes:
        message: User-facing content to emit.
        llm_service_active: Value for the response's ``llmServiceActive`` flag.
        input_guard_failed: Value for the response's ``inputGuardFailed`` flag.
        log_timings: Whether this path logs step timings before exiting. Only
            the repeated-violation path does — matching prior behaviour on both
            entry points.
    """

    message: str
    llm_service_active: bool = True
    input_guard_failed: bool = False
    log_timings: bool = False


def build_block_response(
    request: OrchestrationRequest, block: PreambleBlock
) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
    """Render a :class:`PreambleBlock` as the correct response type.

    Collapses the ``if TEST: TestOrchestrationResponse else: OrchestrationResponse``
    branch that was repeated at every early-exit in the blocking entry point.
    """
    if request.environment == TEST_DEPLOYMENT_ENVIRONMENT:
        return TestOrchestrationResponse(
            llmServiceActive=block.llm_service_active,
            questionOutOfLLMScope=False,
            inputGuardFailed=block.input_guard_failed,
            content=block.message,
            chunks=None,
        )
    return OrchestrationResponse(
        chatId=request.chatId,
        llmServiceActive=block.llm_service_active,
        questionOutOfLLMScope=False,
        inputGuardFailed=block.input_guard_failed,
        content=block.message,
    )


class RequestPreamble:
    """Runs the pre-classifier gates shared by both entry points."""

    def run_early(
        self,
        request: OrchestrationRequest,
        time_metric: Dict[str, float],
        is_streaming: bool = False,
    ) -> Tuple[str, Optional[PreambleBlock]]:
        """Detect language, then check for a post-violation "why?" follow-up.

        Stores the detected language on the request as ``_detected_language``
        for use throughout the pipeline.

        Returns:
            (detected_language, block_or_None)
        """
        # STEP 0: Detect language from user message (with timing)
        start_time = time.time()
        detected_language = detect_language(request.message)
        language_name = get_language_name(detected_language)
        time_metric["language_detection"] = time.time() - start_time

        prefix = "Streaming request - " if is_streaming else ""
        logger.info(
            f"[{request.chatId}] {prefix}Detected language: "
            f"{language_name} ({detected_language})"
        )

        # setattr for type safety — adds a dynamic attribute to the Pydantic model
        setattr(request, "_detected_language", detected_language)  # noqa: B010

        # STEP 0.15: If the previous turn was blocked by guardrails and this
        # message is a bare "why?" follow-up, repeat the same violation message
        # instead of routing a context-less query through the classifier
        # (blocked turns are never persisted to Redis history, so the
        # client-resent conversationHistory is the source of truth).
        repeated_violation = get_repeated_violation_message(request)
        if repeated_violation:
            stream_note = "Streaming - " if is_streaming else ""
            logger.info(
                f"[{request.chatId}] {stream_note}'why' follow-up after guardrail "
                f"violation - repeating violation message"
            )
            return detected_language, PreambleBlock(
                message=repeated_violation.message,
                llm_service_active=True,
                input_guard_failed=repeated_violation.is_input_violation,
                log_timings=True,
            )

        return detected_language, None

    def run_late(
        self,
        request: OrchestrationRequest,
        time_metric: Dict[str, float],
        detected_language: str,
        is_streaming: bool = False,
    ) -> Optional[PreambleBlock]:
        """Validate the query, then check connection status and budget.

        Returns:
            A :class:`PreambleBlock` if the request must stop, else None.
        """
        # STEP 0.5: Basic query validation (before expensive component init)
        start_time = time.time()
        validation_result = validate_query_basic(request.message)
        time_metric["query_validation"] = time.time() - start_time

        if not validation_result.is_valid:
            stream_note = "Streaming - " if is_streaming else ""
            logger.info(
                f"[{request.chatId}] {stream_note}Query validation failed: "
                f"{validation_result.rejection_reason}"
            )
            return PreambleBlock(
                message=get_localized_message(
                    QUERY_VALIDATION_FAILED_MESSAGES, detected_language
                ),
                llm_service_active=True,
                input_guard_failed=False,
            )

        # STEP 0.6: Connection status & budget pre-validation
        is_allowed, block_message = self._validate_connection(
            request, time_metric, detected_language
        )
        if not is_allowed:
            logger.warning(
                f"[{request.chatId}] Connection/budget validation blocked request"
            )
            return PreambleBlock(
                message=block_message or "",
                llm_service_active=False,
                input_guard_failed=False,
            )

        return None

    def _validate_connection(
        self,
        request: OrchestrationRequest,
        time_metric: Dict[str, float],
        detected_language: str,
    ) -> Tuple[bool, Optional[str]]:
        """Resolve the vault UUID if needed, then run the connection/budget gate."""
        from src.orchestration.cost_budget import validate_connection_before_processing

        vault_uuid: Optional[str] = request.connection_id
        if not vault_uuid and request.environment == "production":
            from src.utils.connection_id_fetcher import get_connection_id_fetcher

            fetcher = get_connection_id_fetcher()
            vault_uuid = fetcher.fetch_vault_uuid_sync("production")

        start_time = time.time()
        result = validate_connection_before_processing(
            vault_uuid=vault_uuid,
            environment=request.environment,
            detected_language=detected_language,
        )
        time_metric["connection_budget_validation"] = time.time() - start_time
        return result

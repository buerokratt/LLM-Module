"""API Tool Calling Workflow Executor — Layer 2 of the classification chain."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    cast,
)

from src.loki_logger import LokiLogger

from llm_orchestrator_config.feature_flags import FeatureFlags
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationResponse,
)
from models.session_models import APIToolSession, EndpointSessionState, LastCallContext
from tool_classifier.agentic_loop import AgenticLoop
from tool_classifier.api_caller import APICaller
from tool_classifier.api_response_formatter import APIResponseFormatterModule
from tool_classifier.base_workflow import BaseWorkflow
from tool_classifier.enums import AgenticLoopStatus, ExecutionMode
from tool_classifier.param_extractor import ParamExtractionModule
from utils.api_tool_session_store import APIToolSessionStore
from utils.conversation_history_helpers import get_conversation_history
from utils.conversation_history_store import ConversationHistoryStore
from utils.atc_cache_store import ATCCacheStore
from tool_classifier.constants import ATC_CACHE_DEFAULT_TTL_SECONDS
from tool_classifier.follow_up_detector import FollowUpDetectorModule
from tool_classifier.multi_agentic_loop import MultiEndpointAgenticLoop
from tool_classifier.multi_api_caller import MultiAPICaller
from tool_classifier.multi_response_formatter import MultiResponseFormatterModule

logger = LokiLogger(service_name="api-tool-calling")

if TYPE_CHECKING:
    from guardrails.nemo_rails_adapter import NeMoRailsAdapter


class OrchestrationServiceProtocol(Protocol):
    """Protocol for orchestration service methods used by this workflow."""

    def format_sse(
        self,
        chat_id: str,
        content: str,
        buttons: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format a payload as an SSE message."""
        ...

    async def handle_output_guardrails(
        self,
        guardrails_adapter: Any,  # noqa: ANN401 — NeMoRailsAdapter, avoids circular import
        generated_response: Union[OrchestrationResponse, TestOrchestrationResponse],
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Check output guardrails and return (possibly replaced) response."""
        ...

    async def store_streaming_inference(
        self,
        request: OrchestrationRequest,
        final_answer: str,
    ) -> None:
        """Store streaming inference data for production/testing environments."""
        ...


@dataclass
class _LoopStep:
    """Shared result type from :meth:`APIToolWorkflowExecutor._compute_loop_step`.

    ``kind`` drives both the sync and streaming execution paths:

    * ``"api_call"``       — single endpoint; all params collected; call API and format.
                            Populates: ``endpoint``, ``collected_params``, ``user_query``.
    * ``"multi_api_call"`` — parallel endpoints; call all APIs concurrently and merge results.
                            Populates: ``parallel_endpoints``, ``user_query``.
                            ``endpoint`` and ``collected_params`` are empty/unused.
    * ``"question"``       — agentic loop needs more input; return ``question`` to user.
                            Populates: ``question``, ``question_tokens``.
    * ``"fallback"``       — nothing to do; caller should fall back to RAG.
                            No additional fields are populated.
    * ``"cached_response"`` — cache hit (L1 or L2); skip APICaller; format ``cached_raw_response``.
                             Populates: ``endpoint``, ``cached_raw_response``, ``collected_params``, ``user_query``, ``cache_source``.
    """

    kind: Literal[
        "api_call", "multi_api_call", "question", "fallback", "cached_response"
    ]
    chat_id: str = ""
    endpoint: Dict[str, Any] = field(default_factory=dict)
    parallel_endpoints: List[EndpointSessionState] = field(default_factory=list)
    collected_params: Dict[str, Any] = field(default_factory=dict)
    detected_language: str = "en"
    user_query: str = ""
    question: str = ""
    question_tokens: List[str] = field(default_factory=list)
    custom_instructions: str = ""
    cached_raw_response: Any = None
    cache_source: str = "L1"


class APIToolWorkflowExecutor(BaseWorkflow):
    """Executes API Tool Calling workflow (Layer 2).

    Handles queries that matched an API endpoint in api_tool_collection.

    On the first turn for a chat_id the matched endpoint is read from context
    (populated by ToolClassifier.classify()).  Subsequent turns resume from the
    persisted Redis session — context["matched_endpoint"] is ignored once a
    session exists.

    The executor manages the agentic loop lifecycle:
      - creates the session on turn 1
      - resumes it on turns 2-N
      - deletes it on COMPLETED or MAX_TURNS_REACHED

    When all required params are collected (COMPLETED) the executor calls the
    external API via :class:`APICaller` and formats the raw response into
    natural-language using :class:`APIResponseFormatterModule`. The formatted
    answer is returned directly to the user.
    """

    def __init__(
        self, orchestration_service: Optional[OrchestrationServiceProtocol] = None
    ) -> None:
        self.orchestration_service = orchestration_service
        self._api_caller = APICaller()
        self._prompt_config_loader = (
            getattr(orchestration_service, "prompt_config_loader", None)
            if orchestration_service is not None
            else None
        )
        self._background_tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Internal helpers

    # ------------------------------------------------------------------

    def _create_background_task(self, coro: Coroutine[Any, Any, None]) -> None:
        """Keep fire-and-forget tasks alive until they finish."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._discard_background_task)

    def _discard_background_task(self, task: asyncio.Task[None]) -> None:
        """Remove a completed background task and log any uncaught exception."""
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.warning(f"APIToolWorkflow: background task failed: {exception}")

    def _get_session_store(self) -> Optional[APIToolSessionStore]:
        """Return the session store from the orchestration service, or None."""
        if self.orchestration_service is None:
            return None
        return getattr(self.orchestration_service, "session_store", None)

    def _get_conversation_history_store(self) -> Optional[ConversationHistoryStore]:
        """Return the conversation history store from the orchestration service, or None."""
        if self.orchestration_service is None:
            return None
        return getattr(self.orchestration_service, "conversation_history_store", None)

    def _get_guardrails_adapter(
        self, environment: str, connection_id: Optional[str] = None
    ) -> Optional["NeMoRailsAdapter"]:
        """Return the NeMoRailsAdapter for *environment*, or None if unavailable."""
        if self.orchestration_service is None:
            return None
        shared = getattr(self.orchestration_service, "shared_guardrails_adapters", {})
        if environment in shared:
            return shared[environment]
        # Fallback: per-request initialisation (slower but safe)
        safe_init = getattr(
            self.orchestration_service, "_safe_initialize_guardrails", None
        )
        if safe_init is not None:
            return safe_init(environment, connection_id)
        return None

    async def _get_custom_instructions(self) -> str:
        """Fetch custom prompt instructions from the loader, or return empty string.

        Mirrors LLMOrchestrationService._get_custom_instructions_for_response_generation.
        The PromptConfigurationLoader has a 5-minute TTL cache, so this is cheap.
        Returns empty string on any failure so existing behaviour is preserved.

        Runs the synchronous requests call in a thread pool via asyncio.to_thread so
        that a cache miss or slow Ruuter response never blocks the event loop.
        """
        if self._prompt_config_loader is None:
            return ""
        try:
            custom_prompt = await asyncio.to_thread(
                self._prompt_config_loader.get_custom_instructions
            )
            return custom_prompt if custom_prompt else ""
        except Exception as e:
            logger.error(f"APIToolWorkflow: failed to fetch custom instructions: {e}")
            return ""

    def _build_agentic_loop(
        self, session_store: Any, custom_instructions: str = ""
    ) -> AgenticLoop:
        """Construct a fresh AgenticLoop for one request."""
        return AgenticLoop(
            session_store=session_store,
            param_extractor=ParamExtractionModule(
                custom_instructions=custom_instructions
            ),
        )

    @staticmethod
    def _language_from_custom_instructions(custom_instructions: str) -> Optional[str]:
        """Detect the directed response language from custom instructions.

        Returns the ISO language code (``'en'``, ``'et'``, or ``'ru'``) when the
        instructions contain a recognisable language directive, or ``None`` when no
        directive is found.

        English is checked first so that a prompt written in Estonian that says
        "respond in English" (or the Estonian equivalent ``"inglise keeles"``) is
        correctly identified as directing English responses.  This ensures that both
        the hardcoded continuation question and the LLM-generated clarifying questions
        use the same directed response language.

        ``'inglise'`` (Estonian for "English") and ``'английск'`` (Russian stem for
        "English") are recognised so prompts written entirely in those languages work.

        Args:
            custom_instructions: The raw custom instructions string from the
                ``PromptConfigurationLoader``.

        Returns:
            ``'en'``, ``'et'``, ``'ru'``, or ``None``.
        """
        if not custom_instructions:
            return None
        lower = custom_instructions.lower()
        # "inglise" = Estonian for "English"; "английск" covers "английский/английском"
        if "english" in lower or "inglise" in lower or "английск" in lower:
            return "en"
        if "estonian" in lower or "eesti" in lower:
            return "et"
        if "russian" in lower or "vene" in lower or "русск" in lower:
            return "ru"
        return None

    @staticmethod
    def _required_params(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [p for p in params if isinstance(p, dict) and p.get("required", False)]

    @staticmethod
    def _missing_required_params(
        schema: List[Dict[str, Any]], collected: Dict[str, Any]
    ) -> List[str]:
        """Return names of required params from *schema* not yet present in *collected*."""
        missing: list[str] = []
        for p in schema:
            if not isinstance(p, dict) or not p.get("required", False):
                continue
            name = p.get("name")
            if not isinstance(name, str) or not name:
                continue
            if name not in collected:
                missing.append(name)
        return missing

    async def _execute_api_and_format(
        self,
        chat_id: str,
        endpoint: Dict[str, Any],
        collected_params: Dict[str, Any],
        user_query: str,
        detected_language: str,
        custom_instructions: str = "",
    ) -> OrchestrationResponse:
        """Call the external API with collected params and return a formatted response.

        On success, the raw API response is converted to natural language by
        :class:`APIResponseFormatterModule`.
        On any failure the localized error message is returned directly to the user.
        """
        url = endpoint.get("url", "")
        method = endpoint.get("method", "GET")
        description = endpoint.get("description", "")

        # L1 cache check — collected_params is complete here, so the hash matches
        # what was stored on the previous successful call with the same params.
        if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED and endpoint.get("cacheable", True):
            _cache_store = ATCCacheStore()
            _cached = await _cache_store.get_l1(
                chat_id, endpoint.get("name", ""), collected_params
            )
            if _cached is not None:
                logger.info(
                    f"[{chat_id}] ATC cache: L1 hit for {endpoint.get('name')!r} "
                    f"— skipping API call"
                )
                return await self._format_cached_response(
                    chat_id=chat_id,
                    endpoint=endpoint,
                    user_query=user_query,
                    detected_language=detected_language,
                    cached_raw_response=_cached,
                    collected_params=collected_params,
                    custom_instructions=custom_instructions,
                    cache_source="L1",
                )

        logger.info(
            f"[{chat_id}] APIToolWorkflow: calling API "
            f"{method} {url} with params={list(collected_params.keys())}"
        )
        api_result = await self._api_caller.call(
            url=url,
            method=method,
            params=collected_params,
            language=detected_language,
        )

        if api_result.success:
            logger.info(
                f"[{chat_id}] APIToolWorkflow: API call succeeded "
                f"(status={api_result.status_code})"
            )
            if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED and endpoint.get(
                "cacheable", True
            ):
                _ep_name = endpoint.get("name", "")
                _ttl_override = endpoint.get("cache_ttl_seconds")
                _ttl = (
                    _ttl_override
                    if isinstance(_ttl_override, int) and _ttl_override > 0
                    else ATC_CACHE_DEFAULT_TTL_SECONDS
                )
                _resp_data = api_result.response_data

                async def _write_l1_l2() -> None:
                    try:
                        _cs = ATCCacheStore()
                        await _cs.set_l1(
                            chat_id, _ep_name, collected_params, _resp_data, _ttl
                        )
                        await _cs.set_l2(
                            chat_id,
                            [
                                LastCallContext(
                                    api_name=_ep_name,
                                    endpoint=endpoint,
                                    collected_params=collected_params,
                                    raw_response=_resp_data,
                                    original_query=user_query,
                                    timestamp=time.time(),
                                )
                            ],
                        )
                    except Exception as _exc:
                        logger.warning(
                            f"[{chat_id}] ATC cache: background write failed: {_exc}"
                        )

                self._create_background_task(_write_l1_l2())
            formatter = APIResponseFormatterModule(
                custom_instructions=custom_instructions
            )
            content = await asyncio.to_thread(
                formatter.forward,
                user_query=user_query,
                api_response=api_result.response_data,
                endpoint_description=description,
                detected_language=detected_language,
                collected_params=collected_params,
            )
        else:
            logger.warning(
                f"[{chat_id}] APIToolWorkflow: API call failed "
                f"(status={api_result.status_code}, error={api_result.error!r})"
            )
            content = api_result.error or ""

        return OrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=content,
        )

    @staticmethod
    def _build_question_response(chat_id: str, question: str) -> OrchestrationResponse:
        return OrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=question,
        )

    async def _execute_multi_api_and_format(
        self,
        chat_id: str,
        parallel_endpoints: List[EndpointSessionState],
        user_query: str,
        detected_language: str,
        custom_instructions: str = "",
    ) -> OrchestrationResponse:
        """Call all parallel endpoints concurrently and return a merged natural-language response.

        Builds a ``call_params``-keyed payload for each :class:`EndpointSessionState`,
        dispatches all calls concurrently via :class:`MultiAPICaller`, then synthesises
        the results into one unified answer with :class:`MultiResponseFormatterModule`.
        """
        call_payloads = [
            {**state.endpoint, "call_params": state.collected_params}
            for state in parallel_endpoints
        ]
        ep_names = [
            state.endpoint.get("name", "<unnamed>") for state in parallel_endpoints
        ]
        logger.info(
            f"[{chat_id}] APIToolWorkflow: parallel — calling {len(call_payloads)} APIs "
            f"concurrently: {ep_names}"
        )

        multi_caller = MultiAPICaller(self._api_caller)
        multi_result = await multi_caller.call_all(
            call_payloads, language=detected_language
        )

        logger.info(
            f"[{chat_id}] APIToolWorkflow: parallel batch complete — "
            f"{sum(r.success for r in multi_result.results)}/{len(multi_result.results)} succeeded"
        )

        _pairs = list(zip(parallel_endpoints, multi_result.results, strict=True))

        if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED:

            async def _write_multi_cache() -> None:
                try:
                    _cs = ATCCacheStore()
                    _ctxs: list[LastCallContext] = []
                    for _state, _result in _pairs:
                        if _result.success and _state.endpoint.get("cacheable", True):
                            _ttl_override = _state.endpoint.get("cache_ttl_seconds")
                            _ttl = (
                                _ttl_override
                                if isinstance(_ttl_override, int) and _ttl_override > 0
                                else ATC_CACHE_DEFAULT_TTL_SECONDS
                            )
                            await _cs.set_l1(
                                chat_id,
                                _state.endpoint.get("name", ""),
                                _state.collected_params,
                                _result.response_data,
                                _ttl,
                            )
                            _ctxs.append(
                                LastCallContext(
                                    api_name=_state.endpoint.get("name", ""),
                                    endpoint=_state.endpoint,
                                    collected_params=_state.collected_params,
                                    raw_response=_result.response_data,
                                    original_query=user_query,
                                    timestamp=time.time(),
                                )
                            )
                    if _ctxs:
                        await _cs.set_l2(chat_id, _ctxs)
                except Exception as _exc:
                    logger.warning(
                        f"[{chat_id}] ATC cache: background multi-write failed: {_exc}"
                    )

            self._create_background_task(_write_multi_cache())

        api_results = [
            (
                state.endpoint.get("name", ""),
                state.endpoint.get("description", ""),
                result.response_data if result.success else result.error or "",
                state.collected_params,
            )
            for state, result in zip(
                parallel_endpoints, multi_result.results, strict=True
            )
        ]

        formatter = MultiResponseFormatterModule(
            custom_instructions=custom_instructions
        )
        content = await asyncio.to_thread(
            formatter.forward,
            user_query=user_query,
            api_results=api_results,
            detected_language=detected_language,
        )

        return OrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=content,
        )

    async def _stream_multi_api_and_format(
        self,
        chat_id: str,
        parallel_endpoints: List[EndpointSessionState],
        user_query: str,
        detected_language: str,
        orchestration_service: OrchestrationServiceProtocol,
        request: OrchestrationRequest,
        costs_metric: Optional[Dict[str, Any]] = None,
        custom_instructions: str = "",
    ) -> AsyncIterator[str]:
        """Call all parallel APIs concurrently, then stream the merged answer token by token.

        API calls are pre-resolved before streaming starts so the LLM synthesis step
        receives all results at once.  Uses the same buffer-first guardrails approach as
        :meth:`_stream_api_and_format` — the full response is assembled and validated
        before any token is sent to the client.
        """
        call_payloads = [
            {**state.endpoint, "call_params": state.collected_params}
            for state in parallel_endpoints
        ]
        ep_names = [
            state.endpoint.get("name", "<unnamed>") for state in parallel_endpoints
        ]
        logger.info(
            f"[{chat_id}] APIToolWorkflow (streaming): parallel — calling {len(call_payloads)} APIs "
            f"concurrently: {ep_names}"
        )

        multi_caller = MultiAPICaller(self._api_caller)
        multi_result = await multi_caller.call_all(
            call_payloads, language=detected_language
        )

        logger.info(
            f"[{chat_id}] APIToolWorkflow (streaming): parallel batch complete — "
            f"{sum(r.success for r in multi_result.results)}/{len(multi_result.results)} succeeded"
        )

        _pairs = list(zip(parallel_endpoints, multi_result.results, strict=True))

        if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED:

            async def _write_multi_cache() -> None:
                try:
                    _cs = ATCCacheStore()
                    _ctxs: list[LastCallContext] = []
                    for _state, _result in _pairs:
                        if _result.success and _state.endpoint.get("cacheable", True):
                            _ttl_override = _state.endpoint.get("cache_ttl_seconds")
                            _ttl = (
                                _ttl_override
                                if isinstance(_ttl_override, int) and _ttl_override > 0
                                else ATC_CACHE_DEFAULT_TTL_SECONDS
                            )
                            await _cs.set_l1(
                                chat_id,
                                _state.endpoint.get("name", ""),
                                _state.collected_params,
                                _result.response_data,
                                _ttl,
                            )
                            _ctxs.append(
                                LastCallContext(
                                    api_name=_state.endpoint.get("name", ""),
                                    endpoint=_state.endpoint,
                                    collected_params=_state.collected_params,
                                    raw_response=_result.response_data,
                                    original_query=user_query,
                                    timestamp=time.time(),
                                )
                            )
                    if _ctxs:
                        await _cs.set_l2(chat_id, _ctxs)
                except Exception as _exc:
                    logger.warning(
                        f"[{chat_id}] ATC cache: background multi-write failed: {_exc}"
                    )

            self._create_background_task(_write_multi_cache())

        api_results = [
            (
                state.endpoint.get("name", ""),
                state.endpoint.get("description", ""),
                result.response_data if result.success else result.error or "",
                state.collected_params,
            )
            for state, result in zip(
                parallel_endpoints, multi_result.results, strict=True
            )
        ]

        formatter = MultiResponseFormatterModule(
            custom_instructions=custom_instructions
        )
        buffered_tokens = [
            token
            async for token in formatter.stream_forward_multi(
                user_query=user_query,
                api_results=api_results,
                detected_language=detected_language,
            )
        ]

        full_response = "".join(buffered_tokens)
        final_answer = full_response  # Track what gets sent to user

        guardrails_passed = True
        if orchestration_service is not None:
            guardrails_adapter = self._get_guardrails_adapter(
                request.environment, request.connection_id
            )
            if guardrails_adapter is not None:
                dummy_response = OrchestrationResponse(
                    chatId=chat_id,
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=full_response,
                )
                checked = await orchestration_service.handle_output_guardrails(
                    guardrails_adapter,
                    dummy_response,
                    request,
                    costs_metric if costs_metric is not None else {},
                )
                if checked.content != full_response:
                    logger.warning(
                        f"[{chat_id}] APIToolWorkflow (streaming): "
                        f"parallel output blocked by guardrails"
                    )
                    yield orchestration_service.format_sse(chat_id, checked.content)
                    final_answer = checked.content
                    guardrails_passed = False

        if guardrails_passed:
            for token in buffered_tokens:
                yield orchestration_service.format_sse(chat_id, token)

        await orchestration_service.store_streaming_inference(request, final_answer)
        yield orchestration_service.format_sse(chat_id, "END")

    # ------------------------------------------------------------------
    # Core loop handler — shared by async and streaming paths
    # ------------------------------------------------------------------

    async def _compute_loop_step(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
    ) -> _LoopStep:
        """Run one agentic loop turn and return a tagged outcome.

        This is the single source of truth for session management and loop logic.
        Both the sync (:meth:`execute_async`) and streaming (:meth:`execute_streaming`)
        paths call this method and then handle the result in their own way:

        * ``"api_call"``  → call API + format response (blocking or streaming)
        * ``"question"``  → return clarifying question to user
        * ``"fallback"``  → no valid state; caller falls back to RAG
        """
        chat_id = request.chatId
        session_store = self._get_session_store()

        # ── Try to resume an existing session ────────────────────────────
        session: Optional[APIToolSession] = None
        if session_store is not None:
            session = await session_store.get(chat_id)

        custom_instructions = await self._get_custom_instructions()

        if session is not None:
            # Resume path — endpoint comes from persisted session
            endpoint = session.selected_endpoint
            if endpoint is None:
                logger.warning(
                    f"[{chat_id}] APIToolWorkflow: session has no endpoint — deleting"
                )
                if session_store is not None:
                    await session_store.delete(chat_id)
                return _LoopStep(kind="fallback", chat_id=chat_id)

            if session.execution_mode == ExecutionMode.PARALLEL.value:
                ep_names = [s.endpoint.get("name") for s in session.parallel_endpoints]
                logger.info(
                    f"[{chat_id}] APIToolWorkflow: resuming parallel session "
                    f"(turn={session.turn_count}, endpoints={ep_names})"
                )
            else:
                logger.info(
                    f"[{chat_id}] APIToolWorkflow: resuming session "
                    f"(turn={session.turn_count}, endpoint={endpoint.get('name')!r})"
                )
        else:
            # New-session path — endpoint must come from classifier context.
            # For parallel execution_mode, drive param collection for the first
            # endpoint (Phase 3 MultiEndpointAgenticLoop will advance the index).
            all_matched: list[dict[str, Any]] = []
            if context.get("execution_mode") == ExecutionMode.PARALLEL:
                all_matched = context.get("matched_endpoints", [])
                endpoint = all_matched[0] if all_matched else None
                if endpoint:
                    logger.info(
                        f"[{chat_id}] APIToolWorkflow: parallel mode — "
                        f"starting param collection for first endpoint "
                        f"{endpoint.get('name')!r} "
                        f"({len(all_matched)} endpoints total)"
                    )
            else:
                endpoint = context.get("matched_endpoint")

            if not endpoint:
                logger.warning(
                    f"[{chat_id}] APIToolWorkflow: no matched_endpoint in context "
                    f"and no active session — falling back"
                )
                return _LoopStep(kind="fallback", chat_id=chat_id)

            params_schema: List[Dict[str, Any]] = endpoint.get("params", [])

            # L1 + L2 cache checks — single-mode new queries only; both guarded by
            # the ATC_RESPONSE_CACHE_ENABLED kill-switch.
            if (
                FeatureFlags.ATC_RESPONSE_CACHE_ENABLED
                and not all_matched
                and endpoint.get("cacheable", True)
            ):
                _cache_store = ATCCacheStore()

                # ── L1: exact param-hash hit ─────────────────────────────────
                _cached = await _cache_store.get_l1(
                    chat_id,
                    endpoint.get("name", ""),
                    context.get("pre_extracted_params", {}),
                )
                if _cached is not None:
                    logger.info(
                        f"[{chat_id}] ATC cache: L1 hit for {endpoint.get('name')!r}"
                    )
                    return _LoopStep(
                        kind="cached_response",
                        chat_id=chat_id,
                        endpoint=endpoint,
                        cached_raw_response=_cached,
                        detected_language=getattr(request, "_detected_language", "en"),
                        user_query=request.message,
                        custom_instructions=custom_instructions,
                        collected_params=context.get("pre_extracted_params", {}),
                    )

                # ── L2: follow-up routing based on last call context ─────────
                _last_calls = await _cache_store.get_l2(chat_id)
                if _last_calls:
                    _matching = next(
                        (c for c in _last_calls if c.api_name == endpoint.get("name")),
                        None,
                    )
                    if _matching is not None:
                        try:
                            _detector = FollowUpDetectorModule()
                            _det_result = await asyncio.to_thread(
                                _detector.forward,
                                user_query=request.message,
                                previous_query=_matching.original_query,
                                previous_params=_matching.collected_params,
                                params_schema=endpoint.get("params", []),
                            )
                            if _det_result["follow_up_type"] == "response_question":
                                logger.info(
                                    f"[{chat_id}] ATC cache: L2 follow-up — response_question"
                                )
                                return _LoopStep(
                                    kind="cached_response",
                                    chat_id=chat_id,
                                    endpoint=endpoint,
                                    cached_raw_response=_matching.raw_response,
                                    detected_language=getattr(
                                        request, "_detected_language", "en"
                                    ),
                                    user_query=request.message,
                                    custom_instructions=custom_instructions,
                                    cache_source="L2",
                                    collected_params=_matching.collected_params,
                                )
                            elif _det_result["follow_up_type"] == "param_update":
                                _updated = _det_result["updated_params"]
                                _merged = {**_matching.collected_params, **_updated}
                                _missing = self._missing_required_params(
                                    endpoint.get("params", []), _merged
                                )
                                if not _missing:
                                    # If the merged params are identical to the previous
                                    # call's params (e.g. no genuine new params survived
                                    # schema validation), the user is re-requesting the
                                    # same data → serve from L2 directly without an API
                                    # call or L1 lookup.
                                    if ATCCacheStore._param_hash(
                                        _merged
                                    ) == ATCCacheStore._param_hash(
                                        _matching.collected_params
                                    ):
                                        # Params unchanged — prefer L1 (exact hash hit
                                        # with the actual previously-collected params).
                                        # L1 uses a TTL so it may have expired; fall
                                        # back to L2 raw_response in that case.
                                        _l1_data = await _cache_store.get_l1(
                                            chat_id,
                                            endpoint.get("name", ""),
                                            _matching.collected_params,
                                        )
                                        if _l1_data is not None:
                                            logger.info(
                                                f"[{chat_id}] ATC cache: L2 follow-up — param_update "
                                                f"(params unchanged), L1 hit — serving from L1"
                                            )
                                            return _LoopStep(
                                                kind="cached_response",
                                                chat_id=chat_id,
                                                endpoint=endpoint,
                                                cached_raw_response=_l1_data,
                                                detected_language=getattr(
                                                    request, "_detected_language", "en"
                                                ),
                                                user_query=request.message,
                                                custom_instructions=custom_instructions,
                                                cache_source="L1",
                                                collected_params=_matching.collected_params,
                                            )
                                        logger.info(
                                            f"[{chat_id}] ATC cache: L2 follow-up — param_update "
                                            f"(params unchanged), L1 miss — serving from L2"
                                        )
                                        return _LoopStep(
                                            kind="cached_response",
                                            chat_id=chat_id,
                                            endpoint=endpoint,
                                            cached_raw_response=_matching.raw_response,
                                            detected_language=getattr(
                                                request, "_detected_language", "en"
                                            ),
                                            user_query=request.message,
                                            custom_instructions=custom_instructions,
                                            cache_source="L2",
                                            collected_params=_matching.collected_params,
                                        )
                                    logger.info(
                                        f"[{chat_id}] ATC cache: L2 follow-up — param_update "
                                        f"(all params present), calling API directly"
                                    )
                                    return _LoopStep(
                                        kind="api_call",
                                        chat_id=chat_id,
                                        endpoint=endpoint,
                                        collected_params=_merged,
                                        detected_language=getattr(
                                            request, "_detected_language", "en"
                                        ),
                                        user_query=request.message,
                                        custom_instructions=custom_instructions,
                                    )
                                else:
                                    logger.info(
                                        f"[{chat_id}] ATC cache: L2 follow-up — param_update "
                                        f"(missing: {_missing}), seeding agentic loop"
                                    )
                                    context["seeded_params"] = _merged
                            # new_intent → fall through to normal agentic-loop path
                        except Exception as _exc:
                            logger.warning(
                                f"[{chat_id}] ATC cache: L2 follow-up detection failed: "
                                f"{_exc} — falling through to normal path"
                            )

            # Fast path — skip the agentic loop when no required params need collecting.
            # user_query falls back to request.message on the fast path because no session
            # exists yet (original_query is only stored once a session is created).
            user_query_for_fast_path = request.message
            if all_matched:
                # Parallel mode: fast-path only when ALL endpoints have no required params.
                if all(
                    not self._required_params(ep.get("params", []))
                    for ep in all_matched
                ):
                    logger.info(
                        f"[{chat_id}] APIToolWorkflow: parallel fast path — "
                        f"all {len(all_matched)} endpoints have no required params"
                    )
                    return _LoopStep(
                        kind="multi_api_call",
                        chat_id=chat_id,
                        parallel_endpoints=[
                            EndpointSessionState(endpoint=e) for e in all_matched
                        ],
                        detected_language=getattr(request, "_detected_language", "en"),
                        user_query=user_query_for_fast_path,
                        custom_instructions=custom_instructions,
                    )
            elif not self._required_params(params_schema):
                # Single mode: the only endpoint has no required params.
                logger.info(
                    f"[{chat_id}] APIToolWorkflow: endpoint {endpoint.get('name')!r} "
                    f"has no required params — fast path"
                )
                return _LoopStep(
                    kind="api_call",
                    chat_id=chat_id,
                    endpoint=endpoint,
                    collected_params={},
                    detected_language=getattr(request, "_detected_language", "en"),
                    user_query=user_query_for_fast_path,
                    custom_instructions=custom_instructions,
                )

            # Create a new session before running the first loop turn
            _seeded_params = context.get("seeded_params", {})
            if session_store is not None:
                new_session = APIToolSession(
                    chat_id=chat_id,
                    state="collecting_params",
                    selected_endpoint=endpoint,
                    collected_params=_seeded_params,
                    turn_count=0,
                    max_turns=5,
                    awaiting_continuation=False,
                    detected_language=getattr(request, "_detected_language", "en"),
                    original_query=request.message,
                    execution_mode=ExecutionMode.PARALLEL.value
                    if all_matched
                    else ExecutionMode.SINGLE.value,
                    parallel_endpoints=[
                        EndpointSessionState(endpoint=e) for e in all_matched
                    ],
                )
                await session_store.save(new_session)
                session = new_session
            else:
                logger.warning(
                    f"[{chat_id}] APIToolWorkflow: Redis unavailable — "
                    f"running loop without session persistence"
                )
                session = APIToolSession(
                    chat_id=chat_id,
                    state="collecting_params",
                    selected_endpoint=endpoint,
                    collected_params=_seeded_params,
                    turn_count=0,
                    max_turns=5,
                    awaiting_continuation=False,
                    detected_language=getattr(request, "_detected_language", "en"),
                    original_query=request.message,
                    execution_mode=ExecutionMode.PARALLEL.value
                    if all_matched
                    else ExecutionMode.SINGLE.value,
                    parallel_endpoints=[
                        EndpointSessionState(endpoint=e) for e in all_matched
                    ],
                )

        # ── Run one loop turn ─────────────────────────────────────────────
        if session_store is None:
            logger.warning(
                f"[{chat_id}] APIToolWorkflow: session store unavailable — "
                f"agentic loop running without persistence"
            )

        # If custom_instructions contain a language directive (e.g. "respond in English"
        # inside an Estonian-language prompt), use that directed language for all
        # user-facing questions — both the LLM-generated clarifying questions and the
        # hardcoded continuation question.  Falls back to the session-detected language.
        effective_session_language = (
            self._language_from_custom_instructions(custom_instructions)
            or session.detected_language
        )

        _atc_conversation_summary: Optional[str] = None
        conversation_history_for_loop: List[Dict[str, Any]]
        if session.turn_count == 0:
            # On the first ATC turn there is no prior ATC exchange to pass.
            conversation_history_for_loop = []
        else:
            # On subsequent turns prefer Redis as the authoritative history source.
            _redis_history, _atc_conversation_summary = await get_conversation_history(
                chat_id=chat_id,
                store=self._get_conversation_history_store(),
                fallback=list(request.conversationHistory or []),
            )
            conversation_history_for_loop = [
                {"authorRole": item.authorRole, "message": item.message}
                for item in _redis_history
            ]

        # Incorporate any Redis-supplied conversation summary into custom_instructions
        # so the param extractor and response formatter have full conversational context.
        if _atc_conversation_summary:
            _summary_prefix = (
                f"Summary of earlier conversation: {_atc_conversation_summary}"
            )
            custom_instructions = (
                f"{_summary_prefix}\n\n{custom_instructions}".strip()
                if custom_instructions
                else _summary_prefix
            )

        if session.execution_mode == ExecutionMode.PARALLEL.value:
            # Parallel path: MultiEndpointAgenticLoop operates on the full
            # parallel_endpoints list and builds a merged schema internally so
            # only one clarifying question is asked per turn.
            multi_loop = MultiEndpointAgenticLoop(
                session_store=session_store,
                param_extractor=ParamExtractionModule(
                    custom_instructions=custom_instructions
                ),
            )
            result, question_tokens = await multi_loop.stream_run_turn(
                chat_id=chat_id,
                user_message=request.message,
                conversation_history=conversation_history_for_loop,
                endpoint_states=session.parallel_endpoints,
                turn_count=session.turn_count,
                awaiting_continuation=session.awaiting_continuation,
                session_language=effective_session_language,
            )

            if result.status == AgenticLoopStatus.COMPLETED:
                logger.info(
                    f"[{chat_id}] APIToolWorkflow: parallel params fully collected "
                    f"(turns={result.turn_count}, "
                    f"endpoints={[s.endpoint.get('name') for s in session.parallel_endpoints]})"
                )
                if session_store is not None:
                    await session_store.delete(chat_id)
                return _LoopStep(
                    kind="multi_api_call",
                    chat_id=chat_id,
                    parallel_endpoints=session.parallel_endpoints,
                    detected_language=effective_session_language,
                    user_query=session.original_query or request.message,
                    custom_instructions=custom_instructions,
                )

            if result.status == AgenticLoopStatus.MAX_TURNS_REACHED:
                logger.info(
                    f"[{chat_id}] APIToolWorkflow: parallel max turns reached — deleting session"
                )
                if session_store is not None:
                    await session_store.delete(chat_id)
                return _LoopStep(kind="fallback", chat_id=chat_id)

            # NEEDS_INPUT or AWAITING_CONTINUATION_DECISION
            logger.info(
                f"[{chat_id}] APIToolWorkflow: parallel — asking for more info "
                f"(status={result.status.value}, turn={result.turn_count})"
            )
            return _LoopStep(
                kind="question",
                chat_id=chat_id,
                question=result.clarifying_question,
                question_tokens=question_tokens,
                custom_instructions=custom_instructions,
            )

        # Single path: AgenticLoop (untouched)
        loop = self._build_agentic_loop(session_store, custom_instructions)  # type: ignore[arg-type]

        result, question_tokens = await loop.stream_run_turn(
            chat_id=chat_id,
            user_message=request.message,
            conversation_history=conversation_history_for_loop,
            params_schema=endpoint.get("params", []),
            collected_params=session.collected_params,
            turn_count=session.turn_count,
            max_turns=session.max_turns,
            awaiting_continuation=session.awaiting_continuation,
            session_language=effective_session_language,
            seeded_params=context.get("seeded_params"),
        )

        if result.status == AgenticLoopStatus.COMPLETED:
            logger.info(
                f"[{chat_id}] APIToolWorkflow: all params collected "
                f"(turns={result.turn_count}, params={list(result.collected_params.keys())})"
            )
            if session_store is not None:
                await session_store.delete(chat_id)
            return _LoopStep(
                kind="api_call",
                chat_id=chat_id,
                endpoint=endpoint,
                collected_params=result.collected_params,
                detected_language=effective_session_language,
                user_query=session.original_query or request.message,
                custom_instructions=custom_instructions,
            )

        if result.status == AgenticLoopStatus.MAX_TURNS_REACHED:
            logger.info(
                f"[{chat_id}] APIToolWorkflow: max turns reached — deleting session"
            )
            if session_store is not None:
                await session_store.delete(chat_id)
            return _LoopStep(kind="fallback", chat_id=chat_id)

        # NEEDS_INPUT or AWAITING_CONTINUATION_DECISION
        logger.info(
            f"[{chat_id}] APIToolWorkflow: asking for more info "
            f"(status={result.status.value}, turn={result.turn_count})"
        )
        return _LoopStep(
            kind="question",
            chat_id=chat_id,
            question=result.clarifying_question,
            question_tokens=question_tokens,
            custom_instructions=custom_instructions,
        )

    async def _format_cached_response(
        self,
        chat_id: str,
        endpoint: Dict[str, Any],
        user_query: str,
        detected_language: str,
        cached_raw_response: Any,
        collected_params: Dict[str, Any],
        custom_instructions: str = "",
        cache_source: str = "L1",
    ) -> OrchestrationResponse:
        """Format a cached API response without calling the external API.

        Used when a cache hit (L1 or L2) is detected for a matched endpoint.
        Runs the raw response through APIResponseFormatterModule, bypassing APICaller.
        The collected_params are passed to the formatter to maintain query context consistency.
        """
        description = endpoint.get("description", "")
        logger.info(
            f"[{chat_id}] ATC cache: formatting {cache_source} hit for {endpoint.get('name')!r}"
        )
        formatter = APIResponseFormatterModule(custom_instructions=custom_instructions)
        content = await asyncio.to_thread(
            formatter.forward,
            user_query=user_query,
            api_response=cached_raw_response,
            endpoint_description=description,
            detected_language=detected_language,
            collected_params=collected_params,
        )
        return OrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=content,
        )

    async def _stream_cached_response(
        self,
        chat_id: str,
        endpoint: Dict[str, Any],
        user_query: str,
        detected_language: str,
        cached_raw_response: Any,
        collected_params: Dict[str, Any],
        orchestration_service: OrchestrationServiceProtocol,
        request: OrchestrationRequest,
        costs_metric: Optional[Dict[str, Any]] = None,
        custom_instructions: str = "",
        cache_source: str = "L1",
    ) -> AsyncIterator[str]:
        """Stream a cached API response without calling the external API.

        Used when a cache hit (L1 or L2) is detected for a matched endpoint.
        Runs the raw response through APIResponseFormatterModule streaming, bypassing APICaller.
        The collected_params are passed to the formatter to maintain query context consistency.
        """
        description = endpoint.get("description", "")
        logger.info(
            f"[{chat_id}] ATC cache: streaming {cache_source} hit for {endpoint.get('name')!r}"
        )
        formatter = APIResponseFormatterModule(custom_instructions=custom_instructions)
        buffered_tokens = [
            token
            async for token in formatter.stream_forward(
                user_query=user_query,
                api_response=cached_raw_response,
                endpoint_description=description,
                detected_language=detected_language,
                collected_params=collected_params,
            )
        ]

        full_response = "".join(buffered_tokens)
        final_answer = full_response  # Track what gets sent to user

        guardrails_passed = True
        if orchestration_service is not None:
            guardrails_adapter = self._get_guardrails_adapter(
                request.environment, request.connection_id
            )
            if guardrails_adapter is not None:
                dummy_response = OrchestrationResponse(
                    chatId=chat_id,
                    llmServiceActive=True,
                    questionOutOfLLMScope=False,
                    inputGuardFailed=False,
                    content=full_response,
                )
                checked = await orchestration_service.handle_output_guardrails(
                    guardrails_adapter,
                    dummy_response,
                    request,
                    costs_metric if costs_metric is not None else {},
                )
                if checked.content != full_response:
                    logger.warning(
                        f"[{chat_id}] ATC cache: streaming {cache_source} hit blocked by guardrails"
                    )
                    yield orchestration_service.format_sse(chat_id, checked.content)
                    final_answer = checked.content
                    guardrails_passed = False

        if guardrails_passed:
            for token in buffered_tokens:
                yield orchestration_service.format_sse(chat_id, token)

        await orchestration_service.store_streaming_inference(request, final_answer)
        yield orchestration_service.format_sse(chat_id, "END")

    async def _run(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
    ) -> Optional[OrchestrationResponse]:
        """Blocking execution path — delegates to :meth:`_compute_loop_step`."""
        step = await self._compute_loop_step(request, context)

        if step.kind == "fallback":
            return None

        if step.kind == "question":
            return self._build_question_response(step.chat_id, step.question)

        # multi_api_call — concurrent batch execution + multi-formatter
        if step.kind == "multi_api_call":
            response = await self._execute_multi_api_and_format(
                chat_id=step.chat_id,
                parallel_endpoints=step.parallel_endpoints,
                user_query=step.user_query,
                detected_language=step.detected_language,
                custom_instructions=step.custom_instructions,
            )
            if response.llmServiceActive and self.orchestration_service is not None:
                guardrails_adapter = self._get_guardrails_adapter(
                    request.environment, request.connection_id
                )
                if guardrails_adapter is not None:
                    response = cast(
                        OrchestrationResponse,
                        await self.orchestration_service.handle_output_guardrails(
                            guardrails_adapter,
                            response,
                            request,
                            {},
                        ),
                    )
            return response

        # cached_response — cache hit (L1 or L2): skip APICaller, format cached data
        if step.kind == "cached_response":
            response = await self._format_cached_response(
                chat_id=step.chat_id,
                endpoint=step.endpoint,
                user_query=step.user_query,
                detected_language=step.detected_language,
                collected_params=step.collected_params,
                custom_instructions=step.custom_instructions,
                cached_raw_response=step.cached_raw_response,
                cache_source=step.cache_source,
            )
            if response.llmServiceActive and self.orchestration_service is not None:
                guardrails_adapter = self._get_guardrails_adapter(
                    request.environment, request.connection_id
                )
                if guardrails_adapter is not None:
                    response = cast(
                        OrchestrationResponse,
                        await self.orchestration_service.handle_output_guardrails(
                            guardrails_adapter,
                            response,
                            request,
                            {},
                        ),
                    )
            return response

        # api_call — blocking
        response = await self._execute_api_and_format(
            chat_id=step.chat_id,
            endpoint=step.endpoint,
            collected_params=step.collected_params,
            user_query=step.user_query,
            detected_language=step.detected_language,
            custom_instructions=step.custom_instructions,
        )

        # Output guardrails — only on successful LLM-formatted answers
        if response.llmServiceActive and self.orchestration_service is not None:
            guardrails_adapter = self._get_guardrails_adapter(
                request.environment, request.connection_id
            )
            if guardrails_adapter is not None:
                response = cast(
                    OrchestrationResponse,
                    await self.orchestration_service.handle_output_guardrails(
                        guardrails_adapter,
                        response,
                        request,
                        {},
                    ),
                )

        return response

    # ------------------------------------------------------------------
    # BaseWorkflow interface
    # ------------------------------------------------------------------

    async def execute_async(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[OrchestrationResponse]:
        return await self._run(request, context)

    async def _stream_api_and_format(
        self,
        chat_id: str,
        endpoint: Dict[str, Any],
        collected_params: Dict[str, Any],
        user_query: str,
        detected_language: str,
        orchestration_service: OrchestrationServiceProtocol,
        request: OrchestrationRequest,
        costs_metric: Optional[Dict[str, Any]] = None,
        custom_instructions: str = "",
    ) -> AsyncIterator[str]:
        """Call the external API and stream the formatted response token by token.

        Clarifying questions are short and do not need token streaming; they are
        yielded as a single SSE frame.
        """
        url = endpoint.get("url", "")
        method = endpoint.get("method", "GET")
        description = endpoint.get("description", "")

        # L1 cache check — collected_params is complete here, so the hash matches
        # what was stored on the previous successful call with the same params.
        if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED and endpoint.get("cacheable", True):
            _cache_store = ATCCacheStore()
            _cached = await _cache_store.get_l1(
                chat_id, endpoint.get("name", ""), collected_params
            )
            if _cached is not None:
                logger.info(
                    f"[{chat_id}] ATC cache: L1 hit for {endpoint.get('name')!r} "
                    f"— skipping API call (streaming)"
                )
                async for token in self._stream_cached_response(
                    chat_id=chat_id,
                    endpoint=endpoint,
                    user_query=user_query,
                    detected_language=detected_language,
                    cached_raw_response=_cached,
                    collected_params=collected_params,
                    orchestration_service=orchestration_service,
                    request=request,
                    costs_metric=costs_metric,
                    custom_instructions=custom_instructions,
                    cache_source="L1",
                ):
                    yield token
                return

        logger.info(
            f"[{chat_id}] APIToolWorkflow (streaming): calling API "
            f"{method} {url} with params={list(collected_params.keys())}"
        )
        api_result = await self._api_caller.call(
            url=url,
            method=method,
            params=collected_params,
            language=detected_language,
        )

        if api_result.success:
            logger.info(
                f"[{chat_id}] APIToolWorkflow (streaming): API call succeeded "
                f"(status={api_result.status_code}), streaming formatted response"
            )
            if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED and endpoint.get(
                "cacheable", True
            ):
                _ep_name = endpoint.get("name", "")
                _ttl_override = endpoint.get("cache_ttl_seconds")
                _ttl = (
                    _ttl_override
                    if isinstance(_ttl_override, int) and _ttl_override > 0
                    else ATC_CACHE_DEFAULT_TTL_SECONDS
                )
                _resp_data = api_result.response_data

                async def _write_l1_l2() -> None:
                    try:
                        _cs = ATCCacheStore()
                        await _cs.set_l1(
                            chat_id, _ep_name, collected_params, _resp_data, _ttl
                        )
                        await _cs.set_l2(
                            chat_id,
                            [
                                LastCallContext(
                                    api_name=_ep_name,
                                    endpoint=endpoint,
                                    collected_params=collected_params,
                                    raw_response=_resp_data,
                                    original_query=user_query,
                                    timestamp=time.time(),
                                )
                            ],
                        )
                    except Exception as _exc:
                        logger.warning(
                            f"[{chat_id}] ATC cache: background write failed: {_exc}"
                        )

                self._create_background_task(_write_l1_l2())
            # Buffer all tokens first, then validate with output guardrails before
            # streaming to the client (validate-first approach).
            formatter = APIResponseFormatterModule(
                custom_instructions=custom_instructions
            )
            buffered_tokens = [
                token
                async for token in formatter.stream_forward(
                    user_query=user_query,
                    api_response=api_result.response_data,
                    endpoint_description=description,
                    detected_language=detected_language,
                    collected_params=collected_params,
                )
            ]

            full_response = "".join(buffered_tokens)
            final_answer_to_store = full_response  # Track what gets sent to user

            # Run output guardrails on the complete response
            guardrails_passed = True
            if orchestration_service is not None:
                guardrails_adapter = self._get_guardrails_adapter(
                    request.environment, request.connection_id
                )
                if guardrails_adapter is not None:
                    dummy_response = OrchestrationResponse(
                        chatId=chat_id,
                        llmServiceActive=True,
                        questionOutOfLLMScope=False,
                        inputGuardFailed=False,
                        content=full_response,
                    )

                    checked = await orchestration_service.handle_output_guardrails(
                        guardrails_adapter,
                        dummy_response,
                        request,
                        costs_metric if costs_metric is not None else {},
                    )
                    if checked.content != full_response:
                        # Guardrails replaced the content — yield violation message
                        logger.warning(
                            f"[{chat_id}] APIToolWorkflow (streaming): "
                            f"output blocked by guardrails"
                        )
                        yield orchestration_service.format_sse(chat_id, checked.content)
                        final_answer_to_store = checked.content
                        guardrails_passed = False

            if guardrails_passed:
                for token in buffered_tokens:
                    yield orchestration_service.format_sse(chat_id, token)
        else:
            logger.warning(
                f"[{chat_id}] APIToolWorkflow (streaming): API call failed "
                f"(status={api_result.status_code}, error={api_result.error!r})"
            )
            final_answer_to_store = api_result.error or ""
            yield orchestration_service.format_sse(chat_id, api_result.error or "")

        await orchestration_service.store_streaming_inference(
            request, final_answer_to_store
        )
        yield orchestration_service.format_sse(chat_id, "END")

    async def execute_streaming(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """Streaming mode — run the agentic loop and stream the response token by token.

        Clarifying questions (NEEDS_INPUT / AWAITING_CONTINUATION) are short; they
        are emitted as a single SSE frame followed by END — same as before.

        Final API responses (COMPLETED / fast-path) are streamed word-by-word via
        DSPy native streaming on the ``formatted_answer`` field.
        """
        if self.orchestration_service is None:
            logger.error(
                f"[{request.chatId}] APIToolWorkflow streaming: orchestration_service not set"
            )
            return None

        step = await self._compute_loop_step(request, context)

        if step.kind == "fallback":
            return None

        orchestration_service = self.orchestration_service

        if step.kind == "question":

            async def _stream_question() -> AsyncIterator[str]:
                accumulated_question: list[str] = []
                for token in step.question_tokens or [step.question]:
                    accumulated_question.append(token)
                    yield orchestration_service.format_sse(step.chat_id, token)
                full_question = "".join(accumulated_question)
                await orchestration_service.store_streaming_inference(
                    request, full_question
                )
                yield orchestration_service.format_sse(step.chat_id, "END")

            return _stream_question()

        # multi_api_call — stream the merged answer from all parallel APIs
        if step.kind == "multi_api_call":
            return self._stream_multi_api_and_format(
                chat_id=step.chat_id,
                parallel_endpoints=step.parallel_endpoints,
                user_query=step.user_query,
                detected_language=step.detected_language,
                orchestration_service=orchestration_service,
                request=request,
                costs_metric=context.get("costs_metric"),
                custom_instructions=step.custom_instructions,
            )

        # cached_response — cache hit (L1 or L2): skip APICaller, stream formatted cached data
        if step.kind == "cached_response":
            return self._stream_cached_response(
                chat_id=step.chat_id,
                endpoint=step.endpoint,
                user_query=step.user_query,
                detected_language=step.detected_language,
                collected_params=step.collected_params,
                cached_raw_response=step.cached_raw_response,
                orchestration_service=orchestration_service,
                request=request,
                costs_metric=context.get("costs_metric"),
                custom_instructions=step.custom_instructions,
                cache_source=step.cache_source,
            )

        # api_call — stream the LLM-formatted answer token by token
        return self._stream_api_and_format(
            chat_id=step.chat_id,
            endpoint=step.endpoint,
            collected_params=step.collected_params,
            user_query=step.user_query,
            detected_language=step.detected_language,
            orchestration_service=orchestration_service,
            request=request,
            costs_metric=context.get("costs_metric"),
            custom_instructions=step.custom_instructions,
        )

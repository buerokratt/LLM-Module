"""Multi-endpoint agentic loop for parallel parameter collection."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.loki_logger import LokiLogger
from src.utils.error_utils import generate_error_id

from models.session_models import EndpointSessionState
from utils.api_tool_session_store import APIToolSessionStore
from tool_classifier.constants import (
    CONTINUATION_QUESTION,
    CONTINUATION_QUESTION_ET,
    CONTINUATION_QUESTION_RU,
    MULTI_API_MAX_TURNS,
    MULTI_INTENT_CONTINUATION_TURN,
    MULTI_INTENT_MAX_TURNS,
)
from tool_classifier.continuation_utils import detect_continuation_response
from tool_classifier.enums import AgenticLoopStatus
from tool_classifier.models import AgenticLoopResult
from tool_classifier.param_extractor import ParamExtractionModule, strip_format_hints

logger = LokiLogger(service_name="api-tool-calling")

_CONTINUATION_QUESTIONS: dict[str, str] = {
    "en": CONTINUATION_QUESTION,
    "et": CONTINUATION_QUESTION_ET,
    "ru": CONTINUATION_QUESTION_RU,
}


class MultiEndpointAgenticLoop:
    """Multi-turn parameter collection loop for parallel endpoint execution.

    Merges the parameter schemas of all active endpoints, deduplicates shared
    params, asks the user **once** per turn, and distributes extracted values
    back to the per-endpoint :class:`~models.session_models.EndpointSessionState`
    objects.

    The loop is stateless between HTTP requests — all mutable state is held in
    the :class:`~models.session_models.EndpointSessionState` list passed in on
    each call.  The session store is used exclusively to persist updated state
    between turns.

    Turn limit
    ----------
    * Multi-intent (>1 endpoint): ``max_turns = MULTI_INTENT_MAX_TURNS`` (6)
    * Single-intent: ``max_turns = min(3 * num_endpoints, MULTI_API_MAX_TURNS)``

    Continuation threshold
    ----------------------
    * Multi-intent (>1 endpoint): ``continuation_turn = MULTI_INTENT_CONTINUATION_TURN`` (4)
    * Single-intent: ``continuation_turn = num_endpoints + 1``

    Fixed turn-4 continuation for multi-intent gives the user enough turns to
    supply all intent parameters before being asked whether to keep going.
    All 6 turns (``updated_turn_count`` 1–6) execute normally; the RAG fallback
    is triggered on the **7th** call, when the pre-increment ``turn_count``
    reaches ``MULTI_INTENT_MAX_TURNS`` (6) and the guard ``turn_count >=
    max_turns`` fires before any extraction is attempted.

    Typical usage::

        loop = MultiEndpointAgenticLoop(
            session_store=app.state.session_store,
            param_extractor=ParamExtractionModule(),
        )

        result = await loop.run_turn(
            chat_id=request.chatId,
            user_message=request.message,
            conversation_history=request.conversationHistory,
            endpoint_states=session.parallel_endpoints,
            turn_count=session.turn_count,
        )

        if result.status == AgenticLoopStatus.COMPLETED:
            # All endpoints have their params — proceed to API calls
            ...
        elif result.status == AgenticLoopStatus.NEEDS_INPUT:
            # Session already saved inside run_turn — return question to user
            ...
        else:  # MAX_TURNS_REACHED
            # Delete session and fall back gracefully
            ...
    """

    def __init__(
        self,
        session_store: Optional[APIToolSessionStore],
        param_extractor: ParamExtractionModule,
    ) -> None:
        """Initialise the loop with injected session store and param extractor.

        Args:
            session_store: Redis-backed store used to persist loop state between
                HTTP requests. Injected to allow easy mocking in tests. May be
                ``None`` in environments where session persistence is unavailable.
            param_extractor: DSPy module that extracts parameter values from a
                user message. Injected to allow easy mocking in tests.
        """
        self._session_store: Optional[APIToolSessionStore] = session_store
        self._param_extractor = param_extractor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_turn(
        self,
        chat_id: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        endpoint_states: List[EndpointSessionState],
        turn_count: int,
        awaiting_continuation: bool = False,
        session_language: str = "en",
        continuation_language: Optional[str] = None,
    ) -> AgenticLoopResult:
        """Process one user turn of the multi-endpoint parameter-collection loop.

        Steps:
        0. Continuation decision — if ``awaiting_continuation`` is True, detect
           whether the user said yes (keep going) or no (fall back to RAG).
        1. Turn limit guard — return MAX_TURNS_REACHED when exhausted.
        2. Build merged schema from all incomplete endpoints (deduplicating by name).
        3. Compute merged already_collected (union of per-endpoint collected_params).
        4. Call :class:`~param_extractor.ParamExtractionModule` with the merged schema.
        5. Distribute extracted params back to owning endpoints; mark endpoints
           completed when all their required params are present.
        6. If all endpoints complete → save session, return COMPLETED.
        7. At the continuation threshold → save session, return
           AWAITING_CONTINUATION_DECISION.
        8. Otherwise → save session, return NEEDS_INPUT with the clarifying question.

        Args:
            chat_id: Unique conversation identifier used as the Redis session key.
            user_message: The user's latest message for this turn.
            conversation_history: Recent conversation turns as a list of
                ``{"authorRole": str, "message": str}`` dicts.
            endpoint_states: Per-endpoint state objects.  Mutated in-place when
                params are distributed.
            turn_count: The current turn index (0-based before this call).
            awaiting_continuation: True when the previous turn returned
                AWAITING_CONTINUATION_DECISION and we are now processing the
                user's yes/no reply.
            session_language: Language code for clarifying questions (``"en"``,
                ``"et"``, or ``"ru"``).
            continuation_language: Override language for the continuation
                yes/no question.  Falls back to ``session_language`` when None.

        Returns:
            :class:`~models.AgenticLoopResult` with updated status,
            collected_params (merged across all endpoints), and turn_count.
        """
        if not endpoint_states:
            logger.debug(
                f"MultiEndpointAgenticLoop: no endpoints provided for chat_id={chat_id} — returning COMPLETED"
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.COMPLETED,
                collected_params={},
                clarifying_question="",
                turn_count=turn_count,
            )

        num_endpoints = len(endpoint_states)
        max_turns, continuation_turn = self._compute_turn_limits(num_endpoints)
        updated_turn_count = turn_count + 1

        logger.info(
            f"MultiEndpointAgenticLoop: multi loop turn started | event_type=multi_loop_turn_started"
            f" chat_id={chat_id} turn_count={turn_count} endpoint_count={num_endpoints}"
            f" incomplete_count={sum(1 for s in endpoint_states if not s.completed)}"
        )

        # Step 0 — Continuation decision
        if awaiting_continuation:
            wants_to_continue = detect_continuation_response(user_message)
            if wants_to_continue:
                logger.info(
                    f"MultiEndpointAgenticLoop: continuation user accepted | event_type=continuation_user_accepted"
                    f" chat_id={chat_id} turn_count={turn_count}"
                )
                awaiting_continuation = False
            else:
                logger.info(
                    f"MultiEndpointAgenticLoop: continuation user declined | event_type=continuation_user_declined"
                    f" chat_id={chat_id} turn_count={turn_count} status=max_turns_reached"
                )
                return AgenticLoopResult(
                    status=AgenticLoopStatus.MAX_TURNS_REACHED,
                    collected_params=self._merged_collected(endpoint_states),
                    clarifying_question="",
                    turn_count=updated_turn_count,
                )

        # Step 1 — Turn limit guard (no session save — caller deletes)
        if turn_count >= max_turns:
            logger.warning(
                f"MultiEndpointAgenticLoop: turn limit reached | event_type=turn_limit_reached"
                f" chat_id={chat_id} turn_count={turn_count} max_turns={max_turns}"
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.MAX_TURNS_REACHED,
                collected_params=self._merged_collected(endpoint_states),
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 2 — Build merged schema + param owner map + namespace map
        merged_schema, param_owners, namespace_map = self._build_merged_schema(
            endpoint_states
        )

        logger.debug(
            f"MultiEndpointAgenticLoop: schema merged | event_type=schema_merged"
            f" chat_id={chat_id} turn_count={turn_count}"
            f" total_param_count={sum(len(s.endpoint.get('params', [])) for s in endpoint_states if not s.completed)}"
            f" deduplicated_count={len(merged_schema)} namespaced_count={len(namespace_map)}"
        )

        # Step 3 — Namespaced already_collected for the LLM extractor
        merged_already_collected = self._build_namespaced_already_collected(
            endpoint_states, namespace_map
        )

        # Step 4 — Extract params from the current user message
        intent_groups = self._build_intent_groups(
            endpoint_states, merged_already_collected, namespace_map
        )
        _t0 = time.time()
        try:
            extraction = await asyncio.to_thread(
                self._param_extractor,
                user_message,
                merged_schema,
                conversation_history,
                merged_already_collected,
                session_language,
                turn_count,
                intent_groups,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.debug(
                f"MultiEndpointAgenticLoop: param extraction complete | event_type=param_extraction_complete"
                f" chat_id={chat_id} turn_count={turn_count}"
                f" extracted_count={len(extraction['extracted_params'])} duration_ms={_duration_ms}"
            )
        except Exception as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.error(
                f"MultiEndpointAgenticLoop: param extraction failed | event_type=param_extraction_failed"
                f" chat_id={chat_id} turn_count={turn_count}"
                f" error_id={generate_error_id()} duration_ms={_duration_ms} exc={exc}"
            )
            await self._save_session(
                chat_id,
                endpoint_states,
                updated_turn_count,
                awaiting_continuation=awaiting_continuation,
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.NEEDS_INPUT,
                collected_params=self._merged_collected(endpoint_states),
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 5 — Distribute extracted params to owning endpoints
        previously_completed_indices: set[int] = {
            i for i, s in enumerate(endpoint_states) if s.completed
        }
        self._distribute_params(
            extraction["extracted_params"], endpoint_states, param_owners, namespace_map
        )

        # Step 5b — If the user gave a single value that was duplicated across
        # multiple endpoints' conflicting params, clear the duplicates so the loop
        # re-asks for each endpoint's params separately on the next turn.
        enforced = self._enforce_sequential_conflicting_params(
            endpoint_states, namespace_map
        )

        # Step 5c — If multiple endpoints with non-conflicting param names were all
        # completed by the same single-turn user response (e.g. one date range applied
        # to both electricity prices and parliament stats), keep only the first newly-
        # completed endpoint and clear the rest so they are asked for separately.
        enforced = (
            self._enforce_sequential_parallel_completion(
                endpoint_states, previously_completed_indices
            )
            or enforced
        )

        # Step 6 — Check global completion
        all_done = all(state.completed for state in endpoint_states)
        merged_after = self._merged_collected(endpoint_states)

        if all_done:
            logger.info(
                f"MultiEndpointAgenticLoop: all endpoints completed | event_type=multi_loop_all_endpoints_completed"
                f" chat_id={chat_id} turn_count={turn_count} endpoint_count={num_endpoints}"
                f" status=completed duration_ms={_duration_ms}"
            )
            await self._save_session(
                chat_id,
                endpoint_states,
                updated_turn_count,
                awaiting_continuation=False,
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.COMPLETED,
                collected_params=merged_after,
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 7 — Still missing params
        logger.debug(
            f"MultiEndpointAgenticLoop: loop needs input | event_type=loop_needs_input"
            f" chat_id={chat_id} turn_count={turn_count}"
            f" missing_params={extraction['missing_required']} status=needs_input"
        )

        # At exactly the continuation threshold, ask whether to keep going.
        if updated_turn_count == continuation_turn:
            logger.info(
                f"MultiEndpointAgenticLoop: continuation threshold reached | event_type=continuation_threshold_reached"
                f" chat_id={chat_id} turn_count={turn_count} continuation_turn={continuation_turn}"
                f" missing_count={len(extraction['missing_required'])}"
            )
            effective_lang = continuation_language or session_language
            continuation_q = _CONTINUATION_QUESTIONS.get(
                effective_lang, CONTINUATION_QUESTION
            )
            await self._save_session(
                chat_id, endpoint_states, updated_turn_count, awaiting_continuation=True
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.AWAITING_CONTINUATION_DECISION,
                collected_params=merged_after,
                clarifying_question=continuation_q,
                turn_count=updated_turn_count,
            )

        await self._save_session(
            chat_id, endpoint_states, updated_turn_count, awaiting_continuation=False
        )

        # If enforcement cleared duplicate params from a later endpoint and the LLM's
        # question is stale (it said "none" believing all params were satisfied), we
        # must regenerate a fresh question that covers the remaining missing params.
        clarifying_q = extraction["clarifying_question"]
        if enforced and clarifying_q.strip().lower() == "none":
            clarifying_q = await self._regenerate_question_after_enforcement(
                endpoint_states, merged_schema, namespace_map, session_language
            )
            # If regeneration failed/returned empty, use a fallback question to prevent dead-end
            if not clarifying_q.strip():
                clarifying_q = self._build_fallback_question(
                    endpoint_states, namespace_map, session_language
                )

        return AgenticLoopResult(
            status=AgenticLoopStatus.NEEDS_INPUT,
            collected_params=merged_after,
            clarifying_question=clarifying_q,
            turn_count=updated_turn_count,
        )

    async def stream_run_turn(
        self,
        chat_id: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        endpoint_states: List[EndpointSessionState],
        turn_count: int,
        awaiting_continuation: bool = False,
        session_language: str = "en",
        continuation_language: Optional[str] = None,
    ) -> Tuple[AgenticLoopResult, List[str]]:
        """Process one user turn like :meth:`run_turn` but stream clarifying question tokens.

        Delegates extraction to
        :meth:`~param_extractor.ParamExtractionModule.stream_forward` so
        ``clarifying_question`` tokens are captured as they arrive from the LLM.
        All session management is identical to :meth:`run_turn`.

        Returns:
            Tuple of ``(AgenticLoopResult, question_tokens)``.
            ``question_tokens`` is the list of streamed token strings for the
            clarifying question, or an empty list when no question is needed.
        """
        if not endpoint_states:
            logger.debug(
                f"MultiEndpointAgenticLoop: no endpoints provided for chat_id={chat_id} — returning COMPLETED"
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.COMPLETED,
                    collected_params={},
                    clarifying_question="",
                    turn_count=turn_count,
                ),
                [],
            )

        num_endpoints = len(endpoint_states)
        max_turns, continuation_turn = self._compute_turn_limits(num_endpoints)
        updated_turn_count = turn_count + 1

        logger.info(
            f"MultiEndpointAgenticLoop: multi loop turn started | event_type=multi_loop_turn_started"
            f" chat_id={chat_id} turn_count={turn_count} endpoint_count={num_endpoints}"
            f" incomplete_count={sum(1 for s in endpoint_states if not s.completed)}"
        )

        # Step 0 — Continuation decision
        if awaiting_continuation:
            wants_to_continue = detect_continuation_response(user_message)
            if wants_to_continue:
                logger.info(
                    f"MultiEndpointAgenticLoop: continuation user accepted | event_type=continuation_user_accepted"
                    f" chat_id={chat_id} turn_count={turn_count}"
                )
                awaiting_continuation = False
            else:
                logger.info(
                    f"MultiEndpointAgenticLoop: continuation user declined | event_type=continuation_user_declined"
                    f" chat_id={chat_id} turn_count={turn_count} status=max_turns_reached"
                )
                return (
                    AgenticLoopResult(
                        status=AgenticLoopStatus.MAX_TURNS_REACHED,
                        collected_params=self._merged_collected(endpoint_states),
                        clarifying_question="",
                        turn_count=updated_turn_count,
                    ),
                    [],
                )

        # Step 1 — Turn limit guard
        if turn_count >= max_turns:
            logger.warning(
                f"MultiEndpointAgenticLoop: turn limit reached | event_type=turn_limit_reached"
                f" chat_id={chat_id} turn_count={turn_count} max_turns={max_turns}"
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.MAX_TURNS_REACHED,
                    collected_params=self._merged_collected(endpoint_states),
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 2 — Build merged schema + param owner map + namespace map
        merged_schema, param_owners, namespace_map = self._build_merged_schema(
            endpoint_states
        )

        # Step 3 — Namespaced already_collected for the LLM extractor
        merged_already_collected = self._build_namespaced_already_collected(
            endpoint_states, namespace_map
        )

        # Step 4 — Stream-extract params from the current user message
        intent_groups = self._build_intent_groups(
            endpoint_states, merged_already_collected, namespace_map
        )
        _t0 = time.time()
        try:
            question_tokens, extraction = await self._param_extractor.stream_forward(
                user_message=user_message,
                params_schema=merged_schema,
                conversation_history=conversation_history,
                already_collected=merged_already_collected,
                session_language=session_language,
                turn_count=turn_count,
                intent_groups=intent_groups,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.debug(
                f"MultiEndpointAgenticLoop: param extraction complete | event_type=param_extraction_complete"
                f" chat_id={chat_id} turn_count={turn_count}"
                f" extracted_count={len(extraction['extracted_params'])} duration_ms={_duration_ms}"
            )
        except Exception as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.error(
                f"MultiEndpointAgenticLoop: param extraction failed | event_type=param_extraction_failed"
                f" chat_id={chat_id} turn_count={turn_count}"
                f" error_id={generate_error_id()} duration_ms={_duration_ms} exc={exc}"
            )
            await self._save_session(
                chat_id,
                endpoint_states,
                updated_turn_count,
                awaiting_continuation=awaiting_continuation,
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.NEEDS_INPUT,
                    collected_params=self._merged_collected(endpoint_states),
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 5 — Distribute extracted params to owning endpoints
        previously_completed_indices_stream: set[int] = {
            i for i, s in enumerate(endpoint_states) if s.completed
        }
        self._distribute_params(
            extraction["extracted_params"], endpoint_states, param_owners, namespace_map
        )

        # Step 5b — If the user gave a single value that was duplicated across
        # multiple endpoints' conflicting params, clear the duplicates so the loop
        # re-asks for each endpoint's params separately on the next turn.
        enforced = self._enforce_sequential_conflicting_params(
            endpoint_states, namespace_map
        )

        # Step 5c — If multiple endpoints with non-conflicting param names were all
        # completed by the same single-turn user response, keep only the first newly-
        # completed endpoint and clear the rest so they are asked for separately.
        enforced = (
            self._enforce_sequential_parallel_completion(
                endpoint_states, previously_completed_indices_stream
            )
            or enforced
        )

        # Step 6 — Check global completion
        all_done = all(state.completed for state in endpoint_states)
        merged_after = self._merged_collected(endpoint_states)

        if all_done:
            logger.info(
                f"MultiEndpointAgenticLoop: all endpoints completed | event_type=multi_loop_all_endpoints_completed"
                f" chat_id={chat_id} turn_count={turn_count} endpoint_count={num_endpoints}"
                f" status=completed duration_ms={_duration_ms}"
            )
            await self._save_session(
                chat_id,
                endpoint_states,
                updated_turn_count,
                awaiting_continuation=False,
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.COMPLETED,
                    collected_params=merged_after,
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 7 — Still missing params
        logger.debug(
            f"MultiEndpointAgenticLoop: loop needs input | event_type=loop_needs_input"
            f" chat_id={chat_id} turn_count={turn_count}"
            f" missing_params={extraction['missing_required']} status=needs_input"
        )

        if updated_turn_count == continuation_turn:
            logger.info(
                f"MultiEndpointAgenticLoop: continuation threshold reached | event_type=continuation_threshold_reached"
                f" chat_id={chat_id} turn_count={turn_count} continuation_turn={continuation_turn}"
                f" missing_count={len(extraction['missing_required'])}"
            )
            effective_lang = continuation_language or session_language
            continuation_q = _CONTINUATION_QUESTIONS.get(
                effective_lang, CONTINUATION_QUESTION
            )
            await self._save_session(
                chat_id, endpoint_states, updated_turn_count, awaiting_continuation=True
            )
            words = continuation_q.split(" ")
            continuation_tokens = [
                w + " " if i < len(words) - 1 else w for i, w in enumerate(words)
            ]
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.AWAITING_CONTINUATION_DECISION,
                    collected_params=merged_after,
                    clarifying_question=continuation_q,
                    turn_count=updated_turn_count,
                ),
                continuation_tokens,
            )

        await self._save_session(
            chat_id, endpoint_states, updated_turn_count, awaiting_continuation=False
        )

        # If enforcement cleared duplicate params from a later endpoint and the LLM's
        # question is stale (it said "none" believing all params were satisfied), we
        # must regenerate a fresh question that covers the remaining missing params.
        # The stale question_tokens are also discarded so the workflow falls back to
        # streaming the regenerated question as a single chunk.
        clarifying_q = extraction["clarifying_question"]
        final_question_tokens = question_tokens
        if enforced and clarifying_q.strip().lower() == "none":
            clarifying_q = await self._regenerate_question_after_enforcement(
                endpoint_states, merged_schema, namespace_map, session_language
            )
            final_question_tokens = []
            # If regeneration failed/returned empty, use a fallback question to prevent dead-end
            if not clarifying_q.strip():
                clarifying_q = self._build_fallback_question(
                    endpoint_states, namespace_map, session_language
                )

        return (
            AgenticLoopResult(
                status=AgenticLoopStatus.NEEDS_INPUT,
                collected_params=merged_after,
                clarifying_question=clarifying_q,
                turn_count=updated_turn_count,
            ),
            final_question_tokens,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _regenerate_question_after_enforcement(
        self,
        endpoint_states: List[EndpointSessionState],
        merged_schema: List[Dict[str, Any]],
        namespace_map: Dict[str, Tuple[int, str]],
        session_language: str,
    ) -> str:
        """Generate a fresh clarifying question after sequential param enforcement.

        Called when :meth:`_enforce_sequential_conflicting_params` cleared
        duplicate values from a later endpoint, making it incomplete again after
        the LLM had already reported all params as collected (clarifying_question
        == "none").  The stale "none" question is discarded and a new question is
        produced from the updated endpoint states.

        An empty user message is used so the extractor performs no new extraction
        — it only observes what is still missing in the updated ``already_collected``
        and generates the appropriate clarifying question for those params.

        Args:
            endpoint_states: Per-endpoint states after enforcement (some may have
                had params cleared and ``completed`` reset to ``False``).
            merged_schema: The merged parameter schema built earlier this turn.
            namespace_map: Namespacing map from :meth:`_build_merged_schema`.
            session_language: Language code for the clarifying question.

        Returns:
            A natural-language question for the still-missing params, or an empty
            string if regeneration fails or no params remain missing.
        """
        updated_already_collected = self._build_namespaced_already_collected(
            endpoint_states, namespace_map
        )
        updated_intent_groups = self._build_intent_groups(
            endpoint_states, updated_already_collected, namespace_map
        )
        if not updated_intent_groups:
            return ""
        try:
            regen = await asyncio.to_thread(
                self._param_extractor,
                "",  # empty — no new params to extract, just generate the question
                merged_schema,
                [],
                updated_already_collected,
                session_language,
                0,  # turn_count: no acknowledgment needed for regenerated questions
                updated_intent_groups,
            )
            question = regen["clarifying_question"]
            if question.strip().lower() == "none":
                return ""
            return question
        except Exception as exc:
            logger.warning(
                f"MultiEndpointAgenticLoop: failed to regenerate clarifying question after enforcement | exc={exc}"
            )
            return ""

    def _build_fallback_question(
        self,
        endpoint_states: List[EndpointSessionState],
        namespace_map: Dict[str, Tuple[int, str]],
        session_language: str,
    ) -> str:
        """Build a generic fallback question when LLM regeneration fails.

        Constructs a safe, non-empty question from the still-missing required
        params when ``_regenerate_question_after_enforcement()`` returns empty
        due to failed regeneration or empty intent groups. This prevents the
        workflow from dead-ending with an empty clarifying_question when
        endpoints are still incomplete.

        The question lists all missing required param descriptions from incomplete
        endpoints, providing clear guidance to the user on what information is
        still needed.

        Args:
            endpoint_states: Per-endpoint state objects.
            namespace_map: Mapping of ``namespaced_name → (ep_idx, original_name)``
                for conflicting params.
            session_language: Language code for the question.

        Returns:
            A generic fallback question listing missing params, or a generic
            continuation prompt if no missing params are found.
        """
        conflicting_original_names: set[str] = {
            orig_name for (_, orig_name) in namespace_map.values()
        }

        # Collect all missing required param descriptions from incomplete endpoints
        missing_descriptions: List[str] = []
        seen_descriptions: set[str] = set()

        for state in endpoint_states:
            if state.completed:
                continue
            params_schema: List[Dict[str, Any]] = state.endpoint.get("params", [])
            for param in params_schema:
                if not isinstance(param, dict):
                    continue
                if not param.get("required", False):
                    continue
                name: str = param.get("name", "")
                if not name:
                    continue

                # Check if already collected (accounting for conflicting params)
                if name in conflicting_original_names:
                    if name in state.collected_params:
                        continue
                else:
                    # Non-conflicting: check union of all collected_params
                    found = False
                    for s in endpoint_states:
                        if name in s.collected_params:
                            found = True
                            break
                    if found:
                        continue

                # Add description if not already seen (deduplication)
                desc = strip_format_hints(str(param.get("description", name)))
                if desc and desc not in seen_descriptions:
                    missing_descriptions.append(desc)
                    seen_descriptions.add(desc)

        # Build the fallback question
        if missing_descriptions:
            items_str = ", ".join(missing_descriptions)
            # Localized fallback prompts
            if session_language == "et":
                return f"Palun sisestage järgmine teave: {items_str}"
            elif session_language == "ru":
                return f"Пожалуйста, предоставьте следующую информацию: {items_str}"
            else:  # Default to English
                return f"Please provide the following: {items_str}"
        else:
            # No specific missing params found — use generic continuation prompt
            if session_language == "et":
                return CONTINUATION_QUESTION_ET
            elif session_language == "ru":
                return CONTINUATION_QUESTION_RU
            else:  # Default to English
                return CONTINUATION_QUESTION

    def _compute_turn_limits(self, num_endpoints: int) -> Tuple[int, int]:
        """Compute max_turns and continuation_turn based on endpoint count.

        Args:
            num_endpoints: Number of endpoints in the current session.

        Returns:
            Tuple of (max_turns, continuation_turn).
        """
        is_multi_intent = num_endpoints > 1
        max_turns = (
            MULTI_INTENT_MAX_TURNS
            if is_multi_intent
            else min(3 * num_endpoints, MULTI_API_MAX_TURNS)
        )
        continuation_turn = (
            MULTI_INTENT_CONTINUATION_TURN if is_multi_intent else num_endpoints + 1
        )
        return max_turns, continuation_turn

    def _build_merged_schema(
        self,
        endpoint_states: List[EndpointSessionState],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]], Dict[str, Tuple[int, str]]]:
        """Build a merged parameter schema from all incomplete endpoints.

        Uses a three-pass approach:

        * **Pass 0** — Counts how many *incomplete* endpoints define each param name.
          This detects conflicts among endpoints that are actively requesting params.
          Completed endpoints do not participate in conflict detection.  Names that
          appear in more than one incomplete endpoint are "conflicting" and will
          be namespaced in Pass 1 so each intent can supply an independent value.

        * **Pass 1** — Iterates over *incomplete* endpoints only to build
          ``merged_schema``.  *Non-conflicting* params are deduplicated by name;
          the first occurrence's definition (type, description) is used.  A type
          conflict is logged as a warning.  A param is ``required=True`` in the
          merged schema if it is required in **any** owning endpoint.
          *Conflicting* params are emitted as ``{name}__{endpoint_idx}`` so all
          intents receive their own entry in the schema simultaneously.  Duplicate
          values across endpoints are handled after distribution by
          ``_enforce_sequential_conflicting_params``.

        * **Pass 2** — Iterates over *completed* endpoints and adds them to
          ``param_owners`` for any *non-namespaced* param name already present
          in the merged schema.  This ensures that if a shared non-conflicting
          param is re-extracted or corrected in a later turn the updated value
          is distributed back to completed endpoint states.  Completed endpoints
          do not affect conflict detection or namespacing.

        Args:
            endpoint_states: All per-endpoint states.

        Returns:
            A 3-tuple of:
            - ``merged_schema``: Deduplicated/namespaced list of param dicts.
            - ``param_owners``: Mapping of param key (possibly namespaced) →
              list of endpoint indices that own it.
            - ``namespace_map``: Mapping of ``namespaced_name →
              (ep_idx, original_name)`` for all conflicting params.  Empty dict
              when no conflicts exist.
        """
        # Pass 0: detect conflicting param names across incomplete endpoints only.
        # Conflicts are determined solely among endpoints that need params, not including
        # completed endpoints. This ensures no unnecessary namespacing when only one
        # endpoint is incomplete.
        name_to_endpoints: Dict[str, set[int]] = {}
        for idx, state in enumerate(endpoint_states):
            if state.completed:
                continue
            for param in state.endpoint.get("params", []):
                if not isinstance(param, dict):
                    continue
                name: str = param.get("name", "")
                if name:
                    name_to_endpoints.setdefault(name, set()).add(idx)
        conflicting_names: set[str] = {
            n for n, eps in name_to_endpoints.items() if len(eps) > 1
        }

        merged_schema: List[Dict[str, Any]] = []
        param_owners: Dict[str, List[int]] = {}
        seen_types: Dict[str, str] = {}  # key → type string of first occurrence
        namespace_map: Dict[str, Tuple[int, str]] = {}

        # Pass 1: build merged_schema from incomplete endpoints only.
        for idx, state in enumerate(endpoint_states):
            if state.completed:
                continue
            params_schema: List[Dict[str, Any]] = state.endpoint.get("params", [])
            for param in params_schema:
                if not isinstance(param, dict):
                    continue
                name = param.get("name", "")
                if not name:
                    continue
                param_type: str = str(param.get("type") or "string")

                if name in conflicting_names:
                    # Conflicting — emit as a namespaced key owned by this endpoint only.
                    ns_name: str = f"{name}__{idx}"
                    ns_param = dict(param)
                    ns_param["name"] = ns_name
                    merged_schema.append(ns_param)
                    param_owners[ns_name] = [idx]
                    seen_types[ns_name] = param_type
                    namespace_map[ns_name] = (idx, name)
                elif name not in param_owners:
                    # Non-conflicting first occurrence — add to merged schema.
                    merged_schema.append(dict(param))
                    param_owners[name] = [idx]
                    seen_types[name] = param_type
                else:
                    # Non-conflicting duplicate name — update owner list and required flag.
                    param_owners[name].append(idx)

                    # Warn on type conflicts
                    if param_type != seen_types[name]:
                        logger.warning(
                            f"MultiEndpointAgenticLoop: param '{name}' has conflicting types "
                            f"across endpoints ({seen_types[name]} vs {param_type}). Using first occurrence's type."
                        )

                    # Promote to required if required in any owner
                    if param.get("required", False):
                        for merged_param in merged_schema:
                            if merged_param.get("name") == name:
                                merged_param["required"] = True
                                break

        # Pass 2: register completed endpoints as owners for non-namespaced params
        # that appear in the merged schema.  This ensures that if a shared
        # non-conflicting param is re-extracted or corrected in a later turn the
        # updated value is also written back to already-completed endpoint states.
        #
        # Also apply the same "required in any owner" promotion used in Pass 1.
        for idx, state in enumerate(endpoint_states):
            if not state.completed:
                continue
            params_schema = state.endpoint.get("params", [])
            for param in params_schema:
                if not isinstance(param, dict):
                    continue
                name = param.get("name", "")
                # Only register for non-namespaced (non-conflicting) params.
                if name and name in param_owners:
                    param_owners[name].append(idx)
                    if param.get("required", False):
                        for merged_param in merged_schema:
                            if merged_param.get("name") == name:
                                merged_param["required"] = True
                                break

        return merged_schema, param_owners, namespace_map

    def _build_namespaced_already_collected(
        self,
        endpoint_states: List[EndpointSessionState],
        namespace_map: Dict[str, Tuple[int, str]],
    ) -> Dict[str, Any]:
        """Build the ``already_collected`` dict for the LLM, namespacing conflicting params.

        For non-conflicting params: merged union across all endpoints (original
        names), excluding any names that are conflicting.
        For conflicting params: namespaced keys (e.g. ``startDate__0``,
        ``startDate__1``), read from each endpoint's own ``collected_params``.

        Args:
            endpoint_states: All per-endpoint states.
            namespace_map: Mapping of ``namespaced_name → (ep_idx, original_name)``
                produced by ``_build_merged_schema()``. Empty dict when no conflicts.

        Returns:
            Dict of already-collected param values, ready to pass to the LLM
            extractor.
        """
        if not namespace_map:
            # Fast path — no conflicts, use simple merge.
            merged: Dict[str, Any] = {}
            for state in endpoint_states:
                merged.update(state.collected_params)
            return merged

        conflicting_original_names: set[str] = {
            orig_name for (_, orig_name) in namespace_map.values()
        }

        # Non-conflicting: merged union, excluding conflicting original names.
        result: Dict[str, Any] = {
            k: v
            for state in endpoint_states
            for k, v in state.collected_params.items()
            if k not in conflicting_original_names
        }

        # Conflicting: namespaced keys from each endpoint's own collected_params.
        result.update(
            {
                ns_name: endpoint_states[ep_idx].collected_params[original_name]
                for ns_name, (ep_idx, original_name) in namespace_map.items()
                if original_name in endpoint_states[ep_idx].collected_params
            }
        )

        return result

    def _build_intent_groups(
        self,
        endpoint_states: List[EndpointSessionState],
        already_collected: Dict[str, Any],
        namespace_map: Dict[str, Tuple[int, str]],
    ) -> List[Dict[str, Any]]:
        """Build intent groups for multi-intent clarifying question separation.

        For each incomplete endpoint, collects the required params not yet present
        in ``already_collected``, strips format hints from their descriptions, and
        returns a list of groups suitable for the ``intent_groups`` field of
        :class:`~tool_classifier.param_extractor.ParamExtractionSignature`.

        **Conflicting params** (those in ``namespace_map``) are checked against
        each endpoint's own ``collected_params`` rather than the merged
        ``already_collected``, and deduplication across groups is suppressed so
        that both intents independently list their date-range params.

        **Non-conflicting params** preserve the existing cross-endpoint
        deduplication via ``seen_param_names``.

        Returns an empty list when no groups have missing required params.
        Returns a single-element list when only one endpoint has outstanding
        params — the endpoint name is still included so the LLM can phrase the
        question with proper context (e.g. "For the parliament member participation
        stats, what is the start and end date?").

        Args:
            endpoint_states: All per-endpoint states.
            already_collected: Namespaced already-collected dict (from
                ``_build_namespaced_already_collected()``) used for non-conflicting
                param lookup.
            namespace_map: Mapping of ``namespaced_name → (ep_idx, original_name)``
                produced by ``_build_merged_schema()``.

        Returns:
            List of ``{"intent": str, "missing_param_descriptions": [str, ...]}``
            dicts, or an empty list if no intent has missing required params.
        """
        conflicting_original_names: set[str] = {
            orig_name for (_, orig_name) in namespace_map.values()
        }

        groups: List[Dict[str, Any]] = []
        seen_param_names: set[str] = set()
        for state in endpoint_states:
            if state.completed:
                continue
            params_schema: List[Dict[str, Any]] = state.endpoint.get("params", [])
            missing_descriptions: List[str] = []
            for param in params_schema:
                if not isinstance(param, dict):
                    continue
                if not param.get("required", False):
                    continue
                name: str = param.get("name", "")
                if not name:
                    continue

                if name in conflicting_original_names:
                    # Conflicting param: check per-endpoint collected_params.
                    # Do NOT deduplicate across groups — each intent lists its own entry.
                    if name in state.collected_params:
                        continue
                else:
                    # Non-conflicting: use merged already_collected + seen_param_names.
                    if name in already_collected:
                        continue
                    if name in seen_param_names:
                        continue
                    seen_param_names.add(name)

                desc = strip_format_hints(str(param.get("description", name)))
                missing_descriptions.append(desc)
            if missing_descriptions:
                intent_name: str = state.endpoint.get("name") or state.endpoint.get(
                    "description", ""
                )
                groups.append(
                    {
                        "intent": intent_name,
                        "missing_param_descriptions": missing_descriptions,
                    }
                )
        if not groups:
            return []
        return groups

    def _distribute_params(
        self,
        extracted_params: Dict[str, Any],
        endpoint_states: List[EndpointSessionState],
        param_owners: Dict[str, List[int]],
        namespace_map: Dict[str, Tuple[int, str]],
    ) -> None:
        """Distribute extracted param values to their owning endpoint states.

        For **namespaced** keys (those in ``namespace_map``): the value is
        reverse-translated and written to the specific endpoint's
        ``collected_params`` under the original param name.

        For **non-namespaced** keys: the existing owner-broadcast logic writes
        the value into every owning endpoint's ``collected_params``.

        Marks an endpoint as ``completed=True`` once all its required params are
        present.  For completed endpoints a write only occurs when the incoming
        value differs from the stored one; the override is logged at WARNING
        level.

        Args:
            extracted_params: Dict of newly extracted param values.
            endpoint_states: Per-endpoint state objects (mutated in-place).
            param_owners: Mapping of param key → list of owning endpoint indices.
            namespace_map: Mapping of ``namespaced_name → (ep_idx, original_name)``
                produced by ``_build_merged_schema()``.
        """
        for param_name, value in extracted_params.items():
            if param_name in namespace_map:
                # Namespaced key — write to the specific endpoint under the original name.
                ep_idx, original_name = namespace_map[param_name]
                state = endpoint_states[ep_idx]
                if state.completed:
                    existing = state.collected_params.get(original_name)
                    if value == existing:
                        continue
                    logger.warning(
                        f"MultiEndpointAgenticLoop: overwriting completed endpoint param"
                        f" | event_type=endpoint_param_overwrite"
                        f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                        f" param_name={original_name} old_value={repr(existing)} new_value={repr(value)}"
                    )
                state.collected_params[original_name] = value
            else:
                # Non-namespaced key — broadcast to all owning endpoints.
                owner_indices = param_owners.get(param_name, [])
                for idx in owner_indices:
                    state = endpoint_states[idx]
                    if state.completed:
                        existing = state.collected_params.get(param_name)
                        if value == existing:
                            continue
                        logger.warning(
                            f"MultiEndpointAgenticLoop: overwriting completed endpoint param"
                            f" | event_type=endpoint_param_overwrite"
                            f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                            f" param_name={param_name} old_value={repr(existing)} new_value={repr(value)}"
                        )
                    state.collected_params[param_name] = value

        # Check completion for each non-completed endpoint
        for state in endpoint_states:
            if state.completed:
                continue
            params_schema: List[Dict[str, Any]] = state.endpoint.get("params", [])
            required_names = {
                p["name"]
                for p in params_schema
                if isinstance(p, dict) and p.get("required", False)
            }
            if required_names.issubset(state.collected_params.keys()):
                state.completed = True
                logger.debug(
                    f"MultiEndpointAgenticLoop: endpoint completed | event_type=endpoint_completed"
                    f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                )

    def _enforce_sequential_parallel_completion(
        self,
        endpoint_states: List[EndpointSessionState],
        previously_completed_indices: set[int],
    ) -> bool:
        """Clear params from endpoints newly completed with duplicate values in this turn.

        When endpoints have *different* param names (non-conflicting), the existing
        :meth:`_enforce_sequential_conflicting_params` does not apply.  Instead,
        if a single user response satisfies **multiple** endpoints at once with the
        **same underlying values** (e.g. one date range fills both ``start``/``end``
        on the electricity prices endpoint *and* ``startDate``/``endDate`` on the
        parliament stats endpoint with identical dates), all but the *first*
        (lowest-index) such endpoint have their required params cleared so the loop
        can ask for each endpoint's params separately.

        If the newly-completed endpoints have **disjoint** required-param value sets
        (e.g. the user provided April dates for electricity and January dates for
        parliament in a single reply), both are kept as completed because the user
        intentionally supplied separate values for each intent.

        Endpoints that were already completed *before* this turn are untouched.

        Args:
            endpoint_states: Per-endpoint state objects (mutated in-place).
            previously_completed_indices: Set of endpoint indices that were already
                completed at the start of this turn (before distribution).

        Returns:
            ``True`` if any endpoint had its params cleared; ``False`` otherwise.
        """
        newly_completed = [
            i
            for i, state in enumerate(endpoint_states)
            if state.completed and i not in previously_completed_indices
        ]

        if len(newly_completed) <= 1:
            return False

        first_ep_idx = newly_completed[0]
        first_state = endpoint_states[first_ep_idx]
        first_params_schema: List[Dict[str, Any]] = first_state.endpoint.get(
            "params", []
        )
        first_required_names: set[str] = {
            p["name"]
            for p in first_params_schema
            if isinstance(p, dict) and p.get("required", False) and p.get("name")
        }
        # Normalize values to hashable strings to handle unhashable types (lists, dicts).
        first_values_normalized: set[str] = {
            self._normalize_value_to_hashable(first_state.collected_params[name])
            for name in first_required_names
            if name in first_state.collected_params
        }

        cleared_any = False
        for ep_idx in newly_completed[1:]:
            state = endpoint_states[ep_idx]
            params_schema: List[Dict[str, Any]] = state.endpoint.get("params", [])
            required_names: set[str] = {
                p["name"]
                for p in params_schema
                if isinstance(p, dict) and p.get("required", False) and p.get("name")
            }
            # Normalize values to hashable strings to handle unhashable types (lists, dicts).
            later_values_normalized: set[str] = {
                self._normalize_value_to_hashable(state.collected_params[name])
                for name in required_names
                if name in state.collected_params
            }

            # If both endpoints have non-empty value sets and they are completely
            # disjoint, the user deliberately provided different values for each
            # intent in one reply — keep this endpoint completed.
            if (
                first_values_normalized
                and later_values_normalized
                and later_values_normalized.isdisjoint(first_values_normalized)
            ):
                logger.debug(
                    f"MultiEndpointAgenticLoop: parallel endpoints completed with distinct values — keeping both"
                    f" | event_type=enforcement_parallel_completion"
                    f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                    f" first_endpoint_name={first_state.endpoint.get('name', '<unnamed>')}"
                )
                continue

            # Value sets are exactly equal — the LLM likely applied the same answer
            # to multiple endpoints. Clear this endpoint so the loop re-asks.
            if later_values_normalized == first_values_normalized:
                logger.debug(
                    f"MultiEndpointAgenticLoop: clearing duplicate params from endpoint"
                    f" | event_type=enforcement_sequential_conflict"
                    f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                )
                for param in params_schema:
                    if isinstance(param, dict) and param.get("required", False):
                        name: str = param.get("name", "")
                        if name:
                            state.collected_params.pop(name, None)
                state.completed = False
                cleared_any = True

        return cleared_any

    def _enforce_sequential_conflicting_params(
        self,
        endpoint_states: List[EndpointSessionState],
        namespace_map: Dict[str, Tuple[int, str]],
    ) -> bool:
        """Clear duplicate conflicting param values from later endpoints.

        After distribution, if multiple endpoints received **identical** values
        for all their shared conflicting params it means the user gave a single
        value (e.g. one date range) that the LLM duplicated across all namespaced
        slots.  In that case, keep the values only on the **first** such endpoint
        and clear the rest, so the loop re-asks the user for each remaining
        endpoint's params separately on the next turn.

        When the user provides **distinct** values for each endpoint's conflicting
        params (e.g. two different date ranges) the values will differ, and this
        method leaves all endpoints untouched.

        Args:
            endpoint_states: Per-endpoint state objects (mutated in-place).
            namespace_map: Mapping of ``namespaced_name → (ep_idx, original_name)``
                produced by ``_build_merged_schema()``.  Empty when there are no
                conflicting params.

        Returns:
            ``True`` if any later endpoint had its duplicate params cleared;
            ``False`` otherwise.  The caller uses this flag to detect when the
            LLM's already-generated clarifying question is stale (it said "none"
            because it thought all params were satisfied) and needs regenerating.
        """
        if not namespace_map:
            return False

        # Build a map of endpoint_idx → set of original conflicting param names.
        ep_conflicting: Dict[int, set[str]] = {}
        for ep_idx, original_name in namespace_map.values():
            ep_conflicting.setdefault(ep_idx, set()).add(original_name)

        if len(ep_conflicting) < 2:
            return False

        # Determine the shared conflicting param names across all participating endpoints.
        shared_names: set[str] = set.intersection(*ep_conflicting.values())
        if not shared_names:
            return False

        ep_indices = sorted(ep_conflicting.keys())
        first_ep_idx = ep_indices[0]
        first_state = endpoint_states[first_ep_idx]

        # Collect the first endpoint's values for the shared params (only those
        # that were actually filled this turn). Normalize to hashable strings to
        # handle unhashable types (lists, dicts).
        first_values: Dict[str, str] = {
            name: self._normalize_value_to_hashable(first_state.collected_params[name])
            for name in shared_names
            if name in first_state.collected_params
        }
        if not first_values:
            # First endpoint has no values yet — nothing to compare.
            return False

        # Clear later endpoints that received the same values as the first.
        cleared_any = False
        for ep_idx in ep_indices[1:]:
            state = endpoint_states[ep_idx]
            later_values: Dict[str, str] = {
                name: self._normalize_value_to_hashable(state.collected_params[name])
                for name in shared_names
                if name in state.collected_params
            }
            if not later_values:
                continue
            if later_values == first_values:
                logger.debug(
                    f"MultiEndpointAgenticLoop: clearing duplicate conflicting params from endpoint"
                    f" | event_type=enforcement_sequential_conflict"
                    f" endpoint_name={state.endpoint.get('name', '<unnamed>')}"
                )
                for name in shared_names:
                    state.collected_params.pop(name, None)
                state.completed = False
                cleared_any = True
        return cleared_any

    @staticmethod
    def _normalize_value_to_hashable(value: Any) -> str:  # noqa: ANN401
        """Convert a value to a hashable string for safe comparison.

        Uses JSON serialization with `repr` fallback for non-JSON-serializable
        types (lists, dicts, custom objects, etc.). Ensures that unhashable
        values like lists and dicts can be safely compared in sets or as dict keys.

        Args:
            value: Any value to normalize.

        Returns:
            A string that uniquely represents the value.
        """
        try:
            # JSON serialization with sort_keys ensures consistent repr across identical
            # value structures, using str as fallback for non-JSON-serializable objects.
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            # Fallback to repr for types that can't be JSON-serialized.
            return repr(value)

    @staticmethod
    def _merged_collected(
        endpoint_states: List[EndpointSessionState],
    ) -> Dict[str, Any]:
        """Return the union of all per-endpoint collected_params.

        Later endpoints overwrite earlier ones on key conflicts, consistent with
        how the single-endpoint loop merges extraction results.
        """
        merged: Dict[str, Any] = {}
        for state in endpoint_states:
            merged.update(state.collected_params)
        return merged

    async def _save_session(
        self,
        chat_id: str,
        endpoint_states: List[EndpointSessionState],
        turn_count: int,
        awaiting_continuation: bool = False,
    ) -> None:
        """Persist updated loop state to the Redis session store.

        Only updates fields owned by the loop (parallel_endpoints, turn_count,
        awaiting_continuation).  Workflow-owned fields are preserved.
        A missing or unavailable session is logged but never raises.

        Args:
            chat_id: Unique conversation identifier.
            endpoint_states: Updated per-endpoint states to persist.
            turn_count: Updated turn counter.
            awaiting_continuation: Updated continuation flag.
        """
        try:
            if self._session_store is None:
                logger.debug(
                    f"MultiEndpointAgenticLoop: session store unavailable — skipping save for chat_id={chat_id}"
                )
                return
            await self._session_store.update(
                chat_id,
                parallel_endpoints=endpoint_states,
                turn_count=turn_count,
                awaiting_continuation=awaiting_continuation,
            )
        except Exception as exc:
            logger.error(
                f"MultiEndpointAgenticLoop: failed to save session | event_type=session_save_failed"
                f" chat_id={chat_id} turn_count={turn_count} error_id={generate_error_id()} exc={exc}"
            )

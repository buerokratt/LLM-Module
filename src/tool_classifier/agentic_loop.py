"""Standalone agentic loop for multi-turn parameter collection."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from src.loki_logger import LokiLogger
from src.utils.api_tool_session_store import APIToolSessionStore
from tool_classifier.constants import (
    CONTINUATION_QUESTION,
    CONTINUATION_QUESTION_ET,
    CONTINUATION_QUESTION_RU,
    CONTINUATION_TURN,
)
from tool_classifier.continuation_utils import detect_continuation_response
from tool_classifier.enums import AgenticLoopStatus
from tool_classifier.models import AgenticLoopResult
from tool_classifier.param_extractor import ParamExtractionModule

logger = LokiLogger(service_name="api-tool-calling")

_CONTINUATION_QUESTIONS: dict[str, str] = {
    "en": CONTINUATION_QUESTION,
    "et": CONTINUATION_QUESTION_ET,
    "ru": CONTINUATION_QUESTION_RU,
}


class AgenticLoop:
    """Stateless multi-turn parameter collection loop.

    Each call to run_turn() represents one user message / one loop iteration.
    The loop carries no internal state — all state is passed in as arguments.
    Redis persistence (load from session before calling, save inside run_turn)
    is handled here so callers only need to act on the returned AgenticLoopResult.

    Typical usage::

        loop = AgenticLoop(
            session_store=app.state.session_store,
            param_extractor=ParamExtractionModule(),
        )

        result = await loop.run_turn(
            chat_id=request.chatId,
            user_message=request.message,
            conversation_history=request.conversationHistory,
            params_schema=endpoint["params_schema"],
            collected_params=session.collected_params,
            turn_count=session.turn_count,
            max_turns=session.max_turns,
        )

        if result.status == AgenticLoopStatus.COMPLETED:
            # All params ready — call the API, then delete session
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
        session_store: APIToolSessionStore,
        param_extractor: ParamExtractionModule,
    ) -> None:
        """Initialise the loop with an injected session store and param extractor.

        Args:
            session_store: Redis-backed store used to persist loop state between
                HTTP requests. Injected to allow easy mocking in tests.
            param_extractor: DSPy module that extracts parameter values from a
                user message. Injected to allow easy mocking in tests.
        """
        self._session_store = session_store
        self._param_extractor = param_extractor

    async def run_turn(
        self,
        chat_id: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        params_schema: List[Dict[str, Any]],
        collected_params: Dict[str, Any],
        turn_count: int,
        max_turns: int = 5,
        awaiting_continuation: bool = False,
        continuation_turn: int = CONTINUATION_TURN,
        session_language: str = "en",
        continuation_language: Optional[str] = None,
        seeded_params: Optional[Dict[str, Any]] = None,
    ) -> AgenticLoopResult:
        """Process one user turn of the parameter-collection loop.

        Steps:
        0. Continuation decision — if ``awaiting_continuation`` is True, detect
           whether the user said yes (keep going) or no (fall back to RAG).
           A "no" or ambiguous response returns MAX_TURNS_REACHED immediately.
        1. Guard — return MAX_TURNS_REACHED if the turn limit is reached.
        2. Extract — call ParamExtractionModule for newly mentioned params.
        3. Merge — combine prior collected params with newly extracted ones.
           Prior values are authoritative (not overwritten by this turn).
        4. Completeness check — if all required params are present, save state
           and return COMPLETED.
        5. Incomplete — if this is exactly the ``continuation_turn``, save state
           and return AWAITING_CONTINUATION_DECISION with a yes/no question.
           Otherwise return NEEDS_INPUT with the clarifying question.

        The returned turn_count is always input turn_count + 1.
        Session state is saved automatically on COMPLETED, NEEDS_INPUT, and
        AWAITING_CONTINUATION_DECISION.
        It is NOT saved on MAX_TURNS_REACHED. It is also generally not saved
        on extraction errors, except when a continuation decision was consumed
        and the cleared ``awaiting_continuation`` state must be persisted. The
        caller is expected to delete the session on MAX_TURNS_REACHED and
        extraction errors after handling the failure.

        Args:
            chat_id: Unique conversation identifier, used as the Redis session key.
            user_message: The user's latest message for this turn.
            conversation_history: Recent conversation turns as a list of
                ``{"authorRole": str, "message": str}`` dicts.
            params_schema: Parameter schema defining what to collect. Each
                entry is a dict with at minimum ``name``, ``type``,
                ``required``, and ``description`` keys.
            collected_params: Parameter values collected in prior turns.
                These are treated as authoritative and will not be overwritten.
            turn_count: The current turn index (0-based before this call).
            max_turns: Maximum turns allowed before the loop is abandoned.
            awaiting_continuation: True when the previous turn returned
                AWAITING_CONTINUATION_DECISION and we are now processing the
                user's yes/no reply. Load this from the persisted session.
            continuation_turn: The 1-based turn count at which to ask the
                continuation question when params are still missing.
                Defaults to ``CONTINUATION_TURN`` (3).

        Returns:
            AgenticLoopResult with updated status, collected_params, and
            turn_count.
        """
        updated_turn_count = turn_count + 1

        logger.info(
            f"AgenticLoop: loop turn started | event_type=loop_turn_started chat_id={chat_id} turn_count={turn_count} max_turns={max_turns} awaiting_continuation={awaiting_continuation}"
        )

        # Seed inherited params from L2 follow-up detection — turn 0 only.
        # seeded_params take lower priority than anything already in collected_params
        # (i.e. values explicitly set by the session take precedence).
        if turn_count == 0 and seeded_params:
            collected_params = {**seeded_params, **collected_params}

        # Step 0 — Continuation decision: user is responding to the yes/no prompt
        original_awaiting_continuation = awaiting_continuation
        if awaiting_continuation:
            wants_to_continue = self._detect_continuation_response(user_message)
            if wants_to_continue:
                logger.debug(
                    f"AgenticLoop: user chose to continue on turn {turn_count} for chat_id={chat_id}"
                )
                # Reset the flag so normal extraction takes over from here.
                awaiting_continuation = False
            else:
                logger.info(
                    f"AgenticLoop: user chose to exit on turn {turn_count} for chat_id={chat_id}, "
                    "falling back to RAG"
                )
                return AgenticLoopResult(
                    status=AgenticLoopStatus.MAX_TURNS_REACHED,
                    collected_params=collected_params,
                    clarifying_question="",
                    turn_count=updated_turn_count,
                )

        # Step 1 — Turn limit guard (no session save — caller deletes)
        if turn_count >= max_turns:
            logger.warning(
                f"AgenticLoop: max_turns={max_turns} reached for chat_id={chat_id}, abandoning"
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.MAX_TURNS_REACHED,
                collected_params=collected_params,
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 2 — Extract params from the current user message
        _t0 = time.time()
        try:
            extraction = await asyncio.to_thread(
                self._param_extractor,
                user_message,
                params_schema,
                conversation_history,
                collected_params,
                session_language,
                turn_count,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.debug(
                f"AgenticLoop: param extraction complete | event_type=param_extraction_complete chat_id={chat_id} turn_count={turn_count} extracted_count={len(extraction['extracted_params'])} duration_ms={_duration_ms}"
            )
        except Exception as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.error(
                f"AgenticLoop: param extraction failed on turn {turn_count} for chat_id={chat_id}: {exc}"
            )
            # If a continuation decision was already consumed this turn, persist the
            # updated flag so the next user message is not misread as another
            # yes/no continuation response.
            if awaiting_continuation != original_awaiting_continuation:
                await self._save_session(
                    chat_id,
                    collected_params,
                    updated_turn_count,
                    awaiting_continuation=awaiting_continuation,
                )
            return AgenticLoopResult(
                status=AgenticLoopStatus.NEEDS_INPUT,
                collected_params=collected_params,
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 3 — Merge: newly extracted values override prior ones so the user
        # can correct a value they provided in an earlier turn (e.g. "actually,
        # make that Russia instead of Estonia"). Prior values are kept only for
        # params the extractor did NOT mention in this turn.
        merged_params: Dict[str, Any] = {
            **collected_params,
            **extraction["extracted_params"],
        }

        # Step 4 — Completeness check
        required_param_names = {
            p["name"]
            for p in params_schema
            if isinstance(p, dict) and p.get("required", False)
        }
        all_collected = required_param_names.issubset(merged_params.keys())

        logger.debug(
            f"AgenticLoop: params merged | event_type=params_merged chat_id={chat_id} turn_count={turn_count} required_count={len(required_param_names)} collected_count={len(merged_params)} missing_count={len(required_param_names - merged_params.keys())}"
        )

        if all_collected:
            logger.info(
                f"AgenticLoop: loop completed | event_type=loop_completed chat_id={chat_id} turn_count={turn_count} status=completed collected_count={len(merged_params)} duration_ms={_duration_ms}"
            )
            await self._save_session(
                chat_id, merged_params, updated_turn_count, awaiting_continuation=False
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.COMPLETED,
                collected_params=merged_params,
                clarifying_question="",
                turn_count=updated_turn_count,
            )

        # Step 5 — Still missing params
        logger.debug(
            f"AgenticLoop: loop needs input | event_type=loop_needs_input chat_id={chat_id} turn_count={turn_count} missing_params={extraction['missing_required']} status=needs_input"
        )

        # At exactly the continuation threshold, ask whether to keep going.
        if updated_turn_count == continuation_turn:
            logger.info(
                f"AgenticLoop: continuation threshold reached | event_type=continuation_threshold_reached chat_id={chat_id} turn_count={turn_count} continuation_turn={continuation_turn} missing_count={len(extraction['missing_required'])}"
            )
            effective_continuation_lang = continuation_language or session_language
            continuation_q = _CONTINUATION_QUESTIONS.get(
                effective_continuation_lang, CONTINUATION_QUESTION
            )
            await self._save_session(
                chat_id, merged_params, updated_turn_count, awaiting_continuation=True
            )
            return AgenticLoopResult(
                status=AgenticLoopStatus.AWAITING_CONTINUATION_DECISION,
                collected_params=merged_params,
                clarifying_question=continuation_q,
                turn_count=updated_turn_count,
            )

        await self._save_session(
            chat_id, merged_params, updated_turn_count, awaiting_continuation=False
        )
        return AgenticLoopResult(
            status=AgenticLoopStatus.NEEDS_INPUT,
            collected_params=merged_params,
            clarifying_question=extraction["clarifying_question"],
            turn_count=updated_turn_count,
        )

    async def stream_run_turn(
        self,
        chat_id: str,
        user_message: str,
        conversation_history: List[Dict[str, Any]],
        params_schema: List[Dict[str, Any]],
        collected_params: Dict[str, Any],
        turn_count: int,
        max_turns: int = 5,
        awaiting_continuation: bool = False,
        continuation_turn: int = CONTINUATION_TURN,
        session_language: str = "en",
        continuation_language: Optional[str] = None,
        seeded_params: Optional[Dict[str, Any]] = None,
    ) -> tuple[AgenticLoopResult, List[str]]:
        """Process one user turn like :meth:`run_turn` but stream clarifying_question tokens.

        Delegates extraction to
        :meth:`~param_extractor.ParamExtractionModule.stream_forward` so
        ``clarifying_question`` tokens are captured as they arrive from the LLM.
        All session management (save/delete) is identical to :meth:`run_turn`.

        Returns:
            Tuple of ``(AgenticLoopResult, question_tokens)``.
            ``question_tokens`` is the list of streamed token strings for the
            clarifying question, or an empty list when no question is needed.
        """
        updated_turn_count = turn_count + 1

        logger.info(
            f"AgenticLoop: loop turn started | event_type=loop_turn_started chat_id={chat_id} turn_count={turn_count} max_turns={max_turns} awaiting_continuation={awaiting_continuation}"
        )

        # Seed inherited params from L2 follow-up detection — turn 0 only.
        if turn_count == 0 and seeded_params:
            collected_params = {**seeded_params, **collected_params}

        # Step 0 — Continuation decision
        original_awaiting_continuation = awaiting_continuation
        if awaiting_continuation:
            wants_to_continue = self._detect_continuation_response(user_message)
            if wants_to_continue:
                logger.info(
                    f"AgenticLoop: continuation user accepted | event_type=continuation_user_accepted chat_id={chat_id} turn_count={turn_count}"
                )
                awaiting_continuation = False
            else:
                logger.info(
                    f"AgenticLoop: user chose to exit on turn {turn_count} for chat_id={chat_id}, "
                    "falling back to RAG"
                )
                return (
                    AgenticLoopResult(
                        status=AgenticLoopStatus.MAX_TURNS_REACHED,
                        collected_params=collected_params,
                        clarifying_question="",
                        turn_count=updated_turn_count,
                    ),
                    [],
                )

        # Step 1 — Turn limit guard
        if turn_count >= max_turns:
            logger.warning(
                f"AgenticLoop: max_turns={max_turns} reached for chat_id={chat_id}, abandoning"
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.MAX_TURNS_REACHED,
                    collected_params=collected_params,
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 2 — Stream-extract params from the current user message
        _t0 = time.time()
        try:
            question_tokens, extraction = await self._param_extractor.stream_forward(
                user_message=user_message,
                params_schema=params_schema,
                conversation_history=conversation_history,
                already_collected=collected_params,
                session_language=session_language,
                turn_count=turn_count,
            )
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.debug(
                f"AgenticLoop: param extraction complete | event_type=param_extraction_complete chat_id={chat_id} turn_count={turn_count} extracted_count={len(extraction['extracted_params'])} duration_ms={_duration_ms}"
            )
        except Exception as exc:
            _duration_ms = round((time.time() - _t0) * 1000, 1)
            logger.error(
                f"AgenticLoop: stream param extraction failed on turn {turn_count} for chat_id={chat_id}: {exc}"
            )
            if awaiting_continuation != original_awaiting_continuation:
                await self._save_session(
                    chat_id,
                    collected_params,
                    updated_turn_count,
                    awaiting_continuation=awaiting_continuation,
                )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.NEEDS_INPUT,
                    collected_params=collected_params,
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 3 — Merge
        merged_params: Dict[str, Any] = {
            **collected_params,
            **extraction["extracted_params"],
        }

        # Step 4 — Completeness check
        required_param_names = {
            p["name"]
            for p in params_schema
            if isinstance(p, dict) and p.get("required", False)
        }
        all_collected = required_param_names.issubset(merged_params.keys())

        logger.debug(
            f"AgenticLoop: params merged | event_type=params_merged chat_id={chat_id} turn_count={turn_count} required_count={len(required_param_names)} collected_count={len(merged_params)} missing_count={len(required_param_names - merged_params.keys())}"
        )

        if all_collected:
            logger.debug(
                f"AgenticLoop: all required params collected on turn {turn_count} for chat_id={chat_id}"
            )
            await self._save_session(
                chat_id, merged_params, updated_turn_count, awaiting_continuation=False
            )
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.COMPLETED,
                    collected_params=merged_params,
                    clarifying_question="",
                    turn_count=updated_turn_count,
                ),
                [],
            )

        # Step 5 — Still missing params
        logger.debug(
            f"AgenticLoop: turn {turn_count} for chat_id={chat_id} — still missing: {extraction['missing_required']}"
        )

        if updated_turn_count == continuation_turn:
            logger.info(
                f"AgenticLoop: continuation threshold reached on turn {turn_count} for chat_id={chat_id}"
            )
            effective_continuation_lang = continuation_language or session_language
            continuation_q = _CONTINUATION_QUESTIONS.get(
                effective_continuation_lang, CONTINUATION_QUESTION
            )
            await self._save_session(
                chat_id, merged_params, updated_turn_count, awaiting_continuation=True
            )
            words = continuation_q.split(" ")
            continuation_tokens = [
                w + " " if i < len(words) - 1 else w for i, w in enumerate(words)
            ]
            return (
                AgenticLoopResult(
                    status=AgenticLoopStatus.AWAITING_CONTINUATION_DECISION,
                    collected_params=merged_params,
                    clarifying_question=continuation_q,
                    turn_count=updated_turn_count,
                ),
                continuation_tokens,
            )

        await self._save_session(
            chat_id, merged_params, updated_turn_count, awaiting_continuation=False
        )
        return (
            AgenticLoopResult(
                status=AgenticLoopStatus.NEEDS_INPUT,
                collected_params=merged_params,
                clarifying_question=extraction["clarifying_question"],
                turn_count=updated_turn_count,
            ),
            question_tokens,
        )

    async def _save_session(
        self,
        chat_id: str,
        collected_params: Dict[str, Any],
        turn_count: int,
        awaiting_continuation: bool = False,
    ) -> None:
        """Persist updated loop state to the Redis session store.

        Only updates the fields the loop owns (collected_params, turn_count,
        awaiting_continuation). Workflow-owned fields (selected_endpoint,
        state, max_turns) are preserved.
        A missing or unavailable session is logged but never raises.
        """
        try:
            if self._session_store is None:
                logger.debug(
                    f"AgenticLoop: session store unavailable — skipping save for chat_id={chat_id}"
                )
                return
            await self._session_store.update(
                chat_id,
                collected_params=collected_params,
                turn_count=turn_count,
                awaiting_continuation=awaiting_continuation,
            )
        except Exception as exc:
            logger.error(
                f"AgenticLoop: failed to save session for chat_id={chat_id}: {exc}"
            )

    def _detect_continuation_response(self, user_message: str) -> bool:
        """Detect whether the user's message indicates they want to continue.

        Delegates to :func:`~continuation_utils.detect_continuation_response`.

        Args:
            user_message: The raw user message to inspect.

        Returns:
            True if the user wants to continue, False otherwise.
        """
        return detect_continuation_response(user_message)

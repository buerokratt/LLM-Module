"""Unit tests for the AgenticLoop module."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tool_classifier.agentic_loop import AgenticLoop
from tool_classifier.enums import AgenticLoopStatus
from tool_classifier.param_extractor import ParamExtractionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHAT_ID = "test-chat-1"

_SCHEMA_TWO_REQUIRED: List[Dict[str, Any]] = [
    {
        "name": "countryIsoCode",
        "type": "string",
        "required": True,
        "description": "Country",
    },
    {
        "name": "validFrom",
        "type": "date",
        "required": True,
        "description": "Start date",
    },
]

_SCHEMA_ONE_OPTIONAL: List[Dict[str, Any]] = [
    {
        "name": "limit",
        "type": "integer",
        "required": False,
        "description": "Max results",
    },
]

_SCHEMA_EMPTY: List[Dict[str, Any]] = []

_HISTORY: List[Dict[str, Any]] = [
    {"authorRole": "user", "message": "Get me public holidays"},
    {"authorRole": "bot", "message": "Which country?"},
]


def _make_session_store_mock() -> AsyncMock:
    """Return an AsyncMock standing in for APIToolSessionStore."""
    mock = AsyncMock()
    mock.update = AsyncMock(return_value=None)
    return mock


def _make_extractor_mock(result: ParamExtractionResult) -> MagicMock:
    """Return a MagicMock whose __call__() returns the given ParamExtractionResult."""
    mock = MagicMock(return_value=result)
    return mock


def _make_loop(
    extractor_mock: MagicMock,
    session_store_mock: AsyncMock | None = None,
) -> AgenticLoop:
    """Convenience factory that wires up AgenticLoop with mocked dependencies."""
    return AgenticLoop(
        session_store=session_store_mock or _make_session_store_mock(),
        param_extractor=extractor_mock,
    )


def _extraction(
    extracted: Dict[str, Any],
    missing: List[str],
    question: str,
) -> ParamExtractionResult:
    return ParamExtractionResult(
        extracted_params=extracted,
        missing_required=missing,
        clarifying_question=question,
    )


# ---------------------------------------------------------------------------
# Turn limit guard
# ---------------------------------------------------------------------------


class TestMaxTurnsReached:
    @pytest.mark.asyncio
    async def test_max_turns_reached_when_turn_count_equals_max(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=_HISTORY,
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=5,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        extractor_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_turns_reached_when_turn_count_exceeds_max(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"validFrom": "2026-01-01"},
            turn_count=10,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert result.collected_params == {"validFrom": "2026-01-01"}
        extractor_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_count_incremented_on_max_turns(self) -> None:
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=5,
            max_turns=5,
        )

        assert result.turn_count == 6


# ---------------------------------------------------------------------------
# COMPLETED status
# ---------------------------------------------------------------------------


class TestCompleted:
    @pytest.mark.asyncio
    async def test_completed_when_extractor_finds_last_param(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"validFrom": "2026-01-01"}, [], "none")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="January 2026",
            conversation_history=_HISTORY,
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE"},
            turn_count=2,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert result.collected_params == {
            "countryIsoCode": "EE",
            "validFrom": "2026-01-01",
        }
        assert result.clarifying_question == ""

    @pytest.mark.asyncio
    async def test_completed_when_no_required_params_in_schema(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="list all",
            conversation_history=[],
            params_schema=_SCHEMA_ONE_OPTIONAL,
            collected_params={},
            turn_count=0,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_completed_with_empty_schema(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="go",
            conversation_history=[],
            params_schema=_SCHEMA_EMPTY,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.COMPLETED


# ---------------------------------------------------------------------------
# NEEDS_INPUT status
# ---------------------------------------------------------------------------


class TestNeedsInput:
    @pytest.mark.asyncio
    async def test_needs_input_when_params_still_missing(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction(
                {"countryIsoCode": "EE"},
                ["validFrom"],
                "From which date?",
            )
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=_HISTORY,
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert result.clarifying_question == "From which date?"

    @pytest.mark.asyncio
    async def test_needs_input_when_extractor_finds_nothing(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="I want holidays",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert result.clarifying_question == "Which country and date?"


# ---------------------------------------------------------------------------
# Param merging
# ---------------------------------------------------------------------------


class TestParamMerging:
    @pytest.mark.asyncio
    async def test_new_params_merged_with_prior_params(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"validFrom": "2026-01-01"}, [], "none")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE"},
            turn_count=1,
        )

        assert result.collected_params == {
            "countryIsoCode": "EE",
            "validFrom": "2026-01-01",
        }

    @pytest.mark.asyncio
    async def test_newly_extracted_param_overrides_prior_value(self) -> None:
        # Extractor returns a corrected countryIsoCode; new value should win
        extractor_mock = _make_extractor_mock(
            _extraction({"countryIsoCode": "LV"}, [], "none")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Latvia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE", "validFrom": "2026-01-01"},
            turn_count=2,
        )

        # Newly extracted "LV" overrides the prior "EE" so the user can correct mistakes
        assert result.collected_params["countryIsoCode"] == "LV"
        assert result.status == AgenticLoopStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_empty_extraction_preserves_prior_params(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["validFrom"], "From which date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE"},
            turn_count=0,
        )

        assert result.collected_params == {"countryIsoCode": "EE"}


# ---------------------------------------------------------------------------
# Turn count
# ---------------------------------------------------------------------------


class TestTurnCount:
    @pytest.mark.asyncio
    async def test_turn_count_incremented_on_needs_input(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
        )

        assert result.turn_count == 4

    @pytest.mark.asyncio
    async def test_turn_count_incremented_on_completed(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"countryIsoCode": "EE", "validFrom": "2026-01-01"}, [], "none")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia, January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.turn_count == 1


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_session_saved_on_needs_input(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"countryIsoCode": "EE"}, ["validFrom"], "From which date?")
        )
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        store_mock.update.assert_awaited_once_with(
            _CHAT_ID,
            collected_params={"countryIsoCode": "EE"},
            turn_count=1,
            awaiting_continuation=False,
        )

    @pytest.mark.asyncio
    async def test_session_saved_on_completed(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"validFrom": "2026-01-01"}, [], "none")
        )
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE"},
            turn_count=2,
        )

        store_mock.update.assert_awaited_once_with(
            _CHAT_ID,
            collected_params={"countryIsoCode": "EE", "validFrom": "2026-01-01"},
            turn_count=3,
            awaiting_continuation=False,
        )

    @pytest.mark.asyncio
    async def test_session_not_saved_on_max_turns(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=5,
            max_turns=5,
        )

        store_mock.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_not_saved_on_extractor_error(self) -> None:
        extractor_mock = MagicMock()
        extractor_mock.side_effect = RuntimeError("LLM timeout")
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=1,
        )

        store_mock.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_save_failure_does_not_raise(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        store_mock = _make_session_store_mock()
        store_mock.update.side_effect = RuntimeError("Redis unavailable")
        loop = _make_loop(extractor_mock, store_mock)

        # Should not raise even if Redis save fails
        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_extractor_exception_returns_needs_input(self) -> None:
        extractor_mock = MagicMock()
        extractor_mock.side_effect = RuntimeError("LLM timeout")
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"validFrom": "2026-01-01"},
            turn_count=1,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert result.clarifying_question == ""
        # Prior collected_params preserved on error
        assert result.collected_params == {"validFrom": "2026-01-01"}
        assert result.turn_count == 2

    @pytest.mark.asyncio
    async def test_extractor_called_with_correct_arguments(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({"countryIsoCode": "EE"}, ["validFrom"], "From which date?")
        )

        with patch("tool_classifier.agentic_loop.asyncio.to_thread") as mock_to_thread:
            # Make to_thread call the function synchronously so we can inspect args
            async def fake_to_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
                return fn(*args, **kwargs)

            mock_to_thread.side_effect = fake_to_thread

            loop = _make_loop(extractor_mock)
            await loop.run_turn(
                chat_id=_CHAT_ID,
                user_message="Estonia",
                conversation_history=_HISTORY,
                params_schema=_SCHEMA_TWO_REQUIRED,
                collected_params={"validFrom": "2026-01-01"},
                turn_count=1,
            )

        extractor_mock.assert_called_once_with(
            "Estonia",
            _SCHEMA_TWO_REQUIRED,
            _HISTORY,
            {"validFrom": "2026-01-01"},
            "en",
            1,
        )


# ---------------------------------------------------------------------------
# Continuation decision (turn-3 yes/no prompt)
# ---------------------------------------------------------------------------


class TestContinuationDecision:
    @pytest.mark.asyncio
    async def test_continuation_question_asked_at_threshold(self) -> None:
        """AWAITING_CONTINUATION_DECISION is returned on exactly continuation_turn=3."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="I don't know",
            conversation_history=_HISTORY,
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=2,  # updated_turn_count == 3 == continuation_turn
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.AWAITING_CONTINUATION_DECISION
        assert result.clarifying_question != ""
        assert result.turn_count == 3

    @pytest.mark.asyncio
    async def test_continuation_not_asked_before_threshold(self) -> None:
        """Normal NEEDS_INPUT before the continuation threshold."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hmm",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,  # updated_turn_count == 1, below threshold
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_continuation_not_asked_after_threshold(self) -> None:
        """Normal NEEDS_INPUT after the continuation threshold (user already continued)."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="still not sure",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,  # updated_turn_count == 4, past threshold
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_user_yes_resets_flag_and_continues(self) -> None:
        """When awaiting_continuation=True and user says 'yes', loop continues."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="yes",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            max_turns=5,
            awaiting_continuation=True,
        )

        # Should continue normally — NEEDS_INPUT (params still missing after "yes")
        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_user_estonian_yes_continues(self) -> None:
        """Estonian 'jah' is recognised as an affirmative response."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Mis riik?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="jah",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            max_turns=5,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_user_no_returns_max_turns_reached(self) -> None:
        """When awaiting_continuation=True and user says 'no', RAG fallback is triggered."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            max_turns=5,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert result.clarifying_question == ""

    @pytest.mark.asyncio
    async def test_user_estonian_no_exits(self) -> None:
        """Estonian 'ei' triggers RAG fallback."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Mis riik?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="ei",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_ambiguous_response_exits(self) -> None:
        """An ambiguous response while awaiting continuation defaults to exit."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="I'm not sure what to do",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_exit_preserves_collected_params(self) -> None:
        """Collected params are returned unchanged when the user chooses to exit."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["validFrom"], "Which date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE"},
            turn_count=3,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert result.collected_params == {"countryIsoCode": "EE"}

    @pytest.mark.asyncio
    async def test_session_saved_with_awaiting_continuation_true(self) -> None:
        """Session is persisted with awaiting_continuation=True at the threshold."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="no idea",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=2,
        )

        store_mock.update.assert_awaited_once_with(
            _CHAT_ID,
            collected_params={},
            turn_count=3,
            awaiting_continuation=True,
        )

    @pytest.mark.asyncio
    async def test_session_not_saved_on_user_exit(self) -> None:
        """Session is NOT saved when the user chooses to exit (caller deletes it)."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode"], "Which country?")
        )
        store_mock = _make_session_store_mock()
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=3,
            awaiting_continuation=True,
        )

        store_mock.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# stream_run_turn — streaming path
# ---------------------------------------------------------------------------


def _make_stream_extractor_mock(
    tokens: List[str],
    result: ParamExtractionResult,
) -> MagicMock:
    """Return a MagicMock whose stream_forward() coroutine returns (tokens, result)."""
    mock = MagicMock()
    mock.stream_forward = AsyncMock(return_value=(tokens, result))
    return mock


class TestStreamRunTurn:
    """stream_run_turn() must mirror run_turn() semantics while returning streamed tokens."""

    @pytest.mark.asyncio
    async def test_completed_returns_empty_tokens(self) -> None:
        """All params collected → COMPLETED result with no question tokens."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {"countryIsoCode": "EE", "validFrom": "2026-01-01"}, [], "none"
            ),
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia, January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert tokens == []
        assert result.collected_params == {
            "countryIsoCode": "EE",
            "validFrom": "2026-01-01",
        }

    @pytest.mark.asyncio
    async def test_needs_input_returns_question_tokens(self) -> None:
        """Missing params → NEEDS_INPUT result with streamed question tokens."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country", "?"],
            _extraction(
                {"validFrom": "2026-01-01"}, ["countryIsoCode"], "Which country?"
            ),
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == ["Which", " country", "?"]
        assert result.clarifying_question == "Which country?"

    @pytest.mark.asyncio
    async def test_re_extracted_param_overrides_prior_value(self) -> None:
        """Re-extracted value overwrites the previously collected one (correction allowed)."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction({"countryIsoCode": "LV"}, [], "none"),
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="Latvia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "EE", "validFrom": "2026-01-01"},
            turn_count=1,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert result.collected_params["countryIsoCode"] == "LV"
        assert tokens == []

    @pytest.mark.asyncio
    async def test_max_turns_reached_returns_empty_tokens(self) -> None:
        """Turn limit guard returns MAX_TURNS_REACHED with empty token list."""
        extractor_mock = _make_stream_extractor_mock(
            ["Some", " question"],
            _extraction({}, ["countryIsoCode"], "Which country?"),
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=5,
            max_turns=5,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert tokens == []

    @pytest.mark.asyncio
    async def test_stream_extraction_exception_returns_safe_defaults(self) -> None:
        """An exception from stream_forward must return NEEDS_INPUT with empty tokens."""
        extractor_mock = MagicMock()
        extractor_mock.stream_forward = AsyncMock(
            side_effect=RuntimeError("stream failure")
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == []
        assert result.collected_params == {}

    @pytest.mark.asyncio
    async def test_session_language_forwarded_to_stream_forward(self) -> None:
        """session_language must be passed through to stream_forward."""
        extractor_mock = _make_stream_extractor_mock(
            ["Millist", " riiki?"],
            _extraction({}, ["countryIsoCode"], "Millist riiki?"),
        )
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="Mis riik?",
            conversation_history=_HISTORY,
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
            session_language="et",
        )

        extractor_mock.stream_forward.assert_awaited_once()
        call_kwargs = extractor_mock.stream_forward.call_args.kwargs
        assert call_kwargs["session_language"] == "et"

    @pytest.mark.asyncio
    async def test_user_exit_during_stream_returns_empty_tokens(self) -> None:
        """When awaiting_continuation and user says 'no', return MAX_TURNS_REACHED + []."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country?"],
            _extraction({}, ["countryIsoCode"], "Which country?"),
        )
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"validFrom": "2026-01-01"},
            turn_count=3,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert tokens == []
        # Collected params returned unchanged on exit
        assert result.collected_params == {"validFrom": "2026-01-01"}


# ---------------------------------------------------------------------------
# seeded_params — L2 param_update pre-population at turn 0
# ---------------------------------------------------------------------------


class TestSeededParamsTurn0:
    """Verify that seeded_params from L2 follow-up routing are merged into
    collected_params at turn 0 only, with collected_params taking priority."""

    @pytest.mark.asyncio
    async def test_seeded_params_merged_at_turn_0(self) -> None:
        """seeded_params are prepended to collected_params when turn_count=0."""
        # Extractor returns only validFrom as newly extracted; countryIsoCode comes
        # from seeded_params.
        extractor_mock = _make_extractor_mock(
            _extraction(
                {"validFrom": "2026-01-01"},
                [],  # nothing missing — both params will be present after seed merge
                "none",
            )
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="January 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
            seeded_params={"countryIsoCode": "EE"},
        )

        # Both params present → COMPLETED
        assert result.status == AgenticLoopStatus.COMPLETED
        assert result.collected_params.get("countryIsoCode") == "EE"
        assert result.collected_params.get("validFrom") == "2026-01-01"

    @pytest.mark.asyncio
    async def test_collected_params_override_seeded_params(self) -> None:
        """collected_params values beat seeded_params when the key overlaps."""
        extractor_mock = _make_extractor_mock(
            _extraction(
                {"validFrom": "2026-06-01"},
                [],
                "none",
            )
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="June 2026",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={"countryIsoCode": "LV"},  # explicit value takes priority
            turn_count=0,
            seeded_params={"countryIsoCode": "EE"},  # seeded value must be overridden
        )

        assert result.collected_params.get("countryIsoCode") == "LV"

    @pytest.mark.asyncio
    async def test_seeded_params_not_applied_on_subsequent_turns(self) -> None:
        """seeded_params are ignored when turn_count > 0."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country and date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=1,  # NOT turn 0 → seeded_params must be ignored
            seeded_params={"countryIsoCode": "EE", "validFrom": "2026-01-01"},
        )

        # Even though seeded_params would satisfy all required params, they should
        # not be applied after turn 0 → still NEEDS_INPUT
        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        # seeded values not present in collected_params
        assert "countryIsoCode" not in result.collected_params
        assert "validFrom" not in result.collected_params

    @pytest.mark.asyncio
    async def test_seeded_params_none_does_not_raise(self) -> None:
        """Passing seeded_params=None (default) at turn 0 behaves normally."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["countryIsoCode", "validFrom"], "Which country?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
            seeded_params=None,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_seeded_params_partial_fill_still_asks_for_missing(self) -> None:
        """seeded_params satisfy only one of two required params → still NEEDS_INPUT."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["validFrom"], "From which date?")
        )
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            params_schema=_SCHEMA_TWO_REQUIRED,
            collected_params={},
            turn_count=0,
            seeded_params={"countryIsoCode": "EE"},  # only one param seeded
        )

        # validFrom still missing → NEEDS_INPUT
        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        # But seeded countryIsoCode should be present
        assert result.collected_params.get("countryIsoCode") == "EE"

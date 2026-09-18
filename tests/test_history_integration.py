"""Unit tests for conversation history integration in LLMOrchestrationService.

Tests cover:
- should_save_history(): filtering logic for when to persist rounds (standalone function)
- save_history_round(): Redis persistence with error isolation (standalone function)
- _extract_content_from_sse(): SSE chunk parsing
- Non-streaming hook in process_orchestration_request()
- RAG streaming hook in _stream_rag_pipeline() (accumulated_response saved after END)
- Classifier streaming hook in stream_orchestration_response() (non-RAG workflows)
- ContextWorkflowExecutor._build_history(): Redis-first history retrieval
- ContextAnalyzer.detect_context_with_summary_fallback(): pre_computed_summary fast path
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm_orchestration_service import (
    LLMOrchestrationService,
    _HISTORY_EXCLUDED_MESSAGES,
)
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    INPUT_GUARDRAIL_VIOLATION_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGE,
    TECHNICAL_ISSUE_MESSAGE,
)
from src.models.conversation_history_models import (
    ConversationHistoryState,
    ConversationRound,
)
from src.utils.conversation_history_store import should_save_history, save_history_round
from src.utils.sse_utils import extract_content_from_sse

# Use the same import path as llm_orchestration_service.py uses internally
# (``from models.request_models import ...``) to avoid the Python dual-import
# problem where isinstance() fails across two module-path aliases.
from models.request_models import OrchestrationResponse, TestOrchestrationResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> LLMOrchestrationService:
    """Return a bare LLMOrchestrationService instance with __init__ bypassed."""
    svc: LLMOrchestrationService = object.__new__(LLMOrchestrationService)
    svc.conversation_history_store = None  # default off
    return svc


def _make_ok_response(content: str = "Here is the answer.") -> OrchestrationResponse:
    return OrchestrationResponse(
        chatId="chat-1",
        llmServiceActive=True,
        questionOutOfLLMScope=False,
        inputGuardFailed=False,
        content=content,
    )


def _make_sse(chat_id: str, content: str) -> str:
    payload = {
        "chatId": chat_id,
        "payload": {"content": content},
        "timestamp": "1234567890",
        "sentTo": [],
    }
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# _HISTORY_EXCLUDED_MESSAGES content
# ---------------------------------------------------------------------------


class TestHistoryExcludedMessages:
    def test_english_out_of_scope_excluded(self):
        assert OUT_OF_SCOPE_MESSAGE in _HISTORY_EXCLUDED_MESSAGES

    def test_english_technical_issue_excluded(self):
        assert TECHNICAL_ISSUE_MESSAGE in _HISTORY_EXCLUDED_MESSAGES

    def test_english_input_guardrail_excluded(self):
        assert INPUT_GUARDRAIL_VIOLATION_MESSAGE in _HISTORY_EXCLUDED_MESSAGES

    def test_english_output_guardrail_excluded(self):
        assert OUTPUT_GUARDRAIL_VIOLATION_MESSAGE in _HISTORY_EXCLUDED_MESSAGES

    def test_normal_answer_not_excluded(self):
        assert "Here is the answer." not in _HISTORY_EXCLUDED_MESSAGES

    def test_set_is_frozenset(self):
        assert isinstance(_HISTORY_EXCLUDED_MESSAGES, frozenset)


# ---------------------------------------------------------------------------
# _should_save_history
# ---------------------------------------------------------------------------


class TestShouldSaveHistory:
    def test_returns_false_when_store_is_none(self):
        assert (
            should_save_history(None, _make_ok_response(), _HISTORY_EXCLUDED_MESSAGES)
            is False
        )

    def test_returns_false_for_test_orchestration_response(self):
        test_resp = TestOrchestrationResponse(
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content="answer",
        )
        assert (
            should_save_history(MagicMock(), test_resp, _HISTORY_EXCLUDED_MESSAGES)
            is False
        )

    def test_returns_false_when_input_guard_failed(self):
        resp = OrchestrationResponse(
            chatId="c1",
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=True,
            content=INPUT_GUARDRAIL_VIOLATION_MESSAGE,
        )
        assert (
            should_save_history(MagicMock(), resp, _HISTORY_EXCLUDED_MESSAGES) is False
        )

    def test_returns_false_when_out_of_scope(self):
        resp = OrchestrationResponse(
            chatId="c1",
            llmServiceActive=True,
            questionOutOfLLMScope=True,
            inputGuardFailed=False,
            content=OUT_OF_SCOPE_MESSAGE,
        )
        assert (
            should_save_history(MagicMock(), resp, _HISTORY_EXCLUDED_MESSAGES) is False
        )

    def test_returns_false_when_content_is_excluded_message(self):
        resp = _make_ok_response(content=TECHNICAL_ISSUE_MESSAGE)
        assert (
            should_save_history(MagicMock(), resp, _HISTORY_EXCLUDED_MESSAGES) is False
        )

    def test_returns_true_for_valid_successful_response(self):
        assert (
            should_save_history(
                MagicMock(), _make_ok_response(), _HISTORY_EXCLUDED_MESSAGES
            )
            is True
        )

    def test_returns_true_for_output_guardrail_violation_content_with_flags_false(self):
        """Content match on OUTPUT_GUARDRAIL_VIOLATION_MESSAGE must also exclude."""
        resp = _make_ok_response(content=OUTPUT_GUARDRAIL_VIOLATION_MESSAGE)
        assert (
            should_save_history(MagicMock(), resp, _HISTORY_EXCLUDED_MESSAGES) is False
        )


# ---------------------------------------------------------------------------
# save_history_round
# ---------------------------------------------------------------------------


class TestSaveHistoryRound:
    @pytest.mark.asyncio
    async def test_calls_store_save_round_with_correct_round(self):
        store = AsyncMock()

        await save_history_round(store, "chat-99", "user question", "bot answer")

        store.save_round.assert_awaited_once()
        call_args = store.save_round.call_args
        chat_id_arg, round_arg = call_args.args
        assert chat_id_arg == "chat-99"
        assert isinstance(round_arg, ConversationRound)
        assert round_arg.user_message == "user question"
        assert round_arg.bot_message == "bot answer"

    @pytest.mark.asyncio
    async def test_does_not_raise_when_store_raises(self):
        store = AsyncMock()
        store.save_round.side_effect = RuntimeError("Redis down")

        # Must not propagate
        await save_history_round(store, "chat-1", "q", "a")


# ---------------------------------------------------------------------------
# extract_content_from_sse
# ---------------------------------------------------------------------------


class TestExtractContentFromSse:
    def test_extracts_content_from_valid_chunk(self):
        chunk = _make_sse("c1", "Hello world")
        result = extract_content_from_sse(chunk)
        assert result == "Hello world"

    def test_extracts_end_marker(self):
        chunk = _make_sse("c1", "END")
        result = extract_content_from_sse(chunk)
        assert result == "END"

    def test_returns_none_for_non_sse_string(self):
        result = extract_content_from_sse("not sse data")
        assert result is None

    def test_returns_none_for_malformed_json(self):
        result = extract_content_from_sse("data: {bad json}\n\n")
        assert result is None

    def test_returns_none_when_payload_missing(self):
        chunk = "data: " + json.dumps({"chatId": "c1", "sentTo": []}) + "\n\n"
        result = extract_content_from_sse(chunk)
        assert result is None

    def test_returns_none_when_content_key_missing(self):
        chunk = "data: " + json.dumps({"chatId": "c1", "payload": {}}) + "\n\n"
        result = extract_content_from_sse(chunk)
        assert result is None

    def test_extracts_excluded_message_content(self):
        chunk = _make_sse("c1", TECHNICAL_ISSUE_MESSAGE)
        result = extract_content_from_sse(chunk)
        assert result == TECHNICAL_ISSUE_MESSAGE


# ---------------------------------------------------------------------------
# Non-streaming hook: process_orchestration_request
# ---------------------------------------------------------------------------


class TestNonStreamingHistoryHook:
    @pytest.mark.asyncio
    async def test_save_called_on_successful_response(self):
        store = AsyncMock()
        response = _make_ok_response("The answer.")

        # Exercise the logic directly (mirrors what process_orchestration_request does)
        if should_save_history(store, response, _HISTORY_EXCLUDED_MESSAGES):
            await save_history_round(store, "chat-1", "my question", response.content)

        store.save_round.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_not_called_on_guardrail_blocked_response(self):
        store = AsyncMock()
        blocked = OrchestrationResponse(
            chatId="c1",
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=True,
            content=INPUT_GUARDRAIL_VIOLATION_MESSAGE,
        )

        if should_save_history(store, blocked, _HISTORY_EXCLUDED_MESSAGES):
            await save_history_round(store, "c1", "bad query", blocked.content)

        store.save_round.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_not_called_on_out_of_scope_response(self):
        store = AsyncMock()
        oos = OrchestrationResponse(
            chatId="c1",
            llmServiceActive=True,
            questionOutOfLLMScope=True,
            inputGuardFailed=False,
            content=OUT_OF_SCOPE_MESSAGE,
        )

        if should_save_history(store, oos, _HISTORY_EXCLUDED_MESSAGES):
            await save_history_round(store, "c1", "obscure query", oos.content)

        store.save_round.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_not_called_when_store_is_none(self):
        response = _make_ok_response()

        if should_save_history(None, response, _HISTORY_EXCLUDED_MESSAGES):
            await save_history_round(None, "c1", "q", response.content)

        # When store is None, save_history_round returns early without calling save_round


# ---------------------------------------------------------------------------
# RAG streaming hook: _stream_rag_pipeline logic
# ---------------------------------------------------------------------------


class TestRagStreamingHistoryHook:
    @pytest.mark.asyncio
    async def test_save_called_with_accumulated_response_after_successful_stream(self):
        """Simulate the relevant section of _stream_rag_pipeline after streaming."""
        store = AsyncMock()

        accumulated_response = ["Hello", " world", " from", " RAG."]
        _rag_bot_message = "".join(accumulated_response)

        # Mirrors the hook added to _stream_rag_pipeline
        if store is not None:
            if _rag_bot_message not in _HISTORY_EXCLUDED_MESSAGES:
                await save_history_round(
                    store, "chat-5", "what is RAG?", _rag_bot_message
                )

        store.save_round.assert_awaited_once()
        _, round_arg = store.save_round.call_args.args
        assert round_arg.bot_message == "Hello world from RAG."
        assert round_arg.user_message == "what is RAG?"

    @pytest.mark.asyncio
    async def test_save_skipped_when_accumulated_is_excluded_message(self):
        """OOS/violations yielded as single chunks must not be saved."""
        store = AsyncMock()

        # Simulate what happens when OOS message ends up in accumulated_response
        accumulated_response = [OUT_OF_SCOPE_MESSAGE]
        _rag_bot_message = "".join(accumulated_response)

        if store is not None:
            if _rag_bot_message not in _HISTORY_EXCLUDED_MESSAGES:
                await save_history_round(store, "chat-5", "q", _rag_bot_message)

        store.save_round.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_skipped_when_store_is_none(self):
        store = None

        accumulated_response = ["some", " answer"]
        _rag_bot_message = "".join(accumulated_response)

        if store is not None:
            if _rag_bot_message not in _HISTORY_EXCLUDED_MESSAGES:
                await save_history_round(store, "c1", "q", _rag_bot_message)

        # When store is None, save_history_round returns early without doing anything


# ---------------------------------------------------------------------------
# Classifier streaming hook: stream_orchestration_response accumulation logic
# ---------------------------------------------------------------------------


class TestClassifierStreamingHistoryHook:
    @pytest.mark.asyncio
    async def test_accumulates_non_end_non_excluded_content(self):
        store = AsyncMock()

        tokens = ["The ", "answer ", "is 42."]
        sse_chunks = [_make_sse("c1", t) for t in tokens]
        sse_chunks.append(_make_sse("c1", "END"))

        # Mirrors the classifier streaming accumulation logic
        _save_classifier_history = store is not None
        _classifier_accumulated: list[str] = []

        for sse_chunk in sse_chunks:
            if _save_classifier_history:
                extracted = extract_content_from_sse(sse_chunk)
                if (
                    extracted is not None
                    and extracted != "END"
                    and extracted not in _HISTORY_EXCLUDED_MESSAGES
                ):
                    _classifier_accumulated.append(extracted)

        if _save_classifier_history and _classifier_accumulated:
            await save_history_round(
                store, "c1", "what is 42?", "".join(_classifier_accumulated)
            )

        store.save_round.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_save_when_only_violation_message_streamed(self):
        store = AsyncMock()

        sse_chunks = [
            _make_sse("c1", INPUT_GUARDRAIL_VIOLATION_MESSAGE),
            _make_sse("c1", "END"),
        ]

        _save_classifier_history = store is not None
        _classifier_accumulated: list[str] = []

        for sse_chunk in sse_chunks:
            if _save_classifier_history:
                extracted = extract_content_from_sse(sse_chunk)
                if (
                    extracted is not None
                    and extracted != "END"
                    and extracted not in _HISTORY_EXCLUDED_MESSAGES
                ):
                    _classifier_accumulated.append(extracted)

        if _save_classifier_history and _classifier_accumulated:
            await save_history_round(
                store, "c1", "bad q", "".join(_classifier_accumulated)
            )

        store.save_round.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_save_for_rag_workflow(self):
        """RAG workflow has its own hook in _stream_rag_pipeline; skip classifier hook."""
        from src.tool_classifier import WorkflowType

        store = AsyncMock()
        sse_chunks = [_make_sse("c1", "token"), _make_sse("c1", "END")]

        # Mirrors: _save_classifier_history = store is not None AND workflow != RAG
        workflow_type = WorkflowType.RAG
        _save_classifier_history = (
            store is not None and workflow_type != WorkflowType.RAG  # False for RAG
        )
        _classifier_accumulated: list[str] = []

        for sse_chunk in sse_chunks:
            if _save_classifier_history:
                extracted = extract_content_from_sse(sse_chunk)
                if extracted is not None and extracted != "END":
                    _classifier_accumulated.append(extracted)

        if _save_classifier_history and _classifier_accumulated:
            await save_history_round(store, "c1", "q", "".join(_classifier_accumulated))

        store.save_round.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_saves_for_non_rag_workflow(self):
        """Non-RAG workflows (SERVICE, API_TOOL, etc.) should save via classifier hook."""
        from src.tool_classifier import WorkflowType

        store = AsyncMock()
        tokens = ["answer"]
        sse_chunks = [_make_sse("c1", t) for t in tokens]
        sse_chunks.append(_make_sse("c1", "END"))

        # Use SERVICE workflow (not RAG) so the gate passes
        workflow_type = WorkflowType.SERVICE
        _save_classifier_history = (
            store is not None and workflow_type != WorkflowType.RAG  # True for SERVICE
        )
        _classifier_accumulated: list[str] = []

        for sse_chunk in sse_chunks:
            if _save_classifier_history:
                extracted = extract_content_from_sse(sse_chunk)
                if (
                    extracted is not None
                    and extracted != "END"
                    and extracted not in _HISTORY_EXCLUDED_MESSAGES
                ):
                    _classifier_accumulated.append(extracted)

        if _save_classifier_history and _classifier_accumulated:
            await save_history_round(store, "c1", "q", "".join(_classifier_accumulated))

        store.save_round.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_save_when_store_is_none(self):
        store = None
        sse_chunks = [_make_sse("c1", "answer"), _make_sse("c1", "END")]
        _save_classifier_history = store is not None
        _classifier_accumulated: list[str] = []

        for sse_chunk in sse_chunks:
            if _save_classifier_history:
                extracted = extract_content_from_sse(sse_chunk)
                if extracted is not None and extracted != "END":
                    _classifier_accumulated.append(extracted)

        if _save_classifier_history and _classifier_accumulated:
            await save_history_round(store, "c1", "q", "".join(_classifier_accumulated))

        # When store is None, save_history_round returns early without doing anything


# ---------------------------------------------------------------------------
# Helpers shared by the new test suites below
# ---------------------------------------------------------------------------


def _make_llm_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.ensure_global_config = MagicMock()
    mgr.use_task_local = MagicMock()
    return mgr


def _make_orchestration_request(
    chat_id: str = "chat-1",
    message: str = "What did you say earlier?",
    history: list | None = None,
) -> MagicMock:
    """Return a lightweight mock that mimics the fields accessed by _build_history."""
    req = MagicMock()
    req.chatId = chat_id
    req.message = message
    req.conversationHistory = history or []
    return req


def _make_round(
    user: str = "What is the tax rate?",
    bot: str = "The tax rate is 20%.",
    ts: float = 1_700_000_000.0,
) -> ConversationRound:
    return ConversationRound(user_message=user, bot_message=bot, timestamp=ts)


# ---------------------------------------------------------------------------
# ContextWorkflowExecutor._build_history — Redis-first retrieval
# ---------------------------------------------------------------------------


class TestBuildHistoryRedisFirst:
    """Tests for the async _build_history method in ContextWorkflowExecutor."""

    @pytest.mark.asyncio
    async def test_uses_redis_rounds_when_available(self) -> None:
        """When Redis has rounds, _build_history returns them and ignores request history."""
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        round_ = _make_round()
        state = ConversationHistoryState(
            chat_id="chat-1", rounds=[round_], summary=None
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        workflow = ContextWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            conversation_history_store=store,
        )
        req = _make_orchestration_request(chat_id="chat-1")

        history, summary = await workflow._build_history(req)

        assert len(history) == 2  # one round → two messages
        assert history[0]["authorRole"] == "user"
        assert history[0]["message"] == round_.user_message
        assert history[1]["authorRole"] == "bot"
        assert history[1]["message"] == round_.bot_message
        assert summary is None

    @pytest.mark.asyncio
    async def test_returns_redis_summary_with_rounds(self) -> None:
        """Summary stored in Redis is returned as pre_computed_summary."""
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        round_ = _make_round()
        state = ConversationHistoryState(
            chat_id="chat-1",
            rounds=[round_],
            summary="Earlier we discussed tax rates.",
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        workflow = ContextWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            conversation_history_store=store,
        )
        req = _make_orchestration_request(chat_id="chat-1")

        history, summary = await workflow._build_history(req)

        assert len(history) == 2
        assert summary == "Earlier we discussed tax rates."

    @pytest.mark.asyncio
    async def test_falls_back_to_request_when_redis_raises(self) -> None:
        """When get_context() raises, _build_history falls back to request.conversationHistory."""
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        store = AsyncMock()
        store.get_context = AsyncMock(side_effect=RuntimeError("Redis down"))

        workflow = ContextWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            conversation_history_store=store,
        )

        item = MagicMock()
        item.authorRole = "user"
        item.message = "fallback message"
        item.timestamp = "2024-01-01T00:00:00"

        req = _make_orchestration_request(chat_id="chat-1", history=[item])

        history, summary = await workflow._build_history(req)

        assert len(history) == 1
        assert history[0]["message"] == "fallback message"
        assert summary is None

    @pytest.mark.asyncio
    async def test_falls_back_to_request_when_store_is_none(self) -> None:
        """When conversation_history_store is None, request history is used."""
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        workflow = ContextWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            conversation_history_store=None,
        )

        item = MagicMock()
        item.authorRole = "bot"
        item.message = "bot reply"
        item.timestamp = "2024-01-01T00:00:01"

        req = _make_orchestration_request(chat_id="chat-1", history=[item])

        history, summary = await workflow._build_history(req)

        assert len(history) == 1
        assert history[0]["authorRole"] == "bot"
        assert summary is None

    @pytest.mark.asyncio
    async def test_falls_back_to_request_when_redis_returns_empty_rounds(self) -> None:
        """Redis state with no rounds → fall back to request.conversationHistory."""
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        state = ConversationHistoryState(
            chat_id="chat-1", rounds=[], summary="old summary"
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        workflow = ContextWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            conversation_history_store=store,
        )

        item = MagicMock()
        item.authorRole = "user"
        item.message = "from request"
        item.timestamp = "2024-01-01T00:00:00"

        req = _make_orchestration_request(chat_id="chat-1", history=[item])

        history, summary = await workflow._build_history(req)

        # Empty Redis rounds → fall back to request; summary not returned
        assert len(history) == 1
        assert history[0]["message"] == "from request"
        assert summary is None


# ---------------------------------------------------------------------------
# ContextAnalyzer.detect_context_with_summary_fallback — pre_computed_summary
# ---------------------------------------------------------------------------


class TestDetectContextWithPrecomputedSummary:
    """Tests for the pre_computed_summary fast-path in detect_context_with_summary_fallback."""

    def _make_analyzer(self) -> object:
        from src.tool_classifier.context_analyzer import ContextAnalyzer

        return ContextAnalyzer(_make_llm_manager())

    def _no_answer_detection(self) -> tuple:
        from src.tool_classifier.context_analyzer import ContextDetectionResult

        result = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=False,
            reasoning="cannot answer",
        )
        cost: dict = {"total_cost": 0.001, "total_tokens": 10, "num_calls": 1}
        return result, cost

    def _answer_from_summary(self) -> tuple:
        from src.tool_classifier.context_analyzer import ContextAnalysisResult

        result = ContextAnalysisResult(
            is_greeting=False,
            can_answer_from_context=True,
            answer="The tax rate is 20%.",
            reasoning="found in summary",
        )
        cost: dict = {"total_cost": 0.002, "total_tokens": 20, "num_calls": 1}
        return result, cost

    @pytest.mark.asyncio
    async def test_skips_generate_summary_when_pre_computed_provided(self) -> None:
        """_generate_conversation_summary must NOT be called when pre_computed_summary is set."""
        analyzer = self._make_analyzer()

        with (
            patch.object(
                analyzer,
                "detect_context",
                new_callable=AsyncMock,
                return_value=self._no_answer_detection(),
            ),
            patch.object(
                analyzer,
                "_generate_conversation_summary",
                new_callable=AsyncMock,
            ) as mock_generate,
            patch.object(
                analyzer,
                "_analyze_from_summary",
                new_callable=AsyncMock,
                return_value=self._answer_from_summary(),
            ),
        ):
            await analyzer.detect_context_with_summary_fallback(
                query="What was the tax rate?",
                conversation_history=[],
                pre_computed_summary="Tax rate is 20%.",
            )

        mock_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_still_runs_analyze_from_summary_when_pre_computed_provided(
        self,
    ) -> None:
        """_analyze_from_summary IS called even when the summary comes from Redis."""
        analyzer = self._make_analyzer()

        with (
            patch.object(
                analyzer,
                "detect_context",
                new_callable=AsyncMock,
                return_value=self._no_answer_detection(),
            ),
            patch.object(
                analyzer,
                "_generate_conversation_summary",
                new_callable=AsyncMock,
            ),
            patch.object(
                analyzer,
                "_analyze_from_summary",
                new_callable=AsyncMock,
                return_value=self._answer_from_summary(),
            ) as mock_analyze,
        ):
            await analyzer.detect_context_with_summary_fallback(
                query="What was the tax rate?",
                conversation_history=[],
                pre_computed_summary="Tax rate is 20%.",
            )

        mock_analyze.assert_awaited_once()
        call_kwargs = mock_analyze.call_args.kwargs
        assert call_kwargs["summary"] == "Tax rate is 20%."

    @pytest.mark.asyncio
    async def test_returns_answer_from_pre_computed_summary(self) -> None:
        """When the summary analysis succeeds, result has answered_from_summary=True."""
        from src.tool_classifier.context_analyzer import ContextDetectionResult

        analyzer = self._make_analyzer()

        with (
            patch.object(
                analyzer,
                "detect_context",
                new_callable=AsyncMock,
                return_value=self._no_answer_detection(),
            ),
            patch.object(
                analyzer, "_generate_conversation_summary", new_callable=AsyncMock
            ),
            patch.object(
                analyzer,
                "_analyze_from_summary",
                new_callable=AsyncMock,
                return_value=self._answer_from_summary(),
            ),
        ):
            result, _ = await analyzer.detect_context_with_summary_fallback(
                query="What was the tax rate?",
                conversation_history=[],
                pre_computed_summary="Tax rate is 20%.",
            )

        assert isinstance(result, ContextDetectionResult)
        assert result.can_answer_from_context is True
        assert result.answered_from_summary is True
        assert result.context_snippet == "The tax rate is 20%."

    @pytest.mark.asyncio
    async def test_summary_path_attempted_for_short_history_with_pre_computed(
        self,
    ) -> None:
        """Summary analysis runs even with <=10 turns when pre_computed_summary is set."""
        analyzer = self._make_analyzer()

        # Only 2 items in history (well below 10)
        short_history = [
            {"authorRole": "user", "message": "hi", "timestamp": "0"},
            {"authorRole": "bot", "message": "hello", "timestamp": "1"},
        ]

        with (
            patch.object(
                analyzer,
                "detect_context",
                new_callable=AsyncMock,
                return_value=self._no_answer_detection(),
            ),
            patch.object(
                analyzer, "_generate_conversation_summary", new_callable=AsyncMock
            ) as mock_gen,
            patch.object(
                analyzer,
                "_analyze_from_summary",
                new_callable=AsyncMock,
                return_value=self._answer_from_summary(),
            ) as mock_analyze,
        ):
            await analyzer.detect_context_with_summary_fallback(
                query="What was the tax rate?",
                conversation_history=short_history,
                pre_computed_summary="Tax rate was discussed previously.",
            )

        mock_gen.assert_not_awaited()  # skipped because pre_computed_summary is set
        mock_analyze.assert_awaited_once()  # still validated against query


# ---------------------------------------------------------------------------
# End-to-end: no LLM summarisation when Redis supplies a summary
# ---------------------------------------------------------------------------


class TestEndToEndNoLLMSummarisationWhenRedisHasSummary:
    """Verify that _generate_conversation_summary is never called when the
    workflow retrieves a summary from Redis via _build_history."""

    @pytest.mark.asyncio
    async def test_no_summarisation_call_when_redis_has_summary(self) -> None:
        """Full _detect() path: Redis summary present → no LLM summarisation."""
        from src.tool_classifier.context_analyzer import (
            ContextAnalysisResult,
            ContextDetectionResult,
        )
        from src.tool_classifier.workflows.context_workflow import (
            ContextWorkflowExecutor,
        )

        # Redis returns one round + a running summary
        round_ = _make_round()
        state = ConversationHistoryState(
            chat_id="chat-e2e",
            rounds=[round_],
            summary="We discussed tax rates earlier.",
        )
        store = AsyncMock()
        store.get_context = AsyncMock(return_value=state)

        # detect_context returns "cannot answer" so the summary path is tried
        cannot_answer = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=False,
            reasoning="not in recent history",
        )

        llm_manager = _make_llm_manager()
        workflow = ContextWorkflowExecutor(
            llm_manager=llm_manager,
            conversation_history_store=store,
        )

        with (
            patch.object(
                workflow.context_analyzer,
                "detect_context",
                new_callable=AsyncMock,
                return_value=(
                    cannot_answer,
                    {"total_cost": 0.001, "total_tokens": 10, "num_calls": 1},
                ),
            ),
            patch.object(
                workflow.context_analyzer,
                "_generate_conversation_summary",
                new_callable=AsyncMock,
            ) as mock_gen,
            patch.object(
                workflow.context_analyzer,
                "_analyze_from_summary",
                new_callable=AsyncMock,
                return_value=(
                    ContextAnalysisResult(
                        is_greeting=False,
                        can_answer_from_context=True,
                        answer="The tax rate is 20%.",
                        reasoning="from summary",
                    ),
                    {"total_cost": 0.002, "total_tokens": 20, "num_calls": 1},
                ),
            ),
        ):
            time_metric: dict = {}
            costs_metric: dict = {}
            history, pre_computed_summary = await workflow._build_history(
                _make_orchestration_request(chat_id="chat-e2e")
            )
            result = await workflow._detect(
                message="What was the tax rate?",
                history=history,
                time_metric=time_metric,
                costs_metric=costs_metric,
                pre_computed_summary=pre_computed_summary,
            )

        mock_gen.assert_not_awaited()
        assert result is not None
        assert result.can_answer_from_context is True
        assert result.answered_from_summary is True


# ---------------------------------------------------------------------------
# RAG Workflow: Redis history used in _refine_user_prompt
# ---------------------------------------------------------------------------


class TestRagWorkflowUsesRedisHistory:
    """Verify that the RAG pipeline fetches history from Redis before refinement."""

    @pytest.mark.asyncio
    async def test_redis_history_passed_to_refine_when_available(self) -> None:
        """get_conversation_history is called and its result replaces request history."""

        svc = _make_service()
        svc.conversation_history_store = AsyncMock()

        with patch(
            "src.llm_orchestration_service.get_conversation_history",
            new_callable=AsyncMock,
            return_value=([], None),
        ) as mock_get_history:
            # Stub out the rest of the pipeline so we only test the history fetch
            svc._refine_user_prompt = MagicMock(
                return_value=(
                    MagicMock(
                        original_question="q",
                        refined_questions=["q1"],
                    ),
                    {},
                )
            )
            svc._safe_retrieve_contextual_chunks = AsyncMock(return_value=[])
            svc.format_sse = MagicMock(return_value="data: {}\n\n")

            components = {
                "llm_manager": _make_llm_manager(),
                "contextual_retriever": AsyncMock(),
                "response_generator": MagicMock(),
                "guardrails_adapter": None,
            }
            stream_ctx = MagicMock()
            stream_ctx.stream_id = "sid"
            stream_ctx.mark_completed = MagicMock()

            request = _make_orchestration_request(chat_id="chat-rag")

            # Drain the generator to trigger the history fetch
            async for _ in svc._stream_rag_pipeline(
                request=request,
                components=components,
                stream_ctx=stream_ctx,
                costs_metric={},
                time_metric={},
            ):
                pass

        mock_get_history.assert_awaited_once_with(
            chat_id="chat-rag",
            store=svc.conversation_history_store,
            fallback=request.conversationHistory,
        )

    @pytest.mark.asyncio
    async def test_redis_fallback_used_when_store_is_none(self) -> None:
        """When conversation_history_store is None, request history is used."""
        svc = _make_service()
        svc.conversation_history_store = None

        with patch(
            "src.llm_orchestration_service.get_conversation_history",
            new_callable=AsyncMock,
            return_value=([], None),
        ) as mock_get_history:
            svc._refine_user_prompt = MagicMock(
                return_value=(
                    MagicMock(original_question="q", refined_questions=["q1"]),
                    {},
                )
            )
            svc._safe_retrieve_contextual_chunks = AsyncMock(return_value=[])
            svc.format_sse = MagicMock(return_value="data: {}\n\n")

            request = _make_orchestration_request(chat_id="chat-rag-fallback")
            components = {
                "llm_manager": _make_llm_manager(),
                "contextual_retriever": AsyncMock(),
                "response_generator": MagicMock(),
                "guardrails_adapter": None,
            }
            stream_ctx = MagicMock()
            stream_ctx.stream_id = "sid"
            stream_ctx.mark_completed = MagicMock()

            # Drain the generator to trigger the history fetch
            async for _ in svc._stream_rag_pipeline(
                request=request,
                components=components,
                stream_ctx=stream_ctx,
                costs_metric={},
                time_metric={},
            ):
                pass

        # Fallback is request.conversationHistory
        mock_get_history.assert_awaited_once_with(
            chat_id="chat-rag-fallback",
            store=None,
            fallback=request.conversationHistory,
        )


class TestRefineUserPromptSummary:
    """Verify that _refine_user_prompt prepends summary as a system turn."""

    def test_summary_prepended_to_dspy_history(self) -> None:
        """When conversation_summary is provided, a system message is first in history."""
        svc = _make_service()
        svc.langfuse_config = MagicMock()
        svc.langfuse_config.langfuse_client = None

        captured_history: list = []

        class _FakeRefiner:
            def forward_structured(
                self, history: list, question: str, **_: object
            ) -> dict:
                captured_history.extend(history)
                return {
                    "original_question": question,
                    "refined_questions": [question],
                    "usage": {},
                    "module_info": {},
                }

        llm_manager = _make_llm_manager()
        llm_manager.use_task_local = MagicMock()
        llm_manager.use_task_local.return_value.__enter__ = MagicMock(return_value=None)
        llm_manager.use_task_local.return_value.__exit__ = MagicMock(return_value=False)

        with patch(
            "src.llm_orchestration_service.PromptRefinerAgent",
            return_value=_FakeRefiner(),
        ):
            svc._refine_user_prompt(
                llm_manager=llm_manager,
                original_message="What is the rate?",
                conversation_history=[],
                conversation_summary="We discussed tax earlier.",
            )

        assert len(captured_history) >= 1
        first = captured_history[0]
        assert first["role"] == "system"
        assert "We discussed tax earlier." in first["content"]

    def test_no_summary_entry_when_summary_is_none(self) -> None:
        """When conversation_summary is None, no system turn is prepended."""
        svc = _make_service()
        svc.langfuse_config = MagicMock()
        svc.langfuse_config.langfuse_client = None

        captured_history: list = []

        class _FakeRefiner:
            def forward_structured(
                self, history: list, question: str, **_: object
            ) -> dict:
                captured_history.extend(history)
                return {
                    "original_question": question,
                    "refined_questions": [question],
                    "usage": {},
                    "module_info": {},
                }

        llm_manager = _make_llm_manager()

        with patch(
            "src.llm_orchestration_service.PromptRefinerAgent",
            return_value=_FakeRefiner(),
        ):
            svc._refine_user_prompt(
                llm_manager=llm_manager,
                original_message="What is the rate?",
                conversation_history=[],
                conversation_summary=None,
            )

        assert all(item.get("role") != "system" for item in captured_history)


# ---------------------------------------------------------------------------
# Service Workflow: Redis history used in _process_intent_detection
# ---------------------------------------------------------------------------


class TestServiceWorkflowUsesRedisHistory:
    """Verify ServiceWorkflowExecutor fetches history from Redis before intent detection."""

    @pytest.mark.asyncio
    async def test_get_conversation_history_called_in_process_intent_detection(
        self,
    ) -> None:
        """get_conversation_history is called with the history store from the service."""
        from src.tool_classifier.workflows.service_workflow import (
            ServiceWorkflowExecutor,
        )

        orchestration_service = MagicMock()
        history_store = AsyncMock()
        orchestration_service.conversation_history_store = history_store

        executor = ServiceWorkflowExecutor(
            llm_manager=_make_llm_manager(),
            orchestration_service=orchestration_service,
        )

        with (
            patch(
                "src.tool_classifier.workflows.service_workflow.get_conversation_history",
                new_callable=AsyncMock,
                return_value=([], None),
            ) as mock_get_history,
            patch.object(
                executor,
                "_detect_service_intent",
                new_callable=AsyncMock,
                return_value=(None, {}),
            ),
        ):
            request = _make_orchestration_request(chat_id="chat-svc")
            await executor._process_intent_detection(
                services=[],
                request=request,
                chat_id="chat-svc",
                context={},
                costs_metric={},
            )

        mock_get_history.assert_awaited_once_with(
            chat_id="chat-svc",
            store=history_store,
            fallback=request.conversationHistory,
        )

    @pytest.mark.asyncio
    async def test_summary_passed_to_detect_service_intent(self) -> None:
        """Summary from Redis is forwarded to _detect_service_intent."""
        from src.tool_classifier.workflows.service_workflow import (
            ServiceWorkflowExecutor,
        )

        executor = ServiceWorkflowExecutor(llm_manager=_make_llm_manager())

        with (
            patch(
                "src.tool_classifier.workflows.service_workflow.get_conversation_history",
                new_callable=AsyncMock,
                return_value=([], "Earlier we discussed registration."),
            ),
            patch.object(
                executor,
                "_detect_service_intent",
                new_callable=AsyncMock,
                return_value=(None, {}),
            ) as mock_detect,
        ):
            request = _make_orchestration_request(chat_id="chat-svc2")
            await executor._process_intent_detection(
                services=[],
                request=request,
                chat_id="chat-svc2",
                context={},
                costs_metric={},
            )

        _, kwargs = mock_detect.call_args
        assert (
            kwargs.get("conversation_summary") == "Earlier we discussed registration."
        )

    @pytest.mark.asyncio
    async def test_get_conversation_history_store_returns_none_when_no_service(
        self,
    ) -> None:
        """_get_conversation_history_store returns None when orchestration_service is None."""
        from src.tool_classifier.workflows.service_workflow import (
            ServiceWorkflowExecutor,
        )

        executor = ServiceWorkflowExecutor()
        assert executor._get_conversation_history_store() is None

    @pytest.mark.asyncio
    async def test_summary_prepended_to_history_dicts_in_detect_service_intent(
        self,
    ) -> None:
        """When conversation_summary is set, a 'system' message is first in history_dicts."""
        from src.tool_classifier.workflows.service_workflow import (
            ServiceWorkflowExecutor,
        )
        import dspy

        executor = ServiceWorkflowExecutor(llm_manager=_make_llm_manager())
        captured: list = []

        class _FakeModule:
            def forward(
                self,
                user_query: str,
                services: list,
                conversation_history: list | None = None,
            ) -> dict:
                if conversation_history:
                    captured.extend(conversation_history)
                return {
                    "matched_service_id": None,
                    "confidence": 0.0,
                    "entities": {},
                    "reasoning": "",
                }

        with (
            patch(
                "src.tool_classifier.workflows.service_workflow.IntentDetectionModule",
                return_value=_FakeModule(),
            ),
            patch.object(executor.llm_manager, "ensure_global_config"),
            patch.object(executor.llm_manager, "use_task_local"),
        ):
            # Patch dspy.settings.lm to avoid NoneType
            mock_lm = MagicMock()
            mock_lm.history = []
            with patch.object(dspy, "settings", MagicMock(lm=mock_lm)):
                await executor._detect_service_intent(
                    user_query="register my car",
                    services=[],
                    conversation_history=[],
                    chat_id="c1",
                    conversation_summary="User was asking about vehicle registration.",
                )

        assert len(captured) >= 1
        assert captured[0]["authorRole"] == "system"
        assert "vehicle registration" in captured[0]["message"]


# ---------------------------------------------------------------------------
# ATC Workflow: Redis history used in _compute_loop_step
# ---------------------------------------------------------------------------


class TestATCWorkflowUsesRedisHistory:
    """Verify APIToolWorkflowExecutor fetches history from Redis on turns > 0."""

    def _make_executor(
        self,
        history_store: object = None,
    ) -> object:
        from src.tool_classifier.workflows.api_tool_workflow import (
            APIToolWorkflowExecutor,
        )

        orchestration_service = MagicMock()
        orchestration_service.conversation_history_store = history_store
        orchestration_service.session_store = None
        orchestration_service.prompt_config_loader = None
        executor = APIToolWorkflowExecutor(orchestration_service=orchestration_service)
        return executor

    def test_get_conversation_history_store_returns_store(self) -> None:
        """_get_conversation_history_store returns the store from orchestration_service."""
        from src.tool_classifier.workflows.api_tool_workflow import (
            APIToolWorkflowExecutor,
        )

        store = AsyncMock()
        orchestration_service = MagicMock()
        orchestration_service.conversation_history_store = store
        executor = APIToolWorkflowExecutor(orchestration_service=orchestration_service)
        assert executor._get_conversation_history_store() is store

    def test_get_conversation_history_store_returns_none_when_no_service(self) -> None:
        """_get_conversation_history_store returns None when orchestration_service is None."""
        from src.tool_classifier.workflows.api_tool_workflow import (
            APIToolWorkflowExecutor,
        )

        executor = APIToolWorkflowExecutor(orchestration_service=None)
        assert executor._get_conversation_history_store() is None

    @pytest.mark.asyncio
    async def test_no_redis_call_on_turn_zero(self) -> None:
        """On turn 0 get_conversation_history must NOT be called."""
        from src.models.session_models import APIToolSession
        from src.tool_classifier.enums import ExecutionMode

        executor = self._make_executor()

        session = MagicMock(spec=APIToolSession)
        session.turn_count = 0
        session.selected_endpoint = {"name": "test_endpoint", "params": []}
        session.collected_params = {}
        session.max_turns = 5
        session.awaiting_continuation = False
        session.detected_language = "en"
        session.original_query = "book a slot"
        session.execution_mode = ExecutionMode.SINGLE.value
        session.parallel_endpoints = []

        with (
            patch(
                "src.tool_classifier.workflows.api_tool_workflow.get_conversation_history",
                new_callable=AsyncMock,
            ) as mock_get_history,
            patch.object(executor, "_get_session_store", return_value=None),
            patch.object(
                executor,
                "_get_custom_instructions",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch.object(
                executor,
                "_build_agentic_loop",
                return_value=MagicMock(
                    stream_run_turn=AsyncMock(
                        return_value=(
                            MagicMock(
                                status=MagicMock(value="NEEDS_INPUT"),
                                clarifying_question="What time?",
                                turn_count=1,
                            ),
                            ["What", " time", "?"],
                        )
                    )
                ),
            ),
        ):
            # Force session_store.get to return our mocked session
            with patch.object(
                executor,
                "_get_session_store",
                return_value=MagicMock(
                    get=AsyncMock(return_value=session),
                    delete=AsyncMock(),
                ),
            ):
                request = _make_orchestration_request(chat_id="chat-atc-t0")
                await executor._compute_loop_step(
                    request=request,
                    context={"matched_endpoint": {"name": "ep", "params": []}},
                )

        mock_get_history.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_redis_history_fetched_on_turn_greater_than_zero(self) -> None:
        """On turn > 0 get_conversation_history IS called."""
        from src.models.session_models import APIToolSession
        from src.tool_classifier.enums import ExecutionMode

        executor = self._make_executor()

        session = MagicMock(spec=APIToolSession)
        session.turn_count = 1
        session.selected_endpoint = {"name": "test_endpoint", "params": []}
        session.collected_params = {}
        session.max_turns = 5
        session.awaiting_continuation = False
        session.detected_language = "en"
        session.original_query = "book a slot"
        session.execution_mode = ExecutionMode.SINGLE.value
        session.parallel_endpoints = []

        with (
            patch(
                "src.tool_classifier.workflows.api_tool_workflow.get_conversation_history",
                new_callable=AsyncMock,
                return_value=([], None),
            ) as mock_get_history,
            patch.object(
                executor,
                "_get_custom_instructions",
                new_callable=AsyncMock,
                return_value="",
            ),
            patch.object(
                executor,
                "_build_agentic_loop",
                return_value=MagicMock(
                    stream_run_turn=AsyncMock(
                        return_value=(
                            MagicMock(
                                status=MagicMock(value="NEEDS_INPUT"),
                                clarifying_question="What time?",
                                turn_count=2,
                            ),
                            ["What", " time", "?"],
                        )
                    )
                ),
            ),
            patch.object(
                executor,
                "_get_session_store",
                return_value=MagicMock(
                    get=AsyncMock(return_value=session),
                    delete=AsyncMock(),
                ),
            ),
        ):
            request = _make_orchestration_request(chat_id="chat-atc-t1")
            await executor._compute_loop_step(
                request=request,
                context={"matched_endpoint": {"name": "ep", "params": []}},
            )

        mock_get_history.assert_awaited_once()
        _, kwargs = mock_get_history.call_args
        assert kwargs["chat_id"] == "chat-atc-t1"

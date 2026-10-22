"""Unit tests for context workflow executor."""

import pytest
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
import dspy

from src.tool_classifier.workflows.context_workflow import ContextWorkflowExecutor
from src.tool_classifier.context_analyzer import ContextDetectionResult
from src.models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    ConversationItem,
)


@pytest.fixture
def mock_dspy_lm() -> Generator[MagicMock, None, None]:
    """Mock DSPy LM to prevent 'No LM is loaded' errors."""
    mock_lm = MagicMock()
    mock_lm.history = []
    with patch("dspy.settings") as mock_settings:
        mock_settings.lm = mock_lm
        # Configure DSPy with mock LM
        dspy.configure(lm=mock_lm)
        yield mock_lm


@pytest.fixture
def mock_orchestration_service() -> MagicMock:
    """Create mock orchestration service for streaming tests."""
    import json as _json
    import time as _time

    service = MagicMock()

    def _format_sse_impl(chat_id: str, content: str) -> str:
        payload = {
            "chatId": chat_id,
            "payload": {"content": content},
            "timestamp": int(_time.time() * 1000),
        }
        return f"data: {_json.dumps(payload)}\n\n"

    service.format_sse = _format_sse_impl
    service.log_costs = MagicMock()
    service.store_streaming_inference = AsyncMock()
    return service


@pytest.fixture
def llm_manager() -> MagicMock:
    """Create mock LLM manager."""
    return MagicMock()


@pytest.fixture
def context_workflow(
    llm_manager: MagicMock,
    mock_orchestration_service: MagicMock,
    mock_dspy_lm: MagicMock,
) -> ContextWorkflowExecutor:
    """Create ContextWorkflowExecutor instance."""
    return ContextWorkflowExecutor(
        llm_manager, orchestration_service=mock_orchestration_service
    )


@pytest.fixture
def sample_request() -> OrchestrationRequest:
    """Create sample orchestration request."""
    return OrchestrationRequest(
        chatId="test-chat-123",
        message="Hello!",
        authorId="test-user",
        conversationHistory=[],
        url="https://example.com",
        environment="testing",
        connection_id="test-connection",
    )


class TestContextWorkflowInit:
    """Test context workflow initialization."""

    def test_init_creates_workflow(self, llm_manager: MagicMock) -> None:
        """ContextWorkflowExecutor should initialize with LLM manager."""
        workflow = ContextWorkflowExecutor(llm_manager)

        assert workflow.llm_manager is llm_manager
        assert workflow.context_analyzer is not None


class TestExecuteAsyncGreeting:
    """Test execute_async with greeting queries."""

    @pytest.mark.asyncio
    async def test_execute_async_greeting_estonian(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should handle Estonian greeting and return response."""
        sample_request.message = "Tere!"

        # Mock context analyzer
        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            greeting_type="hello",
            can_answer_from_context=False,
            reasoning="Greeting detected",
            context_snippet=None,
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context_with_summary_fallback",
            new_callable=AsyncMock,
        ) as mock_detect:
            mock_detect.return_value = (
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            )
            context_dict = {}
            response = await context_workflow.execute_async(
                sample_request, context_dict
            )

        assert response is not None
        assert isinstance(response, OrchestrationResponse)
        assert response.chatId == "test-chat-123"
        assert "Tere" in response.content
        assert response.llmServiceActive is True
        assert response.questionOutOfLLMScope is False
        assert response.inputGuardFailed is False

        # Check cost tracking
        assert "costs_dict" in context_dict
        assert "context_detection" in context_dict["costs_dict"]

    @pytest.mark.asyncio
    async def test_execute_async_greeting_english(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should handle English greeting and return response."""
        sample_request.message = "Hello!"

        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            greeting_type="hello",
            can_answer_from_context=False,
            reasoning="English greeting detected",
            context_snippet=None,
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context_with_summary_fallback",
            new_callable=AsyncMock,
        ) as mock_detect:
            mock_detect.return_value = (
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            )
            response = await context_workflow.execute_async(sample_request, {})

        assert response is not None
        assert "Hello" in response.content or "hello" in response.content.lower()


class TestExecuteAsyncContextBased:
    """Test execute_async with context-based queries."""

    @pytest.mark.asyncio
    async def test_execute_async_context_answer(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should answer from conversation history when possible."""
        # Add conversation history
        sample_request.conversationHistory = [
            ConversationItem(
                authorRole="user",
                message="What is the tax rate?",
                timestamp="2024-01-01T12:00:00",
            ),
            ConversationItem(
                authorRole="bot",
                message="The tax rate is 20%.",
                timestamp="2024-01-01T12:00:01",
            ),
        ]
        sample_request.message = "What was the rate you mentioned?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=True,
            reasoning="Referring to previous conversation about tax rate",
            context_snippet="The tax rate is 20%.",
        )

        with (
            patch.object(
                context_workflow.context_analyzer,
                "detect_context",
                return_value=(
                    mock_analysis,
                    {"total_cost": 0.002, "total_tokens": 100, "num_calls": 1},
                ),
            ),
            patch.object(
                context_workflow.context_analyzer,
                "generate_context_response",
                new_callable=AsyncMock,
                return_value=(
                    "The tax rate is 20%.",
                    {"total_cost": 0.003, "num_calls": 1},
                ),
            ),
        ):
            response = await context_workflow.execute_async(sample_request, {})

        assert response is not None
        assert "20%" in response.content

    @pytest.mark.asyncio
    async def test_execute_async_cannot_answer_from_context(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return None when cannot answer from context (fallback to RAG)."""
        sample_request.message = "What is digital signature?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=False,
            reasoning="Query requires knowledge base search",
            context_snippet=None,
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            return_value=(
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            ),
        ):
            response = await context_workflow.execute_async(sample_request, {})

        assert response is None

    @pytest.mark.asyncio
    async def test_execute_async_answer_is_none(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return None when can_answer_from_context=True but context_snippet is absent."""
        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=True,
            context_snippet=None,  # No snippet → cannot generate answer
            reasoning="No relevant snippet found in history",
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            return_value=(
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            ),
        ):
            response = await context_workflow.execute_async(sample_request, {})

        assert response is None


class TestExecuteAsyncErrorHandling:
    """Test error handling in execute_async."""

    @pytest.mark.asyncio
    async def test_execute_async_handles_analyzer_exception(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return None when context analyzer raises exception."""
        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            side_effect=Exception("Analysis failed"),
        ):
            response = await context_workflow.execute_async(sample_request, {})

        assert response is None


class TestExecuteStreaming:
    """Test execute_streaming functionality."""

    @pytest.mark.asyncio
    async def test_execute_streaming_greeting(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should stream greeting response."""
        sample_request.message = "Hello!"

        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            greeting_type="hello",
            can_answer_from_context=False,
            reasoning="Greeting detected",
            context_snippet=None,
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            return_value=(
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            ),
        ):
            stream_gen = await context_workflow.execute_streaming(sample_request, {})

        assert stream_gen is not None

        # Collect streamed chunks
        chunks = [chunk async for chunk in stream_gen]

        # Should have multiple chunks + END marker
        assert len(chunks) > 1

        # Last chunk should be END marker
        last_chunk = chunks[-1]
        assert "END" in last_chunk

        # All chunks should be valid SSE format
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_execute_streaming_context_answer(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should stream context-based answer."""
        sample_request.message = "What did you say earlier?"
        sample_request.conversationHistory = [
            ConversationItem(
                authorRole="bot",
                message="The rate is 20%.",
                timestamp="2024-01-01T12:00:00",
            ),
        ]

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=True,
            reasoning="Referring to previous message",
            context_snippet="I mentioned that the rate is 20%.",
        )

        async def _fake_history_stream(
            *args: object, **kwargs: object
        ) -> AsyncGenerator[str, None]:
            yield context_workflow.orchestration_service.format_sse(
                sample_request.chatId, "I mentioned that the rate is 20%."
            )
            yield context_workflow.orchestration_service.format_sse(
                sample_request.chatId, "END"
            )

        with (
            patch.object(
                context_workflow.context_analyzer,
                "detect_context",
                return_value=(
                    mock_analysis,
                    {"total_cost": 0.002, "total_tokens": 100, "num_calls": 1},
                ),
            ),
            patch.object(
                context_workflow,
                "_create_history_stream",
                new_callable=AsyncMock,
                return_value=_fake_history_stream(),
            ),
        ):
            stream_gen = await context_workflow.execute_streaming(sample_request, {})

        assert stream_gen is not None

        chunks = [chunk async for chunk in stream_gen]

        assert len(chunks) > 0
        # Verify END marker
        assert "END" in chunks[-1]

    @pytest.mark.asyncio
    async def test_execute_streaming_cannot_answer(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return None when cannot answer (fallback to RAG)."""
        sample_request.message = "What is digital signature?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=False,
            reasoning="Requires knowledge base",
        )

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            return_value=(
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            ),
        ):
            stream_gen = await context_workflow.execute_streaming(sample_request, {})

        assert stream_gen is None

    @pytest.mark.asyncio
    async def test_execute_streaming_handles_exception(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return None when analyzer raises exception."""
        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            side_effect=Exception("Analysis failed"),
        ):
            stream_gen = await context_workflow.execute_streaming(sample_request, {})

        assert stream_gen is None


class TestCostTracking:
    """Test cost tracking functionality."""

    @pytest.mark.asyncio
    async def test_cost_tracking_in_context_dict(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should track costs in context dictionary."""
        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            can_answer_from_context=False,
            reasoning="Greeting",
        )

        cost_dict = {
            "total_cost": 0.0015,
            "total_tokens": 75,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 25,
            "num_calls": 1,
        }

        with patch.object(
            context_workflow.context_analyzer,
            "detect_context",
            return_value=(mock_analysis, cost_dict),
        ):
            context_dict = {}
            await context_workflow.execute_async(sample_request, context_dict)

        assert "costs_dict" in context_dict
        assert "context_detection" in context_dict["costs_dict"]
        assert context_dict["costs_dict"]["context_detection"]["total_cost"] == 0.0015
        assert context_dict["costs_dict"]["context_detection"]["total_tokens"] == 75


class TestLanguageDetection:
    """Test language detection integration."""

    @pytest.mark.asyncio
    async def test_detects_estonian_language(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should detect Estonian language from query."""
        sample_request.message = "Tere! Kuidas läheb?"

        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            can_answer_from_context=False,
            reasoning="Estonian greeting",
        )

        with (
            patch.object(
                context_workflow.context_analyzer, "detect_context"
            ) as mock_detect,
            patch(
                "src.tool_classifier.greeting_constants.get_greeting_response"
            ) as mock_greeting,
        ):
            mock_detect.return_value = (
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            )
            mock_greeting.return_value = "Tere! Kuidas ma saan sind aidata?"

            await context_workflow.execute_async(sample_request, {})

            # Verify Estonian language was used for greeting response
            mock_greeting.assert_called_with(greeting_type="hello", language="et")

    @pytest.mark.asyncio
    async def test_detects_english_language(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should detect English language from query."""
        sample_request.message = "Hello! How are you?"

        mock_analysis = ContextDetectionResult(
            is_greeting=True,
            can_answer_from_context=False,
            reasoning="English greeting",
        )

        with (
            patch.object(
                context_workflow.context_analyzer, "detect_context"
            ) as mock_detect,
            patch(
                "src.tool_classifier.greeting_constants.get_greeting_response"
            ) as mock_greeting,
        ):
            mock_detect.return_value = (
                mock_analysis,
                {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
            )
            mock_greeting.return_value = "Hello! How can I help you?"

            await context_workflow.execute_async(sample_request, {})

            # Verify English language was used for greeting response
            mock_greeting.assert_called_with(greeting_type="hello", language="en")


class TestExecuteAsyncSummaryBased:
    """Test execute_async with summary-based answers."""

    @pytest.mark.asyncio
    async def test_execute_async_summary_answer(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should return response when answer comes from conversation summary."""
        sample_request.message = "What was the tax rate we discussed earlier?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=True,
            reasoning="Found in conversation summary",
            context_snippet="Based on our earlier discussion, the tax rate is 20%.",
            answered_from_summary=True,
        )

        with (
            patch.object(
                context_workflow.context_analyzer,
                "detect_context",
                return_value=(
                    mock_analysis,
                    {"total_cost": 0.005, "total_tokens": 200, "num_calls": 3},
                ),
            ),
            patch.object(
                context_workflow.context_analyzer,
                "generate_context_response",
                new_callable=AsyncMock,
                return_value=(
                    "Based on our earlier discussion, the tax rate is 20%.",
                    {"total_cost": 0.003, "num_calls": 1},
                ),
            ),
        ):
            response = await context_workflow.execute_async(sample_request, {})

        assert response is not None
        assert isinstance(response, OrchestrationResponse)
        assert "20%" in response.content
        assert response.llmServiceActive is True

    @pytest.mark.asyncio
    async def test_execute_streaming_summary_answer(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should stream summary-based answer correctly."""
        sample_request.message = "What was the tax rate we discussed earlier?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=True,
            reasoning="Found in conversation summary",
            context_snippet="Based on our earlier discussion, the tax rate is 20%.",
            answered_from_summary=True,
        )

        async def _fake_summary_stream(
            *args: object, **kwargs: object
        ) -> AsyncGenerator[str, None]:
            yield context_workflow.orchestration_service.format_sse(
                sample_request.chatId, "The tax rate is 20%."
            )
            yield context_workflow.orchestration_service.format_sse(
                sample_request.chatId, "END"
            )

        with (
            patch.object(
                context_workflow.context_analyzer,
                "detect_context",
                return_value=(
                    mock_analysis,
                    {"total_cost": 0.005, "total_tokens": 200, "num_calls": 3},
                ),
            ),
            patch.object(
                context_workflow,
                "_create_history_stream",
                new_callable=AsyncMock,
                return_value=_fake_summary_stream(),
            ),
        ):
            stream_gen = await context_workflow.execute_streaming(sample_request, {})

        assert stream_gen is not None

        chunks = [chunk async for chunk in stream_gen]

        # Should have multiple chunks + END marker
        assert len(chunks) > 1
        assert "END" in chunks[-1]

    @pytest.mark.asyncio
    async def test_pre_computed_summary_analysis(
        self,
        context_workflow: ContextWorkflowExecutor,
        sample_request: OrchestrationRequest,
    ) -> None:
        """Should use pre-computed summary analysis from classifier."""
        sample_request.message = "What was the tax rate?"

        mock_analysis = ContextDetectionResult(
            is_greeting=False,
            greeting_type="hello",
            can_answer_from_context=True,
            reasoning="Found in summary",
            context_snippet="The tax rate is 20%.",
            answered_from_summary=True,
        )

        # Pre-computed analysis (from classifier)
        context = {"analysis_result": mock_analysis}

        with patch.object(
            context_workflow.context_analyzer,
            "generate_context_response",
            new_callable=AsyncMock,
            return_value=(
                "The tax rate is 20%.",
                {"total_cost": 0.003, "num_calls": 1},
            ),
        ):
            response = await context_workflow.execute_async(sample_request, context)

        assert response is not None
        assert "20%" in response.content

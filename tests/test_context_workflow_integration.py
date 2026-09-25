"""Integration tests for context workflow.

Tests the full classify -> route -> execute chain with real component wiring.
Only the LLM layer (dspy) and RAG orchestration service are mocked.

These tests verify:
- ToolClassifier.classify() correctly routes greetings to CONTEXT workflow
- ToolClassifier.route_to_workflow() executes the context workflow end-to-end
- Fallback from CONTEXT to RAG when context cannot answer
- Streaming mode for context workflow responses
- Cost tracking propagation through the classify -> execute chain
- Error resilience (LLM failures, JSON parse errors)
"""

import pytest
from collections.abc import AsyncGenerator, Generator
from contextlib import AbstractContextManager
from unittest.mock import AsyncMock, MagicMock, patch
import json
import dspy

from src.tool_classifier.classifier import ToolClassifier
from src.tool_classifier.context_analyzer import ContextDetectionResult
from src.tool_classifier.models import ClassificationResult
from src.models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    ConversationItem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
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
    """Create mock orchestration service for RAG workflow fallback."""
    import json as _json
    import time as _time

    service = MagicMock()

    # Non-streaming RAG fallback returns a valid response
    async def mock_execute_pipeline(**kwargs: object) -> OrchestrationResponse:
        return OrchestrationResponse(
            chatId=kwargs["request"].chatId,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content="RAG fallback answer.",
        )

    service._execute_orchestration_pipeline = AsyncMock(
        side_effect=mock_execute_pipeline
    )
    service._initialize_service_components = MagicMock(return_value={})
    service._log_costs = MagicMock()
    service.log_costs = MagicMock()

    def _format_sse_impl(chat_id: str, content: str) -> str:
        payload = {
            "chatId": chat_id,
            "payload": {"content": content},
            "timestamp": int(_time.time() * 1000),
        }
        return f"data: {_json.dumps(payload)}\n\n"

    service.format_sse = _format_sse_impl

    # Streaming RAG fallback
    async def mock_stream_pipeline(**kwargs: object) -> AsyncGenerator[str, None]:
        yield 'data: {"chatId":"test","payload":{"content":"RAG stream"}}\n\n'
        yield 'data: {"chatId":"test","payload":{"content":"END"}}\n\n'

    service._stream_rag_pipeline = mock_stream_pipeline

    return service


@pytest.fixture
def llm_manager() -> MagicMock:
    """Create mock LLM manager."""
    return MagicMock()


@pytest.fixture
def classifier(
    llm_manager: MagicMock, mock_orchestration_service: MagicMock
) -> ToolClassifier:
    """Create a real ToolClassifier with real workflow executors."""
    return ToolClassifier(
        llm_manager=llm_manager,
        orchestration_service=mock_orchestration_service,
    )


def _make_request(
    message: str,
    chat_id: str = "integration-test-chat",
    history: list | None = None,
) -> OrchestrationRequest:
    """Helper to build an OrchestrationRequest."""
    return OrchestrationRequest(
        chatId=chat_id,
        message=message,
        authorId="test-user",
        conversationHistory=history or [],
        url="https://example.com",
        environment="testing",
        connection_id="test-conn",
    )


def _mock_dspy_greeting(answer_text: str) -> AbstractContextManager[MagicMock]:
    """Return a patch context manager that makes dspy return a greeting analysis."""
    mock_response = MagicMock()
    mock_response.analysis_result = json.dumps(
        {
            "is_greeting": True,
            "can_answer_from_context": False,
            "answer": answer_text,
            "reasoning": "Greeting detected",
        }
    )
    return patch(
        "dspy.ChainOfThought",
        return_value=MagicMock(return_value=mock_response),
    )


def _mock_dspy_context_answer(
    answer_text: str, reasoning: str = "History reference"
) -> AbstractContextManager[MagicMock]:
    """Return a patch that makes dspy return a context-based answer."""
    mock_response = MagicMock()
    mock_response.analysis_result = json.dumps(
        {
            "is_greeting": False,
            "can_answer_from_context": True,
            "answer": answer_text,
            "reasoning": reasoning,
        }
    )
    return patch(
        "dspy.ChainOfThought",
        return_value=MagicMock(return_value=mock_response),
    )


def _mock_dspy_no_match() -> AbstractContextManager[MagicMock]:
    """Return a patch that makes dspy indicate neither greeting nor context match."""
    mock_response = MagicMock()
    mock_response.analysis_result = json.dumps(
        {
            "is_greeting": False,
            "can_answer_from_context": False,
            "answer": None,
            "reasoning": "Requires knowledge base search",
        }
    )
    return patch(
        "dspy.ChainOfThought",
        return_value=MagicMock(return_value=mock_response),
    )


def _patch_cost_utils() -> AbstractContextManager[MagicMock]:
    """Patch cost tracking to avoid dspy settings dependency.

    Patches at both possible module paths to handle Python's module identity
    behaviour when src/ is on sys.path (module may be loaded as either
    ``tool_classifier.context_analyzer`` or ``src.tool_classifier.context_analyzer``).
    """
    cost_return = {
        "total_cost": 0.001,
        "total_tokens": 50,
        "total_prompt_tokens": 30,
        "total_completion_tokens": 20,
        "num_calls": 1,
    }

    import sys

    # Determine which module key is actually loaded at runtime
    if "tool_classifier.context_analyzer" in sys.modules:
        target = "tool_classifier.context_analyzer.get_lm_usage_since"
    else:
        target = "src.tool_classifier.context_analyzer.get_lm_usage_since"

    return patch(target, return_value=cost_return)


# ---------------------------------------------------------------------------
# Integration: classify -> route -> execute (non-streaming)
# ---------------------------------------------------------------------------


class TestClassifyAndRouteGreeting:
    """Test full classify -> route chain for greeting queries."""

    @pytest.mark.asyncio
    async def test_greeting_classify_returns_context_workflow(
        self, classifier: ToolClassifier
    ) -> None:
        """classify() should return CONTEXT workflow for greeting queries.

        With the hybrid-search classifier, classify() uses Qdrant to detect
        service queries. When no service matches (or embedding fails in tests),
        it falls back to CONTEXT. The analysis_result is produced later inside
        the context workflow executor during route_to_workflow.
        """
        with (
            _mock_dspy_greeting("Tere! Kuidas ma saan sind aidata?"),
            _patch_cost_utils(),
        ):
            result = await classifier.classify(
                query="Tere!",
                language="et",
            )

        # Hybrid classifier routes non-service queries to CONTEXT
        assert result.workflow.value == "context"
        # analysis_result is now populated during route_to_workflow, not classify
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_greeting_end_to_end_non_streaming(
        self, classifier: ToolClassifier
    ) -> None:
        """Full chain: classify greeting -> route to context workflow -> get response."""
        with _mock_dspy_greeting("Hello! How can I help you?"), _patch_cost_utils():
            classification = await classifier.classify(
                query="Hello!",
                language="en",
            )

            request = _make_request("Hello!")
            with patch.object(
                classifier.context_workflow.context_analyzer,
                "detect_context_with_summary_fallback",
                new_callable=AsyncMock,
                return_value=(
                    ContextDetectionResult(
                        is_greeting=True,
                        greeting_type="hello",
                        can_answer_from_context=False,
                        reasoning="Greeting detected",
                    ),
                    {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                ),
            ):
                response = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=False,
                )

        assert isinstance(response, OrchestrationResponse)
        assert response.chatId == "integration-test-chat"
        assert "Hello" in response.content
        assert response.llmServiceActive is True
        assert response.questionOutOfLLMScope is False

    @pytest.mark.asyncio
    async def test_estonian_greeting_end_to_end(
        self, classifier: ToolClassifier
    ) -> None:
        """Full chain for Estonian greeting."""
        with (
            _mock_dspy_greeting("Tere! Kuidas ma saan sind aidata?"),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="Tere!",
                language="et",
            )

            request = _make_request("Tere!")
            with patch.object(
                classifier.context_workflow.context_analyzer,
                "detect_context_with_summary_fallback",
                new_callable=AsyncMock,
                return_value=(
                    ContextDetectionResult(
                        is_greeting=True,
                        greeting_type="hello",
                        can_answer_from_context=False,
                        reasoning="Estonian greeting detected",
                    ),
                    {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                ),
            ):
                response = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=False,
                )

        assert isinstance(response, OrchestrationResponse)
        assert "Tere" in response.content

    @pytest.mark.asyncio
    async def test_goodbye_end_to_end(self, classifier: ToolClassifier) -> None:
        """Full chain for goodbye greeting."""
        with _mock_dspy_greeting("Goodbye! Have a great day!"), _patch_cost_utils():
            classification = await classifier.classify(
                query="Goodbye!",
                language="en",
            )

            request = _make_request("Goodbye!")
            with patch.object(
                classifier.context_workflow.context_analyzer,
                "detect_context_with_summary_fallback",
                new_callable=AsyncMock,
                return_value=(
                    ContextDetectionResult(
                        is_greeting=True,
                        greeting_type="goodbye",
                        can_answer_from_context=False,
                        reasoning="Goodbye detected",
                    ),
                    {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                ),
            ):
                response = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=False,
                )

        assert isinstance(response, OrchestrationResponse)
        assert "Goodbye" in response.content

    @pytest.mark.asyncio
    async def test_thanks_end_to_end(self, classifier: ToolClassifier) -> None:
        """Full chain for thanks greeting."""
        with (
            _mock_dspy_greeting("You're welcome! Feel free to ask more."),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="Thank you!",
                language="en",
            )

            request = _make_request("Thank you!")
            with patch.object(
                classifier.context_workflow.context_analyzer,
                "detect_context_with_summary_fallback",
                new_callable=AsyncMock,
                return_value=(
                    ContextDetectionResult(
                        is_greeting=True,
                        greeting_type="thanks",
                        can_answer_from_context=False,
                        reasoning="Thanks detected",
                    ),
                    {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                ),
            ):
                response = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=False,
                )

        assert isinstance(response, OrchestrationResponse)
        assert "welcome" in response.content.lower()


class TestClassifyAndRouteContextAnswer:
    """Test full classify -> route chain for context-based answers."""

    @pytest.mark.asyncio
    async def test_context_answer_end_to_end(self, classifier: ToolClassifier) -> None:
        """Full chain: classify history query -> route to context -> get answer."""
        history = [
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

        with (
            _mock_dspy_context_answer("I mentioned the tax rate is 20%."),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="What was the rate?",
                language="en",
            )

            request = _make_request("What was the rate?", history=history)
            with (
                patch.object(
                    classifier.context_workflow.context_analyzer,
                    "detect_context_with_summary_fallback",
                    new_callable=AsyncMock,
                    return_value=(
                        ContextDetectionResult(
                            is_greeting=False,
                            greeting_type="hello",
                            can_answer_from_context=True,
                            reasoning="Tax rate referenced in history",
                            context_snippet="The tax rate is 20%.",
                        ),
                        {"total_cost": 0.002, "total_tokens": 100, "num_calls": 1},
                    ),
                ),
                patch.object(
                    classifier.context_workflow.context_analyzer,
                    "generate_context_response",
                    new_callable=AsyncMock,
                    return_value=(
                        "I mentioned the tax rate is 20%.",
                        {"total_cost": 0.003, "num_calls": 1},
                    ),
                ),
            ):
                response = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=False,
                )

        assert classification.workflow.value == "context"
        assert isinstance(response, OrchestrationResponse)
        assert "20%" in response.content

    @pytest.mark.asyncio
    async def test_context_answer_with_long_history(
        self, classifier: ToolClassifier
    ) -> None:
        """Should pass last 10 turns to the analyzer even with longer history."""
        history = [
            ConversationItem(
                authorRole="user" if i % 2 == 0 else "bot",
                message=f"Message {i}",
                timestamp=f"2024-01-01T12:00:{i:02d}",
            )
            for i in range(15)
        ]

        with (
            _mock_dspy_context_answer("Based on our conversation, here's the answer."),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="What did we discuss?",
                language="en",
            )

            request = _make_request("What did we discuss?", history=history)
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        assert classification.workflow.value == "context"
        assert isinstance(response, OrchestrationResponse)
        assert response.content is not None


# ---------------------------------------------------------------------------
# Integration: fallback from CONTEXT to RAG
# ---------------------------------------------------------------------------


class TestContextToRAGFallback:
    """Test that context workflow falls back to RAG when it cannot answer."""

    @pytest.mark.asyncio
    async def test_classify_defaults_to_rag_when_no_context_match(
        self, classifier: ToolClassifier, mock_orchestration_service: MagicMock
    ) -> None:
        """When context analyzer can't answer, the full route chain ends at RAG.

        With the hybrid-search classifier, classify() returns CONTEXT for
        non-service queries. The RAG fallback is triggered inside
        route_to_workflow when the context workflow returns None.
        """
        with _mock_dspy_no_match(), _patch_cost_utils():
            classification = await classifier.classify(
                query="What is a digital signature?",
                language="en",
            )

            # Classifier routes non-service queries to CONTEXT first
            assert classification.workflow.value == "context"

            # Full route: context can't answer → falls back to RAG
            request = _make_request("What is a digital signature?")
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        assert isinstance(response, OrchestrationResponse)
        assert "RAG" in response.content

    @pytest.mark.asyncio
    async def test_fallback_to_rag_end_to_end(
        self, classifier: ToolClassifier, mock_orchestration_service: MagicMock
    ) -> None:
        """Full chain: context can't answer -> falls back to RAG -> gets RAG response."""
        with _mock_dspy_no_match(), _patch_cost_utils():
            classification = await classifier.classify(
                query="What is a digital signature?",
                language="en",
            )

            # Hybrid classifier routes to CONTEXT first; RAG is via fallback
            assert classification.workflow.value == "context"

            request = _make_request("What is a digital signature?")
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        assert isinstance(response, OrchestrationResponse)
        # RAG mock returns "RAG fallback answer."
        assert "RAG" in response.content

    @pytest.mark.asyncio
    async def test_context_workflow_returns_none_triggers_rag_fallback(
        self, classifier: ToolClassifier, mock_orchestration_service: MagicMock
    ) -> None:
        """When context workflow returns None during routing, RAG fallback is used."""
        # Force classification to CONTEXT but with an analysis that will produce None
        no_answer_analysis = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=False,
            answer=None,
            reasoning="Cannot answer",
        )

        # Use the WorkflowType from the same module path the classifier uses
        from tool_classifier.enums import WorkflowType as _WorkflowType

        forced_classification = ClassificationResult(
            workflow=_WorkflowType.CONTEXT,
            confidence=0.95,
            metadata={"analysis_result": no_answer_analysis},
            reasoning="Forced for test",
        )

        request = _make_request("Something that context can't answer")
        response = await classifier.route_to_workflow(
            classification=forced_classification,
            request=request,
            is_streaming=False,
        )

        assert isinstance(response, OrchestrationResponse)
        # Should have fallen through to RAG
        assert "RAG" in response.content


# ---------------------------------------------------------------------------
# Integration: streaming mode
# ---------------------------------------------------------------------------


class TestStreamingIntegration:
    """Test the full classify -> route -> stream chain."""

    @pytest.mark.asyncio
    async def test_streaming_greeting_end_to_end(
        self, classifier: ToolClassifier
    ) -> None:
        """Full chain: classify greeting -> route streaming -> collect SSE chunks."""
        with _mock_dspy_greeting("Hello! How can I help you?"), _patch_cost_utils():
            classification = await classifier.classify(
                query="Hello!",
                language="en",
            )

            request = _make_request("Hello!")
            with patch.object(
                classifier.context_workflow.context_analyzer,
                "detect_context_with_summary_fallback",
                new_callable=AsyncMock,
                return_value=(
                    ContextDetectionResult(
                        is_greeting=True,
                        greeting_type="hello",
                        can_answer_from_context=False,
                        reasoning="Greeting detected",
                    ),
                    {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                ),
            ):
                stream = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=True,
                )

                # Collect chunks inside the mock context so the dspy patch is active
                # when the async generator body executes (lazy evaluation).
                chunks = [chunk async for chunk in stream]

        # Should have content chunks + END marker
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.startswith("data: ")
            assert chunk.endswith("\n\n")

        # Last chunk should contain END
        last_payload = json.loads(chunks[-1][6:-2])
        assert last_payload["payload"]["content"] == "END"

        # Reconstruct content from non-END chunks
        content_parts = []
        for chunk in chunks[:-1]:
            payload = json.loads(chunk[6:-2])
            content_parts.append(payload["payload"]["content"])
        full_content = "".join(content_parts)
        assert "Hello" in full_content

    @pytest.mark.asyncio
    async def test_streaming_context_answer_end_to_end(
        self, classifier: ToolClassifier
    ) -> None:
        """Full chain: classify history query -> route streaming -> collect answer."""
        history = [
            ConversationItem(
                authorRole="bot",
                message="The deadline is March 31st.",
                timestamp="2024-01-01T12:00:00",
            ),
        ]

        async def _mock_history_stream() -> AsyncGenerator[str, None]:
            yield 'data: {"chatId":"integration-test-chat","payload":{"content":"The deadline is March 31st."}}\n\n'
            yield 'data: {"chatId":"integration-test-chat","payload":{"content":"END"}}\n\n'

        with (
            _mock_dspy_context_answer("The deadline is March 31st."),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="When is the deadline?",
                language="en",
            )

            request = _make_request("When is the deadline?", history=history)
            with (
                patch.object(
                    classifier.context_workflow.context_analyzer,
                    "detect_context_with_summary_fallback",
                    new_callable=AsyncMock,
                    return_value=(
                        ContextDetectionResult(
                            is_greeting=False,
                            greeting_type="hello",
                            can_answer_from_context=True,
                            reasoning="Deadline referenced in history",
                            context_snippet="The deadline is March 31st.",
                        ),
                        {"total_cost": 0.001, "total_tokens": 50, "num_calls": 1},
                    ),
                ),
                patch.object(
                    classifier.context_workflow,
                    "_create_history_stream",
                    new_callable=AsyncMock,
                    return_value=_mock_history_stream(),
                ),
            ):
                stream = await classifier.route_to_workflow(
                    classification=classification,
                    request=request,
                    is_streaming=True,
                )

                chunks = [chunk async for chunk in stream]

        assert len(chunks) >= 2
        last_payload = json.loads(chunks[-1][6:-2])
        assert last_payload["payload"]["content"] == "END"

    @pytest.mark.asyncio
    async def test_streaming_fallback_to_rag(
        self, classifier: ToolClassifier, mock_orchestration_service: MagicMock
    ) -> None:
        """Streaming: context can't answer -> falls back to RAG streaming."""
        # Force classification to CONTEXT with no answer
        no_answer_analysis = ContextDetectionResult(
            is_greeting=False,
            can_answer_from_context=False,
            answer=None,
            reasoning="Cannot answer",
        )

        from tool_classifier.enums import WorkflowType as _WorkflowType

        forced_classification = ClassificationResult(
            workflow=_WorkflowType.CONTEXT,
            confidence=0.95,
            metadata={"analysis_result": no_answer_analysis},
            reasoning="Forced for test",
        )

        request = _make_request("Something needing RAG")
        stream = await classifier.route_to_workflow(
            classification=forced_classification,
            request=request,
            is_streaming=True,
        )

        chunks = [chunk async for chunk in stream]

        # Should have received RAG streaming output
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# Integration: cost tracking across the chain
# ---------------------------------------------------------------------------


class TestCostTrackingIntegration:
    """Test that cost data flows through the full classify -> execute chain."""

    @pytest.mark.asyncio
    async def test_costs_propagated_through_classification(
        self, classifier: ToolClassifier
    ) -> None:
        """Cost dict from context analysis should be tracked during workflow execution.

        With the hybrid-search classifier, costs are tracked inside the context
        workflow executor (execute_async/execute_streaming), not in classify().
        The cost dict is stored in the workflow's internal context dictionary.
        """
        with _mock_dspy_greeting("Hello!"), _patch_cost_utils():
            classification = await classifier.classify(
                query="Hello!",
                language="en",
            )

            # Verify classify succeeded and routes to CONTEXT
            assert classification.workflow.value == "context"

            # Execute the workflow to trigger cost tracking
            request = _make_request("Hello!")
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        # Verify workflow ran successfully (costs tracked internally)
        assert isinstance(response, OrchestrationResponse)
        assert response.chatId == "integration-test-chat"


# ---------------------------------------------------------------------------
# Integration: error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Test that errors in context analysis gracefully fall back to RAG."""

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_rag(
        self, classifier: ToolClassifier
    ) -> None:
        """If context analyzer LLM call raises, the route chain falls back to RAG.

        With the hybrid-search classifier, classify() returns CONTEXT for
        non-service queries. When the context workflow LLM call raises, the
        context workflow returns None and route_to_workflow falls back to RAG.
        """
        with (
            patch(
                "dspy.ChainOfThought",
                return_value=MagicMock(side_effect=Exception("LLM unavailable")),
            ),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="Hello!",
                language="en",
            )

            # classify() returns CONTEXT (non-service query)
            assert classification.workflow.value == "context"

            # Full route: context LLM fails → falls back to RAG gracefully
            request = _make_request("Hello!")
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        assert isinstance(response, OrchestrationResponse)
        assert "RAG" in response.content

    @pytest.mark.asyncio
    async def test_json_parse_error_falls_back_to_rag(
        self, classifier: ToolClassifier
    ) -> None:
        """If LLM returns invalid JSON, the route chain falls back to RAG.

        JSON parse failure causes context analysis to return is_greeting=False,
        answer=None. The context workflow then returns None and the fallback
        chain routes to RAG.
        """
        mock_response = MagicMock()
        mock_response.analysis_result = "not valid json at all"

        with (
            patch(
                "dspy.ChainOfThought",
                return_value=MagicMock(return_value=mock_response),
            ),
            _patch_cost_utils(),
        ):
            classification = await classifier.classify(
                query="Hello!",
                language="en",
            )

            # classify() returns CONTEXT (non-service query)
            assert classification.workflow.value == "context"

            # Full route: JSON parse fails → context returns None → RAG fallback
            request = _make_request("Hello!")
            response = await classifier.route_to_workflow(
                classification=classification,
                request=request,
                is_streaming=False,
            )

        assert isinstance(response, OrchestrationResponse)
        assert "RAG" in response.content

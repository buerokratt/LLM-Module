"""Unit tests for APIResponseFormatterModule — DSPy JSON-to-natural-language formatter."""

import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import dspy.streaming
import pytest

from tool_classifier.api_response_formatter import (
    APIResponseFormatterModule,
    _FORMATTER_ERROR_MESSAGES,
)


@pytest.fixture(autouse=True)
def mock_dspy_lm() -> Generator[MagicMock, None, None]:
    """Mock DSPy LM to prevent 'No LM is loaded' errors during tests."""
    mock_lm = MagicMock()
    mock_lm.history = []
    with patch("dspy.settings") as mock_settings:
        mock_settings.lm = mock_lm
        dspy.configure(lm=mock_lm)
        yield mock_lm


def _make_mock_result(formatted_answer: str) -> MagicMock:
    """Build a mock DSPy Predict result with the formatted_answer attribute."""
    mock_result = MagicMock()
    mock_result.formatted_answer = formatted_answer
    return mock_result


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestAPIResponseFormatterModuleInit:
    """APIResponseFormatterModule should initialise with the correct attributes."""

    def test_module_has_formatter_attribute(self) -> None:
        module = APIResponseFormatterModule()
        assert hasattr(module, "formatter")

    def test_formatter_is_dspy_predict(self) -> None:
        module = APIResponseFormatterModule()
        assert isinstance(module.formatter, dspy.Predict)


# ---------------------------------------------------------------------------
# Basic formatting
# ---------------------------------------------------------------------------


class TestSimpleFormatting:
    """forward() should return the LLM's formatted answer for valid JSON responses."""

    def test_format_simple_json_response(self) -> None:
        """A valid JSON response should be passed to LLM and its answer returned."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result(
            "The public holidays are: New Year, Independence Day."
        )
        api_response = (
            '{"holidays": [{"name": "New Year"}, {"name": "Independence Day"}]}'
        )

        with patch.object(module, "formatter", return_value=mock_result):
            result = module.forward(
                user_query="What are the public holidays?",
                api_response=api_response,
                endpoint_description="Get public holidays for a country",
            )

        assert result == "The public holidays are: New Year, Independence Day."

    def test_predictor_called_with_correct_fields(self) -> None:
        """forward() must call formatter with all three expected keyword arguments."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Answer")
        api_response = '{"status": "ok"}'

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Is the service running?",
                api_response=api_response,
                endpoint_description="Get service status",
            )

        mock_formatter.assert_called_once()
        call_kwargs = mock_formatter.call_args.kwargs
        assert "user_query" in call_kwargs
        assert "api_response" in call_kwargs
        assert "endpoint_description" in call_kwargs
        assert "response_language" in call_kwargs

    def test_dict_input_converted_to_string(self) -> None:
        """A dict api_response should be JSON-serialised before passing to the LLM."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("The count is 42.")
        api_response_dict = {"count": 42, "items": ["a", "b"]}

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="How many items?",
                api_response=api_response_dict,
                endpoint_description="Get item count",
            )

        assert result == "The count is 42."
        call_kwargs = mock_formatter.call_args.kwargs
        assert isinstance(call_kwargs["api_response"], str)
        parsed = json.loads(call_kwargs["api_response"])
        assert parsed == api_response_dict


# ---------------------------------------------------------------------------
# Empty response handling
# ---------------------------------------------------------------------------


class TestEmptyResponseHandling:
    """forward() should annotate empty responses so the LLM handles them gracefully."""

    def test_format_empty_list_response(self) -> None:
        """An empty list '[]' should be annotated and passed to the LLM."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("No results were found for your query.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="What holidays exist?",
                api_response="[]",
                endpoint_description="Get public holidays",
            )

        assert result == "No results were found for your query."
        call_kwargs = mock_formatter.call_args.kwargs
        assert "EMPTY RESPONSE" in call_kwargs["api_response"]

    def test_format_empty_dict_response(self) -> None:
        """An empty dict '{}' should be annotated before passing to the LLM."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("There is no data available.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="Show me the data",
                api_response="{}",
                endpoint_description="Fetch data",
            )

        assert result == "There is no data available."
        call_kwargs = mock_formatter.call_args.kwargs
        assert "EMPTY RESPONSE" in call_kwargs["api_response"]

    def test_format_null_response(self) -> None:
        """A 'null' JSON response should be annotated before passing to the LLM."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("No information was returned.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="What is the result?",
                api_response="null",
                endpoint_description="Get result",
            )

        assert result == "No information was returned."
        call_kwargs = mock_formatter.call_args.kwargs
        assert "EMPTY RESPONSE" in call_kwargs["api_response"]


# ---------------------------------------------------------------------------
# Error response handling
# ---------------------------------------------------------------------------


class TestErrorResponseHandling:
    """forward() should pass API error responses to the LLM without modification."""

    def test_format_error_response(self) -> None:
        """An error JSON response should be forwarded to the LLM as-is."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Sorry, the requested resource was not found.")
        api_response = '{"error": "not found", "code": 404}'

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="Get my record",
                api_response=api_response,
                endpoint_description="Get record by ID",
            )

        assert result == "Sorry, the requested resource was not found."
        # Error response is not empty — must NOT have the EMPTY RESPONSE annotation
        call_kwargs = mock_formatter.call_args.kwargs
        assert "EMPTY RESPONSE" not in call_kwargs["api_response"]


# ---------------------------------------------------------------------------
# Large response truncation
# ---------------------------------------------------------------------------


class TestLargeResponseTruncation:
    """forward() should truncate responses that exceed the item limit before calling the LLM."""

    def test_format_large_response_truncation(self) -> None:
        """A list with more than 500 items should be truncated and annotated with a NOTE."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Here is a summary of the large dataset.")
        large_response = [{"id": i, "name": f"item_{i}"} for i in range(600)]

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="List all items",
                api_response=large_response,
                endpoint_description="Get all items",
            )

        assert result == "Here is a summary of the large dataset."
        call_kwargs = mock_formatter.call_args.kwargs
        assert "NOTE" in call_kwargs["api_response"]
        assert "500" in call_kwargs["api_response"]
        assert "600" in call_kwargs["api_response"]

    def test_list_within_limit_not_truncated(self) -> None:
        """A list with exactly 500 items should NOT be truncated."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Here are the items.")
        exact_response = [{"id": i} for i in range(500)]

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="List items",
                api_response=exact_response,
                endpoint_description="Get items",
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert "NOTE" not in call_kwargs["api_response"]


# ---------------------------------------------------------------------------
# Language handling
# ---------------------------------------------------------------------------


class TestLanguageHandling:
    """forward() must map detected_language codes to display names for the LLM."""

    @pytest.mark.parametrize(
        "language_code, expected_display",
        [
            ("en", "English"),
            ("et", "Estonian"),
            ("ru", "Russian"),
        ],
    )
    def test_detected_language_mapped_to_display_name(
        self, language_code: str, expected_display: str
    ) -> None:
        """Each ISO code must be forwarded to the LLM as its full display name."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test query",
                api_response='{"key": "value"}',
                endpoint_description="Test endpoint",
                detected_language=language_code,
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["response_language"] == expected_display

    def test_unknown_language_code_defaults_to_english(self) -> None:
        """An unrecognised language code must fall back to 'English'."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test",
                api_response='{"key": "value"}',
                endpoint_description="Test",
                detected_language="fr",  # unsupported code
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["response_language"] == "English"

    def test_default_language_is_english(self) -> None:
        """When detected_language is omitted, response_language must be 'English'."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test",
                api_response='{"key": "value"}',
                endpoint_description="Test",
                # detected_language not passed — should default to 'en'
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["response_language"] == "English"


# ---------------------------------------------------------------------------
# Resilience / error handling
# ---------------------------------------------------------------------------


class TestResilienceHandling:
    """forward() should return a safe fallback message if the LLM call fails."""

    def test_forward_handles_prediction_error(self) -> None:
        """If the DSPy predictor raises an exception, a safe fallback string is returned."""
        module = APIResponseFormatterModule()

        with patch.object(
            module, "formatter", side_effect=RuntimeError("LLM unavailable")
        ):
            result = module.forward(
                user_query="What are the holidays?",
                api_response='{"holidays": []}',
                endpoint_description="Get public holidays",
            )

        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Custom instructions
# ---------------------------------------------------------------------------


class TestCustomInstructions:
    """Custom instructions should be forwarded to the DSPy predictor unchanged."""

    def test_custom_instructions_passed_to_predictor(self) -> None:
        """When custom_instructions is set on the module, it must be sent to the LLM."""
        module = APIResponseFormatterModule(
            custom_instructions="Always respond in Estonian."
        )
        mock_result = _make_mock_result("Pühad on: uusaasta.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="What are the holidays?",
                api_response='{"holidays": ["New Year"]}',
                endpoint_description="Get public holidays",
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["custom_instructions"] == "Always respond in Estonian."

    def test_empty_custom_instructions_passed_by_default(self) -> None:
        """When no custom_instructions provided, the predictor receives an empty string."""
        module = APIResponseFormatterModule()
        mock_result = _make_mock_result("The holidays are: New Year.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="What are the holidays?",
                api_response='{"holidays": ["New Year"]}',
                endpoint_description="Get public holidays",
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["custom_instructions"] == ""

    def test_custom_instructions_stored_on_instance(self) -> None:
        """The custom_instructions value passed to __init__ must be stored."""
        instructions = "Use formal language."
        module = APIResponseFormatterModule(custom_instructions=instructions)
        assert module._custom_instructions == instructions

    def test_default_custom_instructions_is_empty_string(self) -> None:
        """Modules initialised without custom_instructions must default to empty string."""
        module = APIResponseFormatterModule()
        assert module._custom_instructions == ""


# ---------------------------------------------------------------------------
# stream_forward — custom instructions
# ---------------------------------------------------------------------------


def _make_async_iter(*chunks: Any) -> AsyncMock:
    """Return an async context manager that yields the given chunks then closes cleanly."""

    async def _gen() -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = lambda self: _gen()
    mock_stream.aclose = AsyncMock()
    return mock_stream


class TestStreamForwardCustomInstructions:
    """stream_forward() must thread custom_instructions to the stream predictor."""

    @pytest.mark.asyncio
    async def test_custom_instructions_forwarded_to_stream_predictor(self) -> None:
        """When custom_instructions is set, stream_forward must pass it to the predictor."""
        module = APIResponseFormatterModule(
            custom_instructions="Always respond in Estonian."
        )

        captured: dict[str, Any] = {}

        def fake_stream_predictor(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_async_iter()

        mock_predictor = MagicMock(side_effect=fake_stream_predictor)

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="fallback"):
                async for _ in module.stream_forward(
                    user_query="What are the holidays?",
                    api_response='{"holidays": ["New Year"]}',
                    endpoint_description="Get public holidays",
                ):
                    pass

        assert captured.get("custom_instructions") == "Always respond in Estonian."

    @pytest.mark.asyncio
    async def test_empty_custom_instructions_forwarded_to_stream_predictor(
        self,
    ) -> None:
        """When no custom_instructions, stream_forward must pass an empty string."""
        module = APIResponseFormatterModule()

        captured: dict[str, Any] = {}

        def fake_stream_predictor(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_async_iter()

        mock_predictor = MagicMock(side_effect=fake_stream_predictor)

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="fallback"):
                async for _ in module.stream_forward(
                    user_query="Test",
                    api_response='{"key": "value"}',
                    endpoint_description="Test",
                ):
                    pass

        assert captured.get("custom_instructions") == ""

    @pytest.mark.asyncio
    async def test_stream_response_tokens_are_yielded(self) -> None:
        """StreamResponse chunks for formatted_answer must be yielded token by token."""
        module = APIResponseFormatterModule(custom_instructions="Use formal language.")

        token1 = MagicMock(spec=dspy.streaming.StreamResponse)
        token1.signature_field_name = "formatted_answer"
        token1.chunk = "Hello "

        token2 = MagicMock(spec=dspy.streaming.StreamResponse)
        token2.signature_field_name = "formatted_answer"
        token2.chunk = "world."

        mock_predictor = MagicMock(return_value=_make_async_iter(token1, token2))

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward(
                    user_query="Say hello",
                    api_response='{"msg": "hi"}',
                    endpoint_description="Greet",
                )
            ]

        assert tokens == ["Hello ", "world."]

    @pytest.mark.asyncio
    async def test_prediction_fallback_yields_full_answer(self) -> None:
        """When streamify yields a Prediction (no tokens), the full answer is yielded once."""
        module = APIResponseFormatterModule()

        prediction = MagicMock(spec=dspy.Prediction)
        prediction.formatted_answer = "Full answer without streaming."

        mock_predictor = MagicMock(return_value=_make_async_iter(prediction))

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward(
                    user_query="What is the answer?",
                    api_response='{"result": 42}',
                    endpoint_description="Get result",
                )
            ]

        assert tokens == ["Full answer without streaming."]

    @pytest.mark.asyncio
    async def test_blocking_forward_used_as_last_resort(self) -> None:
        """When no tokens and no Prediction, stream_forward falls back to forward()."""
        module = APIResponseFormatterModule()

        mock_predictor = MagicMock(return_value=_make_async_iter())

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            with patch.object(
                module, "forward", return_value="Blocking fallback."
            ) as mock_fwd:
                tokens = [
                    t
                    async for t in module.stream_forward(
                        user_query="Fallback test",
                        api_response='{"x": 1}',
                        endpoint_description="Test",
                        detected_language="en",
                    )
                ]

        mock_fwd.assert_called_once()
        assert tokens == ["Blocking fallback."]

    @pytest.mark.asyncio
    async def test_stream_forward_yields_localized_error_on_exception(self) -> None:
        """If the stream predictor raises, stream_forward yields a non-empty fallback."""
        module = APIResponseFormatterModule()

        mock_predictor = MagicMock(side_effect=RuntimeError("stream broken"))

        with patch.object(module, "_get_stream_predictor", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward(
                    user_query="Anything",
                    api_response='{"x": 1}',
                    endpoint_description="Test",
                    detected_language="en",
                )
            ]

        assert len(tokens) == 1
        assert isinstance(tokens[0], str)
        assert len(tokens[0]) > 0


# ---------------------------------------------------------------------------
# stream_forward
# ---------------------------------------------------------------------------


class TestStreamForward:
    """Tests for stream_forward() on APIResponseFormatterModule."""

    @pytest.mark.asyncio
    async def test_yields_tokens_from_stream_response(self) -> None:
        """stream_forward() should yield token strings from the DSPy stream."""
        formatter = APIResponseFormatterModule()
        tokens_emitted = ["Estonia ", "has ", "10 ", "holidays."]

        async def _fake_stream(
            *args: object, **kwargs: object
        ) -> AsyncIterator[object]:
            from dspy.streaming import StreamResponse

            for token in tokens_emitted:
                yield StreamResponse(
                    predict_name="formatter",
                    signature_field_name="formatted_answer",
                    chunk=token,
                    is_last_chunk=False,
                )

            yield dspy.Prediction(formatted_answer="Estonia has 10 holidays.")

        with patch.object(
            formatter, "_get_stream_predictor", return_value=_fake_stream
        ):
            collected = [
                token
                async for token in formatter.stream_forward(
                    user_query="How many holidays in Estonia?",
                    api_response={"count": 10},
                    endpoint_description="Returns public holidays",
                    detected_language="en",
                )
            ]

        assert collected == tokens_emitted

    @pytest.mark.asyncio
    async def test_exception_in_stream_yields_localized_error(self) -> None:
        """When the stream predictor raises, stream_forward yields the localized error.

        forward() is NOT called — the exception is caught directly and the
        localized error message is yielded instead.
        """
        formatter = APIResponseFormatterModule()

        async def _raise_stream(
            *args: object, **kwargs: object
        ) -> AsyncIterator[object]:
            raise RuntimeError("Streaming unavailable")
            yield  # noqa: F821

        forward_mock = MagicMock(return_value="Should not be called")
        formatter._get_stream_predictor = MagicMock(return_value=_raise_stream)
        formatter.forward = forward_mock

        collected = [
            token
            async for token in formatter.stream_forward(
                user_query="Holidays?",
                api_response={"holidays": []},
                endpoint_description="Returns public holidays",
                detected_language="en",
            )
        ]

        assert collected == [_FORMATTER_ERROR_MESSAGES["en"]]
        forward_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_yields_error_message_on_total_failure(self) -> None:
        """When streaming raises, yield the localized error message for the given language."""
        formatter = APIResponseFormatterModule()

        async def _raise_stream(
            *args: object, **kwargs: object
        ) -> AsyncIterator[object]:
            raise RuntimeError("Streaming unavailable")
            yield  # noqa: F821

        formatter._get_stream_predictor = MagicMock(return_value=_raise_stream)

        collected = [
            token
            async for token in formatter.stream_forward(
                user_query="Holidays?",
                api_response={"holidays": []},
                endpoint_description="Returns public holidays",
                detected_language="en",
            )
        ]

        assert collected == [_FORMATTER_ERROR_MESSAGES["en"]]

"""Unit tests for MultiResponseFormatterModule — multi-API result synthesiser."""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import dspy
import dspy.streaming
import pytest

from tool_classifier.multi_response_formatter import (
    MultiResponseFormatterModule,
    _MULTI_FORMATTER_ERROR_MESSAGES,
    _MAX_TOTAL_RESPONSE_BYTES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_dspy_lm() -> Any:
    """Mock DSPy LM to prevent 'No LM is loaded' errors during tests."""
    mock_lm = MagicMock()
    mock_lm.history = []
    with patch("dspy.settings") as mock_settings:
        mock_settings.lm = mock_lm
        dspy.configure(lm=mock_lm)
        yield mock_lm


def _make_mock_result(unified_answer: str) -> MagicMock:
    """Build a mock DSPy Predict result with the unified_answer attribute."""
    mock_result = MagicMock()
    mock_result.unified_answer = unified_answer
    return mock_result


def _make_async_iter(*chunks: Any) -> AsyncMock:
    """Return an async iterable that yields the given chunks then closes cleanly."""

    async def _gen() -> AsyncGenerator[Any, None]:
        for chunk in chunks:
            yield chunk

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = lambda self: _gen()
    mock_stream.aclose = AsyncMock()
    return mock_stream


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestMultiResponseFormatterModuleInit:
    """MultiResponseFormatterModule should initialise with the correct attributes."""

    def test_module_has_formatter_attribute(self) -> None:
        module = MultiResponseFormatterModule()
        assert hasattr(module, "formatter")

    def test_formatter_is_dspy_predict(self) -> None:
        module = MultiResponseFormatterModule()
        assert isinstance(module.formatter, dspy.Predict)

    def test_default_custom_instructions_is_empty_string(self) -> None:
        module = MultiResponseFormatterModule()
        assert module._custom_instructions == ""

    def test_custom_instructions_stored_on_instance(self) -> None:
        module = MultiResponseFormatterModule(
            custom_instructions="Use formal language."
        )
        assert module._custom_instructions == "Use formal language."


# ---------------------------------------------------------------------------
# _build_results_block
# ---------------------------------------------------------------------------


class TestBuildResultsBlock:
    """_build_results_block() should serialise results into labeled text sections."""

    def test_empty_list_returns_no_results_marker(self) -> None:
        result = MultiResponseFormatterModule._build_results_block([])
        assert "NO RESULTS" in result

    def test_single_result_contains_endpoint_name(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [
                (
                    "get_current_weather",
                    "Too praegused ja kombineeritud ilmaandmed",
                    '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 12.5}]}}',
                    {},
                )
            ]
        )
        assert "get_current_weather" in block
        assert "Too praegused ja kombineeritud ilmaandmed" in block
        assert "12.5" in block

    def test_single_result_section_header_format(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [
                (
                    "get_current_weather",
                    "Too praegused ja kombineeritud ilmaandmed",
                    '{"observations": {"station": [{"name": "Tallinn"}]}}',
                    {},
                )
            ]
        )
        assert "Result 1: get_current_weather" in block

    def test_multiple_results_all_sections_present(self) -> None:
        results = [
            (
                "get_current_weather",
                "Too praegused ilmaandmed",
                '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 12.5}]}}',
                {},
            ),
            (
                "get_public_holidays",
                "Too riiklikud pühad konkreetse riigi kohta",
                '[{"startDate": "2026-01-01", "name": [{"language": "ET", "text": "Uusaasta"}]}]',
                {},
            ),
            (
                "get_public_holidays_lv",
                "Too Läti riiklikud pühad",
                '[{"startDate": "2026-11-18", "name": [{"language": "LV", "text": "Latvijas Republikas proklamēšanas diena"}]}]',
                {},
            ),
        ]
        block = MultiResponseFormatterModule._build_results_block(results)
        assert "Result 1: get_current_weather" in block
        assert "Result 2: get_public_holidays" in block
        assert "Result 3: get_public_holidays_lv" in block

    def test_null_response_annotated_as_empty(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [("get_current_weather", "Too praegused ilmaandmed", "null", {})]
        )
        assert "EMPTY RESPONSE" in block

    def test_empty_list_response_annotated(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [("get_public_holidays", "Too riiklikud pühad", "[]", {})]
        )
        assert "EMPTY RESPONSE" in block

    def test_empty_dict_response_annotated(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [("get_current_weather", "Too praegused ilmaandmed", "{}", {})]
        )
        assert "EMPTY RESPONSE" in block

    def test_dict_api_response_serialised_to_json(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [
                (
                    "get_current_weather",
                    "Too praegused ilmaandmed",
                    {
                        "observations": {
                            "station": [{"name": "Tallinn", "airtemperature": 12.5}]
                        }
                    },
                    {},
                )
            ]
        )
        assert "Tallinn" in block
        assert "12.5" in block

    def test_list_api_response_serialised_to_json(self) -> None:
        block = MultiResponseFormatterModule._build_results_block(
            [
                (
                    "get_public_holidays",
                    "Too riiklikud pühad",
                    [
                        {"startDate": "2026-01-01", "name": [{"text": "Uusaasta"}]},
                        {
                            "startDate": "2026-02-24",
                            "name": [{"text": "Eesti Vabariigi aastapäev"}],
                        },
                    ],
                    {},
                )
            ]
        )
        assert '"startDate"' in block

    def test_large_combined_response_truncated(self) -> None:
        """Combined block exceeding _MAX_TOTAL_RESPONSE_BYTES should be truncated."""
        big_response = "x" * (_MAX_TOTAL_RESPONSE_BYTES + 10_000)
        results = [
            ("get_current_weather", "Too praegused ilmaandmed", big_response, {}),
            ("get_public_holidays", "Too riiklikud pühad", '{"small": true}', {}),
        ]
        block = MultiResponseFormatterModule._build_results_block(results)
        block_bytes = len(block.encode("utf-8"))
        # Allow small overshoot from truncation note text
        assert block_bytes <= _MAX_TOTAL_RESPONSE_BYTES + 200

    def test_large_combined_response_has_truncation_note(self) -> None:
        big_response = "x" * (_MAX_TOTAL_RESPONSE_BYTES + 10_000)
        block = MultiResponseFormatterModule._build_results_block(
            [("get_current_weather", "Too praegused ilmaandmed", big_response, {})]
        )
        assert "truncated" in block.lower()

    def test_per_result_truncation_applied(self) -> None:
        """Individual responses over 50KB should be truncated by _truncate_if_needed."""
        large_items = [{"id": i, "data": "x" * 200} for i in range(600)]
        block = MultiResponseFormatterModule._build_results_block(
            [("get_public_holidays", "Too riiklikud pühad", large_items, {})]
        )
        assert "NOTE" in block


# ---------------------------------------------------------------------------
# forward() — basic field mapping
# ---------------------------------------------------------------------------


class TestForwardFieldMapping:
    """forward() must pass the correct keyword arguments to the DSPy predictor."""

    def test_predictor_called_with_all_fields(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Combined answer.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="What is the current weather and upcoming public holidays in Estonia?",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ja kombineeritud ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 12.5}]}}',
                        {},
                    ),
                    (
                        "get_public_holidays",
                        "Too riiklikud pühad konkreetse riigi kohta",
                        '[{"startDate": "2026-01-01", "name": [{"language": "ET", "text": "Uusaasta"}]}]',
                        {},
                    ),
                ],
            )

        mock_formatter.assert_called_once()
        call_kwargs = mock_formatter.call_args.kwargs
        assert "user_query" in call_kwargs
        assert "api_results_block" in call_kwargs
        assert "response_language" in call_kwargs
        assert "custom_instructions" in call_kwargs
        assert "num_results" in call_kwargs

    def test_num_results_matches_input_length(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Query",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn"}]}}',
                        {},
                    ),
                    (
                        "get_public_holidays",
                        "Too riiklikud pühad",
                        '[{"startDate": "2026-01-01"}]',
                        {},
                    ),
                    (
                        "get_public_holidays_lv",
                        "Too Läti riiklikud pühad",
                        '[{"startDate": "2026-11-18"}]',
                        {},
                    ),
                ],
            )

        call_kwargs = mock_formatter.call_args.kwargs
        assert call_kwargs["num_results"] == "3"

    def test_returns_unified_answer(self) -> None:
        module = MultiResponseFormatterModule()
        expected = "This is the synthesised answer."
        mock_result = _make_mock_result(expected)

        with patch.object(module, "formatter", return_value=mock_result):
            result = module.forward(
                user_query="Query",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 12.5}]}}',
                        {},
                    )
                ],
            )

        assert result == expected

    def test_empty_api_results_still_calls_predictor(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("No results.")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            result = module.forward(
                user_query="Anything",
                api_results=[],
            )

        mock_formatter.assert_called_once()
        assert result == "No results."


# ---------------------------------------------------------------------------
# forward() — language mapping
# ---------------------------------------------------------------------------


class TestForwardLanguageMapping:
    """forward() must map ISO codes to display names."""

    @pytest.mark.parametrize(
        "language_code, expected_display",
        [
            ("en", "English"),
            ("et", "Estonian"),
            ("ru", "Russian"),
        ],
    )
    def test_language_code_mapped_to_display_name(
        self, language_code: str, expected_display: str
    ) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn"}]}}',
                        {},
                    )
                ],
                detected_language=language_code,
            )

        assert mock_formatter.call_args.kwargs["response_language"] == expected_display

    def test_unknown_language_defaults_to_english(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test",
                api_results=[
                    (
                        "get_public_holidays",
                        "Too riiklikud pühad",
                        '[{"startDate": "2026-01-01"}]',
                        {},
                    )
                ],
                detected_language="fr",
            )

        assert mock_formatter.call_args.kwargs["response_language"] == "English"

    def test_default_language_is_english(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Test",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn"}]}}',
                        {},
                    )
                ],
            )

        assert mock_formatter.call_args.kwargs["response_language"] == "English"


# ---------------------------------------------------------------------------
# forward() — custom instructions
# ---------------------------------------------------------------------------


class TestForwardCustomInstructions:
    """Custom instructions must be forwarded verbatim to the predictor."""

    def test_custom_instructions_passed_to_predictor(self) -> None:
        module = MultiResponseFormatterModule(
            custom_instructions="Always respond in Estonian."
        )
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Query",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn"}]}}',
                        {},
                    )
                ],
            )

        assert (
            mock_formatter.call_args.kwargs["custom_instructions"]
            == "Always respond in Estonian."
        )

    def test_empty_custom_instructions_by_default(self) -> None:
        module = MultiResponseFormatterModule()
        mock_result = _make_mock_result("Answer")

        with patch.object(
            module, "formatter", return_value=mock_result
        ) as mock_formatter:
            module.forward(
                user_query="Query",
                api_results=[
                    (
                        "get_public_holidays",
                        "Too riiklikud pühad",
                        '[{"startDate": "2026-01-01"}]',
                        {},
                    )
                ],
            )

        assert mock_formatter.call_args.kwargs["custom_instructions"] == ""


# ---------------------------------------------------------------------------
# forward() — error handling
# ---------------------------------------------------------------------------


class TestForwardErrorHandling:
    """forward() must return a localized fallback if the predictor raises."""

    @pytest.mark.parametrize("language_code", ["en", "et", "ru"])
    def test_predictor_exception_returns_localized_error(
        self, language_code: str
    ) -> None:
        module = MultiResponseFormatterModule()

        with patch.object(
            module, "formatter", side_effect=RuntimeError("LLM unavailable")
        ):
            result = module.forward(
                user_query="Test",
                api_results=[
                    (
                        "get_current_weather",
                        "Too praegused ilmaandmed",
                        '{"observations": {"station": [{"name": "Tallinn"}]}}',
                        {},
                    )
                ],
                detected_language=language_code,
            )

        assert isinstance(result, str)
        assert len(result) > 0
        assert result == _MULTI_FORMATTER_ERROR_MESSAGES[language_code]

    def test_unknown_language_exception_falls_back_to_english_error(self) -> None:
        module = MultiResponseFormatterModule()

        with patch.object(module, "formatter", side_effect=RuntimeError("failure")):
            result = module.forward(
                user_query="Test",
                api_results=[],
                detected_language="fr",
            )

        assert result == _MULTI_FORMATTER_ERROR_MESSAGES["en"]


# ---------------------------------------------------------------------------
# stream_forward_multi — token streaming (Tier 1)
# ---------------------------------------------------------------------------


class TestStreamForwardMultiTokenStreaming:
    """stream_forward_multi() should yield StreamResponse tokens for unified_answer."""

    @pytest.mark.asyncio
    async def test_stream_response_tokens_are_yielded(self) -> None:
        module = MultiResponseFormatterModule()

        token1 = MagicMock(spec=dspy.streaming.StreamResponse)
        token1.signature_field_name = "unified_answer"
        token1.chunk = "Weather is "

        token2 = MagicMock(spec=dspy.streaming.StreamResponse)
        token2.signature_field_name = "unified_answer"
        token2.chunk = "sunny."

        mock_predictor = MagicMock(return_value=_make_async_iter(token1, token2))

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="What's the current weather in Tallinn?",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ja kombineeritud ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 25}]}}',
                            {},
                        )
                    ],
                )
            ]

        assert tokens == ["Weather is ", "sunny."]

    @pytest.mark.asyncio
    async def test_tokens_for_wrong_field_not_yielded(self) -> None:
        """Tokens for a different signature field must not be yielded."""
        module = MultiResponseFormatterModule()

        wrong_token = MagicMock(spec=dspy.streaming.StreamResponse)
        wrong_token.signature_field_name = "other_field"
        wrong_token.chunk = "should not appear"

        right_token = MagicMock(spec=dspy.streaming.StreamResponse)
        right_token.signature_field_name = "unified_answer"
        right_token.chunk = "correct"

        mock_predictor = MagicMock(
            return_value=_make_async_iter(wrong_token, right_token)
        )

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_public_holidays",
                            "Too riiklikud pühad",
                            '[{"startDate": "2026-01-01"}]',
                            {},
                        )
                    ],
                )
            ]

        assert tokens == ["correct"]


# ---------------------------------------------------------------------------
# stream_forward_multi — Prediction fallback (Tier 2)
# ---------------------------------------------------------------------------


class TestStreamForwardMultiPredictionFallback:
    """When no StreamResponse tokens arrive, fall back to Prediction.unified_answer."""

    @pytest.mark.asyncio
    async def test_prediction_fallback_yields_full_answer(self) -> None:
        module = MultiResponseFormatterModule()

        prediction = MagicMock(spec=dspy.Prediction)
        prediction.unified_answer = "Full synthesised answer."

        mock_predictor = MagicMock(return_value=_make_async_iter(prediction))

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn"}]}}',
                            {},
                        )
                    ],
                )
            ]

        assert tokens == ["Full synthesised answer."]

    @pytest.mark.asyncio
    async def test_prediction_without_unified_answer_falls_to_blocking(self) -> None:
        """A Prediction with no unified_answer attribute triggers the blocking fallback."""
        module = MultiResponseFormatterModule()

        prediction = MagicMock(spec=dspy.Prediction)
        prediction.unified_answer = None

        mock_predictor = MagicMock(return_value=_make_async_iter(prediction))

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="blocking result"):
                tokens = [
                    t
                    async for t in module.stream_forward_multi(
                        user_query="Query",
                        api_results=[
                            (
                                "get_public_holidays",
                                "Too riiklikud pühad",
                                '[{"startDate": "2026-01-01"}]',
                                {},
                            )
                        ],
                    )
                ]

        assert tokens == ["blocking result"]


# ---------------------------------------------------------------------------
# stream_forward_multi — blocking forward() fallback (Tier 3)
# ---------------------------------------------------------------------------


class TestStreamForwardMultiBlockingFallback:
    """When streamify yields nothing, fall back to the blocking forward()."""

    @pytest.mark.asyncio
    async def test_blocking_forward_used_when_no_output(self) -> None:
        module = MultiResponseFormatterModule()

        mock_predictor = MagicMock(return_value=_make_async_iter())

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(
                module, "forward", return_value="Blocking fallback."
            ) as mock_fwd:
                tokens = [
                    t
                    async for t in module.stream_forward_multi(
                        user_query="Fallback test",
                        api_results=[("ep", "desc", '{"x": 1}', {})],
                        detected_language="en",
                    )
                ]

        mock_fwd.assert_called_once()
        assert tokens == ["Blocking fallback."]

    @pytest.mark.asyncio
    async def test_blocking_forward_receives_correct_args(self) -> None:
        module = MultiResponseFormatterModule()
        api_results = [
            (
                "get_current_weather",
                "Too praegused ilmaandmed",
                '{"observations": {"station": [{"name": "Tallinn", "airtemperature": 12.5}]}}',
                {},
            ),
            (
                "get_public_holidays",
                "Too riiklikud pühad",
                '[{"startDate": "2026-01-01", "name": [{"language": "ET", "text": "Uusaasta"}]}]',
                {},
            ),
        ]

        mock_predictor = MagicMock(return_value=_make_async_iter())

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="answer") as mock_fwd:
                async for _ in module.stream_forward_multi(
                    user_query="Test query",
                    api_results=api_results,
                    detected_language="et",
                ):
                    pass

        mock_fwd.assert_called_once_with(
            user_query="Test query",
            api_results=api_results,
            detected_language="et",
        )


# ---------------------------------------------------------------------------
# stream_forward_multi — localized error fallback (Tier 4)
# ---------------------------------------------------------------------------


class TestStreamForwardMultiErrorFallback:
    """On exception, stream_forward_multi must yield a localized error string."""

    @pytest.mark.asyncio
    async def test_exception_yields_localized_error(self) -> None:
        module = MultiResponseFormatterModule()

        mock_predictor = MagicMock(side_effect=RuntimeError("stream broken"))

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Anything",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn"}]}}',
                            {},
                        )
                    ],
                    detected_language="en",
                )
            ]

        assert len(tokens) == 1
        assert isinstance(tokens[0], str)
        assert tokens[0] == _MULTI_FORMATTER_ERROR_MESSAGES["en"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language_code", ["en", "et", "ru"])
    async def test_exception_yields_correct_language_error(
        self, language_code: str
    ) -> None:
        module = MultiResponseFormatterModule()

        mock_predictor = MagicMock(side_effect=RuntimeError("failure"))

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Test",
                    api_results=[
                        (
                            "get_public_holidays",
                            "Too riiklikud pühad",
                            '[{"startDate": "2026-01-01"}]',
                            {},
                        ),
                    ],
                    detected_language=language_code,
                )
            ]

        assert tokens[0] == _MULTI_FORMATTER_ERROR_MESSAGES[language_code]

    @pytest.mark.asyncio
    async def test_unknown_language_exception_uses_english_error(self) -> None:
        module = MultiResponseFormatterModule()

        mock_predictor = MagicMock(side_effect=RuntimeError("failure"))

        with patch("dspy.streamify", return_value=mock_predictor):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Test",
                    api_results=[],
                    detected_language="fr",
                )
            ]

        assert tokens[0] == _MULTI_FORMATTER_ERROR_MESSAGES["en"]


# ---------------------------------------------------------------------------
# stream_forward_multi — custom instructions threading
# ---------------------------------------------------------------------------


class TestStreamForwardMultiCustomInstructions:
    """stream_forward_multi must thread custom_instructions to the stream predictor."""

    @pytest.mark.asyncio
    async def test_custom_instructions_forwarded(self) -> None:
        module = MultiResponseFormatterModule(
            custom_instructions="Always respond in Estonian."
        )

        captured: dict[str, Any] = {}

        def fake_stream_predictor(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_async_iter()

        mock_predictor = MagicMock(side_effect=fake_stream_predictor)

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="fallback"):
                async for _ in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn"}]}}',
                            {},
                        )
                    ],
                ):
                    pass

        assert captured.get("custom_instructions") == "Always respond in Estonian."

    @pytest.mark.asyncio
    async def test_empty_custom_instructions_forwarded(self) -> None:
        module = MultiResponseFormatterModule()

        captured: dict[str, Any] = {}

        def fake_stream_predictor(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return _make_async_iter()

        mock_predictor = MagicMock(side_effect=fake_stream_predictor)

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="fallback"):
                async for _ in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_public_holidays",
                            "Too riiklikud pühad",
                            '[{"startDate": "2026-01-01"}]',
                            {},
                        )
                    ],
                ):
                    pass

        assert captured.get("custom_instructions") == ""


# ---------------------------------------------------------------------------
# stream_forward_multi — stream cleanup
# ---------------------------------------------------------------------------


class TestStreamForwardMultiCleanup:
    """stream_forward_multi must call aclose() on the output stream."""

    @pytest.mark.asyncio
    async def test_aclose_called_on_stream(self) -> None:
        module = MultiResponseFormatterModule()
        mock_stream = _make_async_iter()

        mock_predictor = MagicMock(return_value=mock_stream)

        with patch("dspy.streamify", return_value=mock_predictor):
            with patch.object(module, "forward", return_value="fallback"):
                async for _ in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn"}]}}',
                            {},
                        )
                    ],
                ):
                    pass

        mock_stream.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_aclose_called_even_on_exception(self) -> None:
        """Exception during stream_forward_multi must not propagate — error is yielded."""
        module = MultiResponseFormatterModule()

        # Force the exception at the build_results_block step to test finally
        with patch.object(
            module,
            "_build_results_block",
            side_effect=RuntimeError("build error"),
        ):
            tokens = [
                t
                async for t in module.stream_forward_multi(
                    user_query="Query",
                    api_results=[
                        (
                            "get_current_weather",
                            "Too praegused ilmaandmed",
                            '{"observations": {"station": [{"name": "Tallinn"}]}}',
                            {},
                        )
                    ],
                    detected_language="en",
                )
            ]

        # Exception is caught — a localized error is yielded instead of propagating
        assert len(tokens) == 1
        assert tokens[0] == _MULTI_FORMATTER_ERROR_MESSAGES["en"]

"""Unit tests for ParamExtractionModule — DSPy param extraction from natural language."""

import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import dspy
import dspy.streaming
import pytest

from tool_classifier.param_extractor import (
    ParamExtractionModule,
    strip_format_hints,
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


def _make_mock_result(
    extracted_params: dict,
    missing_required: list,
    clarifying_question: str,
) -> MagicMock:
    """Build a mock DSPy Predict result with the expected output attributes."""
    mock_result = MagicMock()
    mock_result.extracted_params = json.dumps(extracted_params, ensure_ascii=False)
    mock_result.missing_required = json.dumps(missing_required, ensure_ascii=False)
    mock_result.clarifying_question = clarifying_question
    return mock_result


def _make_schema(
    *params: tuple[str, str, bool, str],
) -> list[dict]:
    """
    Build a param schema list.

    Each element is (name, type, required, description).
    """
    return [
        {"name": n, "type": t, "required": r, "description": d} for n, t, r, d in params
    ]


class TestParamExtractionModuleInit:
    """ParamExtractionModule should initialise correctly."""

    def test_module_has_extractor_attribute(self) -> None:
        module = ParamExtractionModule()
        assert hasattr(module, "extractor")

    def test_extractor_is_dspy_predict(self) -> None:
        module = ParamExtractionModule()
        assert isinstance(module.extractor, dspy.Predict)


class TestParamExtraction:
    """forward() should extract parameter values from user message."""

    def test_extract_single_param_from_message(self) -> None:
        """A city name in the message should be extracted as the city param."""
        schema = _make_schema(("city", "string", True, "City name for the query"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"city": "Tallinn"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Weather in Tallinn", params_schema=schema
            )

        assert result["extracted_params"] == {"city": "Tallinn"}
        assert result["missing_required"] == []
        assert result["clarifying_question"] == "none"

    def test_extract_multiple_params_from_message(self) -> None:
        """Multiple params present in one message should all be extracted."""
        schema = _make_schema(
            ("fromCurrency", "string", True, "Source currency"),
            ("toCurrency", "string", True, "Target currency"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {"fromCurrency": "EUR", "toCurrency": "USD"}, [], "none"
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Convert 100 EUR to USD", params_schema=schema
            )

        assert result["extracted_params"] == {
            "fromCurrency": "EUR",
            "toCurrency": "USD",
        }
        assert result["missing_required"] == []

    def test_extraction_uses_conversation_history(self) -> None:
        """A param mentioned in prior history, not current message, should be extracted."""
        schema = _make_schema(("countryCode", "string", True, "ISO country code"))
        history = [{"authorRole": "user", "message": "For Estonia"}]
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"countryCode": "EE"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="What are the public holidays?",
                params_schema=schema,
                conversation_history=history,
            )

        assert result["extracted_params"] == {"countryCode": "EE"}

    def test_already_collected_params_not_re_extracted(self) -> None:
        """Params already collected should remain absent from extracted_params."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date of query"),
        )
        module = ParamExtractionModule()

        # LLM only extracted the new param (date); city was passed as already_collected
        mock_result = _make_mock_result({"date": "2026-04-03"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="For April 3rd",
                params_schema=schema,
                already_collected={"city": "Tallinn"},
            )

        assert "city" not in result["extracted_params"]
        assert result["extracted_params"].get("date") == "2026-04-03"
        assert result["missing_required"] == []

    def test_llm_reoutput_of_collected_param_overrides_prior_value(self) -> None:
        """When the LLM re-extracts a param already in already_collected, the new value
        overrides the old one — corrections from the user are intentionally allowed."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date of query"),
        )
        module = ParamExtractionModule()

        # LLM re-extracts 'city' with a corrected value alongside 'date'
        mock_result = _make_mock_result(
            {"city": "Tartu", "date": "2026-04-09"}, [], "none"
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="April 9th in Tartu",
                params_schema=schema,
                already_collected={"city": "Tallinn"},
            )

        # 'city' override IS present in extracted_params (correction allowed)
        assert result["extracted_params"].get("city") == "Tartu"
        # The new param 'date' extracted in this turn should also be present
        assert result["extracted_params"].get("date") == "2026-04-09"
        assert result["missing_required"] == []

    def test_params_not_in_schema_are_ignored(self) -> None:
        """Values for unknown param names should not appear in extracted_params."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        # LLM hallucinated an extra param not in the schema
        mock_result = _make_mock_result(
            {"city": "Tallinn", "unknownParam": "garbage"}, [], "none"
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Tallinn please", params_schema=schema)

        assert "unknownParam" not in result["extracted_params"]
        assert result["extracted_params"] == {"city": "Tallinn"}


class TestTypeValidation:
    """_validate_param_type() and forward() should enforce typed values."""

    def test_valid_date_accepted(self) -> None:
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("2026-04-03", "date")
        assert valid is True
        assert coerced == "2026-04-03"

    def test_invalid_date_rejected(self) -> None:
        module = ParamExtractionModule()
        valid, _ = module._validate_param_type("not-a-date", "date")
        assert valid is False

    def test_valid_integer_accepted(self) -> None:
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("42", "integer")
        assert valid is True
        assert coerced == 42

    def test_float_string_rejected_for_integer(self) -> None:
        module = ParamExtractionModule()
        valid, _ = module._validate_param_type("3.5", "integer")
        assert valid is False

    def test_valid_number_accepted(self) -> None:
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("3.14", "number")
        assert valid is True
        assert coerced == pytest.approx(3.14)

    def test_boolean_true_variants(self) -> None:
        module = ParamExtractionModule()
        for truthy in ("true", "yes", "jah", "1", "on", "õige", "да"):
            valid, coerced = module._validate_param_type(truthy, "boolean")
            assert valid is True, f"Expected {truthy!r} to be truthy"
            assert coerced is True

    def test_boolean_false_variants(self) -> None:
        module = ParamExtractionModule()
        for falsy in ("false", "no", "ei", "0", "off", "vale", "нет"):
            valid, coerced = module._validate_param_type(falsy, "boolean")
            assert valid is True, f"Expected {falsy!r} to be falsy"
            assert coerced is False

    def test_native_python_bool_accepted(self) -> None:
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type(True, "boolean")
        assert valid is True
        assert coerced is True

    def test_string_type_accepts_any_string(self) -> None:
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("hello world", "string")
        assert valid is True
        assert coerced == "hello world"

    def test_invalid_type_invalid_value_moves_to_missing_required(self) -> None:
        """A required param that fails type validation must appear in missing_required."""
        schema = _make_schema(("startDate", "date", True, "Start date for the query"))
        module = ParamExtractionModule()

        # LLM extracted a non-date value for a date param
        mock_result = _make_mock_result({"startDate": "next week"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Starting next week", params_schema=schema
            )

        assert "startDate" not in result["extracted_params"]
        assert "startDate" in result["missing_required"]

    def test_optional_param_type_failure_not_in_missing_required(self) -> None:
        """An optional param failing type validation should NOT appear in missing_required."""
        schema = _make_schema(("maxResults", "integer", False, "Max number of results"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"maxResults": "many"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Give me results", params_schema=schema
            )

        assert "maxResults" not in result["extracted_params"]
        assert "maxResults" not in result["missing_required"]

    def test_integer_with_sign_prefix_accepted(self) -> None:
        """Strings like '+42' are valid integer inputs and should coerce to int."""
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("+42", "integer")
        assert valid is True
        assert coerced == 42

    def test_large_integer_with_sign_prefix_accepted(self) -> None:
        """Large integers with a '+' prefix must not be rejected by float precision loss."""
        module = ParamExtractionModule()
        large = str(2**53 + 1)  # 9007199254740993 — beyond float64 exact range
        valid, coerced = module._validate_param_type("+" + large, "integer")
        assert valid is True
        assert coerced == 2**53 + 1

    def test_none_value_rejected_for_all_types(self) -> None:
        """None input should always return (False, None) regardless of expected type."""
        module = ParamExtractionModule()
        for type_name in ("string", "integer", "number", "date", "boolean"):
            valid, _ = module._validate_param_type(None, type_name)
            assert valid is False, (
                f"Expected None to be rejected for type {type_name!r}"
            )

    def test_unknown_param_type_accepted_as_string(self) -> None:
        """An unrecognised type should fall back to accepting the value as a string."""
        module = ParamExtractionModule()
        valid, coerced = module._validate_param_type("some_value", "unsupported_type")
        assert valid is True
        assert coerced == "some_value"


class TestMissingParamDetection:
    """forward() should accurately report missing required params."""

    def test_identifies_all_missing_required_params(self) -> None:
        schema = _make_schema(
            ("fromCurrency", "string", True, "Source currency"),
            ("toCurrency", "string", True, "Target currency"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["fromCurrency", "toCurrency"],
            "Which source currency and target currency would you like to use?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Convert some money", params_schema=schema
            )

        assert "fromCurrency" in result["missing_required"]
        assert "toCurrency" in result["missing_required"]

    def test_optional_params_not_in_missing_required(self) -> None:
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("language", "string", False, "Response language (optional)"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"city": "Tartu"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Weather in Tartu", params_schema=schema
            )

        assert result["missing_required"] == []
        assert "language" not in result["missing_required"]

    def test_missing_required_empty_when_all_satisfied(self) -> None:
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"city": "Pärnu"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Pärnu", params_schema=schema)

        assert result["missing_required"] == []

    def test_already_collected_fulfils_missing_required(self) -> None:
        """A param in already_collected should not appear in missing_required."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"date": "2026-04-03"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="April 3rd",
                params_schema=schema,
                already_collected={"city": "Narva"},
            )

        assert "city" not in result["missing_required"]
        assert result["missing_required"] == []


class TestClarifyingQuestionGeneration:
    """forward() must return appropriate clarifying questions."""

    def test_returns_none_when_all_params_collected(self) -> None:
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"city": "Tallinn"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Tallinn", params_schema=schema)

        assert result["clarifying_question"] == "none"

    def test_returns_question_when_params_missing(self) -> None:
        """A question covering all missing params should be returned."""
        schema = _make_schema(
            ("countryCode", "string", True, "Country ISO code"),
            ("year", "integer", True, "Year"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["countryCode", "year"],
            "Which country and year would you like holidays for?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="What are the public holidays?", params_schema=schema
            )

        assert result["clarifying_question"] != "none"
        assert "?" in result["clarifying_question"]

    def test_question_covers_all_missing_params(self) -> None:
        """When multiple params are missing the question should address all of them."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date of interest"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["city", "date"],
            "Which city and date are you interested in?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Give me info", params_schema=schema)

        question = result["clarifying_question"]
        assert question != "none"
        # Both param descriptions should be represented in the question
        assert "city" in question.lower() or "date" in question.lower()

    def test_question_does_not_expose_raw_param_name(self) -> None:
        """The clarifying question should NOT contain the raw camelCase param name."""
        schema = _make_schema(
            (
                "countryIsoCode",
                "string",
                True,
                "Which country would you like to search in?",
            )
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["countryIsoCode"],
            "Which country would you like to search in?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Find holidays", params_schema=schema)

        assert "countryIsoCode" not in result["clarifying_question"]

    def test_clarifying_question_forced_to_none_when_no_missing(self) -> None:
        """Even if LLM returns a question, result is 'none' when nothing is missing."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        # LLM erroneously provides a question even though city was extracted
        mock_result = _make_mock_result(
            {"city": "Viljandi"},
            [],
            "Which city are you asking about?",  # wrong — should be overridden
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Weather in Viljandi", params_schema=schema
            )

        assert result["clarifying_question"] == "none"

    def test_estonian_message_triggers_estonian_question(self) -> None:
        """Multilingual: LLM returns question in same language as user message."""
        schema = _make_schema(("city", "string", True, "Linn"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["city"],
            "Millist linna soovite?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Mis on ilm?", params_schema=schema)

        assert result["clarifying_question"] == "Millist linna soovite?"

    def test_russian_message_triggers_russian_question(self) -> None:
        """Multilingual: LLM returns question in Russian when user message is Russian."""
        schema = _make_schema(("city", "string", True, "Город"))
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {},
            ["city"],
            "Какой город вас интересует?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Какая погода?", params_schema=schema)

        assert result["clarifying_question"] == "Какой город вас интересует?"

    def test_second_turn_question_asks_only_for_remaining_params(self) -> None:
        """Multi-turn: after city is collected, second question asks only about date."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date of interest"),
        )
        module = ParamExtractionModule()

        # Turn 2: user provided city in turn 1; only date is still missing
        mock_result = _make_mock_result(
            {},
            ["date"],
            "Which date are you interested in?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="Tell me the weather",
                params_schema=schema,
                already_collected={"city": "Tallinn"},  # city resolved in turn 1
            )

        assert result["missing_required"] == ["date"]
        assert result["clarifying_question"] != "none"
        # city should not be asked for again
        assert "city" not in result["clarifying_question"].lower()

    def test_all_params_satisfied_on_second_turn_returns_none(self) -> None:
        """Multi-turn: once all params are collected the question becomes 'none'."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("date", "date", True, "Date of interest"),
        )
        module = ParamExtractionModule()

        # Turn 2: user just provided date; city was already collected
        mock_result = _make_mock_result({"date": "2026-04-03"}, [], "none")
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(
                user_message="April 3rd",
                params_schema=schema,
                already_collected={"city": "Tallinn"},
            )

        assert result["missing_required"] == []
        assert result["clarifying_question"] == "none"


class TestErrorHandling:
    """forward() should return safe defaults on LLM/JSON failures."""

    def test_json_parse_error_returns_safe_defaults(self) -> None:
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        bad_result = MagicMock()
        bad_result.extracted_params = "NOT VALID JSON {"
        bad_result.missing_required = '["city"]'
        bad_result.clarifying_question = "Which city?"

        with patch.object(module, "extractor", return_value=bad_result):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert result["extracted_params"] == {}
        assert "city" in result["missing_required"]

    def test_llm_exception_returns_safe_defaults(self) -> None:
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        with patch.object(module, "extractor", side_effect=RuntimeError("LLM down")):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert result["extracted_params"] == {}
        assert "city" in result["missing_required"]

    def test_safe_defaults_only_lists_required_params(self) -> None:
        """Safe defaults must not report optional params as missing."""
        schema = _make_schema(
            ("city", "string", True, "City name"),
            ("language", "string", False, "Language"),
        )
        module = ParamExtractionModule()

        with patch.object(module, "extractor", side_effect=RuntimeError("LLM down")):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert "city" in result["missing_required"]
        assert "language" not in result["missing_required"]

    def test_missing_required_output_is_not_dict(self) -> None:
        """If LLM returns a dict for missing_required (wrong type), fall back safely."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        bad_result = MagicMock()
        bad_result.extracted_params = "{}"
        bad_result.missing_required = '{"city": true}'  # wrong type — dict not list
        bad_result.clarifying_question = "Which city?"

        with patch.object(module, "extractor", return_value=bad_result):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert result["extracted_params"] == {}
        assert "city" in result["missing_required"]

    def test_extracted_params_not_dict_falls_back(self) -> None:
        """If LLM returns an array for extracted_params, fall back safely."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        bad_result = MagicMock()
        bad_result.extracted_params = '["Tallinn"]'  # wrong type — list not dict
        bad_result.missing_required = "[]"
        bad_result.clarifying_question = "none"

        with patch.object(module, "extractor", return_value=bad_result):
            result = module.forward(user_message="Tallinn", params_schema=schema)

        assert result["extracted_params"] == {}
        assert "city" in result["missing_required"]

    def test_safe_defaults_clarifying_question_empty_when_params_missing(self) -> None:
        """On LLM failure with missing required params, clarifying_question is '' (caller generates question)."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        with patch.object(module, "extractor", side_effect=RuntimeError("LLM down")):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert result["clarifying_question"] == ""

    def test_safe_defaults_clarifying_question_none_when_nothing_missing(self) -> None:
        """On LLM failure when all params already collected, clarifying_question is 'none'."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        with patch.object(module, "extractor", side_effect=RuntimeError("LLM down")):
            result = module.forward(
                user_message="Hello",
                params_schema=schema,
                already_collected={"city": "Tallinn"},
            )

        assert result["clarifying_question"] == "none"


class TestConversationHistoryFormatting:
    """_format_conversation_history() helper should format turns correctly."""

    def test_none_history_returns_no_history_placeholder(self) -> None:
        module = ParamExtractionModule()
        result = module._format_conversation_history(None)
        assert result == "(No conversation history)"

    def test_empty_history_returns_no_history_placeholder(self) -> None:
        module = ParamExtractionModule()
        result = module._format_conversation_history([])
        assert result == "(No conversation history)"

    def test_single_turn_formatted_correctly(self) -> None:
        module = ParamExtractionModule()
        history = [{"authorRole": "user", "message": "Hello"}]
        result = module._format_conversation_history(history)
        assert result == "user: Hello"

    def test_multiple_turns_formatted_as_lines(self) -> None:
        module = ParamExtractionModule()
        history = [
            {"authorRole": "user", "message": "Hi"},
            {"authorRole": "assistant", "message": "Hello!"},
        ]
        result = module._format_conversation_history(history)
        assert result == "user: Hi\nassistant: Hello!"

    def test_history_truncated_to_max_turns(self) -> None:
        """Only the last 5 turns should be included."""
        module = ParamExtractionModule()
        history = [
            {"authorRole": "user", "message": f"Message {i}"}
            for i in range(8)  # 8 turns — should be trimmed to last 5
        ]
        result = module._format_conversation_history(history)
        lines = result.splitlines()
        assert len(lines) == 5
        assert "Message 7" in result
        assert "Message 2" not in result  # first 3 should be dropped

    def test_turns_with_empty_message_are_skipped(self) -> None:
        module = ParamExtractionModule()
        history = [
            {"authorRole": "user", "message": "Hello"},
            {"authorRole": "assistant", "message": ""},
            {"authorRole": "user", "message": "Continue"},
        ]
        result = module._format_conversation_history(history)
        assert "assistant" not in result
        assert result == "user: Hello\nuser: Continue"


# ---------------------------------------------------------------------------
# Custom instructions
# ---------------------------------------------------------------------------


class TestCustomInstructions:
    """Custom instructions should be forwarded to the DSPy predictor unchanged."""

    def _make_schema(self) -> list[dict]:
        return _make_schema(("city", "string", True, "City name"))

    def test_custom_instructions_passed_to_predictor(self) -> None:
        """When custom_instructions is set on the module, it must be sent to the LLM."""
        schema = self._make_schema()
        module = ParamExtractionModule(custom_instructions="Use formal Estonian.")

        mock_result = _make_mock_result({"city": "Tallinn"}, [], "none")
        with patch.object(
            module, "extractor", return_value=mock_result
        ) as mock_extractor:
            module.forward(user_message="Tallinn", params_schema=schema)

        call_kwargs = mock_extractor.call_args.kwargs
        assert call_kwargs["custom_instructions"] == "Use formal Estonian."

    def test_empty_custom_instructions_passed_by_default(self) -> None:
        """When no custom_instructions provided, the predictor receives an empty string."""
        schema = self._make_schema()
        module = ParamExtractionModule()

        mock_result = _make_mock_result({"city": "Tallinn"}, [], "none")
        with patch.object(
            module, "extractor", return_value=mock_result
        ) as mock_extractor:
            module.forward(user_message="Tallinn", params_schema=schema)

        call_kwargs = mock_extractor.call_args.kwargs
        assert call_kwargs["custom_instructions"] == ""

    def test_custom_instructions_stored_on_instance(self) -> None:
        """The custom_instructions value passed to __init__ must be stored."""
        instructions = "Always reply in Russian."
        module = ParamExtractionModule(custom_instructions=instructions)
        assert module._custom_instructions == instructions

    def test_default_custom_instructions_is_empty_string(self) -> None:
        """Modules initialised without custom_instructions must default to empty string."""
        module = ParamExtractionModule()
        assert module._custom_instructions == ""


# ---------------------------------------------------------------------------
# strip_format_hints helper
# ---------------------------------------------------------------------------


class TestStripFormatHints:
    """strip_format_hints() should remove format hints from descriptions."""

    @pytest.mark.parametrize(
        "description, expected",
        [
            # Parenthesised YYYY-MM-DD pattern
            ("Start date (YYYY-MM-DD)", "Start date"),
            # Parenthesised ISO 8601
            ("End date (ISO 8601)", "End date"),
            # Parenthesised 2-letter code
            ("Country code (2-letter code)", "Country code"),
            # Parenthesised HH:MM:SS
            ("Time of day (HH:MM:SS)", "Time of day"),
            # Trailing "in the format X"
            ("Date in the format YYYY-MM-DD", "Date"),
            # Trailing "in format X"
            ("Date in format YYYY-MM-DD", "Date"),
            # No format hints — unchanged
            ("City name", "City name"),
            # Empty string — unchanged
            ("", ""),
            # Mixed: parenthesised hint plus trailing comma residue
            ("Start date (YYYY-MM-DD)", "Start date"),
        ],
    )
    def test_strips_known_patterns(self, description: str, expected: str) -> None:
        assert strip_format_hints(description) == expected

    def test_preserves_unrelated_parentheses(self) -> None:
        """Parentheses that don't contain format-like keywords are kept."""
        desc = "City name (required)"
        # "required" does not match the keyword list, so it should be preserved
        result = strip_format_hints(desc)
        assert "required" in result

    def test_idempotent(self) -> None:
        """Calling twice produces the same result."""
        desc = "Start date (YYYY-MM-DD)"
        assert strip_format_hints(strip_format_hints(desc)) == strip_format_hints(desc)


# ---------------------------------------------------------------------------
# Sanitized schema in forward()
# ---------------------------------------------------------------------------


class TestSanitizedSchemaInForward:
    """forward() must strip format hints from descriptions before passing to the LLM."""

    def test_format_hints_stripped_from_params_schema_json(self) -> None:
        """The JSON sent to the predictor must not contain YYYY-MM-DD in descriptions."""
        schema = _make_schema(
            ("startDate", "date", True, "Start date (YYYY-MM-DD)"),
            ("country", "string", True, "Country ISO code (2-letter code)"),
        )
        module = ParamExtractionModule()

        mock_result = _make_mock_result(
            {}, ["startDate", "country"], "Which dates and country?"
        )
        with patch.object(
            module, "extractor", return_value=mock_result
        ) as mock_extractor:
            module.forward(user_message="Hello", params_schema=schema)

        call_kwargs = mock_extractor.call_args.kwargs
        params_json = call_kwargs["params_schema"]
        assert "YYYY-MM-DD" not in params_json
        assert "2-letter code" not in params_json
        # Parameter names must still be present
        assert "startDate" in params_json
        assert "country" in params_json

    def test_clarifying_question_does_not_contain_format_hints(self) -> None:
        """End-to-end: even if LLM echoes a hint, result is whatever the LLM returns —
        but the schema fed to the LLM no longer contains the hint text."""
        schema = _make_schema(
            ("startDate", "date", True, "Start date (YYYY-MM-DD)"),
        )
        module = ParamExtractionModule()

        # LLM returns a clean question (as expected when no hint is in the schema)
        mock_result = _make_mock_result(
            {},
            ["startDate"],
            "For which date would you like the information?",
        )
        with patch.object(module, "extractor", return_value=mock_result):
            result = module.forward(user_message="Hello", params_schema=schema)

        assert "YYYY-MM-DD" not in result["clarifying_question"]


# ---------------------------------------------------------------------------
# stream_forward
# ---------------------------------------------------------------------------


class TestStreamForward:
    """stream_forward() should mirror forward() via the streaming predictor."""

    def _make_stream_predictor(
        self,
        stream_tokens: list,
        prediction: dspy.Prediction,
        capture_kwargs: dict | None = None,
    ):
        """Return a callable that yields StreamResponse tokens then a Prediction."""

        async def _stream(**kwargs):
            if capture_kwargs is not None:
                capture_kwargs.update(kwargs)
            for token in stream_tokens:
                yield dspy.streaming.StreamResponse(
                    chunk=token,
                    signature_field_name="clarifying_question",
                    predict_name="extractor",
                    is_last_chunk=False,
                )
            yield prediction

        return _stream

    def _make_prediction(
        self,
        extracted_params: dict,
        missing_required: list,
        clarifying_question: str,
    ) -> dspy.Prediction:
        return dspy.Prediction(
            extracted_params=json.dumps(extracted_params),
            missing_required=json.dumps(missing_required),
            clarifying_question=clarifying_question,
        )

    @pytest.mark.asyncio
    async def test_stream_forward_returns_result_when_all_params_satisfied(
        self,
    ) -> None:
        """stream_forward() returns parsed result and empty tokens when no question needed."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        prediction = self._make_prediction({"city": "Tallinn"}, [], "none")
        stream_predictor = self._make_stream_predictor([], prediction)

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            tokens, result = await module.stream_forward(
                user_message="Tallinn", params_schema=schema
            )

        assert tokens == []
        assert result["extracted_params"] == {"city": "Tallinn"}
        assert result["missing_required"] == []
        assert result["clarifying_question"] == "none"

    @pytest.mark.asyncio
    async def test_stream_forward_collects_tokens_for_clarifying_question(self) -> None:
        """Token strings from StreamResponse chunks are collected and returned."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        prediction = self._make_prediction({}, ["city"], "Which city?")
        stream_predictor = self._make_stream_predictor(
            ["Which", " city", "?"], prediction
        )

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            tokens, result = await module.stream_forward(
                user_message="Give me the weather", params_schema=schema
            )

        assert tokens == ["Which", " city", "?"]
        assert result["missing_required"] == ["city"]
        assert result["clarifying_question"] != "none"

    @pytest.mark.asyncio
    async def test_stream_forward_tokens_cleared_when_question_is_none(self) -> None:
        """Token list is empty even if tokens streamed when the final question is 'none'."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        prediction = self._make_prediction({"city": "Pärnu"}, [], "none")
        stream_predictor = self._make_stream_predictor(
            ["Which", " city", "?"], prediction
        )

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            tokens, result = await module.stream_forward(
                user_message="Pärnu", params_schema=schema
            )

        assert tokens == []
        assert result["clarifying_question"] == "none"

    @pytest.mark.asyncio
    async def test_stream_forward_passes_custom_instructions_to_predictor(self) -> None:
        """custom_instructions on the module must reach the streaming predictor call."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule(custom_instructions="Always reply in Estonian.")

        prediction = self._make_prediction({"city": "Tartu"}, [], "none")
        captured_kwargs: dict = {}
        stream_predictor = self._make_stream_predictor([], prediction, captured_kwargs)

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            await module.stream_forward(user_message="Tartu", params_schema=schema)

        assert captured_kwargs.get("custom_instructions") == "Always reply in Estonian."

    @pytest.mark.asyncio
    async def test_stream_forward_empty_custom_instructions_by_default(self) -> None:
        """When no custom_instructions set, empty string reaches the streaming predictor."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        prediction = self._make_prediction({"city": "Narva"}, [], "none")
        captured_kwargs: dict = {}
        stream_predictor = self._make_stream_predictor([], prediction, captured_kwargs)

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            await module.stream_forward(user_message="Narva", params_schema=schema)

        assert captured_kwargs.get("custom_instructions") == ""

    @pytest.mark.asyncio
    async def test_stream_forward_fallback_when_no_prediction_received(self) -> None:
        """If the stream ends with no Prediction object, forward() fallback is used."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        async def _stream_no_prediction(**kwargs):
            yield dspy.streaming.StreamResponse(
                chunk="Which city?",
                signature_field_name="clarifying_question",
                predict_name="extractor",
                is_last_chunk=True,
            )

        fallback_result = {
            "extracted_params": {},
            "missing_required": ["city"],
            "clarifying_question": "Which city?",
        }

        with patch.object(
            module, "_get_stream_predictor", return_value=_stream_no_prediction
        ):
            with patch.object(module, "forward", return_value=fallback_result):
                tokens, result = await module.stream_forward(
                    user_message="Hello", params_schema=schema
                )

        assert result["missing_required"] == ["city"]
        assert tokens == ["Which city?"]

    @pytest.mark.asyncio
    async def test_stream_forward_exception_returns_safe_defaults(self) -> None:
        """An exception raised by the stream predictor returns safe defaults."""
        schema = _make_schema(("city", "string", True, "City name"))
        module = ParamExtractionModule()

        def _raise(**kwargs):
            raise RuntimeError("Stream failure")

        with patch.object(module, "_get_stream_predictor", return_value=_raise):
            tokens, result = await module.stream_forward(
                user_message="Hello", params_schema=schema
            )

        assert tokens == []
        assert result["extracted_params"] == {}
        assert "city" in result["missing_required"]

    @pytest.mark.asyncio
    async def test_stream_forward_session_language_forwarded(self) -> None:
        """session_language passed to stream_forward must reach the predictor."""
        schema = _make_schema(("city", "string", True, "Linn"))
        module = ParamExtractionModule()

        prediction = self._make_prediction({}, ["city"], "Millist linna soovite?")
        captured_kwargs: dict = {}
        stream_predictor = self._make_stream_predictor(
            ["Millist linna soovite?"], prediction, captured_kwargs
        )

        with patch.object(
            module, "_get_stream_predictor", return_value=stream_predictor
        ):
            tokens, result = await module.stream_forward(
                user_message="Mis on ilm?",
                params_schema=schema,
                session_language="et",
            )

        assert captured_kwargs.get("session_language") == "et"
        assert tokens == ["Millist linna soovite?"]

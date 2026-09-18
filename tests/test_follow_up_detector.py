"""Unit tests for FollowUpDetectorModule — DSPy follow-up query classification."""

import json
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import dspy
import pytest

from tool_classifier.follow_up_detector import (
    FollowUpDetectorModule,
    _validate_updated_params,
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
    follow_up_type: str,
    updated_params: dict,
) -> MagicMock:
    """Build a mock DSPy Predict result with the expected output attributes."""
    mock_result = MagicMock()
    mock_result.follow_up_type = follow_up_type
    mock_result.updated_params = json.dumps(updated_params, ensure_ascii=False)
    return mock_result


_SAMPLE_SCHEMA = [
    {
        "name": "year",
        "type": "integer",
        "required": True,
        "description": "Year for the query",
    },
    {
        "name": "country",
        "type": "string",
        "required": True,
        "description": "Country ISO code",
    },
]

_SAMPLE_PARAMS = {"year": 2026, "country": "EE"}


class TestFollowUpDetectorModuleInit:
    """FollowUpDetectorModule should initialise correctly."""

    def test_module_has_detector_attribute(self) -> None:
        module = FollowUpDetectorModule()
        assert hasattr(module, "detector")

    def test_detector_is_dspy_predict(self) -> None:
        module = FollowUpDetectorModule()
        assert isinstance(module.detector, dspy.Predict)


class TestFollowUpDetectorForward:
    """forward() should classify follow-up queries correctly."""

    def test_response_question_classification(self) -> None:
        """Query asking about returned data should be classified as response_question."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result("response_question", {})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="Which of those holidays falls on a Monday?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "response_question"
        assert result["updated_params"] == {}

    def test_param_update_classification(self) -> None:
        """Query refining parameters should be classified as param_update with updated_params."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result("param_update", {"year": 2025})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="What about 2025 instead?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "param_update"
        assert result["updated_params"] == {"year": 2025}

    def test_new_intent_classification(self) -> None:
        """Unrelated query should be classified as new_intent with empty updated_params."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result("new_intent", {})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="How do I apply for a driving licence?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "new_intent"
        assert result["updated_params"] == {}

    def test_invalid_follow_up_type_defaults_to_new_intent(self) -> None:
        """An unrecognised follow_up_type value should fall back to new_intent."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result("unknown_value", {})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="Some query",
                previous_query="Previous query",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "new_intent"
        assert result["updated_params"] == {}

    def test_json_parse_error_defaults_to_empty_params(self) -> None:
        """Malformed JSON in updated_params for param_update should set params to {} but keep type."""
        module = FollowUpDetectorModule()
        mock_result = MagicMock()
        mock_result.follow_up_type = "param_update"
        mock_result.updated_params = "not valid json {"

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="What about 2025?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        # Follow-up type is still recognized as param_update, but params default to {}
        assert result["follow_up_type"] == "param_update"
        assert result["updated_params"] == {}

    def test_exception_defaults_to_new_intent(self) -> None:
        """A predictor exception should return the safe fallback without raising."""
        module = FollowUpDetectorModule()

        with patch.object(module, "detector", side_effect=RuntimeError("LLM failure")):
            result = module.forward(
                user_query="Some query",
                previous_query="Previous query",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "new_intent"
        assert result["updated_params"] == {}

    def test_invalid_updated_params_defaults_to_empty_dict(self) -> None:
        """Non-dict updated_params (e.g. a JSON list) should be replaced with {}."""
        module = FollowUpDetectorModule()
        mock_result = MagicMock()
        mock_result.follow_up_type = "param_update"
        mock_result.updated_params = json.dumps(["year", 2025])  # list, not dict

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="What about 2025?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "param_update"
        assert result["updated_params"] == {}


class TestFollowUpDetectorOutputRulesEnforcement:
    """OUTPUT RULES normalization: updated_params must be {} unless follow_up_type is param_update."""

    def test_normalize_updated_params_to_empty_for_response_question(self) -> None:
        """Verify that updated_params is forced to {} for response_question even if LLM returns non-empty.

        Per OUTPUT RULES: "For response_question or new_intent: return updated_params as empty {}"
        This prevents unintended params from leaking downstream.
        """
        module = FollowUpDetectorModule()
        # LLM returns non-empty updated_params, but we should normalize to {}
        mock_result = _make_mock_result("response_question", {"year": 2025})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="Which of those falls on a Monday?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "response_question"
        assert result["updated_params"] == {}, (
            "updated_params must be {} for response_question per OUTPUT RULES"
        )

    def test_normalize_updated_params_to_empty_for_new_intent(self) -> None:
        """Verify that updated_params is forced to {} for new_intent even if LLM returns non-empty.

        Per OUTPUT RULES: "For response_question or new_intent: return updated_params as empty {}"
        This prevents unintended params from leaking downstream.
        """
        module = FollowUpDetectorModule()
        # LLM returns non-empty updated_params, but we should normalize to {}
        mock_result = _make_mock_result("new_intent", {"country": "US"})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="How do I apply for a driving licence?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "new_intent"
        assert result["updated_params"] == {}, (
            "updated_params must be {} for new_intent per OUTPUT RULES"
        )

    def test_preserve_updated_params_only_for_param_update(self) -> None:
        """Verify that updated_params is only preserved when follow_up_type is param_update."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result("param_update", {"year": 2025, "country": "US"})

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="What about 2025 and US?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "param_update"
        assert result["updated_params"] == {"year": 2025, "country": "US"}, (
            "updated_params should be preserved for param_update"
        )


class TestValidateUpdatedParams:
    """_validate_updated_params should filter and validate against the schema."""

    def test_keeps_valid_params_in_schema(self) -> None:
        """Valid parameters that are in the schema should be kept."""
        updated_params = {"year": 2025, "country": "US"}
        result = _validate_updated_params(updated_params, _SAMPLE_SCHEMA)
        assert result == {"year": 2025, "country": "US"}

    def test_drops_unknown_params_not_in_schema(self) -> None:
        """Parameters not in the schema should be dropped to prevent injection."""
        updated_params = {"year": 2025, "country": "US", "malicious_param": "value"}
        result = _validate_updated_params(updated_params, _SAMPLE_SCHEMA)
        assert result == {"year": 2025, "country": "US"}
        assert "malicious_param" not in result

    def test_drops_multiple_unknown_params(self) -> None:
        """Multiple injected parameters should all be dropped."""
        updated_params = {
            "year": 2025,
            "injected1": "danger1",
            "injected2": "danger2",
        }
        result = _validate_updated_params(updated_params, _SAMPLE_SCHEMA)
        assert result == {"year": 2025}
        assert len(result) == 1

    def test_empty_params_returns_empty_dict(self) -> None:
        """Empty updated_params should return empty dict."""
        result = _validate_updated_params({}, _SAMPLE_SCHEMA)
        assert result == {}

    def test_all_params_unknown_returns_empty_dict(self) -> None:
        """If all parameters are unknown, result should be empty dict."""
        updated_params = {"unknown1": "val1", "unknown2": "val2"}
        result = _validate_updated_params(updated_params, _SAMPLE_SCHEMA)
        assert result == {}

    def test_partial_unknown_mixed_with_valid(self) -> None:
        """Mix of valid and unknown params should keep only valid ones."""
        updated_params = {
            "year": 2025,
            "country": "EE",
            "extra_field": "injected",
            "another_injection": 123,
        }
        result = _validate_updated_params(updated_params, _SAMPLE_SCHEMA)
        assert result == {"year": 2025, "country": "EE"}

    def test_schema_with_null_entries_handles_gracefully(self) -> None:
        """Schema entries without 'name' key should be skipped without errors."""
        schema_with_invalid = [
            {"name": "year", "type": "integer", "required": True},
            {"type": "string"},  # Missing 'name' key
            {"name": "country", "type": "string", "required": True},
        ]
        updated_params = {"year": 2025, "country": "US"}
        result = _validate_updated_params(updated_params, schema_with_invalid)
        assert result == {"year": 2025, "country": "US"}

    def test_validation_inside_param_update_flow(self) -> None:
        """Integration: param_update with injected params should be filtered."""
        module = FollowUpDetectorModule()
        mock_result = _make_mock_result(
            "param_update",
            {"year": 2025, "country": "US", "injected_param": "danger"},
        )

        with patch.object(module, "detector", return_value=mock_result):
            result = module.forward(
                user_query="What about 2025 and US?",
                previous_query="Show public holidays in Estonia for 2026",
                previous_params=_SAMPLE_PARAMS,
                params_schema=_SAMPLE_SCHEMA,
            )

        assert result["follow_up_type"] == "param_update"
        # Injected param should be dropped
        assert result["updated_params"] == {"year": 2025, "country": "US"}
        assert "injected_param" not in result["updated_params"]

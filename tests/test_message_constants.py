"""Guards for ticket #158 - message catalogue completeness and consistency.

Every user-facing message the system produces itself (as opposed to LLM output)
lives in ``llm_ochestrator_constants.py`` as a dict keyed by language code, and is
read through ``get_localized_message``.

The most important test here is ``test_every_message_dict_is_excluded_from_history``:
a message dict that is not registered in ``_HISTORY_EXCLUDED_MESSAGES`` gets its
failed exchange persisted to Redis and replayed to the prompt refiner on the
user's next turn, degrading the following answer.
"""

from typing import Dict

import pytest

from src.llm_orchestrator_config.llm_ochestrator_constants import (
    ENTITY_EXTRACTION_FAILED_MESSAGES,
    INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    INSUFFICIENT_CONTEXT_MESSAGES,
    NO_CONTEXT_AVAILABLE_MESSAGES,
    OUT_OF_SCOPE_MESSAGES,
    OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
    QUERY_VALIDATION_FAILED_MESSAGES,
    SERVICE_EXECUTION_ERROR_MESSAGES,
    SERVICE_NOT_FOUND_MESSAGES,
    SERVICE_TIMEOUT_ERROR_MESSAGES,
    SERVICE_VALIDATION_FAILED_MESSAGES,
    TECHNICAL_ISSUE_MESSAGES,
    get_localized_message,
)

# Decision D1 (see docs/TICKET_158_IMPLEMENTATION_PLAN.md): the four pre-existing
# OOD dictionaries already carried "ru", so new messages match rather than regress.
REQUIRED_LANGUAGES = frozenset({"et", "en", "ru"})

ALL_MESSAGE_DICTS: Dict[str, Dict[str, str]] = {
    # Pre-existing
    "OUT_OF_SCOPE_MESSAGES": OUT_OF_SCOPE_MESSAGES,
    "TECHNICAL_ISSUE_MESSAGES": TECHNICAL_ISSUE_MESSAGES,
    "INPUT_GUARDRAIL_VIOLATION_MESSAGES": INPUT_GUARDRAIL_VIOLATION_MESSAGES,
    "OUTPUT_GUARDRAIL_VIOLATION_MESSAGES": OUTPUT_GUARDRAIL_VIOLATION_MESSAGES,
    "QUERY_VALIDATION_FAILED_MESSAGES": QUERY_VALIDATION_FAILED_MESSAGES,
    # Service workflow (Layer 1) - added by #158
    "SERVICE_NOT_FOUND_MESSAGES": SERVICE_NOT_FOUND_MESSAGES,
    "SERVICE_VALIDATION_FAILED_MESSAGES": SERVICE_VALIDATION_FAILED_MESSAGES,
    "SERVICE_TIMEOUT_ERROR_MESSAGES": SERVICE_TIMEOUT_ERROR_MESSAGES,
    "SERVICE_EXECUTION_ERROR_MESSAGES": SERVICE_EXECUTION_ERROR_MESSAGES,
    "ENTITY_EXTRACTION_FAILED_MESSAGES": ENTITY_EXTRACTION_FAILED_MESSAGES,
    # Context workflow (Layer 3) - added by #158
    "INSUFFICIENT_CONTEXT_MESSAGES": INSUFFICIENT_CONTEXT_MESSAGES,
    "NO_CONTEXT_AVAILABLE_MESSAGES": NO_CONTEXT_AVAILABLE_MESSAGES,
}


class TestMessageCatalogue:
    """Every message dict is complete, non-empty and correctly localized."""

    @pytest.mark.parametrize("name,messages", sorted(ALL_MESSAGE_DICTS.items()))
    def test_all_required_languages_present(
        self, name: str, messages: Dict[str, str]
    ) -> None:
        missing = REQUIRED_LANGUAGES - set(messages)
        assert not missing, f"{name} is missing translations for: {sorted(missing)}"

    @pytest.mark.parametrize("name,messages", sorted(ALL_MESSAGE_DICTS.items()))
    def test_no_empty_messages(self, name: str, messages: Dict[str, str]) -> None:
        for lang, text in messages.items():
            assert text and text.strip(), f"{name}[{lang}] is empty"

    @pytest.mark.parametrize("name,messages", sorted(ALL_MESSAGE_DICTS.items()))
    def test_lookup_returns_correct_language(
        self, name: str, messages: Dict[str, str]
    ) -> None:
        for lang in REQUIRED_LANGUAGES:
            assert get_localized_message(messages, lang) == messages[lang], (
                f"{name}: lookup for '{lang}' did not return the '{lang}' text"
            )

    def test_unknown_language_falls_back_to_estonian(self) -> None:
        assert (
            get_localized_message(OUT_OF_SCOPE_MESSAGES, "fr")
            == OUT_OF_SCOPE_MESSAGES["et"]
        )


class TestHistoryExclusion:
    """Error/refusal messages must never reach conversation history."""

    def test_every_message_dict_is_excluded_from_history(self) -> None:
        """A message dict not registered here pollutes the prompt refiner.

        _HISTORY_EXCLUDED_MESSAGES matches by exact string equality, so adding a
        new message dictionary without registering it means that failed exchange
        is written to Redis and replayed on the user's next turn.
        """
        from src.llm_orchestration_service import _HISTORY_EXCLUDED_MESSAGES

        unregistered = [
            f"{name}[{lang}]"
            for name, messages in ALL_MESSAGE_DICTS.items()
            for lang, text in messages.items()
            if text not in _HISTORY_EXCLUDED_MESSAGES
        ]

        assert not unregistered, (
            "These messages are not in _HISTORY_EXCLUDED_MESSAGES "
            "(llm_orchestration_service.py) and would be persisted to Redis, "
            f"polluting the prompt refiner on the next turn: {unregistered}"
        )


class TestGreetings:
    """Greetings stay in greeting_constants.py (decision D2) but match D1."""

    def test_all_languages_present(self) -> None:
        from src.tool_classifier.greeting_constants import GREETINGS_BY_LANGUAGE

        missing = REQUIRED_LANGUAGES - set(GREETINGS_BY_LANGUAGE)
        assert not missing, f"GREETINGS_BY_LANGUAGE is missing: {sorted(missing)}"

    def test_all_greeting_types_present_in_every_language(self) -> None:
        from src.tool_classifier.greeting_constants import GREETINGS_BY_LANGUAGE

        required_types = {"hello", "goodbye", "thanks", "casual"}
        for lang, greetings in GREETINGS_BY_LANGUAGE.items():
            missing = required_types - set(greetings)
            assert not missing, f"GREETINGS[{lang}] is missing types: {sorted(missing)}"
            for gtype, text in greetings.items():
                assert text and text.strip(), f"GREETINGS[{lang}][{gtype}] is empty"

    def test_lookup_returns_requested_language(self) -> None:
        from src.tool_classifier.greeting_constants import (
            GREETINGS_BY_LANGUAGE,
            get_greeting_response,
        )

        for lang in REQUIRED_LANGUAGES:
            assert (
                get_greeting_response("hello", lang)
                == GREETINGS_BY_LANGUAGE[lang]["hello"]
            )

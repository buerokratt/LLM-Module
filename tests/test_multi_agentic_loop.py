"""Unit tests for MultiEndpointAgenticLoop."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from models.session_models import EndpointSessionState
from tool_classifier.enums import AgenticLoopStatus
from tool_classifier.multi_agentic_loop import MultiEndpointAgenticLoop
from tool_classifier.param_extractor import ParamExtractionResult


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

_CHAT_ID = "test-multi-chat-1"

_HISTORY: List[Dict[str, Any]] = [
    {
        "authorRole": "user",
        "message": "Show me the weather and public holidays for Estonia",
    },
    {"authorRole": "bot", "message": "I can help with both!"},
]

# Endpoint A — get_public_holidays: languageIsoCode (required, shared with B),
#              countryIsoCode + validFrom + validTo (required, unique to A)
_ENDPOINT_A: Dict[str, Any] = {
    "name": "get_public_holidays",
    "url": "https://openholidaysapi.org/PublicHolidays",
    "params": [
        {
            "name": "languageIsoCode",
            "type": "string",
            "required": True,
            "description": "Response language code (e.g. ET, EN)",
        },
        {
            "name": "countryIsoCode",
            "type": "string",
            "required": True,
            "description": "Two-letter country ISO code (e.g. EE, LV)",
        },
        {
            "name": "validFrom",
            "type": "date",
            "required": True,
            "description": "Start date (YYYY-MM-DD)",
        },
        {
            "name": "validTo",
            "type": "date",
            "required": True,
            "description": "End date (YYYY-MM-DD)",
        },
    ],
}

# Endpoint B — get_current_weather: languageIsoCode (required, shared with A),
#              station (required, unique to B)
_ENDPOINT_B: Dict[str, Any] = {
    "name": "get_current_weather",
    "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
    "params": [
        {
            "name": "languageIsoCode",
            "type": "string",
            "required": True,
            "description": "Response language code",
        },
        {
            "name": "station",
            "type": "string",
            "required": True,
            "description": "Weather station identifier",
        },
    ],
}

# Endpoint C — get_current_weather with no required params
_ENDPOINT_C: Dict[str, Any] = {
    "name": "get_current_weather",
    "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
    "params": [
        {
            "name": "station",
            "type": "string",
            "required": False,
            "description": "Weather station identifier (optional)",
        },
    ],
}

# Endpoint D — get_current_weather with single required param (station)
_ENDPOINT_D: Dict[str, Any] = {
    "name": "get_current_weather",
    "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
    "params": [
        {
            "name": "station",
            "type": "string",
            "required": True,
            "description": "Weather station identifier",
        },
    ],
}


def _make_state(
    endpoint: Dict[str, Any],
    collected: Dict[str, Any] | None = None,
    completed: bool = False,
) -> EndpointSessionState:
    return EndpointSessionState(
        endpoint=endpoint,
        collected_params=collected or {},
        completed=completed,
    )


def _make_session_store_mock() -> AsyncMock:
    mock = AsyncMock()
    mock.update = AsyncMock(return_value=None)
    return mock


def _make_extractor_mock(result: ParamExtractionResult) -> MagicMock:
    mock = MagicMock(return_value=result)
    return mock


def _make_loop(
    extractor_mock: MagicMock,
    session_store_mock: AsyncMock | None = None,
) -> MultiEndpointAgenticLoop:
    return MultiEndpointAgenticLoop(
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
# Schema merging & deduplication
# ---------------------------------------------------------------------------


class TestSchemaMerging:
    def test_shared_param_namespaced_when_both_endpoints_incomplete(self) -> None:
        """'languageIsoCode' is shared by A and B (both incomplete) — it must be
        namespaced to 'languageIsoCode__0' and 'languageIsoCode__1', not merged."""
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, param_owners, namespace_map = loop._build_merged_schema(states)

        names = [p["name"] for p in merged_schema]
        # Original name must not appear (it's been namespaced)
        assert "languageIsoCode" not in names
        # Both namespaced entries must appear
        assert "languageIsoCode__0" in names
        assert "languageIsoCode__1" in names

    def test_param_owners_for_namespaced_params(self) -> None:
        """Each namespaced entry must be owned by exactly its corresponding endpoint."""
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, param_owners, namespace_map = loop._build_merged_schema(states)

        assert param_owners["languageIsoCode__0"] == [0]
        assert param_owners["languageIsoCode__1"] == [1]

    def test_namespace_map_populated_for_conflicting_params(self) -> None:
        """namespace_map must map namespaced keys to (ep_idx, original_name)."""
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, _, namespace_map = loop._build_merged_schema(states)

        assert namespace_map["languageIsoCode__0"] == (0, "languageIsoCode")
        assert namespace_map["languageIsoCode__1"] == (1, "languageIsoCode")

    def test_unique_params_included(self) -> None:
        """'countryIsoCode' (unique to A) and 'station' (unique to B) must appear."""
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, _, _ = loop._build_merged_schema(states)

        names = [p["name"] for p in merged_schema]
        assert "countryIsoCode" in names
        assert "station" in names

    def test_completed_endpoint_skipped_in_schema(self) -> None:
        """Params from a completed endpoint should NOT appear in merged schema."""
        states = [
            _make_state(
                _ENDPOINT_A,
                collected={
                    "languageIsoCode": "ET",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                },
                completed=True,
            ),
            _make_state(_ENDPOINT_D),
        ]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, _, _ = loop._build_merged_schema(states)

        names = [p["name"] for p in merged_schema]
        assert "countryIsoCode" not in names
        assert "languageIsoCode" not in names
        assert "station" in names

    def test_namespace_map_empty_when_only_one_endpoint_incomplete(self) -> None:
        """When only one incomplete endpoint exists, no conflicts arise and
        namespace_map must be empty."""
        states = [
            _make_state(
                _ENDPOINT_A,
                collected={
                    "languageIsoCode": "ET",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                },
                completed=True,
            ),
            _make_state(_ENDPOINT_D),
        ]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, _, namespace_map = loop._build_merged_schema(states)

        assert namespace_map == {}

    def test_required_promoted_if_any_owner_requires_it(self) -> None:
        """For a shared *non-conflicting* param (one completed + one incomplete
        endpoint), Pass 2 must promote it to required=True when the completed
        endpoint marks it required while the incomplete endpoint marks it optional."""
        incomplete_ep = {
            "name": "get_current_weather",
            "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
            "params": [
                {
                    "name": "languageIsoCode",
                    "type": "string",
                    "required": False,
                    "description": "Language code (optional)",
                },
                {
                    "name": "station",
                    "type": "string",
                    "required": True,
                    "description": "Weather station identifier",
                },
            ],
        }
        states = [
            _make_state(incomplete_ep),
            _make_state(
                _ENDPOINT_A,
                collected={
                    "languageIsoCode": "ET",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                },
                completed=True,
            ),
        ]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, _, namespace_map = loop._build_merged_schema(states)

        # Only incomplete_ep is counted in Pass 0 — no conflict, namespace_map empty.
        assert namespace_map == {}
        lang_param = next(p for p in merged_schema if p["name"] == "languageIsoCode")
        # Pass 2 promotes to required because completed _ENDPOINT_A marks it required.
        assert lang_param["required"] is True

    def test_required_promoted_by_completed_endpoint_owner(self) -> None:
        """Pass 2 must promote a shared param to required=True when the sole
        incomplete endpoint marks it optional but a completed endpoint marks it
        required — "any owner" semantics apply across both passes."""
        incomplete_ep = {
            "name": "get_current_weather",
            "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
            "params": [
                {
                    "name": "languageIsoCode",
                    "type": "string",
                    "required": False,
                    "description": "Language code (optional for weather)",
                },
                {
                    "name": "station",
                    "type": "string",
                    "required": True,
                    "description": "Weather station identifier",
                },
            ],
        }
        completed_ep = {
            "name": "get_public_holidays",
            "url": "https://openholidaysapi.org/PublicHolidays",
            "params": [
                {
                    "name": "languageIsoCode",
                    "type": "string",
                    "required": True,
                    "description": "Response language code (required for holidays)",
                },
            ],
        }
        states = [
            _make_state(incomplete_ep),
            _make_state(
                completed_ep, collected={"languageIsoCode": "ET"}, completed=True
            ),
        ]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, _, _ = loop._build_merged_schema(states)

        lang_param = next(p for p in merged_schema if p["name"] == "languageIsoCode")
        assert lang_param["required"] is True

    def test_type_conflict_for_conflicting_params_each_preserves_own_type(self) -> None:
        """When two incomplete endpoints share a param name with different types,
        both get their own namespaced entry with their own type (no merging)."""
        endpoint_date_type = {
            "name": "get_something",
            "url": "https://example.com/api",
            "params": [
                {
                    "name": "languageIsoCode",
                    "type": "date",  # different type from _ENDPOINT_A
                    "required": True,
                    "description": "Some date field",
                },
            ],
        }
        states = [_make_state(_ENDPOINT_A), _make_state(endpoint_date_type)]
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, _, namespace_map = loop._build_merged_schema(states)

        # Both endpoints conflict — both get namespaced entries.
        assert "languageIsoCode__0" in namespace_map
        assert "languageIsoCode__1" in namespace_map
        schema_by_name = {p["name"]: p for p in merged_schema}
        assert (
            schema_by_name["languageIsoCode__0"]["type"] == "string"
        )  # from _ENDPOINT_A
        assert (
            schema_by_name["languageIsoCode__1"]["type"] == "date"
        )  # from endpoint_date_type


# ---------------------------------------------------------------------------
# Param distribution
# ---------------------------------------------------------------------------


class TestParamDistribution:
    def test_namespaced_param_written_to_correct_endpoint(self) -> None:
        """Namespaced key 'languageIsoCode__0' must be reverse-translated and
        written to endpoint 0's collected_params under the original name."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]

        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, param_owners, namespace_map = loop._build_merged_schema(states)
        loop._distribute_params(
            {"languageIsoCode__0": "ET", "languageIsoCode__1": "EN"},
            states,
            param_owners,
            namespace_map,
        )

        assert state_a.collected_params["languageIsoCode"] == "ET"
        assert state_b.collected_params["languageIsoCode"] == "EN"

    def test_unique_param_written_only_to_owner(self) -> None:
        """Extracted 'countryIsoCode' must be written only to endpoint A, not B."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]

        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, param_owners, namespace_map = loop._build_merged_schema(states)
        loop._distribute_params(
            {"countryIsoCode": "EE"}, states, param_owners, namespace_map
        )

        assert state_a.collected_params.get("countryIsoCode") == "EE"
        assert "countryIsoCode" not in state_b.collected_params

    def test_endpoint_marked_completed_when_all_required_present(self) -> None:
        """Endpoint A must be marked completed once all required params are set."""
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        # A and B share languageIsoCode (conflicting); use A+D instead so there
        # are no shared params and no namespace conflict.
        state_a2 = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
        )
        state_d = _make_state(_ENDPOINT_D)
        states2 = [state_a2, state_d]
        _, param_owners2, namespace_map2 = loop._build_merged_schema(states2)
        # namespace_map2 is empty (no conflicts between A and D)
        assert namespace_map2 == {}
        loop._distribute_params(
            {"countryIsoCode": "EE"}, states2, param_owners2, namespace_map2
        )

        assert state_a2.completed is True
        assert state_d.completed is False

    def test_endpoint_with_no_required_params_immediately_completed(self) -> None:
        """Endpoint C has no required params — distribute with empty dict should
        mark it completed."""
        state_c = _make_state(_ENDPOINT_C)
        states = [state_c]

        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, param_owners, namespace_map = loop._build_merged_schema(states)
        loop._distribute_params({}, states, param_owners, namespace_map)

        assert state_c.completed is True

    def test_completed_endpoint_not_overwritten_when_value_unchanged(self) -> None:
        """A completed endpoint's param must NOT be touched when the extracted
        value is identical to the already-stored value."""
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
            completed=True,
        )
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]

        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        # A is completed — Pass 0 only sees B's params, so no conflict.
        _, param_owners, namespace_map = loop._build_merged_schema(states)
        assert namespace_map == {}
        original_collected = dict(state_a.collected_params)
        # languageIsoCode is in param_owners (non-namespaced) — same value → no-op.
        loop._distribute_params(
            {"languageIsoCode": "ET"}, states, param_owners, namespace_map
        )

        assert state_a.collected_params == original_collected
        assert state_a.completed is True

    def test_completed_endpoint_overwritten_when_value_differs(self) -> None:
        """A completed endpoint's param IS overwritten when the extracted value
        differs — the intentional shared-param correction path."""
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
            completed=True,
        )
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]

        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        # A is completed — no conflict.
        _, param_owners, namespace_map = loop._build_merged_schema(states)
        assert namespace_map == {}
        loop._distribute_params(
            {"languageIsoCode": "EN"}, states, param_owners, namespace_map
        )

        assert state_a.collected_params["languageIsoCode"] == "EN"
        # Still completed — the flag is not rolled back
        assert state_a.completed is True


# ---------------------------------------------------------------------------
# Single-turn completion
# ---------------------------------------------------------------------------


class TestSingleTurnCompletion:
    @pytest.mark.asyncio
    async def test_completed_when_all_params_provided_in_first_message(self) -> None:
        """If the user provides all params in one message, status should be COMPLETED.
        With A+B both incomplete, languageIsoCode is conflicting — the extractor
        must return namespaced keys 'languageIsoCode__0' and 'languageIsoCode__1'."""
        extractor_mock = _make_extractor_mock(
            _extraction(
                {
                    "languageIsoCode__0": "ET",
                    "languageIsoCode__1": "EN",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                    "station": "Tallinn",
                },
                [],
                "none",
            )
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="ET, EE, 2026-01-01, 2026-12-31, Tallinn",
            conversation_history=_HISTORY,
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert result.turn_count == 1

    @pytest.mark.asyncio
    async def test_endpoint_with_no_required_params_completes_on_any_turn(self) -> None:
        """Endpoint C has no required params — should be marked completed immediately."""
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        states = [_make_state(_ENDPOINT_C)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="go",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.COMPLETED


# ---------------------------------------------------------------------------
# Multi-turn accumulation
# ---------------------------------------------------------------------------


class TestMultiTurnAccumulation:
    @pytest.mark.asyncio
    async def test_params_accumulate_across_turns(self) -> None:
        """Turn 1 collects namespaced 'languageIsoCode__0/1', turn 2 completes both."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]
        store_mock = _make_session_store_mock()

        # Turn 1: collect shared languageIsoCode (namespaced because both incomplete)
        extractor1 = _make_extractor_mock(
            _extraction(
                {"languageIsoCode__0": "ET", "languageIsoCode__1": "EN"},
                ["countryIsoCode", "validFrom", "validTo", "station"],
                "Which country, date range, and weather station?",
            )
        )
        loop = _make_loop(extractor1, store_mock)
        result1 = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )
        assert result1.status == AgenticLoopStatus.NEEDS_INPUT
        assert state_a.collected_params["languageIsoCode"] == "ET"
        assert state_b.collected_params["languageIsoCode"] == "EN"

        # Turn 2: collect remaining params for both endpoints
        extractor2 = _make_extractor_mock(
            _extraction(
                {
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                    "station": "Tallinn",
                },
                [],
                "none",
            )
        )
        loop2 = _make_loop(extractor2, store_mock)
        result2 = await loop2.run_turn(
            chat_id=_CHAT_ID,
            user_message="EE, 2026-01-01, 2026-12-31, Tallinn",
            conversation_history=[],
            endpoint_states=states,
            turn_count=1,
        )
        assert result2.status == AgenticLoopStatus.COMPLETED
        assert state_a.completed is True
        assert state_b.completed is True

    @pytest.mark.asyncio
    async def test_per_endpoint_partial_completion(self) -> None:
        """One endpoint can complete before the others."""
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
        )
        state_d = _make_state(_ENDPOINT_D)  # needs 'station'
        states = [state_a, state_d]

        extractor_mock = _make_extractor_mock(
            _extraction({"countryIsoCode": "EE"}, ["station"], "Which weather station?")
        )
        loop = _make_loop(extractor_mock)
        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="EE",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        # A is now complete (has all 4 required params), D still missing station
        assert state_a.completed is True
        assert state_d.completed is False
        assert result.status == AgenticLoopStatus.NEEDS_INPUT


# ---------------------------------------------------------------------------
# Turn limit enforcement
# ---------------------------------------------------------------------------


class TestTurnLimitEnforcement:
    @pytest.mark.asyncio
    async def test_max_turns_reached_when_turn_count_equals_limit(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        # max_turns = min(3 * 2, 9) = 6
        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=6,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        extractor_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_turns_capped_at_multi_api_max_turns(self) -> None:
        """4 endpoints → 3*4=12 → capped at MULTI_API_MAX_TURNS=9."""
        endpoint_e = {
            "name": "get_public_holidays_lv",
            "url": "https://openholidaysapi.org/PublicHolidays",
            "params": [
                {
                    "name": "countryIsoCode",
                    "type": "string",
                    "required": True,
                    "description": "Country ISO code for Latvia",
                }
            ],
        }
        endpoint_f = {
            "name": "get_current_weather_lv",
            "url": "https://publicapi.envir.ee/v1/combinedWeatherData",
            "params": [
                {
                    "name": "station",
                    "type": "string",
                    "required": True,
                    "description": "Latvian weather station identifier",
                }
            ],
        }
        states = [
            _make_state(_ENDPOINT_A),
            _make_state(_ENDPOINT_B),
            _make_state(endpoint_e),
            _make_state(endpoint_f),
        ]
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        loop = _make_loop(extractor_mock)

        # turn_count=9 → max_turns=9 → should hit guard
        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=9,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        extractor_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_turn_count_incremented_on_max_turns(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # max_turns = min(3*1, 9) = 3 (holidays has 4 required params)
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert result.turn_count == 4

    @pytest.mark.asyncio
    async def test_session_not_saved_on_max_turns(self) -> None:
        store_mock = _make_session_store_mock()
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,
        )

        store_mock.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_multi_intent_last_valid_turn_executes_normally(self) -> None:
        """With 2 endpoints (multi-intent), max_turns = MULTI_INTENT_MAX_TURNS = 6.

        turn_count=5 → updated_turn_count=6 is the last turn that passes the
        guard (5 < 6).  Extraction must be attempted and the result must NOT be
        MAX_TURNS_REACHED, proving that 6 full turns execute before the fallback.
        """
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=5,  # updated_turn_count=6 — last turn before guard fires
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert result.turn_count == 6
        extractor_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_multi_intent_fixed_cap_overrides_per_endpoint_formula(self) -> None:
        """The multi-intent cap is fixed at MULTI_INTENT_MAX_TURNS=6, regardless of
        how many endpoints are active.

        With 3 endpoints the single-intent formula would give
        min(3 * 3, MULTI_API_MAX_TURNS) = min(9, 9) = 9 turns, but the
        multi-intent path uses MULTI_INTENT_MAX_TURNS=6 instead.  Asserting that
        turn_count=6 already triggers the guard for 3 endpoints confirms the
        fixed cap is applied and the per-endpoint formula is not.
        """
        endpoint_third = {
            "name": "get_public_holidays_lv",
            "url": "https://openholidaysapi.org/PublicHolidays",
            "params": [
                {
                    "name": "countryIsoCode",
                    "type": "string",
                    "required": True,
                    "description": "Country ISO code for Latvia",
                }
            ],
        }
        states = [
            _make_state(_ENDPOINT_A),
            _make_state(_ENDPOINT_B),
            _make_state(endpoint_third),
        ]
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        loop = _make_loop(extractor_mock)

        # With the single-endpoint formula: min(3*3, 9)=9 → turn_count=6 would NOT
        # trigger the guard.  With MULTI_INTENT_MAX_TURNS=6 it must.
        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=6,  # == MULTI_INTENT_MAX_TURNS — must fire for 3 endpoints too
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert result.turn_count == 7
        extractor_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Continuation threshold
# ---------------------------------------------------------------------------


class TestContinuationThreshold:
    @pytest.mark.asyncio
    async def test_continuation_asked_at_threshold(self) -> None:
        """With 2 endpoints (multi-intent), continuation_turn = MULTI_INTENT_CONTINUATION_TURN = 4.
        At turn_count=3, updated=4 → trigger."""
        extractor_mock = _make_extractor_mock(
            _extraction(
                {},
                ["languageIsoCode", "countryIsoCode", "station"],
                "Still missing params",
            )
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # updated_turn_count = 4 == MULTI_INTENT_CONTINUATION_TURN
        )

        assert result.status == AgenticLoopStatus.AWAITING_CONTINUATION_DECISION
        assert result.clarifying_question != ""
        assert result.turn_count == 4

    @pytest.mark.asyncio
    async def test_continuation_not_asked_before_threshold(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hmm",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,  # updated = 1, threshold = 3
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_continuation_not_asked_after_threshold(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="still no",
            conversation_history=[],
            endpoint_states=states,
            turn_count=4,  # updated = 5, past MULTI_INTENT_CONTINUATION_TURN=4
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_single_endpoint_continuation_threshold_is_two(self) -> None:
        """Single endpoint → continuation_turn = 2. turn_count=1 → updated=2 → trigger."""
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["station"], "Which weather station?")
        )
        states = [_make_state(_ENDPOINT_D)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hmm",
            conversation_history=[],
            endpoint_states=states,
            turn_count=1,  # updated = 2 == num_endpoints+1 = 2
        )

        assert result.status == AgenticLoopStatus.AWAITING_CONTINUATION_DECISION


# ---------------------------------------------------------------------------
# Continuation yes/no detection
# ---------------------------------------------------------------------------


class TestContinuationYesNo:
    @pytest.mark.asyncio
    async def test_user_yes_continues_normally(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="yes",
            conversation_history=[],
            endpoint_states=states,
            turn_count=4,  # updated=5, past MULTI_INTENT_CONTINUATION_TURN=4
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_estonian_yes_continues(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Mis keelekood?")
        )
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="jah",
            conversation_history=[],
            endpoint_states=states,
            turn_count=2,
            awaiting_continuation=True,
        )

        assert result.status != AgenticLoopStatus.MAX_TURNS_REACHED

    @pytest.mark.asyncio
    async def test_user_no_returns_max_turns_reached(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            endpoint_states=states,
            turn_count=2,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        extractor_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_response_treated_as_no(self) -> None:
        extractor_mock = _make_extractor_mock(_extraction({}, [], "none"))
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="maybe later",
            conversation_history=[],
            endpoint_states=states,
            turn_count=2,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    @pytest.mark.asyncio
    async def test_session_saved_on_needs_input(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction(
                {"languageIsoCode": "ET"},
                ["countryIsoCode", "validFrom", "validTo"],
                "Which country and date range?",
            )
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        store_mock.update.assert_awaited_once()
        call_kwargs = store_mock.update.await_args
        assert call_kwargs.args[0] == _CHAT_ID
        assert call_kwargs.kwargs["turn_count"] == 1
        assert call_kwargs.kwargs["awaiting_continuation"] is False

    @pytest.mark.asyncio
    async def test_session_saved_on_completed(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction(
                {
                    "languageIsoCode": "ET",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                    "station": "Tallinn",
                },
                [],
                "none",
            )
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="all params",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        store_mock.update.assert_awaited_once()
        call_kwargs = store_mock.update.await_args
        assert call_kwargs.kwargs["awaiting_continuation"] is False

    @pytest.mark.asyncio
    async def test_session_save_failure_does_not_raise(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction({}, ["languageIsoCode"], "Which language code?")
        )
        store_mock = _make_session_store_mock()
        store_mock.update.side_effect = RuntimeError("Dragonfly unavailable")
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_awaiting_continuation_saved_as_true_at_threshold(self) -> None:
        extractor_mock = _make_extractor_mock(
            _extraction(
                {}, ["languageIsoCode", "countryIsoCode", "station"], "Still missing"
            )
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # triggers continuation threshold (updated=4 == MULTI_INTENT_CONTINUATION_TURN)
        )

        call_kwargs = store_mock.update.await_args
        assert call_kwargs.kwargs["awaiting_continuation"] is True

    @pytest.mark.asyncio
    async def test_session_saved_on_extractor_error(self) -> None:
        """turn_count must be persisted to Dragonfly even when param extraction raises."""
        extractor_mock = MagicMock(side_effect=RuntimeError("LLM timeout"))
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        result = await loop.run_turn(
            chat_id=_CHAT_ID,
            user_message="Estonia",
            conversation_history=[],
            endpoint_states=states,
            turn_count=1,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        store_mock.update.assert_awaited_once()
        assert store_mock.update.await_args.kwargs["turn_count"] == 2


# ---------------------------------------------------------------------------
# Merged collected params helper
# ---------------------------------------------------------------------------


class TestMergedCollected:
    def test_merged_collected_returns_union(self) -> None:
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
        )
        state_b = _make_state(_ENDPOINT_B, collected={"station": "Tallinn"})
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        merged = loop._merged_collected([state_a, state_b])
        assert merged == {
            "languageIsoCode": "ET",
            "countryIsoCode": "EE",
            "validFrom": "2026-01-01",
            "validTo": "2026-12-31",
            "station": "Tallinn",
        }

    def test_merged_collected_empty_states(self) -> None:
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        assert loop._merged_collected([]) == {}


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
    """stream_run_turn() must mirror run_turn() semantics while returning (result, tokens)."""

    @pytest.mark.asyncio
    async def test_completed_returns_empty_tokens(self) -> None:
        """All params collected → COMPLETED result with no question tokens.
        With A+B both incomplete, languageIsoCode is namespaced."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {
                    "languageIsoCode__0": "ET",
                    "languageIsoCode__1": "EN",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                    "station": "Tallinn",
                },
                [],
                "none",
            ),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET, EE, 2026-01-01, 2026-12-31, Tallinn",
            conversation_history=_HISTORY,
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert tokens == []
        assert result.turn_count == 1

    @pytest.mark.asyncio
    async def test_needs_input_returns_question_tokens(self) -> None:
        """Missing params → NEEDS_INPUT result with streamed question tokens."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country", "?"],
            _extraction(
                {"languageIsoCode": "ET"},
                ["countryIsoCode", "station"],
                "Which country?",
            ),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == ["Which", " country", "?"]
        assert result.clarifying_question == "Which country?"

    @pytest.mark.asyncio
    async def test_re_extracted_param_overrides_prior_value(self) -> None:
        """Re-extracted value overwrites the previously collected one (correction allowed).
        A+B both incomplete — languageIsoCode is namespaced."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {"languageIsoCode__0": "EN", "languageIsoCode__1": "FR"}, [], "none"
            ),
        )
        # Both endpoints have unique params but not the shared languageIsoCode yet
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
        )
        state_b = _make_state(_ENDPOINT_B, collected={"station": "Tallinn"})
        states = [state_a, state_b]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="English please",
            conversation_history=[],
            endpoint_states=states,
            turn_count=1,
        )

        assert result.status == AgenticLoopStatus.COMPLETED
        assert state_a.collected_params["languageIsoCode"] == "EN"
        assert state_b.collected_params["languageIsoCode"] == "FR"
        assert tokens == []

    @pytest.mark.asyncio
    async def test_max_turns_reached_returns_empty_tokens(self) -> None:
        """Turn limit guard returns MAX_TURNS_REACHED with empty token list."""
        extractor_mock = _make_stream_extractor_mock(
            ["Some", " question?"],
            _extraction({}, ["languageIsoCode"], "Which language code?"),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        # max_turns = min(3 * 2, 9) = 6
        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=6,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert tokens == []
        extractor_mock.stream_forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stream_extraction_exception_returns_safe_defaults(self) -> None:
        """An exception from stream_forward must return NEEDS_INPUT with empty tokens."""
        extractor_mock = MagicMock()
        extractor_mock.stream_forward = AsyncMock(
            side_effect=RuntimeError("stream failure")
        )
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hello",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == []
        assert result.collected_params == {}

    @pytest.mark.asyncio
    async def test_session_language_forwarded_to_stream_forward(self) -> None:
        """session_language must be passed through to stream_forward."""
        extractor_mock = _make_stream_extractor_mock(
            ["Millist", " keelt?"],
            _extraction({}, ["language"], "Millist keelt?"),
        )
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="Mis keel?",
            conversation_history=_HISTORY,
            endpoint_states=states,
            turn_count=0,
            session_language="et",
        )

        extractor_mock.stream_forward.assert_awaited_once()
        call_kwargs = extractor_mock.stream_forward.call_args.kwargs
        assert call_kwargs["session_language"] == "et"

    @pytest.mark.asyncio
    async def test_user_exit_during_stream_returns_empty_tokens(self) -> None:
        """awaiting_continuation=True + 'no' → MAX_TURNS_REACHED with empty tokens."""
        extractor_mock = _make_stream_extractor_mock(
            ["Some", " tokens"],
            _extraction({}, ["language"], "Which language?"),
        )
        states = [_make_state(_ENDPOINT_A, collected={"countryIsoCode": "EE"})]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            endpoint_states=states,
            turn_count=2,
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.MAX_TURNS_REACHED
        assert tokens == []
        assert result.collected_params == {"countryIsoCode": "EE"}
        extractor_mock.stream_forward.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_user_yes_continues_normally(self) -> None:
        """awaiting_continuation=True + 'yes' → loop continues with NEEDS_INPUT."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " language?"],
            _extraction({}, ["languageIsoCode"], "Which language?"),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="yes",
            conversation_history=[],
            endpoint_states=states,
            turn_count=4,  # updated=5, past MULTI_INTENT_CONTINUATION_TURN=4
            awaiting_continuation=True,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == ["Which", " language?"]

    @pytest.mark.asyncio
    async def test_continuation_threshold_returns_word_tokens(self) -> None:
        """At the continuation threshold, tokens are the continuation question split word-by-word."""
        from tool_classifier.constants import CONTINUATION_QUESTION

        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {},
                ["languageIsoCode", "countryIsoCode", "station"],
                "Still missing",
            ),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # updated_turn_count = 4 == MULTI_INTENT_CONTINUATION_TURN
        )

        assert result.status == AgenticLoopStatus.AWAITING_CONTINUATION_DECISION
        assert "".join(tokens) == CONTINUATION_QUESTION
        assert all(t.endswith(" ") for t in tokens[:-1])
        assert not tokens[-1].endswith(" ")

    @pytest.mark.asyncio
    async def test_continuation_threshold_uses_continuation_language(self) -> None:
        """continuation_language overrides session_language for the threshold tokens."""
        from tool_classifier.constants import CONTINUATION_QUESTION_ET

        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {},
                ["languageIsoCode", "countryIsoCode", "station"],
                "Veel puudub",
            ),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ei tea",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # updated_turn_count = 4 == MULTI_INTENT_CONTINUATION_TURN
            session_language="et",
            continuation_language="et",
        )

        assert result.status == AgenticLoopStatus.AWAITING_CONTINUATION_DECISION
        assert "".join(tokens) == CONTINUATION_QUESTION_ET

    @pytest.mark.asyncio
    async def test_shared_params_distributed_to_both_endpoints(self) -> None:
        """Namespaced params are written to their corresponding endpoint states."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country?"],
            _extraction(
                {"languageIsoCode__0": "ET", "languageIsoCode__1": "EN"},
                ["countryIsoCode", "station"],
                "Which country?",
            ),
        )
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        states = [state_a, state_b]
        loop = _make_loop(extractor_mock)

        result, _ = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert state_a.collected_params["languageIsoCode"] == "ET"
        assert state_b.collected_params["languageIsoCode"] == "EN"
        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_session_saved_on_completed(self) -> None:
        """Session is persisted with awaiting_continuation=False on COMPLETED."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {
                    "languageIsoCode__0": "ET",
                    "languageIsoCode__1": "ET",
                    "countryIsoCode": "EE",
                    "validFrom": "2026-01-01",
                    "validTo": "2026-12-31",
                    "station": "Tallinn",
                },
                [],
                "none",
            ),
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="all params",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        store_mock.update.assert_awaited_once()
        assert store_mock.update.await_args.kwargs["awaiting_continuation"] is False

    @pytest.mark.asyncio
    async def test_session_saved_on_needs_input(self) -> None:
        """Session is persisted with awaiting_continuation=False on NEEDS_INPUT."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country?"],
            _extraction(
                {"languageIsoCode": "ET"},
                ["countryIsoCode", "validFrom", "validTo"],
                "Which country?",
            ),
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        store_mock.update.assert_awaited_once()
        call_kwargs = store_mock.update.await_args.kwargs
        assert call_kwargs["awaiting_continuation"] is False
        assert call_kwargs["turn_count"] == 1

    @pytest.mark.asyncio
    async def test_session_saved_with_awaiting_continuation_true_at_threshold(
        self,
    ) -> None:
        """Session is persisted with awaiting_continuation=True at the threshold."""
        extractor_mock = _make_stream_extractor_mock(
            [],
            _extraction(
                {},
                ["languageIsoCode", "countryIsoCode", "station"],
                "Still missing",
            ),
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # triggers continuation threshold (updated=4 == MULTI_INTENT_CONTINUATION_TURN)
        )

        call_kwargs = store_mock.update.await_args.kwargs
        assert call_kwargs["awaiting_continuation"] is True
        assert call_kwargs["turn_count"] == 4

    @pytest.mark.asyncio
    async def test_session_not_saved_on_max_turns(self) -> None:
        """No session save when the turn limit is hit."""
        extractor_mock = _make_stream_extractor_mock([], _extraction({}, [], "none"))
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=3,  # max_turns = min(3 * 1, 9) = 3
        )

        store_mock.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_not_saved_on_user_exit(self) -> None:
        """No session save when the user chooses to exit (awaiting_continuation + 'no')."""
        extractor_mock = _make_stream_extractor_mock([], _extraction({}, [], "none"))
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="no",
            conversation_history=[],
            endpoint_states=states,
            turn_count=2,
            awaiting_continuation=True,
        )

        store_mock.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_session_save_failure_does_not_raise(self) -> None:
        """A Dragonfly failure must not propagate — NEEDS_INPUT is still returned."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " language?"],
            _extraction({}, ["languageIsoCode"], "Which language?"),
        )
        store_mock = _make_session_store_mock()
        store_mock.update.side_effect = RuntimeError("Dragonfly unavailable")
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        result, _ = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT

    @pytest.mark.asyncio
    async def test_session_saved_on_stream_extractor_error(self) -> None:
        """turn_count must be persisted to Dragonfly even when stream_forward raises."""
        extractor_mock = MagicMock()
        extractor_mock.stream_forward = AsyncMock(
            side_effect=RuntimeError("stream failure")
        )
        store_mock = _make_session_store_mock()
        states = [_make_state(_ENDPOINT_A)]
        loop = _make_loop(extractor_mock, store_mock)

        result, tokens = await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="hi",
            conversation_history=[],
            endpoint_states=states,
            turn_count=1,
        )

        assert result.status == AgenticLoopStatus.NEEDS_INPUT
        assert tokens == []
        store_mock.update.assert_awaited_once()
        assert store_mock.update.await_args.kwargs["turn_count"] == 2


# ---------------------------------------------------------------------------
# _build_intent_groups — unit tests
# ---------------------------------------------------------------------------


class TestBuildIntentGroups:
    """Unit tests for MultiEndpointAgenticLoop._build_intent_groups()."""

    def test_two_intents_with_missing_required_params_returns_two_groups(
        self,
    ) -> None:
        """Two incomplete endpoints each with missing required params → two groups."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [state_a, state_b], already_collected={}, namespace_map={}
        )

        assert len(groups) == 2
        intents = {g["intent"] for g in groups}
        assert "get_public_holidays" in intents
        assert "get_current_weather" in intents

    def test_single_incomplete_endpoint_returns_single_element_list(self) -> None:
        """Only one incomplete endpoint has missing required params → single-element list."""
        state_a = _make_state(_ENDPOINT_A)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [state_a], already_collected={}, namespace_map={}
        )

        assert len(groups) == 1

    def test_no_endpoints_with_missing_returns_empty_list(self) -> None:
        """No incomplete endpoints → empty list."""
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups([], already_collected={}, namespace_map={})

        assert groups == []

    def test_completed_endpoints_are_skipped(self) -> None:
        """A completed endpoint must not contribute a group."""
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
            completed=True,
        )
        state_b = _make_state(_ENDPOINT_B)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        # Only one incomplete endpoint → single-element list for the incomplete endpoint
        groups = loop._build_intent_groups(
            [state_a, state_b],
            already_collected={"languageIsoCode": "ET"},
            namespace_map={},
        )

        assert len(groups) == 1
        assert groups[0]["intent"] == "get_current_weather"

    def test_already_collected_params_excluded_from_descriptions(self) -> None:
        """Params already in already_collected must not appear in missing_param_descriptions."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        already_collected = {"languageIsoCode": "ET"}
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [state_a, state_b],
            already_collected=already_collected,
            namespace_map={},
        )

        # Groups still returned because both endpoints still have missing params
        assert len(groups) == 2
        for group in groups:
            # languageIsoCode is collected — its description must not appear
            assert not any(
                "language" in desc.lower()
                for desc in group["missing_param_descriptions"]
            )

    def test_optional_params_excluded_from_descriptions(self) -> None:
        """Optional (required=False) params must not appear in missing_param_descriptions."""
        endpoint_with_optional = {
            "name": "get_data_a",
            "url": "https://api.example.com/a",
            "params": [
                {
                    "name": "requiredParam",
                    "type": "string",
                    "required": True,
                    "description": "The required value",
                },
                {
                    "name": "optionalParam",
                    "type": "string",
                    "required": False,
                    "description": "An optional filter",
                },
            ],
        }
        endpoint_b_required = {
            "name": "get_data_b",
            "url": "https://api.example.com/b",
            "params": [
                {
                    "name": "anotherRequired",
                    "type": "string",
                    "required": True,
                    "description": "Another required field",
                },
            ],
        }
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [_make_state(endpoint_with_optional), _make_state(endpoint_b_required)],
            already_collected={},
            namespace_map={},
        )

        assert len(groups) == 2
        group_a = next(g for g in groups if g["intent"] == "get_data_a")
        descriptions = group_a["missing_param_descriptions"]
        assert "The required value" in descriptions
        assert not any("optional" in d.lower() for d in descriptions)

    def test_format_hints_stripped_from_descriptions(self) -> None:
        """(YYYY-MM-DD) format hints must be stripped from missing_param_descriptions."""
        state_a = _make_state(_ENDPOINT_A)  # validFrom/validTo have (YYYY-MM-DD) hints
        state_b = _make_state(_ENDPOINT_B)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [state_a, state_b],
            already_collected={"languageIsoCode": "ET"},
            namespace_map={},
        )

        group_a = next(g for g in groups if g["intent"] == "get_public_holidays")
        for desc in group_a["missing_param_descriptions"]:
            assert "YYYY" not in desc
            assert "MM-DD" not in desc

    def test_intent_name_from_endpoint_name_key(self) -> None:
        """Group 'intent' field must be taken from the endpoint's 'name' key."""
        state_a = _make_state(_ENDPOINT_A)
        state_b = _make_state(_ENDPOINT_B)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [state_a, state_b], already_collected={}, namespace_map={}
        )

        intents = {g["intent"] for g in groups}
        assert "get_public_holidays" in intents
        assert "get_current_weather" in intents

    def test_intent_name_falls_back_to_description_when_name_absent(self) -> None:
        """When endpoint has no 'name' key, 'intent' must fall back to 'description'."""
        endpoint_no_name = {
            "description": "lookup_initiative",
            "url": "https://api.example.com/initiative",
            "params": [
                {
                    "name": "initiativeId",
                    "type": "string",
                    "required": True,
                    "description": "Unique initiative identifier",
                },
            ],
        }
        endpoint_b = {
            "name": "search_address",
            "url": "https://api.example.com/address",
            "params": [
                {
                    "name": "address",
                    "type": "string",
                    "required": True,
                    "description": "Street address or place name",
                },
            ],
        }
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        groups = loop._build_intent_groups(
            [_make_state(endpoint_no_name), _make_state(endpoint_b)],
            already_collected={},
            namespace_map={},
        )

        assert len(groups) == 2
        intents = {g["intent"] for g in groups}
        assert "lookup_initiative" in intents
        assert "search_address" in intents

    def test_endpoint_with_only_optional_params_excluded_from_groups(self) -> None:
        """An endpoint with no missing required params contributes no group;
        the remaining endpoint's required params produce a single-element list."""
        state_c = _make_state(_ENDPOINT_C)  # only optional params
        state_b = _make_state(_ENDPOINT_B)  # has required 'station'
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        # C has no required params → no group for C → only 1 group (for B)
        groups = loop._build_intent_groups(
            [state_c, state_b], already_collected={}, namespace_map={}
        )

        assert len(groups) == 1


# ---------------------------------------------------------------------------
# intent_groups forwarding in stream_run_turn
# ---------------------------------------------------------------------------


class TestStreamRunTurnIntentGroupsForwarding:
    """Verify that intent_groups is built correctly and forwarded to stream_forward."""

    @pytest.mark.asyncio
    async def test_intent_groups_forwarded_for_two_incomplete_endpoints(self) -> None:
        """stream_forward must receive a non-empty intent_groups list when two
        incomplete endpoints each have missing required params."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country?"],
            _extraction(
                {},
                ["languageIsoCode", "countryIsoCode", "station"],
                "Which country and station?",
            ),
        )
        states = [_make_state(_ENDPOINT_A), _make_state(_ENDPOINT_B)]
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        extractor_mock.stream_forward.assert_awaited_once()
        call_kwargs = extractor_mock.stream_forward.call_args.kwargs
        intent_groups = call_kwargs["intent_groups"]
        assert isinstance(intent_groups, list)
        assert len(intent_groups) == 2
        intents = {g["intent"] for g in intent_groups}
        assert "get_public_holidays" in intents
        assert "get_current_weather" in intents
        for group in intent_groups:
            assert "missing_param_descriptions" in group
            assert isinstance(group["missing_param_descriptions"], list)
            assert len(group["missing_param_descriptions"]) > 0

    @pytest.mark.asyncio
    async def test_intent_groups_single_element_for_single_endpoint(self) -> None:
        """stream_forward must receive a single-element intent_groups list when only one endpoint is active."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " station?"],
            _extraction({}, ["station"], "Which station?"),
        )
        states = [_make_state(_ENDPOINT_D)]
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="not sure",
            conversation_history=[],
            endpoint_states=states,
            turn_count=0,
        )

        extractor_mock.stream_forward.assert_awaited_once()
        call_kwargs = extractor_mock.stream_forward.call_args.kwargs
        assert len(call_kwargs["intent_groups"]) == 1
        assert call_kwargs["intent_groups"][0]["intent"] == "get_current_weather"

    @pytest.mark.asyncio
    async def test_intent_groups_single_element_when_other_endpoint_complete(
        self,
    ) -> None:
        """stream_forward must receive a single-element intent_groups list when only one of
        two endpoints has missing required params (the other has all params collected)."""
        state_a = _make_state(
            _ENDPOINT_A,
            collected={
                "languageIsoCode": "ET",
                "countryIsoCode": "EE",
                "validFrom": "2026-01-01",
                "validTo": "2026-12-31",
            },
            completed=True,
        )
        state_b = _make_state(_ENDPOINT_B)  # still missing 'station'
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " station?"],
            _extraction({"languageIsoCode": "ET"}, ["station"], "Which station?"),
        )
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=[state_a, state_b],
            turn_count=0,
        )

        extractor_mock.stream_forward.assert_awaited_once()
        call_kwargs = extractor_mock.stream_forward.call_args.kwargs
        assert len(call_kwargs["intent_groups"]) == 1
        assert call_kwargs["intent_groups"][0]["intent"] == "get_current_weather"

    @pytest.mark.asyncio
    async def test_intent_groups_excludes_already_collected_params(self) -> None:
        """Already-collected params must not appear in intent_groups descriptions
        forwarded to stream_forward."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " country?"],
            _extraction(
                {},
                ["countryIsoCode", "validFrom", "validTo", "station"],
                "Which country and station?",
            ),
        )
        state_a = _make_state(_ENDPOINT_A, collected={"languageIsoCode": "ET"})
        state_b = _make_state(_ENDPOINT_B, collected={"languageIsoCode": "ET"})
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=[state_a, state_b],
            turn_count=1,
        )

        extractor_mock.stream_forward.assert_awaited_once()
        intent_groups = extractor_mock.stream_forward.call_args.kwargs["intent_groups"]
        assert len(intent_groups) == 2
        # languageIsoCode is collected — its description must not appear
        for group in intent_groups:
            assert not any(
                "language" in desc.lower()
                for desc in group["missing_param_descriptions"]
            )

    @pytest.mark.asyncio
    async def test_intent_groups_format_hints_stripped_before_forwarding(self) -> None:
        """Format hints in descriptions must be stripped before being passed to stream_forward."""
        extractor_mock = _make_stream_extractor_mock(
            ["Which", " dates?"],
            _extraction(
                {"languageIsoCode": "ET"},
                ["countryIsoCode", "validFrom", "validTo", "station"],
                "Which dates and station?",
            ),
        )
        state_a = _make_state(_ENDPOINT_A, collected={"languageIsoCode": "ET"})
        state_b = _make_state(_ENDPOINT_B, collected={"languageIsoCode": "ET"})
        loop = _make_loop(extractor_mock)

        await loop.stream_run_turn(
            chat_id=_CHAT_ID,
            user_message="ET",
            conversation_history=[],
            endpoint_states=[state_a, state_b],
            turn_count=1,
        )

        intent_groups = extractor_mock.stream_forward.call_args.kwargs["intent_groups"]
        group_a = next(g for g in intent_groups if g["intent"] == "get_public_holidays")
        for desc in group_a["missing_param_descriptions"]:
            assert "YYYY" not in desc
            assert "MM-DD" not in desc


# ---------------------------------------------------------------------------
# Conflict namespace fixtures — both endpoints define startDate/endDate
# ---------------------------------------------------------------------------

_ENDPOINT_E: Dict[str, Any] = {
    "name": "get_first_intent_data",
    "url": "https://api.example.com/first",
    "params": [
        {
            "name": "startDate",
            "type": "date",
            "required": True,
            "description": "Start date for first intent (YYYY-MM-DD)",
        },
        {
            "name": "endDate",
            "type": "date",
            "required": True,
            "description": "End date for first intent (YYYY-MM-DD)",
        },
    ],
}

_ENDPOINT_F: Dict[str, Any] = {
    "name": "get_second_intent_data",
    "url": "https://api.example.com/second",
    "params": [
        {
            "name": "startDate",
            "type": "date",
            "required": True,
            "description": "Start date for second intent (YYYY-MM-DD)",
        },
        {
            "name": "endDate",
            "type": "date",
            "required": True,
            "description": "End date for second intent (YYYY-MM-DD)",
        },
    ],
}


# ---------------------------------------------------------------------------
# TestSchemaNamespacing — conflicting param namespacing
# ---------------------------------------------------------------------------


class TestSchemaNamespacing:
    """Unit tests verifying that conflicting params (same name in 2+ incomplete
    endpoints) are namespaced as {name}__{ep_idx} in the merged schema."""

    def test_build_merged_schema_produces_namespaced_keys_for_conflicting_params(
        self,
    ) -> None:
        """startDate/endDate appear in both E and F ? namespaced keys in schema."""
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        merged_schema, param_owners, namespace_map = loop._build_merged_schema(
            [_make_state(_ENDPOINT_E), _make_state(_ENDPOINT_F)]
        )

        names = [p["name"] for p in merged_schema]
        assert "startDate__0" in names
        assert "startDate__1" in names
        assert "endDate__0" in names
        assert "endDate__1" in names
        # Original names must NOT appear
        assert "startDate" not in names
        assert "endDate" not in names

    def test_namespace_map_populated_for_startdate_enddate_conflict(self) -> None:
        """namespace_map maps each namespaced key to (ep_idx, original_name)."""
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, _, namespace_map = loop._build_merged_schema(
            [_make_state(_ENDPOINT_E), _make_state(_ENDPOINT_F)]
        )

        assert namespace_map["startDate__0"] == (0, "startDate")
        assert namespace_map["startDate__1"] == (1, "startDate")
        assert namespace_map["endDate__0"] == (0, "endDate")
        assert namespace_map["endDate__1"] == (1, "endDate")

    def test_param_owners_maps_each_namespaced_key_to_single_endpoint(self) -> None:
        """param_owners for namespaced keys lists only one endpoint each."""
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        _, param_owners, _ = loop._build_merged_schema(
            [_make_state(_ENDPOINT_E), _make_state(_ENDPOINT_F)]
        )

        assert param_owners["startDate__0"] == [0]
        assert param_owners["startDate__1"] == [1]
        assert param_owners["endDate__0"] == [0]
        assert param_owners["endDate__1"] == [1]

    def test_distribute_params_writes_original_names_to_each_endpoint(self) -> None:
        """Namespaced keys in extracted_params are written under original names
        to the correct endpoint's collected_params."""
        state_e = _make_state(_ENDPOINT_E)
        state_f = _make_state(_ENDPOINT_F)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        _, param_owners, namespace_map = loop._build_merged_schema([state_e, state_f])
        loop._distribute_params(
            {
                "startDate__0": "2026-01-01",
                "endDate__0": "2026-06-30",
                "startDate__1": "2026-07-01",
                "endDate__1": "2026-12-31",
            },
            [state_e, state_f],
            param_owners,
            namespace_map,
        )

        assert state_e.collected_params["startDate"] == "2026-01-01"
        assert state_e.collected_params["endDate"] == "2026-06-30"
        assert state_f.collected_params["startDate"] == "2026-07-01"
        assert state_f.collected_params["endDate"] == "2026-12-31"

    def test_build_intent_groups_shows_date_params_in_both_groups(self) -> None:
        """With namespaced dates, both groups should list their own date descriptions."""
        state_e = _make_state(_ENDPOINT_E)
        state_f = _make_state(_ENDPOINT_F)
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))

        _, _, namespace_map = loop._build_merged_schema([state_e, state_f])
        groups = loop._build_intent_groups(
            [state_e, state_f], already_collected={}, namespace_map=namespace_map
        )

        assert len(groups) == 2
        for group in groups:
            assert len(group["missing_param_descriptions"]) > 0


# ---------------------------------------------------------------------------
# TestBuildNamespacedAlreadyCollected
# ---------------------------------------------------------------------------


class TestBuildNamespacedAlreadyCollected:
    """Unit tests for _build_namespaced_already_collected()."""

    def test_fast_path_when_no_namespace_map(self) -> None:
        """When namespace_map is empty, returns simple union of all collected_params."""
        state_a = _make_state(_ENDPOINT_A, collected={"languageIsoCode": "ET"})
        state_b = _make_state(_ENDPOINT_B, collected={"station": "Tallinn"})
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        result = loop._build_namespaced_already_collected(
            [state_a, state_b], namespace_map={}
        )
        assert result["languageIsoCode"] == "ET"
        assert result["station"] == "Tallinn"

    def test_conflicting_params_appear_under_namespaced_keys(self) -> None:
        """When namespace_map is set, each endpoint's conflicting params are
        returned under their namespaced keys."""
        state_e = _make_state(
            _ENDPOINT_E, collected={"startDate": "2026-01-01", "endDate": "2026-06-30"}
        )
        state_f = _make_state(_ENDPOINT_F, collected={"startDate": "2026-07-01"})

        namespace_map = {
            "startDate__0": (0, "startDate"),
            "startDate__1": (1, "startDate"),
            "endDate__0": (0, "endDate"),
            "endDate__1": (1, "endDate"),
        }
        loop = _make_loop(_make_extractor_mock(_extraction({}, [], "none")))
        result = loop._build_namespaced_already_collected(
            [state_e, state_f], namespace_map=namespace_map
        )

        assert result["startDate__0"] == "2026-01-01"
        assert result["endDate__0"] == "2026-06-30"
        assert result["startDate__1"] == "2026-07-01"
        # endDate__1 not yet collected — absent
        assert "endDate__1" not in result
        # Original conflicting names must not appear at top level
        assert "startDate" not in result
        assert "endDate" not in result

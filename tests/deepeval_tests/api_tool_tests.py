"""
DeepEval tests for the API Tool Calling feature (issue #447).

Covers the single-intent (Scenarios 1-4) and multi-intent (MI-1..MI-8) scenarios
listed in the issue against the running orchestration service. Each scenario
walks the ``/orchestrate`` endpoint turn-by-turn via the testcontainers-backed
``orchestration_client`` fixture, extracts the final agentic-loop tool call,
and scores it with a DeepEval agentic metric:

* **Strict single-intent (S1, S2a, S2b, S3)** — the issue specifies an
  "Expected endpoint" + URL/params per scenario. Scored with
  ``ToolCorrectnessMetric`` (deterministic name + input-parameter comparison,
  ``threshold=1.0``).

* **Loose single-intent (S4)** — the issue documents the 5-turn flow but does
  not specify an expected resolution. Scored with
  ``ArgumentCorrectnessMetric`` (LLM-as-judge over the conversation input and
  the resolved tool call).

* **Multi-intent (MI-1..MI-8)** — issue lists only queries (EN+ET). If the
  system resolves a tool call, scored with ``ArgumentCorrectnessMetric``; if
  it asks a clarifying question instead, the test only verifies routing to
  ATC (i.e. non-empty reply, not silent RAG/OOD).

Endpoints are matched by ``name`` — the UUIDs in the issue do happen to match
those in ``tests/api_tool_eval/test-endpoints.json``, but name matching is the
stable contract.

Depends on:
* ``orchestration_client`` — provides the testcontainers-mapped base URL.
* ``api_tool_endpoints_indexed`` — seeds the API tool fixture into Qdrant's
  ``api_tool_collection`` so the agentic loop can find the endpoints.
"""

import datetime
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import requests
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

REQUEST_TIMEOUT = 60
ENVIRONMENT = "development"
AUTHOR_ID = "api-tool-deepeval"

# Where the result-collector writes the per-scenario record consumed by
# tests/deepeval_tests/api_tool_report_generator.py to render the PR
# comment / artifact markdown.
RESULTS_FILE = Path("api_tool_test_results.json")

# Strict scenarios assert the exact expected tool was called with the exact
# expected params (extras allowed — see ToolCorrectnessMetric docs on
# should_exact_match). Threshold 1.0 because the deterministic comparison
# scores fractionally over expected_tools, and we want every expected param
# present and correct.
STRICT_TOOL_THRESHOLD = 1.0

# Loose scenarios are graded by an LLM judge — 0.7 matches the threshold used
# for the RAG metrics in standard_tests.py.
JUDGE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# HTTP helpers (mirror tests/api_tool_eval/integration_test_*.py)
# ---------------------------------------------------------------------------


def _make_chat_id(label: str) -> str:
    return f"deepeval-api-tool-{label}-{uuid.uuid4().hex[:8]}"


def _send_turn(
    base_url: str,
    chat_id: str,
    message: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "chatId": chat_id,
        "message": message,
        "authorId": AUTHOR_ID,
        "conversationHistory": history,
        "url": "deepeval-test",
        "environment": ENVIRONMENT,
    }
    resp = requests.post(
        f"{base_url}/orchestrate",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _append_history(
    history: List[Dict[str, str]],
    user_message: str,
    bot_response: str,
) -> List[Dict[str, str]]:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return history + [
        {"authorRole": "user", "message": user_message, "timestamp": ts},
        {"authorRole": "bot", "message": bot_response, "timestamp": ts},
    ]


def _parse_completed(content: str) -> Optional[Dict[str, Any]]:
    """Return the parsed JSON if content is a completed agentic-loop payload,
    else None. A completed payload carries both ``endpoint`` and
    ``collected_params`` keys."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if "collected_params" in data and "endpoint" in data:
        return data
    return None


def _to_tool_call(content: str) -> Optional[ToolCall]:
    """Build a DeepEval ToolCall from a completed agentic-loop payload, or
    None if the response wasn't a completed JSON."""
    data = _parse_completed(content)
    if data is None:
        return None
    ep = data.get("endpoint", {})
    name = ep.get("name", "") if isinstance(ep, dict) else str(ep)
    params = data.get("collected_params", {}) or {}
    return ToolCall(name=name, input_parameters=params)


def _conversation_text(turns: List[Dict[str, Any]]) -> str:
    """Flatten the user turns into a single string for LLMTestCase.input.

    DeepEval's agentic single-turn metrics take a single ``input`` string;
    this approximation gives the LLM judge the full conversational context.
    """
    return "\n".join(f"USER: {t['user']}" for t in turns)


def _walk_turns(base_url: str, label: str, turns: List[Dict[str, Any]]) -> str:
    """POST each turn in sequence, maintaining a stable chatId + history.
    Returns the final bot response content."""
    chat_id = _make_chat_id(label)
    history: List[Dict[str, str]] = []
    final_content = ""
    for turn in turns:
        resp = _send_turn(base_url, chat_id, turn["user"], history)
        final_content = resp.get("content", "")
        history = _append_history(history, turn["user"], final_content)
    return final_content


def _tool_call_to_dict(tc: Optional[ToolCall]) -> Optional[Dict[str, Any]]:
    """Serialize a ToolCall for the results JSON (None-tolerant)."""
    if tc is None:
        return None
    return {"name": tc.name, "input_parameters": dict(tc.input_parameters or {})}


# ---------------------------------------------------------------------------
# Result collector — every test pushes one record, autouse fixture flushes to
# disk at session end. tests/deepeval_tests/api_tool_report_generator.py
# reads the JSON and renders the markdown report consumed by the workflow.
# ---------------------------------------------------------------------------


class ApiToolResultCollector:
    """Accumulates per-scenario results from the API tool tests."""

    def __init__(self) -> None:
        self.results: Dict[str, Any] = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "errored_tests": 0,
            "test_start_time": datetime.datetime.now().isoformat(),
            "scenarios": [],
        }

    def add(
        self,
        scenario_id: str,
        scenario_type: str,
        metric_name: str,
        threshold: float,
        score: Optional[float],
        passed: bool,
        reason: str = "",
        error: str = "",
        expected_tool: Optional[Dict[str, Any]] = None,
        actual_tool: Optional[Dict[str, Any]] = None,
        final_response_preview: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.results["total_tests"] += 1
        if error:
            self.results["errored_tests"] += 1
        elif passed:
            self.results["passed_tests"] += 1
        else:
            self.results["failed_tests"] += 1
        self.results["scenarios"].append(
            {
                "id": scenario_id,
                "type": scenario_type,
                "metric": metric_name,
                "threshold": threshold,
                "score": score,
                "passed": passed,
                "reason": reason,
                "error": error,
                "expected_tool": expected_tool,
                "actual_tool": actual_tool,
                "final_response_preview": final_response_preview,
                "extra": extra or {},
            }
        )

    def save(self, path: Path = RESULTS_FILE) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, default=str, ensure_ascii=False)
        print(
            f"Saved API tool results to {path}: "
            f"{self.results['passed_tests']}/{self.results['total_tests']} passed, "
            f"{self.results['failed_tests']} failed, "
            f"{self.results['errored_tests']} errored"
        )


_collector = ApiToolResultCollector()


@pytest.fixture(scope="session", autouse=True)
def _save_api_tool_results():
    """Flush collected results to RESULTS_FILE at end of session, even on
    failure — mirrors save_results_fixture in standard_tests.py."""
    yield
    _collector.save()


# ---------------------------------------------------------------------------
# Strict single-intent scenarios — issue specifies expected endpoint + params
# (S1, S2a, S2b, S3). Scored with ToolCorrectnessMetric.
# ---------------------------------------------------------------------------


STRICT_SINGLE_INTENT_SCENARIOS: List[Dict[str, Any]] = [
    # Scenario 1 — Normal Workflow (citizen initiative details)
    {
        "id": "S1-citizen-initiative-EN",
        "label": "s1-en",
        "turns": [
            {"user": "Can I see the details of a citizen initiative?"},
            {"user": "1790"},
        ],
        "expected_tool": ToolCall(
            name="get_initiative_details",
            input_parameters={"id": "1790"},
        ),
    },
    {
        "id": "S1-citizen-initiative-ET",
        "label": "s1-et",
        "turns": [
            {"user": "Kas ma saan kodanikualgatuse üksikasju vaadata?"},
            {"user": "1790"},
        ],
        "expected_tool": ToolCall(
            name="get_initiative_details",
            input_parameters={"id": "1790"},
        ),
    },
    # Scenario 2a — Public Holidays (date range correction)
    {
        "id": "S2a-public-holidays-date-correction-EN",
        "label": "s2a-en",
        "turns": [
            {"user": "What are the public holidays in Estonia?"},
            {"user": "From 2026-01-01"},
            {
                "user": (
                    "My mistake — the correct period is April 1, 2026 "
                    "through December 31, 2026."
                )
            },
        ],
        "expected_tool": ToolCall(
            name="get_public_holidays",
            input_parameters={
                "countryIsoCode": "EE",
                "validFrom": "2026-04-01",
                "validTo": "2026-12-31",
            },
        ),
    },
    {
        "id": "S2a-public-holidays-date-correction-ET",
        "label": "s2a-et",
        "turns": [
            {"user": "Millised on riigipühad Eestis?"},
            {"user": "Alates 1. jaanuarist 2026"},
            {
                "user": (
                    "Minu viga, õige periood on 01.04.2026 kuni 31.12.2026. "
                    "Tegelikult tahan 2026-04-01 kuni 2026-12-31."
                )
            },
        ],
        "expected_tool": ToolCall(
            name="get_public_holidays",
            input_parameters={
                "countryIsoCode": "EE",
                "validFrom": "2026-04-01",
                "validTo": "2026-12-31",
            },
        ),
    },
    # Scenario 2b — Parliament Votings (date range correction)
    {
        "id": "S2b-parliament-votings-date-correction-EN",
        "label": "s2b-en",
        "turns": [
            {"user": "What votes took place in the Estonian parliament?"},
            {"user": "2026-04-05"},
            {
                "user": (
                    "My mistake — the correct period is April 6, 2026 "
                    "through April 7, 2026."
                )
            },
        ],
        "expected_tool": ToolCall(
            name="get_parliament_votings",
            input_parameters={
                "startDate": "2026-04-06",
                "endDate": "2026-04-07",
            },
        ),
    },
    {
        "id": "S2b-parliament-votings-date-correction-DE",
        "label": "s2b-de",
        "turns": [
            {"user": ("Welche Abstimmungen fanden im estnischen Parlament statt?")},
            {"user": "2026-04-05"},
            {
                "user": (
                    "Mein Fehler — der richtige Zeitraum ist vom "
                    "6. April 2026 bis zum 7. April 2026."
                )
            },
        ],
        "expected_tool": ToolCall(
            name="get_parliament_votings",
            input_parameters={
                "startDate": "2026-04-06",
                "endDate": "2026-04-07",
            },
        ),
    },
    # Scenario 3 — Intent Switch (electricity prices -> address search)
    {
        "id": "S3-intent-switch-EN",
        "label": "s3-en",
        "turns": [
            {"user": "Show last week's electricity prices in Estonia."},
            {
                "user": (
                    "Wait — could you check the following location instead: "
                    "Viru tn 4, Tallinn?"
                )
            },
        ],
        "expected_tool": ToolCall(
            name="search_address",
            input_parameters={"address": "Viru tn 4, Tallinn"},
        ),
    },
    {
        "id": "S3-intent-switch-ET",
        "label": "s3-et",
        "turns": [
            {"user": "Näita eelmise nädala elektrienergia hindu Eestis."},
            {
                "user": (
                    "Oota, kas saaksid hoopis järgmist asukohta kontrollida: "
                    "Viru tn 4, Tallinn?"
                )
            },
        ],
        "expected_tool": ToolCall(
            name="search_address",
            input_parameters={"address": "Viru tn 4, Tallinn"},
        ),
    },
]


@pytest.mark.parametrize(
    "scenario",
    STRICT_SINGLE_INTENT_SCENARIOS,
    ids=[s["id"] for s in STRICT_SINGLE_INTENT_SCENARIOS],
)
def test_api_tool_strict_single_intent(
    scenario: Dict[str, Any],
    orchestration_client: Any,
    api_tool_endpoints_indexed: None,
) -> None:
    """Scenarios 1, 2a, 2b, 3 — issue specifies the expected tool call.

    Scored deterministically with ``ToolCorrectnessMetric``:
    * the resolved tool's name must equal the expected name, and
    * every expected input parameter must be present and equal in the
      ``collected_params`` (extras allowed).
    """
    del api_tool_endpoints_indexed  # consumed for its setup side effect only

    expected_tool: ToolCall = scenario["expected_tool"]
    final_content = ""
    actual_tool: Optional[ToolCall] = None
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""

    try:
        final_content = _walk_turns(
            orchestration_client.base_url, scenario["label"], scenario["turns"]
        )
        actual_tool = _to_tool_call(final_content)
        test_case = LLMTestCase(
            input=_conversation_text(scenario["turns"]),
            actual_output=final_content,
            tools_called=[actual_tool] if actual_tool is not None else [],
            expected_tools=[expected_tool],
        )
        metric = ToolCorrectnessMetric(
            threshold=STRICT_TOOL_THRESHOLD,
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        )
        metric.measure(test_case)
        score = metric.score
        reason = metric.reason or ""
        passed = score is not None and score >= STRICT_TOOL_THRESHOLD
        assert passed, (
            f"[{scenario['id']}] tool correctness {score} < "
            f"{STRICT_TOOL_THRESHOLD}: {reason}\n"
            f"Expected tool: {expected_tool.name}({expected_tool.input_parameters})\n"
            f"Final response: {final_content[:300]}"
        )
    except AssertionError:
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _collector.add(
            scenario_id=scenario["id"],
            scenario_type="strict",
            metric_name="ToolCorrectnessMetric",
            threshold=STRICT_TOOL_THRESHOLD,
            score=score,
            passed=passed,
            reason=reason,
            error=error,
            expected_tool=_tool_call_to_dict(expected_tool),
            actual_tool=_tool_call_to_dict(actual_tool),
            final_response_preview=final_content[:300],
        )


# ---------------------------------------------------------------------------
# Loose single-intent scenarios — issue documents the flow but does not
# specify an "Expected endpoint" (S4, both languages). Scored with
# ArgumentCorrectnessMetric (LLM judges arg correctness vs. the
# conversation input).
# ---------------------------------------------------------------------------


LOOSE_SINGLE_INTENT_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "S4-parliament-attendance-multi-turn-EN",
        "label": "s4-en",
        "turns": [
            {
                "user": (
                    "Can you show me the parliament attendance of former "
                    "Finance Minister Martin Helme?"
                )
            },
            {"user": "Can you just check with what you have?"},
            {"user": "2026-04-01"},
            {"user": "Yes"},
            {"user": "2026-04-20"},
        ],
    },
    {
        "id": "S4-parliament-attendance-multi-turn-ET",
        "label": "s4-et",
        "turns": [
            {
                "user": (
                    "Kas saaksite mulle näidata endise rahandusministri "
                    "Martin Helme parlamendi kohaloleku andmeid?"
                )
            },
            {"user": "Kas saaksite lihtsalt oma andmetest järele vaadata?"},
            {"user": "2026-04-01"},
            {"user": "Jah"},
            {"user": "2026-04-20"},
        ],
    },
]


@pytest.mark.parametrize(
    "scenario",
    LOOSE_SINGLE_INTENT_SCENARIOS,
    ids=[s["id"] for s in LOOSE_SINGLE_INTENT_SCENARIOS],
)
def test_api_tool_loose_single_intent(
    scenario: Dict[str, Any],
    orchestration_client: Any,
    api_tool_endpoints_indexed: None,
) -> None:
    """Scenario 4 (EN/ET) — issue gives no expected resolution.

    Asserts:
    1. The final turn produced a completed JSON tool call (i.e. the agentic
       loop actually resolved, didn't fall through to RAG).
    2. The LLM judge ``ArgumentCorrectnessMetric`` is satisfied that the
       chosen tool's arguments fit the conversation input.
    """
    del api_tool_endpoints_indexed  # consumed for its setup side effect only

    final_content = ""
    actual_tool: Optional[ToolCall] = None
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""

    try:
        final_content = _walk_turns(
            orchestration_client.base_url, scenario["label"], scenario["turns"]
        )
        actual_tool = _to_tool_call(final_content)

        assert actual_tool is not None, (
            f"[{scenario['id']}] expected a completed JSON tool call on the "
            f"final turn but got: {final_content[:300]}"
        )

        test_case = LLMTestCase(
            input=_conversation_text(scenario["turns"]),
            actual_output=final_content,
            tools_called=[actual_tool],
        )
        metric = ArgumentCorrectnessMetric(threshold=JUDGE_THRESHOLD)
        metric.measure(test_case)
        score = metric.score
        reason = metric.reason or ""
        passed = score is not None and score >= JUDGE_THRESHOLD
        assert passed, (
            f"[{scenario['id']}] argument correctness {score} < "
            f"{JUDGE_THRESHOLD}: {reason}\n"
            f"Resolved tool: {actual_tool.name}({actual_tool.input_parameters})"
        )
    except AssertionError:
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _collector.add(
            scenario_id=scenario["id"],
            scenario_type="loose",
            metric_name="ArgumentCorrectnessMetric",
            threshold=JUDGE_THRESHOLD,
            score=score,
            passed=passed,
            reason=reason,
            error=error,
            actual_tool=_tool_call_to_dict(actual_tool),
            final_response_preview=final_content[:300],
        )


# ---------------------------------------------------------------------------
# Multi-Intent scenarios (issue #447, MI-1..MI-8)
#
# Issue lists 8 queries (EN + ET) with no expected resolution. Under Phase 1
# the orchestrator decomposes the query and falls back to a single endpoint
# (see tests/api_tool_eval/integration_test_multi_intent.py docstring). We:
#
# * If the system resolves a tool call → score with ArgumentCorrectnessMetric
#   (LLM judges whether the chosen args fit the multi-intent query).
# * If the system asks a clarifying question → accept it (still ATC-routed),
#   only assert the reply is non-empty (rules out a silent RAG/OOD fallthrough).
# ---------------------------------------------------------------------------


MULTI_INTENT_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "MI-1-address-and-vehicle-tax",
        "label": "mi1",
        "query_en": (
            "Can you find an address for me and also calculate my vehicle "
            "tax? (Address: Viru tn 4, Tallinn / Plate: 123ABC / Year: 2026)"
        ),
        "query_et": (
            "Kas saaksite mulle aadressi leida ja arvutada ka mu sõiduki "
            "maksu? (Aadress: Viru tn 4, Tallinn / Registreerimismärk: "
            "123ABC / Aasta: 2026)"
        ),
    },
    {
        "id": "MI-2-address-and-initiative-details",
        "label": "mi2",
        "query_en": (
            "I need to find an address and also check details of an "
            "initiative. (Address: Viru tn 4, Tallinn / Initiative ID: 1790)"
        ),
        "query_et": (
            "Mul on vaja leida aadress ja vaadata ka kodanikualgatuse "
            "üksikasju. (Aadress: Viru tn 4, Tallinn / Algatuse ID: 1790)"
        ),
    },
    {
        "id": "MI-3-electricity-and-public-holidays",
        "label": "mi3",
        "query_en": (
            "Show me electricity prices in Estonia and list Estonia's public holidays."
        ),
        "query_et": ("Näita mulle Eesti elektrihindu ja too välja Eesti riigipühad."),
    },
    {
        "id": "MI-4-parliament-votings-and-initiatives",
        "label": "mi4",
        "query_en": (
            "Show me the parliament voting results and also list the "
            "citizen initiatives."
        ),
        "query_et": (
            "Näita mulle parlamendi hääletustulemusi ja too välja ka kodanikualgatused."
        ),
    },
    {
        "id": "MI-5-address-and-parliament-participation",
        "label": "mi5",
        "query_en": (
            "Could you find me an address and also show the attendance "
            "statistics of Riigikogu members?"
        ),
        "query_et": (
            "Kas saaksite mulle leida aadressi ja näidata ka Riigikogu "
            "liikmete osalusstatistikat?"
        ),
    },
    {
        "id": "MI-6-public-holidays-and-vehicle-tax",
        "label": "mi6",
        "query_en": (
            "What are the public holidays in Estonia and can you also "
            "calculate my vehicle tax?"
        ),
        "query_et": (
            "Millised on riigipühad Eestis ja kas saate arvutada ka mu sõiduki maksu?"
        ),
    },
    {
        "id": "MI-7-initiatives-and-parliament-votings",
        "label": "mi7",
        "query_en": (
            "Show me all citizen initiatives and also display the parliament "
            "voting results."
        ),
        "query_et": (
            "Kuva mulle kõik kodanikualgatused ja näita ka riigikogu hääletustulemusi."
        ),
    },
    {
        "id": "MI-8-vehicle-tax-and-electricity",
        "label": "mi8",
        "query_en": (
            "Calculate my vehicle tax and also show me the electricity "
            "market price in Estonia."
        ),
        "query_et": (
            "Arvuta mu sõiduki maks ja näita mulle ka Eesti elektri turuhinda."
        ),
    },
]


@pytest.mark.parametrize(
    "scenario",
    MULTI_INTENT_SCENARIOS,
    ids=[s["id"] for s in MULTI_INTENT_SCENARIOS],
)
@pytest.mark.parametrize("lang", ["en", "et"])
def test_api_tool_multi_intent(
    scenario: Dict[str, Any],
    lang: str,
    orchestration_client: Any,
    api_tool_endpoints_indexed: None,
) -> None:
    del api_tool_endpoints_indexed  # consumed for its setup side effect only

    query = scenario[f"query_{lang}"]
    content = ""
    actual_tool: Optional[ToolCall] = None
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""
    outcome = "unknown"  # "tool_call" | "clarifying_question"

    try:
        base_url = orchestration_client.base_url
        chat_id = _make_chat_id(f"{scenario['label']}-{lang}")
        resp = _send_turn(base_url, chat_id, query, [])
        content = resp.get("content", "")
        actual_tool = _to_tool_call(content)

        if actual_tool is not None:
            outcome = "tool_call"
            test_case = LLMTestCase(
                input=query,
                actual_output=content,
                tools_called=[actual_tool],
            )
            metric = ArgumentCorrectnessMetric(threshold=JUDGE_THRESHOLD)
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason or ""
            passed = score is not None and score >= JUDGE_THRESHOLD
            assert passed, (
                f"[{scenario['id']} {lang}] argument correctness {score} "
                f"< {JUDGE_THRESHOLD}: {reason}\n"
                f"Resolved tool: {actual_tool.name}({actual_tool.input_parameters})"
            )
        else:
            outcome = "clarifying_question"
            # No tool call yet — must at least be a non-empty clarifying reply
            # (rules out silent failure / RAG fallthrough returning no content).
            passed = bool(content.strip())
            assert passed, (
                f"[{scenario['id']} {lang}] empty response — multi-intent query "
                f"failed routing entirely. Got: {content!r}"
            )
            reason = "Resolved to a clarifying question (no tool call yet)"
    except AssertionError:
        raise
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _collector.add(
            scenario_id=f"{scenario['id']}-{lang}",
            scenario_type="multi_intent",
            metric_name=(
                "ArgumentCorrectnessMetric" if outcome == "tool_call" else "RoutingOnly"
            ),
            threshold=JUDGE_THRESHOLD if outcome == "tool_call" else 0.0,
            score=score,
            passed=passed,
            reason=reason,
            error=error,
            actual_tool=_tool_call_to_dict(actual_tool),
            final_response_preview=content[:300],
            extra={"language": lang, "outcome": outcome, "query": query},
        )

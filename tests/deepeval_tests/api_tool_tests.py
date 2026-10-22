"""
DeepEval tests for the API Tool Calling feature (issue #447).

Covers the single-intent (Scenarios 1-4) and multi-intent (MI-1..MI-8) scenarios
listed in the issue against the running orchestration service. Each scenario
walks the ``/orchestrate-eval`` endpoint turn-by-turn via the
testcontainers-backed ``orchestration_client`` fixture. When the agentic loop
completes, the service (running with ``EVAL_MODE=true``) attaches the resolved
external call(s) to the response as ``tool_calls`` — name, method, url,
input_parameters, success, status_code — so the tests score the *resolved call*
rather than parsing the natural-language answer. Each scenario is scored with a
DeepEval agentic metric:

* **Strict single-intent (S1, S2a, S2b, S3)** — the issue specifies an
  "Expected endpoint" + URL/params per scenario. Scored with
  ``ToolCorrectnessMetric`` (deterministic name + input-parameter comparison,
  ``threshold=1.0``). The called tool's parameters are restricted to the
  expected keys before scoring, so optional extras the loop also collected
  (e.g. ``languageIsoCode``) do not fail the scenario.

* **Loose single-intent (S4)** — the issue documents the 5-turn flow but does
  not specify an expected resolution. Scored with
  ``ArgumentCorrectnessMetric`` (LLM-as-judge over the conversation input and
  the resolved tool call).

* **Multi-intent (MI-1..MI-8)** — issue lists only queries (EN+ET). If the
  system resolves tool call(s), scored with ``ArgumentCorrectnessMetric``; if
  it asks a clarifying question instead, the test verifies via the eval-only
  ``workflow`` field that the API Tool Calling workflow answered (a silent
  fall-through to CONTEXT/RAG/OOD fails).

Endpoints are matched by ``name`` — the UUIDs in the issue do happen to match
those in ``tests/api_tool_eval/test-endpoints.json``, but name matching is the
stable contract.

Depends on:
* ``orchestration_client`` — provides the testcontainers-mapped base URL.
* ``api_tool_endpoints_indexed`` — seeds the API tool fixture into Qdrant's
  ``api_tool_collection`` so the agentic loop can find the endpoints.
* The eval stack (``docker-compose-eval.yml``) running with
  ``TOOL_CLASSIFIER_ENABLED=true`` and ``EVAL_MODE=true``.
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

# A single turn can take well over a minute: agentic-loop LLM calls, the external
# API call, LLM response formatting and the output-guardrail check all run in
# sequence (a 60 s limit produced a spurious ReadTimeout in a real run).
REQUEST_TIMEOUT = 180
# Must match the connection seeded into Vault by the ``rag_stack`` fixture in
# conftest.py (``llm/connections/azure_openai/evalconnection-1``) and the values
# used by standard_tests.py. Vault paths are keyed by connection_id only, so a
# missing connection_id makes the LLM manager raise before any workflow runs.
ENVIRONMENT = "testing"
CONNECTION_ID = "evalconnection-1"
AUTHOR_ID = "api-tool-deepeval"

# Where the result-collector writes the per-scenario record consumed by
# tests/deepeval_tests/api_tool_report_generator.py to render the PR
# comment / artifact markdown.
RESULTS_FILE = Path("api_tool_test_results.json")

# Strict scenarios assert the expected tool was called and every expected
# param is present with the expected value. ToolCorrectnessMetric scores
# fractionally over the parameter dict *including* extras (an extra key
# lowers the score below 1.0), so the called tool's params are first
# restricted to the expected keys — see _restrict_to_expected(). Threshold
# 1.0 then means: right tool, all expected params correct.
STRICT_TOOL_THRESHOLD = 1.0

# Loose scenarios are graded by an LLM judge — 0.7 matches the threshold used
# for the RAG metrics in standard_tests.py.
JUDGE_THRESHOLD = 0.7

# Value of the eval-only ``workflow`` response field when the API Tool Calling
# workflow produced the answer (tool_classifier.enums.WorkflowType.API_TOOL_CALLING).
API_TOOL_WORKFLOW = "api_tool_calling"


# ---------------------------------------------------------------------------
# HTTP helpers
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
        "connection_id": CONNECTION_ID,
    }
    # /orchestrate-eval runs the same pipeline as /orchestrate but, under
    # EVAL_MODE=true, also returns eval-only fields (``tool_calls``).
    resp = requests.post(
        f"{base_url}/orchestrate-eval",
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


def _tool_calls_from_response(resp: Dict[str, Any]) -> List[ToolCall]:
    """Build DeepEval ToolCalls from the eval-only ``tool_calls`` field.

    Empty when the turn did not complete a tool call (clarifying question,
    fallback to RAG/OOD, or the service is not running in EVAL_MODE).
    """
    records = resp.get("tool_calls") or []
    return [
        ToolCall(
            name=str(rec.get("name", "")),
            input_parameters=dict(rec.get("input_parameters") or {}),
        )
        for rec in records
        if isinstance(rec, dict)
    ]


def _restrict_to_expected(called: ToolCall, expected: ToolCall) -> ToolCall:
    """Return ``called`` with input_parameters limited to the keys in ``expected``.

    ToolCorrectnessMetric compares parameter dicts key-by-key over the *union*
    of keys, so an optional parameter the loop also collected would lower the
    score even when every expected parameter is correct. Dropping the extras
    makes the metric express "all expected params present and correct".
    Missing or wrong expected params still fail (they are not added).
    """
    if called.name != expected.name:
        return called
    expected_keys = set((expected.input_parameters or {}).keys())
    params = {
        k: v for k, v in (called.input_parameters or {}).items() if k in expected_keys
    }
    return ToolCall(name=called.name, input_parameters=params)


def _conversation_text(turns: List[Dict[str, Any]]) -> str:
    """Flatten the user turns into a single string for LLMTestCase.input.

    DeepEval's agentic single-turn metrics take a single ``input`` string;
    this approximation gives the LLM judge the full conversational context.
    """
    return "\n".join(f"USER: {t['user']}" for t in turns)


def _run_turn(
    base_url: str,
    chat_id: str,
    message: str,
    history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Send one turn and return a self-contained record of it.

    The record (``user``, ``bot``, ``workflow``, ``tool_calls``,
    ``input_guard_failed``, ``latency_s``) is what the results JSON stores for
    every turn so failures can be diagnosed from the report alone. The raw
    response is kept under ``raw`` for the metric code.
    """
    started = time.monotonic()
    resp = _send_turn(base_url, chat_id, message, history)
    return {
        "user": message,
        "bot": resp.get("content", ""),
        "workflow": resp.get("workflow"),
        "tool_calls": [
            _tool_call_to_dict(tc) for tc in _tool_calls_from_response(resp)
        ],
        "input_guard_failed": bool(resp.get("inputGuardFailed", False)),
        "latency_s": round(time.monotonic() - started, 1),
        "raw": resp,
    }


def _walk_turns(
    base_url: str, label: str, turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """POST each turn in sequence, maintaining a stable chatId + history.
    Returns one record per turn (see :func:`_run_turn`); the last one is the
    final answer."""
    chat_id = _make_chat_id(label)
    history: List[Dict[str, str]] = []
    records: List[Dict[str, Any]] = []
    for turn in turns:
        rec = _run_turn(base_url, chat_id, turn["user"], history)
        records.append(rec)
        history = _append_history(history, turn["user"], rec["bot"])
    return records


def _transcript(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn records without the raw response, for the results JSON."""
    return [{k: v for k, v in rec.items() if k != "raw"} for rec in records]


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
        turns: Optional[List[Dict[str, Any]]] = None,
        input_guard_failed: Optional[bool] = None,
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
                "input_guard_failed": input_guard_failed,
                "turns": turns or [],
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
    workflow: Optional[str] = None
    tool_calls: List[ToolCall] = []
    actual_tool: Optional[ToolCall] = None
    records: List[Dict[str, Any]] = []
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""

    try:
        records = _walk_turns(
            orchestration_client.base_url, scenario["label"], scenario["turns"]
        )
        final_resp = records[-1]["raw"]
        final_content = final_resp.get("content", "")
        workflow = final_resp.get("workflow")
        tool_calls = _tool_calls_from_response(final_resp)
        actual_tool = tool_calls[0] if tool_calls else None
        test_case = LLMTestCase(
            input=_conversation_text(scenario["turns"]),
            actual_output=final_content,
            tools_called=[
                _restrict_to_expected(tc, expected_tool) for tc in tool_calls
            ],
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
            f"Resolved tool calls: {[_tool_call_to_dict(tc) for tc in tool_calls]}\n"
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
            turns=_transcript(records),
            input_guard_failed=records[-1]["input_guard_failed"] if records else None,
            extra={
                "workflow": workflow,
                "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
            },
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
    1. The final turn resolved at least one tool call (``tool_calls`` is
       non-empty), i.e. the agentic loop actually completed and did not fall
       through to RAG or stall on a clarifying question.
    2. The LLM judge ``ArgumentCorrectnessMetric`` is satisfied that the
       chosen tool's arguments fit the conversation input.
    """
    del api_tool_endpoints_indexed  # consumed for its setup side effect only

    final_content = ""
    workflow: Optional[str] = None
    tool_calls: List[ToolCall] = []
    actual_tool: Optional[ToolCall] = None
    records: List[Dict[str, Any]] = []
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""

    try:
        records = _walk_turns(
            orchestration_client.base_url, scenario["label"], scenario["turns"]
        )
        final_resp = records[-1]["raw"]
        final_content = final_resp.get("content", "")
        workflow = final_resp.get("workflow")
        tool_calls = _tool_calls_from_response(final_resp)
        actual_tool = tool_calls[0] if tool_calls else None

        assert tool_calls, (
            f"[{scenario['id']}] expected a resolved tool call on the final "
            f"turn but tool_calls was empty (workflow={workflow!r}). "
            f"Response: {final_content[:300]}"
        )

        test_case = LLMTestCase(
            input=_conversation_text(scenario["turns"]),
            actual_output=final_content,
            tools_called=tool_calls,
        )
        metric = ArgumentCorrectnessMetric(threshold=JUDGE_THRESHOLD)
        metric.measure(test_case)
        score = metric.score
        reason = metric.reason or ""
        passed = score is not None and score >= JUDGE_THRESHOLD
        assert passed, (
            f"[{scenario['id']}] argument correctness {score} < "
            f"{JUDGE_THRESHOLD}: {reason}\n"
            f"Resolved tool calls: {[_tool_call_to_dict(tc) for tc in tool_calls]}"
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
            turns=_transcript(records),
            input_guard_failed=records[-1]["input_guard_failed"] if records else None,
            extra={
                "workflow": workflow,
                "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
            },
        )


# ---------------------------------------------------------------------------
# Multi-Intent scenarios (issue #447, MI-1..MI-8)
#
# Issue lists 8 queries (EN + ET) with no expected resolution. The orchestrator
# decomposes the query; depending on the ATC phase it either collects params for
# all matched endpoints in parallel or falls back to a single endpoint. We:
#
# * If the system resolves tool call(s) → score with ArgumentCorrectnessMetric
#   (LLM judges whether the chosen args fit the multi-intent query).
# * If the input guardrail blocked the query → fail, reported as such.
# * If the system asks a clarifying question → pass only when the eval-only
#   ``workflow`` field says the API Tool Calling workflow asked it (a silent
#   CONTEXT/RAG/OOD fallthrough fails).
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
    workflow: Optional[str] = None
    tool_calls: List[ToolCall] = []
    actual_tool: Optional[ToolCall] = None
    records: List[Dict[str, Any]] = []
    score: Optional[float] = None
    reason = ""
    passed = False
    error = ""
    outcome = "unknown"  # "tool_call" | "clarifying_question" | "input_guard_blocked"

    try:
        base_url = orchestration_client.base_url
        chat_id = _make_chat_id(f"{scenario['label']}-{lang}")
        records = [_run_turn(base_url, chat_id, query, [])]
        resp = records[-1]["raw"]
        content = resp.get("content", "")
        workflow = resp.get("workflow")
        tool_calls = _tool_calls_from_response(resp)
        actual_tool = tool_calls[0] if tool_calls else None

        if tool_calls:
            # Parallel multi-intent execution yields one record per endpoint;
            # the judge sees all of them.
            outcome = "tool_call"
            test_case = LLMTestCase(
                input=query,
                actual_output=content,
                tools_called=tool_calls,
            )
            metric = ArgumentCorrectnessMetric(threshold=JUDGE_THRESHOLD)
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason or ""
            passed = score is not None and score >= JUDGE_THRESHOLD
            assert passed, (
                f"[{scenario['id']} {lang}] argument correctness {score} "
                f"< {JUDGE_THRESHOLD}: {reason}\n"
                f"Resolved tool calls: {[_tool_call_to_dict(tc) for tc in tool_calls]}"
            )
        elif records[-1]["input_guard_failed"]:
            # Blocked before classification ever ran — a guardrail finding,
            # reported distinctly from a routing failure.
            outcome = "input_guard_blocked"
            passed = False
            reason = "Input guardrail blocked the query before classification"
            raise AssertionError(
                f"[{scenario['id']} {lang}] input guardrail blocked the query; "
                f"the API Tool Calling workflow never ran. Got: {content[:300]!r}"
            )
        else:
            outcome = "clarifying_question"
            # No tool call yet — acceptable only if the API Tool Calling
            # workflow is the one asking. ``workflow`` is the eval-only field
            # set by ToolClassifier under EVAL_MODE; a CONTEXT/RAG/OOD answer
            # here means the multi-intent query was misrouted.
            passed = workflow == API_TOOL_WORKFLOW and bool(content.strip())
            assert passed, (
                f"[{scenario['id']} {lang}] expected the API Tool Calling "
                f"workflow to answer (tool call or clarifying question) but "
                f"workflow={workflow!r}. Got: {content[:300]!r}"
            )
            reason = (
                "Resolved to a clarifying question from the API Tool Calling "
                "workflow (no tool call yet)"
            )
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
            turns=_transcript(records),
            input_guard_failed=records[-1]["input_guard_failed"] if records else None,
            extra={
                "language": lang,
                "outcome": outcome,
                "query": query,
                "workflow": workflow,
                "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
            },
        )

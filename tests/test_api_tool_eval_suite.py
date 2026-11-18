"""Offline unit tests for the API-tool deepeval suite's own logic.

``tests/deepeval_tests/api_tool_tests.py`` normally runs against the full
``docker-compose-eval.yml`` stack with Azure secrets. These tests load that
module by path and drive its helpers, metric wiring, result collector and
report generator against an in-process stub of ``/orchestrate-eval``, so the
suite's *own* behaviour is checked in CI in well under a second:

* ``tool_calls`` / ``workflow`` / ``inputGuardFailed`` are read correctly;
* ``ToolCorrectnessMetric`` is fed the parameters it needs (extras tolerated,
  missing/wrong params still fail) — the deepeval behaviour that motivated
  ``_restrict_to_expected``;
* strict / loose / multi-intent verdicts and their failure messages;
* every turn is recorded and rendered as a transcript in the report.

Only deterministic paths are exercised; ``ArgumentCorrectnessMetric`` (LLM
judge) is never invoked.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Dict, List

import pytest

# Must be set before deepeval is imported (it is imported by the suite module).
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

_DEEPEVAL_DIR = Path(__file__).parent / "deepeval_tests"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"_api_tool_eval_{name}", _DEEPEVAL_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


suite = _load("api_tool_tests")
report = _load("api_tool_report_generator")

ToolCall = suite.ToolCall


# ---------------------------------------------------------------------------
# Stub /orchestrate-eval
# ---------------------------------------------------------------------------

# A script maps (label, turn_number) -> partial response overrides.
Script = Callable[[str, int, Dict[str, Any]], Dict[str, Any]]


def _base_response(chat_id: str) -> Dict[str, Any]:
    return {
        "chatId": chat_id,
        "llmServiceActive": True,
        "questionOutOfLLMScope": False,
        "inputGuardFailed": False,
        "content": "Which value do you mean?",
        "tool_calls": None,
        "workflow": suite.API_TOOL_WORKFLOW,
    }


def _tool_call(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "method": "GET",
        "url": "https://stub.example/" + name,
        "input_parameters": params,
        "success": True,
        "status_code": 200,
        "cache_source": None,
    }


class _Stub:
    """Threaded HTTP stub; ``script`` is swapped per test."""

    def __init__(self) -> None:
        self.script: Script = lambda label, turn, body: {}
        self.requests: List[Dict[str, Any]] = []
        self._turns: Dict[str, int] = {}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests.append({"path": self.path, "body": body})
                chat_id = body["chatId"]
                match = re.match(r"deepeval-api-tool-(.+)-[0-9a-f]{8}$", chat_id)
                label = match.group(1) if match else chat_id
                stub._turns[chat_id] = stub._turns.get(chat_id, 0) + 1
                resp = _base_response(chat_id)
                resp.update(stub.script(label, stub._turns[chat_id], body))
                out = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture(scope="module")
def stub():
    s = _Stub()
    yield s
    s.close()


@pytest.fixture
def client(stub):
    """Stands in for the ``orchestration_client`` fixture of the real suite."""
    stub.requests.clear()
    return SimpleNamespace(base_url=stub.base_url)


@pytest.fixture(autouse=True)
def fresh_collector():
    """Isolate the suite's module-level result collector per test."""
    suite._collector = suite.ApiToolResultCollector()
    yield suite._collector


def _scenario(group: List[Dict[str, Any]], scenario_id: str) -> Dict[str, Any]:
    return next(s for s in group if s["id"] == scenario_id)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_tool_calls_from_response_tolerates_missing_and_malformed():
    assert suite._tool_calls_from_response({}) == []
    assert suite._tool_calls_from_response({"tool_calls": None}) == []
    calls = suite._tool_calls_from_response(
        {"tool_calls": ["junk", {"name": "x", "input_parameters": None}, {"name": "y"}]}
    )
    assert [(c.name, c.input_parameters) for c in calls] == [("x", {}), ("y", {})]


def test_restrict_to_expected_drops_extras_but_never_adds():
    expected = ToolCall(name="t", input_parameters={"a": "1", "b": "2"})
    restricted = suite._restrict_to_expected(
        ToolCall(name="t", input_parameters={"a": "1", "extra": "z"}), expected
    )
    assert restricted.input_parameters == {"a": "1"}  # 'b' still missing
    other = ToolCall(name="other", input_parameters={"a": "1"})
    assert suite._restrict_to_expected(other, expected) is other


def test_tool_correctness_metric_needs_the_restriction():
    """Documents why _restrict_to_expected exists: deepeval scores extras down."""
    expected = ToolCall(name="t", input_parameters={"a": "1"})
    called = ToolCall(name="t", input_parameters={"a": "1", "extra": "z"})

    def score(tools_called: List[Any]) -> float:
        case = suite.LLMTestCase(
            input="q",
            actual_output="a",
            tools_called=tools_called,
            expected_tools=[expected],
        )
        metric = suite.ToolCorrectnessMetric(
            threshold=1.0, evaluation_params=[suite.ToolCallParams.INPUT_PARAMETERS]
        )
        metric.measure(case)
        return metric.score or 0.0

    assert score([called]) < 1.0
    assert score([suite._restrict_to_expected(called, expected)]) == 1.0
    assert score([]) == 0.0


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def test_run_turn_hits_eval_endpoint_with_connection_and_records_turn(stub, client):
    stub.script = lambda label, turn, body: {
        "content": "done",
        "tool_calls": [_tool_call("search_address", {"address": "Viru tn 4"})],
    }

    rec = suite._run_turn(client.base_url, "chat-x", "hello", [])

    req = stub.requests[-1]
    assert req["path"] == "/orchestrate-eval"
    assert req["body"]["connection_id"] == suite.CONNECTION_ID
    assert req["body"]["environment"] == suite.ENVIRONMENT
    assert rec["user"] == "hello" and rec["bot"] == "done"
    assert rec["workflow"] == suite.API_TOOL_WORKFLOW
    assert rec["tool_calls"] == [
        {"name": "search_address", "input_parameters": {"address": "Viru tn 4"}}
    ]
    assert rec["input_guard_failed"] is False
    assert isinstance(rec["latency_s"], float)


def test_walk_turns_threads_history(stub, client):
    stub.script = lambda label, turn, body: {"content": f"reply {turn}"}
    turns = [{"user": "one"}, {"user": "two"}, {"user": "three"}]

    records = suite._walk_turns(client.base_url, "lbl", turns)

    assert [r["bot"] for r in records] == ["reply 1", "reply 2", "reply 3"]
    last_history = stub.requests[-1]["body"]["conversationHistory"]
    assert [h["message"] for h in last_history] == ["one", "reply 1", "two", "reply 2"]


# ---------------------------------------------------------------------------
# Strict scenarios (ToolCorrectnessMetric, deterministic)
# ---------------------------------------------------------------------------


def test_strict_passes_with_extra_optional_param(stub, client, fresh_collector):
    scenario = _scenario(
        suite.STRICT_SINGLE_INTENT_SCENARIOS, "S1-citizen-initiative-EN"
    )
    expected = scenario["expected_tool"]

    def script(label: str, turn: int, body: Dict[str, Any]) -> Dict[str, Any]:
        if turn < len(scenario["turns"]):
            return {"content": "Which initiative id?"}
        params = {**expected.input_parameters, "languageIsoCode": "EN"}
        return {
            "content": "Here it is.",
            "tool_calls": [_tool_call(expected.name, params)],
        }

    stub.script = script
    suite.test_api_tool_strict_single_intent(scenario, client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert record["passed"] is True and record["score"] == 1.0
    assert [t["workflow"] for t in record["turns"]] == [suite.API_TOOL_WORKFLOW] * 2
    assert (
        record["turns"][-1]["tool_calls"][0]["input_parameters"]["languageIsoCode"]
        == "EN"
    )


def test_strict_fails_on_wrong_tool_and_records_transcript(
    stub, client, fresh_collector
):
    scenario = _scenario(suite.STRICT_SINGLE_INTENT_SCENARIOS, "S3-intent-switch-EN")
    stub.script = lambda label, turn, body: (
        {"content": "From when?"}
        if turn < 2
        else {
            "content": "Prices...",
            "tool_calls": [_tool_call("get_electricity_prices", {"start": "x"})],
        }
    )

    with pytest.raises(AssertionError, match="Expected tool: search_address"):
        suite.test_api_tool_strict_single_intent(scenario, client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert record["passed"] is False and record["score"] == 0.0
    assert record["actual_tool"]["name"] == "get_electricity_prices"
    assert len(record["turns"]) == 2


def test_strict_fails_on_uncorrected_param(stub, client, fresh_collector):
    scenario = _scenario(
        suite.STRICT_SINGLE_INTENT_SCENARIOS,
        "S2b-parliament-votings-date-correction-EN",
    )
    stub.script = lambda label, turn, body: (
        {"content": "Dates?"}
        if turn < 3
        else {
            "content": "Votes...",
            "tool_calls": [
                _tool_call(
                    "get_parliament_votings",
                    {
                        "startDate": "2026-04-05",
                        "endDate": "2026-04-07",
                    },  # start not corrected
                )
            ],
        }
    )

    with pytest.raises(AssertionError, match="tool correctness 0.5 < 1.0"):
        suite.test_api_tool_strict_single_intent(scenario, client, None)


# ---------------------------------------------------------------------------
# Loose scenario — only the deterministic precondition
# ---------------------------------------------------------------------------


def test_loose_fails_clearly_when_nothing_was_resolved(stub, client, fresh_collector):
    scenario = suite.LOOSE_SINGLE_INTENT_SCENARIOS[0]
    stub.script = lambda label, turn, body: {"content": "Still need more info."}

    with pytest.raises(AssertionError, match="tool_calls was empty"):
        suite.test_api_tool_loose_single_intent(scenario, client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert len(record["turns"]) == len(scenario["turns"])
    assert record["metric"] == "ArgumentCorrectnessMetric" and record["score"] is None


# ---------------------------------------------------------------------------
# Multi-intent — routing verdicts
# ---------------------------------------------------------------------------


def test_multi_intent_clarifying_question_from_atc_passes(
    stub, client, fresh_collector
):
    scenario = suite.MULTI_INTENT_SCENARIOS[2]  # MI-3, no params in the query
    stub.script = lambda label, turn, body: {"content": "Which dates?"}

    suite.test_api_tool_multi_intent(scenario, "en", client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert record["passed"] is True
    assert record["extra"]["outcome"] == "clarifying_question"
    assert record["metric"] == "RoutingOnly"


def test_multi_intent_rag_fallthrough_fails(stub, client, fresh_collector):
    scenario = suite.MULTI_INTENT_SCENARIOS[6]  # MI-7
    stub.script = lambda label, turn, body: {
        "content": "From the knowledge base...",
        "workflow": "rag",
    }

    with pytest.raises(AssertionError, match="workflow='rag'"):
        suite.test_api_tool_multi_intent(scenario, "et", client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert record["extra"]["workflow"] == "rag" and record["passed"] is False


def test_multi_intent_input_guard_block_is_reported_distinctly(
    stub, client, fresh_collector
):
    scenario = suite.MULTI_INTENT_SCENARIOS[0]  # MI-1
    stub.script = lambda label, turn, body: {
        "content": "I cannot assist with that request.",
        "inputGuardFailed": True,
        "workflow": None,
    }

    with pytest.raises(AssertionError, match="input guardrail blocked"):
        suite.test_api_tool_multi_intent(scenario, "en", client, None)

    (record,) = fresh_collector.results["scenarios"]
    assert record["extra"]["outcome"] == "input_guard_blocked"
    assert record["input_guard_failed"] is True
    assert record["reason"].startswith("Input guardrail blocked")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_renders_transcript_and_escapes_pipes(stub, client, fresh_collector):
    scenario = _scenario(suite.STRICT_SINGLE_INTENT_SCENARIOS, "S3-intent-switch-EN")
    stub.script = lambda label, turn, body: (
        {"content": "From when?"}
        if turn < 2
        else {
            "content": "Prices | table",
            "tool_calls": [_tool_call("get_electricity_prices", {"start": "a|b"})],
        }
    )
    with pytest.raises(AssertionError):
        suite.test_api_tool_strict_single_intent(scenario, client, None)

    md = report.render_report(fresh_collector.results)

    assert "### Failed scenarios" in md and "- **Turns:**" in md
    assert "| 2 |" in md  # second turn rendered
    assert "a\\|b" in md and "Prices \\| table" in md  # pipes escaped in cells
    assert "**Workflow:** `api_tool_calling`" in md
    assert "Total scenarios | 1" in md


def test_report_handles_missing_results_file(tmp_path):
    result = report.load_results(tmp_path / "nope.json")
    assert "error" in result
    assert "ERROR" in report.render_report(result)

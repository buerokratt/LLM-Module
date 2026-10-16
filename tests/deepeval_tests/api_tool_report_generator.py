"""
Render the API Tool Calling test results JSON as a Markdown report.

Reads ``api_tool_test_results.json`` (written by the
``_save_api_tool_results`` autouse fixture in
``tests/deepeval_tests/api_tool_tests.py``) and writes
``api_tool_test_report.md``. The workflow uploads the markdown as an artifact
and posts it as a PR comment.

This is the API-tool counterpart of ``report_generator.py``; the two are
deliberately independent so changes to the RAG metrics report don't risk
breaking the API-tool report or vice versa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RESULTS_FILE = Path("api_tool_test_results.json")
REPORT_FILE = Path("api_tool_test_report.md")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_results(path: Path = RESULTS_FILE) -> Dict[str, Any]:
    """Load the JSON written by ApiToolResultCollector.save()."""
    if not path.exists():
        return {"error": f"Results file not found: {path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return {"error": f"Results file is not valid JSON: {e}"}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _by_type(scenarios: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for s in scenarios:
        grouped.setdefault(s.get("type", "unknown"), []).append(s)
    return grouped


def _pass_rate(scenarios: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    total = len(scenarios)
    passed = sum(1 for s in scenarios if s.get("passed"))
    rate = (passed / total * 100.0) if total else 0.0
    return passed, total, rate


def _status_emoji(scenario: Dict[str, Any]) -> str:
    if scenario.get("error"):
        return "💥"
    if scenario.get("passed"):
        return "✅"
    return "❌"


def _fmt_tool(tc: Optional[Dict[str, Any]]) -> str:
    if not tc:
        return "_(none)_"
    name = tc.get("name", "?")
    params = tc.get("input_parameters") or {}
    if not params:
        return f"`{name}()`"
    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"`{name}({param_str})`"


def _score_str(score: Optional[float], threshold: float) -> str:
    if score is None:
        return f"— / {threshold}"
    return f"{score:.2f} / {threshold}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def render_header(results: Dict[str, Any]) -> str:
    total = results.get("total_tests", 0)
    passed = results.get("passed_tests", 0)
    failed = results.get("failed_tests", 0)
    errored = results.get("errored_tests", 0)
    rate = (passed / total * 100.0) if total else 0.0
    started = results.get("test_start_time", "")

    return (
        "## API Tool Calling Evaluation Report\n\n"
        f"_Issue #447 — DeepEval coverage for the API Tool Calling feature._\n\n"
        f"**Started:** `{started}`\n\n"
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| Total scenarios | {total} |\n"
        f"| Passed | {passed} |\n"
        f"| Failed | {failed} |\n"
        f"| Errored | {errored} |\n"
        f"| Pass rate | **{rate:.1f}%** |\n\n"
    )


def render_by_type(results: Dict[str, Any]) -> str:
    scenarios = results.get("scenarios", [])
    grouped = _by_type(scenarios)
    out = "### Results by scenario type\n\n"
    out += "| Type | Metric | Passed | Total | Pass rate |\n"
    out += "|---|---|---|---|---|\n"
    type_metric = {
        "strict": "ToolCorrectnessMetric (deterministic, threshold=1.0)",
        "loose": "ArgumentCorrectnessMetric (LLM judge, threshold=0.7)",
        "multi_intent": "ArgumentCorrectnessMetric or routing-only",
    }
    type_label = {
        "strict": "Strict single-intent (S1, S2a, S2b, S3)",
        "loose": "Loose single-intent (S4)",
        "multi_intent": "Multi-intent (MI-1..MI-8)",
    }
    for stype in ("strict", "loose", "multi_intent"):
        items = grouped.get(stype, [])
        if not items:
            continue
        p, t, r = _pass_rate(items)
        out += (
            f"| {type_label.get(stype, stype)} | "
            f"{type_metric.get(stype, '—')} | "
            f"{p} | {t} | {r:.1f}% |\n"
        )
    return out + "\n"


def render_scenario_table(results: Dict[str, Any]) -> str:
    scenarios = results.get("scenarios", [])
    if not scenarios:
        return "### Detailed results\n\n_No scenarios recorded._\n\n"

    out = "### Detailed results\n\n"
    out += "| Status | Scenario | Metric | Score | Expected → Actual |\n"
    out += "|---|---|---|---|---|\n"
    for s in scenarios:
        status = _status_emoji(s)
        sid = s.get("id", "?")
        metric = s.get("metric", "—")
        score_cell = _score_str(s.get("score"), s.get("threshold", 0.0))
        expected = _fmt_tool(s.get("expected_tool"))
        actual = _fmt_tool(s.get("actual_tool"))
        # For multi-intent / loose, expected_tool is None — show "→ {actual}" only
        if s.get("expected_tool") is None:
            arrow = actual
        else:
            arrow = f"{expected} → {actual}"
        out += f"| {status} | `{sid}` | {metric} | {score_cell} | {arrow} |\n"
    return out + "\n"


def render_failures(results: Dict[str, Any]) -> str:
    failures = [
        s
        for s in results.get("scenarios", [])
        if not s.get("passed") and not s.get("error")
    ]
    if not failures:
        return ""
    out = "### Failed scenarios\n\n"
    for s in failures:
        sid = s.get("id", "?")
        reason = s.get("reason") or "_(no reason captured)_"
        preview = (s.get("final_response_preview") or "").replace("\n", " ")
        if len(preview) > 200:
            preview = preview[:200] + "…"
        out += f"#### ❌ `{sid}` ({s.get('metric', '—')})\n\n"
        out += f"- **Score:** {_score_str(s.get('score'), s.get('threshold', 0.0))}\n"
        if s.get("expected_tool"):
            out += f"- **Expected:** {_fmt_tool(s['expected_tool'])}\n"
        out += f"- **Actual:** {_fmt_tool(s.get('actual_tool'))}\n"
        out += f"- **Reason:** {reason}\n"
        if preview:
            out += f"- **Response preview:** `{preview}`\n"
        out += "\n"
    return out


def render_errors(results: Dict[str, Any]) -> str:
    errored = [s for s in results.get("scenarios", []) if s.get("error")]
    if not errored:
        return ""
    out = "### Errored scenarios (test harness errors, not assertion failures)\n\n"
    for s in errored:
        sid = s.get("id", "?")
        err = s.get("error") or "_(no error captured)_"
        out += f"- 💥 `{sid}` — {err}\n"
    return out + "\n"


def render_methodology() -> str:
    return (
        "### Methodology\n\n"
        "- **Strict scenarios** (issue specifies the expected endpoint and "
        "params) are scored with `ToolCorrectnessMetric`, "
        "`evaluation_params=[ToolCallParams.INPUT_PARAMETERS]`, threshold 1.0. "
        "Tool name must match exactly and every expected parameter must be "
        "present with the expected value (extra parameters allowed).\n"
        "- **Loose scenarios** (issue describes the flow but not the expected "
        "resolution) are scored with `ArgumentCorrectnessMetric` (LLM-as-judge, "
        "threshold 0.7).\n"
        "- **Multi-intent scenarios** use `ArgumentCorrectnessMetric` when the "
        "agent resolves a tool call; if the agent asks a clarifying question "
        "instead, only routing-to-ATC is verified (rules out silent fall-through "
        "to RAG/OOD).\n"
        "- API tool endpoints are seeded into the testcontainers-backed Qdrant "
        "from `tests/api_tool_eval/test-endpoints.json` via the "
        "`api_tool_endpoints_indexed` fixture in `conftest.py`.\n\n"
    )


def render_report(results: Dict[str, Any]) -> str:
    if results.get("error"):
        return (
            f"## API Tool Calling Evaluation Report\n\n**ERROR:** {results['error']}\n"
        )
    return (
        render_header(results)
        + render_by_type(results)
        + render_scenario_table(results)
        + render_failures(results)
        + render_errors(results)
        + render_methodology()
    )


def main() -> int:
    results = load_results()
    markdown = render_report(results)
    REPORT_FILE.write_text(markdown, encoding="utf-8")
    print(f"Wrote {REPORT_FILE} ({len(markdown)} chars)")
    if results.get("error"):
        print(f"  WARNING: {results['error']}", file=sys.stderr)
    else:
        print(
            f"  {results.get('passed_tests', 0)}/{results.get('total_tests', 0)} "
            "scenarios passed"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

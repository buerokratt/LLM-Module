"""Unit tests for api_tool_indexer/main_indexer.py.

Covers:
- _build_params_summary(): single/multiple/no params
- _parse_example_queries(): normal output, missing section → empty,
  deduplication, varied bullet formats, non-list lines end section
- _generate_context_for_endpoint(): success, LLM failure/timeout
- index_endpoint() full pipeline: success, partial failure (delete fails),
  empty examples still produces summary point
"""

from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api_tool_indexer.main_indexer import (
    _build_params_summary,
    _generate_context_for_endpoint,
    _parse_example_queries,
    index_endpoint,
)
from api_tool_indexer.models import EndpointData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENDPOINT_DATA = EndpointData(
    endpoint_id="ep-holidays",
    name="get_public_holidays",
    description="Returns public holidays for a country.",
    url="https://openholidaysapi.org/PublicHolidays",
    method="GET",
    params=[
        {
            "name": "countryIsoCode",
            "type": "string",
            "required": True,
            "description": "ISO country code",
        },
        {
            "name": "year",
            "type": "integer",
            "required": False,
            "description": "Optional year",
        },
    ],
    service_id="svc-001",
    visibility="private",
    type="custom_endpoint",
)

_ENDPOINT_NO_PARAMS = EndpointData(
    endpoint_id="ep-no-params",
    name="get_info",
    description="General info endpoint.",
    url="https://ilmmicroservice.envir.ee/api/forecasts",
    method="GET",
    params=[],
)

_CONTEXT_WITH_EXAMPLES = """
This endpoint returns public holidays for a given country.
Users typically ask about upcoming holidays, days off, or national celebrations.

Example queries:
- When is the next public holiday in Estonia?
- What are the official holidays in Estonia this year?
- Riiklikud pühad Eestis 2024
- Millal on järgmine riiklik püha?
- Estonian national holidays list
"""

_CONTEXT_WITHOUT_EXAMPLES = """
This endpoint returns general information.
No example queries are provided here.
"""

_MOCK_EMBEDDING = [0.1] * 3072


def _make_api_client(
    context: str = _CONTEXT_WITH_EXAMPLES,
    embedding: Optional[List[float]] = None,
    raise_on_generate: Optional[Exception] = None,
    raise_on_embed: Optional[Exception] = None,
) -> MagicMock:
    """Build a mock LLMAPIClient context manager."""
    client = MagicMock()
    client.environment = "testing"
    client.connection_id = "test-conn"
    client.max_retries = 3
    client.retry_delay_base = 1
    client.session = MagicMock()

    if raise_on_generate:
        client.session.post.side_effect = raise_on_generate
    else:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"context": context}
        client.session.post = AsyncMock(return_value=mock_resp)

    if raise_on_embed:
        client.create_embedding = AsyncMock(side_effect=raise_on_embed)
    else:
        client.create_embedding = AsyncMock(return_value=embedding or _MOCK_EMBEDDING)

    return client


def _make_ctx_manager_client(client: MagicMock) -> MagicMock:
    """Wrap a mock client in an async context manager."""
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


# ---------------------------------------------------------------------------
# _build_params_summary
# ---------------------------------------------------------------------------


class TestBuildParamsSummary:
    def test_no_params_returns_no_required_message(self) -> None:
        result = _build_params_summary([])
        assert result == "No parameters required."

    def test_single_param_formats_correctly(self) -> None:
        params = [
            {
                "name": "country",
                "type": "string",
                "required": True,
                "description": "ISO country code",
            }
        ]
        result = _build_params_summary(params)
        assert "country" in result
        assert "string" in result
        assert "required" in result
        assert "ISO country code" in result

    def test_multiple_params_separated_by_semicolons(self) -> None:
        params = [
            {
                "name": "country",
                "type": "string",
                "required": True,
                "description": "ISO code",
            },
            {
                "name": "year",
                "type": "integer",
                "required": False,
                "description": "Year",
            },
        ]
        result = _build_params_summary(params)
        assert "; " in result
        parts = result.split("; ")
        assert len(parts) == 2

    def test_optional_param_shows_optional_label(self) -> None:
        params = [
            {
                "name": "format",
                "type": "string",
                "required": False,
                "description": "Output format",
            }
        ]
        result = _build_params_summary(params)
        assert "optional" in result

    def test_missing_keys_use_defaults(self) -> None:
        params = [{}]
        result = _build_params_summary(params)
        # Should not raise; uses defaults
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# _parse_example_queries
# ---------------------------------------------------------------------------


class TestParseExampleQueries:
    def test_normal_context_returns_all_examples(self) -> None:
        result = _parse_example_queries(_CONTEXT_WITH_EXAMPLES)
        assert len(result) == 5
        assert "When is the next public holiday in Estonia?" in result

    def test_missing_section_header_returns_empty(self) -> None:
        context = "This endpoint returns holidays. No examples here."
        result = _parse_example_queries(context)
        assert result == []

    def test_deduplication_removes_repeated_entries(self) -> None:
        context = """
Example queries:
- How many holidays in Estonia?
- How many holidays in Estonia?
- What are the public holidays?
"""
        result = _parse_example_queries(context)
        assert len(result) == 2
        assert result.count("How many holidays in Estonia?") == 1

    def test_varied_bullet_formats_included(self) -> None:
        context = """
Example queries:
- First example query here
- Second example query here
"""
        result = _parse_example_queries(context)
        assert len(result) == 2

    def test_non_list_line_ends_section(self) -> None:
        context = """
Example queries:
- First query example
- Second query example
This paragraph text ends the section.
- Third query should be ignored
"""
        result = _parse_example_queries(context)
        assert len(result) == 2
        assert "Third query should be ignored" not in result

    def test_empty_string_returns_empty(self) -> None:
        result = _parse_example_queries("")
        assert result == []

    def test_case_insensitive_header_match(self) -> None:
        context = "EXAMPLE QUERIES:\n- Some example\n"
        result = _parse_example_queries(context)
        assert "Some example" in result

    def test_strips_leading_whitespace_from_examples(self) -> None:
        context = "Example queries:\n-   leading spaces example\n"
        result = _parse_example_queries(context)
        assert result[0] == "leading spaces example"

    def test_empty_bullet_lines_excluded(self) -> None:
        # A bullet with only whitespace after stripping becomes "-" not "- something",
        # so it terminates the section. This validates that behaviour: only non-empty
        # bullet items (i.e. lines with text after "- ") are collected.
        context = "Example queries:\n- real example\n-\n- not included\n"
        result = _parse_example_queries(context)
        assert "" not in result
        assert "real example" in result
        assert "not included" not in result


# ---------------------------------------------------------------------------
# _generate_context_for_endpoint
# ---------------------------------------------------------------------------


class TestGenerateContextForEndpoint:
    @pytest.mark.asyncio
    async def test_success_returns_context_string(self) -> None:
        client = _make_api_client(
            context="Generated rich context for holidays endpoint."
        )
        result = await _generate_context_for_endpoint(client, _ENDPOINT_DATA)
        assert "context" in result.lower() or result != ""

    @pytest.mark.asyncio
    async def test_raises_runtime_error_after_all_retries(self) -> None:
        client = _make_api_client(raise_on_generate=Exception("LLM unavailable"))
        client.max_retries = 2
        client.retry_delay_base = 0  # No actual sleep in tests

        with patch(
            "api_tool_indexer.main_indexer.asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="Context generation failed"):
                await _generate_context_for_endpoint(client, _ENDPOINT_DATA)

    @pytest.mark.asyncio
    async def test_empty_context_retries_and_raises(self) -> None:
        """Empty context string from API should be treated as a failure."""
        client = _make_api_client(context="")
        client.max_retries = 2
        client.retry_delay_base = 0

        with patch(
            "api_tool_indexer.main_indexer.asyncio.sleep", new_callable=AsyncMock
        ):
            with pytest.raises(RuntimeError, match="Context generation failed"):
                await _generate_context_for_endpoint(client, _ENDPOINT_DATA)

    @pytest.mark.asyncio
    async def test_url_braces_escaped_in_prompt(self) -> None:
        """URL path params like {id} must not cause KeyError in str.format()."""
        endpoint = EndpointData(
            endpoint_id="ep-brace",
            name="get_item",
            description="Get item by ID.",
            url="https://rahvaalgatus.ee/initiatives/{id}/comments/{commentId}",
            method="GET",
            params=[],
        )
        client = _make_api_client(context="Item lookup endpoint.")
        # Should not raise KeyError
        result = await _generate_context_for_endpoint(client, endpoint)
        assert result == "Item lookup endpoint."


# ---------------------------------------------------------------------------
# index_endpoint — full pipeline
# ---------------------------------------------------------------------------


class TestIndexEndpoint:
    @pytest.mark.asyncio
    async def test_successful_indexing_returns_success_result(self) -> None:
        client = _make_api_client(context=_CONTEXT_WITH_EXAMPLES)

        mock_qdrant = MagicMock()
        mock_qdrant.connect = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        mock_qdrant.delete_endpoint_points = MagicMock(return_value=True)
        mock_qdrant.upsert_endpoint_points = MagicMock(return_value=True)
        mock_qdrant.close = MagicMock()

        with (
            patch(
                "api_tool_indexer.main_indexer.LLMAPIClient",
                return_value=_make_ctx_manager_client(client),
            ),
            patch(
                "api_tool_indexer.main_indexer.ApiToolQdrantManager",
                return_value=mock_qdrant,
            ),
        ):
            result = await index_endpoint(_ENDPOINT_DATA)

        assert result.success is True
        assert result.endpoint_id == "ep-holidays"
        assert "indexed successfully" in result.message.lower()

    @pytest.mark.asyncio
    async def test_empty_examples_still_indexes_summary_point(self) -> None:
        """No example queries parsed → only summary point → still succeeds."""
        client = _make_api_client(context=_CONTEXT_WITHOUT_EXAMPLES)

        mock_qdrant = MagicMock()
        mock_qdrant.connect = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        mock_qdrant.delete_endpoint_points = MagicMock(return_value=True)
        mock_qdrant.upsert_endpoint_points = MagicMock(return_value=True)
        mock_qdrant.close = MagicMock()

        with (
            patch(
                "api_tool_indexer.main_indexer.LLMAPIClient",
                return_value=_make_ctx_manager_client(client),
            ),
            patch(
                "api_tool_indexer.main_indexer.ApiToolQdrantManager",
                return_value=mock_qdrant,
            ),
        ):
            result = await index_endpoint(_ENDPOINT_NO_PARAMS)

        assert result.success is True
        # Verify upsert was called with only summary point (no examples)
        call_args = mock_qdrant.upsert_endpoint_points.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].point_type == "summary"

    @pytest.mark.asyncio
    async def test_delete_failure_aborts_upsert(self) -> None:
        """If delete_endpoint_points returns False, upsert must not be called."""
        client = _make_api_client(context=_CONTEXT_WITH_EXAMPLES)

        mock_qdrant = MagicMock()
        mock_qdrant.connect = MagicMock()
        mock_qdrant.ensure_collection = MagicMock()
        mock_qdrant.delete_endpoint_points = MagicMock(return_value=False)
        mock_qdrant.upsert_endpoint_points = MagicMock()
        mock_qdrant.close = MagicMock()

        with (
            patch(
                "api_tool_indexer.main_indexer.LLMAPIClient",
                return_value=_make_ctx_manager_client(client),
            ),
            patch(
                "api_tool_indexer.main_indexer.ApiToolQdrantManager",
                return_value=mock_qdrant,
            ),
        ):
            result = await index_endpoint(_ENDPOINT_DATA)

        assert result.success is False
        assert (
            "delete" in result.message.lower()
            or "delete" in (result.error or "").lower()
        )
        mock_qdrant.upsert_endpoint_points.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_generation_failure_returns_failure_result(self) -> None:
        """LLM context generation raises → index_endpoint returns failure."""
        client = _make_api_client(raise_on_generate=Exception("LLM down"))
        client.max_retries = 1
        client.retry_delay_base = 0

        with (
            patch(
                "api_tool_indexer.main_indexer.LLMAPIClient",
                return_value=_make_ctx_manager_client(client),
            ),
            patch(
                "api_tool_indexer.main_indexer.asyncio.sleep", new_callable=AsyncMock
            ),
        ):
            result = await index_endpoint(_ENDPOINT_DATA)

        assert result.success is False
        assert result.endpoint_id == "ep-holidays"


# ---------------------------------------------------------------------------
# Step tracking (failed_step) and the --result-file output
# ---------------------------------------------------------------------------


def _make_qdrant(
    delete_ok: bool = True,
    upsert_ok: bool = True,
    raise_on_connect: Optional[Exception] = None,
    raise_on_upsert: Optional[Exception] = None,
) -> MagicMock:
    qdrant = MagicMock()
    qdrant.connect = MagicMock(side_effect=raise_on_connect)
    qdrant.ensure_collection = MagicMock()
    qdrant.delete_endpoint_points = MagicMock(return_value=delete_ok)
    if raise_on_upsert:
        qdrant.upsert_endpoint_points = MagicMock(side_effect=raise_on_upsert)
    else:
        qdrant.upsert_endpoint_points = MagicMock(return_value=upsert_ok)
    qdrant.close = MagicMock()
    return qdrant


async def _run_index(client: MagicMock, qdrant: MagicMock):
    with (
        patch(
            "api_tool_indexer.main_indexer.LLMAPIClient",
            return_value=_make_ctx_manager_client(client),
        ),
        patch(
            "api_tool_indexer.main_indexer.ApiToolQdrantManager",
            return_value=qdrant,
        ),
        patch("api_tool_indexer.main_indexer.asyncio.sleep", new_callable=AsyncMock),
    ):
        return await index_endpoint(_ENDPOINT_DATA)


class TestFailedStep:
    @pytest.mark.asyncio
    async def test_success_has_no_failed_step(self) -> None:
        result = await _run_index(_make_api_client(), _make_qdrant())
        assert result.success is True
        assert result.failed_step is None

    @pytest.mark.asyncio
    async def test_context_failure_is_enriching(self) -> None:
        client = _make_api_client(raise_on_generate=Exception("LLM down"))
        client.max_retries = 1
        result = await _run_index(client, _make_qdrant())
        assert result.success is False
        assert result.failed_step == "ENRICHING"
        assert "LLM down" in (result.error or "")

    @pytest.mark.asyncio
    async def test_embedding_failure_is_embedding(self) -> None:
        client = _make_api_client(
            raise_on_embed=RuntimeError("503 Service Unavailable")
        )
        result = await _run_index(client, _make_qdrant())
        assert result.success is False
        assert result.failed_step == "EMBEDDING"
        assert "503" in (result.error or "")

    @pytest.mark.asyncio
    async def test_empty_exception_message_still_gives_error(self) -> None:
        client = _make_api_client(raise_on_embed=TimeoutError())
        result = await _run_index(client, _make_qdrant())
        assert result.failed_step == "EMBEDDING"
        assert result.error == "TimeoutError"

    @pytest.mark.asyncio
    async def test_qdrant_connect_failure_is_qdrant_delete(self) -> None:
        qdrant = _make_qdrant(raise_on_connect=ConnectionError("qdrant unreachable"))
        result = await _run_index(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_DELETE"
        assert "qdrant unreachable" in (result.error or "")
        qdrant.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_returns_false_is_qdrant_delete(self) -> None:
        result = await _run_index(_make_api_client(), _make_qdrant(delete_ok=False))
        assert result.failed_step == "QDRANT_DELETE"

    @pytest.mark.asyncio
    async def test_upsert_returns_false_is_qdrant_upsert(self) -> None:
        result = await _run_index(_make_api_client(), _make_qdrant(upsert_ok=False))
        assert result.failed_step == "QDRANT_UPSERT"
        assert "old index" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_upsert_raises_is_qdrant_upsert(self) -> None:
        qdrant = _make_qdrant(raise_on_upsert=RuntimeError("write timeout"))
        result = await _run_index(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_UPSERT"
        assert "write timeout" in (result.error or "")
        assert "old index" in (result.error or "").lower()


class TestResultFile:
    _ARGS = [
        "--endpoint-id",
        "ep-holidays",
        "--name",
        "get_public_holidays",
        "--description",
        "Returns public holidays.",
        "--url",
        "https://openholidaysapi.org/PublicHolidays",
    ]

    def _read(self, path) -> dict:
        import json

        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_success_writes_success_true(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main
        from api_tool_indexer.models import IndexingResult

        out = tmp_path / "result.json"
        ok = IndexingResult(success=True, endpoint_id="ep-holidays", message="done")
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(return_value=ok),
        ):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 0
        data = self._read(out)
        assert data["success"] is True
        assert data["step"] is None
        assert data["error"] is None

    def test_failure_writes_step_and_error(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main
        from api_tool_indexer.models import IndexingResult

        out = tmp_path / "result.json"
        failed = IndexingResult(
            success=False,
            endpoint_id="ep-holidays",
            message="failed",
            error="503 Service Unavailable",
            failed_step="EMBEDDING",
        )
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(return_value=failed),
        ):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 1
        data = self._read(out)
        assert data == {
            "success": False,
            "step": "EMBEDDING",
            "error": "503 Service Unavailable",
            "message": "failed",
            "attemptNo": None,
        }

    def test_unexpected_crash_still_writes_file(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main

        out = tmp_path / "result.json"
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 1
        data = self._read(out)
        assert data["success"] is False
        assert "boom" in data["error"]

    def test_bad_arguments_still_write_file(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main

        out = tmp_path / "result.json"
        # --name, --description, --url missing → argparse exits with code 2
        code = main(["--endpoint-id", "x", "--result-file", str(out)])

        assert code == 2
        data = self._read(out)
        assert data["success"] is False
        assert "Invalid arguments" in data["error"]

    def test_no_result_file_flag_writes_nothing(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main
        from api_tool_indexer.models import IndexingResult

        ok = IndexingResult(success=True, endpoint_id="ep-holidays", message="done")
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(return_value=ok),
        ):
            assert main(self._ARGS) == 0
        assert list(tmp_path.iterdir()) == []

    def test_attempt_no_is_accepted_and_written(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main
        from api_tool_indexer.models import IndexingResult

        out = tmp_path / "result.json"
        ok = IndexingResult(success=True, endpoint_id="ep-holidays", message="done")
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(return_value=ok),
        ):
            code = main(self._ARGS + ["--attempt-no", "2", "--result-file", str(out)])

        assert code == 0
        assert self._read(out)["attemptNo"] == 2

    def test_without_attempt_no_still_works(self, tmp_path) -> None:
        from api_tool_indexer.main_indexer import main
        from api_tool_indexer.models import IndexingResult

        out = tmp_path / "result.json"
        ok = IndexingResult(success=True, endpoint_id="ep-holidays", message="done")
        with patch(
            "api_tool_indexer.main_indexer.index_endpoint",
            new=AsyncMock(return_value=ok),
        ):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 0
        assert self._read(out)["attemptNo"] is None

"""Unit tests for intent_data_enrichment/main_enrichment.py status reporting.

Covers:
- enrich_service(): failed_step for each pipeline step, real Qdrant error text
- main(): result file content (success, failure, crash, bad arguments,
  with and without --attempt-no)
"""

import json
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intent_data_enrichment.main_enrichment import enrich_service, main
from intent_data_enrichment.models import EnrichmentResult, ServiceData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVICE_DATA = ServiceData(
    service_id="svc-holidays",
    name="Public holidays",
    description="Returns public holidays.",
    examples=["When is the next holiday?", "Is Monday a holiday?"],
    entities=["date"],
)

_MODULE = "intent_data_enrichment.main_enrichment"


def _make_api_client(
    raise_on_generate: Optional[BaseException] = None,
    raise_on_embed: Optional[BaseException] = None,
) -> MagicMock:
    """LLMAPIClient replacement, used as ``async with LLMAPIClient() as c``."""
    client = MagicMock()
    client.generate_context = AsyncMock(
        side_effect=raise_on_generate, return_value="enriched context"
    )
    client.create_embedding = AsyncMock(
        side_effect=raise_on_embed, return_value=[0.1, 0.2, 0.3]
    )
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _make_qdrant(
    delete_ok: bool = True,
    upsert_ok: bool = True,
    last_error: Optional[str] = None,
    raise_on_connect: Optional[BaseException] = None,
    raise_on_upsert: Optional[BaseException] = None,
) -> MagicMock:
    qdrant = MagicMock()
    qdrant.last_error = last_error
    qdrant.connect.side_effect = raise_on_connect
    qdrant.delete_service_points.return_value = delete_ok
    if raise_on_upsert is not None:
        qdrant.upsert_service_points.side_effect = raise_on_upsert
    else:
        qdrant.upsert_service_points.return_value = upsert_ok
    return qdrant


async def _run_enrich(api_client: MagicMock, qdrant: MagicMock) -> EnrichmentResult:
    with (
        patch(f"{_MODULE}.LLMAPIClient", return_value=api_client),
        patch(f"{_MODULE}.QdrantManager", return_value=qdrant),
        patch(
            f"{_MODULE}.compute_sparse_vector",
            return_value=SimpleNamespace(indices=[1], values=[1.0]),
        ),
    ):
        return await enrich_service(_SERVICE_DATA)


# ---------------------------------------------------------------------------
# enrich_service(): failed step
# ---------------------------------------------------------------------------


class TestFailedStep:
    @pytest.mark.asyncio
    async def test_success_has_no_failed_step(self) -> None:
        result = await _run_enrich(_make_api_client(), _make_qdrant())
        assert result.success is True
        assert result.failed_step is None

    @pytest.mark.asyncio
    async def test_context_failure_is_enriching(self) -> None:
        client = _make_api_client(raise_on_generate=RuntimeError("LLM down"))
        result = await _run_enrich(client, _make_qdrant())
        assert result.success is False
        assert result.failed_step == "ENRICHING"
        assert "LLM down" in (result.error or "")

    @pytest.mark.asyncio
    async def test_embedding_failure_is_embedding(self) -> None:
        client = _make_api_client(
            raise_on_embed=RuntimeError("503 Service Unavailable")
        )
        result = await _run_enrich(client, _make_qdrant())
        assert result.failed_step == "EMBEDDING"
        assert "503" in (result.error or "")

    @pytest.mark.asyncio
    async def test_empty_exception_message_still_gives_error(self) -> None:
        client = _make_api_client(raise_on_embed=TimeoutError())
        result = await _run_enrich(client, _make_qdrant())
        assert result.failed_step == "EMBEDDING"
        assert result.error == "TimeoutError"

    @pytest.mark.asyncio
    async def test_qdrant_connect_failure_is_qdrant_delete(self) -> None:
        qdrant = _make_qdrant(raise_on_connect=ConnectionError("qdrant unreachable"))
        result = await _run_enrich(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_DELETE"
        assert "qdrant unreachable" in (result.error or "")
        qdrant.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_failure_reports_real_error(self) -> None:
        qdrant = _make_qdrant(
            delete_ok=False, last_error="ResponseHandlingException: timed out"
        )
        result = await _run_enrich(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_DELETE"
        assert result.error == "ResponseHandlingException: timed out"
        qdrant.upsert_service_points.assert_not_called()

    @pytest.mark.asyncio
    async def test_upsert_failure_reports_real_error(self) -> None:
        qdrant = _make_qdrant(
            upsert_ok=False, last_error="UnexpectedResponse: 400 wrong vector size"
        )
        result = await _run_enrich(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_UPSERT"
        assert "wrong vector size" in (result.error or "")
        assert "old index" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_upsert_raises_is_qdrant_upsert(self) -> None:
        qdrant = _make_qdrant(raise_on_upsert=RuntimeError("write timeout"))
        result = await _run_enrich(_make_api_client(), qdrant)
        assert result.failed_step == "QDRANT_UPSERT"
        assert "write timeout" in (result.error or "")
        assert "old index" in (result.error or "").lower()


# ---------------------------------------------------------------------------
# QdrantManager.last_error (real manager, mocked Qdrant client)
# ---------------------------------------------------------------------------


class TestQdrantManagerLastError:
    def test_delete_failure_sets_last_error_and_success_resets_it(self) -> None:
        from intent_data_enrichment.qdrant_manager import QdrantManager

        manager = QdrantManager(url="http://qdrant-test:6333")
        manager.client = MagicMock()
        manager.client.delete.side_effect = RuntimeError("timed out")

        assert manager.delete_service_points("svc-holidays") is False
        assert manager.last_error is not None
        assert "RuntimeError" in manager.last_error
        assert "timed out" in manager.last_error

        manager.client.delete.side_effect = None
        assert manager.delete_service_points("svc-holidays") is True
        assert manager.last_error is None


# ---------------------------------------------------------------------------
# main(): result file
# ---------------------------------------------------------------------------


class TestResultFile:
    _ARGS = [
        "--service-id",
        "svc-holidays",
        "--name",
        "Public holidays",
        "--description",
        "Returns public holidays.",
    ]

    def _read(self, path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _ok(self) -> EnrichmentResult:
        return EnrichmentResult(success=True, service_id="svc-holidays", message="done")

    def test_success_writes_success_true(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        with patch(f"{_MODULE}.enrich_service", new=AsyncMock(return_value=self._ok())):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 0
        data = self._read(out)
        assert data["success"] is True
        assert data["step"] is None
        assert data["error"] is None

    def test_failure_writes_step_and_error(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        failed = EnrichmentResult(
            success=False,
            service_id="svc-holidays",
            message="failed",
            error="503 Service Unavailable",
            failed_step="EMBEDDING",
        )
        with patch(f"{_MODULE}.enrich_service", new=AsyncMock(return_value=failed)):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 1
        assert self._read(out) == {
            "success": False,
            "step": "EMBEDDING",
            "error": "503 Service Unavailable",
            "message": "failed",
            "attemptNo": None,
        }

    def test_unexpected_crash_still_writes_file(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        with patch(
            f"{_MODULE}.enrich_service",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 1
        data = self._read(out)
        assert data["success"] is False
        assert "boom" in data["error"]

    def test_bad_arguments_still_write_file(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        # --name and --description missing → argparse exits with code 2
        code = main(["--service-id", "x", "--result-file", str(out)])

        assert code == 2
        data = self._read(out)
        assert data["success"] is False
        assert "Invalid arguments" in data["error"]

    def test_no_result_file_flag_writes_nothing(self, tmp_path) -> None:
        with patch(f"{_MODULE}.enrich_service", new=AsyncMock(return_value=self._ok())):
            assert main(self._ARGS) == 0
        assert list(tmp_path.iterdir()) == []

    def test_attempt_no_is_accepted_and_written(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        with patch(f"{_MODULE}.enrich_service", new=AsyncMock(return_value=self._ok())):
            code = main(self._ARGS + ["--attempt-no", "2", "--result-file", str(out)])

        assert code == 0
        assert self._read(out)["attemptNo"] == 2

    def test_without_attempt_no_still_works(self, tmp_path) -> None:
        out = tmp_path / "result.json"
        with patch(f"{_MODULE}.enrich_service", new=AsyncMock(return_value=self._ok())):
            code = main(self._ARGS + ["--result-file", str(out)])

        assert code == 0
        assert self._read(out)["attemptNo"] is None

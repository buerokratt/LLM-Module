"""Unit tests for IndexerSupport._get_context_manager connection resolution.

The /generate-context endpoint (used by the API-tool and vector indexers) must
honour the connection_id supplied in the request, like /embeddings does, and only
fall back to the Resql lookup of the environment's vault_uuid when none is given.

Scope: these tests verify *which* connection id and environment reach the
LLMManager constructor and whether Resql is consulted. They do not (and cannot,
with LLMManager mocked) verify that the LM is bound per connection: LLMManager
is a process-wide singleton fixed by the first successful construction, so the
arguments only take effect when this is the first LLM manager in the process.
The method is exercised unbound on a bare stub object so nothing heavy is built.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.vector_indexer.indexer_support import IndexerSupport


@pytest.fixture
def stub_support() -> SimpleNamespace:
    """Bare object standing in for ``self``; no cached _context_manager yet."""
    return SimpleNamespace(_context_manager=None)


def _call(stub: SimpleNamespace, **kwargs) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Run _get_context_manager with LLMManager, ContextGenerationManager and the
    connection-id fetcher patched; return the three mocks."""
    fetcher = MagicMock()
    fetcher.fetch_vault_uuid_sync.return_value = "uuid-from-db"
    with (
        patch("src.vector_indexer.indexer_support.LLMManager") as llm_manager_cls,
        patch(
            "src.llm_orchestrator_config.context_manager.ContextGenerationManager"
        ) as ctx_cls,
        patch(
            "src.utils.connection_id_fetcher.get_connection_id_fetcher",
            return_value=fetcher,
        ),
    ):
        IndexerSupport._get_context_manager(stub, **kwargs)  # type: ignore[arg-type]
    return llm_manager_cls, ctx_cls, fetcher


def test_supplied_connection_id_skips_db_lookup(stub_support) -> None:
    llm_manager_cls, ctx_cls, fetcher = _call(
        stub_support, environment="testing", connection_id="evalconnection-1"
    )

    fetcher.fetch_vault_uuid_sync.assert_not_called()
    llm_manager_cls.assert_called_once_with(
        environment="testing", connection_id="evalconnection-1"
    )
    ctx_cls.assert_called_once_with(llm_manager_cls.return_value)


def test_missing_connection_id_resolves_from_db_for_environment(stub_support) -> None:
    llm_manager_cls, _, fetcher = _call(stub_support, environment="production")

    fetcher.fetch_vault_uuid_sync.assert_called_once_with("production")
    llm_manager_cls.assert_called_once_with(
        environment="production", connection_id="uuid-from-db"
    )


def test_defaults_match_previous_behaviour(stub_support) -> None:
    """No arguments → production lookup, exactly what the CronManager indexer relied on."""
    llm_manager_cls, _, fetcher = _call(stub_support)

    fetcher.fetch_vault_uuid_sync.assert_called_once_with("production")
    assert llm_manager_cls.call_args.kwargs["environment"] == "production"


def test_context_manager_is_cached(stub_support) -> None:
    llm_manager_cls, ctx_cls, _ = _call(
        stub_support, environment="testing", connection_id="evalconnection-1"
    )
    first = stub_support._context_manager

    # Second call with different args must not rebuild (process-wide LLMManager).
    _call(stub_support, environment="production")
    assert stub_support._context_manager is first

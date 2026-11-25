"""Unit tests for IndexerSupport._get_context_manager connection resolution.

The /generate-context endpoint (used by the API-tool and vector indexers) must
honour the connection_id supplied in the request, like /embeddings does, and only
fall back to the Resql lookup of the environment's vault_uuid when none is given.

Scope: these tests verify *which* connection id and environment reach the
LLMManager constructor, whether Resql is consulted, and that one context manager
is kept per connection and rebuilt when its LLMManager is reloaded.
The method is exercised unbound on a bare stub object so nothing heavy is built.
"""

from types import SimpleNamespace
from typing import Dict, Iterator, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.vector_indexer.indexer_support import IndexerSupport


@pytest.fixture
def stub_support() -> SimpleNamespace:
    """Bare object standing in for ``self``; no cached context managers yet."""
    return SimpleNamespace(_context_managers={})


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


# --- One context manager per connection ----------------------------------------


@pytest.fixture
def per_connection(stub_support: SimpleNamespace) -> Iterator[Dict[str, object]]:
    """Patch LLMManager to return one object per connection (like the real
    per-connection cache) and the DB lookup to return the current production id."""
    managers: Dict[str, SimpleNamespace] = {}
    state: Dict[str, object] = {"production_id": "prod-1", "managers": managers}

    def llm_manager(environment: str, connection_id: Optional[str]) -> SimpleNamespace:
        key = str(connection_id)
        if key not in managers:
            managers[key] = SimpleNamespace(connection_id=connection_id)
        return managers[key]

    fetcher = MagicMock()
    fetcher.fetch_vault_uuid_sync.side_effect = lambda env: state["production_id"]

    with (
        patch("src.vector_indexer.indexer_support.LLMManager", side_effect=llm_manager),
        patch(
            "src.llm_orchestrator_config.context_manager.ContextGenerationManager",
            side_effect=lambda manager: SimpleNamespace(llm_manager=manager),
        ),
        patch(
            "src.utils.connection_id_fetcher.get_connection_id_fetcher",
            return_value=fetcher,
        ),
    ):
        yield state


def _get(stub: SimpleNamespace, **kwargs) -> SimpleNamespace:
    return IndexerSupport._get_context_manager(stub, **kwargs)  # type: ignore[arg-type,return-value]


def test_same_connection_reuses_context_manager(stub_support, per_connection) -> None:
    first = _get(stub_support, environment="production")
    second = _get(stub_support, environment="production")

    assert second is first


def test_production_switch_uses_new_connection(stub_support, per_connection) -> None:
    before = _get(stub_support, environment="production")

    # Production is switched to another connection (e.g. after /cache/clear).
    per_connection["production_id"] = "prod-2"
    after = _get(stub_support, environment="production")

    assert after is not before
    assert after.llm_manager.connection_id == "prod-2"


def test_reloaded_connection_rebuilds_context_manager(
    stub_support, per_connection
) -> None:
    before = _get(stub_support, environment="production")

    # /cache/clear drops the loaded connection; the same id is loaded again.
    per_connection["managers"].clear()  # type: ignore[union-attr]
    after = _get(stub_support, environment="production")

    assert after is not before
    assert after.llm_manager is not before.llm_manager

"""LLMManager keeps one instance per connection and each request uses its own connection."""

import asyncio
from typing import Any, Dict, Generator, List, Optional

import dspy
import pytest

import llm_orchestrator_config.llm_manager as llm_manager_module
from llm_orchestrator_config.llm_manager import LLMManager, request_llm_scope

PRODUCTION_ID = "prod-uuid"


class _FakeProvider:
    """Stands in for a provider; holds one DSPy LM per connection."""

    def __init__(self, connection_id: Optional[str]) -> None:
        self.lm = dspy.LM(f"openai/model-{connection_id}", api_key="test-key")

    def get_dspy_client(self) -> dspy.LM:
        return self.lm


@pytest.fixture(autouse=True)
def _fake_connections(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[List[str], None, None]:
    """Load fake connections instead of reading config and Vault."""
    loaded: List[str] = []

    def fake_load(self: LLMManager) -> None:
        if self._connection_id == "broken":
            raise RuntimeError("Vault unavailable")
        loaded.append(str(self._connection_id))

    def fake_init_providers(self: LLMManager) -> None:
        self._default_provider = _FakeProvider(self._connection_id)  # type: ignore[assignment]

    monkeypatch.setattr(LLMManager, "_load_configuration", fake_load)
    monkeypatch.setattr(LLMManager, "_initialize_providers", fake_init_providers)
    monkeypatch.setattr(
        llm_manager_module, "_resolve_connection_id", lambda environment: PRODUCTION_ID
    )
    monkeypatch.setattr(llm_manager_module, "ConfigurationLoader", lambda *a: None)

    LLMManager.clear_instances()
    yield loaded
    LLMManager.clear_instances()


def _model(lm: Any) -> str:
    return str(getattr(lm, "model", lm))


# --- One instance per connection -------------------------------------------


def test_same_connection_returns_same_instance(_fake_connections: List[str]) -> None:
    first = LLMManager(environment="testing", connection_id="a")
    second = LLMManager(environment="testing", connection_id="a")

    assert first is second
    assert _fake_connections == ["a"]  # loaded once


def test_different_connections_return_different_instances() -> None:
    a = LLMManager(environment="testing", connection_id="a")
    b = LLMManager(environment="testing", connection_id="b")

    assert a is not b
    assert a.connection_id == "a"
    assert b.connection_id == "b"
    assert _model(a.get_dspy_client()) != _model(b.get_dspy_client())


def test_missing_connection_id_resolves_current_connection() -> None:
    manager = LLMManager()

    assert manager.connection_id == PRODUCTION_ID
    assert manager is LLMManager(environment="production", connection_id=PRODUCTION_ID)


def test_failed_load_is_not_cached(_fake_connections: List[str]) -> None:
    with pytest.raises(RuntimeError):
        LLMManager(environment="testing", connection_id="broken")

    assert ("testing", "broken") not in LLMManager._instances


def test_clear_instances_reloads_connection(
    _fake_connections: List[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    before = LLMManager(environment="production", connection_id=PRODUCTION_ID)
    monkeypatch.setattr(LLMManager, "_configured", True)

    LLMManager.clear_instances()
    after = LLMManager(environment="production", connection_id=PRODUCTION_ID)

    assert after is not before
    assert _fake_connections == [PRODUCTION_ID, PRODUCTION_ID]
    # DSPy only allows dspy.configure() from the first task, so the global
    # default must not be set again after a cache clear.
    assert LLMManager._configured is True


def test_cache_clear_in_another_task_does_not_reconfigure_dspy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup configures DSPy in one task; requests after /cache/clear run in others."""
    monkeypatch.setattr(LLMManager, "_configured", False)
    configure_calls: List[Any] = []
    monkeypatch.setattr(
        llm_manager_module.dspy,
        "configure",
        lambda **kwargs: configure_calls.append(kwargs["lm"]),
    )

    LLMManager(
        environment="production", connection_id=PRODUCTION_ID
    ).ensure_global_config()
    LLMManager.clear_instances()

    async def request_after_cache_clear() -> None:
        manager = LLMManager(environment="testing", connection_id="b")
        manager.ensure_global_config()
        with request_llm_scope():
            manager.activate_for_request()
            assert dspy.settings.lm is manager.get_dspy_client()

    asyncio.run(request_after_cache_clear())

    assert len(configure_calls) == 1


# --- Request scope -----------------------------------------------------------


def test_use_task_local_outside_request_uses_own_connection() -> None:
    a = LLMManager(environment="testing", connection_id="a")

    with a.use_task_local():
        assert dspy.settings.lm is a.get_dspy_client()


def test_request_scope_uses_request_connection_everywhere() -> None:
    production = LLMManager(environment="production", connection_id=PRODUCTION_ID)
    testing = LLMManager(environment="testing", connection_id="b")
    default_lm = dspy.settings.lm

    with request_llm_scope():
        testing.activate_for_request()

        # Code reading dspy.settings.lm directly (e.g. guardrails)
        assert dspy.settings.lm is testing.get_dspy_client()
        # A component holding another connection's manager (e.g. the tool classifier)
        with production.use_task_local():
            assert dspy.settings.lm is testing.get_dspy_client()

    assert dspy.settings.lm is default_lm


def test_activate_outside_request_scope_has_no_effect() -> None:
    default_lm = dspy.settings.lm

    LLMManager(environment="testing", connection_id="b").activate_for_request()

    assert dspy.settings.lm is default_lm


def test_activate_twice_with_same_connection_is_harmless() -> None:
    testing = LLMManager(environment="testing", connection_id="b")
    default_lm = dspy.settings.lm

    with request_llm_scope():
        testing.activate_for_request()
        testing.activate_for_request()  # e.g. workflows re-initializing components
        assert dspy.settings.lm is testing.get_dspy_client()

    assert dspy.settings.lm is default_lm


# --- Streaming and concurrency -------------------------------------------------


async def _stream(manager: LLMManager) -> Any:
    """Mimics stream_orchestration_response: scope wraps a generator that yields."""
    with request_llm_scope():
        manager.activate_for_request()
        yield _model(dspy.settings.lm)
        await asyncio.sleep(0)
        yield _model(dspy.settings.lm)
        # Blocking LLM calls run in threads via asyncio.to_thread
        yield await asyncio.to_thread(lambda: _model(dspy.settings.lm))


def test_streaming_keeps_request_connection_across_yields() -> None:
    testing = LLMManager(environment="testing", connection_id="b")

    async def collect() -> List[str]:
        return [chunk async for chunk in _stream(testing)]

    chunks = asyncio.run(collect())

    assert chunks == [_model(testing.get_dspy_client())] * 3


def test_concurrent_requests_use_their_own_connections() -> None:
    a = LLMManager(environment="testing", connection_id="a")
    b = LLMManager(environment="testing", connection_id="b")

    async def request(manager: LLMManager) -> List[str]:
        return [chunk async for chunk in _stream(manager)]

    async def run_both() -> Dict[str, List[str]]:
        result_a, result_b = await asyncio.gather(request(a), request(b))
        return {"a": result_a, "b": result_b}

    results = asyncio.run(run_both())

    assert results["a"] == [_model(a.get_dspy_client())] * 3
    assert results["b"] == [_model(b.get_dspy_client())] * 3

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.guardrails.preload import preload_missing, retry_preload


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv('LLM_GENERATION_ENABLED', 'true')


def service(factory, initial=None):
    obj = SimpleNamespace(shared_guardrails_adapters=initial or {}, _initialize_guardrails=factory)
    obj._initialize_shared_guardrails_at_startup = lambda: preload_missing(obj, Mock())
    return obj


def test_failed_adapters_not_published_and_no_exception_values_logged():
    adapter = Mock()
    adapter._ensure_initialized.side_effect = RuntimeError('secret-remote-detail')
    logger = Mock()
    obj = service(Mock(return_value=adapter))
    assert preload_missing(obj, logger) == {}
    assert 'secret-remote-detail' not in str(logger.mock_calls)
    assert adapter._ensure_initialized.call_count == 2


def test_ready_environment_is_not_rebuilt_and_missing_production_stays_absent():
    ready = object()
    factory = Mock(side_effect=ValueError('not configured'))
    obj = service(factory, {'testing': ready})
    assert preload_missing(obj, Mock()) == {'testing': ready}
    factory.assert_called_once_with(environment='production', connection_id=None)


def test_generation_disabled_never_initializes(monkeypatch):
    monkeypatch.setenv('LLM_GENERATION_ENABLED', 'false')
    factory = Mock()
    obj = service(factory)
    assert preload_missing(obj, Mock()) == {}
    asyncio.run(retry_preload(obj, Mock(), initial_delay=0))
    factory.assert_not_called()


def test_dependency_recovers_without_request_or_process_restart():
    async def scenario():
        attempts = 0
        adapter = Mock()
        def initialize(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TimeoutError('Resql starting')
            return adapter
        obj = service(initialize, {'production': object()})
        obj.shared_guardrails_adapters = obj._initialize_shared_guardrails_at_startup()
        assert 'testing' not in obj.shared_guardrails_adapters
        await asyncio.wait_for(retry_preload(obj, Mock(), initial_delay=.001, max_delay=.002), 2)
        assert obj.shared_guardrails_adapters['testing'] is adapter
        adapter._ensure_initialized.assert_called_once()
        assert attempts == 2
    asyncio.run(scenario())


def test_backoff_is_capped_and_no_parallel_attempts(monkeypatch):
    async def scenario():
        sleeps = []
        active = 0
        def initialize(**kwargs):
            nonlocal active
            active += 1
            assert active == 1
            active -= 1
            raise TimeoutError()
        obj = service(initialize)
        async def sleep(delay):
            sleeps.append(delay)
            if len(sleeps) == 6:
                raise asyncio.CancelledError()
        monkeypatch.setattr('src.guardrails.preload.asyncio.sleep', sleep)
        with pytest.raises(asyncio.CancelledError):
            await retry_preload(obj, Mock(), initial_delay=5, max_delay=20)
        assert sleeps == [5, 10, 20, 20, 20, 20]
        assert obj.shared_guardrails_adapters == {}
    asyncio.run(scenario())


def test_slow_sync_initialization_does_not_block_event_loop_and_shutdown_drains():
    async def scenario():
        started, release = threading.Event(), threading.Event()
        adapter = Mock()
        def initialize(**kwargs):
            started.set()
            assert release.wait(2)
            return adapter
        obj = service(initialize, {'production': object()})
        task = asyncio.create_task(retry_preload(obj, Mock(), initial_delay=.001))
        while not started.is_set():
            await asyncio.sleep(.001)
        # Reaching this point while the initializer is blocked proves loop liveness.
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert 'testing' not in obj.shared_guardrails_adapters
        adapter._ensure_initialized.assert_called_once()
    asyncio.run(scenario())


def test_fastapi_lifespan_recovers_preload_and_reports_environment(monkeypatch):
    import time
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    from src import llm_orchestration_service_api as api
    from src.guardrails import offline_embeddings
    factory = Mock(return_value=Mock())
    obj = service(factory)
    obj._prewarm_shared_bm25 = AsyncMock()
    obj.aclose = AsyncMock()
    monkeypatch.setattr(api, 'LLMOrchestrationService', lambda: obj)
    monkeypatch.setattr(offline_embeddings, 'verify_and_load', lambda: True)
    monkeypatch.setattr(api, 'init_redis_client', AsyncMock())
    monkeypatch.setattr(api, 'close_redis_client', AsyncMock())
    monkeypatch.setattr(api, 'check_redis_health', AsyncMock(return_value='connected'))
    monkeypatch.setattr(api, 'LLMManager', Mock())
    monkeypatch.setattr(api, 'create_incremental_summarizer', Mock(return_value=None))
    monkeypatch.setattr(api.StreamConfig, 'RATE_LIMIT_ENABLED', False)
    async def fast_retry(service, logger):
        await retry_preload(service, logger, initial_delay=.001, max_delay=.002)
    monkeypatch.setattr('src.guardrails.preload.retry_preload', fast_retry)
    app = FastAPI(lifespan=api.lifespan)
    app.add_api_route('/health', api.health_check)
    with TestClient(app) as client:
        deadline = time.monotonic() + 2
        while True:
            response = client.get('/health')
            assert response.status_code == 200
            health = response.json()
            if health['guardrails_environments'] == ['production', 'testing']:
                break
            assert time.monotonic() < deadline
            time.sleep(.001)
        assert health['guardrails_loaded'] and health['guardrails_embedding_loaded']
        assert health['redis_session_store'] == 'connected'
        task = app.state.guardrails_preload_task
    assert task.done()
    obj.aclose.assert_awaited_once()

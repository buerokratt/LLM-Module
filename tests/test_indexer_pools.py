"""The indexer's endpoints run on their own capped pools, and context generation
has a token budget (src/utils/indexer_pools).

During a sync the indexer sent 15 context-generation calls at once. On the
default thread pool they took the threads user queries need, and their tokens
used up the model quota the queries share (50k tokens/minute): every query
timed out while the knowledge base was indexed.
"""

import asyncio
import contextvars
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import List

import pytest

from src.utils import indexer_pools
from src.utils.indexer_pools import TokenBudget, estimate_context_tokens, run_in_pool


@pytest.mark.asyncio
async def test_indexer_calls_are_capped_at_the_pool_size() -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def generate_context() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    pool = ThreadPoolExecutor(max_workers=3)
    try:
        # What the indexer sends during a sync: 3 documents x 5 chunks.
        await asyncio.gather(*(run_in_pool(pool, generate_context) for _ in range(15)))
    finally:
        pool.shutdown()

    assert peak == 3


@pytest.mark.asyncio
async def test_user_queries_are_not_starved_while_the_indexer_pool_is_busy() -> None:
    release = threading.Event()
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        # Indexer calls hang (as they did while the model quota was exhausted).
        indexer_calls = [
            asyncio.ensure_future(run_in_pool(pool, release.wait, 10))
            for _ in range(15)
        ]
        await asyncio.sleep(0.05)

        # A user query's own LLM call runs on the default pool meanwhile.
        start = time.monotonic()
        answer = await asyncio.wait_for(asyncio.to_thread(lambda: "answer"), 2)

        assert answer == "answer"
        assert time.monotonic() - start < 1
    finally:
        release.set()
        await asyncio.gather(*indexer_calls)
        pool.shutdown()


@pytest.mark.asyncio
async def test_context_variables_reach_the_pool_thread() -> None:
    request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id")
    request_id.set("sync-request")
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        assert await run_in_pool(pool, request_id.get) == "sync-request"
    finally:
        pool.shutdown()


@pytest.mark.asyncio
async def test_keyword_arguments_are_passed_through() -> None:
    seen: List[tuple] = []
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        await run_in_pool(pool, lambda a, b=0: seen.append((a, b)), 1, b=2)
    finally:
        pool.shutdown()
    assert seen == [(1, 2)]


@pytest.mark.parametrize(
    "value, expected", [(None, 3), ("5", 5), ("0", 1), ("-2", 1), ("many", 3)]
)
def test_pool_size_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    if value is None:
        monkeypatch.delenv("INDEXER_CONTEXT_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("INDEXER_CONTEXT_CONCURRENCY", value)

    assert indexer_pools._pool_size("INDEXER_CONTEXT_CONCURRENCY", 3) == expected


@pytest.mark.asyncio
async def test_indexer_endpoints_run_on_their_own_pools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_orchestration_service_api import (
        app,
        create_embeddings,
        generate_context_with_caching,
    )
    from models.request_models import ContextGenerationRequest, EmbeddingRequest

    threads: List[str] = []

    def generate_context(request: ContextGenerationRequest) -> dict:
        threads.append(threading.current_thread().name)
        return {
            "context": "context",
            "usage": {},
            "cache_performance": {},
            "model_used": "gpt-4.1",
        }

    def embed(**kwargs: object) -> dict:
        threads.append(threading.current_thread().name)
        return {"embeddings": [[0.1]], "model_used": "emb", "processing_info": {}}

    monkeypatch.setattr(
        app.state,
        "orchestration_service",
        SimpleNamespace(
            generate_context_for_chunks=generate_context,
            create_embeddings_for_indexer=embed,
        ),
        raising=False,
    )

    await generate_context_with_caching(
        ContextGenerationRequest(
            document_prompt="document", chunk_prompt="chunk", environment="production"
        )
    )
    await create_embeddings(EmbeddingRequest(texts=["chunk"], environment="production"))

    assert threads[0].startswith("indexer-context")
    assert threads[1].startswith("indexer-embedding")


def test_endpoint_pools_are_dedicated_and_small() -> None:
    assert indexer_pools.context_pool._max_workers == indexer_pools.CONTEXT_CONCURRENCY
    assert (
        indexer_pools.embedding_pool._max_workers == indexer_pools.EMBEDDING_CONCURRENCY
    )
    assert indexer_pools.CONTEXT_CONCURRENCY < 15


# ---------------------------------------------------------------------------
# Token budget for context generation
# ---------------------------------------------------------------------------


class FakeClock:
    """Simulated time: sleep() advances it instantly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _budget(tokens_per_minute: int) -> "tuple[TokenBudget, FakeClock]":
    clock = FakeClock()
    return TokenBudget(tokens_per_minute, clock=clock, sleep=clock.sleep), clock


@pytest.mark.asyncio
async def test_calls_within_the_budget_do_not_wait() -> None:
    budget, clock = _budget(20_000)

    assert await budget.acquire(8_000) == 0
    assert await budget.acquire(12_000) == 0
    assert clock.now == 0


@pytest.mark.asyncio
async def test_calls_over_the_budget_wait_for_it_to_refill() -> None:
    budget, clock = _budget(20_000)  # refills 20k per 60 s
    await budget.acquire(20_000)

    waited = await budget.acquire(5_000)

    assert waited == pytest.approx(15.0)
    assert clock.now == pytest.approx(15.0)


@pytest.mark.asyncio
async def test_indexer_never_exceeds_its_tokens_per_minute() -> None:
    """15 calls of 10k tokens (what the indexer sent at once) take as long as
    a 20k/minute budget needs: the first two minutes' worth at once, the rest
    paced."""
    budget, clock = _budget(20_000)

    await asyncio.gather(*(budget.acquire(10_000) for _ in range(15)))

    # 150k tokens: 20k right away, 130k at 20k per minute.
    assert clock.now == pytest.approx(130_000 / 20_000 * 60)


@pytest.mark.asyncio
async def test_a_call_larger_than_the_budget_waits_for_a_full_bucket() -> None:
    budget, clock = _budget(20_000)
    await budget.acquire(20_000)

    waited = await asyncio.wait_for(budget.acquire(45_000), 1)

    assert waited == pytest.approx(60.0)


def test_estimate_counts_the_prompt_and_reserved_output() -> None:
    document = "x" * 40_000  # ~10k tokens

    assert estimate_context_tokens(document, "y" * 400, max_tokens=1000) == 11_100


@pytest.mark.parametrize(
    "value, enabled", [(None, True), ("50000", True), ("0", False)]
)
def test_budget_can_be_set_or_turned_off(
    monkeypatch: pytest.MonkeyPatch, value: "str | None", enabled: bool
) -> None:
    if value is None:
        monkeypatch.delenv("INDEXER_CONTEXT_TOKENS_PER_MINUTE", raising=False)
    else:
        monkeypatch.setenv("INDEXER_CONTEXT_TOKENS_PER_MINUTE", value)

    budget = indexer_pools._context_budget()

    assert (budget is not None) is enabled
    if value == "50000":
        assert budget is not None and budget.capacity == 50_000


@pytest.mark.asyncio
async def test_generate_context_endpoint_waits_for_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_orchestration_service_api as api
    from models.request_models import ContextGenerationRequest

    requested: List[int] = []

    class RecordingBudget:
        async def acquire(self, tokens: int) -> float:
            requested.append(tokens)
            return 0.0

    monkeypatch.setattr(api, "context_budget", RecordingBudget())
    monkeypatch.setattr(
        api.app.state,
        "orchestration_service",
        SimpleNamespace(
            generate_context_for_chunks=lambda request: {
                "context": "context",
                "usage": {},
                "cache_performance": {},
                "model_used": "gpt-4.1",
            }
        ),
        raising=False,
    )

    await api.generate_context_with_caching(
        ContextGenerationRequest(
            document_prompt="d" * 4_000,
            chunk_prompt="c" * 400,
            environment="production",
            max_tokens=1000,
        )
    )

    assert requested == [1_100 + 1_000]

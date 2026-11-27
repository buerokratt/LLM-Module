"""Worker pools for the vector indexer's endpoints (/generate-context, /embeddings).

During a sync the indexer sends many calls at once (3 documents x 5 chunks by
default). Run on the default thread pool, they took the threads user queries
need, and that many LLM calls at once used up the model quota the queries share
with them: every query timed out while the knowledge base was being indexed.

Each indexer endpoint gets its own small pool instead. Its size caps how many
indexer calls run at a time; the rest wait in the pool's queue (the indexer
allows 300 s per request and retries). User queries keep the default pool.

Sizes can be changed with INDEXER_CONTEXT_CONCURRENCY and
INDEXER_EMBEDDING_CONCURRENCY.

Capping calls is not enough on its own: the model's quota is tokens per minute
(50,000 on the shared gpt-4.1 deployment), and every context call sends the
chunk's whole document (up to ~19k tokens). Even 3 calls at a time used the whole
quota and user queries got 429s. So context generation also has a token budget,
INDEXER_CONTEXT_TOKENS_PER_MINUTE (default 20,000; 0 turns it off): calls wait
until their estimated tokens fit, leaving the rest of the quota to user queries.
Raise it with the deployment's quota.
"""

import asyncio
import contextvars
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Awaitable, Callable, Optional, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

DEFAULT_CONTEXT_CONCURRENCY = 3
DEFAULT_EMBEDDING_CONCURRENCY = 2
DEFAULT_CONTEXT_TOKENS_PER_MINUTE = 20_000
CHARS_PER_TOKEN = 4


def _int_from_env(env_name: str, default: int) -> int:
    """An integer from ``env_name``, else ``default``."""
    try:
        return int(os.getenv(env_name, str(default)))
    except ValueError:
        return default


def _pool_size(env_name: str, default: int) -> int:
    """A positive worker count from ``env_name``, else ``default``."""
    return max(1, _int_from_env(env_name, default))


CONTEXT_CONCURRENCY = _pool_size(
    "INDEXER_CONTEXT_CONCURRENCY", DEFAULT_CONTEXT_CONCURRENCY
)
EMBEDDING_CONCURRENCY = _pool_size(
    "INDEXER_EMBEDDING_CONCURRENCY", DEFAULT_EMBEDDING_CONCURRENCY
)

context_pool = ThreadPoolExecutor(
    max_workers=CONTEXT_CONCURRENCY, thread_name_prefix="indexer-context"
)
embedding_pool = ThreadPoolExecutor(
    max_workers=EMBEDDING_CONCURRENCY, thread_name_prefix="indexer-embedding"
)


async def run_in_pool(
    pool: ThreadPoolExecutor, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
) -> T:
    """Run blocking ``fn`` on ``pool`` and await the result.

    Like asyncio.to_thread, but on the given pool; context variables (Langfuse
    spans, DSPy settings) are carried over the same way.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(pool, partial(context.run, fn, *args, **kwargs))


class TokenBudget:
    """At most ``tokens_per_minute`` (estimated) tokens per minute, as a token
    bucket: up to a minute's worth at once, refilled continuously.

    Callers queue in arrival order. A call larger than a minute's budget waits
    for a full bucket instead of forever.
    """

    def __init__(
        self,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.capacity = float(tokens_per_minute)
        self._rate = tokens_per_minute / 60.0
        self._available = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> float:
        """Wait until ``tokens`` fit in the budget; returns the seconds waited."""
        needed = min(float(max(tokens, 0)), self.capacity)
        waited = 0.0
        async with self._lock:
            while True:
                now = self._clock()
                self._available = min(
                    self.capacity, self._available + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._available >= needed:
                    self._available -= needed
                    return waited
                delay = (needed - self._available) / self._rate
                waited += delay
                await self._sleep(delay)


def estimate_context_tokens(
    document_prompt: str, chunk_prompt: str, max_tokens: int
) -> int:
    """Tokens a context call counts against the model's quota: the prompt
    (estimated from its length) plus max_tokens, which Azure reserves."""
    return (len(document_prompt) + len(chunk_prompt)) // CHARS_PER_TOKEN + max_tokens


def _context_budget() -> Optional[TokenBudget]:
    tokens_per_minute = _int_from_env(
        "INDEXER_CONTEXT_TOKENS_PER_MINUTE", DEFAULT_CONTEXT_TOKENS_PER_MINUTE
    )
    return TokenBudget(tokens_per_minute) if tokens_per_minute > 0 else None


context_budget: Optional[TokenBudget] = _context_budget()


def shutdown_pools() -> None:
    """Stop the pools at service shutdown; queued calls are cancelled."""
    for pool in (context_pool, embedding_pool):
        pool.shutdown(wait=False, cancel_futures=True)

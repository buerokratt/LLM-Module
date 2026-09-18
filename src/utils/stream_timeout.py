"""Stream timeout utilities for async streaming operations."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Union

from src.llm_orchestrator_config.exceptions import StreamTimeoutError

# An SSE comment frame. Ignored by EventSource and by the notification server's
# relay (which only forwards `data: ` lines), but it is bytes on the wire, which
# is what keeps proxy idle timers from closing a slow stream.
HEARTBEAT_FRAME = ": ping\n\n"


class _StreamExhausted:
    """Sentinel type marking a normally-completed source iterator.

    A dedicated class rather than a bare ``object()`` so that an ``isinstance``
    check narrows the value back to ``str`` for type checkers.
    """


_STREAM_EXHAUSTED = _StreamExhausted()


async def _next_or_sentinel(
    iterator: AsyncIterator[str],
) -> Union[str, _StreamExhausted]:
    """Advance an async iterator, returning a sentinel instead of raising at the end.

    Returning a sentinel keeps StopAsyncIteration out of the asyncio.Task that
    wraps this call, where it would be an awkward special case.
    """
    try:
        return await iterator.__anext__()
    except StopAsyncIteration:
        return _STREAM_EXHAUSTED


@asynccontextmanager
async def stream_timeout(seconds: int) -> AsyncIterator[None]:
    """
    Context manager for stream timeout enforcement.

    Args:
        seconds: Maximum duration in seconds

    Raises:
        StreamTimeoutError: When timeout is exceeded

    Example:
        async with stream_timeout(300):
            async for chunk in stream_generator():
                yield chunk
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except asyncio.TimeoutError as e:
        raise StreamTimeoutError(
            f"Stream exceeded maximum duration of {seconds} seconds"
        ) from e


async def with_heartbeat(
    source: AsyncIterator[str],
    heartbeat_interval: float,
    idle_timeout: float,
) -> AsyncIterator[str]:
    """
    Relay a stream, emitting SSE comment frames during quiet periods.

    Two problems are solved together. A long pause between chunks lets any proxy
    on the path close the connection, so we keep writing; and a stream that has
    genuinely stalled should fail fast rather than sit until the total-duration
    cap expires, so we enforce an idle budget.

    Both timers measure the gap *between* chunks - a long answer that keeps
    producing is never interrupted, however long it runs in total.

    Args:
        source: The upstream chunk iterator.
        heartbeat_interval: Seconds of quiet before emitting a heartbeat frame.
        idle_timeout: Seconds of continuous quiet before giving up.

    Yields:
        Chunks from ``source``, interleaved with ``HEARTBEAT_FRAME``.

    Raises:
        StreamTimeoutError: If no chunk arrives for ``idle_timeout`` seconds.
    """
    iterator = source.__aiter__()
    pending: Optional["asyncio.Task[Union[str, _StreamExhausted]]"] = None

    try:
        while True:
            pending = asyncio.ensure_future(_next_or_sentinel(iterator))
            idle_elapsed = 0.0

            while True:
                try:
                    # Shielded so a heartbeat timeout does not cancel the pull;
                    # the same task is awaited again on the next pass.
                    chunk = await asyncio.wait_for(
                        asyncio.shield(pending), heartbeat_interval
                    )
                    break
                except asyncio.TimeoutError:
                    idle_elapsed += heartbeat_interval
                    if idle_elapsed >= idle_timeout:
                        pending.cancel()
                        raise StreamTimeoutError(
                            f"Stream produced no output for {idle_elapsed:.1f} "
                            f"seconds (idle limit {idle_timeout:.1f}s)"
                        ) from None
                    yield HEARTBEAT_FRAME

            if isinstance(chunk, _StreamExhausted):
                return

            yield chunk
    finally:
        # Covers early consumer exit (client disconnect) as well as errors.
        if pending is not None and not pending.done():
            pending.cancel()

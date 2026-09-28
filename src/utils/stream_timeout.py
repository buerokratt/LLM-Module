"""Stream timeout utilities for async streaming operations."""

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any, AsyncIterator, Optional, Tuple

from src.llm_orchestrator_config.exceptions import StreamTimeoutError

# An SSE comment frame. Ignored by EventSource and by the notification server's
# relay (which only forwards `data: ` lines), but it is bytes on the wire, which
# is what keeps proxy idle timers from closing a slow stream.
HEARTBEAT_FRAME = ": ping\n\n"


# Queue message kinds exchanged between the pump task and the relay loop.
_CHUNK = "chunk"
_DONE = "done"
_ERROR = "error"


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
    # The source is drained by a single long-lived task rather than one task per
    # chunk. Advancing an async generator from a different task on each pull
    # splits any task-affine state it holds across a `yield`: anyio cancel scopes
    # (used by the NeMo/DSPy streaming stack) raise "Attempted to exit cancel
    # scope in a different task", and OTel context tokens fail to detach. Pinning
    # the whole `async for` to one task keeps both paired correctly.
    #
    # maxsize=1 preserves backpressure - the pump stays at most one chunk ahead
    # of the consumer instead of buffering a whole answer in memory.
    queue: "asyncio.Queue[Tuple[str, Any]]" = asyncio.Queue(maxsize=1)

    async def pump() -> None:
        try:
            async for chunk in source:
                await queue.put((_CHUNK, chunk))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer side
            await queue.put((_ERROR, exc))
        else:
            await queue.put((_DONE, None))

    pump_task: Optional["asyncio.Task[None]"] = asyncio.ensure_future(pump())

    try:
        while True:
            idle_elapsed = 0.0

            while True:
                try:
                    # Only the queue read is timed out. The source pull is never
                    # cancelled or restarted, so a heartbeat cannot disturb it.
                    kind, value = await asyncio.wait_for(
                        queue.get(), heartbeat_interval
                    )
                    break
                except asyncio.TimeoutError:
                    idle_elapsed += heartbeat_interval
                    if idle_elapsed >= idle_timeout:
                        raise StreamTimeoutError(
                            f"Stream produced no output for {idle_elapsed:.1f} "
                            f"seconds (idle limit {idle_timeout:.1f}s)"
                        ) from None
                    yield HEARTBEAT_FRAME

            if kind == _DONE:
                return
            if kind == _ERROR:
                raise value

            yield value
    finally:
        # Covers early consumer exit (client disconnect), timeout and errors.
        # Awaiting the cancellation matters: it lets the source generator unwind
        # inside the pump task, which is the task that entered its cancel scopes.
        if pump_task is not None and not pump_task.done():
            pump_task.cancel()
            with suppress(BaseException):
                await pump_task

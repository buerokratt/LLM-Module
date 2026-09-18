"""Stream Manager - Centralized tracking and lifecycle management for streaming responses."""

from typing import Dict, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio
from src.loki_logger import LokiLogger
from pydantic import BaseModel, Field, ConfigDict

from src.llm_orchestrator_config.stream_config import StreamConfig
from src.llm_orchestrator_config.exceptions import StreamError
from src.utils.error_utils import generate_error_id

# Initialize Loki logger
logger = LokiLogger(service_name="stream-manager")


class StreamContext(BaseModel):
    """Context for tracking a single stream's lifecycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow AsyncIterator type

    stream_id: str
    chat_id: str
    author_id: str
    start_time: datetime
    token_count: int = 0
    status: str = Field(
        default="active", description="active, completed, error, timeout, cancelled"
    )
    error_id: Optional[str] = None
    bot_generator: Optional[AsyncIterator[str]] = Field(
        default=None, exclude=True, repr=False
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring."""
        return {
            "stream_id": self.stream_id,
            "chat_id": self.chat_id,
            "author_id": self.author_id,
            "start_time": self.start_time.isoformat(),
            "token_count": self.token_count,
            "status": self.status,
            "error_id": self.error_id,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }

    async def cleanup(self) -> None:
        """Clean up resources associated with this stream."""
        if self.bot_generator is not None:
            try:
                logger.debug(f"[{self.stream_id}] Closing bot generator")
                # AsyncIterator might be AsyncGenerator which has aclose()
                if hasattr(self.bot_generator, "aclose"):
                    await self.bot_generator.aclose()  # type: ignore
                    logger.debug(
                        f"[{self.stream_id}] Bot generator closed successfully"
                    )
            except Exception as e:
                # Expected during normal completion or cancellation
                logger.debug(
                    f"[{self.stream_id}] Generator cleanup exception (may be normal): {e}"
                )
            finally:
                self.bot_generator = None

    def mark_completed(self) -> None:
        """Mark stream as successfully completed."""
        self.status = "completed"
        logger.info(
            f"[{self.stream_id}] Stream completed successfully "
            f"({self.token_count} tokens, "
            f"{(datetime.now() - self.start_time).total_seconds():.2f}s)"
        )

    def mark_error(self, error_id: str) -> None:
        """Mark stream as failed with error."""
        self.status = "error"
        self.error_id = error_id
        logger.error(
            f"[{self.stream_id}] Stream failed with error_id={error_id} "
            f"({self.token_count} tokens generated before failure)"
        )

    def mark_timeout(self) -> None:
        """Mark stream as timed out."""
        self.status = "timeout"
        logger.warning(
            f"[{self.stream_id}] Stream timed out "
            f"({self.token_count} tokens, "
            f"{(datetime.now() - self.start_time).total_seconds():.2f}s)"
        )

    def mark_cancelled(self) -> None:
        """Mark stream as cancelled (client disconnect)."""
        self.status = "cancelled"
        logger.info(
            f"[{self.stream_id}] Stream cancelled by client "
            f"({self.token_count} tokens, "
            f"{(datetime.now() - self.start_time).total_seconds():.2f}s)"
        )


class StreamManager:
    """
    Singleton manager for tracking and managing active streaming connections.

    Features:
    - Concurrent stream limiting (system-wide and per-user)
    - Stream lifecycle tracking
    - Guaranteed resource cleanup
    - Operational visibility and debugging
    """

    _instance: Optional["StreamManager"] = None

    def __new__(cls) -> "StreamManager":
        """Singleton pattern - ensure only one manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the stream manager."""
        if not hasattr(self, "_initialized"):
            self._streams: Dict[str, StreamContext] = {}
            self._user_streams: Dict[
                str, set[str]
            ] = {}  # author_id -> set of stream_ids
            self._registry_lock = asyncio.Lock()
            self._initialized = True
            logger.info("StreamManager initialized")

    def _generate_stream_id(self) -> str:
        """Generate unique stream ID."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        import random
        import string

        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"stream-{timestamp}-{suffix}"

    async def check_capacity(self, author_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if new stream can be created within capacity limits.

        Args:
            author_id: User identifier

        Returns:
            Tuple of (can_create, error_message)
        """
        async with self._registry_lock:
            total_streams = len(self._streams)
            user_streams = len(self._user_streams.get(author_id, set()))

            # Check system-wide limit
            if total_streams >= StreamConfig.MAX_CONCURRENT_STREAMS:
                error_msg = (
                    f"Service at capacity ({total_streams}/{StreamConfig.MAX_CONCURRENT_STREAMS} "
                    f"concurrent streams). Please retry in a moment."
                )
                logger.warning(
                    f"Stream capacity exceeded: {total_streams}/{StreamConfig.MAX_CONCURRENT_STREAMS}"
                )
                return False, error_msg

            # Check per-user limit
            if user_streams >= StreamConfig.MAX_STREAMS_PER_USER:
                error_msg = (
                    f"You have reached the maximum of {StreamConfig.MAX_STREAMS_PER_USER} "
                    f"concurrent streams. Please wait for existing streams to complete."
                )
                logger.warning(
                    f"User {author_id} exceeded stream limit: "
                    f"{user_streams}/{StreamConfig.MAX_STREAMS_PER_USER}"
                )
                return False, error_msg

            return True, None

    async def register_stream(self, chat_id: str, author_id: str) -> StreamContext:
        """
        Register a new stream and return its context.

        Args:
            chat_id: Chat identifier
            author_id: User identifier

        Returns:
            StreamContext for the new stream
        """
        async with self._registry_lock:
            stream_id = self._generate_stream_id()

            ctx = StreamContext(
                stream_id=stream_id,
                chat_id=chat_id,
                author_id=author_id,
                start_time=datetime.now(),
            )

            self._streams[stream_id] = ctx

            # Track user streams
            if author_id not in self._user_streams:
                self._user_streams[author_id] = set()
            self._user_streams[author_id].add(stream_id)

            logger.info(
                f"[{stream_id}] Stream registered: "
                f"chatId={chat_id}, authorId={author_id}, "
                f"total_streams={len(self._streams)}, "
                f"user_streams={len(self._user_streams[author_id])}"
            )

            return ctx

    async def unregister_stream(self, stream_id: str) -> None:
        """
        Unregister a stream from tracking.

        Args:
            stream_id: Stream identifier
        """
        async with self._registry_lock:
            ctx = self._streams.get(stream_id)
            if ctx is None:
                logger.warning(f"[{stream_id}] Attempted to unregister unknown stream")
                return

            # Remove from main registry
            del self._streams[stream_id]

            # Remove from user tracking
            author_id = ctx.author_id
            if author_id in self._user_streams:
                self._user_streams[author_id].discard(stream_id)
                if not self._user_streams[author_id]:
                    del self._user_streams[author_id]

            logger.info(
                f"[{stream_id}] Stream unregistered: "
                f"status={ctx.status}, "
                f"tokens={ctx.token_count}, "
                f"duration={(datetime.now() - ctx.start_time).total_seconds():.2f}s, "
                f"remaining_streams={len(self._streams)}"
            )

    @asynccontextmanager
    async def managed_stream(
        self, chat_id: str, author_id: str
    ) -> AsyncIterator[StreamContext]:
        """
        Context manager for stream lifecycle management with guaranteed cleanup.

        Usage:
            async with stream_manager.managed_stream(chat_id, author_id) as ctx:
                ctx.bot_generator = some_async_generator()
                async for token in ctx.bot_generator:
                    ctx.token_count += len(token) // 4
                    yield token
                ctx.mark_completed()

        Args:
            chat_id: Chat identifier
            author_id: User identifier

        Yields:
            StreamContext for the managed stream
        """
        # Check capacity before registering
        can_create, error_msg = await self.check_capacity(author_id)
        if not can_create:
            # Create a minimal error context without registering
            error_id = generate_error_id()
            logger.error(
                f"Stream creation rejected for chatId={chat_id}, authorId={author_id}: {error_msg}",
                extra={"error_id": error_id},
            )
            raise StreamError(f"Cannot create stream: {error_msg}", error_id=error_id)

        # Register the stream
        ctx = await self.register_stream(chat_id, author_id)

        try:
            yield ctx
        except GeneratorExit:
            # Client disconnected
            ctx.mark_cancelled()
            raise
        except Exception as e:
            # Any other error - will be handled by caller with error_id
            if not ctx.error_id:
                # Mark error if not already marked
                error_id = getattr(e, "error_id", generate_error_id())
                ctx.mark_error(error_id)
            raise
        finally:
            # GUARANTEED cleanup - runs in all cases
            await ctx.cleanup()
            await self.unregister_stream(ctx.stream_id)

    async def get_active_streams(self) -> int:
        """Get count of active streams."""
        async with self._registry_lock:
            return len(self._streams)

    async def get_user_streams(self, author_id: str) -> int:
        """Get count of active streams for a specific user."""
        async with self._registry_lock:
            return len(self._user_streams.get(author_id, set()))

    async def get_stream_info(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific stream."""
        async with self._registry_lock:
            ctx = self._streams.get(stream_id)
            return ctx.to_dict() if ctx else None

    async def get_all_stream_info(self) -> list[Dict[str, Any]]:
        """Get information about all active streams."""
        async with self._registry_lock:
            return [ctx.to_dict() for ctx in self._streams.values()]

    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics about streaming."""
        async with self._registry_lock:
            total_streams = len(self._streams)
            total_users = len(self._user_streams)

            status_counts: Dict[str, int] = {}
            for ctx in self._streams.values():
                status_counts[ctx.status] = status_counts.get(ctx.status, 0) + 1

            return {
                "total_active_streams": total_streams,
                "total_active_users": total_users,
                "status_breakdown": status_counts,
                "capacity_used_pct": (
                    total_streams / StreamConfig.MAX_CONCURRENT_STREAMS
                )
                * 100,
                "max_concurrent_streams": StreamConfig.MAX_CONCURRENT_STREAMS,
                "max_streams_per_user": StreamConfig.MAX_STREAMS_PER_USER,
            }


# Global singleton instance
stream_manager = StreamManager()

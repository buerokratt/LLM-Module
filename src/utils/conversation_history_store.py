"""Redis-backed conversation history store."""

import asyncio
import json
import weakref
from typing import TYPE_CHECKING, Optional, Union, cast

from src.loki_logger import LokiLogger
from redis import WatchError

from src.models.conversation_history_models import (
    ConversationHistoryState,
    ConversationRound,
)
from src.utils.redis_client import get_redis_client

if TYPE_CHECKING:
    from src.models.request_models import (
        OrchestrationResponse,
        TestOrchestrationResponse,
    )
    from src.utils.conversation_summary_generator import SummarizerCallable

logger = LokiLogger(service_name="conversation-history")

_HISTORY_KEY_PREFIX = "conv:"
_SUMMARY_KEY_PREFIX = "conv:summary:"
_HISTORY_TTL_SECONDS = 1800  # 30 minutes, sliding
_MAX_ROUNDS = 10
_APPEND_MAX_RETRIES = 3


def _history_key(chat_id: str) -> str:
    return f"{_HISTORY_KEY_PREFIX}{chat_id}"


def _summary_key(chat_id: str) -> str:
    return f"{_SUMMARY_KEY_PREFIX}{chat_id}"


class ConversationHistoryStore:
    """CRUD store for per-session conversation history backed by Redis.

    All operations are async and safe to call from FastAPI handlers.
    The TTL is reset (sliding expiry) on every write that touches a key.

    Key layout (db=1, same as session store):
      ``conv:{chat_id}``         — JSON list of up to 10 ``ConversationRound`` objects
      ``conv:summary:{chat_id}`` — plain string summary (optional)

    An optional *summarizer* callable is injected at construction time.  When
    trimming evicts rounds (``len(rounds) > _MAX_ROUNDS``), a background
    ``asyncio.Task`` is created to merge the evicted rounds into the existing
    summary via the summarizer.  If *summarizer* is ``None``, trimming still
    occurs but no summary is generated.
    """

    def __init__(
        self,
        summarizer: Optional["SummarizerCallable"] = None,
    ) -> None:
        """Initialise the store.

        Args:
            summarizer: Optional async callable that merges evicted rounds into
                the running conversation summary.  See
                :func:`~src.utils.conversation_summary_generator.create_incremental_summarizer`
                for a factory that creates one.
        """
        self._summarizer = summarizer
        # Hold strong references to background tasks to prevent GC collection
        # before they complete (asyncio tasks are only weakly referenced by the
        # event loop).
        self._pending_tasks: set[asyncio.Task[None]] = set()
        # Per-chat locks to serialize summary updates and prevent concurrent
        # write races when multiple save_round() calls trigger evictions.
        # WeakValueDictionary allows entries to be GC'd once no task holds a
        # strong reference, preventing unbounded growth under high-cardinality
        # chat_ids.
        self._summary_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    async def save_round(self, chat_id: str, round: ConversationRound) -> None:
        """Append a round to the history, trim to ``_MAX_ROUNDS``, reset both TTLs.

        Uses optimistic locking (WATCH/MULTI/EXEC) to detect concurrent writes and
        retries up to ``_APPEND_MAX_RETRIES`` times on conflict.

        Args:
            chat_id: The conversation identifier.
            round: The completed user+bot exchange to persist.
        """
        client = get_redis_client()
        if client is None:
            logger.warning(
                f"[ConversationHistoryStore] Redis unavailable - save_round({chat_id}) skipped"
            )
            return

        hkey = _history_key(chat_id)
        skey = _summary_key(chat_id)

        for attempt in range(_APPEND_MAX_RETRIES):
            try:
                async with client.pipeline(transaction=True) as pipe:
                    await pipe.watch(hkey)

                    raw = await pipe.get(hkey)
                    if raw is not None:
                        rounds: list[dict] = json.loads(raw)
                    else:
                        rounds = []

                    rounds.append(round.model_dump())

                    # Capture rounds that will be evicted before trimming.
                    evicted: list[ConversationRound] = []
                    if len(rounds) > _MAX_ROUNDS:
                        evicted = [
                            ConversationRound.model_validate(r)
                            for r in rounds[: len(rounds) - _MAX_ROUNDS]
                        ]
                        rounds = rounds[-_MAX_ROUNDS:]

                    pipe.multi()
                    pipe.set(hkey, json.dumps(rounds), ex=_HISTORY_TTL_SECONDS)
                    # Reset summary TTL without overwriting its value
                    pipe.expire(skey, _HISTORY_TTL_SECONDS)
                    await pipe.execute()

                logger.debug(
                    f"[ConversationHistoryStore] Round saved for chat_id={chat_id}"
                )

                # Fire-and-forget incremental summary generation for evicted rounds.
                if evicted and self._summarizer is not None:
                    task = asyncio.create_task(
                        self._run_incremental_summary(chat_id, evicted)
                    )
                    self._pending_tasks.add(task)
                    task.add_done_callback(self._pending_tasks.discard)

                return

            except WatchError:
                logger.debug(
                    f"[ConversationHistoryStore] save_round({chat_id}) - concurrent modification "
                    f"detected, retrying (attempt {attempt + 1}/{_APPEND_MAX_RETRIES})"
                )
                continue
            except Exception as exc:
                logger.error(
                    f"[ConversationHistoryStore] save_round({chat_id}) failed: {exc}"
                )
                return

        logger.error(
            f"[ConversationHistoryStore] save_round({chat_id}) - exhausted {_APPEND_MAX_RETRIES} retries due to "
            f"concurrent writes"
        )

    def _get_summary_lock(self, chat_id: str) -> asyncio.Lock:
        """Get or create a lock for serializing summary updates for this chat_id.

        Args:
            chat_id: The conversation identifier.

        Returns:
            An asyncio.Lock that serializes summary merges for this chat_id.
        """
        if chat_id not in self._summary_locks:
            self._summary_locks[chat_id] = asyncio.Lock()
        return self._summary_locks[chat_id]

    async def _run_incremental_summary(
        self,
        chat_id: str,
        evicted_rounds: list[ConversationRound],
    ) -> None:
        """Background task: merge *evicted_rounds* into the stored summary.

        Acquires a per-chat lock to serialize summary updates for the same
        chat_id. This ensures that concurrent evictions do not lose information
        due to simultaneous reads and writes. Only one summarizer runs at a time
        for each chat_id, making summary merges deterministic.

        Fetches the current summary, calls the injected summarizer, and persists
        the result.  All exceptions are caught so the task never propagates.

        Args:
            chat_id: The conversation identifier.
            evicted_rounds: Rounds that were just trimmed from the active window.
        """
        lock = self._get_summary_lock(chat_id)
        try:
            async with lock:
                existing_summary = await self.get_summary(chat_id)
                updated = await self._summarizer(existing_summary, evicted_rounds)  # type: ignore[misc]
                if updated:
                    await self.save_summary(chat_id, updated)
                    logger.debug(
                        f"[ConversationHistoryStore] Incremental summary updated for chat_id={chat_id}"
                    )
        except Exception as exc:
            logger.error(
                f"[ConversationHistoryStore] _run_incremental_summary({chat_id}) failed: {exc}"
            )

    async def get_history(self, chat_id: str) -> list[ConversationRound]:
        """Retrieve the stored rounds for a conversation.

        Returns:
            Ordered list of ``ConversationRound`` objects (newest last),
            or an empty list if the key is missing or Redis is unavailable.
        """
        client = get_redis_client()
        if client is None:
            logger.warning(
                f"[ConversationHistoryStore] Redis unavailable - get_history({chat_id}) skipped"
            )
            return []

        try:
            raw = await client.get(_history_key(chat_id))
            if raw is None:
                return []
            return [ConversationRound.model_validate(r) for r in json.loads(raw)]
        except Exception as exc:
            logger.error(
                f"[ConversationHistoryStore] get_history({chat_id}) failed: {exc}"
            )
            return []

    async def get_summary(self, chat_id: str) -> Optional[str]:
        """Retrieve the optional summary for a conversation.

        Returns:
            The summary string, or None if not set or Redis is unavailable.
        """
        client = get_redis_client()
        if client is None:
            logger.warning(
                f"[ConversationHistoryStore] Redis unavailable - get_summary({chat_id}) skipped"
            )
            return None

        try:
            raw = await client.get(_summary_key(chat_id))
            return raw if raw is not None else None
        except Exception as exc:
            logger.error(
                f"[ConversationHistoryStore] get_summary({chat_id}) failed: {exc}"
            )
            return None

    async def save_summary(self, chat_id: str, summary: str) -> None:
        """Persist a summary string and reset the TTL on both keys.

        Args:
            chat_id: The conversation identifier.
            summary: The condensed text to store.
        """
        client = get_redis_client()
        if client is None:
            logger.warning(
                f"[ConversationHistoryStore] Redis unavailable - save_summary({chat_id}) skipped"
            )
            return

        hkey = _history_key(chat_id)
        skey = _summary_key(chat_id)

        try:
            async with client.pipeline(transaction=False) as pipe:
                pipe.set(skey, summary, ex=_HISTORY_TTL_SECONDS)
                # Reset history key TTL to keep both keys in sync
                pipe.expire(hkey, _HISTORY_TTL_SECONDS)
                await pipe.execute()
            logger.debug(
                f"[ConversationHistoryStore] Summary saved for chat_id={chat_id}"
            )
        except Exception as exc:
            logger.error(
                f"[ConversationHistoryStore] save_summary({chat_id}) failed: {exc}"
            )

    async def get_context(self, chat_id: str) -> ConversationHistoryState:
        """Return the full conversation context (rounds + summary) for a chat.

        Fetches both keys concurrently via ``asyncio.gather``.

        Returns:
            A ``ConversationHistoryState`` instance. Always succeeds — both
            fields fall back to safe defaults if Redis is unavailable.
        """
        rounds, summary = await asyncio.gather(
            self.get_history(chat_id),
            self.get_summary(chat_id),
        )
        return ConversationHistoryState(
            chat_id=chat_id,
            rounds=rounds,
            summary=summary,
        )


def should_save_history(
    conversation_history_store: Optional["ConversationHistoryStore"],
    response: Union["OrchestrationResponse", "TestOrchestrationResponse"],
    excluded_messages: frozenset[str],
) -> bool:
    """Return True when a successful exchange should be persisted to history.

    Args:
        conversation_history_store: The active store instance, or None when Redis is unavailable.
        response: The response produced by the orchestration pipeline.
        excluded_messages: Set of content strings that must never be persisted (OOS, error, etc.).
    """
    if conversation_history_store is None:
        return False
    # Use duck typing to distinguish response types, avoiding isinstance() issues
    # caused by import path aliasing (models.request_models vs src.models.request_models).
    # OrchestrationResponse has chatId; TestOrchestrationResponse does not.
    if not hasattr(response, "chatId"):
        # TestOrchestrationResponse (testing env) — skip history.
        return False

    # After hasattr check, safely access chatId via cast for type safety
    orch_response = cast("OrchestrationResponse", response)
    if orch_response.chatId is None:
        return False
    if response.inputGuardFailed or response.questionOutOfLLMScope:
        return False
    if response.content in excluded_messages:
        return False
    return True


async def save_history_round(
    store: ConversationHistoryStore,
    chat_id: str,
    user_message: str,
    bot_message: str,
) -> None:
    """Persist a completed user+bot exchange to Redis. Never raises.

    Args:
        store: The active ConversationHistoryStore.
        chat_id: Conversation identifier.
        user_message: The user's original message.
        bot_message: The bot's full response.
    """
    try:
        round_ = ConversationRound(
            user_message=user_message,
            bot_message=bot_message,
        )
        await store.save_round(chat_id, round_)
        logger.debug(
            f"[{chat_id}] Conversation history round saved ({len(bot_message)} chars)"
        )
    except Exception as exc:
        logger.warning(f"[{chat_id}] Failed to save conversation history round: {exc}")

"""Shared helper for fetching conversation history from Redis.

Mirrors the pattern established by ``ContextWorkflowExecutor._build_history()``
so that all workflow entry points (RAG, service, ATC) resolve history in the
same way: Redis is preferred over the GUI-supplied request payload, and any
running conversation summary stored alongside the rounds is surfaced to callers
so they can skip an expensive LLM summarisation step.
"""

from typing import List, Optional

from src.loki_logger import LokiLogger

from src.models.conversation_history_models import ConversationHistoryState
from models.request_models import ConversationItem
from src.utils.conversation_history_store import ConversationHistoryStore

logger = LokiLogger(service_name="conversation-history")


async def get_conversation_history(
    chat_id: str,
    store: Optional[ConversationHistoryStore],
    fallback: List[ConversationItem],
) -> tuple[List[ConversationItem], Optional[str]]:
    """Fetch conversation history, preferring Redis over the request payload.

    When *store* is provided and the Redis session has rounds, those rounds are
    returned as the authoritative history and *fallback* is ignored.  Any running
    summary attached to the stored state is returned as the second tuple element
    so that callers can skip an expensive LLM summarisation step.

    If the store is absent, raises, or contains no rounds the function returns
    *fallback* with ``None`` as the summary.

    Args:
        chat_id: The conversation identifier.
        store: Optional Redis-backed conversation history store.
        fallback: ``request.conversationHistory`` — used when Redis is unavailable
            or has no rounds for this session.

    Returns:
        ``(history, summary)`` where *history* is a list of
        :class:`~src.models.request_models.ConversationItem` objects (two per
        stored round: one ``"user"`` and one ``"bot"`` item) and *summary* is
        the Redis summary string or ``None``.
    """
    if store is not None:
        try:
            state: ConversationHistoryState = await store.get_context(chat_id)
            if state.rounds:
                history: List[ConversationItem] = []
                for round_ in state.rounds:
                    history.append(
                        ConversationItem(
                            authorRole="user",
                            message=round_.user_message,
                            timestamp=str(round_.timestamp),
                        )
                    )
                    history.append(
                        ConversationItem(
                            authorRole="bot",
                            message=round_.bot_message,
                            timestamp=str(round_.timestamp),
                        )
                    )
                logger.debug(
                    f"[{chat_id}] Using Redis history: {len(state.rounds)} rounds, "
                    f"summary={'present' if state.summary else 'absent'}"
                )
                return history, state.summary
        except Exception as exc:
            logger.warning(
                f"[{chat_id}] Redis history fetch failed, falling back to request history: {exc}"
            )

    return fallback, None

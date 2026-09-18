"""Two-tier Redis response cache for the ATC workflow."""

import hashlib
import json
from typing import Any

from src.loki_logger import LokiLogger

from models.session_models import LastCallContext
from tool_classifier.constants import (
    ATC_CACHE_DEFAULT_TTL_SECONDS,
    ATC_CACHE_KEY_PREFIX,
    ATC_LAST_CALL_KEY_PREFIX,
    ATC_LAST_CALL_TTL_SECONDS,
)
from src.utils.redis_client import get_redis_client

logger = LokiLogger(service_name="atc-cache-store")


class ATCCacheStore:
    """Two-tier response cache for the ATC workflow.

    Tier 1 (L1) — Exact response cache
        Key:   atc:cache:{chat_id}:{api_name}:{param_hash}
        Value: raw API response JSON
        TTL:   per-endpoint ``cache_ttl_seconds`` or ``ATC_CACHE_DEFAULT_TTL_SECONDS``

    Tier 2 (L2) — Last call context
        Key:   atc:last:{chat_id}
        Value: JSON ``list[LastCallContext]``
        TTL:   ``ATC_LAST_CALL_TTL_SECONDS`` (sliding — reset on every write)

    All public methods are async and fail-open: a Redis unavailability or
    serialisation error is logged as a warning and the caller receives
    ``None`` / silent no-op rather than an exception.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_params(params: dict[str, Any]) -> dict[str, Any]:
        """Normalise param values so equivalent inputs produce the same hash.

        Rules applied per string value:
        - Strip leading/trailing whitespace.
        - If the stripped string is purely numeric (``"2026"``), cast to ``int``.
        - If the stripped string is all-alpha with no spaces (enum-like), lowercase it.
        """
        result: dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, str):
                stripped = v.strip()
                if stripped.isdigit():
                    result[k] = int(stripped)
                elif stripped.isalpha():
                    result[k] = stripped.lower()
                else:
                    result[k] = stripped
            else:
                result[k] = v
        return result

    @staticmethod
    def _param_hash(params: dict[str, Any]) -> str:
        """Return a 16-char hex digest of the normalised, sorted params dict."""
        normalised = ATCCacheStore._normalise_params(params)
        serialised = json.dumps(normalised, sort_keys=True)
        return hashlib.sha256(serialised.encode()).hexdigest()[:16]

    @staticmethod
    def _l1_key(chat_id: str, api_name: str, params: dict[str, Any]) -> str:
        return f"{ATC_CACHE_KEY_PREFIX}:{chat_id}:{api_name}:{ATCCacheStore._param_hash(params)}"

    @staticmethod
    def _l2_key(chat_id: str) -> str:
        return f"{ATC_LAST_CALL_KEY_PREFIX}:{chat_id}"

    # ------------------------------------------------------------------
    # L1 — Exact response cache
    # ------------------------------------------------------------------

    async def get_l1(
        self, chat_id: str, api_name: str, params: dict[str, Any]
    ) -> Any | None:
        """Return the cached raw API response, or ``None`` on cache miss or error."""
        client = get_redis_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._l1_key(chat_id, api_name, params))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning(
                f"[ATCCache] get_l1 failed: chat_id={chat_id} api_name={api_name!r} error={exc}"
            )
            return None

    async def set_l1(
        self,
        chat_id: str,
        api_name: str,
        params: dict[str, Any],
        raw_response: Any,
        ttl: int = ATC_CACHE_DEFAULT_TTL_SECONDS,
    ) -> None:
        """Serialise and store a raw API response in L1.

        Args:
            chat_id: Conversation identifier.
            api_name: Endpoint name (snake_case, matches ``EnrichedEndpoint.name``).
            params: The collected param values used for this API call.
            raw_response: Parsed API JSON (dict or list).
            ttl: Cache TTL in seconds. Defaults to ``ATC_CACHE_DEFAULT_TTL_SECONDS``.
        """
        client = get_redis_client()
        if client is None:
            return
        try:
            await client.set(
                self._l1_key(chat_id, api_name, params),
                json.dumps(raw_response),
                ex=ttl,
            )
            logger.debug(
                f"[ATCCache] L1 set: chat_id={chat_id} api_name={api_name!r} ttl={ttl}s"
            )
        except Exception as exc:
            logger.warning(
                f"[ATCCache] set_l1 failed: chat_id={chat_id} api_name={api_name!r} error={exc}"
            )

    # ------------------------------------------------------------------
    # L2 — Last call context
    # ------------------------------------------------------------------

    async def get_l2(self, chat_id: str) -> list[LastCallContext] | None:
        """Return the last-call context list, or ``None`` on miss or error."""
        client = get_redis_client()
        if client is None:
            return None
        try:
            raw = await client.get(self._l2_key(chat_id))
            if raw is None:
                return None
            data: list[dict[str, Any]] = json.loads(raw)
            return [LastCallContext.model_validate(item) for item in data]
        except Exception as exc:
            logger.warning(f"[ATCCache] get_l2 failed: chat_id={chat_id} error={exc}")
            return None

    async def set_l2(self, chat_id: str, contexts: list[LastCallContext]) -> None:
        """Serialise and store the last-call context list in L2.

        The TTL is reset on every write (sliding expiry) so an active conversation
        always has a live L2 entry while the session exists.

        For single-intent calls pass a one-element list; for multi-intent calls
        pass one ``LastCallContext`` per succeeded endpoint.
        """
        client = get_redis_client()
        if client is None:
            return
        try:
            payload = json.dumps([ctx.model_dump() for ctx in contexts])
            await client.set(
                self._l2_key(chat_id),
                payload,
                ex=ATC_LAST_CALL_TTL_SECONDS,
            )
            logger.debug(
                f"[ATCCache] L2 set: chat_id={chat_id} entries={len(contexts)}"
            )
        except Exception as exc:
            logger.warning(f"[ATCCache] set_l2 failed: chat_id={chat_id} error={exc}")

    async def invalidate_l2(self, chat_id: str) -> None:
        """Delete the L2 last-call context key for a chat session.

        Called on intent switch so the follow-up detector does not carry stale
        context into a brand-new query. L1 entries are **not** deleted — they
        are param-hash scoped and expire on their own TTL.
        """
        client = get_redis_client()
        if client is None:
            return
        try:
            await client.delete(self._l2_key(chat_id))
            logger.debug(f"[ATCCache] L2 invalidated: chat_id={chat_id}")
        except Exception as exc:
            logger.warning(
                f"[ATCCache] invalidate_l2 failed: chat_id={chat_id} error={exc}"
            )

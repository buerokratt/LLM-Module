"""Relay the orchestration SSE stream to the central notifications node."""

import asyncio
import codecs
import json
import logging
from typing import AsyncIterable, Protocol
from uuid import uuid4

import aiohttp

logger = logging.getLogger(__name__)


class Publisher(Protocol):
    async def publish(
        self, chat_id: str, event_type: str, payload: dict[str, object]
    ) -> None: ...


class NotificationPublisher:
    """Publish events to the notifications node's private API."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession) -> None:
        self._url = f"{base_url.rstrip('/')}/private/v1/notifications/events"
        self._session = session

    async def publish(
        self, chat_id: str, event_type: str, payload: dict[str, object]
    ) -> None:
        event = {
            "eventUuid": str(uuid4()),
            "recipient": "CHAT",
            "recipientUuid": [chat_id],
            "type": event_type,
            "payload": payload,
        }
        async with self._session.post(
            self._url,
            json=event,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as response:
            response.raise_for_status()


class StreamRelay:
    """Convert SSE frames into notification events without exposing raw errors."""

    def __init__(self, publisher: Publisher, chat_id: str, request_id: str) -> None:
        self.publisher = publisher
        self.chat_id = chat_id
        self.request_id = request_id

    async def _publish(self, event_type: str, **payload: object) -> None:
        await self.publisher.publish(
            self.chat_id,
            event_type,
            {"channelId": self.chat_id, "requestId": self.request_id, **payload},
        )

    async def run(self, body: AsyncIterable[str | bytes | memoryview]) -> None:
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            await self._publish("stream_start")
            async for chunk in body:
                buffer += chunk if isinstance(chunk, str) else decoder.decode(chunk)
                buffer = buffer.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    lines = [
                        line[5:].lstrip()
                        for line in frame.split("\n")
                        if line.startswith("data:")
                    ]
                    if not lines:
                        continue
                    content = json.loads("\n".join(lines))["payload"]
                    text = content.get("content")
                    if text == "END":
                        await self._publish("stream_complete")
                        return
                    if text or content.get("buttons"):
                        payload = {"content": text or ""}
                        if content.get("buttons"):
                            payload["buttons"] = content["buttons"]
                        await self._publish("stream_chunk", **payload)
            raise RuntimeError("Stream ended without END marker")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "Notification relay failed for request %s", self.request_id
            )
            try:
                await self._publish(
                    "stream_error", error="Unable to complete the stream"
                )
            except Exception:
                logger.exception(
                    "Could not publish stream error for request %s", self.request_id
                )
        finally:
            aclose = getattr(body, "aclose", None)
            if aclose is not None:
                await aclose()

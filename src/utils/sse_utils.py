"""Utilities for parsing Server-Sent Events (SSE) messages."""

import json
from typing import Optional


def extract_content_from_sse(sse_chunk: str) -> Optional[str]:
    """Parse an SSE chunk and return payload.content, or None on failure."""
    if not sse_chunk.startswith("data: "):
        return None
    json_part = sse_chunk[len("data: ") :].strip()
    try:
        parsed = json.loads(json_part)
        return parsed.get("payload", {}).get("content")
    except (json.JSONDecodeError, AttributeError):
        return None

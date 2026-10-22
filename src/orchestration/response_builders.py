"""Pure response-building helpers extracted from LLMOrchestrationService.

All functions here are stateless — they take explicit arguments and return
values with no side effects. Suitable for direct import and unit testing
without any service instance.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from models.request_models import (
    ChunkInfo,
    DocumentReference,
    OrchestrationRequest,
    OrchestrationResponse,
)
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    OUT_OF_SCOPE_MESSAGES,
    TECHNICAL_ISSUE_MESSAGES,
    get_localized_message,
)


def format_sse(
    chat_id: str,
    content: str,
    buttons: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format an SSE message with the canonical payload shape.

    Args:
        chat_id: Chat/channel identifier
        content: Content to send (token, "END", error message, etc.)
        buttons: Optional list of choice button dicts for MCQ step responses

    Returns:
        SSE-formatted string: "data: {json}\\n\\n"
    """
    inner_payload: Dict[str, Any] = {"content": content}
    if buttons:
        inner_payload["buttons"] = buttons

    payload: Dict[str, Any] = {
        "chatId": chat_id,
        "payload": inner_payload,
        "timestamp": str(int(datetime.now().timestamp() * 1000)),
        "sentTo": [],
    }
    return f"data: {json.dumps(payload)}\n\n"


def create_error_response(request: OrchestrationRequest) -> OrchestrationResponse:
    """Create a standardized technical-error response with a localized message."""
    detected_lang = getattr(request, "_detected_language", "en")
    localized_message = get_localized_message(TECHNICAL_ISSUE_MESSAGES, detected_lang)

    return OrchestrationResponse(
        chatId=request.chatId,
        llmServiceActive=False,
        questionOutOfLLMScope=False,
        inputGuardFailed=False,
        content=localized_message,
    )


def create_out_of_scope_response(
    request: OrchestrationRequest,
) -> OrchestrationResponse:
    """Create a standardized out-of-scope response with a localized message."""
    detected_lang = getattr(request, "_detected_language", "en")
    localized_message = get_localized_message(OUT_OF_SCOPE_MESSAGES, detected_lang)

    return OrchestrationResponse(
        chatId=request.chatId,
        llmServiceActive=True,
        questionOutOfLLMScope=True,
        inputGuardFailed=False,
        content=localized_message,
    )


def format_chunks_for_test_response(
    relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
) -> Optional[List[ChunkInfo]]:
    """Format retrieved chunks for a TestOrchestrationResponse.

    Args:
        relevant_chunks: List of retrieved chunks with metadata

    Returns:
        List of ChunkInfo objects with rank and content, or None if no chunks
    """
    if not relevant_chunks:
        return None

    formatted_chunks = []
    for rank, chunk in enumerate(relevant_chunks, start=1):
        chunk_text = chunk.get("text", chunk.get("content", ""))
        if isinstance(chunk_text, str) and chunk_text.strip():
            formatted_chunks.append(ChunkInfo(rank=rank, chunkRetrieved=chunk_text))

    return formatted_chunks if formatted_chunks else None


def extract_document_references(
    relevant_chunks: Optional[List[Dict[str, Union[str, float, Dict[str, Any]]]]],
) -> Optional[List[DocumentReference]]:
    """Extract unique document references from retrieved chunks.

    Args:
        relevant_chunks: List of retrieved chunks with metadata

    Returns:
        List of DocumentReference objects (deduplicated by URL), or None
    """
    if not relevant_chunks:
        return None

    seen_urls: set[str] = set()
    references: List[DocumentReference] = []

    for rank, chunk in enumerate(relevant_chunks, start=1):
        doc_url = chunk.get("document_url")
        if not doc_url:
            meta = chunk.get("meta", {})
            if isinstance(meta, dict):
                doc_url = (
                    meta.get("document_url")
                    or meta.get("source_file")
                    or meta.get("source")
                )

        if doc_url and isinstance(doc_url, str) and doc_url.strip():
            if doc_url not in seen_urls:
                seen_urls.add(doc_url)

                score_value = chunk.get("fused_score") or chunk.get("score", 0.0)
                try:
                    score = (
                        float(score_value)
                        if isinstance(score_value, (int, float))
                        else 0.0
                    )
                except (ValueError, TypeError):
                    score = 0.0

                references.append(
                    DocumentReference(
                        document_url=doc_url,
                        chunk_rank=rank,
                        relevance_score=round(score, 4),
                    )
                )

    return references if references else None

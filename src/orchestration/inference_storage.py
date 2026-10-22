"""Inference-result persistence for production/testing environments.

Extracted from LLMOrchestrationService (Group J). Persists completed
exchanges to the production store. Called by every streaming workflow
(RAG, SERVICE, ATC, CONTEXT) via the orchestration-service façade.
"""

from typing import Any, Dict, List

from src.loki_logger import LokiLogger
from models.request_models import OrchestrationRequest
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    PRODUCTION_DEPLOYMENT_ENVIRONMENT,
    TEST_DEPLOYMENT_ENVIRONMENT,
)
from src.utils.production_store import get_production_store

logger = LokiLogger(service_name="llm-orchestration-service")


async def store_streaming_inference(
    request: OrchestrationRequest,
    final_answer: str,
) -> None:
    """Store inference data for production and testing environments.

    Reads RAG data from dynamic request attributes (_rag_refined_questions,
    _rag_ranked_chunks) set by the RAG pipeline; falls back to empty lists so
    non-RAG workflows (SERVICE, ATC, CONTEXT) can call the same function.
    """
    if request.environment not in [
        PRODUCTION_DEPLOYMENT_ENVIRONMENT,
        TEST_DEPLOYMENT_ENVIRONMENT,
    ]:
        return

    try:
        refined_questions: List[str] = getattr(request, "_rag_refined_questions", [])
        ranked_chunks: List[Dict[str, Any]] = getattr(request, "_rag_ranked_chunks", [])

        embedding_scores: List[float] = []
        for chunk in ranked_chunks:
            score_value = chunk.get("fused_score", chunk.get("score", 0.0))
            try:
                embedding_scores.append(
                    float(score_value) if isinstance(score_value, (int, float)) else 0.0
                )
            except (ValueError, TypeError):
                embedding_scores.append(0.0)

        conversation_history_list = [
            {"role": item.authorRole, "content": item.message}
            for item in (request.conversationHistory or [])
        ]

        production_store = get_production_store()
        result = await production_store.store_inference_result_async(
            chat_id=request.chatId,
            user_question=request.message,
            refined_questions=refined_questions,
            conversation_history=conversation_history_list,
            ranked_chunks=ranked_chunks,
            embedding_scores=embedding_scores,
            final_answer=final_answer,
            environment=request.environment,
            vault_uuid=request.connection_id,
        )

        if result["success"]:
            logger.info(
                f"Successfully stored streaming inference for chat_id: {request.chatId}, "
                f"environment: {request.environment}, "
                f"has_rag_data: {bool(refined_questions)}"
            )
        else:
            logger.warning(
                f"Failed to store streaming inference for chat_id: {request.chatId} - "
                f"Error: {result['error']}"
            )

    except Exception as e:
        logger.error(
            f"Error storing streaming inference for chat_id: {request.chatId} - {str(e)}"
        )

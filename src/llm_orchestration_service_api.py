"""LLM Orchestration Service API - FastAPI application."""

import asyncio
import os
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import uvicorn

from llm_orchestration_service import LLMOrchestrationService
from llm_orchestrator_config.llm_manager import LLMManager
from src.utils.redis_client import (
    init_redis_client,
    close_redis_client,
    check_redis_health,
)
from src.utils.api_tool_session_store import APIToolSessionStore
from src.utils.conversation_history_store import ConversationHistoryStore
from src.utils.conversation_summary_generator import create_incremental_summarizer
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    STREAMING_ALLOWED_ENVS,
    STREAM_TIMEOUT_MESSAGE,
    RATE_LIMIT_REQUESTS_EXCEEDED_MESSAGE,
    RATE_LIMIT_TOKENS_EXCEEDED_MESSAGE,
    VALIDATION_MESSAGE_TOO_SHORT,
    VALIDATION_MESSAGE_TOO_LONG,
    VALIDATION_MESSAGE_INVALID_FORMAT,
    VALIDATION_MESSAGE_GENERIC,
    VALIDATION_CONVERSATION_HISTORY_ERROR,
    VALIDATION_REQUEST_TOO_LARGE,
    VALIDATION_REQUIRED_FIELDS_MISSING,
    VALIDATION_GENERIC_ERROR,
)
from src.llm_orchestrator_config.stream_config import StreamConfig
from src.llm_orchestrator_config.exceptions import StreamTimeoutError

# NOTE: imported via the bare package path, not "src.llm_orchestrator_config".
# Both spellings resolve to separate module objects at runtime, so the class
# imported here must match the one the config loader raises or `except` misses.
from llm_orchestrator_config.exceptions import ConfigurationError
from src.utils.stream_timeout import stream_timeout, with_heartbeat
from src.utils.observation_utils import safe_observation_context
from src.utils.error_utils import generate_error_id, log_error_with_context
from src.utils.rate_limiter import RateLimiter
from src.utils.prompt_config_loader import RefreshStatus
from models.request_models import (
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationRequest,
    TestOrchestrationResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    ContextGenerationRequest,
    ContextGenerationResponse,
    EmbeddingErrorResponse,
    DeepEvalTestOrchestrationResponse,
)
from src.utils.connection_id_fetcher import get_connection_id_fetcher
from src.loki_logger import LokiLogger

# Initialize Loki logger for centralized logging
logger = LokiLogger(service_name="llm-orchestration-api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Orchestration Service API")

    # nemoguardrails.actions.action_dispatcher logs every action it registers
    logging.getLogger("nemoguardrails.actions.action_dispatcher").setLevel(
        logging.WARNING
    )
    logging.getLogger("langfuse").setLevel(logging.ERROR)

    try:
        app.state.orchestration_service = LLMOrchestrationService()
        logger.info("LLM Orchestration Service initialized successfully")

        # Pre-warm shared BM25 index so the first query is never penalised by
        # the cold-start cost of scrolling all Qdrant chunks + building the index.
        logger.info("Pre-warming shared BM25 index...")
        await app.state.orchestration_service._prewarm_shared_bm25()
        logger.info("BM25 pre-warming complete")

        # Initialize rate limiter if enabled
        if StreamConfig.RATE_LIMIT_ENABLED:
            app.state.rate_limiter = RateLimiter(
                requests_per_minute=StreamConfig.RATE_LIMIT_REQUESTS_PER_MINUTE,
                tokens_per_minute=StreamConfig.RATE_LIMIT_TOKENS_PER_MINUTE,
            )
            logger.info("Rate limiter initialized successfully")
        else:
            app.state.rate_limiter = None
            logger.info("Rate limiting disabled")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Orchestration Service: {e}")
        raise

    # Initialize Redis session store (non-fatal: service continues without it)
    try:
        await init_redis_client()
        app.state.session_store = APIToolSessionStore()

        # Wire an incremental summarizer if the LLM manager singleton is available.
        summarizer = None
        try:
            summarizer = create_incremental_summarizer(LLMManager())
            logger.info("Incremental conversation summarizer initialized")
        except Exception as e:
            logger.warning(
                f"Could not create incremental summarizer, continuing without it: {e}"
            )

        app.state.conversation_history_store = ConversationHistoryStore(
            summarizer=summarizer
        )
        logger.info("Redis session store initialized successfully")
    except Exception as e:
        logger.warning(f"Redis session store unavailable, continuing without it: {e}")
        app.state.session_store = None
        app.state.conversation_history_store = None

    # Expose session_store and conversation_history_store on the orchestration
    # service so downstream components can reach them via self.orchestration_service.
    if (
        hasattr(app.state, "orchestration_service")
        and app.state.orchestration_service is not None
    ):
        app.state.orchestration_service.session_store = app.state.session_store
        app.state.orchestration_service.conversation_history_store = (
            app.state.conversation_history_store
        )

    yield

    # Shutdown
    logger.info("Shutting down LLM Orchestration Service API")

    # Await any in-flight incremental summary tasks to avoid lost work.
    store = getattr(app.state, "conversation_history_store", None)
    if store is not None and store._pending_tasks:
        logger.info(
            f"Waiting for {len(store._pending_tasks)} pending summary task(s) to complete..."
        )
        await asyncio.gather(*store._pending_tasks, return_exceptions=True)

    if (
        hasattr(app.state, "orchestration_service")
        and app.state.orchestration_service is not None
    ):
        await app.state.orchestration_service.aclose()
        app.state.orchestration_service = None

    try:
        await close_redis_client()
    except Exception as e:
        logger.warning(f"Error closing Redis client during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="LLM Orchestration Service API",
    description="API for orchestrating LLM requests with configuration management",
    version="1.0.0",
    lifespan=lifespan,
)


# Custom exception handlers for user-friendly error messages
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> StreamingResponse | JSONResponse:
    """
    Handle Pydantic validation errors with user-friendly messages.

    For streaming endpoints: Returns SSE format
    For non-streaming endpoints: Returns JSON format
    """
    import json as json_module
    from datetime import datetime

    error_id = generate_error_id()

    # Extract the first error for user-friendly message
    from typing import Dict, Any

    first_error: Dict[str, Any] = exc.errors()[0] if exc.errors() else {}
    error_msg = str(first_error.get("msg", ""))
    field_location: Any = first_error.get("loc", [])

    # Log full technical details for debugging (internal only)
    logger.error(
        f"[{error_id}] Request validation failed at {field_location}: {error_msg} | "
        f"Full errors: {exc.errors()}"
    )

    # Map technical errors to user-friendly messages
    user_message = VALIDATION_GENERIC_ERROR

    if "message" in field_location:
        if "at least 3 characters" in error_msg.lower():
            user_message = VALIDATION_MESSAGE_TOO_SHORT
        elif "maximum length" in error_msg.lower() or "exceeds" in error_msg.lower():
            user_message = VALIDATION_MESSAGE_TOO_LONG
        elif "sanitization" in error_msg.lower():
            user_message = VALIDATION_MESSAGE_INVALID_FORMAT
        else:
            user_message = VALIDATION_MESSAGE_GENERIC

    elif "conversationhistory" in "".join(str(loc).lower() for loc in field_location):
        user_message = VALIDATION_CONVERSATION_HISTORY_ERROR

    elif "payload" in error_msg.lower() or "size" in error_msg.lower():
        user_message = VALIDATION_REQUEST_TOO_LARGE

    elif any(
        field in field_location
        for field in ["chatId", "authorId", "url", "environment"]
    ):
        user_message = VALIDATION_REQUIRED_FIELDS_MISSING

    # Check if this is a streaming endpoint request
    if request.url.path == "/orchestrate/stream":
        # Extract chatId from request body if available
        chat_id = "unknown"
        try:
            body = await request.body()
            if body:
                body_json = json_module.loads(body)
                chat_id = body_json.get("chatId", "unknown")
        except Exception:
            # Silently fall back to "unknown" if body parsing fails
            # This is a validation error handler, so body is already malformed
            pass

        # Return SSE format for streaming endpoint
        async def validation_error_stream() -> AsyncGenerator[str, None]:
            error_payload: Dict[str, Any] = {
                "chatId": chat_id,
                "payload": {"content": user_message},
                "timestamp": str(int(datetime.now().timestamp() * 1000)),
                "sentTo": [],
            }
            yield f"data: {json_module.dumps(error_payload)}\n\n"

        return StreamingResponse(
            validation_error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Return JSON format for non-streaming endpoints
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": user_message,
            "error_id": error_id,
            "type": "validation_error",
        },
    )


@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic ValidationError with user-friendly messages."""
    error_id = generate_error_id()

    # Log technical details internally
    logger.error(f"[{error_id}] Pydantic validation error: {exc.errors()} | {str(exc)}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "I apologize, but I couldn't process your request due to invalid data format. Please check your input and try again.",
            "error_id": error_id,
            "type": "validation_error",
        },
    )


@app.post("/cache/clear")
async def clear_connection_cache() -> dict[str, str]:
    """Clear cached connection IDs and vault UUIDs."""
    try:
        fetcher = get_connection_id_fetcher()
        fetcher.clear_cache()
        logger.info("Connection cache cleared via /cache/clear endpoint")
        return {"status": "ok", "message": "Connection cache cleared"}
    except Exception as e:
        logger.error(f"Failed to clear connection cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear connection cache",
        ) from e


@app.get("/health")
async def health_check(request: Request) -> dict[str, str]:
    """Health check endpoint."""
    service_status = (
        "initialized"
        if hasattr(request.app.state, "orchestration_service")
        and request.app.state.orchestration_service is not None
        else "not_initialized"
    )
    redis_status = await check_redis_health()
    return {
        "status": "healthy",
        "service": "llm-orchestration-service",
        "orchestration_service": service_status,
        "redis_session_store": redis_status,
    }


@app.post(
    "/orchestrate",
    response_model=OrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Process LLM orchestration request",
    description="Processes a user message through the LLM orchestration pipeline",
)
async def orchestrate_llm_request(
    http_request: Request,
    request: OrchestrationRequest,
) -> OrchestrationResponse:
    """
    Process LLM orchestration request.

    Args:
        http_request: FastAPI Request object for accessing app state
        request: OrchestrationRequest containing user message and context

    Returns:
        OrchestrationResponse: Response with LLM output and status flags

    Raises:
        HTTPException: For processing errors
    """
    try:
        logger.info(f"Received orchestration request for chatId: {request.chatId}")

        # Get the orchestration service from app state
        if not hasattr(http_request.app.state, "orchestration_service"):
            logger.error("Orchestration service not found in app state")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        orchestration_service = http_request.app.state.orchestration_service
        if orchestration_service is None:
            logger.error("Orchestration service is None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        # Process the request
        response = await orchestration_service.process_orchestration_request(request)

        buttons_present = bool(response.buttons)
        buttons_count = len(response.buttons) if response.buttons else 0
        logger.info(
            f"[orchestrate] buttons in response for chatId {request.chatId}: "
            f"present={buttons_present}, count={buttons_count}"
        )
        logger.info(f"Successfully processed request for chatId: {request.chatId}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(
            logger, error_id, "orchestrate_endpoint", request.chatId, e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred",
        ) from e


@app.post(
    "/orchestrate/test",
    response_model=TestOrchestrationResponse,
    status_code=status.HTTP_200_OK,
    summary="Process test LLM orchestration request",
    description="Processes a simplified test message through the LLM orchestration pipeline",
)
async def test_orchestrate_llm_request(
    http_request: Request,
    request: TestOrchestrationRequest,
) -> TestOrchestrationResponse:
    """
    Process test LLM orchestration request with simplified input.

    Args:
        http_request: FastAPI Request object for accessing app state
        request: TestOrchestrationRequest containing only message, environment, and connection_id

    Returns:
        TestOrchestrationResponse: Response with LLM output and status flags (without chatId)

    Raises:
        HTTPException: For processing errors
    """
    try:
        logger.info(
            f"Received test orchestration request for environment: {request.environment}"
        )

        # Get the orchestration service from app state
        if not hasattr(http_request.app.state, "orchestration_service"):
            logger.error("Orchestration service not found in app state")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        orchestration_service = http_request.app.state.orchestration_service
        if orchestration_service is None:
            logger.error("Orchestration service is None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        # Map TestOrchestrationRequest to OrchestrationRequest with defaults
        full_request = OrchestrationRequest(
            chatId="test-session",
            message=request.message,
            authorId="test-user",
            conversationHistory=[],
            url="test-context",
            environment=request.environment,
            connection_id=str(request.connectionId)
            if request.connectionId is not None
            else None,
        )

        # test-LLM is single-turn only (no conversationHistory, no multi-turn loops).
        # Clear any stale API tool session so each request starts fresh and never
        # accidentally resumes a parameter-collection loop from a previous test query.
        session_store = getattr(http_request.app.state, "session_store", None)
        if session_store is not None:
            await session_store.delete("test-session")

        logger.info(f"This is full request constructed for testing: {full_request}")

        # Process the request using the same logic
        response = await orchestration_service.process_orchestration_request(
            full_request
        )

        # If response is already TestOrchestrationResponse (when environment is testing), return it directly
        if isinstance(response, TestOrchestrationResponse):
            buttons_count = len(response.buttons) if response.buttons else 0
            logger.info(
                f"[test_orchestrate] buttons present in response: {buttons_count}"
            )
            logger.info(
                f"Successfully processed test request for environment: {request.environment}"
            )
            return response

        # Convert to TestOrchestrationResponse (exclude chatId) for other cases
        test_response = TestOrchestrationResponse(
            llmServiceActive=response.llmServiceActive,
            questionOutOfLLMScope=response.questionOutOfLLMScope,
            inputGuardFailed=response.inputGuardFailed,
            content=response.content,
            buttons=response.buttons,
            chunks=None,  # OrchestrationResponse doesn't have chunks
        )
        logger.info(
            f"Successfully processed test request for environment: {request.environment}"
        )
        return test_response

    except HTTPException:
        raise
    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(
            logger, error_id, "test_orchestrate_endpoint", "test-session", e
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred",
        ) from e


@app.post(
    "/orchestrate/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream LLM orchestration response with validation-first guardrails",
    description="Streams LLM response with NeMo Guardrails validation-first approach",
)
async def stream_orchestrated_response(
    http_request: Request,
    request: OrchestrationRequest,
) -> StreamingResponse:
    """
    Stream LLM orchestration response with validation-first guardrails.

    Flow:
    1. Validate input with guardrails (blocking)
    2. Refine prompt (blocking)
    3. Retrieve context chunks (blocking)
    4. Check if question is in scope (blocking)
    5. Stream through NeMo Guardrails (validation-first)
       - Tokens buffered (chunk_size=200)
       - Each buffer validated before streaming
       - Only validated tokens reach client

    Request Body:
        Same as /orchestrate endpoint - OrchestrationRequest

    Response:
        Server-Sent Events (SSE) stream with format:
        data: {"chatId": "...", "payload": {"content": "..."}, "timestamp": "...", "sentTo": []}

    Content Types:
        - Regular token: "Token1", "Token2", "Token3", ...
        - Stream complete: "END"
        - Input blocked: Fixed message from constants
        - Out of scope: Fixed message from constants
        - Guardrail failed: Fixed message from constants
        - Validation error: User-friendly validation message
        - Technical error: Fixed message from constants

    Notes:
        - Available for configured environments (see STREAMING_ALLOWED_ENVS)
        - All responses use SSE format for consistency
        - Streaming uses validation-first approach (stream_first=False)
        - All tokens are validated before being sent to client
    """

    import json as json_module
    from datetime import datetime

    def create_sse_error_stream(chat_id: str, error_message: str) -> str:
        """Create an SSE error response, terminated by the END marker.

        The END frame is what the notification server translates into the
        browser's ``stream_end``. Without it an error frame is indistinguishable
        from ordinary content, so the client keeps waiting on a stream that has
        already finished - which is how an upstream timeout turned into a chat
        that hung indefinitely. Every caller is a terminal error path, so
        closing the stream here is always correct.
        """
        from typing import Dict, Any

        def frame(content: str) -> str:
            payload: Dict[str, Any] = {
                "chatId": chat_id,
                "payload": {"content": content},
                "timestamp": str(int(datetime.now().timestamp() * 1000)),
                "sentTo": [],
            }
            return f"data: {json_module.dumps(payload)}\n\n"

        return frame(error_message) + frame("END")

    try:
        logger.info(
            f"Streaming request received - "
            f"chatId: {request.chatId}, "
            f"environment: {request.environment}, "
            f"message: {request.message[:100]}..."
            f"connection_id: {request.connection_id}"
        )

        # Streaming is only for allowed environments
        if request.environment not in STREAMING_ALLOWED_ENVS:
            error_msg = f"Streaming is only available for production and testing environments. Current environment: {request.environment}. Please use /orchestrate endpoint for non-streaming environments."
            logger.warning(error_msg)

            async def env_error_stream() -> AsyncGenerator[str, None]:
                yield create_sse_error_stream(request.chatId, error_msg)

            return StreamingResponse(
                env_error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Get the orchestration service from app state
        if not hasattr(http_request.app.state, "orchestration_service"):
            error_msg = "I apologize, but the service is not available at the moment. Please try again later."
            logger.error("Orchestration service not found in app state")

            async def service_error_stream() -> AsyncGenerator[str, None]:
                yield create_sse_error_stream(request.chatId, error_msg)

            return StreamingResponse(
                service_error_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        orchestration_service = http_request.app.state.orchestration_service
        if orchestration_service is None:
            error_msg = "I apologize, but the service is not available at the moment. Please try again later."
            logger.error("Orchestration service is None")

            async def service_none_stream() -> AsyncGenerator[str, None]:
                yield create_sse_error_stream(request.chatId, error_msg)

            return StreamingResponse(
                service_none_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Check rate limits if enabled
        if StreamConfig.RATE_LIMIT_ENABLED and hasattr(
            http_request.app.state, "rate_limiter"
        ):
            rate_limiter = http_request.app.state.rate_limiter

            # Estimate tokens for this request (message + history)
            estimated_tokens = len(request.message) // 4  # 4 chars = 1 token
            for item in request.conversationHistory:
                estimated_tokens += len(item.message) // 4

            # Check rate limit
            rate_limit_result = rate_limiter.check_rate_limit(
                author_id=request.authorId,
                estimated_tokens=estimated_tokens,
            )

            if not rate_limit_result.allowed:
                # Determine appropriate error message
                if rate_limit_result.limit_type == "requests":
                    error_msg = RATE_LIMIT_REQUESTS_EXCEEDED_MESSAGE
                else:
                    error_msg = RATE_LIMIT_TOKENS_EXCEEDED_MESSAGE

                logger.warning(
                    f"Rate limit exceeded for {request.authorId} - "
                    f"type: {rate_limit_result.limit_type}, "
                    f"usage: {rate_limit_result.current_usage}/{rate_limit_result.limit}, "
                    f"retry_after: {rate_limit_result.retry_after}s"
                )

                # Return SSE format with rate limit error
                async def rate_limit_error_stream() -> AsyncGenerator[str, None]:
                    yield create_sse_error_stream(request.chatId, error_msg)

                return StreamingResponse(
                    rate_limit_error_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "Retry-After": str(rate_limit_result.retry_after),
                    },
                    status_code=429,
                )

        # Wrap streaming response with timeout
        async def timeout_wrapped_stream() -> AsyncGenerator[str, None]:
            """Generator wrapper with timeout enforcement."""
            with safe_observation_context(
                as_type="generation",
                name="streaming_generation",
                input={"message": request.message[:500], "chat_id": request.chatId},
            ):
                try:
                    async with stream_timeout(StreamConfig.MAX_STREAM_DURATION_SECONDS):
                        # Heartbeat frames keep proxies from closing a slow stream,
                        # and the idle budget fails fast on one that has truly
                        # stalled rather than waiting out the total-duration cap.
                        async for chunk in with_heartbeat(
                            orchestration_service.stream_orchestration_response(
                                request
                            ),
                            heartbeat_interval=StreamConfig.HEARTBEAT_INTERVAL_SECONDS,
                            idle_timeout=StreamConfig.IDLE_TIMEOUT_SECONDS,
                        ):
                            yield chunk
                except StreamTimeoutError as timeout_exc:
                    # StreamTimeoutError already has error_id
                    log_error_with_context(
                        logger,
                        timeout_exc.error_id,
                        "streaming_timeout",
                        request.chatId,
                        timeout_exc,
                    )
                    # Send timeout message to client
                    yield create_sse_error_stream(
                        request.chatId, STREAM_TIMEOUT_MESSAGE
                    )
                except Exception as stream_error:
                    error_id = generate_error_id()
                    log_error_with_context(
                        logger,
                        error_id,
                        "streaming_error",
                        request.chatId,
                        stream_error,
                    )
                    # Send generic error message to client
                    yield create_sse_error_stream(
                        request.chatId,
                        "I apologize, but I encountered an issue while generating your response. Please try again.",
                    )

        # Stream the response
        return StreamingResponse(
            timeout_wrapped_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        # Catch any unexpected errors and return SSE format
        error_id = generate_error_id()
        logger.error(f"[{error_id}] Unexpected error in streaming endpoint: {str(e)}")

        async def unexpected_error_stream() -> AsyncGenerator[str, None]:
            yield create_sse_error_stream(
                request.chatId if hasattr(request, "chatId") else "unknown",
                "I apologize, but I encountered an unexpected issue. Please try again.",
            )

        return StreamingResponse(
            unexpected_error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


@app.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    responses={500: {"model": EmbeddingErrorResponse}},
)
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Create embeddings using DSPy with vault-driven model resolution.

    Model selection is automatic based on environment and connection_id:
    - Production: Uses first available embedding model from vault
    - Development/Test: Uses model associated with connection_id

    Supports Azure OpenAI, AWS Bedrock, and OpenAI embedding models.
    Includes automatic retry with exponential backoff.
    """
    try:
        logger.info(
            f"Creating embeddings for {len(request.texts)} texts in {request.environment} environment"
        )

        result: Dict[str, Any] = (
            app.state.orchestration_service.create_embeddings_for_indexer(
                texts=request.texts,
                environment=request.environment,
                connection_id=request.connection_id,
                batch_size=request.batch_size or 50,
            )
        )

        return EmbeddingResponse(**result)

    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(
            logger,
            error_id,
            "embeddings_endpoint",
            None,
            e,
            {"num_texts": len(request.texts), "environment": request.environment},
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Embedding creation failed",
                "retry_after": 30,
            },
        ) from e


@app.post("/generate-context", response_model=ContextGenerationResponse)
async def generate_context_with_caching(
    request: ContextGenerationRequest,
) -> ContextGenerationResponse:
    """
    Generate contextual descriptions using Anthropic methodology.

    Uses exact Anthropic prompt templates and supports structure for
    future prompt caching implementation for cost optimization.
    """
    try:
        result = app.state.orchestration_service.generate_context_for_chunks(request)

        return ContextGenerationResponse(**result)

    except ConfigurationError as e:
        # No usable LLM connection for this environment. This is an operator
        # action, not a transient fault - 503 with the reason so callers (e.g.
        # the vector indexer) can stop retrying and surface something useful.
        error_id = generate_error_id()
        log_error_with_context(logger, error_id, "context_generation_endpoint", None, e)
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(logger, error_id, "context_generation_endpoint", None, e)
        raise HTTPException(status_code=500, detail="Context generation failed") from e


@app.get("/embedding-models")
async def get_available_embedding_models(
    environment: str = "production",
) -> Dict[str, Any]:
    """Get available embedding models from vault configuration.

    Args:
        environment: Environment to get models for (production, development, test)

    Returns:
        Dictionary with available models and default model information
    """
    try:
        # Get available embedding models using vault-driven resolution
        result: Dict[str, Any] = (
            app.state.orchestration_service.get_available_embedding_models_for_indexer(
                environment=environment
            )
        )
        return result

    except Exception as e:
        error_id = generate_error_id()
        log_error_with_context(
            logger,
            error_id,
            "embedding_models_endpoint",
            None,
            e,
            {"environment": environment},
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve embedding models"
        ) from e


@app.post("/orchestrate-eval")
async def orchestrate_llm_request_eval(
    http_request: Request,
    request: OrchestrationRequest,
) -> DeepEvalTestOrchestrationResponse:
    """
    Process LLM orchestration request with additional testing data.

    This endpoint is only available when EVAL_MODE=true and returns
    retrieval context and refined questions for DeepEval metrics evaluation.

    Args:
        http_request: FastAPI Request object for accessing app state
        request: OrchestrationRequest containing user message and context

    Returns:
        OrchestrationResponse: Response with LLM output, status flags, and test data

    Raises:
        HTTPException: For processing errors or if not in testing mode
    """
    # Check if eval mode is enabled
    eval_mode = os.getenv("EVAL_MODE", "false").lower() == "true"
    if not eval_mode:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Eval endpoint not available in production mode",
        )

    try:
        logger.info(f"Received EVAL orchestration request for chatId: {request.chatId}")

        if not hasattr(http_request.app.state, "orchestration_service"):
            logger.error("Orchestration service not found in app state")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        orchestration_service = http_request.app.state.orchestration_service
        if orchestration_service is None:
            logger.error("Orchestration service is None")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Service not initialized",
            )

        # Process the request (will include test data due to EVAL_MODE env var)
        response = await orchestration_service.process_orchestration_request(request)

        # Convert to test response with additional fields
        # Response may be OrchestrationResponse or TestOrchestrationResponse
        chat_id = getattr(response, "chatId", request.chatId)
        retrieval_ctx = getattr(response, "retrieval_context", None)

        test_response = DeepEvalTestOrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=response.llmServiceActive,
            questionOutOfLLMScope=response.questionOutOfLLMScope,
            inputGuardFailed=response.inputGuardFailed,
            content=response.content,
            retrieval_context=retrieval_ctx,
            expected_output=None,  # Will be populated by test framework
        )

        logger.info(f"Successfully processed TEST request for chatId: {request.chatId}")
        return test_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing TEST request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred",
        ) from e


@app.post("/prompt-config/refresh")
def refresh_prompt_config(http_request: Request) -> Dict[str, Any]:
    """
    Force immediate refresh of prompt configuration cache.

    This endpoint is called by Ruuter after admin updates the prompt configuration
    in the database, ensuring the changes are reflected immediately without waiting
    for the cache TTL to expire.

    Returns:
        Dictionary with refresh status and message

    Raises:
        HTTPException (503): If prompt configuration loader is not initialized
        HTTPException (404): If no prompt configuration found in database
        HTTPException (500): If refresh operation fails
    """
    orchestration_service = http_request.app.state.orchestration_service

    # Check if loader is initialized
    if not orchestration_service or not hasattr(
        orchestration_service, "prompt_config_loader"
    ):
        error_id = generate_error_id()
        logger.error(f"[{error_id}] Prompt configuration loader not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Prompt configuration loader not initialized",
                "error_id": error_id,
            },
        )

    try:
        # Use new method that returns detailed status
        refresh_result = (
            orchestration_service.prompt_config_loader.force_refresh_with_status()
        )
        refresh_status = refresh_result.get("status")

        if refresh_status == RefreshStatus.SUCCESS:
            # Success - configuration loaded
            logger.info("Prompt configuration refreshed successfully")
            return {
                "refreshed": True,
                "message": refresh_result.get("message"),
                "prompt_length": refresh_result.get("length"),
            }

        elif refresh_status == RefreshStatus.NOT_FOUND:
            # Configuration absent in database
            error_id = generate_error_id()
            logger.warning(f"[{error_id}] Prompt configuration not found in database")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": refresh_result.get("message"),
                    "error_id": error_id,
                },
            )

        elif refresh_status == RefreshStatus.FETCH_FAILED:
            # Upstream service failure (network/HTTP/timeout errors)
            error_id = generate_error_id()
            had_stale = refresh_result.get("had_stale_cache", False)

            if had_stale:
                logger.warning(
                    f"[{error_id}] Upstream service unavailable, stale cache exists"
                )
                # Temporarily unavailable but we have fallback
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail={
                        "error": "Upstream service temporarily unavailable",
                        "error_id": error_id,
                        "message": "Stale configuration available as fallback",
                    },
                )
            else:
                logger.warning(
                    f"[{error_id}] Upstream service unavailable, no cache exists"
                )
                # Service gateway error or timeout
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "error": refresh_result.get("message"),
                        "error_id": error_id,
                        "details": refresh_result.get("error"),
                    },
                )

        else:
            # Unexpected status - should never happen but handle defensively
            error_id = generate_error_id()
            logger.error(f"[{error_id}] Unexpected refresh status: {refresh_status}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Unexpected error during refresh",
                    "error_id": error_id,
                },
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Unexpected errors during refresh
        error_id = generate_error_id()
        logger.error(f"[{error_id}] Failed to refresh prompt configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Failed to refresh prompt configuration",
                "error_id": error_id,
            },
        ) from e


if __name__ == "__main__":
    logger.info("Starting LLM Orchestration Service API server on port 8100")
    uvicorn.run(
        "llm_orchestration_service_api:app",
        host="0.0.0.0",
        port=8100,
        log_level="info",
    )

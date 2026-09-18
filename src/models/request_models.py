"""Pydantic models for API requests and responses."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
import json

from src.utils.input_sanitizer import InputSanitizer
from src.llm_orchestrator_config.stream_config import StreamConfig
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="request-models")


class ConversationItem(BaseModel):
    """Model for conversation history item."""

    authorRole: Literal["user", "bot"] = Field(
        ..., description="Role of the message author"
    )
    message: str = Field(..., description="Content of the message")
    timestamp: str = Field(..., description="Timestamp in ISO format")

    @field_validator("message")
    @classmethod
    def validate_and_sanitize_message(cls, v: str) -> str:
        """Sanitize and validate conversation message."""

        # Sanitize HTML and normalize whitespace
        v = InputSanitizer.sanitize_message(v)

        # Check length
        if len(v) > StreamConfig.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Conversation message exceeds maximum length of {StreamConfig.MAX_MESSAGE_LENGTH} characters"
            )

        return v


class PromptRefinerOutput(BaseModel):
    """Model for prompt refiner output."""

    original_question: str = Field(..., description="The original user question")
    refined_questions: List[str] = Field(
        ..., description="List of refined question variants"
    )


class OrchestrationRequest(BaseModel):
    """Model for LLM orchestration request."""

    chatId: str = Field(..., description="Unique identifier for the chat session")
    message: str = Field(..., description="User's message/query")
    authorId: str = Field(..., description="Unique identifier for the user")
    conversationHistory: List[ConversationItem] = Field(
        ..., description="Previous conversation history"
    )
    url: str = Field(..., description="Source URL context")
    environment: Literal["production", "testing"] = Field(
        "production", description="Environment context (defaults to production)"
    )
    connection_id: Optional[str] = Field(
        None, description="Vault UUID for the connection (required for testing)"
    )

    @model_validator(mode="after")
    def validate_connection_id_for_testing(self) -> "OrchestrationRequest":
        """Ensure connection_id is provided when environment is testing."""
        if self.environment == "testing" and not self.connection_id:
            raise ValueError("connection_id is required when environment is 'testing'")
        return self

    @field_validator("message")
    @classmethod
    def validate_and_sanitize_message(cls, v: str) -> str:
        """Sanitize and validate user message.

        Note: This validator only handles security/format concerns:
        - XSS/HTML sanitization
        - Maximum length enforcement

        Query quality validation (empty messages, special chars, etc.) is handled
        by the business logic layer (query_validator) with localized error messages.

        Content safety checks (prompt injection, PII, harmful content)
        are handled by NeMo Guardrails after this validation layer.
        """
        # Sanitize HTML/XSS and normalize whitespace
        v = InputSanitizer.sanitize_message(v)

        # Check length after sanitization (resource protection)
        if len(v) > StreamConfig.MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Message exceeds maximum length of {StreamConfig.MAX_MESSAGE_LENGTH} characters"
            )

        return v

    @field_validator("conversationHistory")
    @classmethod
    def validate_conversation_history(
        cls, v: List[ConversationItem]
    ) -> List[ConversationItem]:
        """Validate conversation history limits."""

        # Limit number of conversation history items
        max_history_items = 100

        if len(v) > max_history_items:
            logger.warning(
                f"Conversation history truncated: {len(v)} -> {max_history_items} items"
            )
            # Truncate to most recent items
            v = v[-max_history_items:]

        return v

    @model_validator(mode="after")
    def validate_payload_size(self) -> "OrchestrationRequest":
        """Validate total payload size does not exceed limit."""

        try:
            payload_size = len(json.dumps(self.model_dump()).encode("utf-8"))
            if payload_size > StreamConfig.MAX_PAYLOAD_SIZE_BYTES:
                raise ValueError(
                    f"Request payload exceeds maximum size of {StreamConfig.MAX_PAYLOAD_SIZE_BYTES} bytes"
                )
        except (TypeError, ValueError, OverflowError) as e:
            # Catch specific serialization errors and log them
            # ValueError: raised when size limit exceeded (re-raise this)
            # TypeError: circular references or non-serializable objects
            # OverflowError: data too large to serialize
            if "exceeds maximum size" in str(e):
                raise  # Re-raise size limit violations
            logger.warning(
                f"Payload size validation skipped due to serialization error: {type(e).__name__}: {e}"
            )
        return self


class DocumentReference(BaseModel):
    """Model for document reference with URL."""

    document_url: str = Field(..., description="Source document URL")
    chunk_rank: int = Field(..., description="Rank of chunk in retrieval (1-based)")
    relevance_score: float = Field(..., description="Relevance score (0-1)")


class ChoiceButton(BaseModel):
    """A single MCQ choice button returned in an orchestration response."""

    title: str = Field(..., description="Button label shown to the user")
    payload: str = Field(
        ...,
        description="Routing string sent when the button is clicked (e.g. '#service, /POST/...')",
    )


class OrchestrationResponse(BaseModel):
    """Model for LLM orchestration response."""

    chatId: str = Field(..., description="Chat session identifier from request")
    llmServiceActive: bool = Field(..., description="Whether LLM service is active")
    questionOutOfLLMScope: bool = Field(
        ..., description="Whether question is out of LLM scope"
    )
    inputGuardFailed: bool = Field(
        ..., description="Whether input guard validation failed"
    )
    content: str = Field(..., description="Response content with citations")
    buttons: Optional[List[ChoiceButton]] = Field(
        default=None,
        description="Optional list of choice buttons for MCQ step responses",
    )
    retrieval_context: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Retrieved chunks with metadata, populated only in EVAL_MODE",
    )


# New models for embedding and context generation


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation.

    Model name is resolved from vault based on environment and connection_id.
    No explicit model_name parameter needed - uses vault-driven model selection.
    """

    texts: List[str] = Field(..., description="List of texts to embed", max_length=1000)
    environment: Literal["production", "development", "testing"] = Field(
        ..., description="Environment for model resolution"
    )
    batch_size: Optional[int] = Field(
        50,  # Using small batch size as requested
        description="Batch size for processing",
        ge=1,
        le=100,
    )
    connection_id: Optional[str] = Field(
        None,
        description="Connection ID for dev/test environments (required for non-production)",
    )


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""

    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model_used: str = Field(..., description="Actual model used for embeddings")
    processing_info: Dict[str, Any] = Field(..., description="Processing metadata")
    total_tokens: Optional[int] = Field(None, description="Total tokens processed")


class ContextGenerationRequest(BaseModel):
    """Request model for context generation using Anthropic methodology."""

    document_prompt: str = Field(
        ..., description="Document content for caching", max_length=100000
    )
    chunk_prompt: str = Field(..., description="Chunk-specific prompt", max_length=5000)
    environment: Literal["production", "development", "testing"] = Field(
        ..., description="Environment for model resolution"
    )
    use_cache: bool = Field(default=True, description="Enable prompt caching")
    connection_id: Optional[str] = Field(
        None, description="Connection ID for dev/test environments"
    )
    max_tokens: int = Field(
        default=1000, description="Maximum tokens for response", ge=1, le=8192
    )
    temperature: float = Field(
        default=0.1, description="Temperature for response generation", ge=0.0, le=2.0
    )
    context_type: Literal["chunk", "api_tool"] = Field(
        default="chunk",
        description="Controls which prompt template is used. 'chunk' uses the Anthropic "
        "RAG template (short succinct context). 'api_tool' passes the prompt through "
        "unmodified so the caller's own instructions are respected.",
    )


class ContextGenerationResponse(BaseModel):
    """Response model for context generation."""

    context: str = Field(..., description="Generated contextual description")
    usage: Dict[str, int] = Field(..., description="Token usage breakdown")
    cache_performance: Dict[str, Any] = Field(
        ..., description="Caching performance metrics"
    )
    model_used: str = Field(..., description="Model used for generation")


class EmbeddingErrorResponse(BaseModel):
    """Error response for embedding failures."""

    error: str = Field(..., description="Error message")
    failed_texts: List[str] = Field(..., description="Texts that failed to embed")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")


# Test endpoint models


class TestOrchestrationRequest(BaseModel):
    """Model for simplified test orchestration request."""

    message: str = Field(..., description="User's message/query")
    environment: Literal["production", "testing"] = Field(
        ..., description="Environment context"
    )
    connectionId: Optional[int] = Field(
        None, description="Optional connection identifier"
    )


class ChunkInfo(BaseModel):
    """Model for chunk information in test response."""

    rank: int = Field(..., description="Rank of the retrieved chunk")
    chunkRetrieved: str = Field(..., description="Content of the retrieved chunk")


class TestOrchestrationResponse(BaseModel):
    """Model for test orchestration response (without chatId)."""

    llmServiceActive: bool = Field(..., description="Whether LLM service is active")
    questionOutOfLLMScope: bool = Field(
        ..., description="Whether question is out of LLM scope"
    )
    inputGuardFailed: bool = Field(
        ..., description="Whether input guard validation failed"
    )
    content: str = Field(..., description="Response content with citations")
    buttons: Optional[List[ChoiceButton]] = Field(
        default=None,
        description="Optional list of choice buttons for MCQ step responses",
    )
    chunks: Optional[List[ChunkInfo]] = Field(
        default=None, description="Retrieved chunks with rank and content"
    )


class DeepEvalTestOrchestrationResponse(BaseModel):
    """Extended response model for testing with additional evaluation data."""

    chatId: str
    llmServiceActive: bool
    questionOutOfLLMScope: bool
    inputGuardFailed: bool
    content: str
    retrieval_context: Optional[List[Dict[str, Any]]] = None
    refined_questions: Optional[List[str]] = None
    expected_output: Optional[str] = None  # For DeepEval

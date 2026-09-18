"""Data models for vector indexer."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Information about a document to be processed."""

    document_hash: str = Field(..., description="Document hash identifier")
    cleaned_txt_path: str = Field(..., description="Path to cleaned.txt file")
    source_meta_path: str = Field(..., description="Path to cleaned.meta.json file")
    dataset_collection: str = Field(..., description="Dataset collection name")


class ProcessingDocument(BaseModel):
    """Document loaded and ready for processing."""

    content: str = Field(..., description="Document content from cleaned.txt")
    metadata: Dict[str, Any] = Field(..., description="Metadata from cleaned.meta.json")
    document_hash: str = Field(..., description="Document hash identifier")

    @property
    def source_url(self) -> Optional[str]:
        """Get source URL from metadata."""
        return self.metadata.get("source_url")


class BaseChunk(BaseModel):
    """Base chunk before context generation."""

    content: str = Field(..., description="Original chunk content")
    tokens: int = Field(..., description="Estimated token count")
    start_index: int = Field(..., description="Start character index in document")
    end_index: int = Field(..., description="End character index in document")


class ContextualChunk(BaseModel):
    """Chunk with generated context and embeddings."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_hash: str = Field(..., description="Parent document hash")
    chunk_index: int = Field(..., description="Chunk index within document")
    total_chunks: int = Field(..., description="Total chunks in document")

    # Content
    original_content: str = Field(..., description="Original chunk content")
    context: str = Field(..., description="Generated contextual description")
    contextual_content: str = Field(..., description="Context + original content")

    # Embedding information
    embedding: Optional[List[float]] = Field(None, description="Embedding vector")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    vector_dimensions: Optional[int] = Field(None, description="Vector dimensions")

    # Metadata
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    processing_timestamp: datetime = Field(default_factory=datetime.now)
    tokens_count: int = Field(..., description="Token count of contextual content")

    @property
    def source_url(self) -> Optional[str]:
        """Get source URL from metadata."""
        return self.metadata.get("source_url")

    @property
    def dataset_collection(self) -> Optional[str]:
        """Extract dataset collection from chunk_id."""
        # chunk_id format: {document_hash}_chunk_{index}
        return self.metadata.get("dataset_collection")


class ProcessingStats(BaseModel):
    """Statistics for processing session."""

    total_documents: int = 0
    documents_processed: int = 0
    documents_failed: int = 0
    total_chunks_processed: int = 0
    total_chunks_failed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration(self) -> Optional[str]:
        """Calculate processing duration."""
        if self.start_time and self.end_time:
            return str(self.end_time - self.start_time)
        return None

    @property
    def success_rate(self) -> float:
        """Calculate document success rate."""
        if self.total_documents > 0:
            return self.documents_processed / self.total_documents
        return 0.0

    @property
    def chunk_success_rate(self) -> float:
        """Calculate chunk success rate (processed vs processed + failed)."""
        total_chunks = self.total_chunks_processed + self.total_chunks_failed
        if total_chunks > 0:
            return self.total_chunks_processed / total_chunks
        return 0.0


class ProcessingError(BaseModel):
    """Error information for failed processing."""

    timestamp: datetime = Field(default_factory=datetime.now)
    error_type: str = Field(..., description="Type of error")
    document_hash: Optional[str] = Field(
        None, description="Document hash if applicable"
    )
    chunk_index: Optional[int] = Field(None, description="Chunk index if applicable")
    error_message: str = Field(..., description="Error message")
    retry_count: int = Field(0, description="Number of retries attempted")
    action_taken: str = Field(..., description="Action taken after error")

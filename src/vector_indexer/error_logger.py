"""Enhanced error logging for vector indexer."""

import json
from pathlib import Path
from loki_logger import LokiLogger

from vector_indexer.config.config_loader import VectorIndexerConfig
from vector_indexer.models import ProcessingError, ProcessingStats

# Initialize Loki logger
logger = LokiLogger(service_name="error-logger")


class ErrorLogger:
    """Enhanced error logging with file-based failure tracking."""

    def __init__(self, config: VectorIndexerConfig) -> None:
        self.config = config
        self._ensure_log_directories()
        self._setup_logging()

    def _ensure_log_directories(self) -> None:
        """Create log directories if they don't exist."""
        for log_file in [
            self.config.failure_log_file,
            self.config.processing_log_file,
            self.config.stats_log_file,
        ]:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        """Setup logging - LokiLogger handles all logging (console + Loki service)."""
        # LokiLogger is already configured and handles console output + Loki integration
        # No additional configuration needed
        pass

    def log_document_failure(
        self, document_hash: str, error: str, retry_count: int = 0
    ) -> None:
        """Log document processing failure."""
        if not self.config.log_failures:
            return

        error_entry = ProcessingError(
            error_type="document_processing_failed",
            document_hash=document_hash,
            chunk_index=None,
            error_message=str(error),
            retry_count=retry_count,
            action_taken="skipped_document",
        )

        # Append to JSONL failure log
        try:
            with open(self.config.failure_log_file, "a", encoding="utf-8") as f:
                f.write(error_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write failure log: {e}")

        logger.error(f"Document {document_hash} failed: {error}")

    def log_chunk_failure(
        self, document_hash: str, chunk_index: int, error: str, retry_count: int
    ) -> None:
        """Log individual chunk processing failure."""
        if not self.config.log_failures:
            return

        error_entry = ProcessingError(
            error_type="chunk_processing_failed",
            document_hash=document_hash,
            chunk_index=chunk_index,
            error_message=str(error),
            retry_count=retry_count,
            action_taken="skipped_chunk",
        )

        try:
            with open(self.config.failure_log_file, "a", encoding="utf-8") as f:
                f.write(error_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write failure log: {e}")

        logger.warning(
            f"Chunk {chunk_index} in document {document_hash} failed: {error}"
        )

    def log_context_generation_failure(
        self, document_hash: str, chunk_index: int, error: str, retry_count: int
    ) -> None:
        """Log context generation failure."""
        if not self.config.log_failures:
            return

        error_entry = ProcessingError(
            error_type="context_generation_failed",
            document_hash=document_hash,
            chunk_index=chunk_index,
            error_message=str(error),
            retry_count=retry_count,
            action_taken="skipped_chunk_context",
        )

        try:
            with open(self.config.failure_log_file, "a", encoding="utf-8") as f:
                f.write(error_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write failure log: {e}")

        logger.warning(
            f"Context generation failed for chunk {chunk_index} in document {document_hash}: {error}"
        )

    def log_embedding_failure(
        self, document_hash: str, error: str, retry_count: int
    ) -> None:
        """Log embedding creation failure."""
        if not self.config.log_failures:
            return

        error_entry = ProcessingError(
            error_type="embedding_creation_failed",
            document_hash=document_hash,
            chunk_index=None,
            error_message=str(error),
            retry_count=retry_count,
            action_taken="skipped_document_embedding",
        )

        try:
            with open(self.config.failure_log_file, "a", encoding="utf-8") as f:
                f.write(error_entry.model_dump_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to write failure log: {e}")

        logger.error(f"Embedding creation failed for document {document_hash}: {error}")

    def log_processing_stats(self, stats: ProcessingStats) -> None:
        """Log final processing statistics."""
        try:
            stats_dict = stats.model_dump()
            # Convert datetime objects to ISO format strings
            if stats.start_time is not None:
                stats_dict["start_time"] = stats.start_time.isoformat()
            if stats.end_time is not None:
                stats_dict["end_time"] = stats.end_time.isoformat()
            stats_dict["duration"] = stats.duration
            stats_dict["success_rate"] = stats.success_rate
            stats_dict["chunk_success_rate"] = stats.chunk_success_rate

            with open(self.config.stats_log_file, "w", encoding="utf-8") as f:
                json.dump(stats_dict, f, indent=2)

            logger.info(
                f"Processing completed - Success rate: {stats.success_rate:.1%}, "
                f"Chunk success rate: {stats.chunk_success_rate:.1%}, "
                f"Duration: {stats.duration}, "
                f"Processed: {stats.documents_processed}/{stats.total_documents} documents, "
                f"Chunks: {stats.total_chunks_processed} ok / {stats.total_chunks_failed} failed"
            )
        except Exception as e:
            logger.error(f"Failed to write stats log: {e}")

    def log_progress(
        self, completed: int, total: int, current_document: str = ""
    ) -> None:
        """Log processing progress."""
        percentage = (completed / total * 100) if total > 0 else 0
        if current_document:
            logger.info(
                f"Progress: {completed}/{total} ({percentage:.1f}%) - Processing: {current_document}"
            )
        else:
            logger.info(f"Progress: {completed}/{total} ({percentage:.1f}%)")

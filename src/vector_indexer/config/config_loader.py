"""Configuration loader for vector indexer."""

import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from loki_logger import LokiLogger

from vector_indexer.constants import (
    DocumentConstants,
    ValidationConstants,
    ChunkingConstants,
    ProcessingConstants,
)

# Initialize Loki logger
logger = LokiLogger(service_name="config-loader")


class ChunkingConfig(BaseModel):
    """Configuration for document chunking operations"""

    min_chunk_size: int = Field(
        default=ChunkingConstants.MIN_CHUNK_SIZE_TOKENS,
        ge=10,
        description="Minimum chunk size in tokens",
    )
    max_chunk_size: int = Field(
        default=4000, ge=100, description="Maximum chunk size in tokens"
    )
    tokenizer_encoding: str = Field(
        default=ChunkingConstants.DEFAULT_TOKENIZER_ENCODING,
        description="Tokenizer encoding to use (e.g., cl100k_base)",
    )
    chars_per_token: float = Field(
        default=ChunkingConstants.CHARS_PER_TOKEN,
        gt=0.0,
        description="Estimated characters per token for pre-chunking",
    )
    templates: Dict[str, str] = Field(
        default_factory=lambda: {
            "chunk_id_pattern": "chunk_{document_hash}_{index:04d}",
            "context_separator": "\n\n--- Chunk {chunk_id} ---\n\n",
        },
        description="Templates for chunk formatting",
    )


class ProcessingConfig(BaseModel):
    """Configuration for document processing operations"""

    batch_delay_seconds: float = Field(
        default=ProcessingConstants.BATCH_DELAY_SECONDS,
        ge=0.0,
        description="Delay between batch processing operations",
    )
    context_delay_seconds: float = Field(
        default=ProcessingConstants.CONTEXT_DELAY_SECONDS,
        ge=0.0,
        description="Delay between context generation operations",
    )
    provider_detection_patterns: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "openai": [r"\bGPT\b", r"\bOpenAI\b", r"\btext-embedding\b", r"\bada\b"],
            "aws_bedrock": [r"\btitan\b", r"\bamazon\b", r"\bbedrock\b"],
            "azure_openai": [r"\bazure\b", r"\btext-embedding-3\b", r"\bada-002\b"],
        },
        description="Regex patterns for provider detection in content",
    )


class QdrantConfig(BaseModel):
    """Qdrant database configuration."""

    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "chunks"


class VectorIndexerConfig(BaseModel):
    """Configuration model for vector indexer."""

    # API Configuration
    api_base_url: str = "http://llm-orchestration-service:8100"
    qdrant_url: str = "http://qdrant:6333"
    api_timeout: int = 300

    # Processing Configuration
    environment: str = "production"
    connection_id: Optional[str] = None

    # Chunking Configuration
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Concurrency Configuration
    max_concurrent_documents: int = 3
    max_concurrent_chunks_per_doc: int = 5

    # Batch Configuration (Small batches)
    embedding_batch_size: int = 10
    context_batch_size: int = 5

    # Error Handling
    max_retries: int = 3
    retry_delay_base: int = 2
    continue_on_failure: bool = True
    log_failures: bool = True

    # Logging Configuration
    log_level: str = "INFO"
    failure_log_file: str = "logs/vector_indexer_failures.jsonl"
    processing_log_file: str = "logs/vector_indexer_processing.log"
    stats_log_file: str = "logs/vector_indexer_stats.json"

    # Dataset Configuration
    dataset_base_path: str = "datasets"
    target_file: str = "cleaned.txt"
    metadata_file: str = "cleaned.meta.json"

    # Enhanced Configuration Models
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)


class DocumentLoaderConfig(BaseModel):
    """Enhanced configuration model for document loader with validation."""

    # File discovery
    target_file: str = Field(
        default=DocumentConstants.DEFAULT_TARGET_FILE, min_length=1
    )
    metadata_file: str = Field(
        default=DocumentConstants.DEFAULT_METADATA_FILE, min_length=1
    )

    # Content validation
    min_content_length: int = Field(default=DocumentConstants.MIN_CONTENT_LENGTH, gt=0)
    max_content_length: int = Field(default=DocumentConstants.MAX_CONTENT_LENGTH, gt=0)
    encoding: str = Field(default=DocumentConstants.ENCODING)

    # Metadata validation
    required_metadata_fields: List[str] = Field(
        default=ValidationConstants.REQUIRED_METADATA_FIELDS
    )

    # File validation
    min_file_size_bytes: int = Field(
        default=ValidationConstants.MIN_FILE_SIZE_BYTES, gt=0
    )
    max_file_size_bytes: int = Field(
        default=ValidationConstants.MAX_FILE_SIZE_BYTES, gt=0
    )

    # Performance settings
    enable_content_caching: bool = Field(default=False)
    max_scan_depth: int = Field(default=DocumentConstants.MAX_SCAN_DEPTH, gt=0, le=10)

    @model_validator(mode="after")
    def validate_content_length_range(self) -> "DocumentLoaderConfig":
        """Ensure min_content_length < max_content_length."""
        if self.min_content_length >= self.max_content_length:
            raise ValueError(
                f"min_content_length ({self.min_content_length}) must be less than "
                f"max_content_length ({self.max_content_length})"
            )
        return self

    @model_validator(mode="after")
    def validate_file_size_range(self) -> "DocumentLoaderConfig":
        """Ensure min_file_size_bytes < max_file_size_bytes."""
        if self.min_file_size_bytes >= self.max_file_size_bytes:
            raise ValueError(
                f"min_file_size_bytes ({self.min_file_size_bytes}) must be less than "
                f"max_file_size_bytes ({self.max_file_size_bytes})"
            )
        return self

    @field_validator("required_metadata_fields")
    @classmethod
    def validate_metadata_fields(cls, v: List[str]) -> List[str]:
        """Ensure at least one metadata field is required."""
        if not v or len(v) == 0:
            raise ValueError("At least one metadata field must be required")
        return v


class ConfigLoader:
    """Load configuration from YAML file."""

    @staticmethod
    def load_config(
        config_path: str = "src/vector_indexer/config/vector_indexer_config.yaml",
    ) -> VectorIndexerConfig:
        """Load configuration from YAML file."""

        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return VectorIndexerConfig()

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
            return VectorIndexerConfig()

        # Extract vector_indexer section
        indexer_config = yaml_config.get("vector_indexer", {})

        # Flatten nested configuration
        flattened_config: Dict[str, Any] = {}

        # API config
        api_config = indexer_config.get("api", {})
        flattened_config["api_base_url"] = api_config.get("base_url")
        flattened_config["api_timeout"] = api_config.get("timeout", 300)

        # Processing config
        processing_config = indexer_config.get("processing", {})
        flattened_config["environment"] = processing_config.get(
            "environment", "production"
        )
        flattened_config["connection_id"] = processing_config.get("connection_id")

        # Chunking config
        chunking_config = indexer_config.get("chunking", {})
        flattened_config["chunk_size"] = chunking_config.get("chunk_size", 800)
        flattened_config["chunk_overlap"] = chunking_config.get("chunk_overlap", 100)

        # Concurrency config
        concurrency_config = indexer_config.get("concurrency", {})
        flattened_config["max_concurrent_documents"] = concurrency_config.get(
            "max_concurrent_documents", 3
        )
        flattened_config["max_concurrent_chunks_per_doc"] = concurrency_config.get(
            "max_concurrent_chunks_per_doc", 5
        )

        # Batching config
        batching_config = indexer_config.get("batching", {})
        flattened_config["embedding_batch_size"] = batching_config.get(
            "embedding_batch_size", 10
        )
        flattened_config["context_batch_size"] = batching_config.get(
            "context_batch_size", 5
        )

        # Error handling config
        error_config = indexer_config.get("error_handling", {})
        flattened_config["max_retries"] = error_config.get("max_retries", 3)
        flattened_config["retry_delay_base"] = error_config.get("retry_delay_base", 2)
        flattened_config["continue_on_failure"] = error_config.get(
            "continue_on_failure", True
        )
        flattened_config["log_failures"] = error_config.get("log_failures", True)

        # Logging config
        logging_config = indexer_config.get("logging", {})
        flattened_config["log_level"] = logging_config.get("level", "INFO")
        flattened_config["failure_log_file"] = logging_config.get(
            "failure_log_file", "logs/vector_indexer_failures.jsonl"
        )
        flattened_config["processing_log_file"] = logging_config.get(
            "processing_log_file", "logs/vector_indexer_processing.log"
        )
        flattened_config["stats_log_file"] = logging_config.get(
            "stats_log_file", "logs/vector_indexer_stats.json"
        )

        # Dataset config
        dataset_config = indexer_config.get("dataset", {})
        flattened_config["dataset_base_path"] = dataset_config.get(
            "base_path", "datasets"
        )
        flattened_config["target_file"] = dataset_config.get(
            "target_file", "cleaned.txt"
        )
        flattened_config["metadata_file"] = dataset_config.get(
            "metadata_file", "cleaned.meta.json"
        )

        try:
            # Create config dict with only values that were actually found in YAML
            config_kwargs: Dict[str, Any] = {}

            # Define the fields we want to extract from flattened_config
            config_fields = [
                "api_base_url",
                "api_timeout",
                "environment",
                "connection_id",
                "chunk_size",
                "chunk_overlap",
                "max_concurrent_documents",
                "max_concurrent_chunks_per_doc",
                "embedding_batch_size",
                "context_batch_size",
                "max_retries",
                "retry_delay_base",
                "continue_on_failure",
                "log_failures",
                "log_level",
                "failure_log_file",
                "processing_log_file",
                "stats_log_file",
                "dataset_base_path",
                "target_file",
                "metadata_file",
            ]

            # Only add values that exist in flattened_config (no defaults)
            for field in config_fields:
                if field in flattened_config:
                    config_kwargs[field] = flattened_config[field]

            # Always add nested config objects
            config_kwargs["chunking"] = ChunkingConfig()
            config_kwargs["processing"] = ProcessingConfig()

            return VectorIndexerConfig(**config_kwargs)
        except Exception as e:
            logger.error(f"Failed to create config object: {e}")
            return VectorIndexerConfig()

    @staticmethod
    def load_document_loader_config(
        config_path: str = "src/vector_indexer/config/vector_indexer_config.yaml",
    ) -> DocumentLoaderConfig:
        """
        Load document loader specific configuration from YAML file.

        Args:
            config_path: Path to the configuration YAML file

        Returns:
            DocumentLoaderConfig: Enhanced document loader configuration with validation
        """
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return DocumentLoaderConfig()

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")
            return DocumentLoaderConfig()

        # Extract document_loader section
        indexer_config = yaml_config.get("vector_indexer", {})
        doc_loader_config = indexer_config.get("document_loader", {})

        try:
            return DocumentLoaderConfig(**doc_loader_config)
        except Exception as e:
            logger.error(f"Failed to create document loader config object: {e}")
            return DocumentLoaderConfig()

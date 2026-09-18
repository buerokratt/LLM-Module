"""Contextual processor for implementing Anthropic's contextual retrieval methodology."""

import asyncio
import tiktoken
from typing import List, Dict, Any, Optional
from loki_logger import LokiLogger

from vector_indexer.config.config_loader import VectorIndexerConfig
from vector_indexer.models import ProcessingDocument, BaseChunk, ContextualChunk
from vector_indexer.api_client import LLMOrchestrationAPIClient
from vector_indexer.error_logger import ErrorLogger
from vector_indexer.constants import ChunkingConstants, ProcessingConstants

# Initialize Loki logger
logger = LokiLogger(service_name="contextual-processor")


class ContextualProcessor:
    """Processes documents into contextual chunks using Anthropic methodology."""

    def __init__(
        self,
        api_client: LLMOrchestrationAPIClient,
        config: VectorIndexerConfig,
        error_logger: ErrorLogger,
    ) -> None:
        self.api_client = api_client
        self.config = config
        self.error_logger = error_logger

        # Initialize tokenizer for chunk splitting
        try:
            # Use chunking config if available, otherwise fallback to constant
            if hasattr(self.config, "chunking") and self.config.chunking:
                encoding_name = self.config.chunking.tokenizer_encoding
            else:
                encoding_name = ChunkingConstants.DEFAULT_TOKENIZER_ENCODING
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(
                f"Failed to load tiktoken encoder: {e}, using simple token estimation"
            )
            self.tokenizer = None

    async def process_document(
        self, document: ProcessingDocument
    ) -> tuple[List[ContextualChunk], int]:
        """
        Process single document into contextual chunks.

        Args:
            document: Document to process

        Returns:
            Tuple of (contextual chunks with embeddings, number of chunks
            dropped due to context-generation failure)
        """
        logger.info(
            f"Processing document {document.document_hash} ({len(document.content)} characters)"
        )

        try:
            # Step 1: Split document into base chunks
            base_chunks = self._split_into_chunks(document.content)
            logger.info(f"Split document into {len(base_chunks)} chunks")

            # Step 2: Generate contexts for all chunks concurrently (but controlled)
            chunk_contents = [chunk.content for chunk in base_chunks]
            contexts = await self.api_client.generate_context_batch(
                document.content, chunk_contents
            )

            # Step 3: Create contextual chunks (filter out failed context generations)
            contextual_chunks: List[ContextualChunk] = []
            valid_contextual_contents: List[str] = []
            failed_chunks = 0

            for i, (base_chunk, context) in enumerate(
                zip(base_chunks, contexts, strict=True)
            ):
                if isinstance(context, Exception):
                    failed_chunks += 1
                    self.error_logger.log_context_generation_failure(
                        document.document_hash, i, str(context), self.config.max_retries
                    )
                    logger.warning(
                        f"Skipping chunk {i} due to context generation failure"
                    )
                    continue

                # Ensure context is string (it should be at this point since we filter out exceptions)
                context_str = str(context) if not isinstance(context, str) else context

                # Create contextual content (Anthropic methodology)
                contextual_content = f"{context_str}\n\n{base_chunk.content}"
                valid_contextual_contents.append(contextual_content)

                # Create contextual chunk object with configurable ID pattern
                if (
                    hasattr(self.config, "chunking")
                    and self.config.chunking
                    and "chunk_id_pattern" in self.config.chunking.templates
                ):
                    chunk_id_pattern = self.config.chunking.templates[
                        "chunk_id_pattern"
                    ]
                    chunk_id = chunk_id_pattern.format(
                        document_hash=document.document_hash, index=i
                    )
                else:
                    chunk_id = ChunkingConstants.CHUNK_ID_PATTERN.format(
                        document_hash=document.document_hash, index=i
                    )

                chunk = ContextualChunk(
                    chunk_id=chunk_id,
                    document_hash=document.document_hash,
                    chunk_index=i,
                    total_chunks=len(base_chunks),
                    original_content=base_chunk.content,
                    context=context_str,
                    contextual_content=contextual_content,
                    metadata=document.metadata,
                    tokens_count=self._estimate_tokens(contextual_content),
                    # Embedding fields will be set later after embedding generation
                    embedding=None,
                    embedding_model=None,
                    vector_dimensions=None,
                )

                contextual_chunks.append(chunk)

            if not contextual_chunks:
                logger.error(
                    f"No valid chunks created for document {document.document_hash}"
                )
                return [], failed_chunks

            # Step 4: Create embeddings for all valid contextual chunks
            try:
                embeddings_response = await self._create_embeddings_in_batches(
                    valid_contextual_contents
                )

                # Step 5: Add embeddings to chunks
                for chunk, embedding in zip(
                    contextual_chunks, embeddings_response["embeddings"], strict=True
                ):
                    chunk.embedding = embedding
                    chunk.embedding_model = embeddings_response["model_used"]
                    chunk.vector_dimensions = len(embedding)

            except Exception as e:
                self.error_logger.log_embedding_failure(
                    document.document_hash, str(e), self.config.max_retries
                )
                logger.error(
                    f"Failed to create embeddings for document {document.document_hash}: {e}"
                )
                raise

            logger.info(
                f"Successfully processed document {document.document_hash}: "
                f"{len(contextual_chunks)} chunks ({failed_chunks} dropped)"
            )
            return contextual_chunks, failed_chunks

        except Exception as e:
            logger.error(
                f"Document processing failed for {document.document_hash}: {e}"
            )
            raise

    def _split_into_chunks(self, content: str) -> List[BaseChunk]:
        """
        Split document content into base chunks with overlap.

        Args:
            content: Document content to split

        Returns:
            List of base chunks
        """
        chunks: List[BaseChunk] = []

        if self.tokenizer:
            # Use tiktoken for accurate token-based splitting
            tokens = self.tokenizer.encode(content)

            chunk_start = 0
            chunk_index = 0

            while chunk_start < len(tokens):
                # Calculate chunk end
                chunk_end = min(chunk_start + self.config.chunk_size, len(tokens))

                # Extract chunk tokens
                chunk_tokens = tokens[chunk_start:chunk_end]
                chunk_content = self.tokenizer.decode(chunk_tokens)

                # Find character positions in original content
                if chunk_index == 0:
                    start_char = 0
                else:
                    # Approximate character position
                    start_char = int(chunk_start * len(content) / len(tokens))

                end_char = int(chunk_end * len(content) / len(tokens))
                end_char = min(end_char, len(content))

                chunks.append(
                    BaseChunk(
                        content=chunk_content.strip(),
                        tokens=len(chunk_tokens),
                        start_index=start_char,
                        end_index=end_char,
                    )
                )

                # Move to next chunk with overlap
                chunk_start = chunk_end - self.config.chunk_overlap
                chunk_index += 1

                # Break if we've reached the end
                if chunk_end >= len(tokens):
                    break
        else:
            # Fallback: Simple character-based splitting with token estimation
            # Use configuration if available, otherwise fallback to constant
            if hasattr(self.config, "chunking") and self.config.chunking:
                char_per_token = self.config.chunking.chars_per_token
            else:
                char_per_token = ChunkingConstants.CHARS_PER_TOKEN
            chunk_size_chars = self.config.chunk_size * char_per_token
            overlap_chars = self.config.chunk_overlap * char_per_token

            start = 0
            chunk_index = 0

            while start < len(content):
                end = min(start + chunk_size_chars, len(content))

                chunk_content = content[start:end].strip()
                if chunk_content:
                    estimated_tokens = self._estimate_tokens(chunk_content)

                    chunks.append(
                        BaseChunk(
                            content=chunk_content,
                            tokens=estimated_tokens,
                            start_index=start,
                            end_index=end,
                        )
                    )

                start = end - overlap_chars
                chunk_index += 1

                if end >= len(content):
                    break

        # Filter out very small chunks using configuration
        if hasattr(self.config, "chunking") and self.config.chunking:
            min_chunk_size = self.config.chunking.min_chunk_size
        else:
            min_chunk_size = ChunkingConstants.MIN_CHUNK_SIZE_TOKENS
        chunks = [chunk for chunk in chunks if chunk.tokens >= min_chunk_size]

        logger.debug(
            f"Created {len(chunks)} chunks with average {sum(c.tokens for c in chunks) / len(chunks):.0f} tokens each"
        )
        return chunks

    async def _create_embeddings_in_batches(
        self, contextual_contents: List[str]
    ) -> Dict[str, Any]:
        """
        Create embeddings for contextual chunks in small batches.

        Args:
            contextual_contents: List of contextual content to embed

        Returns:
            Combined embeddings response
        """
        all_embeddings: List[List[float]] = []
        model_used: Optional[str] = None
        total_tokens: int = 0

        # Process in batches of embedding_batch_size (10)
        for i in range(0, len(contextual_contents), self.config.embedding_batch_size):
            batch = contextual_contents[i : i + self.config.embedding_batch_size]

            logger.debug(
                f"Creating embeddings for batch {i // self.config.embedding_batch_size + 1} ({len(batch)} chunks)"
            )

            try:
                batch_response = await self.api_client.create_embeddings_batch(batch)
                all_embeddings.extend(batch_response["embeddings"])

                if model_used is None:
                    model_used = batch_response["model_used"]

                total_tokens += batch_response.get("total_tokens", 0)

            except Exception as e:
                logger.error(
                    f"Embedding batch {i // self.config.embedding_batch_size + 1} failed: {e}"
                )
                raise

            # Small delay between batches using configuration
            if i + self.config.embedding_batch_size < len(contextual_contents):
                if hasattr(self.config, "processing") and self.config.processing:
                    delay = self.config.processing.batch_delay_seconds
                else:
                    delay = ProcessingConstants.BATCH_DELAY_SECONDS
                await asyncio.sleep(delay)

        return {
            "embeddings": all_embeddings,
            "model_used": model_used,
            "total_tokens": total_tokens,
            "provider": self._extract_provider_from_model(model_used)
            if model_used
            else "unknown",
            "dimensions": len(all_embeddings[0]) if all_embeddings else 0,
        }

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough estimation: 1 token ≈ 4 characters
            return int(len(text) / 4)

    def _extract_provider_from_model(self, model_name: str) -> str:
        """
        Extract provider name from model name.

        Args:
            model_name: Model name

        Returns:
            Provider name
        """
        if not model_name:
            return "unknown"

        if "azure" in model_name.lower() or "text-embedding-3" in model_name:
            return "azure_openai"
        elif "amazon" in model_name.lower() or "titan" in model_name.lower():
            return "aws_bedrock"
        else:
            return "unknown"

"""HTTP API client for LLM Orchestration Service."""

import asyncio
from typing import List, Dict, Any, Optional, Union
import httpx
from typing_extensions import Self
from loki_logger import LokiLogger

from vector_indexer.config.config_loader import VectorIndexerConfig

# Initialize Loki logger
logger = LokiLogger(service_name="api-client")


class LLMOrchestrationAPIClient:
    """Client for calling LLM Orchestration Service API endpoints."""

    def __init__(self, config: VectorIndexerConfig) -> None:
        self.config = config
        self.session = httpx.AsyncClient(
            timeout=config.api_timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit."""
        await self.session.aclose()

    async def generate_context_batch(
        self, document_content: str, chunks: List[str]
    ) -> List[Union[str, BaseException]]:
        """
        Generate contexts for multiple chunks concurrently with controlled batching.

        Args:
            document_content: Full document content for context
            chunks: List of chunk contents to generate context for

        Returns:
            List of generated contexts (or BaseException objects for failures)
        """
        contexts: List[Union[str, BaseException]] = []

        # Process chunks in small concurrent batches (context_batch_size = 5)
        for i in range(0, len(chunks), self.config.context_batch_size):
            batch = chunks[i : i + self.config.context_batch_size]

            # Create semaphore to limit concurrent requests (max_concurrent_chunks_per_doc = 5)
            semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks_per_doc)

            async def generate_context_with_semaphore(
                chunk_content: str, sem: asyncio.Semaphore = semaphore
            ) -> str:
                async with sem:
                    return await self._generate_context_with_retry(
                        document_content, chunk_content
                    )

            # Process batch concurrently
            batch_contexts = await asyncio.gather(
                *[generate_context_with_semaphore(chunk) for chunk in batch],
                return_exceptions=True,
            )

            contexts.extend(batch_contexts)

            # Small delay between batches to be gentle on the API
            if i + self.config.context_batch_size < len(chunks):
                await asyncio.sleep(0.1)

        return contexts

    async def _generate_context_with_retry(
        self, document_content: str, chunk_content: str
    ) -> str:
        """Generate context with retry logic - calls /generate-context endpoint."""

        # Construct the exact Anthropic prompt structure
        request_data = {
            "document_prompt": f"<document>\n{document_content}\n</document>",
            "chunk_prompt": f"""Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.""",
            "environment": self.config.environment,
            "use_cache": True,
            "connection_id": self.config.connection_id,
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"Calling /generate-context (attempt {attempt + 1}/{self.config.max_retries})"
                )

                response = await self.session.post(
                    f"{self.config.api_base_url}/generate-context", json=request_data
                )
                response.raise_for_status()
                result = response.json()

                context = result.get("context", "").strip()
                if not context:
                    raise ValueError("Empty context returned from API")

                logger.debug(
                    f"Successfully generated context: {len(context)} characters"
                )
                return context

            except Exception as e:
                last_error = e
                logger.warning(f"Context generation attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_base**attempt
                    logger.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries failed
        error_msg = f"Context generation failed after {self.config.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def create_embeddings_batch(
        self, contextual_texts: List[str]
    ) -> Dict[str, Any]:
        """Create embeddings with smaller batch size and retry logic."""

        request_data = {
            "texts": contextual_texts,
            "environment": self.config.environment,
            "connection_id": self.config.connection_id,
            "batch_size": self.config.embedding_batch_size,  # Small batch size (10)
        }

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    f"Calling /embeddings for {len(contextual_texts)} texts (attempt {attempt + 1}/{self.config.max_retries})"
                )

                response = await self.session.post(
                    f"{self.config.api_base_url}/embeddings", json=request_data
                )
                response.raise_for_status()
                result = response.json()

                # Validate response
                embeddings = result.get("embeddings", [])
                if len(embeddings) != len(contextual_texts):
                    raise ValueError(
                        f"Expected {len(contextual_texts)} embeddings, got {len(embeddings)}"
                    )

                logger.debug(
                    f"Successfully created {len(embeddings)} embeddings using {result.get('model_used')}"
                )
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Embedding creation attempt {attempt + 1} failed: {e}")

                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay_base**attempt
                    logger.debug(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries failed
        error_msg = f"Embedding creation failed after {self.config.max_retries} attempts: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def health_check(self) -> bool:
        """Check if the LLM Orchestration Service is accessible."""
        try:
            # Simple connectivity test - try to make a minimal request
            response = await self.session.get(
                f"{self.config.api_base_url}/health", timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the HTTP session."""
        await self.session.aclose()

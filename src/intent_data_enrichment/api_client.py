"""API client for LLM Orchestration Service."""

import asyncio
import httpx
from typing import List, Optional
from types import TracebackType
from src.loki_logger import LokiLogger

from intent_data_enrichment.constants import EnrichmentConstants
from intent_data_enrichment.models import ServiceData

# Initialize Loki logger
logger = LokiLogger(service_name="intent-enrichment-api-client")


class LLMAPIClient:
    """Client for calling LLM Orchestration Service endpoints."""

    def __init__(
        self,
        api_base_url: str = EnrichmentConstants.DEFAULT_API_BASE_URL,
        environment: str = EnrichmentConstants.DEFAULT_ENVIRONMENT,
        connection_id: Optional[str] = EnrichmentConstants.DEFAULT_CONNECTION_ID,
        max_retries: int = EnrichmentConstants.MAX_RETRIES,
        retry_delay_base: int = EnrichmentConstants.RETRY_DELAY_BASE,
        timeout: int = EnrichmentConstants.REQUEST_TIMEOUT,
    ) -> None:
        self.api_base_url = api_base_url
        self.environment = environment
        self.connection_id = connection_id
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base
        self.timeout = timeout
        self.session: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "LLMAPIClient":
        """Async context manager entry."""
        self.session = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()

    async def generate_context(self, service_data: ServiceData) -> str:
        """
        Generate rich context for service using LLM.

        Args:
            service_data: Service data to enrich

        Returns:
            Generated context string

        Raises:
            RuntimeError: If context generation fails after all retries
        """
        # Build full service information
        full_service_info = f"""Service: {service_data.name}
ID: {service_data.service_id}
Description: {service_data.description}
Examples: {", ".join(service_data.examples)}
Entities: {", ".join(service_data.entities)}"""

        # Build context generation prompt
        context_prompt = EnrichmentConstants.CONTEXT_TEMPLATE.format(
            full_service_info=full_service_info,
            name=service_data.name,
            description=service_data.description,
            examples=", ".join(service_data.examples),
        )

        request_data = {
            "document_prompt": "",  # Empty for service enrichment
            "chunk_prompt": context_prompt,
            "environment": self.environment,
            "use_cache": True,
            "connection_id": self.connection_id,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Generating context for service '{service_data.service_id}' "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )

                if not self.session:
                    raise RuntimeError("HTTP session not initialized")

                response = await self.session.post(
                    f"{self.api_base_url}/generate-context", json=request_data
                )
                response.raise_for_status()
                result = response.json()

                context = result.get("context", "").strip()
                if not context:
                    raise ValueError("Empty context returned from API")

                logger.success(
                    f"Successfully generated context for '{service_data.service_id}': "
                    f"{len(context)} characters"
                )
                return context

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Context generation attempt {attempt + 1} failed for "
                    f"'{service_data.service_id}': {e}"
                )

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay_base**attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries failed
        error_msg = (
            f"Context generation failed for '{service_data.service_id}' "
            f"after {self.max_retries} attempts: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    async def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding creation fails after all retries
        """
        request_data = {
            "texts": [text],
            "environment": self.environment,
            "connection_id": self.connection_id,
            "batch_size": 1,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Creating embedding (attempt {attempt + 1}/{self.max_retries})"
                )

                if not self.session:
                    raise RuntimeError("HTTP session not initialized")

                response = await self.session.post(
                    f"{self.api_base_url}/embeddings", json=request_data
                )
                response.raise_for_status()
                result = response.json()

                embeddings = result.get("embeddings", [])
                if not embeddings or not embeddings[0]:
                    raise ValueError("Empty embedding returned from API")

                embedding = embeddings[0]
                logger.success(
                    f"Successfully created embedding: dimension {len(embedding)}"
                )
                return embedding

            except Exception as e:
                last_error = e
                logger.warning(f"Embedding creation attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay_base**attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        # All retries failed
        error_msg = (
            f"Embedding creation failed after {self.max_retries} attempts: {last_error}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

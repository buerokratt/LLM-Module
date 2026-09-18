"""Embedding Manager for DSPy integration with vault secrets."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import dspy
import numpy as np
from src.loki_logger import LokiLogger
from pydantic import BaseModel
from .vault.vault_client import VaultAgentClient
from .config.loader import ConfigurationLoader
from .exceptions import ConfigurationError

# Initialize Loki logger
logger = LokiLogger(service_name="embedding-manager")


class EmbeddingFailure(BaseModel):
    """Model for tracking embedding failures."""

    texts: List[str]
    error_message: str
    timestamp: float
    attempt_count: int
    model_name: str


class EmbeddingManager:
    """Manager for DSPy embedding models with vault integration."""

    def __init__(
        self, vault_client: VaultAgentClient, config_loader: ConfigurationLoader
    ) -> None:
        """Initialize embedding manager."""
        self.vault_client = vault_client
        self.config_loader = config_loader
        self.embedders: Dict[str, dspy.Embedder] = {}
        self.failure_log_path = Path("logs/embedding_failures.jsonl")
        self.failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    def get_embedder(
        self, environment: str = "production", connection_id: Optional[str] = None
    ) -> dspy.Embedder:
        """Get or create DSPy Embedder instance using vault-driven model resolution.

        Args:
            environment: Environment (production, development, test)
            connection_id: Optional connection ID for dev/test environments

        Returns:
            Configured DSPy embedder instance

        Raises:
            ConfigurationError: If no embedding models are available or configuration fails
        """
        # Resolve model from vault using ConfigurationLoader
        try:
            provider_name, model_name = self.config_loader.resolve_embedding_model(
                environment, connection_id
            )

            cache_key: str = f"{provider_name}_{model_name}_{environment}_{connection_id or 'default'}"

            if cache_key in self.embedders:
                logger.debug(f"Using cached embedder: {provider_name}/{model_name}")
                return self.embedders[cache_key]

            # Get full configuration with secrets from embeddings vault path
            config: Dict[str, Any] = self.config_loader.get_embedding_provider_config(
                provider_name, model_name, environment, connection_id
            )

            # Create DSPy embedder based on provider
            embedder: dspy.Embedder = self._create_dspy_embedder(config)
            self.embedders[cache_key] = embedder

            logger.info(f"Created embedder for model: {provider_name}/{model_name}")
            return embedder

        except Exception as e:
            logger.error(f"Failed to create embedder: {e}")
            raise ConfigurationError(f"Embedder creation failed: {e}") from e

    def create_embeddings(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 50,
    ) -> Dict[str, Any]:
        """Create embeddings using DSPy with vault-driven model resolution.

        Args:
            texts: List of texts to embed
            environment: Environment (production, development, test)
            connection_id: Optional connection ID for dev/test environments
            batch_size: Batch size for processing

        Returns:
            Dictionary with embeddings and metadata

        Raises:
            ConfigurationError: If embedding creation fails
        """
        embedder: dspy.Embedder = self.get_embedder(environment, connection_id)

        # Get the resolved model information for metadata
        provider_name, model_name = self.config_loader.resolve_embedding_model(
            environment, connection_id
        )
        model_identifier: str = f"{provider_name}/{model_name}"

        try:
            # Process in batches
            all_embeddings: List[List[float]] = []
            total_tokens: int = 0

            for i in range(0, len(texts), batch_size):
                batch_texts: List[str] = texts[i : i + batch_size]
                logger.info(f"Processing embedding batch {i // batch_size + 1}")

                # Use Python's generic exponential backoff
                batch_embeddings: np.ndarray = self._create_embeddings_with_retry(
                    embedder, batch_texts, model_identifier
                )

                # DEBUG: Log embedding conversion process
                logger.info("=== EMBEDDING CONVERSION DEBUG ===")
                logger.info(f"Batch texts: {len(batch_texts)}")
                logger.info(f"batch_embeddings shape: {batch_embeddings.shape}")

                embedding_list: List[List[float]] = batch_embeddings.tolist()
                logger.info(f"After .tolist() - type: {type(embedding_list)}")
                logger.info(f"After .tolist() - length: {len(embedding_list)}")

                if len(embedding_list) > 0:
                    logger.info(f"First item type: {type(embedding_list[0])}")
                    logger.info(f"First embedding dimensions: {len(embedding_list[0])}")

                logger.info(
                    f"all_embeddings count before extend: {len(all_embeddings)}"
                )
                all_embeddings.extend(embedding_list)
                logger.info(f"all_embeddings count after extend: {len(all_embeddings)}")
                logger.info("=== END EMBEDDING CONVERSION DEBUG ===")

                # Estimate tokens (rough approximation)
                total_tokens += int(
                    sum(len(text.split()) * 1.3 for text in batch_texts)
                )

            return {
                "embeddings": all_embeddings,
                "model_used": model_identifier,
                "processing_info": {
                    "batch_count": (len(texts) + batch_size - 1) // batch_size,
                    "total_texts": len(texts),
                    "batch_size": batch_size,
                },
                "total_tokens": int(total_tokens),
            }

        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            self._log_embedding_failure(texts, str(e), model_identifier)
            raise

    def _create_embeddings_with_retry(
        self,
        embedder: dspy.Embedder,
        texts: List[str],
        model_name: str,
        max_attempts: int = 3,
    ) -> np.ndarray:
        """Create embeddings with Python's generic exponential backoff."""
        last_exception: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                logger.info(f"Embedding attempt {attempt + 1}/{max_attempts}")
                raw_embeddings = embedder(texts)

                return raw_embeddings

            except Exception as e:
                last_exception = e
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")

                if attempt < max_attempts - 1:
                    # Exponential backoff: 2^attempt seconds (1, 2, 4, 8...)
                    delay = 2**attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Final attempt failed, log and raise
                    self._log_embedding_failure(texts, str(e), model_name, attempt + 1)

        if last_exception:
            raise last_exception

        # This should never be reached, but makes pyright happy
        raise RuntimeError("Unexpected error in retry logic")

    def _create_dspy_embedder(self, config: Dict[str, Any]) -> dspy.Embedder:
        """Create DSPy embedder from vault configuration."""
        try:
            # For Azure OpenAI
            if "azure" in config.get("endpoint", "").lower():
                model_string = f"azure/{config['deployment_name']}"
                # DSPy will use environment variables or we can pass them
                return dspy.Embedder(
                    model=model_string,
                    api_key=config["api_key"],
                    api_base=config["endpoint"],  # or extract base URL
                    api_version=config["api_version"],
                    batch_size=50,
                    caching=True,
                )

            # For OpenAI
            elif "openai" in config.get("endpoint", "").lower():
                return dspy.Embedder(
                    model=f"openai/{config['model']}", batch_size=50, caching=True
                )

            # For AWS Bedrock
            else:
                return dspy.Embedder(
                    model=f"bedrock/{config['model']}", batch_size=50, caching=True
                )

        except Exception as e:
            logger.error(f"Failed to create DSPy embedder: {e}")
            raise ConfigurationError(f"Could not create embedder: {e}") from e

    def _log_embedding_failure(
        self,
        texts: List[str],
        error_message: str,
        model_name: str,
        attempt_count: int = 1,
    ) -> None:
        """Log embedding failure to file for later retry."""
        failure = EmbeddingFailure(
            texts=texts,
            error_message=error_message,
            timestamp=time.time(),
            attempt_count=attempt_count,
            model_name=model_name,
        )

        try:
            with open(self.failure_log_path, "a", encoding="utf-8") as f:
                f.write(failure.model_dump_json() + "\n")
            logger.info(f"Logged embedding failure to {self.failure_log_path}")
        except Exception as e:
            logger.error(f"Failed to log embedding failure: {e}")

    def get_available_models(self, environment: str) -> List[str]:
        """Get available embedding models from vault using ConfigurationLoader."""
        try:
            available_models: Dict[str, List[str]] = (
                self.config_loader.get_available_embedding_models(environment)
            )
            # Flatten the dictionary values into a single list
            all_models: List[str] = []
            for provider_models in available_models.values():
                all_models.extend(provider_models)
            return all_models
        except ConfigurationError as e:
            logger.warning(f"Could not get available embedding models: {e}")
            # Fallback to static list if vault query fails
            return [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return ["text-embedding-3-small"]  # Fallback

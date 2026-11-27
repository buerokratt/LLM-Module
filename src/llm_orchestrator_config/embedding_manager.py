"""Embedding Manager for DSPy integration with vault secrets."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import dspy
import numpy as np
from src.loki_logger import LokiLogger
from pydantic import BaseModel
from .vault.vault_client import VaultAgentClient
from .config.loader import ConfigurationLoader
from .exceptions import ConfigurationError

# Initialize Loki logger
logger = LokiLogger(service_name="embedding-manager")


class EndpointHostChangedError(ConfigurationError):
    """A test embedding with saved credentials was asked to use an endpoint on
    another host than the one those credentials are saved for."""

    # Read by callers instead of isinstance: this module is imported under two
    # package paths (src.llm_orchestrator_config / llm_orchestrator_config).
    detail = "endpoint_host_changed"


def ensure_same_endpoint_host(saved_endpoint: str, requested_endpoint: str) -> None:
    """Allow a requested endpoint only on the saved endpoint's host, over https.

    A connection's saved API key must never be sent anywhere new: an endpoint
    override comes from the request, so without this a caller could point the
    test embedding at its own server and receive the key (and probe internal
    hosts). A different host needs the key entered again (a new secret).

    Raises:
        EndpointHostChangedError: The requested endpoint is not https or is on
            another host or port than the saved one.
    """
    saved = urlparse(saved_endpoint.strip())
    requested = urlparse(requested_endpoint.strip())
    if requested.scheme.lower() != "https" or not requested.hostname:
        raise EndpointHostChangedError("The embedding endpoint must be an https URL")
    if (requested.hostname.lower(), requested.port) != (
        (saved.hostname or "").lower(),
        saved.port,
    ):
        raise EndpointHostChangedError(
            "The embedding endpoint is on another host than the saved API key; "
            "enter the API key again to use it"
        )


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

    def wait_until_embedding_secret_written(
        self, connection_id: str, wait_seconds: float = 30.0
    ) -> bool:
        """Wait until a connection's embedding secret exists in Vault.

        Only checks that it was written, not that it is valid: CronManager writes
        the LLM secret first and the embedding secret last, so once it exists the
        store is finished and the secrets can safely be deleted.

        Returns:
            True when it appeared, False after ``wait_seconds``.
        """
        deadline = time.monotonic() + wait_seconds
        while True:
            for provider in ("azure_openai", "aws_bedrock"):
                try:
                    if self.vault_client.get_secret(
                        f"embeddings/connections/{provider}/{connection_id}"
                    ):
                        return True
                except Exception as e:
                    logger.debug(f"Waiting for embedding secret {connection_id}: {e}")
            if time.monotonic() >= deadline:
                return False
            time.sleep(1.0)

    def wait_for_embedding_secret(
        self, connection_id: str, wait_seconds: float = 30.0
    ) -> tuple[str, str]:
        """Wait until a connection's embedding secret can be read from Vault.

        CronManager stores secrets asynchronously, LLM secret first and the
        embedding secret last, so once this returns the whole store is finished
        and the secret can safely be deleted.

        Returns:
            (provider_name, model_name) of the embedding secret

        Raises:
            Exception: The secret did not appear within ``wait_seconds``.
        """
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                return self.config_loader.resolve_embedding_model(
                    "testing", connection_id
                )
            # Not `except ConfigurationError`: the loader imports that class as
            # llm_orchestrator_config.exceptions and this module as
            # src.llm_orchestrator_config.exceptions, which Python treats as two
            # different classes, so it would never match and nothing would wait.
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1.0)

    def measure_test_embedding(
        self,
        connection_id: str,
        overrides: Optional[Dict[str, str]] = None,
        wait_seconds: float = 30.0,
    ) -> Dict[str, Any]:
        """Embed one short text with a connection's real credentials.

        Used by the knowledge base embedding lock (issue #806) to find the model
        that is really behind a connection: on Azure the deployment name decides
        it, not the dropdown, so the vector size is what tells.

        Args:
            connection_id: Vault UUID holding the embedding credentials. The GUI
                stores them through CronManager, which returns before the secret
                is written, so a missing secret is retried for ``wait_seconds``.
            overrides: Non-secret Azure fields (``deployment_name``,
                ``endpoint``) to test instead of the stored ones, for edits that
                keep the saved key. An ``endpoint`` must be on the stored
                endpoint's host (see ensure_same_endpoint_host).
            wait_seconds: How long to wait for the secret to appear in Vault.

        Returns:
            ``{"provider", "model", "vector_dimensions"}``

        Raises:
            EndpointHostChangedError: The endpoint override is not on the stored
                endpoint's host; nothing was sent.
            ConfigurationError: Secret not found in time, or the test embedding failed.
        """
        provider_name, model_name = self.wait_for_embedding_secret(
            connection_id, wait_seconds
        )

        config: Dict[str, Any] = self.config_loader.get_embedding_provider_config(
            provider_name, model_name, "testing", connection_id
        )
        given = {key: value for key, value in (overrides or {}).items() if value}
        if "endpoint" in given:
            ensure_same_endpoint_host(
                str(config.get("endpoint") or ""), given["endpoint"]
            )
        config.update(given)

        # Not cached: a cached result for the same text could hide bad credentials.
        embedder = self._create_dspy_embedder(config, caching=False)
        try:
            vectors = embedder(["Knowledge base embedding model check"])
        except Exception as e:
            raise ConfigurationError(f"Test embedding failed: {e}") from e

        return {
            "provider": provider_name,
            "model": model_name,
            "vector_dimensions": int(np.asarray(vectors).shape[-1]),
        }

    def _create_dspy_embedder(
        self, config: Dict[str, Any], caching: bool = True
    ) -> dspy.Embedder:
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
                    caching=caching,
                )

            # For OpenAI
            elif "openai" in config.get("endpoint", "").lower():
                return dspy.Embedder(
                    model=f"openai/{config['model']}", batch_size=50, caching=caching
                )

            # For AWS Bedrock
            else:
                return dspy.Embedder(
                    model=f"bedrock/{config['model']}", batch_size=50, caching=caching
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

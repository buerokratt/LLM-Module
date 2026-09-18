"""Secret resolver with TTL cache and background refresh for LLM Config Module."""

import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, ValidationError
from src.loki_logger import LokiLogger

from llm_orchestrator_config.vault.vault_client import (
    VaultAgentClient,
    get_vault_client,
)
from llm_orchestrator_config.vault.models import (
    AzureOpenAISecret,
    AWSBedrockSecret,
    get_secret_model,
)
from llm_orchestrator_config.vault.exceptions import VaultConnectionError

# Initialize Loki logger
logger = LokiLogger(service_name="secret-resolver")


class CachedSecret(BaseModel):
    """Cached secret with TTL information."""

    data: Union[AzureOpenAISecret, AWSBedrockSecret]
    expires_at: datetime
    path: str


class SecretResolver:
    """Vault secret resolver with TTL cache and fallback support."""

    def __init__(
        self,
        vault_client: Optional[VaultAgentClient] = None,
        cache_ttl_minutes: int = 5,
        background_refresh: bool = True,
    ) -> None:
        """Initialize Secret Resolver.

        Args:
            vault_client: Vault client instance (creates default if None)
            cache_ttl_minutes: Cache TTL in minutes
            background_refresh: Enable background refresh of expired secrets
        """
        self.vault_client = vault_client or get_vault_client()
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.background_refresh = background_refresh

        # In-memory cache: path -> CachedSecret
        self._cache: Dict[str, CachedSecret] = {}
        self._cache_lock = threading.RLock()

        # Last-known-good fallback cache
        self._fallback_cache: Dict[str, Union[AzureOpenAISecret, AWSBedrockSecret]] = {}

        logger.info(f"SecretResolver initialized with {cache_ttl_minutes}min TTL")

    def get_secret_for_model(
        self,
        provider: str,
        environment: str,
        model_name: str,
        connection_id: Optional[str] = None,
    ) -> Optional[Union[AzureOpenAISecret, AWSBedrockSecret]]:
        """Get secret for a specific model.

        Args:
            provider: Provider name (azure_openai, aws_bedrock)
            environment: Environment (production, development, test)
            model_name: Model name from llm_config.yaml
            connection_id: Optional connection ID for dev/test environments

        Returns:
            Validated secret object or None if not found
        """
        # Determine the vault path
        vault_path = self._build_vault_path(
            provider, environment, model_name, connection_id
        )

        # Try cache first
        cached_secret = self._get_from_cache(vault_path)
        if cached_secret:
            return cached_secret

        # Fetch from Vault
        try:
            secret_data = self.vault_client.get_secret(vault_path)
            if not secret_data:
                logger.debug(f"Secret not found in Vault: {vault_path}")
                return self._get_fallback(vault_path)

            # Validate and parse secret
            secret_model = get_secret_model(provider)
            validated_secret = secret_model(**secret_data)

            # Verify model name matches (more flexible for production)
            if environment == "production":
                # For production, trust the model name from vault secret
                logger.debug(
                    f"Production secret model: {validated_secret.model}, requested: {model_name}"
                )
            elif validated_secret.model != model_name:
                logger.warning(
                    f"Model name mismatch: vault={validated_secret.model}, "
                    f"requested={model_name}"
                )
                # Continue anyway - vault might have updated model name

            # Cache the secret
            self._cache_secret(vault_path, validated_secret)

            # Update fallback cache
            self._fallback_cache[vault_path] = validated_secret

            logger.debug(f"Successfully resolved secret for {provider}/{model_name}")
            return validated_secret

        except VaultConnectionError:
            logger.warning(f"Vault unavailable, trying fallback for {vault_path}")
            return self._get_fallback(vault_path)
        except ValidationError as e:
            # The secret exists but does not match the provider's schema. A
            # last-known-good fallback would mask a stored-data problem that no
            # retry can fix, so report it as a misconfiguration and stop here.
            logger.error(
                f"Malformed secret at {vault_path} for provider {provider} - the "
                f"stored value does not match {get_secret_model(provider).__name__} "
                f"and must be rewritten: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Error resolving secret for {vault_path}: {e}")
            return self._get_fallback(vault_path)

    def list_available_models(self, provider: str, environment: str) -> list[str]:
        """List available models for a provider.

        Args:
            provider: Provider name (azure_openai, aws_bedrock)
            environment: Environment (kept for API compatibility, not used in path)

        Returns:
            List of available vault UUIDs under this provider
        """
        # List all UUIDs under provider (no environment in path)
        base_path = f"llm/connections/{provider}"
        try:
            entries = self.vault_client.list_secrets(base_path)
            if entries:
                logger.debug(f"Found {len(entries)} entries for {provider}: {entries}")
                return entries
            else:
                logger.debug(f"No entries found for {provider}")
                return []

        except Exception as e:
            logger.error(f"Error listing models for {provider}: {e}")
            return []

    def refresh_secret(self, vault_path: str) -> bool:
        """Manually refresh a specific secret.

        Args:
            vault_path: Vault path to refresh

        Returns:
            True if refresh successful
        """
        try:
            secret_data = self.vault_client.get_secret(vault_path)
            if not secret_data:
                logger.debug(f"Secret not found during refresh: {vault_path}")
                return False

            # Determine provider from path
            path_parts = vault_path.split("/")
            if len(path_parts) >= 4:
                provider = path_parts[2]  # llm/connections/PROVIDER/...
                secret_model = get_secret_model(provider)
                validated_secret = secret_model(**secret_data)

                # Update cache
                self._cache_secret(vault_path, validated_secret)
                self._fallback_cache[vault_path] = validated_secret

                logger.debug(f"Successfully refreshed secret: {vault_path}")
                return True

        except Exception as e:
            logger.error(f"Error refreshing secret {vault_path}: {e}")

        return False

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status information.

        Returns:
            Dictionary with cache statistics
        """
        with self._cache_lock:
            now = datetime.now()
            total_secrets = len(self._cache)
            expired_secrets = sum(
                1 for cached in self._cache.values() if cached.expires_at <= now
            )

            return {
                "total_cached_secrets": total_secrets,
                "expired_secrets": expired_secrets,
                "active_secrets": total_secrets - expired_secrets,
                "fallback_secrets": len(self._fallback_cache),
                "cache_ttl_minutes": int(self.cache_ttl.total_seconds() / 60),
            }

    def clear_cache(self) -> None:
        """Clear all cached secrets."""
        with self._cache_lock:
            self._cache.clear()
            logger.info("Secret cache cleared")

    def _build_vault_path(
        self,
        provider: str,
        environment: str,
        model_name: str,
        connection_id: Optional[str] = None,
    ) -> str:
        """Build Vault path for a secret.

        Uses UUID (connection_id) as the sole path terminal.
        Environment is not part of the path — swaps are DB-only.

        Path format: llm/connections/{provider}/{uuid}
        """
        if not connection_id:
            raise ValueError(
                f"connection_id (vault_uuid) is required to build vault path "
                f"for provider={provider}, environment={environment}"
            )
        return f"llm/connections/{provider}/{connection_id}"

    def _get_from_cache(
        self, vault_path: str
    ) -> Optional[Union[AzureOpenAISecret, AWSBedrockSecret]]:
        """Get secret from cache if valid."""
        with self._cache_lock:
            cached = self._cache.get(vault_path)
            if cached and cached.expires_at > datetime.now():
                logger.debug(f"Cache hit for {vault_path}")
                return cached.data
            elif cached:
                logger.debug(f"Cache expired for {vault_path}")
                # Optionally trigger background refresh
                if self.background_refresh:
                    self._schedule_background_refresh(vault_path)
            return None

    def _cache_secret(
        self, vault_path: str, secret: Union[AzureOpenAISecret, AWSBedrockSecret]
    ) -> None:
        """Cache a secret with TTL."""
        with self._cache_lock:
            expires_at = datetime.now() + self.cache_ttl
            self._cache[vault_path] = CachedSecret(
                data=secret, expires_at=expires_at, path=vault_path
            )
            logger.debug(f"Cached secret {vault_path} until {expires_at}")

    def _get_fallback(
        self, vault_path: str
    ) -> Optional[Union[AzureOpenAISecret, AWSBedrockSecret]]:
        """Get secret from fallback cache."""
        fallback = self._fallback_cache.get(vault_path)
        if fallback:
            logger.info(f"Using fallback secret for {vault_path}")
            return fallback
        return None

    def _schedule_background_refresh(self, vault_path: str) -> None:
        """Schedule background refresh of an expired secret."""

        def refresh_task() -> None:
            logger.debug(f"Background refresh for {vault_path}")
            self.refresh_secret(vault_path)

        # Use threading for background refresh
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()

    # Embedding-specific methods using separate vault paths

    def get_embedding_secret_for_model(
        self,
        provider: str,
        environment: str,
        model_name: str,
        connection_id: Optional[str] = None,
    ) -> Optional[Union[AzureOpenAISecret, AWSBedrockSecret]]:
        """Get secret for a specific embedding model.

        Args:
            provider: Provider name (azure_openai, aws_bedrock)
            environment: Environment (production, development, test)
            model_name: Embedding model name from vault
            connection_id: Optional connection ID for dev/test environments

        Returns:
            Validated secret object or None if not found
        """
        # Build embeddings-specific vault path
        vault_path: str = self._build_embedding_vault_path(
            provider, environment, model_name, connection_id
        )

        # Try cache first
        cached_secret: Optional[Union[AzureOpenAISecret, AWSBedrockSecret]] = (
            self._get_from_cache(vault_path)
        )
        if cached_secret:
            return cached_secret

        # Fetch from Vault
        try:
            secret_data: Optional[Dict[str, Any]] = self.vault_client.get_secret(
                vault_path
            )
            if not secret_data:
                logger.debug(f"Embedding secret not found in Vault: {vault_path}")
                return self._get_fallback(vault_path)

            # Validate and parse secret
            secret_model: type = get_secret_model(provider)
            validated_secret: Union[AzureOpenAISecret, AWSBedrockSecret] = secret_model(
                **secret_data
            )

            # Verify model name matches (more flexible for production)
            if environment == "production":
                # For production, trust the model name from vault secret
                logger.debug(
                    f"Production embedding model: {validated_secret.model}, requested: {model_name}"
                )
            elif validated_secret.model != model_name:
                logger.warning(
                    f"Embedding model name mismatch: vault={validated_secret.model}, "
                    f"requested={model_name}"
                )
                # Continue anyway - vault might have updated model name

            # Cache the secret
            self._cache_secret(vault_path, validated_secret)

            # Update fallback cache
            self._fallback_cache[vault_path] = validated_secret

            logger.debug(
                f"Successfully resolved embedding secret for {provider}/{model_name}"
            )
            return validated_secret

        except VaultConnectionError:
            logger.warning(
                f"Vault unavailable, trying fallback for embedding {vault_path}"
            )
            return self._get_fallback(vault_path)
        except ValidationError as e:
            # See get_secret_for_model: a schema mismatch is a stored-data
            # problem, not a transient one, so do not serve a stale fallback.
            logger.error(
                f"Malformed embedding secret at {vault_path} for provider {provider} - "
                f"the stored value does not match "
                f"{get_secret_model(provider).__name__} and must be rewritten: {e}"
            )
            return None
        except Exception as e:
            logger.error(f"Error resolving embedding secret for {vault_path}: {e}")
            return self._get_fallback(vault_path)

    def list_available_embedding_models(
        self, provider: str, environment: str
    ) -> List[str]:
        """List available embedding models for a provider.

        Args:
            provider: Provider name (azure_openai, aws_bedrock)
            environment: Environment (kept for API compatibility, not used in path)

        Returns:
            List of available vault UUIDs under this provider
        """
        # List all UUIDs under provider (no environment in path)
        base_path: str = f"embeddings/connections/{provider}"
        try:
            entries: Optional[list[str]] = self.vault_client.list_secrets(base_path)
            if entries:
                logger.debug(
                    f"Found {len(entries)} embedding entries for {provider}: {entries}"
                )
                return entries
            else:
                logger.debug(f"No embedding entries found for {provider}")
                return []

        except Exception as e:
            logger.error(f"Error listing embedding models for {provider}: {e}")
            return []

    def _build_embedding_vault_path(
        self,
        provider: str,
        environment: str,
        model_name: str,
        connection_id: Optional[str] = None,
    ) -> str:
        """Build Vault path for embedding secrets.

        Uses UUID (connection_id) as the sole path terminal.
        Environment is not part of the path — swaps are DB-only.

        Path format: embeddings/connections/{provider}/{uuid}
        """
        if not connection_id:
            raise ValueError(
                f"connection_id (vault_uuid) is required to build embedding vault path "
                f"for provider={provider}, environment={environment}"
            )
        return f"embeddings/connections/{provider}/{connection_id}"

"""Configuration schema definitions for the LLM Config Module."""

from typing import Dict, Any, Optional
from pydantic import BaseModel
from llm_orchestrator_config.types import LLMProvider


class VaultConfig(BaseModel):
    """Configuration for HashiCorp Vault integration."""

    url: str = "http://vault:8200"
    token: str = ""
    enabled: bool = True


class ProviderConfig(BaseModel):
    """Base configuration for LLM providers."""

    enabled: bool
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


class AzureOpenAIConfig(ProviderConfig):
    """Configuration for Azure OpenAI provider."""

    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2025-01-01-preview"
    deployment_name: str = ""
    model_type: str = "chat"
    cache: bool = False
    num_retries: int = 3
    # Reasoning models only; ignored by standard chat models such as gpt-4.1.
    reasoning_effort: str = "medium"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "endpoint": self.endpoint,
                "api_key": self.api_key,
                "api_version": self.api_version,
                "deployment_name": self.deployment_name,
                "model_type": self.model_type,
                "cache": self.cache,
                "num_retries": self.num_retries,
                "reasoning_effort": self.reasoning_effort,
            }
        )
        return base_dict


class AWSBedrockConfig(ProviderConfig):
    """Configuration for AWS Bedrock provider."""

    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    session_token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "region": self.region,
                "access_key_id": self.access_key_id,
                "secret_access_key": self.secret_access_key,
                "session_token": self.session_token,
            }
        )
        return base_dict


class LLMConfiguration(BaseModel):
    """Main configuration container for LLM settings."""

    vault: Optional[VaultConfig] = None
    default_provider: LLMProvider
    providers: Dict[str, ProviderConfig]

    def get_provider_config(self, provider: LLMProvider) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider."""
        return self.providers.get(provider.value)

    def is_provider_enabled(self, provider: LLMProvider) -> bool:
        """Check if a provider is enabled."""
        config = self.get_provider_config(provider)
        is_enabled = config is not None and config.enabled
        return is_enabled

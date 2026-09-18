"""Pydantic models for Vault connection secrets."""

from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field, field_validator

# Type alias for cleaner code
TagsInput = Union[str, List[str], None]


class BaseConnectionSecret(BaseModel):
    """Base model for connection secrets stored in Vault."""

    connection_id: str = Field(..., description="Unique connection identifier")
    model: str = Field(..., description="Model name (must match llm_config.yaml)")
    environment: Optional[str] = Field(
        None,
        description="Environment: production/development/test (legacy, not stored in new secrets)",
    )
    tags: List[str] = Field(default_factory=list, description="Connection tags")

    @field_validator("tags", mode="before")
    @classmethod
    def parse_tags(cls, value: TagsInput) -> List[str]:
        """Convert string tags to list if needed."""
        if value is None:
            return []

        # Handle string case - split by comma
        if isinstance(value, str):
            return [tag.strip() for tag in value.split(",") if tag.strip()]

        # Handle list case - convert all items to strings
        return [str(tag).strip() for tag in value if str(tag).strip()]


class AzureOpenAISecret(BaseConnectionSecret):
    """Azure OpenAI connection secrets from Vault."""

    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    deployment_name: str = Field(..., description="Azure deployment name")
    api_version: str = Field(..., description="Azure OpenAI API version")


class AWSBedrockSecret(BaseConnectionSecret):
    """AWS Bedrock connection secrets from Vault."""

    region: str = Field(..., description="AWS region")
    access_key_id: str = Field(..., description="AWS access key ID")
    secret_access_key: str = Field(..., description="AWS secret access key")
    session_token: Optional[str] = Field(None, description="AWS session token")


# Type mapping for provider secrets
PROVIDER_SECRET_DATA_MODELS: Dict[str, type] = {
    "azure_openai": AzureOpenAISecret,
    "aws_bedrock": AWSBedrockSecret,
}


def get_secret_model(provider: str) -> type:
    """Get the appropriate secret model for a provider.

    Args:
        provider: Provider name (azure_openai, aws_bedrock)

    Returns:
        Pydantic model class for the provider

    Raises:
        ValueError: If provider is not supported
    """
    if provider not in PROVIDER_SECRET_DATA_MODELS:
        raise ValueError(f"Unsupported provider: {provider}")
    return PROVIDER_SECRET_DATA_MODELS[provider]

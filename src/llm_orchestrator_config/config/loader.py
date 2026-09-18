"""Configuration loader for the LLM Config Module."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import yaml
from dotenv import load_dotenv
from src.loki_logger import LokiLogger

from llm_orchestrator_config.config.schema import (
    LLMConfiguration,
    ProviderConfig,
    AzureOpenAIConfig,
    AWSBedrockConfig,
    VaultConfig,
)
from llm_orchestrator_config.vault.secret_resolver import SecretResolver
from llm_orchestrator_config.vault.models import AzureOpenAISecret, AWSBedrockSecret
from llm_orchestrator_config.types import LLMProvider
from llm_orchestrator_config.exceptions import (
    ConfigurationError,
    InvalidConfigurationError,
)

# Constants
DEFAULT_CONFIG_FILENAME = "llm_config.yaml"

# Initialize Loki logger
logger = LokiLogger(service_name="config-loader")

# Type alias for configuration values that can be processed
ConfigValue = Union[str, Dict[str, Any], List[Any], int, float, bool, None]


class ConfigurationLoader:
    """Loads and processes LLM configuration from YAML files with environment variable support."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: str = "development",
        connection_id: Optional[str] = None,
    ) -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Path to the configuration file. If None, uses default location.
            environment: Environment type ("production", "development", "test")
            connection_id: Connection ID (required for development/test environments)
        """
        # Load environment variables from .env file if it exists
        self._load_environment_variables()

        self.config_path = self._resolve_config_path(config_path)
        self.environment = environment
        self.connection_id = connection_id

    def _load_environment_variables(self) -> None:
        """Load environment variables from .env file if it exists."""
        try:
            # Look for .env file in the project root
            # Start from the config file's directory and go up to find project root
            current_dir = Path(__file__).parent
            project_root = current_dir

            # Go up until we find the project root (containing pyproject.toml or similar)
            while project_root.parent != project_root:
                if (project_root / "pyproject.toml").exists() or (
                    project_root / ".git"
                ).exists():
                    break
                project_root = project_root.parent

            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                logger.debug(f"Loaded environment variables from {env_file}")
            else:
                # Try loading from current directory as fallback
                load_dotenv(
                    verbose=False
                )  # This will look for .env in current directory

        except Exception as e:
            # Don't fail if .env loading fails, just log a warning
            logger.warning(f"Could not load .env file: {e}")

    def _resolve_config_path(self, config_path: Optional[str]) -> Path:
        """Resolve the configuration file path."""
        if config_path:
            return Path(config_path)

        # Default locations to search for config
        default_locations = [
            Path(DEFAULT_CONFIG_FILENAME),
            Path("config") / DEFAULT_CONFIG_FILENAME,
            Path(__file__).parent / DEFAULT_CONFIG_FILENAME,
        ]

        for location in default_locations:
            if location.exists():
                return location

        # If no config file found, use the default location in the config directory
        return Path(__file__).parent / DEFAULT_CONFIG_FILENAME

    def load_config(self) -> LLMConfiguration:
        """Load and parse the configuration file.

        Returns:
            Parsed LLM configuration.

        Raises:
            ConfigurationError: If configuration loading fails.
        """
        try:
            if not self.config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.config_path}"
                )
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)

            if not raw_config or "llm" not in raw_config:
                raise ConfigurationError("Invalid configuration: missing 'llm' section")

            # Process vault configuration and resolve secrets
            processed_config = self._resolve_vault_secrets(raw_config["llm"])

            # Parse and validate configuration
            return self._parse_configuration(processed_config)

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
        except Exception as e:
            # Already a ConfigurationError with a specific, operator-facing
            # message (e.g. "No production LLM connection configured") - keep it
            # verbatim instead of nesting another prefix in front of it.
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def _resolve_vault_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve secrets from Vault for provider configurations.

        Args:
            config: Raw configuration dictionary.

        Returns:
            Configuration with Vault secrets resolved.

        Raises:
            ConfigurationError: If vault configuration is invalid or secrets cannot be resolved
        """
        try:
            # First process any remaining environment variables (like vault config)
            config = self._process_environment_variables(config)

            # Initialize vault resolver
            resolver = self._initialize_vault_resolver(config)

            # Process provider configurations
            self._resolve_provider_secrets(config, resolver)

            return config

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to resolve vault secrets: {e}") from e

    def _initialize_vault_resolver(self, config: Dict[str, Any]) -> SecretResolver:
        """Initialize vault secret resolver from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Initialized SecretResolver or None if vault not available

        Raises:
            ConfigurationError: If vault configuration is invalid
        """
        vault_config = config.get("vault", {})
        if not vault_config.get("enabled", True):
            raise ConfigurationError("Vault is disabled in configuration")

        return SecretResolver()

    def _resolve_provider_secrets(
        self, config: Dict[str, Any], resolver: SecretResolver
    ) -> None:
        """Resolve secrets for providers using the new vault structure.

        Args:
            config: Configuration dictionary to update
            resolver: Secret resolver instance

        Raises:
            ConfigurationError: If secret resolution fails
        """
        if "providers" not in config:
            return

        # connection_id (vault_uuid) is required for all environments. A missing
        # one means no connection row exists for this environment at all - the
        # operator has not configured one yet, so say that rather than naming an
        # internal field.
        if not self.connection_id:
            logger.error(
                f"No {self.environment} LLM connection configured. "
                f"Create a {self.environment} connection so its credentials are "
                f"stored in Vault."
            )
            raise ConfigurationError(f"No {self.environment} LLM connection configured")

        try:
            providers_to_update: Dict[str, Dict[str, Any]] = {}

            # Process each provider defined in the config
            for provider_name, provider_config in config["providers"].items():
                try:
                    if not isinstance(provider_config, dict):
                        logger.warning(
                            f"Provider {provider_name} config is not a dictionary, skipping"
                        )
                        continue
                    provider_config = cast(Dict[str, Any], provider_config)

                    # Skip if provider doesn't have models defined
                    if "models" not in provider_config:
                        logger.warning(
                            f"Provider {provider_name} has no models defined, skipping"
                        )
                        continue

                    # Use connection_id (vault_uuid) directly for all environments
                    secret = resolver.get_secret_for_model(
                        provider_name,
                        self.environment,
                        "",
                        self.connection_id,
                    )
                    if secret:
                        model_name = secret.model
                        updated_config = self._merge_config_with_secrets(
                            provider_config, secret, model_name
                        )
                        providers_to_update[provider_name] = updated_config
                        logger.info(
                            f"Configured {provider_name} with model {model_name} "
                            f"(vault_uuid: {self.connection_id})"
                        )
                    else:
                        # Either nothing is stored for this provider under this
                        # connection, or what is stored failed validation - the
                        # resolver logs which one.
                        logger.warning(
                            f"No usable {provider_name} credentials for vault_uuid "
                            f"{self.connection_id} - skipping this provider"
                        )

                except Exception as e:
                    logger.error(f"Failed to process provider {provider_name}: {e}")
                    logger.debug(
                        f"Provider {provider_name} processing error details",
                        exc_info=True,
                    )
                    # Continue to next provider instead of failing completely
                    continue

            # Check if we have any providers configured. Reaching here means a
            # connection row exists but none of its provider secrets could be
            # loaded from Vault - either the cron has not written them yet or
            # what it wrote does not match the expected schema (see the
            # per-provider warnings above for which, and why).
            if not providers_to_update:
                logger.error(
                    f"No usable LLM provider for the {self.environment} connection "
                    f"(vault_uuid: {self.connection_id}). The connection exists but "
                    f"none of its credentials could be loaded from Vault - see the "
                    f"per-provider messages above."
                )
                raise ConfigurationError(
                    f"No usable LLM provider for the {self.environment} connection"
                )

            # Update the configuration with only available providers
            config["providers"] = providers_to_update

            # Update default_provider if needed
            self._update_default_provider(config)

            logger.info(
                f"Successfully configured {len(providers_to_update)} providers: {list(providers_to_update.keys())}"
            )

        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to resolve provider secrets: {e}") from e

    def _merge_config_with_secrets(
        self,
        provider_config: Dict[str, Any],
        secret: Any,  # noqa: ANN401  # Runtime type discrimination via hasattr (Union causes type errors)
        model_name: str,
    ) -> Dict[str, Any]:
        """Merge provider configuration with secrets from Vault.

        Args:
            provider_config: Provider configuration from YAML
            secret: Secret object from Vault (AzureOpenAISecret or AWSBedrockSecret)
            model_name: Model name being configured

        Returns:
            Updated provider configuration with secrets and model-specific config
        """
        # Start with the original config (provider-level settings)
        updated_config = provider_config.copy()

        # Set the active model
        updated_config["model"] = model_name

        # Get model-specific configuration from YAML
        model_config: Dict[str, Any] = {}
        if "models" in provider_config and model_name in provider_config["models"]:
            raw_model_config = provider_config["models"][model_name]
            if isinstance(raw_model_config, dict):
                model_config = cast(Dict[str, Any], raw_model_config.copy())

        # Add secret data based on provider type
        if hasattr(secret, "endpoint"):  # Azure OpenAI
            updated_config.update(
                {
                    "endpoint": secret.endpoint,
                    "api_key": secret.api_key,
                    "enabled": True,
                }
            )
            # Merge model-specific config, with secret taking precedence for deployment_name
            if model_config:
                # Add model-specific settings (max_tokens, temperature, etc.)
                updated_config.update(model_config)
                # Override deployment_name from secret if provided
                updated_config["deployment_name"] = secret.deployment_name

        elif hasattr(secret, "region"):  # AWS Bedrock
            updated_config.update(
                {
                    "region": secret.region,
                    "access_key_id": secret.access_key_id,
                    "secret_access_key": secret.secret_access_key,
                    "enabled": True,
                }
            )
            if hasattr(secret, "session_token") and secret.session_token:
                updated_config["session_token"] = secret.session_token

            # Merge model-specific config
            if model_config:
                updated_config.update(model_config)

        # Remove the models section since we're now using a single active model
        if "models" in updated_config:
            del updated_config["models"]

        return updated_config

    def _build_provider_configs(
        self, config: Dict[str, Any], available_providers: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Build configuration for available providers.

        Args:
            config: Original configuration
            available_providers: Available providers from vault

        Returns:
            Dictionary of provider configurations
        """
        providers_to_process: Dict[str, Dict[str, Any]] = {}

        for provider_name, connection in available_providers.items():
            # Check if provider is defined in config
            if provider_name in config["providers"]:
                provider_config = config["providers"][provider_name]

                # Copy the template configuration
                if isinstance(provider_config, dict):
                    providers_to_process[provider_name] = {
                        **provider_config,
                        "enabled": True,  # Force enable since it's available in vault
                    }
                    logger.info(
                        f"Using provider {provider_name} from vault connection {connection.metadata.id}"
                    )
                else:
                    logger.warning(
                        f"Invalid configuration for provider {provider_name}, skipping"
                    )
            else:
                # Provider available in vault but not in config template
                # Create a minimal config for it
                providers_to_process[provider_name] = {
                    "enabled": True,
                    "cache": True,
                    "num_retries": 3,
                }
                logger.info(
                    f"Provider {provider_name} available in vault but not in config, using minimal configuration"
                )

        return providers_to_process

    def _update_default_provider(self, config: Dict[str, Any]) -> None:
        """Update default_provider if it's not available or set automatically from vault-resolved providers.

        Args:
            config: Configuration dictionary to update
        """
        if "providers" not in config or not config["providers"]:
            return
        available_providers = config["providers"]

        # Auto-set default provider if not specified
        if "default_provider" not in config:
            new_default = next(iter(available_providers.keys()))
            logger.info(
                f"No default provider specified, auto-selected '{new_default}' "
                f"from vault-resolved providers"
            )
            config["default_provider"] = new_default
        else:
            # Check if existing default provider is available
            default_provider = config["default_provider"]
            if default_provider not in available_providers:
                # Set default to the first available provider
                new_default = next(iter(available_providers.keys()))
                logger.warning(
                    f"Default provider '{default_provider}' not available, "
                    f"using '{new_default}' instead"
                )
                config["default_provider"] = new_default

    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variable substitutions in configuration.

        This method is now only used for vault configuration processing.

        Args:
            config: Raw configuration dictionary.

        Returns:
            Configuration with environment variables substituted.
        """

        def substitute_env_vars(obj: ConfigValue) -> ConfigValue:
            if isinstance(obj, str):
                # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
                pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

                def replace_env_var(match: re.Match[str]) -> str:
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ""
                    return os.getenv(var_name, default_value)

                return re.sub(pattern, replace_env_var, obj)
            elif isinstance(obj, dict):
                result: Dict[str, ConfigValue] = {}
                for key, value in obj.items():
                    result[str(key)] = substitute_env_vars(value)
                return result
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            else:
                return obj

        result = substitute_env_vars(config)
        if isinstance(result, dict):
            return result
        else:
            # This should never happen given our input type, but provide a fallback
            raise ConfigurationError(
                "Environment variable substitution resulted in non-dictionary type"
            )

    def _parse_configuration(self, config: Dict[str, Any]) -> LLMConfiguration:
        """Parse the processed configuration into structured objects.

        Args:
            config: Processed configuration dictionary.

        Returns:
            Structured LLM configuration.

        Raises:
            InvalidConfigurationError: If configuration validation fails.
        """
        try:
            # Validate required fields
            if "providers" not in config:
                raise InvalidConfigurationError("Missing required field: providers")

            # Parse default provider - it might be auto-selected after vault resolution
            default_provider = None
            if "default_provider" in config:
                try:
                    default_provider = LLMProvider(config["default_provider"])
                except ValueError as e:
                    raise InvalidConfigurationError(
                        f"Invalid default_provider: {config['default_provider']}"
                    ) from e

            # Parse provider configurations
            providers: Dict[str, ProviderConfig] = {}

            for provider_name, provider_config in config["providers"].items():
                try:
                    provider_type = LLMProvider(provider_name)
                    providers[provider_name] = self._parse_provider_config(
                        provider_type, provider_config
                    )
                except ValueError as e:
                    raise InvalidConfigurationError(
                        f"Invalid provider name: {provider_name}"
                    ) from e

            # Auto-select default provider if not set
            if default_provider is None:
                # Find the first enabled provider
                enabled_providers = [
                    name for name, config in providers.items() if config.enabled
                ]
                if not enabled_providers:
                    raise InvalidConfigurationError("No enabled providers found")

                try:
                    default_provider = LLMProvider(enabled_providers[0])
                    logger.info(
                        f"Auto-selected default provider: {default_provider.value}"
                    )
                except ValueError as e:
                    raise InvalidConfigurationError(
                        f"Invalid auto-selected provider: {enabled_providers[0]}"
                    ) from e

            # Validate that default provider exists and is enabled
            if default_provider.value not in providers:
                raise InvalidConfigurationError(
                    f"Default provider '{default_provider.value}' not found in providers"
                )

            if not providers[default_provider.value].enabled:
                raise InvalidConfigurationError(
                    f"Default provider '{default_provider.value}' is not enabled"
                )

            # Parse vault configuration
            vault_config = None
            if "vault" in config:
                vault_config = VaultConfig(**config["vault"])

            return LLMConfiguration(
                vault=vault_config,
                default_provider=default_provider,
                providers=providers,
            )

        except Exception as e:
            if isinstance(e, InvalidConfigurationError):
                raise
            raise InvalidConfigurationError(f"Configuration parsing failed: {e}") from e

    def _parse_provider_config(
        self, provider_type: LLMProvider, config: Dict[str, Any]
    ) -> ProviderConfig:
        """Parse provider-specific configuration.

        Args:
            provider_type: Type of the provider.
            config: Provider configuration dictionary.

        Returns:
            Parsed provider configuration.
        """
        # Validate required base fields
        required_fields = ["enabled", "model"]
        for field in required_fields:
            if field not in config:
                raise InvalidConfigurationError(
                    f"Missing required field '{field}' for provider {provider_type.value}"
                )

        if provider_type == LLMProvider.AZURE_OPENAI:
            return AzureOpenAIConfig(
                enabled=config["enabled"],
                model=config["model"],
                max_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                endpoint=config.get("endpoint", ""),
                api_key=config.get("api_key", ""),
                api_version=config.get("api_version", "2024-02-15-preview"),
                deployment_name=config.get("deployment_name", ""),
            )
        elif provider_type == LLMProvider.AWS_BEDROCK:
            return AWSBedrockConfig(
                enabled=config["enabled"],
                model=config["model"],
                max_tokens=config.get("max_tokens", 4096),
                temperature=config.get("temperature", 0.7),
                region=config.get("region", ""),
                access_key_id=config.get("access_key_id", ""),
                secret_access_key=config.get("secret_access_key", ""),
                session_token=config.get("session_token"),
            )
        else:
            raise InvalidConfigurationError(
                f"Unsupported provider type: {provider_type}"
            )

    # Embedding-specific methods for vault-driven model resolution

    def resolve_embedding_model(
        self, environment: str, connection_id: Optional[str] = None
    ) -> tuple[str, str]:
        """Resolve embedding model from vault based on environment and connection_id.

        Uses the same vault_uuid as the LLM model resolver. The embedding secret
        is stored at `secret/embeddings/connections/{platform}/{vault_uuid}`.

        Args:
            environment: Environment (production, testing)
            connection_id: Vault UUID for the connection (same as LLM connection)

        Returns:
            Tuple of (provider_name, model_name) resolved from vault

        Raises:
            ConfigurationError: If no embedding models are available
        """
        # Resolve vault_uuid for production if not provided
        if not connection_id:
            from src.utils.connection_id_fetcher import get_connection_id_fetcher

            fetcher = get_connection_id_fetcher()
            connection_id = fetcher.fetch_vault_uuid_sync(environment)
            if not connection_id:
                raise ConfigurationError(
                    f"No {environment} connection found in database for embedding resolution"
                )

        # Load raw config to get vault settings
        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config: Dict[str, Any] = yaml.safe_load(file)

            if not raw_config or "llm" not in raw_config:
                raise ConfigurationError("Invalid configuration: missing 'llm' section")

            config: Dict[str, Any] = self._process_environment_variables(
                raw_config["llm"]
            )
            resolver: SecretResolver = self._initialize_vault_resolver(config)

            # Get available providers from config
            providers: List[str] = ["azure_openai", "aws_bedrock"]

            # Use vault_uuid directly to find embedding secret (same UUID as LLM)
            for provider in providers:
                try:
                    secret: Optional[Union[AzureOpenAISecret, AWSBedrockSecret]] = (
                        resolver.get_embedding_secret_for_model(
                            provider, environment, "", connection_id
                        )
                    )
                    if secret and self._is_embedding_model(secret.model):
                        logger.info(
                            f"Resolved embedding model: {provider}/{secret.model} "
                            f"(vault_uuid: {connection_id})"
                        )
                        return provider, secret.model
                except Exception as e:
                    logger.debug(
                        f"Provider {provider} not available for embeddings "
                        f"with vault_uuid {connection_id}: {e}"
                    )
                    continue

            raise ConfigurationError(
                f"No embedding models available for {environment} "
                f"with vault_uuid {connection_id}"
            )

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Failed to resolve embedding model: {e}") from e

    def get_embedding_provider_config(
        self,
        provider: str,
        model: str,
        environment: str,
        connection_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get embedding provider configuration with vault secrets merged.

        Args:
            provider: Provider name (azure_openai, aws_bedrock)
            model: Embedding model name
            environment: Environment (production, development, test)
            connection_id: Optional connection ID for dev/test environments

        Returns:
            Complete provider configuration with secrets

        Raises:
            ConfigurationError: If configuration cannot be loaded or secrets not found
        """
        try:
            # Auto-resolve vault_uuid from DB if not provided
            if not connection_id:
                from src.utils.connection_id_fetcher import get_connection_id_fetcher

                fetcher = get_connection_id_fetcher()
                connection_id = fetcher.fetch_vault_uuid_sync(environment)
                if not connection_id:
                    raise ConfigurationError(
                        f"No {environment} connection found in database for embedding config"
                    )

            # Load raw config
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config: Dict[str, Any] = yaml.safe_load(file)

            if not raw_config or "llm" not in raw_config:
                raise ConfigurationError("Invalid configuration: missing 'llm' section")

            config: Dict[str, Any] = self._process_environment_variables(
                raw_config["llm"]
            )
            resolver: SecretResolver = self._initialize_vault_resolver(config)

            # Get base provider config from llm_config.yaml
            base_config: Dict[str, Any] = config.get("providers", {}).get(provider, {})
            if not base_config:
                raise ConfigurationError(
                    f"Provider {provider} not found in configuration"
                )

            # Get secrets from embeddings vault path
            secret: Optional[Union[AzureOpenAISecret, AWSBedrockSecret]] = (
                resolver.get_embedding_secret_for_model(
                    provider, environment, model, connection_id
                )
            )

            if not secret:
                raise ConfigurationError(
                    f"No embedding secrets found for {provider}/{model} in {environment}"
                )

            # Merge configuration with secrets using existing method
            merged_config: Dict[str, Any] = self._merge_config_with_secrets(
                base_config, secret, model
            )

            logger.debug(f"Successfully loaded embedding config for {provider}/{model}")
            return merged_config

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to get embedding provider config: {e}"
            ) from e

    def get_available_embedding_models(self, environment: str) -> Dict[str, List[str]]:
        """Get available embedding models across all providers.

        Args:
            environment: Environment (production, development, test)

        Returns:
            Dictionary mapping provider names to available embedding models

        Raises:
            ConfigurationError: If configuration cannot be loaded
        """
        try:
            # Load raw config
            with open(self.config_path, "r", encoding="utf-8") as file:
                raw_config: Dict[str, Any] = yaml.safe_load(file)

            if not raw_config or "llm" not in raw_config:
                raise ConfigurationError("Invalid configuration: missing 'llm' section")

            config: Dict[str, Any] = self._process_environment_variables(
                raw_config["llm"]
            )
            resolver: SecretResolver = self._initialize_vault_resolver(config)

            available_models: Dict[str, List[str]] = {}
            providers: List[str] = ["azure_openai", "aws_bedrock"]

            for provider in providers:
                try:
                    models: List[str] = resolver.list_available_embedding_models(
                        provider, environment
                    )
                    embedding_models: List[str] = [
                        m for m in models if self._is_embedding_model(m)
                    ]
                    if embedding_models:
                        available_models[provider] = embedding_models
                except Exception as e:
                    logger.debug(f"Provider {provider} not available: {e}")
                    continue

            return available_models

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}") from e
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(
                f"Failed to get available embedding models: {e}"
            ) from e

    def _is_embedding_model(self, model_name: str) -> bool:
        """Detect if model is an embedding model based on name patterns.

        Args:
            model_name: Model name to check

        Returns:
            True if model appears to be an embedding model
        """
        embedding_patterns: List[str] = [
            "embedding",
            "embed",
            "text-embedding",
            "titan-embed",
            "e5-",
            "instructor-",
            "sentence-transformer",
        ]

        model_lower: str = model_name.lower()
        return any(pattern in model_lower for pattern in embedding_patterns)

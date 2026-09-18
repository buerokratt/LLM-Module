"""LLM Manager - Main entry point for the LLM Config Module."""

from typing import Any, Dict, Generator, List, Optional
from contextlib import contextmanager
import threading

import dspy

from llm_orchestrator_config.llm_factory import LLMFactory
from llm_orchestrator_config.config.loader import ConfigurationLoader
from llm_orchestrator_config.config.schema import LLMConfiguration
from llm_orchestrator_config.providers.base import BaseLLMProvider
from llm_orchestrator_config.types import LLMProvider
from llm_orchestrator_config.exceptions import ConfigurationError


class LLMManager:
    """Singleton manager for LLM providers.

    This class provides a centralized way to manage and access LLM providers
    throughout the application. It follows the Singleton pattern to ensure
    consistent configuration and provider instances.
    """

    _instance: Optional["LLMManager"] = None
    _instance_lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    _configured: bool = False
    _config_lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        config_path: Optional[str] = None,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> "LLMManager":
        """Create or return the singleton instance.

        Args:
            config_path: Optional path to configuration file.
            environment: Environment type ("production", "development", "test")
            connection_id: Connection ID (required for development/test environments)

        Returns:
            LLMManager singleton instance.
        """
        # Thread-safe singleton creation
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> None:
        """Initialize the LLM Manager.

        Args:
            config_path: Optional path to configuration file.
            environment: Environment type ("production", "development", "test")
            connection_id: Connection ID (required for development/test environments)
        """
        if not self._initialized:
            self._environment = environment
            self._connection_id = connection_id
            self._config_loader = ConfigurationLoader(
                config_path, environment, connection_id
            )
            self._config: Optional[LLMConfiguration] = None
            self._providers: Dict[LLMProvider, BaseLLMProvider] = {}
            self._default_provider: Optional[BaseLLMProvider] = None
            self._load_configuration()
            self._initialize_providers()
            LLMManager._initialized = True

    @property
    def connection_id(self) -> Optional[str]:
        """Return the connection ID (vault_uuid) used by this manager."""
        return self._connection_id

    def _load_configuration(self) -> None:
        """Load configuration from file.

        Raises:
            ConfigurationError: If configuration loading fails.
        """
        try:
            self._config = self._config_loader.load_config()
        except ConfigurationError:
            # The loader's message already identifies the environment and what
            # the operator needs to do; re-wrapping only buries it.
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to load LLM configuration: {e}") from e

    def _initialize_providers(self) -> None:
        """Initialize all enabled providers.

        Raises:
            ConfigurationError: If provider initialization fails.
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded")

        enabled_providers: List[str] = []

        # Initialize all enabled providers
        for provider_name, provider_config in self._config.providers.items():
            if provider_config.enabled:
                try:
                    provider_type = LLMProvider(provider_name)
                    provider = LLMFactory.create_provider(
                        provider_type, provider_config.to_dict()
                    )
                    self._providers[provider_type] = provider
                    enabled_providers.append(provider_name)
                except Exception as e:
                    raise ConfigurationError(
                        f"Failed to initialize provider '{provider_name}': {e}"
                    ) from e

        if not enabled_providers:
            raise ConfigurationError("No providers are enabled")

        # Set default provider
        default_provider_type = self._config.default_provider
        if default_provider_type not in self._providers:
            raise ConfigurationError(
                f"Default provider '{default_provider_type.value}' is not enabled or failed to initialize"
            )

        self._default_provider = self._providers[default_provider_type]

    def get_llm(self, provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get LLM provider instance.

        Args:
            provider: Optional specific provider to get. If None, returns default provider.

        Returns:
            LLM provider instance.

        Raises:
            ConfigurationError: If the requested provider is not available.
        """
        if provider is None:
            if self._default_provider is None:
                raise ConfigurationError("No default provider configured")
            return self._default_provider

        if provider not in self._providers:
            available_providers = ", ".join([p.value for p in self._providers.keys()])
            raise ConfigurationError(
                f"Provider '{provider.value}' is not available. "
                f"Available providers: {available_providers}"
            )

        return self._providers[provider]

    def get_dspy_client(self, provider: Optional[LLMProvider] = None) -> dspy.LM:
        """Get DSPY-compatible client.

        Args:
            provider: Optional specific provider to get client for.

        Returns:
            DSPY LM client instance.
        """
        llm_provider = self.get_llm(provider)
        return llm_provider.get_dspy_client()

    def configure_dspy(self, provider: Optional[LLMProvider] = None) -> None:
        """Configure DSPY with the specified or default provider.

        Args:
            provider: Optional specific provider to configure DSPY with.
        """
        dspy_client = self.get_dspy_client(provider)
        dspy.configure(lm=dspy_client, track_usage=True)

    def ensure_global_config(self, provider: Optional[LLMProvider] = None) -> None:
        """Configure DSPy exactly once per process (thread-safe)."""
        # Double-checked locking pattern for performance
        if not self._configured:
            with self._config_lock:
                # Re-check inside the lock to prevent race condition
                if not self._configured:
                    dspy_client = self.get_dspy_client(provider)
                    dspy.configure(lm=dspy_client, track_usage=True)
                    self._configured = True

    @contextmanager
    def use_task_local(
        self, provider: Optional[LLMProvider] = None
    ) -> Generator[None, None, None]:
        """Use a task/thread-local DSPy LM without reconfiguring globally."""
        lm = self.get_dspy_client(provider)
        with dspy.context(lm=lm):
            yield

    def get_available_providers(self) -> Dict[LLMProvider, str]:
        """Get information about available providers.

        Returns:
            Dictionary mapping provider types to their names.
        """
        return {
            provider_type: provider.provider_name
            for provider_type, provider in self._providers.items()
        }

    def get_provider_info(
        self, provider: Optional[LLMProvider] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a provider.

        Args:
            provider: Optional specific provider to get info for.

        Returns:
            Dictionary containing provider information.
        """
        llm_provider = self.get_llm(provider)
        return llm_provider.get_model_info()

    def is_provider_available(self, provider: LLMProvider) -> bool:
        """Check if a provider is available.

        Args:
            provider: Provider type to check.

        Returns:
            True if the provider is available, False otherwise.
        """
        return provider in self._providers

    def reload_configuration(self, config_path: Optional[str] = None) -> None:
        """Reload configuration and reinitialize providers.

        Args:
            config_path: Optional new configuration file path.

        Raises:
            ConfigurationError: If reloading fails.
        """
        if config_path:
            self._config_loader = ConfigurationLoader(config_path)

        # Clear existing providers
        self._providers.clear()
        self._default_provider = None

        # Reload configuration and providers
        self._load_configuration()
        self._initialize_providers()

    def get_configuration(self) -> Optional[LLMConfiguration]:
        """Get the current configuration.

        Returns:
            Current LLM configuration or None if not loaded
        """
        return self._config

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance.

        This is primarily useful for testing purposes.
        """
        with cls._instance_lock:
            cls._instance = None
            cls._initialized = False
            cls._configured = False

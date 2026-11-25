"""LLM Manager - Main entry point for the LLM Config Module."""

from typing import Any, Dict, Generator, List, Optional, Tuple
from contextlib import ExitStack, contextmanager, suppress
from contextvars import ContextVar
import threading

import dspy

from llm_orchestrator_config.llm_factory import LLMFactory
from llm_orchestrator_config.config.loader import ConfigurationLoader
from llm_orchestrator_config.config.schema import LLMConfiguration
from llm_orchestrator_config.providers.base import BaseLLMProvider
from llm_orchestrator_config.types import LLMProvider
from llm_orchestrator_config.exceptions import ConfigurationError

# The DSPy LM of the request being processed. use_task_local() prefers it over
# the manager's own LM, so components created once (e.g. the tool classifier)
# still use the connection of the current request.
_request_lm: ContextVar[Optional[dspy.LM]] = ContextVar("request_lm", default=None)

# Cleanup for the current request scope; see request_llm_scope().
_request_scope: ContextVar[Optional[ExitStack]] = ContextVar(
    "request_llm_scope", default=None
)


@contextmanager
def request_llm_scope() -> Generator[None, None, None]:
    """Scope one orchestration request.

    Inside the scope, LLMManager.activate_for_request() makes that manager's LM
    the model for the rest of the request: dspy.settings.lm, use_task_local()
    and everything reading them (guardrails, classifier, RAG) use it. The
    request's model is reset when the scope ends.
    """
    stack = ExitStack()
    token = _request_scope.set(stack)
    try:
        yield
    finally:
        # A ValueError means the scope is being closed from another context,
        # e.g. a streaming generator finalized elsewhere; nothing to restore.
        with suppress(ValueError):
            stack.close()
        with suppress(ValueError):
            _request_scope.reset(token)


def _resolve_connection_id(environment: str) -> Optional[str]:
    """Resolve the vault_uuid of the environment's current connection from the DB."""
    try:
        from src.utils.connection_id_fetcher import get_connection_id_fetcher
    except ImportError:
        return None
    return get_connection_id_fetcher().fetch_vault_uuid_sync(environment)


class LLMManager:
    """Manager for the LLM providers of one LLM connection.

    One instance is created per connection (environment + vault_uuid) and
    reused, so each connection's model and credentials are loaded once.
    ``LLMManager(environment=..., connection_id=...)`` returns that
    connection's instance. Without a connection_id, the environment's current
    connection is resolved from the database.
    """

    _instances: Dict[Tuple[str, str], "LLMManager"] = {}
    # Re-entrant: loading a connection must not deadlock if it creates a manager.
    _instance_lock = threading.RLock()
    _configured: bool = False
    _config_lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        config_path: Optional[str] = None,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> "LLMManager":
        """Return the instance for the connection, creating it on first use.

        Args:
            config_path: Optional path to configuration file.
            environment: Environment type ("production", "testing", ...)
            connection_id: Connection vault_uuid. Resolved from the database
                when omitted.

        Returns:
            The LLMManager of that connection.
        """
        resolved_id = connection_id or _resolve_connection_id(environment)
        key = (environment, resolved_id or "")
        with cls._instance_lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = super().__new__(cls)
                # Only cached once loaded, so a failed load is retried next time.
                instance._setup(config_path, environment, resolved_id)
                cls._instances[key] = instance
        return instance

    def __init__(
        self,
        config_path: Optional[str] = None,
        environment: str = "production",
        connection_id: Optional[str] = None,
    ) -> None:
        """No-op: each instance is set up once, in __new__."""

    def _setup(
        self,
        config_path: Optional[str],
        environment: str,
        connection_id: Optional[str],
    ) -> None:
        """Load the connection's configuration and initialize its providers."""
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
        """Set the process-wide default DSPy LM once (thread-safe).

        The default only applies outside requests and use_task_local(); requests
        use their own connection via activate_for_request(). It is set only once
        per process (DSPy rejects dspy.configure() from other async tasks), so it
        is kept after clear_instances().
        """
        # Double-checked locking pattern for performance
        if not LLMManager._configured:
            with LLMManager._config_lock:
                # Re-check inside the lock to prevent race condition
                if not LLMManager._configured:
                    dspy_client = self.get_dspy_client(provider)
                    dspy.configure(lm=dspy_client, track_usage=True)
                    LLMManager._configured = True

    def activate_for_request(self, provider: Optional[LLMProvider] = None) -> None:
        """Use this manager's LM for the rest of the current request.

        Only has an effect inside request_llm_scope(); the model is reset when
        the scope ends.
        """
        stack = _request_scope.get()
        if stack is None:
            return
        lm = self.get_dspy_client(provider)
        if _request_lm.get() is lm:
            return
        stack.enter_context(dspy.context(lm=lm))
        token = _request_lm.set(lm)
        stack.callback(_request_lm.reset, token)

    @contextmanager
    def use_task_local(
        self, provider: Optional[LLMProvider] = None
    ) -> Generator[None, None, None]:
        """Use a task/thread-local DSPy LM without reconfiguring globally.

        Inside a request, the request's LM is used, so components holding a
        manager from an earlier request still call the current connection.
        """
        lm = _request_lm.get() or self.get_dspy_client(provider)
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
    def clear_instances(cls) -> None:
        """Drop all cached connections, so the next use reloads them.

        Called by /cache/clear when a connection is added, changed or deleted.
        Requests already running keep the manager they hold.

        The process-wide DSPy default is not set again: DSPy only allows
        dspy.configure() from the task that called it first, and requests use
        their own connection via activate_for_request() anyway.
        """
        with cls._instance_lock:
            cls._instances.clear()

    @classmethod
    def reset_instance(cls) -> None:
        """Drop all cached connections (kept for existing callers and tests)."""
        cls.clear_instances()

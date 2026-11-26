"""Vault Agent client using hvac library."""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, cast
from src.constants_loader import get_constant, get_optional_constant
from src.loki_logger import LokiLogger
import hvac
from hvac.exceptions import InvalidPath, Forbidden

from llm_orchestrator_config.vault.exceptions import (
    VaultConnectionError,
    VaultSecretError,
    VaultTokenError,
)

# Initialize Loki logger
logger = LokiLogger(service_name="vault-client")

# Global singleton instance
_vault_client_instance: Optional["VaultAgentClient"] = None
_vault_client_lock = threading.Lock()


def _resolve_path_prefix() -> str:
    """Resolve the KV path prefix, normalized without surrounding slashes.

    constants.ini is the source of truth so that Python and the CronManager
    shell scripts read the same value; VAULT_PATH_PREFIX in the environment
    overrides it for ad-hoc runs and tests. Empty means no prefix.
    """
    raw = os.getenv("VAULT_PATH_PREFIX")
    if raw is None:
        raw = get_optional_constant("VAULT_PATH_PREFIX")
    return raw.strip().strip("/")


def _resolve_vault_endpoint() -> str:
    """Resolve the HTTP endpoint for KV reads, with no built-in default.

    Normally the Vault Agent (VAULT_AGENT_LLM_URL in constants.ini), which proxies
    to Vault and injects the token. VAULT_ADDR overrides it with the address of
    Vault itself, which is how the test stacks bypass the agent.
    """
    return os.getenv("VAULT_ADDR", "").strip() or get_constant("VAULT_AGENT_LLM_URL")


def _resolve_token_file() -> Optional[Path]:
    """Path of the Vault token file, or None when a Vault Agent supplies it.

    Both real deployments talk to a Vault Agent -- vault-agent-llm:8201 under
    compose, the localhost:8201 sidecar in Kubernetes -- and the agent injects
    its own token (api_proxy.use_auto_auth_token), so this client sends none.
    Talking straight to Vault is the exception (the integration-test stack) and
    it says so explicitly by setting VAULT_TOKEN_FILE.
    """
    path = os.getenv("VAULT_TOKEN_FILE", "").strip()
    return Path(path) if path else None


def get_vault_client(
    vault_endpoint: Optional[str] = None,
    token_path: Optional[str] = None,
    mount_point: str = "secret",
    timeout: int = 10,
) -> "VaultAgentClient":
    """Get or create singleton VaultAgentClient instance.

    This ensures only one Vault client is created per process,
    avoiding redundant token loading and health checks (~35ms overhead per instantiation).

    Args:
        vault_endpoint: Agent or Vault address (defaults to VAULT_AGENT_LLM_URL
            from constants.ini, or VAULT_ADDR when set)
        token_path: Token file to authenticate with (defaults to VAULT_TOKEN_FILE;
            None means the Vault Agent supplies the token)
        mount_point: KV v2 mount point
        timeout: Request timeout in seconds

    Returns:
        Singleton VaultAgentClient instance
    """
    global _vault_client_instance

    if _vault_client_instance is None:
        with _vault_client_lock:
            if _vault_client_instance is None:
                _vault_client_instance = VaultAgentClient(
                    vault_endpoint=vault_endpoint,
                    token_path=token_path,
                    mount_point=mount_point,
                    timeout=timeout,
                )
                logger.info("Created singleton VaultAgentClient instance")

    return _vault_client_instance


class VaultAgentClient:
    """HashiCorp Vault client using Vault Agent token."""

    def __init__(
        self,
        vault_endpoint: Optional[str] = None,
        token_path: Optional[str] = None,
        mount_point: str = "secret",
        timeout: int = 10,
    ) -> None:
        """Initialize Vault Agent client.

        Args:
            vault_endpoint: Agent or Vault address (defaults to
                VAULT_AGENT_LLM_URL from constants.ini, or VAULT_ADDR when set)
            token_path: Token file to authenticate with (defaults to
                VAULT_TOKEN_FILE; None means the Vault Agent supplies the token)
            mount_point: KV v2 mount point
            timeout: Request timeout in seconds
        """
        self.vault_endpoint = vault_endpoint or _resolve_vault_endpoint()
        self.token_path = Path(token_path) if token_path else _resolve_token_file()
        self.mount_point = mount_point
        self.timeout = timeout
        self.use_token_file = self.token_path is not None

        # Applied here, at the single chokepoint for every Vault read. Callers
        # (SecretResolver, the Langfuse lookup) pass unprefixed logical paths
        # and must not prefix them again.
        self.path_prefix = _resolve_path_prefix()
        if self.path_prefix:
            logger.info(f"Vault path prefix active: {self.path_prefix}/")

        # Initialize hvac client
        self.client = hvac.Client(
            url=self.vault_endpoint,
            timeout=timeout,
        )

        if self.use_token_file:
            self._load_token()

        logger.info(f"Vault Agent client initialized: {self.vault_endpoint}")

    def _load_token(self) -> None:
        """Load the token from the configured token file.

        No-op when no token file is configured: the Vault Agent is then the one
        authenticating, and there is nothing on disk to read.

        Raises:
            VaultTokenError: If token file is missing or unreadable
        """
        if self.token_path is None:
            return

        try:
            if not self.token_path.exists():
                raise VaultTokenError(
                    f"Vault Agent token file not found: {self.token_path}"
                )

            with open(self.token_path, "r") as f:
                token = f.read().strip()

            if not token:
                raise VaultTokenError("Vault Agent token file is empty")

            # Log token info for debugging (first and last 4 chars only)
            token_preview = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
            logger.debug(f"Loaded token: {token_preview} (length: {len(token)})")

            self.client.token = token
            logger.debug("Vault Agent token loaded successfully")

        except (OSError, IOError) as e:
            raise VaultTokenError(f"Failed to read Vault Agent token: {e}") from e

    def is_authenticated(self) -> bool:
        """Check if client is authenticated with Vault.

        Returns:
            True if authenticated, False otherwise
        """
        try:
            # Agent mode: nothing to verify here, the agent attaches the token.
            if not self.use_token_file:
                return self.is_vault_available()

            # Check token is available
            if not hasattr(self.client, "token") or not self.client.token:
                logger.debug("No token set on client")
                return False

            # Test authentication with a simple lookup_self call
            result = self.client.is_authenticated()
            logger.debug(f"Vault authentication result: {result}")

            # Additional debug - try to get token info
            try:
                token_info = cast(Dict[str, Any], self.client.lookup_token())
                logger.debug(
                    f"Token info: policies={token_info.get('data', {}).get('policies', [])}"
                )
            except Exception as token_e:
                logger.debug(
                    f"Could not get token info (this might be normal): {token_e}"
                )

            return result

        except Exception as e:
            logger.warning(f"Vault authentication check failed: {e}")
            return False

    def is_vault_available(self) -> bool:
        """Check if Vault is available and accessible.

        Returns:
            True if Vault is available, False otherwise
        """
        try:
            response = self.client.sys.read_health_status()

            # For Vault health endpoint, we primarily check the HTTP status code
            if hasattr(response, "status_code"):
                return response.status_code == 200
            else:
                # Fallback for non-Response objects (direct dict)
                if isinstance(response, dict):
                    is_available = response.get(
                        "initialized", False
                    ) and not response.get("sealed", True)
                    logger.debug(f"Vault availability check from dict: {is_available}")
                    return is_available
                else:
                    logger.warning(f"Unexpected response type: {type(response)}")
                    return False

        except Exception as e:
            logger.warning(f"Vault not available: {e}")
            return False

    def _apply_prefix(self, path: str) -> str:
        """Prepend the configured KV path prefix to a logical secret path.

        Mount-relative in, mount-relative out: "llm/connections/azure_openai/x"
        becomes "llm-module/llm/connections/azure_openai/x". hvac still adds the
        mount and the data/ or metadata/ segment on top.
        """
        clean = path.strip("/")
        if not self.path_prefix:
            return clean
        return f"{self.path_prefix}/{clean}"

    def _require_vault(self) -> None:
        """Guard every read: Vault reachable, and authenticated if we hold a token.

        Raises:
            VaultConnectionError: If Vault is unreachable or authentication failed
        """
        if not self.is_vault_available():
            raise VaultConnectionError("Vault is not available")

        if not self.is_authenticated():
            self._load_token()  # the agent may have rotated it on disk
            if not self.is_authenticated():
                raise VaultConnectionError("Vault authentication failed")

    def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault KV v2 store.

        Args:
            path: Mount-relative logical path WITHOUT the configured prefix,
                e.g. "llm/connections/azure_openai/<vault_uuid>". The prefix is
                applied internally.

        Returns:
            Secret data or None if not found

        Raises:
            VaultConnectionError: If Vault is not available
            VaultSecretError: If secret retrieval fails
        """
        self._require_vault()
        full_path = self._apply_prefix(path)

        try:
            # Use KV v2 API
            response = self.client.secrets.kv.v2.read_secret_version(
                path=full_path,
                mount_point=self.mount_point,
            )

            if response and "data" in response:
                secret_data = response["data"]["data"]
                logger.debug(f"Successfully retrieved secret from path: {full_path}")
                return secret_data
            else:
                logger.debug(f"Secret not found at path: {full_path}")
                return None

        except InvalidPath:
            logger.debug(f"Secret not found at path: {full_path}")
            return None
        except Forbidden as e:
            raise VaultSecretError(
                f"Access denied to secret path {full_path}: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Error retrieving secret from path {full_path}: {e}")
            raise VaultSecretError(f"Failed to retrieve secret: {e}") from e

    def list_secrets(self, path: str) -> Optional[list[str]]:
        """List secrets at the given path.

        Args:
            path: Directory path to list

        Returns:
            List of secret names or None if path doesn't exist

        Raises:
            VaultConnectionError: If Vault is not available
            VaultSecretError: If listing fails
        """
        self._require_vault()
        full_path = self._apply_prefix(path)

        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=full_path,
                mount_point=self.mount_point,
            )

            if response and "data" in response:
                keys = response["data"].get("keys", [])
                logger.debug(f"Listed {len(keys)} secrets at path: {full_path}")
                return keys
            else:
                logger.debug(f"No secrets found at path: {full_path}")
                return None

        except InvalidPath:
            logger.debug(f"Path not found: {full_path}")
            return None
        except Exception as e:
            logger.error(f"Error listing secrets at path {full_path}: {e}")
            raise VaultSecretError(f"Failed to list secrets: {e}") from e

    def refresh_token(self) -> bool:
        """Re-read the token file, if one is configured.

        Returns:
            True if the client is authenticated afterwards
        """
        try:
            self._load_token()
            return self.is_authenticated()
        except Exception as e:
            logger.error(f"Failed to refresh Vault token: {e}")
            return False

"""Vault Agent client using hvac library."""

import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any, cast
from src.loki_logger import LokiLogger
import hvac
from hvac.exceptions import InvalidPath, Forbidden

from llm_orchestrator_config.vault.exceptions import (
    VaultConnectionError,
    VaultSecretError,
    VaultTokenError,
)

from llm_orchestrator_config.vault.central_paths import settings, physical_path

# Initialize Loki logger
logger = LokiLogger(service_name="vault-client")

# Global singleton instance
_vault_client_instance: Optional["VaultAgentClient"] = None
_vault_client_lock = threading.Lock()


def get_vault_client(
    vault_url: Optional[str] = None,
    token_path: str = "/agent/llm-token/token",
    mount_point: Optional[str] = None,
    timeout: int = 10,
    use_token_file: bool = True,
) -> "VaultAgentClient":
    """Get or create singleton VaultAgentClient instance.

    This ensures only one Vault client is created per process,
    avoiding redundant token loading and health checks (~35ms overhead per instantiation).

    Args:
        vault_url: Vault server URL (defaults to VAULT_ADDR env var)
        token_path: Path to Vault Agent token file (only used if use_token_file=True)
        mount_point: KV v2 mount point
        timeout: Request timeout in seconds
        use_token_file: Whether to read token from file (False when using vault agent proxy)

    Returns:
        Singleton VaultAgentClient instance
    """
    global _vault_client_instance

    if _vault_client_instance is None:
        with _vault_client_lock:
            if _vault_client_instance is None:
                _vault_client_instance = VaultAgentClient(
                    vault_url=vault_url,
                    token_path=token_path,
                    mount_point=mount_point,
                    timeout=timeout,
                    use_token_file=use_token_file,
                )
                logger.info("Created singleton VaultAgentClient instance")

    return _vault_client_instance


class VaultAgentClient:
    """HashiCorp Vault client using Vault Agent token."""

    def __init__(
        self,
        vault_url: Optional[str] = None,
        token_path: str = "/agent/llm-token/token",
        mount_point: Optional[str] = None,
        timeout: int = 10,
        use_token_file: bool = True,
    ) -> None:
        """Initialize Vault Agent client.

        Args:
            vault_url: Vault server URL (defaults to VAULT_ADDR env var)
            token_path: Path to Vault Agent token file (only used if use_token_file=True)
            mount_point: KV v2 mount point
            timeout: Request timeout in seconds
            use_token_file: Whether to read token from file (False when using vault agent proxy)
        """
        self.vault_url = vault_url or os.getenv("VAULT_ADDR", "http://vault:8200")
        self.token_path = Path(token_path)
        self.path_prefix, configured_mount = settings()
        self.mount_point = mount_point or configured_mount
        self.timeout = timeout
        self.use_token_file = use_token_file

        # Explicit proxy mode also covers localhost and every singleton caller.
        token_setting = os.getenv("VAULT_USE_TOKEN_FILE")
        if token_setting is not None:
            if token_setting not in ("true", "false"):
                raise ValueError("VAULT_USE_TOKEN_FILE must be true or false")
            self.use_token_file = token_setting == "true"

        # Initialize hvac client
        self.client = hvac.Client(
            url=self.vault_url,
            timeout=timeout,
            verify=os.getenv("VAULT_CACERT") or True,
        )
        self.client.adapter.session.trust_env = False

        # Load token from file only if not using proxy
        if self.use_token_file:
            self._load_token()
        else:
            logger.info("Using vault agent proxy - no token file required")

        logger.info(f"Vault Agent client initialized: {self.vault_url}")

    def _load_token(self) -> None:
        """Load token from Vault Agent token file.

        Raises:
            VaultTokenError: If token file is missing or unreadable
        """
        try:
            if not self.token_path.exists():
                raise VaultTokenError(
                    f"Vault Agent token file not found: {self.token_path}"
                )

            with open(self.token_path, "r") as f:
                token = f.read().strip()

            if not token:
                raise VaultTokenError("Vault Agent token file is empty")

            self.client.token = token
            logger.debug("Vault Agent token loaded successfully")

        except (OSError, IOError) as e:
            raise VaultTokenError(f"Failed to read Vault Agent token: {type(e).__name__}") from e

    def is_authenticated(self) -> bool:
        """Check if client is authenticated with Vault.

        Returns:
            True if authenticated, False otherwise
        """
        try:
            # If using proxy mode, skip token checks
            if not self.use_token_file:
                # Just verify vault is accessible (no token needed with proxy)
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
                    f"Could not get token info (this might be normal): {type(token_e).__name__}"
                )

            return result

        except Exception as e:
            logger.warning(f"Vault authentication check failed: {type(e).__name__}")
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
            logger.warning(f"Vault not available: {type(e).__name__}")
            return False

    def get_secret(self, path: str) -> Optional[Dict[str, Any]]:
        """Retrieve secret from Vault KV v2 store.

        Args:
            path: Secret path (e.g., "llm/connections/azure_openai/production/gpt-4o")

        Returns:
            Secret data or None if not found

        Raises:
            VaultConnectionError: If Vault is not available
            VaultSecretError: If secret retrieval fails
        """
        if not self.is_vault_available():
            raise VaultConnectionError("Vault is not available")

        if not self.is_authenticated():
            # Try to reload token
            self._load_token()
            if not self.is_authenticated():
                raise VaultConnectionError("Vault authentication failed")

        try:
            # Use KV v2 API
            response = self.client.secrets.kv.v2.read_secret_version(
                path=physical_path(path, self.path_prefix),
                mount_point=self.mount_point,
            )

            if response and "data" in response:
                secret_data = response["data"]["data"]
                logger.debug(f"Successfully retrieved secret from path: {path}")
                return secret_data
            else:
                logger.debug(f"Secret not found at path: {path}")
                return None

        except InvalidPath:
            logger.debug(f"Secret not found at path: {path}")
            return None
        except Forbidden as e:
            raise VaultSecretError(f"Access denied to secret path {path}: {type(e).__name__}") from e
        except Exception as e:
            logger.error(f"Error retrieving secret from path {path}: {type(e).__name__}")
            raise VaultSecretError(f"Failed to retrieve secret: {type(e).__name__}") from e

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
        if not self.is_vault_available():
            raise VaultConnectionError("Vault is not available")

        if not self.is_authenticated():
            self._load_token()
            if not self.is_authenticated():
                raise VaultConnectionError("Vault authentication failed")

        try:
            response = self.client.secrets.kv.v2.list_secrets(
                path=physical_path(path, self.path_prefix),
                mount_point=self.mount_point,
            )

            if response and "data" in response:
                keys = response["data"].get("keys", [])
                logger.debug(f"Listed {len(keys)} secrets at path: {path}")
                return keys
            else:
                logger.debug(f"No secrets found at path: {path}")
                return None

        except InvalidPath:
            logger.debug(f"Path not found: {path}")
            return None
        except Exception as e:
            logger.error(f"Error listing secrets at path {path}: {type(e).__name__}")
            raise VaultSecretError(f"Failed to list secrets: {type(e).__name__}") from e

    def refresh_token(self) -> bool:
        """Refresh token from Vault Agent.

        Returns:
            True if token was refreshed successfully
        """
        try:
            if self.use_token_file:
                self._load_token()
            return self.is_authenticated()
        except Exception as e:
            logger.error(f"Failed to refresh Vault token: {type(e).__name__}")
            return False

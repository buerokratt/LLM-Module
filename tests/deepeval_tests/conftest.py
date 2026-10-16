import os
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Generator

import pytest
import hvac
import requests
from loguru import logger
from testcontainers.compose import DockerCompose  # type: ignore
from azure.storage.blob import BlobServiceClient


# ===================== Azure Blob Storage Helper =====================


def download_embeddings_from_azure(
    connection_string: str, container_name: str, blob_name: str, local_path: Path
) -> None:
    """
    Download pre-computed embeddings from Azure Blob Storage.

    Args:
        connection_string: Azure Storage connection string
        container_name: Name of the blob container
        blob_name: Name of the blob to download
        local_path: Local path to save the downloaded file
    """
    logger.info("Downloading embeddings from Azure Blob Storage...")
    logger.info(f"  Container: {container_name}")
    logger.info(f"  Blob: {blob_name}")
    logger.info(f"  Local path: {local_path}")

    try:
        # Create BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )

        # Get blob client
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the blob
        with open(local_path, "wb") as download_file:
            download_stream = blob_client.download_blob()
            download_file.write(download_stream.readall())

        file_size_kb = local_path.stat().st_size / 1024
        logger.info(f"✓ Downloaded embeddings successfully ({file_size_kb:.2f} KB)")

    except Exception as e:
        logger.error(f"Failed to download embeddings from Azure: {e}")
        raise


# ===================== VaultAgentClient =====================


class VaultAgentClient:
    """Client for interacting with Vault using a token written by Vault Agent"""

    def __init__(
        self,
        vault_url: str,
        token_path: Path = Path("test-vault/agent-out/token"),
        mount_point: str = "secret",
        timeout: int = 10,
    ):
        self.vault_url = vault_url
        self.token_path = token_path
        self.mount_point = mount_point

        self.client = hvac.Client(url=self.vault_url, timeout=timeout)
        self._load_token()

    def _load_token(self) -> None:
        """Load token from file written by Vault Agent"""
        if not self.token_path.exists():
            raise FileNotFoundError(f"Vault token file missing: {self.token_path}")
        token = self.token_path.read_text().strip()
        if not token:
            raise ValueError("Vault token file is empty")
        self.client.token = token

    def is_authenticated(self) -> bool:
        """Check if the current token is valid"""
        try:
            return self.client.is_authenticated()
        except Exception as e:
            logger.warning(f"Vault token is not valid: {e}")
            return False

    def is_vault_available(self) -> bool:
        """Check if Vault is initialized and unsealed"""
        try:
            status = self.client.sys.read_health_status(method="GET")
            return (
                isinstance(status, dict)
                and status.get("initialized", False)
                and not status.get("sealed", True)
            )
        except Exception as e:
            logger.warning(f"Vault availability check failed: {e}")
            return False

    def get_secret(self, path: str) -> dict:
        """Read a secret from Vault KV v2"""
        try:
            result = self.client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self.mount_point
            )
            return result["data"]["data"]
        except Exception as e:
            logger.error(f"Failed to read Vault secret at {path}: {e}")
            raise


# ===================== RAGStackTestContainers =====================


class RAGStackTestContainers:
    """Manages test containers for RAG stack including Vault, Qdrant, Langfuse, and LLM orchestration service"""

    def __init__(self, compose_file_name: str = "docker-compose-eval.yml"):
        self.project_root = Path(__file__).parent.parent.parent
        self.compose_file_path = self.project_root / compose_file_name
        self.compose: Optional[DockerCompose] = None
        self.services_info: Dict[str, Dict[str, Any]] = {}

        if not self.compose_file_path.exists():
            raise FileNotFoundError(
                f"Docker compose file not found: {self.compose_file_path}"
            )

    def start(self) -> None:
        """Start all test containers and bootstrap Vault"""
        logger.info("Starting RAG Stack testcontainers...")
        os.environ["EVAL_MODE"] = "true"

        # Download embeddings from Azure before starting containers
        self._download_embeddings_from_azure()

        # Prepare Vault Agent directories
        agent_in = self.project_root / "test-vault" / "agents" / "llm"
        agent_out = self.project_root / "test-vault" / "agent-out"
        agent_in.mkdir(parents=True, exist_ok=True)
        agent_out.mkdir(parents=True, exist_ok=True)

        # Clean up any stale files from previous runs
        for f in ["role_id", "secret_id", "token", "pidfile", "dummy"]:
            (agent_in / f).unlink(missing_ok=True)
            (agent_out / f).unlink(missing_ok=True)

        # Start all Docker Compose services
        logger.info("Starting Docker Compose services...")
        self.compose = DockerCompose(
            str(self.project_root),
            compose_file_name=self.compose_file_path.name,
            pull=False,
        )
        self.compose.start()

        # Get Vault connection details
        vault_url = self._get_vault_url()
        logger.info(f"Vault URL: {vault_url}")

        # Wait for Vault to be ready
        self._wait_for_vault_ready(vault_url)

        # Configure Vault with AppRole, policies, and test secrets
        self._bootstrap_vault_dev(agent_in, vault_url)

        # Verify credentials were written successfully
        role_id = (agent_in / "role_id").read_text().strip()
        secret_id = (agent_in / "secret_id").read_text().strip()
        logger.info(
            f"AppRole credentials written: role_id={role_id[:8]}..., secret_id={secret_id[:8]}..."
        )

        # Wait for Vault Agent to authenticate and write token
        logger.info("Waiting for vault-agent to authenticate...")
        self._wait_for_valid_token(agent_out / "token", vault_url, max_attempts=20)

        logger.info("Vault Agent authenticated successfully")

        # Wait for other services to be ready
        self._wait_for_services()
        self._collect_service_info()

        # Index test data into Qdrant
        self._index_test_data()

        logger.info("RAG Stack testcontainers ready")

    def stop(self) -> None:
        """Stop all test containers"""
        if self.compose:
            logger.info("Stopping RAG Stack testcontainers...")
            self.compose.stop()
            logger.info("Testcontainers stopped")

    def _download_embeddings_from_azure(self) -> None:
        """Download embeddings from Azure Blob Storage if configured."""
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "test-embeddings")
        blob_name = os.getenv("AZURE_STORAGE_BLOB_NAME", "test_embeddings.json")

        # Local path where embeddings should be saved
        embeddings_file = self.project_root / "tests" / "data" / "test_embeddings.json"

        # Require Azure configuration for CI/CD
        if not connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is required to download embeddings. "
                "Either set this environment variable or ensure test_embeddings.json "
                f"exists at {embeddings_file}"
            )

        logger.info("=" * 80)
        logger.info("DOWNLOADING EMBEDDINGS FROM AZURE BLOB STORAGE")
        logger.info("=" * 80)

        try:
            download_embeddings_from_azure(
                connection_string=connection_string,
                container_name=container_name,
                blob_name=blob_name,
                local_path=embeddings_file,
            )
            logger.info("Embeddings download complete")
        except Exception as e:
            logger.error(f"Failed to download embeddings from Azure: {e}")
            raise

    def _get_vault_url(self) -> str:
        """Get the mapped Vault URL accessible from the host"""
        if not self.compose:
            raise RuntimeError("Docker Compose not initialized")
        host = self.compose.get_service_host("vault", 8200)
        port = self.compose.get_service_port("vault", 8200)
        return f"http://{host}:{port}"

    def _wait_for_vault_ready(self, vault_url: str, timeout: int = 60) -> None:
        """Wait for Vault to be initialized and unsealed"""
        logger.info("Waiting for Vault to be available...")
        client = hvac.Client(url=vault_url, token="root", timeout=10)

        start = time.time()
        while time.time() - start < timeout:
            try:
                status = client.sys.read_health_status(method="GET")
                if status.get("initialized", False) and not status.get("sealed", True):
                    logger.info("Vault is available and unsealed")
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError("Vault did not become available within 60s")

    def _bootstrap_vault_dev(self, agent_in: Path, vault_url: str) -> None:
        """
        Bootstrap Vault dev instance with:
        - AppRole auth method
        - Policy for LLM orchestration service
        - AppRole role and credentials
        - Test secrets (LLM connections, Langfuse, embeddings, guardrails)
        """
        logger.info("Bootstrapping Vault with AppRole and test secrets...")
        client = hvac.Client(url=vault_url, token="root")

        # Enable AppRole authentication method
        if "approle/" not in client.sys.list_auth_methods():
            client.sys.enable_auth_method("approle")
            logger.info("AppRole enabled")

        # Create policy with permissions for all secret paths (updated with correct embedding paths)
        policy = """
path "secret/metadata/llm/*" { capabilities = ["list"] }
path "secret/data/llm/*"     { capabilities = ["read"] }
path "secret/metadata/langfuse/*" { capabilities = ["list"] }
path "secret/data/langfuse/*"     { capabilities = ["read"] }
path "secret/metadata/embeddings/*" { capabilities = ["list"] }
path "secret/data/embeddings/*"     { capabilities = ["read"] }
path "secret/metadata/guardrails/*" { capabilities = ["list"] }
path "secret/data/guardrails/*"     { capabilities = ["read"] }
path "auth/token/lookup-self" { capabilities = ["read"] }
path "auth/token/renew-self" { capabilities = ["update"] }
"""
        client.sys.create_or_update_policy("llm-orchestration", policy)
        logger.info("Policy 'llm-orchestration' created")

        # Create AppRole role with service token type
        role_name = "llm-orchestration-service"
        client.write(
            f"auth/approle/role/{role_name}",
            **{
                "token_policies": "llm-orchestration",
                "secret_id_ttl": "24h",
                "token_ttl": "1h",
                "token_max_ttl": "24h",
                "secret_id_num_uses": 0,
                "bind_secret_id": True,
                "token_no_default_policy": True,
                "token_type": "service",
            },
        )
        logger.info(f"AppRole '{role_name}' created")

        # Generate credentials for the AppRole
        role_id = client.read(f"auth/approle/role/{role_name}/role-id")["data"][
            "role_id"
        ]
        secret_id = client.write(f"auth/approle/role/{role_name}/secret-id")["data"][
            "secret_id"
        ]

        # Write credentials to files that Vault Agent will read
        (agent_in / "role_id").write_text(role_id, encoding="utf-8")
        (agent_in / "secret_id").write_text(secret_id, encoding="utf-8")
        logger.info("AppRole credentials written to agent-in/")

        # Write test secrets
        self._write_test_secrets(client)

    def _write_test_secrets(self, client: hvac.Client) -> None:
        """Write all test secrets to Vault with correct path structure"""

        # ============================================================
        # CRITICAL DEBUG SECTION - Environment Variables
        # ============================================================
        logger.info("=" * 80)
        logger.info("VAULT SECRET BOOTSTRAP - ENVIRONMENT VARIABLES DEBUG")
        logger.info("=" * 80)

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        # Validate critical environment variables
        missing_vars = []
        if not azure_endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not azure_api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not azure_embedding_deployment:
            missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        if missing_vars:
            error_msg = f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("All required environment variables are set")
        logger.info("=" * 80)

        # ============================================================
        # CHAT MODEL SECRET (LLM path)
        # ============================================================
        logger.info("")
        logger.info("Writing LLM connection secret (chat model)...")
        llm_secret = {
            "connection_id": "evalconnection-1",
            "endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "deployment_name": azure_deployment or "gpt-4o-mini",
            "environment": "testing",
            "model": "gpt-4o-mini",
            "model_type": "chat",
            "api_version": "2024-02-15-preview",
            "tags": "azure,test,chat",
        }

        logger.info(f"  → chat deployment: {llm_secret['deployment_name']}")
        logger.info(f"  → endpoint: {llm_secret['endpoint']}")
        logger.info(f"  → connection_id: {llm_secret['connection_id']}")

        client.secrets.kv.v2.create_or_update_secret(
            mount_point="secret",
            path="llm/connections/azure_openai/evalconnection-1",
            secret=llm_secret,
        )
        logger.info(
            "LLM connection secret written to llm/connections/azure_openai/evalconnection-1"
        )

        # ============================================================
        # EMBEDDING MODEL SECRET (Embeddings path)
        # ============================================================
        logger.info("")
        logger.info("Writing embedding model secret...")
        embedding_secret = {
            "connection_id": "evalconnection-1",
            "endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "deployment_name": azure_embedding_deployment,  # This is the embedding deployment
            "environment": "testing",
            "model": "text-embedding-3-large",
            "model_type": "embedding",
            "api_version": "2024-02-15-preview",
            "max_tokens": 2048,
            "vector_size": 3072,
            "tags": "azure,embedding,test",
        }

        logger.info(f"  → model: {embedding_secret['model']}")
        logger.info(f"  → connection_id: {embedding_secret['connection_id']}")
        logger.info(
            "  → Vault path: embeddings/connections/azure_openai/evalconnection-1"
        )

        # Write to embeddings path with connection_id in the path
        client.secrets.kv.v2.create_or_update_secret(
            mount_point="secret",
            path="embeddings/connections/azure_openai/evalconnection-1",
            secret=embedding_secret,
        )
        logger.info(
            "Embedding secret written to embeddings/connections/azure_openai/evalconnection-1"
        )

        # ============================================================
        # VERIFY SECRETS WERE WRITTEN CORRECTLY
        # ============================================================
        logger.info("")
        logger.info("Verifying secrets in Vault...")
        try:
            # Verify LLM path
            verify_llm = client.secrets.kv.v2.read_secret_version(
                path="llm/connections/azure_openai/development/evalconnection-1",
                mount_point="secret",
            )
            llm_data = verify_llm["data"]["data"]
            logger.info("LLM path verified:")
            logger.info(f"    • connection_id: {llm_data.get('connection_id')}")

            # Verify embeddings path
            verify_embedding = client.secrets.kv.v2.read_secret_version(
                path="embeddings/connections/azure_openai/development/evalconnection-1",
                mount_point="secret",
            )
            embedding_data = verify_embedding["data"]["data"]
            logger.info("Embeddings path verified:")
            logger.info(f"    • model: {embedding_data.get('model')}")
            logger.info(f"    • connection_id: {embedding_data.get('connection_id')}")

            # Critical validation
            if embedding_data.get("deployment_name") != azure_embedding_deployment:
                error_msg = (
                    "VAULT SECRET MISMATCH! "
                    f"Expected deployment_name='{azure_embedding_deployment}' "
                    f"but Vault has '{embedding_data.get('deployment_name')}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if embedding_data.get("connection_id") != "evalconnection-1":
                error_msg = (
                    "VAULT SECRET MISMATCH! "
                    "Expected connection_id='evalconnection-1' "
                    f"but Vault has '{embedding_data.get('connection_id')}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info("Secret verification PASSED")

        except Exception as e:
            logger.error(f"Failed to verify secrets: {e}")
            raise

        # ============================================================
        # LANGFUSE CONFIGURATION
        # ============================================================
        logger.info("")
        logger.info("Writing Langfuse configuration secret...")
        langfuse_secret = {
            "public_key": "pk-lf-test",
            "secret_key": "sk-lf-test",
            "host": "http://langfuse-web:3000",
        }
        client.secrets.kv.v2.create_or_update_secret(
            mount_point="secret", path="langfuse/config", secret=langfuse_secret
        )
        logger.info("Langfuse configuration secret written")

        logger.info("=" * 80)
        logger.info("ALL SECRETS WRITTEN SUCCESSFULLY")
        logger.info("=" * 80)

    def _capture_service_logs(self) -> None:
        """Capture logs from all services before cleanup."""
        services = ["llm-orchestration-service", "vault", "qdrant", "langfuse-web"]

        for service in services:
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"LOGS: {service}")
                logger.info("=" * 60)

                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(self.compose_file_path),
                        "logs",
                        "--tail",
                        "200",
                        service,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(self.project_root),
                )

                if result.stdout:
                    logger.info(result.stdout)
                if result.stderr:
                    logger.error(result.stderr)

            except Exception as e:
                logger.error(f"Failed to capture logs for {service}: {e}")

    def _wait_for_valid_token(
        self, token_path: Path, vault_url: str, max_attempts: int = 20
    ) -> None:
        """Wait for Vault Agent to write a valid token and verify it works"""
        for attempt in range(max_attempts):
            if token_path.exists() and token_path.stat().st_size > 0:
                try:
                    # Fix permissions before reading
                    self._fix_token_file_permissions(token_path)

                    token = token_path.read_text().strip()

                    client = hvac.Client(url=vault_url, token=token)
                    try:
                        client.lookup_token()

                        if client.is_authenticated():
                            logger.info(f"Valid token obtained (attempt {attempt + 1})")
                            self._verify_token_permissions(client)
                            return
                    except Exception as e:
                        if attempt < max_attempts - 1:
                            logger.debug(
                                f"Token validation error (attempt {attempt + 1}): {type(e).__name__}"
                            )
                except PermissionError as e:
                    logger.warning(
                        f"Permission error reading token file (attempt {attempt + 1}): {e}"
                    )
                    # Try to fix permissions again
                    self._fix_token_file_permissions(token_path, force=True)

            time.sleep(2)

        logger.error("Failed to obtain valid Vault token")
        self._check_agent_logs()
        raise TimeoutError(
            f"Failed to obtain valid Vault token after {max_attempts} attempts"
        )

    def _fix_token_file_permissions(
        self, token_path: Path, force: bool = False
    ) -> None:
        """Fix permissions on token file to make it readable by host user"""
        try:
            # Try to change permissions using subprocess (requires Docker to be accessible)
            if force:
                logger.info(
                    "Attempting to fix token file permissions using docker exec..."
                )
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        "vault-agent-llm",
                        "chmod",
                        "644",
                        "/agent/llm-token/token",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info(
                        "Successfully fixed token file permissions via docker exec"
                    )
                else:
                    logger.warning(
                        f"Failed to fix permissions via docker exec: {result.stderr}"
                    )

            # Also try direct chmod (may not work in all environments)
            try:
                os.chmod(token_path, 0o644)
            except Exception as chmod_error:
                logger.debug(
                    f"Direct chmod failed (expected in some environments): {chmod_error}"
                )

        except Exception as e:
            logger.debug(f"Could not fix token file permissions: {e}")

    def _verify_token_permissions(self, client: hvac.Client) -> None:
        """Verify the token has correct permissions to read secrets"""
        try:
            client.secrets.kv.v2.read_secret_version(
                path="llm/connections/azure_openai/development/evalconnection-1",
                mount_point="secret",
            )
            logger.info("Token has correct permissions to read secrets")
        except Exception as e:
            logger.error(f"Token cannot read secrets: {e}")
            raise

    def _check_agent_logs(self) -> None:
        """Check vault-agent logs for debugging authentication issues"""
        result = subprocess.run(
            ["docker", "logs", "--tail", "50", "vault-agent-llm"],
            capture_output=True,
            text=True,
        )
        logger.error(f"Vault Agent Logs:\n{result.stdout}\n{result.stderr}")

    def _wait_for_services(self, total_timeout: int = 300) -> None:
        """Wait for all services to be healthy"""
        services = [
            ("qdrant", 6333, self._check_qdrant, 60),
            ("langfuse-web", 3000, self._check_langfuse, 120),
            ("llm-orchestration-service", 8100, self._check_orchestration, 180),
        ]
        start = time.time()
        for name, port, check, timeout in services:
            self._wait_single(name, port, check, timeout, start, total_timeout)

    def _wait_single(
        self,
        name: str,
        port: int,
        check: Any,
        timeout: int,
        global_start: float,
        total_timeout: int,
    ) -> None:
        """Wait for a single service to be ready"""
        if self.compose is None:
            return

        logger.info(f"Waiting for {name}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                host = self.compose.get_service_host(name, port)
                mapped_port = self.compose.get_service_port(name, port)
                if check(host, mapped_port):
                    logger.info(f"{name} ready at {host}:{mapped_port}")
                    self.services_info[name] = {
                        "host": host,
                        "port": mapped_port,
                        "url": f"http://{host}:{mapped_port}",
                    }
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(f"Timeout waiting for {name}")

    def _check_qdrant(self, host: str, port: int) -> bool:
        """Check if Qdrant is ready"""
        try:
            r = requests.get(f"http://{host}:{port}/collections", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _check_langfuse(self, host: str, port: int) -> bool:
        """Check if Langfuse is ready"""
        try:
            r = requests.get(f"http://{host}:{port}/api/public/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _check_orchestration(self, host: str, port: int) -> bool:
        """Check if LLM orchestration service is healthy"""
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=5)
            return r.status_code == 200 and r.json().get("status") == "healthy"
        except Exception:
            return False

    def _collect_service_info(self) -> None:
        """Collect service connection information"""
        if self.compose:
            self.services_info["vault"] = {
                "host": self.compose.get_service_host("vault", 8200),
                "port": self.compose.get_service_port("vault", 8200),
                "url": self._get_vault_url(),
            }

    def get_orchestration_service_url(self) -> str:
        """Get the URL for the LLM orchestration service"""
        return self.services_info["llm-orchestration-service"]["url"]

    def get_qdrant_url(self) -> str:
        """Get the URL for Qdrant"""
        return self.services_info["qdrant"]["url"]

    def get_vault_url(self) -> str:
        """Get the URL for Vault"""
        return self.services_info["vault"]["url"]

    def get_langfuse_url(self) -> str:
        """Get the URL for Langfuse"""
        return self.services_info.get("langfuse-web", {}).get(
            "url", "http://localhost:3000"
        )

    def is_service_available(self, service_name: str) -> bool:
        """Check if a service is available"""
        return service_name in self.services_info

    def _index_test_data(self) -> None:
        """Index test documents into Qdrant for retrieval testing."""
        logger.info("Indexing test data into Qdrant contextual collections...")

        try:
            from tests.helpers.test_data_loader import load_test_data_into_qdrant

            load_test_data_into_qdrant(
                orchestration_url=self.get_orchestration_service_url(),
                qdrant_url=self.get_qdrant_url(),
            )

            logger.info("Test data indexing complete")

        except Exception as e:
            logger.error(f"Failed to index test data: {e}")
            raise


# ===================== Pytest Fixtures =====================


@pytest.fixture(scope="session")
def rag_stack() -> Generator[RAGStackTestContainers, None, None]:
    """
    Session-scoped fixture that starts all test containers once per test session.
    Containers are automatically stopped after all tests complete.
    """
    stack = RAGStackTestContainers()
    try:
        stack.start()
        yield stack
    except Exception as e:
        # If startup fails, capture logs before cleanup
        logger.error(f"RAG stack startup failed: {e}")
        try:
            stack._capture_service_logs()
        except Exception as e:
            logger.error(f"Could not capture logs after startup failure: {e}")
            pass
        raise
    finally:
        logger.info("=" * 80)
        logger.info("CAPTURING SERVICE LOGS BEFORE CLEANUP")
        logger.info("=" * 80)
        try:
            stack._capture_service_logs()
        except Exception as e:
            logger.error(f"Could not capture logs: {e}")
        stack.stop()


@pytest.fixture(scope="function")
def orchestration_client(rag_stack: RAGStackTestContainers):
    """
    function-scoped fixture that provides the orchestration service URL.
    Tests can use either requests (sync) or httpx (async).
    """

    class OrchestrationClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

    return OrchestrationClient(rag_stack.get_orchestration_service_url())


@pytest.fixture(scope="session")
def api_tool_endpoints_indexed(rag_stack: RAGStackTestContainers):
    """
    Session-scoped fixture that seeds the API tool endpoints from
    ``tests/api_tool_eval/test-endpoints.json`` into Qdrant's
    ``api_tool_collection`` so the agentic loop can find them.

    Required by tests in ``tests/deepeval_tests/api_tool_tests.py`` — without
    it, semantic search over API tools returns nothing and the orchestrator
    falls back to RAG.

    The indexer module hard-codes container-internal URLs in its constants
    (``http://llm-orchestration-service:8100``, ``qdrant:6333``). Two of the
    three are used in code paths that re-read the constant at call time
    (monkey-patching the class attribute works), but ``ApiToolQdrantManager``
    is instantiated with no arguments inside ``index_endpoint`` and binds the
    host/port at function-definition time as defaults. To override those, the
    fixture also replaces the ``ApiToolQdrantManager`` name imported into
    ``main_indexer`` with a factory that injects the testcontainers-mapped
    host/port.
    """
    import asyncio
    import json as _json
    from pathlib import Path as _Path
    from unittest.mock import patch
    from urllib.parse import urlparse

    from api_tool_indexer import main_indexer
    from api_tool_indexer.constants import ApiToolIndexerConstants
    from api_tool_indexer.main_indexer import index_endpoint
    from api_tool_indexer.models import EndpointData
    from api_tool_indexer.qdrant_manager import ApiToolQdrantManager

    orch_url = rag_stack.get_orchestration_service_url()
    qdrant_url = rag_stack.get_qdrant_url()
    parsed = urlparse(qdrant_url)
    qdrant_host = parsed.hostname or "localhost"
    qdrant_port = parsed.port or 6333

    def _qdrant_factory(*args: Any, **kwargs: Any) -> ApiToolQdrantManager:
        kwargs.setdefault("host", qdrant_host)
        kwargs.setdefault("port", qdrant_port)
        return ApiToolQdrantManager(*args, **kwargs)

    endpoints_file = (
        _Path(__file__).parent.parent / "api_tool_eval" / "test-endpoints.json"
    )
    if not endpoints_file.exists():
        pytest.skip(
            f"API tool endpoint fixture not found: {endpoints_file} — "
            "cannot seed api_tool_collection"
        )

    with open(endpoints_file, encoding="utf-8") as f:
        endpoints = _json.load(f)

    async def _seed_all() -> list:
        results = []
        for ep in endpoints:
            endpoint_data = EndpointData(
                endpoint_id=ep["endpointId"],
                name=ep["name"],
                description=ep["description"],
                url=ep["url"],
                method=ep["method"],
                params=ep.get("params", []),
            )
            res = await index_endpoint(endpoint_data)
            logger.info(
                f"Seeded endpoint '{ep['name']}': success={res.success} ({res.message})"
            )
            results.append((ep["name"], res))
        return results

    # Connection ID "evalconnection-1" matches the one used by standard_tests.py
    # and the EVAL_MODE setup; vault is seeded for it. The default
    # "gpt-4o-mini" used by the indexer in production has no test fixture.
    with (
        patch.object(ApiToolIndexerConstants, "DEFAULT_API_BASE_URL", orch_url),
        patch.object(
            ApiToolIndexerConstants,
            "DEFAULT_CONNECTION_ID",
            "evalconnection-1",
        ),
        patch.object(main_indexer, "ApiToolQdrantManager", _qdrant_factory),
    ):
        logger.info(
            f"Seeding {len(endpoints)} API tool endpoints "
            f"(orch={orch_url}, qdrant={qdrant_host}:{qdrant_port})"
        )
        results = asyncio.run(_seed_all())

    failed = [name for name, r in results if not r.success]
    if failed:
        pytest.skip(
            f"API tool seeding failed for {len(failed)}/{len(endpoints)} "
            f"endpoints ({failed}) — skipping API tool tests"
        )

    logger.info(f"Indexed all {len(endpoints)} API tool endpoints successfully")
    yield

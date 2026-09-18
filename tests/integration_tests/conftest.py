import pytest
from testcontainers.compose import DockerCompose
from minio import Minio
from qdrant_client import QdrantClient
from pathlib import Path
import time
import json
import os
import hvac
import subprocess
from typing import Dict, Any, Optional, Generator
import requests
from loguru import logger


# Deterministic vault_uuid used for the integration-test LLM connection.
# Vault secret paths now terminate in the connection's vault_uuid (no environment
# segment) — see SecretResolver._build_vault_path. We pin a fixed, valid UUID so
# the secret we write, the DB connection row, and the value the DSL forwards to
# /orchestrate/test all line up without depending on a DB-generated UUID.
TEST_VAULT_UUID = "00000000-0000-0000-0000-000000000001"


# ===================== VaultAgentClient =====================
class VaultAgentClient:
    """Client for interacting with Vault using a token written by Vault Agent"""

    def __init__(
        self,
        vault_url: str,
        token_path: Path = Path("test-vault/agent-out/token"),
        mount_point: str = "secret",
        timeout: int = 10,
    ) -> None:
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


class RAGStackTestContainers:
    """Manages test containers for RAG stack including Vault, Qdrant, Langfuse, and LLM orchestration service"""

    def __init__(self, compose_file_name: str = "docker-compose-test.yml") -> None:
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
        # Prepare Vault Agent directories
        agent_in = self.project_root / "test-vault" / "agents" / "llm"
        agent_out = self.project_root / "test-vault" / "agent-out"
        agent_in.mkdir(parents=True, exist_ok=True)
        agent_out.mkdir(parents=True, exist_ok=True)

        # Clean up any stale files from previous runs
        for f in ["role_id", "secret_id", "token", "pidfile", "dummy"]:
            (agent_in / f).unlink(missing_ok=True)
            (agent_out / f).unlink(missing_ok=True)

        # Create Ruuter health endpoint for tests
        self._create_ruuter_health_endpoint()

        # Remove .guard files BEFORE starting containers
        # (Ruuter loads DSL on startup, so guards must be removed before that)
        self._remove_ruuter_guard_files()

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

        # Restart vault-agent to ensure it picks up the credentials
        logger.info("Restarting vault-agent to authenticate...")
        try:
            import subprocess

            subprocess.run(
                ["docker", "restart", "vault-agent-llm"],
                check=True,
                capture_output=True,
            )
            logger.info("vault-agent restarted")
            time.sleep(3)  # Give it time to start
        except Exception as e:
            logger.warning(f"Could not restart vault-agent: {e}")

        # Wait for Vault Agent to authenticate and write token
        logger.info("Waiting for vault-agent to authenticate...")
        self._wait_for_valid_token(agent_out / "token", vault_url, max_attempts=20)

        logger.info("Vault Agent authenticated successfully")

        # Wait for other services to be ready
        self._wait_for_services()
        self._collect_service_info()

        # Run database migration
        self._run_database_migration()

        logger.info("RAG Stack testcontainers ready")

    def stop(self) -> None:
        """Stop all test containers"""
        if self.compose:
            logger.info("Stopping RAG Stack testcontainers...")
            self.compose.stop()
            logger.info("Testcontainers stopped")

        # Clean up test files
        self._remove_ruuter_health_endpoint()

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
            except Exception as e:
                logger.debug(f"Vault not ready yet: {e}")
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
        logger.info("VAULT SECRET BOOTSTRAP - ENVIRONMENT VARIABLES DEBUG")

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        azure_embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

        # Validate critical environment variables
        missing_vars = []
        if not azure_endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not azure_api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not azure_embedding_deployment:
            missing_vars.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        if not azure_embedding_endpoint:
            missing_vars.append("AZURE_OPENAI_EMBEDDING_ENDPOINT")

        if missing_vars:
            error_msg = f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("All required environment variables are set")
        logger.info("=" * 80)

        # ============================================================
        # SECRET PATHS
        # ============================================================
        # Vault paths now terminate in the connection's vault_uuid (no environment
        # segment) — see SecretResolver._build_vault_path / _build_embedding_vault_path.
        # We write both the chat and embedding secrets under the SAME pinned
        # vault_uuid (TEST_VAULT_UUID); the testing DB connection is forced to carry
        # this UUID in ensure_testing_connection, and inference/test.yml forwards it
        # to /orchestrate/test. The LLM, embedding and guardrails resolvers all read
        # the secret using this UUID as the connection_id.
        llm_vault_path = f"llm/connections/azure_openai/{TEST_VAULT_UUID}"
        embedding_vault_path = f"embeddings/connections/azure_openai/{TEST_VAULT_UUID}"

        # ============================================================
        # CHAT MODEL SECRET (LLM path)
        # ============================================================
        logger.info("")
        logger.info("Writing LLM connection secret (chat model)...")
        llm_secret = {
            "connection_id": TEST_VAULT_UUID,
            "endpoint": azure_endpoint,
            "api_key": azure_api_key,
            "deployment_name": azure_deployment or "gpt-4o-mini",
            "environment": "test",
            "model": "gpt-4o-mini",
            "model_type": "chat",
            "api_version": "2024-02-15-preview",
            "tags": "azure,test,chat",
        }

        logger.info(f"  chat deployment: {llm_secret['deployment_name']}")
        logger.info(f"  endpoint: {llm_secret['endpoint']}")
        logger.info(f"  vault path: {llm_vault_path}")

        client.secrets.kv.v2.create_or_update_secret(
            mount_point="secret",
            path=llm_vault_path,
            secret=llm_secret,
        )
        logger.info(f"LLM connection secret written to {llm_vault_path}")

        # ============================================================
        # EMBEDDING MODEL SECRET (Embeddings path)
        # ============================================================
        logger.info("")
        logger.info("Writing embedding model secret...")
        embedding_secret = {
            "connection_id": TEST_VAULT_UUID,
            "endpoint": azure_embedding_endpoint,
            "api_key": azure_api_key,
            "deployment_name": azure_embedding_deployment,
            "environment": "test",
            "model": "text-embedding-3-large",
            "api_version": "2024-12-01-preview",
            "tags": "azure,test,text-embedding-3-large",
        }

        logger.info(f"  → model: {embedding_secret['model']}")
        logger.info(f"  → vault path: {embedding_vault_path}")

        client.secrets.kv.v2.create_or_update_secret(
            mount_point="secret",
            path=embedding_vault_path,
            secret=embedding_secret,
        )
        logger.info(f"Embedding secret written to {embedding_vault_path}")

        # ============================================================
        # VERIFY SECRETS WERE WRITTEN CORRECTLY
        # ============================================================
        logger.info("")
        logger.info("Verifying secrets in Vault...")
        try:
            # Verify LLM path
            verify_llm = client.secrets.kv.v2.read_secret_version(
                path=llm_vault_path,
                mount_point="secret",
            )
            llm_data = verify_llm["data"]["data"]
            logger.info("LLM path verified:")
            logger.info(f"    connection_id: {llm_data.get('connection_id')}")

            # Verify embeddings path
            verify_embedding = client.secrets.kv.v2.read_secret_version(
                path=embedding_vault_path,
                mount_point="secret",
            )
            embedding_data = verify_embedding["data"]["data"]
            logger.info("Embeddings path verified:")
            logger.info(f"    model: {embedding_data.get('model')}")
            logger.info(f"    connection_id: {embedding_data.get('connection_id')}")

            # Critical validation
            if embedding_data.get("deployment_name") != azure_embedding_deployment:
                error_msg = (
                    "VAULT SECRET MISMATCH! "
                    f"Expected deployment_name='{azure_embedding_deployment}' "
                    f"but Vault has '{embedding_data.get('deployment_name')}'"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if embedding_data.get("connection_id") != TEST_VAULT_UUID:
                error_msg = (
                    "VAULT SECRET MISMATCH! "
                    f"Expected connection_id='{TEST_VAULT_UUID}' "
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

        # ============================================================
        # GUARDRAILS CONFIGURATION
        # ============================================================

        logger.info("ALL SECRETS WRITTEN SUCCESSFULLY")

    def _run_database_migration(self) -> None:
        """Run Liquibase database migration using migrate.sh script."""
        logger.info("Running database migration...")

        try:
            # Run the migrate.sh script from the project root
            # Note: migrate.sh uses network 'bykstack' but we use 'test-network'
            # So we need to run Liquibase directly with the test network
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "test-network",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/changelog:/liquibase/changelog",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/master.yml:/liquibase/master.yml",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/data:/liquibase/data",
                    "liquibase/liquibase:4.33",
                    "--defaultsFile=/liquibase/changelog/liquibase.properties",
                    "--changelog-file=master.yml",
                    "--url=jdbc:postgresql://rag-search-db:5432/rag-search?user=postgres",
                    "--password=dbadmin",
                    "update",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            if result.returncode == 0:
                logger.info("Database migration completed successfully")
                logger.debug(f"Migration output: {result.stdout}")
            else:
                logger.error(f"Database migration failed with code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Database migration failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Database migration timed out after 120 seconds")
            raise
        except Exception as e:
            logger.error(f"Failed to run database migration: {e}")
            raise

    def _run_database_migration(self) -> None:
        """Run Liquibase database migration using migrate.sh script."""
        logger.info("Running database migration...")

        try:
            # Run the migrate.sh script from the project root
            # Note: migrate.sh uses network 'bykstack' but we use 'test-network'
            # So we need to run Liquibase directly with the test network
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "test-network",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/changelog:/liquibase/changelog",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/master.yml:/liquibase/master.yml",
                    "-v",
                    f"{self.project_root}/DSL/Liquibase/data:/liquibase/data",
                    "liquibase/liquibase:4.33",
                    "--defaultsFile=/liquibase/changelog/liquibase.properties",
                    "--changelog-file=master.yml",
                    "--url=jdbc:postgresql://rag-search-db:5432/rag-search?user=postgres",
                    "--password=dbadmin",
                    "update",
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.project_root),
            )

            if result.returncode == 0:
                logger.info("Database migration completed successfully")
                logger.debug(f"Migration output: {result.stdout}")
            else:
                logger.error(f"Database migration failed with code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"Database migration failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("Database migration timed out after 120 seconds")
            raise
        except Exception as e:
            logger.error(f"Failed to run database migration: {e}")
            raise

    def _capture_service_logs(self) -> None:
        """Capture logs from all services before cleanup."""
        services = [
            "llm-orchestration-service",
            "ruuter-public",
            "ruuter-private",
            "cron-manager",
            "vault",
            "qdrant",
            "langfuse-web",
        ]

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
                        "/agent/out/token",
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
                path=f"llm/connections/azure_openai/{TEST_VAULT_UUID}",
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
            ("ruuter-private", 8088, self._check_ruuter_private, 90),
            ("ruuter-public", 8086, self._check_ruuter_public, 90),
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
        attempt = 0
        while time.time() - start < timeout:
            attempt += 1
            elapsed = time.time() - start
            try:
                host = self.compose.get_service_host(name, port)
                mapped_port = self.compose.get_service_port(name, port)
                logger.debug(
                    f"{name} - Attempt {attempt} ({elapsed:.1f}s) - Checking {host}:{mapped_port}"
                )
                if check(host, mapped_port):
                    logger.info(
                        f"{name} ready at {host}:{mapped_port} (took {elapsed:.1f}s, {attempt} attempts)"
                    )
                    self.services_info[name] = {
                        "host": host,
                        "port": mapped_port,
                        "url": f"http://{host}:{mapped_port}",
                    }
                    return
            except Exception as e:
                logger.debug(f"{name} - Attempt {attempt} failed: {e}")
            time.sleep(3)

        elapsed_total = time.time() - start
        raise TimeoutError(
            f"Timeout waiting for {name} after {elapsed_total:.1f}s ({attempt} attempts)"
        )

    def _check_qdrant(self, host: str, port: int) -> bool:
        """Check if Qdrant is ready"""
        try:
            r = requests.get(f"http://{host}:{port}/collections", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def _check_ruuter_private(self, host: str, port: int) -> bool:
        """Check if Ruuter Private is ready using the /health endpoint"""
        try:
            # Use the health endpoint we created for testing
            r = requests.get(f"http://{host}:{port}/rag-search/health", timeout=5)
            logger.debug(
                f"Ruuter Private health check - Status: {r.status_code}, Response: {r.text[:100]}"
            )

            # If we get 200, Ruuter is processing DSL correctly
            if r.status_code == 200:
                logger.debug("Ruuter Private health check passed with 200 status")
                return True

            logger.debug(
                f"Ruuter Private health check failed - unexpected status: {r.status_code}"
            )
            return False
        except Exception as e:
            logger.debug(
                f"Ruuter Private health check exception: {type(e).__name__}: {e}"
            )
            return False

    def _check_ruuter_public(self, host: str, port: int) -> bool:
        """Check if Ruuter Public is ready using the /health endpoint"""
        try:
            # Use the health endpoint we created for testing
            r = requests.get(f"http://{host}:{port}/rag-search/health", timeout=5)
            logger.debug(
                f"Ruuter Public health check - Status: {r.status_code}, Response: {r.text[:100]}"
            )

            # If we get 200, Ruuter is processing DSL correctly
            if r.status_code == 200:
                logger.debug("Ruuter Public health check passed with 200 status")
                return True

            logger.debug(
                f"Ruuter Public health check failed - unexpected status: {r.status_code}"
            )
            return False
        except Exception as e:
            logger.debug(
                f"Ruuter Public health check exception: {type(e).__name__}: {e}"
            )
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

    def _remove_ruuter_guard_files(self) -> None:
        """
        Remove .guard files from Ruuter DSL to disable authentication during tests.

        The .guard files are used by Ruuter to enforce authentication on endpoints.
        For integration tests, we need to disable this authentication.

        Note: Files are simply removed (not backed up) since they're in git.
        After tests, `git restore` can be used to restore them if needed.
        """
        guard_files = [
            "DSL/Ruuter.private/rag-search/GET/.guard",
            "DSL/Ruuter.private/rag-search/POST/.guard",
            "DSL/Ruuter.private/rag-search/POST/accounts/.guard",
        ]

        for guard_file in guard_files:
            guard_path = self.project_root / guard_file
            if guard_path.exists():
                try:
                    guard_path.unlink()
                    logger.info(f"Removed guard file: {guard_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove guard file {guard_file}: {e}")
            else:
                logger.debug(f"Guard file not found (already removed?): {guard_file}")

    def _create_ruuter_health_endpoint(self) -> None:
        """
        Create a simple /health endpoint for Ruuter health checks during tests.

        This endpoint is created dynamically and not committed to the repository.
        It's used to verify Ruuter is responding properly during test setup.
        Creates health endpoints for both Ruuter.private and Ruuter.public.
        """
        health_dsl_content = """declaration:
  call: declare
  version: 0.1
  description: "Health check endpoint for tests"
  method: get
  accepts: json
  returns: json
  namespace: rag-search

return_health:
  return: '{"status":"healthy","service":"ruuter"}'
  next: end
"""

        # Create health endpoint for both Ruuter.private and Ruuter.public
        for ruuter_dir in ["Ruuter.private", "Ruuter.public"]:
            health_endpoint_dir = (
                self.project_root / "DSL" / ruuter_dir / "rag-search" / "GET"
            )
            health_endpoint_dir.mkdir(parents=True, exist_ok=True)

            health_endpoint_path = health_endpoint_dir / "health.yml"

            try:
                health_endpoint_path.write_text(health_dsl_content)
                logger.info(
                    f"Created {ruuter_dir} health endpoint: {health_endpoint_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to create {ruuter_dir} health endpoint: {e}")

    def _remove_ruuter_health_endpoint(self) -> None:
        """
        Remove the dynamically created /health endpoint after tests complete.
        Removes health endpoints from both Ruuter.private and Ruuter.public.
        """
        # Remove health endpoint from both Ruuter.private and Ruuter.public
        for ruuter_dir in ["Ruuter.private", "Ruuter.public"]:
            health_endpoint_path = (
                self.project_root
                / "DSL"
                / ruuter_dir
                / "rag-search"
                / "GET"
                / "health.yml"
            )

            if health_endpoint_path.exists():
                try:
                    health_endpoint_path.unlink()
                    logger.info(f"Removed {ruuter_dir} health endpoint")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove {ruuter_dir} health endpoint: {e}"
                    )
            else:
                logger.debug(
                    f"{ruuter_dir} health endpoint file not found (already removed?)"
                )

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
def orchestration_client(rag_stack: RAGStackTestContainers) -> Any:
    """
    Function-scoped fixture that provides a configured requests session
    for testing the LLM orchestration service API.
    """
    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json", "Accept": "application/json"}
    )
    session.base_url = rag_stack.get_orchestration_service_url()
    return session


@pytest.fixture(scope="session")
def minio_client(rag_stack: RAGStackTestContainers) -> Minio:
    """Create MinIO client connected to test instance."""
    client = Minio(
        "localhost:9000",
        access_key="minio",
        secret_key="miniosecret",
        secure=False,
    )
    return client


@pytest.fixture(scope="session")
def qdrant_client(rag_stack: RAGStackTestContainers) -> QdrantClient:
    """Create Qdrant client connected to test instance."""
    client = QdrantClient(host="localhost", port=6333)
    return client


@pytest.fixture
def test_bucket(minio_client: Minio):
    """Create a test bucket with public read access and clean it up after test."""
    bucket_name = "test-integration-bucket"

    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

        # Set bucket policy to allow public read access
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": "*"},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{bucket_name}/*"],
                }
            ],
        }

        minio_client.set_bucket_policy(bucket_name, json.dumps(policy))

    yield bucket_name

    # Cleanup: remove all objects and bucket
    try:
        objects = minio_client.list_objects(bucket_name, recursive=True)
        for obj in objects:
            minio_client.remove_object(bucket_name, obj.object_name)
        minio_client.remove_bucket(bucket_name)
    except Exception as e:
        # Ignore cleanup errors - bucket may not exist or objects already deleted
        # This is acceptable in test teardown as it doesn't affect test results
        logger.debug(f"MinIO cleanup failed for bucket {bucket_name}: {e}")


@pytest.fixture
def test_document(test_bucket: str, minio_client: Minio, tmp_path: Path):
    """
    Create a test document with cleaned.txt and source.meta.json.

    Returns tuple of (bucket_name, object_prefix, local_path)
    """
    # Create test document directory structure
    doc_dir = tmp_path / "test_doc"
    doc_dir.mkdir()

    # Create cleaned.txt with sample content
    cleaned_content = """This is a test document for integration testing.

It contains multiple paragraphs to test chunking.

The document discusses RAG (Retrieval-Augmented Generation) systems.

RAG combines retrieval mechanisms with language models.

This helps provide accurate and contextual responses.

Integration testing ensures all components work together correctly.
"""
    cleaned_file = doc_dir / "cleaned.txt"
    cleaned_file.write_text(cleaned_content)

    # Create source.meta.json
    meta_content = {
        "source": "integration_test",
        "title": "Test Document",
        "created_at": "2025-01-01T00:00:00Z",
        "author": "Test Suite",
    }
    meta_file = doc_dir / "source.meta.json"
    meta_file.write_text(json.dumps(meta_content))

    # Upload to MinIO
    object_prefix = "test_documents/doc1"

    minio_client.fput_object(
        test_bucket, f"{object_prefix}/cleaned.txt", str(cleaned_file)
    )
    minio_client.fput_object(
        test_bucket, f"{object_prefix}/source.meta.json", str(meta_file)
    )

    return test_bucket, object_prefix, doc_dir


@pytest.fixture
def presigned_url(minio_client: Minio, test_document):
    """
    Generate presigned URL for test document.

    Note: For actual testing, you may need to create a zip archive
    and generate a presigned URL for that.
    """
    bucket_name, object_prefix, _ = test_document

    # Generate presigned URL (valid for 1 hour)
    from datetime import timedelta

    url = minio_client.presigned_get_object(
        bucket_name, f"{object_prefix}/cleaned.txt", expires=timedelta(hours=1)
    )

    return url


@pytest.fixture(scope="session")
def qdrant_collections():
    """List of Qdrant collection names used by the indexer."""
    return ["contextual_chunks_azure", "contextual_chunks_aws"]


@pytest.fixture(scope="session")
def llm_orchestration_url(rag_stack: RAGStackTestContainers) -> str:
    """
    URL for the LLM orchestration service.

    Depends on rag_stack to ensure all services are started and Vault is populated
    with LLM connection secrets before tests run.
    """
    return rag_stack.get_orchestration_service_url()


@pytest.fixture(scope="session")
def vault_client(rag_stack: RAGStackTestContainers):
    """Create Vault client connected to test instance using root token (dev mode)."""
    vault_url = rag_stack.get_vault_url()

    # In test environment, Vault runs in dev mode with known root token
    # This is simpler and avoids permission issues with agent-out token files
    client = hvac.Client(url=vault_url, token="root")

    # Verify connection
    if not client.is_authenticated():
        raise RuntimeError("Failed to authenticate with Vault using root token")

    logger.info("Vault client authenticated using dev mode root token")

    # Create a simple wrapper to match VaultAgentClient interface
    class SimpleVaultClient:
        def __init__(self, hvac_client):
            self.client = hvac_client

        def get_secret(self, path: str, mount_point: str = "secret") -> dict:
            """Read a secret from Vault KV v2"""
            result = self.client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=mount_point
            )
            return result["data"]["data"]

    return SimpleVaultClient(client)


@pytest.fixture(scope="session")
def postgres_client(rag_stack: RAGStackTestContainers):
    """Create PostgreSQL client connected to test database."""
    import psycopg2

    # Wait for database to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host="localhost",
                port=5436,
                database="rag-search",
                user="postgres",
                password="dbadmin",
                # RAG tables were moved to the rag_search schema (Liquibase v7
                # schema migration). Put it on the search_path so unqualified
                # queries (e.g. FROM llm_connections) resolve.
                options="-c search_path=rag_search,public",
            )
            logger.info("PostgreSQL connection established")
            yield conn
            conn.close()
            return
        except psycopg2.OperationalError:
            if attempt < max_attempts - 1:
                time.sleep(2)
            else:
                raise

    raise TimeoutError("Could not connect to PostgreSQL")


@pytest.fixture(scope="session")
def setup_agency_sync_schema(postgres_client):
    """Create agency_sync and mock_ckb tables for data update tests."""
    cursor = postgres_client.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.agency_sync (
                id VARCHAR(255) PRIMARY KEY,
                agency_data_hash VARCHAR(255),
                data_url TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS public.mock_ckb (
                client_id VARCHAR(255) PRIMARY KEY,
                client_data_hash VARCHAR(255) NOT NULL,
                signed_s3_url TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        postgres_client.commit()
        logger.info("Agency sync and mock CKB tables created")
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        raise
    finally:
        cursor.close()


@pytest.fixture(scope="function")
def ruuter_private_client(rag_stack: RAGStackTestContainers):
    """
    Function-scoped fixture that provides a configured requests session
    for testing Ruuter Private API endpoints.

    Ruuter Private is the routing layer that handles requests to the LLM orchestration
    service via DSL-defined endpoints.

    If Ruuter Private service is not available, tests using this fixture will be skipped.
    """
    # Check if Ruuter Private service is available
    if "ruuter-private" not in rag_stack.services_info:
        pytest.skip("Ruuter Private service not available")

    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json", "Accept": "application/json"}
    )
    # Ruuter Private runs on port 8088 in test environment
    session.base_url = "http://localhost:8088"
    return session


@pytest.fixture(scope="function")
def ruuter_public_client(rag_stack: RAGStackTestContainers):
    """
    Function-scoped fixture that provides a configured requests session
    for testing Ruuter Public API endpoints.

    Ruuter Public is the routing layer that handles requests to the LLM orchestration
    service via DSL-defined endpoints.

    If Ruuter Public service is not available, tests using this fixture will be skipped.
    """
    # Check if Ruuter Public service is available
    if "ruuter-public" not in rag_stack.services_info:
        pytest.skip("Ruuter Public service not available")

    session = requests.Session()
    session.headers.update(
        {"Content-Type": "application/json", "Accept": "application/json"}
    )
    # Ruuter Public runs on port 8088 in test environment
    session.base_url = "http://localhost:8086"
    return session


@pytest.fixture(scope="session")
def sample_test_data():
    """Load test data for inference tests."""
    test_data_path = Path(__file__).parent / "inference_test_data.json"

    if not test_data_path.exists():
        # Fallback to inline data if file doesn't exist
        logger.warning(
            f"Test data file not found at {test_data_path}, using fallback data"
        )
        return [
            {
                "question": "What is the retirement age?",
                "category": "pension_information",
                "expected_scope": True,
                "expected_keywords": ["retirement", "age", "pension"],
                "description": "Simple pension question",
            },
            {
                "question": "What is the capital of Mars?",
                "category": "out_of_scope",
                "expected_scope": False,
                "expected_keywords": [],
                "description": "Out of scope question",
            },
        ]

    with open(test_data_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} test cases from {test_data_path}")
    return data


@pytest.fixture(scope="function")
def ensure_testing_connection(postgres_client, ruuter_private_client, rag_stack):
    """
    Ensure a testing gpt-4o-mini LLM connection exists for testing inference tests.

    This fixture checks if a testing connection with gpt-4o-mini exists.
    If not found, it creates one via the Ruuter API.

    Note: Uses 'testing' environment to leverage the simpler /inference/test endpoint.
    """
    cursor = postgres_client.cursor()
    try:
        # First, check what connections exist in the database
        cursor.execute(
            "SELECT id, connection_name, environment, llm_model FROM llm_connections "
            "ORDER BY id"
        )
        all_connections = cursor.fetchall()
        logger.info(f"All connections in database: {len(all_connections)}")
        for conn in all_connections:
            logger.info(
                f"  - ID={conn[0]}, Name='{conn[1]}', Env={conn[2]}, Model={conn[3]}"
            )

        # Check for existing testing connection with gpt-4o-mini
        cursor.execute(
            "SELECT id, connection_name FROM llm_connections "
            "WHERE environment = 'testing' AND llm_model = 'gpt-4o-mini' "
            "LIMIT 1"
        )
        row = cursor.fetchone()

        if row is not None:
            connection_id, connection_name = row
            logger.info(
                f"Found existing testing gpt-4o-mini connection: "
                f"ID={connection_id}, Name='{connection_name}'"
            )
            # Pin the row's vault_uuid so it matches the vault secret path and the
            # value inference/test.yml forwards to /orchestrate/test.
            _pin_vault_uuid(postgres_client, connection_id)
            return connection_id

        # No testing gpt-4o-mini found - create one
        logger.info("No testing gpt-4o-mini connection found. Creating one...")

        payload = {
            "connection_name": "Testing gpt-4o-mini for Production Tests",
            "llm_platform": "azure",
            "llm_model": "gpt-4o-mini",
            "deployment_name": "gpt-4o-mini-deployment-test",
            "target_uri": "https://test-production.openai.azure.com/",
            "api_key": "test-production-api-key",
            "embedding_platform": "azure",
            "embedding_model": "text-embedding-3-large",
            "embedding_deployment_name": "text-embedding-prod-deployment",
            "embedding_target_uri": "https://test-production.openai.azure.com/",
            "embedding_azure_api_key": "test-embedding-prod-key",
            "monthly_budget": 10000.00,
            "warn_budget_threshold": 80,
            "stop_budget_threshold": 95,
            "disconnect_on_budget_exceed": False,
            "deployment_environment": "testing",
        }

        response = requests.post(
            f"{ruuter_private_client.base_url}/rag-search/llm-connections/add",
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Failed to create testing connection: {response.text}")

        data = response.json()
        response_data = data.get("response", data)
        connection_id = response_data["id"]

        logger.info(f"Created testing gpt-4o-mini connection with ID: {connection_id}")

        # Wait for database write
        time.sleep(2)

        # Pin the row's vault_uuid so it matches the vault secret path and the
        # value inference/test.yml forwards to /orchestrate/test.
        _pin_vault_uuid(postgres_client, connection_id)

        return connection_id

    finally:
        cursor.close()


def _pin_vault_uuid(postgres_client, connection_id: int) -> None:
    """Force a testing connection's vault_uuid to the pinned TEST_VAULT_UUID.

    Vault secret paths terminate in the connection's vault_uuid (no environment
    segment). We write the integration-test secrets under TEST_VAULT_UUID, so the
    DB row that inference/test.yml looks up must carry the same UUID for the
    forwarded value to resolve to the right vault path.
    """
    cursor = postgres_client.cursor()
    try:
        cursor.execute(
            "UPDATE llm_connections SET vault_uuid = %s WHERE id = %s",
            (TEST_VAULT_UUID, connection_id),
        )
        postgres_client.commit()
        logger.info(
            f"Pinned vault_uuid={TEST_VAULT_UUID} on testing connection id={connection_id}"
        )
    except Exception as e:
        postgres_client.rollback()
        logger.error(f"Failed to pin vault_uuid on connection {connection_id}: {e}")
        raise
    finally:
        cursor.close()


@pytest.fixture(scope="session", autouse=True)
def capture_container_logs_on_exit(rag_stack):
    """
    Capture Docker container logs at the end of the test session.

    This runs automatically after all tests complete but before testcontainers
    shuts down the Docker containers. Logs are printed to pytest output which
    appears in GitHub Actions logs.
    """
    yield  # Let all tests run first

    # After all tests complete, capture logs before containers are destroyed
    import subprocess

    logger.info("")
    logger.info("=" * 80)
    logger.info("CAPTURING CONTAINER LOGS BEFORE SHUTDOWN")
    logger.info("=" * 80)

    containers = [
        ("llm-orchestration-service", 500),
        ("ruuter", 200),
        ("resql", 200),
        ("qdrant", 100),
    ]

    for container_name, tail_lines in containers:
        try:
            logger.info("")
            logger.info(f"{'=' * 80}")
            logger.info(f"{container_name.upper()} LOGS (last {tail_lines} lines)")
            logger.info(f"{'=' * 80}")

            result = subprocess.run(
                ["docker", "logs", container_name, "--tail", str(tail_lines)],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.info("--- STDERR ---")
                logger.info(result.stderr)

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout while capturing logs from {container_name}")
        except Exception as e:
            logger.warning(f"Failed to capture logs from {container_name}: {e}")

    logger.info("LOG CAPTURE COMPLETE")

"""Tests for the Vault KV path prefix applied by VaultAgentClient.

The prefix is inserted at the client, not in SecretResolver, so that:
  - the Langfuse lookup, which calls get_secret() directly and bypasses the
    resolver, is covered without a second code path, and
  - resolver paths stay unprefixed, keeping the positional provider parsing in
    refresh_secret() correct.

Both properties are asserted here because breaking either fails silently in
production: a missed Langfuse prefix disables tracing, and a broken refresh
parser leaves cached secrets stale forever.
"""

import pytest

from src.llm_orchestrator_config.vault.vault_client import (
    VaultAgentClient,
    _resolve_path_prefix,
)
from src.llm_orchestrator_config.vault.secret_resolver import SecretResolver


def make_client(prefix):
    """Build a client without touching the network or a token file."""
    client = VaultAgentClient.__new__(VaultAgentClient)
    client.path_prefix = prefix
    return client


class TestResolvePathPrefix:
    def test_env_var_wins_over_constants_ini(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH_PREFIX", "llm-module")
        assert _resolve_path_prefix() == "llm-module"

    def test_surrounding_slashes_are_stripped(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH_PREFIX", "/llm-module/")
        assert _resolve_path_prefix() == "llm-module"

    def test_whitespace_only_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH_PREFIX", "   ")
        assert _resolve_path_prefix() == ""

    def test_empty_env_var_is_respected_not_ignored(self, monkeypatch):
        # An explicitly empty env var must mean "no prefix", not "fall through
        # to constants.ini" — otherwise a deployment cannot opt out.
        monkeypatch.setenv("VAULT_PATH_PREFIX", "")
        assert _resolve_path_prefix() == ""


class TestApplyPrefix:
    @pytest.mark.parametrize(
        "path",
        [
            "llm/connections/azure_openai/00000000-0000-0000-0000-000000000001",
            "embeddings/connections/aws_bedrock/abc-123",
            "langfuse/config",
            "encryption/public_key",
        ],
    )
    def test_no_prefix_leaves_path_untouched(self, path):
        assert make_client("")._apply_prefix(path) == path

    @pytest.mark.parametrize(
        "path",
        [
            "llm/connections/azure_openai/abc-123",
            "embeddings/connections/aws_bedrock/abc-123",
            "langfuse/config",
        ],
    )
    def test_prefix_is_prepended_once(self, path):
        result = make_client("llm-module")._apply_prefix(path)
        assert result == f"llm-module/{path}"
        assert result.count("llm-module") == 1

    def test_no_double_slash_when_path_has_leading_slash(self):
        result = make_client("llm-module")._apply_prefix("/langfuse/config")
        assert result == "llm-module/langfuse/config"
        assert "//" not in result


class TestResolverStaysUnprefixed:
    """The resolver must keep building mount-relative, unprefixed paths."""

    def test_build_vault_path_has_no_prefix(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH_PREFIX", "llm-module")
        resolver = SecretResolver.__new__(SecretResolver)
        path = SecretResolver._build_vault_path(
            resolver, "azure_openai", "production", "gpt-4o", connection_id="abc-123"
        )
        assert path == "llm/connections/azure_openai/abc-123"
        assert "llm-module" not in path

    def test_refresh_parser_still_finds_provider(self, monkeypatch):
        # refresh_secret() reads the provider at path_parts[2]. If a prefix ever
        # leaks into resolver paths this lands on "connections", the resulting
        # exception is swallowed, and secrets silently stop refreshing.
        monkeypatch.setenv("VAULT_PATH_PREFIX", "llm-module")
        resolver = SecretResolver.__new__(SecretResolver)
        path = SecretResolver._build_vault_path(
            resolver, "aws_bedrock", "production", "claude-3", connection_id="abc-123"
        )
        assert path.split("/")[2] == "aws_bedrock"

    def test_embedding_path_has_no_prefix(self, monkeypatch):
        monkeypatch.setenv("VAULT_PATH_PREFIX", "llm-module")
        resolver = SecretResolver.__new__(SecretResolver)
        path = SecretResolver._build_embedding_vault_path(
            resolver, "azure_openai", "production", "text-embedding-3-large", "abc-123"
        )
        assert path == "embeddings/connections/azure_openai/abc-123"

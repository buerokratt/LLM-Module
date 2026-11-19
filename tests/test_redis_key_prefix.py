"""Every store key must carry REDIS_KEY_PREFIX (risk R8: no mixed keyspace)."""

import pytest
import tool_classifier  # noqa: F401

from src.utils.api_tool_session_store import _key
from src.utils.conversation_history_store import _history_key, _summary_key
from utils.atc_cache_store import ATCCacheStore


@pytest.mark.parametrize("prefix", ["", "rag:"])
def test_every_store_key_starts_with_prefix(monkeypatch, prefix):
    monkeypatch.setenv("REDIS_KEY_PREFIX", prefix)
    keys = [
        _key("chat-1"),
        _history_key("chat-1"),
        _summary_key("chat-1"),
        ATCCacheStore._l1_key("chat-1", "api", {"a": 1}),
        ATCCacheStore._l2_key("chat-1"),
    ]
    assert all(key.startswith(prefix) for key in keys)


def test_default_prefix_keeps_existing_keys(monkeypatch):
    monkeypatch.delenv("REDIS_KEY_PREFIX", raising=False)
    assert _key("abc") == "session:abc"
    assert _history_key("abc") == "conv:abc"


def test_prefix_is_prepended(monkeypatch):
    monkeypatch.setenv("REDIS_KEY_PREFIX", "rag:")
    assert _key("abc") == "rag:session:abc"
    assert _summary_key("abc") == "rag:conv:summary:abc"

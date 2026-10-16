"""One boundary for Central KV paths; resolver caches use logical paths."""
import os
import re


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or any(
        not re.fullmatch(r"[A-Za-z0-9_.-]+", segment) or segment in (".", "..")
        for segment in value.split("/")
    ):
        raise ValueError("Invalid relative Vault path")
    return value


def settings() -> tuple[str, str]:
    prefix = relative_path(os.environ.get("VAULT_PATH_PREFIX", ""))
    mount = relative_path(os.environ.get("VAULT_KV_MOUNT", "secret"))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in prefix.split("/")) or not re.fullmatch(r"[A-Za-z0-9_-]+", mount):
        raise ValueError("KV mount must be one segment")
    return prefix, mount


def logical_path(value: str, prefix: str) -> str:
    value = relative_path(value)
    if value.startswith(prefix + "/"):
        value = value[len(prefix) + 1:]
    # Accept one physical prefix for refresh calls, but never silently double it.
    if value == prefix or value.startswith(prefix + "/"):
        raise ValueError("Repeated Vault prefix")
    return relative_path(value)


def physical_path(value: str, prefix: str) -> str:
    return prefix + "/" + logical_path(value, prefix)

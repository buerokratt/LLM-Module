"""Log Dragonfly metrics (from INFO) to Loki every minute.

Dragonfly runs with ``maxmemory_policy: noeviction``: once ``used_memory`` reaches
``maxmemory`` it refuses writes. The Grafana alerts ``dragonfly-memory-*`` read
``memory_used_pct`` and ``dragonfly-restarted`` reads ``uptime_seconds``; the
"Dragonfly" dashboard graphs every field logged here.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from redis import Redis as DragonflyClient, RedisError as DragonflyError

# /app holds the mounted src package (src/__init__.py, src/loki_logger.py)
sys.path.insert(0, "/app")
from src.loki_logger import LokiLogger  # noqa: E402

ENV_FILE = Path("/app/.env")

logger = LokiLogger(service_name="dragonfly-metrics")


def load_env_file(path: Path) -> dict[str, str]:
    """Read KEY=VALUE lines from the mounted .env file."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def collect_metrics(info: dict[str, Any]) -> dict[str, float | int]:
    """Pick the fields we monitor from an INFO reply."""
    used = int(info.get("used_memory", 0))
    maximum = int(info.get("maxmemory", 0))
    keys = sum(
        stats.get("keys", 0)
        for name, stats in info.items()
        if name.startswith("db") and isinstance(stats, dict)
    )
    return {
        "used_memory": used,
        "maxmemory": maximum,
        "memory_used_pct": round(used / maximum * 100, 2) if maximum else 0,
        "used_memory_rss": int(info.get("used_memory_rss", 0)),
        "connected_clients": int(info.get("connected_clients", 0)),
        "ops_per_sec": int(info.get("instantaneous_ops_per_sec", 0)),
        "keys": keys,
        "uptime_seconds": int(info.get("uptime_in_seconds", 0)),
    }


def main() -> int:
    env = load_env_file(ENV_FILE)

    def setting(name: str, default: str) -> str:
        return os.getenv(name) or env.get(name) or default

    client = DragonflyClient(
        host=setting("DRAGONFLY_HOST", "dragonfly"),
        port=int(setting("DRAGONFLY_PORT", "6379")),
        password=setting("DRAGONFLY_PASSWORD", "") or None,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        # Sync client: info() returns a plain dict (the stubs also allow an awaitable)
        metrics = collect_metrics(cast(dict[str, Any], client.info()))
    except DragonflyError as exc:
        logger.error(f"Dragonfly metrics check failed: {exc}")
        logger.log_queue.join()
        return 1

    logger.info(json.dumps(metrics))

    # LokiLogger sends from a background thread; wait so the log is not lost on exit
    logger.log_queue.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())

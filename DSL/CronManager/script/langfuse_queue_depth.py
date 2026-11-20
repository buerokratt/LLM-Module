"""Log Langfuse (BullMQ) queue depth from Dragonfly to Loki.

Runs every minute from CronManager. Langfuse keeps its job queues in db 0 under
``bull:<queue>:*``; jobs normally leave the wait list within seconds, so a waiting
count that never returns to 0 means the Langfuse worker is falling behind.
Grafana alert ``langfuse-queue-backlog`` reads the ``waiting_total`` field.
"""

import json
import os
import sys
from pathlib import Path
from typing import cast

from redis import Redis as DragonflyClient, RedisError as DragonflyError

# /app holds the mounted src package (src/__init__.py, src/loki_logger.py)
sys.path.insert(0, "/app")
from src.loki_logger import LokiLogger  # noqa: E402

ENV_FILE = Path("/app/.env")
LANGFUSE_DB = 0

logger = LokiLogger(service_name="langfuse-queue-depth")


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


def queue_depths(client: DragonflyClient) -> dict[str, dict[str, int]]:
    """Return waiting/active/delayed job counts for every BullMQ queue."""
    depths: dict[str, dict[str, int]] = {}
    for meta_key in client.scan_iter(match="bull:*:meta", count=200):
        queue = meta_key[len("bull:") : -len(":meta")]
        prefix = f"bull:{queue}"
        # Sync client: these calls return plain ints (the stubs also allow awaitables)
        depths[queue] = {
            "waiting": cast(int, client.llen(f"{prefix}:wait"))
            + cast(int, client.zcard(f"{prefix}:prioritized")),
            "active": cast(int, client.llen(f"{prefix}:active")),
            "delayed": cast(int, client.zcard(f"{prefix}:delayed")),
        }
    return depths


def main() -> int:
    env = load_env_file(ENV_FILE)

    def setting(name: str, default: str) -> str:
        return os.getenv(name) or env.get(name) or default

    client = DragonflyClient(
        host=setting("DRAGONFLY_HOST", "dragonfly"),
        port=int(setting("DRAGONFLY_PORT", "6379")),
        password=setting("DRAGONFLY_PASSWORD", "") or None,
        db=LANGFUSE_DB,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        depths = queue_depths(client)
    except DragonflyError as exc:
        logger.error(f"Langfuse queue depth check failed: {exc}")
        logger.log_queue.join()
        return 1

    summary = {
        "queues": len(depths),
        "waiting_total": sum(d["waiting"] for d in depths.values()),
        "active_total": sum(d["active"] for d in depths.values()),
        "delayed_total": sum(d["delayed"] for d in depths.values()),
        # Only queues with work in them, to keep the log line short
        "busy_queues": {
            name: d for name, d in sorted(depths.items()) if d["waiting"] or d["active"]
        },
    }
    logger.info(json.dumps(summary))

    # LokiLogger sends from a background thread; wait so the log is not lost on exit
    logger.log_queue.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())

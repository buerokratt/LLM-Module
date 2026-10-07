#!/usr/bin/env python3
"""
Loki Logger for RAG Module
Sends logs directly to Loki API for centralized logging
"""

import configparser
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from threading import Thread
from queue import Full, Queue
from typing import Optional

import requests


def _read_loki_url_from_constants() -> Optional[str]:
    """Read LOKI_URL from the [DSL] section of the nearest constants.ini.

    Searches upward from this file so every copy of the logger finds the
    project-root constants.ini wherever it is mounted (e.g. /app/constants.ini).
    Returns None when no file or key is found.
    """
    for directory in Path(__file__).resolve().parents:
        ini_path = directory / "constants.ini"
        if ini_path.is_file():
            parser = configparser.ConfigParser(interpolation=None, strict=False)
            parser.optionxform = str  # type: ignore[assignment,method-assign]
            try:
                parser.read(ini_path, encoding="utf-8")
            except configparser.Error:
                # Not our INI format (e.g. CronManager's own /app/constants.ini
                # has no section header) - never let logging break the caller.
                return None
            return parser.get("DSL", "LOKI_URL", fallback="").strip() or None
    return None


_CONFIGURED_LOKI_URL = _read_loki_url_from_constants()


class LokiLogger:
    """Simple logger that sends logs directly to Loki API with async background thread"""

    _instances: dict[str, "LokiLogger"] = {}

    def __new__(
        cls, loki_url: Optional[str] = None, service_name: str = "default"
    ) -> "LokiLogger":
        key = f"{loki_url or _CONFIGURED_LOKI_URL}:{service_name}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self, loki_url: Optional[str] = None, service_name: str = "default"
    ) -> None:
        """
        Initialize LokiLogger

        Args:
            loki_url: URL for Loki service (default: LOKI_URL from constants.ini).
                If neither is available, logs go to the console only.
            service_name: Name of the service for labeling logs
        """
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.loki_url = loki_url or _CONFIGURED_LOKI_URL
        if not self.loki_url:
            print(  # noqa: T201
                f"LokiLogger[{service_name}]: LOKI_URL not found in constants.ini; "
                "logging to console only",
                file=sys.stderr,
            )
        self.service_name = service_name
        self.session = requests.Session()
        # Set default timeout for all requests
        self.timeout = 5

        # Queue for async log processing (bounded to avoid unbounded memory growth under load)
        self.log_queue: Queue[tuple[str, str]] = Queue(maxsize=10_000)

        # Start background worker thread
        self.worker_thread = Thread(target=self._process_logs, daemon=True)
        self.worker_thread.start()

    def _process_logs(self) -> None:
        """Background worker that processes log queue"""
        while True:
            try:
                # Get log entry from queue (blocking)
                level, message = self.log_queue.get()

                # Send to Loki
                self._send_to_loki_sync(level, message)

                # Mark task as done
                self.log_queue.task_done()
            except Exception:
                # Silently ignore errors in background thread
                pass

    def _send_to_loki_sync(self, level: str, message: str) -> None:
        """Send log entry directly to Loki API (called from background thread)"""
        try:
            # Create timestamp in nanoseconds (Loki requirement)
            timestamp_ns = str(int(time.time() * 1_000_000_000))

            # Prepare labels for Loki
            labels = {
                "service": self.service_name,
                "level": level,
            }

            # Create log entry
            log_entry = {
                "level": level,
                "message": message,
                "service": self.service_name,
            }

            # Prepare Loki payload
            payload = {
                "streams": [
                    {
                        "stream": labels,
                        "values": [[timestamp_ns, json.dumps(log_entry)]],
                    }
                ]
            }

            # Send to Loki
            self.session.post(
                f"{self.loki_url}/loki/api/v1/push",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

        except Exception:
            # Silently ignore logging errors to not affect main application
            pass

    def _log(self, level: str, message: str) -> None:
        """Queue log entry for async processing (non-blocking)"""
        # Print to console immediately for real-time feedback. Written to
        # stderr (not stdout) so callers that capture a subprocess's stdout
        # for its return value (e.g. decrypt_vault_secrets.py) never pick up
        # log lines mixed in with the actual output.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level: <8} | {message}", file=sys.stderr)  # noqa: T201

        if not self.loki_url:
            return

        # Queue for async Loki sending (non-blocking)
        try:
            self.log_queue.put_nowait((level, message))
        except Full:
            # Queue full (Loki may be slow/unreachable) - drop log to avoid blocking
            pass

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("INFO", message)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("ERROR", message)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("WARNING", message)

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("DEBUG", message)

    def success(self, message: str, **kwargs: object) -> None:
        """Log success message (loguru compatibility). Extra kwargs ignored."""
        self._log("SUCCESS", message)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("CRITICAL", message)

    def exception(self, message: str, **kwargs: object) -> None:
        """Log exception message. Extra kwargs (extra, exc_info) are ignored for compatibility."""
        self._log("EXCEPTION", message)

    def add(self, *args: object, **kwargs: object) -> None:
        """
        No-op method for loguru compatibility.

        LokiLogger sends logs to Loki/console only, not to files.
        This method exists for backward compatibility with loguru code.
        """
        pass  # Silently ignore - logs go to Loki instead of files

    def remove(self, *args: object, **kwargs: object) -> None:
        """No-op method for loguru compatibility."""
        pass  # Silently ignore

    def bind(self, **kwargs: object) -> "LokiLogger":
        """No-op method for loguru compatibility. Returns self for chaining."""
        return self  # Allow method chaining

    def opt(self, **kwargs: object) -> "LokiLogger":
        """No-op method for loguru compatibility. Returns self for chaining."""
        return self  # Allow method chaining

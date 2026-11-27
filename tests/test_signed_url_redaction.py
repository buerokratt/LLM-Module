"""The presigned CKB dataset URL never reaches the logs (review finding H2).

Its query string carries the S3 credential and signature: anyone who can read
the logs could download the CKB dataset until the URL expires.
vector_indexer_pipeline.sh already redacts it; the indexer must as well.
"""

import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import requests

# Like cron-manager's PYTHONPATH; appended so src/vector_indexer/models.py does
# not shadow the src/models package.
sys.path.append(str(Path(__file__).resolve().parent.parent / "src" / "vector_indexer"))

from vector_indexer import dataset_download, main_indexer  # noqa: E402
from vector_indexer.dataset_download import redact_url  # noqa: E402

SIGNATURE = "c0ffee5ignature"
SIGNED_URL = (
    "http://ckb-seaweedfs:8333/ckb/zipped/5eb19249.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=ckbaccesskey%2F20261127%2Fus-east-1%2Fs3%2Faws4_request"
    f"&X-Amz-Date=20261127T053800Z&X-Amz-Expires=604800&X-Amz-Signature={SIGNATURE}"
)


class RecordingLogger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def __getattr__(self, _level: str):
        return lambda message, *a, **k: self.lines.append(str(message))


def test_redact_url_keeps_the_path_and_drops_the_query() -> None:
    redacted = redact_url(SIGNED_URL)

    assert redacted == "http://ckb-seaweedfs:8333/ckb/zipped/5eb19249.zip?<redacted>"
    assert SIGNATURE not in redacted
    assert "ckbaccesskey" not in redacted


def test_redact_url_handles_urls_inside_messages() -> None:
    message = f"404 Client Error: Not Found for url: {SIGNED_URL}"

    assert SIGNATURE not in redact_url(message)
    assert redact_url(message).startswith(
        "404 Client Error: Not Found for url: http://"
    )


@pytest.mark.parametrize("busy", [False, True])
def test_lock_logs_do_not_contain_the_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy: bool
) -> None:
    log = RecordingLogger()
    fcntl = MagicMock()
    fcntl.flock.side_effect = OSError("busy") if busy else None
    monkeypatch.setattr(main_indexer, "logger", log)
    monkeypatch.setattr(main_indexer, "fcntl", fcntl)
    monkeypatch.setattr(main_indexer, "INDEXER_LOCK_DIR", tmp_path)

    handle = main_indexer._acquire_indexer_lock(SIGNED_URL)

    assert (handle is None) is busy
    if handle:
        handle.close()
    assert log.lines and all(SIGNATURE not in line for line in log.lines)
    assert any("5eb19249.zip" in line for line in log.lines)


def test_download_logs_and_errors_do_not_contain_the_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = RecordingLogger()
    monkeypatch.setattr(dataset_download, "logger", log)
    response = requests.Response()
    response.status_code = 403
    response.url = SIGNED_URL

    with patch.object(dataset_download.requests, "get", return_value=response):
        with pytest.raises(requests.exceptions.HTTPError) as raised:
            dataset_download.download_and_extract_dataset(SIGNED_URL, tmp_path)

    assert SIGNATURE not in str(raised.value)
    # Not chained: the original message (with the URL) is not in the traceback
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    assert all(SIGNATURE not in line for line in log.lines)


def test_connection_errors_are_redacted_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset_download, "logger", RecordingLogger())
    failure = requests.ConnectionError(f"Max retries exceeded with url: {SIGNED_URL}")

    with patch.object(dataset_download.requests, "get", side_effect=failure):
        with pytest.raises(requests.RequestException) as raised:
            dataset_download.download_and_extract_dataset(SIGNED_URL, tmp_path)

    assert SIGNATURE not in str(raised.value)

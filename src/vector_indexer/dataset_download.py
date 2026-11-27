"""Simple dataset download utility using requests."""

import re
import shutil
import zipfile
import tempfile
from pathlib import Path
import requests
from loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="dataset-download")

DATASETS_PATH = Path("/app/datasets")

# The query string of a presigned URL carries the S3 credential and signature.
_URL_QUERY = re.compile(r"(https?://[^\s?'\"<>]+)\?[^\s'\"<>]*")


def redact_url(text: str) -> str:
    """Replace the query string of every URL in ``text`` with ``<redacted>``.

    Presigned dataset URLs must never reach the logs: anyone who can read them
    could download the CKB dataset until the URL expires.
    """
    return _URL_QUERY.sub(r"\1?<redacted>", text or "")


def _empty_folder(folder: Path) -> int:
    """Remove a folder's contents (not the folder: it is a mounted volume)."""
    removed = 0
    for child in folder.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
        removed += 1
    return removed


def download_and_extract_dataset(
    signed_url: str, datasets_path: Path = DATASETS_PATH
) -> tuple[str, int]:
    """
    Download ZIP from signed URL and extract it to datasets folder.

    The folder is emptied first: files left by a failed or interrupted run
    must not be indexed with the new dataset (issue #588: a retried re-index
    must embed exactly the current content, not documents CKB has removed).

    Args:
        signed_url: URL to download ZIP from
        datasets_path: Folder to extract into (the indexer's datasets volume)

    Returns:
        tuple: (extraction_path, files_extracted_count)

    Raises:
        requests.RequestException: If download fails
        zipfile.BadZipFile: If ZIP file is corrupted
        IOError: If extraction fails
    """
    if not signed_url:
        raise ValueError("signed_url cannot be empty")

    logger.info("Starting dataset download...")
    logger.debug(f"Download URL: {redact_url(signed_url)}")

    # Create datasets folder
    datasets_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Dataset directory ready: {datasets_path}")

    # Download ZIP to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
        temp_zip_path = Path(temp_file.name)

    try:
        # Download file with progress logging
        logger.info("Downloading ZIP file...")
        response = requests.get(
            signed_url, stream=True, timeout=300, allow_redirects=True
        )
        response.raise_for_status()

        # Write to temp file
        with open(temp_zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        file_size_mb = temp_zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Downloaded {file_size_mb:.1f} MB")

        # Only after a successful download, so a failed one changes nothing
        leftovers = _empty_folder(datasets_path)
        if leftovers:
            logger.warning(
                f"Removed {leftovers} leftover entries from {datasets_path} "
                "(an earlier run did not finish) before extracting the new dataset"
            )

        # Extract ZIP
        logger.info("Extracting files...")
        files_count = 0
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            files_count = len(zip_ref.namelist())
            zip_ref.extractall(datasets_path)

        logger.info(f"Extracted {files_count} files to {datasets_path}")
        logger.info("Cleaning up temporary files...")

        return str(datasets_path), files_count

    # requests puts the full URL (with its signature) in error messages, and
    # callers log them with their traceback: re-raise without the URL's query
    # and without chaining the original exception.
    except requests.exceptions.HTTPError as e:
        logger.error(f"Download failed with HTTP error {e.response.status_code}")
        raise requests.exceptions.HTTPError(
            redact_url(str(e)), response=e.response
        ) from None
    except requests.exceptions.Timeout:
        logger.error("Download timed out after 300 seconds")
        raise requests.exceptions.Timeout(
            "Dataset download timed out after 300 seconds"
        ) from None
    except requests.RequestException as e:
        message = redact_url(str(e))
        logger.error(f"Download request failed: {message}")
        try:
            redacted = type(e)(message)  # same type (ConnectionError, ...)
        except Exception:
            redacted = requests.RequestException(message)
        raise redacted from None
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid or corrupted ZIP file: {e}")
        raise
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise
    finally:
        # Always clean up temp file
        if temp_zip_path.exists():
            try:
                temp_zip_path.unlink()
                logger.debug("Temporary ZIP file cleaned up")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file: {e}")

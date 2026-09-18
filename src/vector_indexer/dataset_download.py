"""Simple dataset download utility using requests."""

import zipfile
import tempfile
from pathlib import Path
import requests
from loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="dataset-download")


def download_and_extract_dataset(signed_url: str) -> tuple[str, int]:
    """
    Download ZIP from signed URL and extract it to datasets folder.

    Args:
        signed_url: URL to download ZIP from

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
    logger.debug(f"Download URL (first 100 chars): {signed_url[:100]}...")

    # Create datasets folder
    datasets_path = Path("/app/datasets")
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

        # Extract ZIP
        logger.info("Extracting files...")
        files_count = 0
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            files_count = len(zip_ref.namelist())
            zip_ref.extractall(datasets_path)

        logger.info(f"Extracted {files_count} files to {datasets_path}")
        logger.info("Cleaning up temporary files...")

        return str(datasets_path), files_count

    except requests.exceptions.HTTPError as e:
        logger.error(f"Download failed with HTTP error {e.response.status_code}")
        raise
    except requests.exceptions.Timeout:
        logger.error("Download timed out after 300 seconds")
        raise
    except requests.RequestException as e:
        logger.error(f"Download request failed: {e}")
        raise
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

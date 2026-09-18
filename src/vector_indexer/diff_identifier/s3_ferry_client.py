"""S3Ferry client for file transfer operations."""

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional
import requests
from loki_logger import LokiLogger

from typing_extensions import Self

from diff_identifier.diff_models import DiffConfig, DiffError
from constants import get_s3_ferry_payload

# Initialize Loki logger
logger = LokiLogger(service_name="s3-ferry-client")


class S3Ferry:
    """Client for interacting with S3Ferry service."""

    def __init__(self, url: str) -> None:
        self.url = url

    def transfer_file(
        self,
        destination_file_path: str,
        destination_storage_type: str,
        source_file_path: str,
        source_storage_type: str,
    ) -> requests.Response:
        """
        Transfer file using S3Ferry service.

        Args:
            destination_file_path: Path where file should be stored
            destination_storage_type: "S3" or "FS" (filesystem)
            source_file_path: Path of source file
            source_storage_type: "S3" or "FS" (filesystem)

        Returns:
            requests.Response: Response from S3Ferry service
        """
        payload = get_s3_ferry_payload(
            destination_file_path,
            destination_storage_type,
            source_file_path,
            source_storage_type,
        )

        response = requests.post(self.url, json=payload)

        return response


class S3FerryClient:
    """High-level client for S3Ferry operations with metadata handling.

    S3Ferry service handles all S3 configuration internally.
    This client only needs to know the S3Ferry URL and metadata paths.
    """

    def __init__(self, config: DiffConfig) -> None:
        self.config = config
        self.s3_ferry = S3Ferry(config.s3_ferry_url)

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        """Async context manager exit."""
        pass

    async def upload_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Upload metadata to S3 via S3Ferry.

        Args:
            metadata: Dictionary containing metadata to upload

        Returns:
            True if successful, False otherwise

        Raises:
            DiffError: If upload fails
        """
        try:
            # Create temporary file with metadata (run in thread pool)
            temp_file_path = await asyncio.to_thread(
                self._create_temp_metadata_file, metadata
            )

            try:
                # Transfer from FS to S3 using S3Ferry (run in thread pool)
                # Convert absolute path to S3Ferry-relative path
                s3ferry_source_path = self._convert_to_s3ferry_path(temp_file_path)

                response = await asyncio.to_thread(
                    self._retry_with_backoff,
                    lambda: self.s3_ferry.transfer_file(
                        destination_file_path=self.config.metadata_s3_path,
                        destination_storage_type="S3",
                        source_file_path=s3ferry_source_path,
                        source_storage_type="FS",
                    ),
                )

                if response.status_code in [
                    200,
                    201,
                ]:  # Accept both 200 OK and 201 Created
                    logger.info(
                        f"Metadata uploaded successfully to {self.config.metadata_s3_path} (status: {response.status_code})"
                    )
                    return True
                else:
                    logger.error(
                        f"S3Ferry upload failed: {response.status_code} - {response.text}"
                    )
                    return False

            finally:
                # Clean up temporary file (run in thread pool)
                await asyncio.to_thread(self._cleanup_temp_file, temp_file_path)

        except Exception as e:
            raise DiffError(f"Failed to upload metadata: {str(e)}", e) from e

    async def download_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Download metadata from S3 via S3Ferry.

        Returns:
            Dict containing metadata or None if not found

        Raises:
            DiffError: If download fails (except for file not found)
        """
        try:
            # Create temporary file for download (run in thread pool)
            temp_file_path = await asyncio.to_thread(self._create_temp_file)

            try:
                # Transfer from S3 to FS using S3Ferry (run in thread pool)
                # Convert absolute path to S3Ferry-relative path
                s3ferry_dest_path = self._convert_to_s3ferry_path(temp_file_path)

                response = await asyncio.to_thread(
                    self._retry_with_backoff,
                    lambda: self.s3_ferry.transfer_file(
                        destination_file_path=s3ferry_dest_path,
                        destination_storage_type="FS",
                        source_file_path=self.config.metadata_s3_path,
                        source_storage_type="S3",
                    ),
                )

                if response.status_code in [
                    200,
                    201,
                ]:  # Accept both 200 OK and 201 Created
                    # Read metadata from downloaded file (run in thread pool)
                    metadata = await asyncio.to_thread(
                        self._read_metadata_from_file, temp_file_path
                    )
                    logger.info(
                        f"Metadata downloaded successfully from {self.config.metadata_s3_path} (status: {response.status_code})"
                    )
                    return metadata
                elif response.status_code == 404:
                    logger.info(
                        "No previous metadata found - this appears to be the first run"
                    )
                    return None
                else:
                    logger.error(
                        f"S3Ferry download failed: {response.status_code} - {response.text}"
                    )
                    return None

            finally:
                # Clean up temporary file (run in thread pool)
                await asyncio.to_thread(self._cleanup_temp_file, temp_file_path)

        except json.JSONDecodeError as e:
            raise DiffError(
                f"Failed to parse downloaded metadata JSON: {str(e)}", e
            ) from e
        except Exception as e:
            # Don't raise for file not found - it's expected on first run
            logger.warning(f"Failed to download metadata (may be first run): {str(e)}")
            return None

    def _create_temp_metadata_file(self, metadata: Dict[str, Any]) -> str:
        """Create a temporary file with metadata content in shared folder."""
        import os
        import uuid

        # Create temp file in shared folder accessible by both containers
        shared_dir = "/app/shared"
        os.makedirs(shared_dir, exist_ok=True)

        temp_filename = f"temp_metadata_{uuid.uuid4().hex[:8]}.json"
        temp_file_path = os.path.join(shared_dir, temp_filename)

        with open(temp_file_path, "w") as temp_file:
            json.dump(metadata, temp_file, indent=2)

        # Set broad permissions so S3Ferry can read the file
        os.chmod(temp_file_path, 0o666)  # rw-rw-rw-

        return temp_file_path

    def _create_temp_file(self) -> str:
        """Create an empty temporary file in shared folder."""
        import os
        import uuid

        # Create temp file in shared folder accessible by both containers
        shared_dir = "/app/shared"
        os.makedirs(shared_dir, exist_ok=True)

        temp_filename = f"temp_download_{uuid.uuid4().hex[:8]}.json"
        temp_file_path = os.path.join(shared_dir, temp_filename)

        # Create empty file
        with open(temp_file_path, "w"):
            pass  # Create empty file

        # Set broad permissions so S3Ferry can write to the file
        os.chmod(temp_file_path, 0o666)  # rw-rw-rw-

        return temp_file_path

    def _read_metadata_from_file(self, file_path: str) -> Dict[str, Any]:
        """Read metadata from a file."""
        with open(file_path, "r") as f:
            return json.load(f)

    def _convert_to_s3ferry_path(self, absolute_path: str) -> str:
        """Convert absolute path to S3Ferry-relative path.

        S3Ferry expects paths relative to /app/ working directory.
        Converts: /app/shared/filename.json -> shared/filename.json
        """
        if absolute_path.startswith("/app/"):
            return absolute_path[5:]  # Remove '/app/' prefix
        return absolute_path

    def _cleanup_temp_file(self, file_path: str) -> None:
        """Clean up a temporary file."""
        import os

        try:
            os.unlink(file_path)
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup temp file {file_path}: {cleanup_error}")

    def _retry_with_backoff(
        self, operation: Callable[[], requests.Response]
    ) -> requests.Response:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: Operation to retry (callable that returns Response)

        Returns:
            Response from the operation

        Raises:
            DiffError: If all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                response = operation()

                # Consider non-2xx responses as failures for retry purposes
                if response.status_code >= 400:
                    if attempt == self.config.max_retries - 1:
                        return response  # Last attempt - return the error response

                    delay = min(1 * (2**attempt), self.config.max_delay_seconds)
                    time.sleep(delay)
                    continue

                return response

            except Exception as e:
                last_exception = e

                if attempt == self.config.max_retries - 1:
                    raise DiffError(
                        f"Operation failed after {self.config.max_retries} attempts: {str(e)}",
                        e,
                    ) from e

                delay = min(1 * (2**attempt), self.config.max_delay_seconds)
                time.sleep(delay)

        raise DiffError(
            f"Operation failed after {self.config.max_retries} attempts: {str(last_exception)}",
            last_exception,
        )

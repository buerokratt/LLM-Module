"""Main diff detector for identifying changed files."""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from loki_logger import LokiLogger
import hashlib

from diff_identifier.diff_models import DiffConfig, DiffError, DiffResult
from diff_identifier.version_manager import VersionManager
from dotenv import load_dotenv

load_dotenv(".env")

# Initialize Loki logger
logger = LokiLogger(service_name="diff-detector")


class DiffDetector:
    """Main orchestrator for diff identification."""

    def __init__(self, config: DiffConfig) -> None:
        self.config = config
        self.version_manager = VersionManager(config)

    async def get_changed_files(self) -> DiffResult:
        """
        Get list of files that need processing.

        Returns:
            DiffResult with files to process and metadata

        Raises:
            DiffError: If diff detection fails critically
        """
        try:
            logger.info("Starting diff identification process...")

            # First, check for existing processed files metadata (this is the source of truth)
            logger.info("Checking for existing processed files metadata...")
            processed_state = await self.version_manager.get_processed_files_metadata()

            # Initialize DVC if needed (but don't rely on it for first-run detection)
            if not self.version_manager.is_dvc_initialized():
                logger.info("DVC not initialized - initializing now")
                await self.version_manager.initialize_dvc()

            # Scan current files
            logger.info("Scanning current dataset files...")
            current_files = self.version_manager.scan_current_files()

            if not current_files:
                logger.info("No files found in datasets directory")
                return DiffResult(
                    new_files=[],
                    total_files_scanned=0,
                    previously_processed_count=0
                    if processed_state is None
                    else processed_state.total_processed,
                    is_first_run=processed_state is None,
                )

            # Determine if this is truly a first run based on metadata existence
            if processed_state is None:
                logger.info("No previous metadata found - this is a first run")
                return DiffResult(
                    new_files=list(current_files.values()),
                    total_files_scanned=len(current_files),
                    previously_processed_count=0,
                    is_first_run=True,
                )

            # This is an incremental run - identify all types of changes
            logger.info(
                f"Previous metadata found with {processed_state.total_processed} processed files"
            )
            changes = self.version_manager.identify_comprehensive_changes(
                current_files, processed_state
            )

            result = DiffResult(
                new_files=changes["new_files"],
                modified_files=changes["modified_files"],
                deleted_files=changes["deleted_files"],
                unchanged_files=changes["unchanged_files"],
                total_files_scanned=len(current_files),
                previously_processed_count=processed_state.total_processed,
                is_first_run=False,
                chunks_to_delete=changes["chunks_to_delete"],
                estimated_cleanup_count=changes["estimated_cleanup_count"],
            )

            logger.info(
                f"Diff identification complete: {len(result.new_files)} files need processing"
            )
            return result

        except Exception as e:
            # Log error but don't fail - fall back to processing all files
            logger.error(f"Diff identification failed: {e}")
            logger.info("Falling back to processing all files as safety measure")

            try:
                # Get all files as fallback
                current_files = self.version_manager.scan_current_files()
                return DiffResult(
                    new_files=list(current_files.values()),
                    total_files_scanned=len(current_files),
                    previously_processed_count=0,
                    is_first_run=True,
                )
            except Exception as fallback_error:
                raise DiffError(
                    f"Both diff identification and fallback failed: {fallback_error}", e
                ) from fallback_error

    async def mark_files_processed(
        self,
        processed_file_paths: List[str],
        force_metadata_update: bool = False,
        chunks_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Mark files as successfully processed and update metadata.

        Args:
            processed_file_paths: List of file paths that were processed successfully
            force_metadata_update: Force metadata update even if no new files processed (for cleanup operations)
            chunks_info: Optional dict mapping document_hash to {"chunk_count": int}

        Raises:
            DiffError: If marking files fails
        """
        try:
            # Handle cleanup-only scenarios (no new files processed)
            if not processed_file_paths and force_metadata_update:
                logger.info(
                    "Updating metadata for cleanup operations (no new files processed)..."
                )
                await self.version_manager.update_processed_files_metadata({})
                logger.info("Metadata updated for cleanup operations")
                return

            if not processed_file_paths:
                logger.info("No files to mark as processed")
                return

            logger.info(f"Marking {len(processed_file_paths)} files as processed...")

            # Log chunks_info summary only (avoid massive logs)
            if chunks_info:
                total_chunks = sum(
                    info.get("chunk_count", 0) for info in chunks_info.values()
                )
                logger.info(
                    f"RECEIVED CHUNKS INFO: {len(chunks_info)} documents, {total_chunks} total chunks"
                )
            else:
                logger.warning("No chunks_info provided to mark_files_processed")

                # Calculate hashes for processed files
            processed_files: Dict[str, str] = {}
            for file_path in processed_file_paths:
                try:
                    full_path = Path(file_path)
                    if full_path.exists():
                        # IMPORTANT: Read file exactly the same way as document_loader.py
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()  # Match document_loader exactly

                        file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                        processed_files[file_hash] = file_path
                        logger.debug(
                            f"PROCESSED FILE: {file_path} -> {file_hash[:12]}..."
                        )

                        # Debug: Check if this file_hash exists in chunks_info
                        if chunks_info and file_hash in chunks_info:
                            chunk_count = chunks_info[file_hash].get("chunk_count", 0)
                            logger.info(
                                f"MATCHED CHUNK INFO: {file_hash[:12]}... has {chunk_count} chunks"
                            )
                        elif chunks_info:
                            logger.warning(
                                f"NO MATCH: {file_hash[:12]}... not found in chunks_info"
                            )
                            logger.info(
                                f"   Available chunks_info keys: {[k[:12] + '...' for k in chunks_info.keys()]}"
                            )

                    else:
                        logger.warning(f"Processed file not found: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to hash processed file {file_path}: {e}")

            if not processed_files:
                logger.warning("No valid processed files to record")
                return

            # Update metadata
            await self.version_manager.update_processed_files_metadata(
                processed_files, chunks_info
            )

            # Commit to DVC if initialized
            if self.version_manager.is_dvc_initialized():
                await self.version_manager.commit_dvc_changes()

            logger.info(
                f"Successfully marked {len(processed_files)} files as processed"
            )

        except Exception as e:
            raise DiffError(f"Failed to mark files as processed: {str(e)}", e) from e

    async def _handle_first_run(self) -> DiffResult:
        """
        Handle first run setup.

        Returns:
            DiffResult for first run

        Raises:
            DiffError: If first run setup fails
        """
        try:
            logger.info("Setting up DVC for first run...")

            # Initialize DVC
            await self.version_manager.initialize_dvc()

            # Get all files for processing
            current_files = self.version_manager.scan_current_files()

            logger.info(
                f"First run setup complete: {len(current_files)} files to process"
            )

            return DiffResult(
                new_files=list(current_files.values()),
                total_files_scanned=len(current_files),
                previously_processed_count=0,
                is_first_run=True,
            )

        except Exception as e:
            raise DiffError(f"First run setup failed: {str(e)}", e) from e


def create_diff_config() -> DiffConfig:
    """
    Create DiffConfig from environment variables.

    Hybrid approach:
    - S3Ferry handles metadata operations (processed files tracking)
    - DVC needs direct S3 access for version control operations

    Returns:
        DiffConfig instance

    Raises:
        DiffError: If required environment variables are missing
    """
    try:
        # S3Ferry Configuration
        s3_ferry_url = os.getenv("S3_FERRY_URL")
        if not s3_ferry_url:
            raise DiffError("Missing required environment variable: S3_FERRY_URL")

        # Path configurations
        datasets_path = os.getenv("DATASETS_PATH")
        if not datasets_path:
            raise DiffError("Missing required environment variable: DATASETS_PATH")
        metadata_filename = os.getenv("METADATA_FILENAME")
        if not metadata_filename:
            raise DiffError("Missing required environment variable: METADATA_FILENAME")

        # S3 configuration (required for DVC operations)
        s3_bucket_name = os.getenv("S3_DATA_BUCKET_NAME")
        s3_bucket_path = os.getenv("S3_DATA_BUCKET_PATH")
        s3_endpoint_url = os.getenv("S3_ENDPOINT_URL")
        s3_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
        s3_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")

        # Validate required S3 credentials for DVC
        if not all(
            [s3_bucket_name, s3_endpoint_url, s3_access_key_id, s3_secret_access_key]
        ):
            missing = [
                var
                for var, val in [
                    ("S3_DATA_BUCKET_NAME", s3_bucket_name),
                    ("S3_ENDPOINT_URL", s3_endpoint_url),
                    ("S3_ACCESS_KEY_ID", s3_access_key_id),
                    ("S3_SECRET_ACCESS_KEY", s3_secret_access_key),
                ]
                if not val
            ]
            raise DiffError(
                f"Missing required S3 environment variables for DVC: {', '.join(missing)}"
            )

        # Build paths
        # S3Ferry is already configured with bucket context, so no need for s3_bucket_path prefix
        metadata_s3_path = f"datasets/{metadata_filename}"
        dvc_remote_url = f"s3://{s3_bucket_name}/{s3_bucket_path}/datasets/dvc-cache"

        config = DiffConfig(
            s3_ferry_url=s3_ferry_url,
            metadata_s3_path=metadata_s3_path,
            datasets_path=datasets_path,
            metadata_filename=metadata_filename,
            dvc_remote_url=dvc_remote_url,
            s3_endpoint_url=str(s3_endpoint_url),
            s3_access_key_id=str(s3_access_key_id),
            s3_secret_access_key=str(s3_secret_access_key),
        )

        logger.info("Diff configuration loaded successfully")
        logger.info(f"S3Ferry URL: {config.s3_ferry_url}")
        logger.info(f"Metadata S3 Path: {config.metadata_s3_path}")
        logger.info(f"DVC Remote URL: {config.dvc_remote_url}")
        logger.info(f"Datasets Path: {config.datasets_path}")

        return config

    except Exception as e:
        raise DiffError(f"Failed to create diff configuration: {str(e)}", e) from e

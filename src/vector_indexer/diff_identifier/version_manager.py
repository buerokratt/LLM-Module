"""Version manager for DVC operations and metadata handling."""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from loki_logger import LokiLogger

from typing_extensions import Self

from diff_identifier.diff_models import (
    DiffConfig,
    DiffError,
    ProcessedFileInfo,
    VersionState,
)
from diff_identifier.s3_ferry_client import S3FerryClient

# Initialize Loki logger
logger = LokiLogger(service_name="version-manager")


class VersionManager:
    """Manages DVC operations and version tracking."""

    def __init__(self, config: DiffConfig) -> None:
        self.config = config
        self.datasets_path = Path(config.datasets_path)

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

    def is_dvc_initialized(self) -> bool:
        """Check if DVC is initialized in datasets directory."""
        dvc_dir = self.datasets_path / ".dvc"
        return dvc_dir.exists() and dvc_dir.is_dir()

    async def initialize_dvc(self) -> None:
        """
        Initialize DVC in datasets directory with S3 remote.

        Raises:
            DiffError: If DVC initialization fails
        """
        try:
            logger.info("Initializing DVC in datasets directory...")

            # Ensure datasets directory exists
            self.datasets_path.mkdir(parents=True, exist_ok=True)

            # Initialize DVC (no SCM integration)
            await self._run_dvc_command(["dvc", "init", "--no-scm"])

            # Add S3 remote
            remote_url = self.config.dvc_remote_url
            logger.info(f"Adding DVC remote: {remote_url}")
            await self._run_dvc_command(
                ["dvc", "remote", "add", "-d", "rag-storage", remote_url]
            )

            # Configure S3 credentials
            await self._run_dvc_command(
                [
                    "dvc",
                    "remote",
                    "modify",
                    "rag-storage",
                    "endpointurl",
                    self.config.s3_endpoint_url,
                ]
            )
            await self._run_dvc_command(
                [
                    "dvc",
                    "remote",
                    "modify",
                    "rag-storage",
                    "access_key_id",
                    self.config.s3_access_key_id,
                ]
            )
            await self._run_dvc_command(
                [
                    "dvc",
                    "remote",
                    "modify",
                    "rag-storage",
                    "secret_access_key",
                    self.config.s3_secret_access_key,
                ]
            )

            logger.info("DVC initialized successfully")

        except Exception as e:
            raise DiffError(f"Failed to initialize DVC: {str(e)}", e) from e

    async def get_processed_files_metadata(self) -> Optional[VersionState]:
        """
        Download and parse processed files metadata from S3.

        Returns:
            VersionState if metadata exists, None if first run

        Raises:
            DiffError: If metadata exists but cannot be parsed
        """
        try:
            async with S3FerryClient(self.config) as s3_client:
                metadata_dict = await s3_client.download_metadata()

                if metadata_dict is None:
                    return None

                # Parse metadata into VersionState
                return VersionState(
                    last_updated=metadata_dict["last_updated"],
                    processed_files={
                        file_hash: ProcessedFileInfo(**file_info)
                        for file_hash, file_info in metadata_dict[
                            "processed_files"
                        ].items()
                    },
                    total_processed=metadata_dict.get(
                        "total_processed", len(metadata_dict["processed_files"])
                    ),
                )

        except Exception as e:
            raise DiffError(
                f"Failed to get processed files metadata: {str(e)}", e
            ) from e

    async def update_processed_files_metadata(
        self,
        processed_files: Dict[str, str],
        chunks_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """
        Update processed files metadata and upload to S3.

        Args:
            processed_files: Dict mapping file hash to file path for newly processed files
            chunks_info: Optional dict mapping file_hash to {"chunk_count": int}

        Raises:
            DiffError: If metadata update fails
        """
        try:
            # Get existing metadata or create new
            existing_state = await self.get_processed_files_metadata()
            processed_files_dict: Dict[str, ProcessedFileInfo] = (
                existing_state.processed_files.copy() if existing_state else {}
            )
            processing_stats: Dict[str, Any] = (
                existing_state.processing_stats.copy()
                if existing_state and existing_state.processing_stats
                else {}
            )

            # Handle cleanup-only operation
            if not processed_files and existing_state:
                current_files = self.scan_current_files()
                current_hashes: Set[str] = set(current_files.keys())
                deleted_count = sum(
                    1 for h in processed_files_dict if h not in current_hashes
                )
                processed_files_dict = {
                    h: info
                    for h, info in processed_files_dict.items()
                    if h in current_hashes
                }
                if deleted_count > 0:
                    logger.info(f"Removed {deleted_count} deleted files from metadata")
                    processing_stats["last_run_deleted_files"] = deleted_count

            # Build path-to-hash map for deduplication
            path_to_hash: Dict[str, str] = {
                info.original_path: h for h, info in processed_files_dict.items()
            }
            current_time = datetime.now().isoformat()

            # Add/update new and modified files
            for file_hash, file_path in processed_files.items():
                file_stats = Path(file_path).stat()

                # Remove old entry if file was modified
                if file_path in path_to_hash and path_to_hash[file_path] != file_hash:
                    old_hash = path_to_hash[file_path]
                    del processed_files_dict[old_hash]
                    logger.info(
                        f"DEDUPLICATING: {file_path} (old: {old_hash[:12]}..., new: {file_hash[:12]}...)"
                    )

                # Get chunk count
                chunk_count = (
                    chunks_info.get(file_hash, {}).get("chunk_count", 0)
                    if chunks_info
                    else 0
                )
                if chunks_info and file_hash in chunks_info:
                    logger.info(f"Found {chunk_count} chunks for {file_hash[:12]}...")

                # Add/update file entry
                processed_files_dict[file_hash] = ProcessedFileInfo(
                    content_hash=file_hash,
                    original_path=file_path,
                    file_size=file_stats.st_size,
                    processed_at=current_time,
                    chunk_count=chunk_count,
                )
                path_to_hash[file_path] = file_hash

            # Update stats and create new state
            if processed_files:
                processing_stats["last_run_new_files"] = len(processed_files)
            processing_stats["last_run_timestamp"] = current_time

            new_state = VersionState(
                last_updated=current_time,
                processed_files=processed_files_dict,
                total_processed=len(processed_files_dict),
                processing_stats=processing_stats,
            )

            # Upload to S3
            metadata_dict = {
                "last_updated": new_state.last_updated,
                "total_processed": new_state.total_processed,
                "processing_stats": new_state.processing_stats,
                "processed_files": {
                    fh: {
                        "content_hash": fi.content_hash,
                        "original_path": fi.original_path,
                        "file_size": fi.file_size,
                        "processed_at": fi.processed_at,
                        "chunk_count": fi.chunk_count,
                    }
                    for fh, fi in new_state.processed_files.items()
                },
            }

            async with S3FerryClient(self.config) as s3_client:
                if not await s3_client.upload_metadata(metadata_dict):
                    raise DiffError("Failed to upload metadata to S3")

            logger.info(
                f"Updated processed files metadata: {len(processed_files)} new files"
            )

        except DiffError:
            raise
        except Exception as e:
            raise DiffError(
                f"Failed to update processed files metadata: {str(e)}", e
            ) from e

    def scan_current_files(self) -> Dict[str, str]:
        """
        Scan datasets directory and calculate file hashes.

        Returns:
            Dict mapping file hash to file path

        Raises:
            DiffError: If file scanning fails
        """
        try:
            files_map: Dict[str, str] = {}

            if not self.datasets_path.exists():
                logger.warning(f"Datasets path does not exist: {self.datasets_path}")
                return files_map

            # Find all cleaned.txt files
            cleaned_files = list(self.datasets_path.glob("**/cleaned.txt"))
            logger.info(f"Found {len(cleaned_files)} files to scan")

            for cleaned_file in cleaned_files:
                try:
                    # Calculate file hash consistently with document_loader.py
                    # Use text mode and encode to match document processing pipeline
                    with open(cleaned_file, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    # Store relative path from datasets directory
                    relative_path = str(
                        cleaned_file.relative_to(self.datasets_path.parent)
                    )
                    files_map[file_hash] = relative_path

                    logger.debug(
                        f"Scanned file: {relative_path} -> {file_hash[:12]}..."
                    )

                except Exception as e:
                    logger.warning(f"Failed to process file {cleaned_file}: {e}")
                    continue

            logger.info(f"Successfully scanned {len(files_map)} files")
            return files_map

        except Exception as e:
            raise DiffError(f"Failed to scan current files: {str(e)}", e) from e

    def identify_comprehensive_changes(
        self, current_files: Dict[str, str], processed_state: Optional[VersionState]
    ) -> Dict[str, Any]:
        """
        Identify all types of file changes: new, modified, deleted, unchanged.

        Args:
            current_files: Current files map (hash -> path)
            processed_state: Previously processed state

        Returns:
            Dict with lists of different file change types and cleanup information
        """
        if processed_state is None:
            # First run - all files are new
            logger.info("First run detected - all files are new")
            return {
                "new_files": list(current_files.values()),
                "modified_files": [],
                "deleted_files": [],
                "unchanged_files": [],
                "chunks_to_delete": {},
                "estimated_cleanup_count": 0,
            }

        # Initialize result lists
        new_files: List[str] = []
        modified_files: List[str] = []
        deleted_files: List[str] = []
        unchanged_files: List[str] = []
        chunks_to_delete: Dict[str, str] = {}
        total_chunks_to_delete = 0

        # Create lookup maps for efficient searching
        current_hash_to_path: Dict[str, str] = current_files  # hash -> path
        processed_hash_to_info: Dict[str, ProcessedFileInfo] = (
            processed_state.processed_files
        )  # hash -> ProcessedFileInfo
        processed_path_to_hash: Dict[str, str] = {
            info.original_path: hash
            for hash, info in processed_state.processed_files.items()
        }  # path -> hash

        # 1. Find deleted files (in processed_state but not in current scan)
        logger.debug("Identifying deleted files...")
        for old_hash, old_info in processed_hash_to_info.items():
            if old_hash not in current_hash_to_path:
                deleted_files.append(old_info.original_path)
                # Use content hash (old_hash) as document_hash for cleanup - now they match!
                chunks_to_delete[old_hash] = old_info.original_path
                # Estimate chunks to delete (use chunk_count if available, otherwise assume some exist)
                estimated_chunks = max(
                    old_info.chunk_count, 1
                )  # Assume at least 1 chunk if processed before
                total_chunks_to_delete += estimated_chunks
                logger.debug(
                    f"Deleted file: {old_info.original_path} (content_hash/document_hash: {old_hash[:12]}..., estimated chunks: {estimated_chunks})"
                )

        # 2. Find new, modified, and unchanged files
        logger.debug("Identifying new, modified, and unchanged files...")
        for current_hash, current_path in current_hash_to_path.items():
            if current_hash in processed_hash_to_info:
                # File exists with same content hash - unchanged
                unchanged_files.append(current_path)
                logger.debug(f"Unchanged file: {current_path}")
            else:
                # Check if this is a modified file (same path, different hash)
                if current_path in processed_path_to_hash:
                    old_hash = processed_path_to_hash[current_path]
                    old_info = processed_hash_to_info[old_hash]
                    modified_files.append(current_path)
                    # Mark old chunks for deletion - use content hash (old_hash) as document_hash
                    chunks_to_delete[old_hash] = old_info.original_path
                    total_chunks_to_delete += max(old_info.chunk_count, 1)
                    logger.debug(
                        f"Modified file: {current_path} (old_content_hash/document_hash: {old_hash[:12]}..., new_content_hash: {current_hash[:12]}..., estimated old chunks: {max(old_info.chunk_count, 1)})"
                    )
                else:
                    # Completely new file
                    new_files.append(current_path)
                    logger.debug(f"New file: {current_path}")

        # Log summary
        logger.info("COMPREHENSIVE DIFF ANALYSIS COMPLETE:")
        logger.info(f"New files: {len(new_files)}")
        logger.info(f"Modified files: {len(modified_files)}")
        logger.info(f"Deleted files: {len(deleted_files)}")
        logger.info(f"Unchanged files: {len(unchanged_files)}")
        logger.info(f"Total chunks to cleanup: {total_chunks_to_delete}")

        return {
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "unchanged_files": unchanged_files,
            "chunks_to_delete": chunks_to_delete,
            "estimated_cleanup_count": total_chunks_to_delete,
        }

    def identify_changed_files(
        self, current_files: Dict[str, str], processed_state: Optional[VersionState]
    ) -> Set[str]:
        """
        Legacy method - kept for backward compatibility.
        Use identify_comprehensive_changes for new functionality.

        Args:
            current_files: Current files map (hash -> path)
            processed_state: Previously processed state

        Returns:
            Set of file paths that need processing
        """
        changes = self.identify_comprehensive_changes(current_files, processed_state)
        # Return new + modified files (files that need processing)
        all_changed: List[str] = changes["new_files"] + changes["modified_files"]
        return set(all_changed)

    async def commit_dvc_changes(self) -> None:
        """
        Commit current datasets state to DVC and push to remote.

        Raises:
            DiffError: If DVC operations fail
        """
        try:
            # Add all cleaned.txt files to DVC tracking instead of using "."
            logger.info("Adding files to DVC tracking...")

            # Find all cleaned.txt files relative to datasets directory
            cleaned_files = list(self.datasets_path.glob("**/cleaned.txt"))
            if cleaned_files:
                # Add each file individually using relative paths
                for cleaned_file in cleaned_files:
                    try:
                        # Get relative path from datasets directory
                        relative_path = cleaned_file.relative_to(self.datasets_path)
                        logger.debug(f"Adding file to DVC: {relative_path}")
                        await self._run_dvc_command(["dvc", "add", str(relative_path)])
                    except Exception as e:
                        logger.warning(f"Failed to add {cleaned_file} to DVC: {e}")
                        # Continue with other files
                        continue

                logger.info(f"Added {len(cleaned_files)} files to DVC tracking")
            else:
                logger.warning("No cleaned.txt files found to add to DVC")

            # Push to remote storage
            logger.info("Pushing to DVC remote storage...")
            await self._run_dvc_command(["dvc", "push"])

            logger.info("DVC commit completed successfully")

        except Exception as e:
            raise DiffError(f"Failed to commit DVC changes: {str(e)}", e) from e

    async def _run_dvc_command(self, command: List[str]) -> str:
        """
        Run DVC command asynchronously.

        Args:
            command: DVC command as list of strings

        Returns:
            Command output

        Raises:
            DiffError: If command fails
        """
        try:
            logger.debug(f"Running DVC command: {' '.join(command)}")

            # Ensure DVC commands run from the datasets directory
            cwd = str(self.datasets_path.resolve())
            logger.debug(f"Running DVC command in directory: {cwd}")
            logger.debug(f"datasets_path: {self.datasets_path}")
            logger.debug(f"datasets_path.resolve(): {self.datasets_path.resolve()}")
            logger.debug(f"datasets_path exists: {self.datasets_path.exists()}")

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(
                    f"DVC command failed with return code {process.returncode}"
                )
                logger.error(f"Command: {' '.join(command)}")
                logger.error(f"Working directory: {cwd}")
                logger.error(f"Stdout: {stdout.decode().strip()}")
                logger.error(f"Stderr: {error_msg}")
                raise DiffError(
                    f"DVC command failed: {' '.join(command)} - {error_msg}"
                )

            output = stdout.decode().strip()
            logger.debug(f"DVC command output: {output}")

            return output

        except Exception as e:
            if isinstance(e, DiffError):
                raise
            raise DiffError(
                f"Failed to run DVC command {' '.join(command)}: {str(e)}", e
            ) from e

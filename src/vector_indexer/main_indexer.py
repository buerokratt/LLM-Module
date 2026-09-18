"""Main vector indexer script for processing documents with contextual retrieval."""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from loki_logger import LokiLogger
import hashlib

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vector_indexer.config.config_loader import ConfigLoader
from vector_indexer.document_loader import DocumentLoader, DocumentLoadError
from vector_indexer.contextual_processor import ContextualProcessor
from vector_indexer.qdrant_manager import QdrantManager
from vector_indexer.error_logger import ErrorLogger
from vector_indexer.models import ProcessingStats, DocumentInfo
from vector_indexer.diff_identifier import DiffDetector, create_diff_config, DiffError
from vector_indexer.diff_identifier.diff_models import DiffResult
from vector_indexer.dataset_download import download_and_extract_dataset

# Initialize Loki logger
logger = LokiLogger(service_name="main-indexer")


class VectorIndexer:
    """Main vector indexer orchestrating the full pipeline."""

    def __init__(
        self, config_path: Optional[str] = None, signed_url: Optional[str] = None
    ) -> None:
        # Load configuration
        self.config_path = (
            config_path or "src/vector_indexer/config/vector_indexer_config.yaml"
        )
        self.config = ConfigLoader.load_config(self.config_path)

        # Store signed URL for future dataset download implementation
        self.signed_url = signed_url

        # Initialize components
        self.document_loader = DocumentLoader(self.config)
        self.error_logger = ErrorLogger(self.config)

        # Initialize API client
        from vector_indexer.api_client import LLMOrchestrationAPIClient

        self.api_client = LLMOrchestrationAPIClient(self.config)

        # Initialize contextual processor with all required arguments
        self.contextual_processor = ContextualProcessor(
            self.api_client, self.config, self.error_logger
        )

        # Processing statistics
        self.stats = ProcessingStats()

        logger.info(f"Vector Indexer initialized with config: {self.config_path}")
        logger.info(f"Dataset path: {self.config.dataset_base_path}")
        logger.info(f"Max concurrent documents: {self.config.max_concurrent_documents}")
        logger.info(
            f"Max concurrent chunks: {self.config.max_concurrent_chunks_per_doc}"
        )

        if self.signed_url:
            logger.info(f"Signed URL provided: {self.signed_url[:50]}...")

    async def process_all_documents(self) -> ProcessingStats:
        """
        Process all documents in the dataset with contextual retrieval.

        Returns:
            ProcessingStats: Overall processing statistics
        """
        logger.info("Starting Vector Indexer - Contextual Retrieval Pipeline")

        self.stats.start_time = datetime.now()

        try:
            # Step 1: Dataset download
            if self.signed_url:
                logger.info("Dataset download URL provided - starting download")
                try:
                    extraction_path, files_count = download_and_extract_dataset(
                        self.signed_url
                    )
                    logger.info(
                        f"Dataset extracted: {files_count} files to {extraction_path}"
                    )
                    # Update config to use the downloaded dataset
                    self.config.dataset_base_path = extraction_path
                except Exception as e:
                    logger.error(f"Dataset download failed: {e}")
                    raise

            # Step 2: Diff identification - determine what files need processing
            logger.info("Step 1: Identifying changed files...")
            try:
                diff_config = create_diff_config()
                diff_detector = DiffDetector(diff_config)
                diff_result = await diff_detector.get_changed_files()

                logger.info("Diff identification complete:")
                logger.info(
                    f"  • Total files scanned: {diff_result.total_files_scanned}"
                )
                logger.info(
                    f"  • Previously processed: {diff_result.previously_processed_count}"
                )
                logger.info(f"  • New files: {len(diff_result.new_files)}")
                logger.info(f"  • Modified files: {len(diff_result.modified_files)}")
                logger.info(f"  • Deleted files: {len(diff_result.deleted_files)}")
                logger.info(f"  • Unchanged files: {len(diff_result.unchanged_files)}")
                logger.info(f"  • Is first run: {diff_result.is_first_run}")

                files_to_process = diff_result.new_files + diff_result.modified_files

            except DiffError as e:
                logger.error(f"Diff identification failed: {e}")
                logger.info("Continuing with full document discovery as fallback")
                diff_result = None
                diff_detector = None
                files_to_process = []

            # Initialize Qdrant collections
            async with QdrantManager(self.config) as qdrant_manager:
                await qdrant_manager.ensure_collections_exist()

                # Step 2.5: Execute cleanup operations for deleted/modified files
                if diff_result and diff_result.chunks_to_delete:
                    logger.info("EXECUTING CLEANUP OPERATIONS...")
                    await self._execute_cleanup_operations(qdrant_manager, diff_result)

                # Early exit check AFTER cleanup operations
                # Only exit if there's nothing to process AND no cleanup was needed
                if diff_result and not files_to_process:
                    logger.info("No new or modified files to process.")
                    # ALWAYS update metadata when there were deletions or modifications
                    if diff_detector and (
                        diff_result.deleted_files or diff_result.modified_files
                    ):
                        logger.info("Updating metadata to reflect file changes...")
                        await diff_detector.mark_files_processed(
                            [], force_metadata_update=True
                        )
                        logger.info("Metadata updated successfully.")
                    else:
                        logger.info("No changes detected - no metadata update needed.")
                    return self.stats

                # Step 3: Document discovery (filtered by diff results if available)
                logger.info("Step 2: Discovering documents...")
                if diff_result and files_to_process:
                    # Filter documents to only those identified as changed
                    documents = self._filter_documents_by_paths(files_to_process)
                else:
                    # Fallback: discover all documents
                    documents = self.document_loader.discover_all_documents()

                if not documents:
                    logger.warning("No documents found to process")
                    self._cleanup_datasets()
                    return self.stats

                logger.info(f"Found {len(documents)} documents to process")
                self.stats.total_documents = len(documents)

                # Process documents with controlled concurrency
                semaphore = asyncio.Semaphore(self.config.max_concurrent_documents)
                tasks: List[asyncio.Task[tuple[int, str, int]]] = []

                for doc_info in documents:
                    task = asyncio.create_task(
                        self._process_single_document(
                            doc_info, qdrant_manager, semaphore
                        )
                    )
                    tasks.append(task)

                # Execute all document processing tasks
                logger.info(
                    f"Processing {len(tasks)} documents with max {self.config.max_concurrent_documents} concurrent"
                )
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Collect results and handle exceptions
                chunks_info: Dict[
                    str, Dict[str, Any]
                ] = {}  # Track chunk counts for metadata update
                # Only documents that processed successfully are marked as
                # processed in DVC tracking, so failures are retried next run.
                processed_documents: List[DocumentInfo] = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        doc_info = documents[i]
                        logger.error(
                            f"Document processing failed: {doc_info.document_hash} - {result}"
                        )
                        self.stats.documents_failed += 1
                        self.error_logger.log_document_failure(
                            doc_info.document_hash, str(result)
                        )
                    else:
                        # Result should be tuple of (chunk_count, content_hash, failed_chunks)
                        doc_info = documents[i]
                        self.stats.documents_processed += 1
                        processed_documents.append(doc_info)
                        if isinstance(result, tuple) and len(result) == 3:
                            chunk_count, content_hash, failed_chunks = result
                            self.stats.total_chunks_processed += chunk_count
                            self.stats.total_chunks_failed += failed_chunks
                            # Track chunk count using content_hash (not directory hash)
                            chunks_info[content_hash] = {"chunk_count": chunk_count}
                            logger.info(
                                f"CHUNK COUNT: Document {doc_info.document_hash[:12]}... (content: {content_hash[:12]}...) -> {chunk_count} chunks ({failed_chunks} failed)"
                            )

                # Log summary only (avoid massive logs for large datasets)
                total_chunks = sum(info["chunk_count"] for info in chunks_info.values())
                logger.info(
                    f"CHUNKS INFO SUMMARY: {len(chunks_info)} documents tracked, {total_chunks} total chunks"
                )

                # Calculate final statistics
                self.stats.end_time = datetime.now()

                # Step 4: Update processed files tracking (even if no new documents processed)
                if diff_detector:
                    try:
                        # Update metadata for newly processed files (successful only)
                        if processed_documents:
                            processed_paths = [
                                doc.cleaned_txt_path for doc in processed_documents
                            ]
                            if processed_paths:
                                logger.debug(
                                    f"Passing chunks_info with keys: {[k[:12] + '...' for k in chunks_info.keys()]} to mark_files_processed"
                                )
                                await diff_detector.mark_files_processed(
                                    processed_paths, chunks_info=chunks_info
                                )
                                logger.info(
                                    "Updated processed files tracking for new documents"
                                )

                        # CRITICAL: Update metadata even when only cleanup operations occurred
                        # This ensures deleted files are removed from metadata
                        elif diff_result and (
                            diff_result.deleted_files or diff_result.modified_files
                        ):
                            logger.info(
                                "Updating metadata to reflect file deletions/modifications..."
                            )
                            # Force metadata update for cleanup operations
                            await diff_detector.mark_files_processed(
                                [], force_metadata_update=True
                            )
                            logger.info(
                                "Updated processed files tracking for cleanup operations"
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to update processed files tracking: {e}"
                        )

                # Log final statistics
                self.error_logger.log_processing_stats(self.stats)
                self._log_final_summary()

                # Step 5: Cleanup datasets folder after successful processing
                self._cleanup_datasets()

                return self.stats

        except Exception as e:
            logger.error(f"Critical error in vector indexer: {e}")
            self.stats.end_time = datetime.now()
            self.error_logger.log_processing_stats(self.stats)
            raise
        finally:
            # Clean up API client AFTER all processing is complete
            try:
                await self.api_client.close()
            except Exception as e:
                logger.warning(f"Error closing API client: {e}")

    async def _process_single_document(
        self,
        doc_info: DocumentInfo,
        qdrant_manager: QdrantManager,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, str, int]:
        """
        Process a single document with contextual retrieval.

        Args:
            doc_info: Document information
            qdrant_manager: Qdrant manager instance
            semaphore: Concurrency control semaphore

        Returns:
            tuple: (chunk_count: int, content_hash: str, failed_chunks: int).
            Raises on any failure (including load failure or zero usable chunks),
            so the document is counted as failed rather than as success.
        """
        async with semaphore:
            logger.info(f"Processing document: {doc_info.document_hash}")

            try:
                # Load document content
                document = self.document_loader.load_document(doc_info)

                if not document:
                    raise DocumentLoadError(
                        f"Could not load document: {doc_info.document_hash}"
                    )

                # Process document with contextual retrieval
                (
                    contextual_chunks,
                    failed_chunks,
                ) = await self.contextual_processor.process_document(document)

                if not contextual_chunks:
                    raise RuntimeError(
                        f"No chunks created for document: {doc_info.document_hash} "
                        f"({failed_chunks} chunks failed context generation)"
                    )

                # Store chunks in Qdrant
                await qdrant_manager.store_chunks(contextual_chunks)

                logger.info(
                    f"Successfully processed document {doc_info.document_hash}: "
                    f"{len(contextual_chunks)} chunks ({failed_chunks} dropped)"
                )

                return (len(contextual_chunks), document.document_hash, failed_chunks)

            except Exception as e:
                logger.error(f"Error processing document {doc_info.document_hash}: {e}")
                self.error_logger.log_document_failure(doc_info.document_hash, str(e))
                raise

    def _log_final_summary(self) -> None:
        """Log final processing summary."""

        logger.info("VECTOR INDEXER PROCESSING COMPLETE")

        logger.info("Processing Statistics:")
        logger.info(f"   • Total Documents: {self.stats.total_documents}")
        logger.info(f"   • Successful Documents: {self.stats.documents_processed}")
        logger.info(f"   • Failed Documents: {self.stats.documents_failed}")
        logger.info(f"   • Total Chunks: {self.stats.total_chunks_processed}")
        logger.info(f"   • Failed Chunks: {self.stats.total_chunks_failed}")

        if self.stats.total_documents > 0:
            logger.info(f"Success Rate: {self.stats.success_rate * 100:.1f}%")

        if self.stats.total_chunks_processed + self.stats.total_chunks_failed > 0:
            logger.info(
                f"Chunk Success Rate: {self.stats.chunk_success_rate * 100:.1f}%"
            )

        logger.info(f"Processing Duration: {self.stats.duration}")

        if self.stats.documents_failed > 0:
            logger.warning(
                f"  {self.stats.documents_failed} documents failed processing"
            )
            logger.info("Check failure logs for details")

        if self.stats.total_chunks_failed > 0:
            logger.warning(
                f"  {self.stats.total_chunks_failed} chunks failed processing"
            )

    async def run_health_check(self) -> bool:
        """
        Run health check on all components.

        Returns:
            bool: True if all components are healthy
        """
        logger.info("Running Vector Indexer health check...")

        try:
            # Check Qdrant connection
            async with QdrantManager(self.config) as qdrant_manager:
                # Test basic Qdrant connectivity by trying to list collections
                try:
                    qdrant_url = self.config.qdrant_url
                    response = await qdrant_manager.client.get(
                        f"{qdrant_url}/collections"
                    )
                    if response.status_code == 200:
                        logger.info("Qdrant server: Connected")

                        # Check if collections exist, create them if they don't
                        collections_info = {}
                        for collection_name in qdrant_manager.collections_config.keys():
                            info = await qdrant_manager.get_collection_info(
                                collection_name
                            )
                            if info:
                                count = await qdrant_manager.count_points(
                                    collection_name
                                )
                                collections_info[collection_name] = count
                                logger.info(
                                    f"Qdrant collection '{collection_name}': {count} points"
                                )
                            else:
                                logger.info(
                                    f"Qdrant collection '{collection_name}': Not found (will be created automatically)"
                                )
                    else:
                        logger.error(
                            f"Qdrant server not accessible: {response.status_code}"
                        )
                        return False
                except Exception as e:
                    logger.error(f"Qdrant connection failed: {e}")
                    return False

            # Check API client connectivity
            api_healthy = await self.api_client.health_check()
            if api_healthy:
                logger.info("LLM Orchestration Service API: Connected")
            else:
                logger.error("LLM Orchestration Service API: Not accessible")
                return False

            # Check dataset path
            if Path(self.config.dataset_base_path).exists():
                logger.info(f"Dataset path: {self.config.dataset_base_path}")
            else:
                logger.error(f"Dataset path not found: {self.config.dataset_base_path}")
                return False

            logger.info("All health checks passed!")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
        # NOTE: Don't close API client here - it will be used by main processing

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.api_client.close()
            logger.debug("API client closed successfully")
        except Exception as e:
            logger.warning(f"Error closing API client: {e}")

    def _filter_documents_by_paths(self, file_paths: List[str]) -> List[DocumentInfo]:
        """
        Filter documents by specific file paths.

        IMPORTANT: This method now uses discover_all_documents() to get the correct
        content hashes that were already calculated, instead of recalculating them.
        This ensures consistency throughout the pipeline.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of DocumentInfo for matching files
        """
        documents: List[DocumentInfo] = []

        # FIX: Discover ALL documents first to get their content hashes
        # This ensures we use the same hash that was calculated in discover_all_documents()
        logger.debug("Discovering all documents to get content hashes...")
        all_documents = self.document_loader.discover_all_documents()

        # Create a lookup map: file_path -> DocumentInfo
        path_to_doc_map: Dict[str, DocumentInfo] = {
            doc.cleaned_txt_path: doc for doc in all_documents
        }
        logger.debug(f"Created path lookup map with {len(path_to_doc_map)} documents")

        for file_path in file_paths:
            # Check if this file path exists in our discovered documents
            if file_path in path_to_doc_map:
                # Use the DocumentInfo that was already discovered (with correct content hash)
                doc_info = path_to_doc_map[file_path]
                documents.append(doc_info)
                logger.debug(
                    f"Added document: {doc_info.document_hash[:12]}... from {file_path}"
                )
            else:
                logger.warning(
                    f"File path {file_path} not found in discovered documents"
                )

        logger.info(
            f"Filtered to {len(documents)} documents from {len(file_paths)} paths"
        )
        return documents

    async def _execute_cleanup_operations(
        self, qdrant_manager: QdrantManager, diff_result: "DiffResult"
    ) -> int:
        """
        Execute cleanup operations for deleted and modified files.

        Args:
            qdrant_manager: Qdrant manager instance
            diff_result: DiffResult containing cleanup information

        Returns:
            Total number of chunks deleted
        """
        total_deleted = 0

        if not diff_result.chunks_to_delete:
            logger.info("No cleanup operations needed")
            return total_deleted

        logger.info(
            f"STARTING CLEANUP: {len(diff_result.chunks_to_delete)} documents with chunks to delete"
        )

        # Get appropriate collection names
        collections_to_clean = ["contextual_chunks_azure", "contextual_chunks_aws"]

        for document_hash, original_path in diff_result.chunks_to_delete.items():
            logger.info(
                f"ATTEMPTING CLEANUP for document {document_hash[:12]}... (path: {original_path})"
            )
            logger.debug(f"DEBUG: Full document_hash for deletion: {document_hash}")
            logger.info(
                "DEBUG: This could be a retry if chunks were deleted in a previous run but metadata wasn't updated"
            )

            chunks_deleted_for_doc = 0
            fallback_hash = None

            for collection_name in collections_to_clean:
                try:
                    # Try with current document_hash first
                    deleted_count = await qdrant_manager.delete_chunks_by_document_hash(
                        collection_name, document_hash
                    )
                    chunks_deleted_for_doc += deleted_count

                    if deleted_count > 0:
                        logger.info(
                            f"Deleted {deleted_count} chunks from {collection_name}"
                        )
                    else:
                        # If no chunks found with current hash, try fallback with old hash calculation method
                        if fallback_hash is None and Path(original_path).exists():
                            try:
                                # Calculate hash using old method (read_bytes) for backward compatibility

                                file_content = Path(original_path).read_bytes()
                                fallback_hash = hashlib.sha256(file_content).hexdigest()
                                logger.info(
                                    f"Trying fallback hash calculation for backward compatibility: {fallback_hash[:12]}..."
                                )
                            except Exception as fallback_error:
                                logger.warning(
                                    f"Could not calculate fallback hash: {fallback_error}"
                                )
                                fallback_hash = "FAILED"

                        if (
                            fallback_hash
                            and fallback_hash != "FAILED"
                            and fallback_hash != document_hash
                        ):
                            fallback_deleted = (
                                await qdrant_manager.delete_chunks_by_document_hash(
                                    collection_name, fallback_hash
                                )
                            )
                            chunks_deleted_for_doc += fallback_deleted
                            if fallback_deleted > 0:
                                logger.info(
                                    f"  ✅ Deleted {fallback_deleted} chunks from {collection_name} using fallback hash"
                                )

                except Exception as e:
                    logger.error(f"Failed to delete chunks from {collection_name}: {e}")
                    continue

            total_deleted += chunks_deleted_for_doc
            if chunks_deleted_for_doc > 0:
                logger.info(
                    f"Total deleted for document {document_hash[:12]}...: {chunks_deleted_for_doc} chunks"
                )
            else:
                if (
                    fallback_hash
                    and fallback_hash != "FAILED"
                    and fallback_hash != document_hash
                ):
                    logger.info(
                        f"No chunks found for document {document_hash[:12]}... or fallback hash {fallback_hash[:12]}... (may have been deleted previously or stored with different hash)"
                    )
                else:
                    logger.info(
                        f"No chunks found for document {document_hash[:12]}... (file tracked in metadata but chunks not in vector store)"
                    )

        if total_deleted > 0:
            logger.info(
                f"CLEANUP COMPLETED: {total_deleted} total chunks removed from {len(diff_result.chunks_to_delete)} documents"
            )
        else:
            logger.info(
                f"CLEANUP COMPLETED: No chunks removed (0 chunks found in vector store for {len(diff_result.chunks_to_delete)} tracked documents)"
            )

        # Log cleanup summary by file type
        deleted_files = diff_result.deleted_files
        modified_files = diff_result.modified_files

        if deleted_files:
            logger.info(f"Processed cleanup for {len(deleted_files)} deleted files")
        if modified_files:
            logger.info(f"Processed cleanup for {len(modified_files)} modified files")

        return total_deleted

    def _cleanup_datasets(self) -> None:
        """Remove datasets folder contents after processing.

        Only the folder's contents are removed, not the folder itself, since
        the datasets path is a mounted volume in the container.
        """
        try:
            datasets_path = Path(self.config.dataset_base_path)
            if datasets_path.exists():
                for child in datasets_path.iterdir():
                    if child.is_dir():
                        shutil.rmtree(str(child))
                    else:
                        child.unlink()
                logger.info(f"Datasets folder contents cleaned up: {datasets_path}")
            else:
                logger.debug(f"Datasets folder does not exist: {datasets_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup datasets folder: {e}")
            # Non-critical error - don't fail the entire process


async def main() -> int:
    """Main entry point for the vector indexer."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Vector Indexer with Diff Identification"
    )
    parser.add_argument("--signed-url", help="Signed URL for dataset download")
    args = parser.parse_args()

    # LokiLogger handles all logging (console + Loki service)
    # No additional configuration needed

    indexer = None
    try:
        # Initialize vector indexer with signed URL
        indexer = VectorIndexer(signed_url=args.signed_url)

        # Run health check first
        logger.info("Performing pre-processing health check...")
        health_ok = await indexer.run_health_check()

        if not health_ok:
            logger.error("Health check failed. Aborting processing.")
            return 1  # Return exit code instead of sys.exit()

        # Process all documents
        logger.info("Health check passed. Starting document processing...")
        stats = await indexer.process_all_documents()

        # Exit with appropriate code
        if stats.documents_failed > 0:
            logger.warning(
                f"Processing completed with {stats.documents_failed} failures"
            )
            return 2  # Partial success
        else:
            logger.info("Processing completed successfully")
            return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1
    finally:
        # Ensure cleanup happens
        if indexer:
            try:
                await indexer.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Run the async main function and exit with the returned code
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

"""Document loader for scanning and loading documents from datasets folder."""

import hashlib
import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from loki_logger import LokiLogger

from vector_indexer.config.config_loader import VectorIndexerConfig
from vector_indexer.models import DocumentInfo, ProcessingDocument
from vector_indexer.constants import DocumentConstants

# Initialize Loki logger
logger = LokiLogger(service_name="document-loader")


class DocumentLoadError(Exception):
    """Custom exception for document loading failures."""

    pass


class DocumentLoader:
    """Handles document discovery and loading from datasets folder."""

    def __init__(self, config: VectorIndexerConfig) -> None:
        self.config = config
        self.datasets_path = Path(config.dataset_base_path)

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        """Validate that a URL has a proper scheme and network location."""
        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            return False

    def discover_all_documents(self) -> List[DocumentInfo]:
        """
        Optimized document discovery using pathlib.glob for better performance.

        Scans for any folder structure containing cleaned.txt and source.meta.json files.
        No assumptions about collection naming patterns - works with any folder structure.

        Expected structure (flexible):
        datasets/
        └── any_collection_name/
            ├── any_hash_directory/
            │   ├── cleaned.txt          <- Target file
            │   ├── source.meta.json     <- Metadata file
            │   └── other files...
            └── another_hash/
                ├── cleaned.txt
                └── source.meta.json

        Returns:
            List of DocumentInfo objects for processing
        """
        documents: List[DocumentInfo] = []

        if not self.datasets_path.exists():
            logger.error(f"Datasets path does not exist: {self.datasets_path}")
            return documents

        logger.info(f"Scanning datasets folder: {self.datasets_path}")

        # Use glob to find all target files recursively (any folder structure)
        pattern = f"**/{self.config.target_file}"

        for cleaned_file in self.datasets_path.glob(pattern):
            hash_dir = cleaned_file.parent

            # Skip if we're at root level (need at least one parent for collection)
            if hash_dir == self.datasets_path:
                continue

            # Get collection name (parent of hash directory)
            collection_dir = hash_dir.parent
            if collection_dir == self.datasets_path.parent:
                collection_name = DocumentConstants.DEFAULT_COLLECTION_NAME
            else:
                collection_name = collection_dir.name

            # This ensures document_hash is always the SHA-256 of file content
            try:
                with open(cleaned_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Calculate SHA-256 hash of content (same method used everywhere)
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                logger.debug(
                    f"Calculated content hash for {cleaned_file.name}: {content_hash[:12]}..."
                )

            except Exception as e:
                logger.warning(f"Failed to calculate hash for {cleaned_file}: {e}")
                continue

            # Check metadata file exists
            metadata_file = hash_dir / self.config.metadata_file
            if not metadata_file.exists():
                logger.warning(
                    f"Skipping document in {hash_dir.name}: "
                    f"missing {self.config.metadata_file}"
                )
                continue

            # Validate source_url before accepting the document
            try:
                with open(metadata_file, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                source_url = meta.get("source_url") or ""
            except Exception as e:
                logger.warning(
                    f"Skipping document in {hash_dir.name}: "
                    f"failed to read metadata: {e}"
                )
                continue

            if not self._is_valid_url(source_url):
                logger.warning(
                    f"Skipping document in {hash_dir.name}: "
                    f"invalid source_url '{source_url}'"
                )
                continue

            documents.append(
                DocumentInfo(
                    document_hash=content_hash,  # Use content hash consistently
                    cleaned_txt_path=str(cleaned_file),
                    source_meta_path=str(metadata_file),
                    dataset_collection=collection_name,
                )
            )
            logger.debug(
                f"Found document: {content_hash[:12]}... "
                f"in collection: {collection_name}"
            )

        logger.info(f"Discovered {len(documents)} documents for processing")
        return documents

    def load_document(self, doc_info: DocumentInfo) -> ProcessingDocument:
        """
        Load document content and metadata.

        Args:
            doc_info: Document information with content hash as document_hash

        Returns:
            ProcessingDocument with content and metadata

        Raises:
            DocumentLoadError: If document cannot be loaded
        """
        try:
            # Load cleaned text content
            with open(doc_info.cleaned_txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                raise ValueError(f"Empty content in {doc_info.cleaned_txt_path}")

            # Load metadata
            with open(doc_info.source_meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Add dataset collection to metadata
            metadata["dataset_collection"] = doc_info.dataset_collection

            logger.debug(
                f"Loaded document {doc_info.document_hash[:12]}...: {len(content)} characters"
            )

            # It's already the content hash (calculated in discover_all_documents)
            # No need to recalculate here - keeps the hash consistent
            return ProcessingDocument(
                content=content,
                metadata=metadata,
                document_hash=doc_info.document_hash,  # Already the content hash
            )

        except Exception as e:
            error_msg = f"Failed to load document {doc_info.document_hash[:12]}...: {e}"
            logger.error(error_msg)
            raise DocumentLoadError(error_msg) from e

    def get_document_by_hash(self, document_hash: str) -> DocumentInfo:
        """
        Find document by content hash.

        Args:
            document_hash: Document content hash to find

        Returns:
            DocumentInfo object

        Raises:
            ValueError: If document not found
        """
        all_documents = self.discover_all_documents()

        for doc_info in all_documents:
            if doc_info.document_hash == document_hash:
                return doc_info

        raise ValueError(f"Document not found with hash: {document_hash[:12]}...")

    def validate_document_structure(self, doc_info: DocumentInfo) -> bool:
        """
        Validate that document has required structure.

        Args:
            doc_info: Document information to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check files exist
            if not Path(doc_info.cleaned_txt_path).exists():
                logger.error(
                    f"Missing cleaned.txt for document {doc_info.document_hash[:12]}..."
                )
                return False

            if not Path(doc_info.source_meta_path).exists():
                logger.error(
                    f"Missing cleaned.meta.json for document {doc_info.document_hash[:12]}..."
                )
                return False

            # Try to load content with configurable validation
            with open(
                doc_info.cleaned_txt_path, "r", encoding=DocumentConstants.ENCODING
            ) as f:
                content = f.read().strip()
                if len(content) < DocumentConstants.MIN_CONTENT_LENGTH:
                    logger.error(
                        f"Content too short for document {doc_info.document_hash[:12]}...: "
                        f"{len(content)} chars (min: {DocumentConstants.MIN_CONTENT_LENGTH})"
                    )
                    return False

            # Try to load metadata
            with open(doc_info.source_meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                if not isinstance(metadata, dict):
                    logger.error(
                        f"Invalid metadata format for document {doc_info.document_hash[:12]}..."
                    )
                    return False

            return True

        except Exception as e:
            logger.error(
                f"Document validation failed for {doc_info.document_hash[:12]}...: {e}"
            )
            return False

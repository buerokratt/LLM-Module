#!/usr/bin/env python3
"""
Service Data Enrichment Script

This script receives service data, enriches it with LLM-generated context,
creates embeddings (dense + sparse per example), and stores in Qdrant intent_collections.

Indexing strategy:
- One 'example' point per example query (dense + sparse vectors of the example text)
- One 'summary' point per service (dense + sparse vectors of name + description + context)
"""

import os
import sys
import json
import argparse
import asyncio
import tempfile

from typing import List, Optional, Sequence, Tuple
from src.loki_logger import LokiLogger

from intent_data_enrichment.models import (
    ServiceData,
    EnrichedService,
    EnrichmentResult,
    EnrichmentStep,
)
from intent_data_enrichment.api_client import LLMAPIClient
from intent_data_enrichment.qdrant_manager import QdrantManager

# Initialize Loki logger
logger = LokiLogger(service_name="intent-enrichment-main")

# Import sparse encoder from tool_classifier (shared module)
sys.path.insert(0, "/app/src")
try:
    from tool_classifier.sparse_encoder import compute_sparse_vector
except ImportError:
    # Fallback for local development
    try:
        from src.tool_classifier.sparse_encoder import compute_sparse_vector
    except ImportError:
        logger.warning(
            "Could not import sparse_encoder from tool_classifier, "
            "attempting direct import"
        )
        import importlib.util
        import os

        # Try to find the module relative to this file
        module_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "tool_classifier",
            "sparse_encoder.py",
        )
        if os.path.exists(module_path):
            spec = importlib.util.spec_from_file_location("sparse_encoder", module_path)
            if spec is not None and spec.loader is not None:
                sparse_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sparse_module)
                compute_sparse_vector = sparse_module.compute_sparse_vector
            else:
                raise ImportError(
                    f"Cannot load spec or loader for sparse_encoder.py at {module_path}"
                ) from None
        else:
            raise ImportError(
                f"Cannot find sparse_encoder.py at {module_path}"
            ) from None


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[ServiceData, Optional[str], Optional[int]]:
    """Parse command line arguments.

    Returns:
        (ServiceData, result_file path or None, attempt_no or None)
    """
    parser = argparse.ArgumentParser(description="Service Data Enrichment")
    parser.add_argument("--service-id", type=str, required=True, help="Service ID")
    parser.add_argument("--name", type=str, required=True, help="Service name")
    parser.add_argument(
        "--description", type=str, required=True, help="Service description"
    )
    parser.add_argument("--examples-file", type=str, help="Path to examples JSON file")
    parser.add_argument("--entities-file", type=str, help="Path to entities JSON file")
    parser.add_argument("--ruuter-type", type=str, default="GET", help="Ruuter type")
    parser.add_argument(
        "--current-state", type=str, default="draft", help="Current state"
    )
    parser.add_argument(
        "--is-common",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Is common service",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        help="Where to write {success, step, error} JSON for the status callback",
    )
    parser.add_argument(
        "--attempt-no",
        type=int,
        default=None,
        help="Service Module attempt number (logged and written to the result file)",
    )

    args = parser.parse_args(argv)

    # Read and parse JSON arrays from files
    examples = []
    if args.examples_file:
        try:
            with open(args.examples_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    examples = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read/parse examples file: {e}")

    entities = []
    if args.entities_file:
        try:
            with open(args.entities_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    entities = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read/parse entities file: {e}")

    service_data = ServiceData(
        service_id=args.service_id,
        name=args.name,
        description=args.description,
        examples=examples,
        entities=entities,
        ruuter_type=args.ruuter_type,
        current_state=args.current_state,
        is_common=args.is_common.lower() == "true",
    )
    return service_data, args.result_file, args.attempt_no


async def enrich_service(service_data: ServiceData) -> EnrichmentResult:
    """
    Main enrichment pipeline: generate context, create per-example embeddings,
    store in Qdrant with hybrid vectors (dense + sparse).

    Args:
        service_data: Service data to enrich

    Returns:
        EnrichmentResult with success/failure information. On failure,
        ``failed_step`` names the step that failed (see EnrichmentStep).
    """
    # Updated as the pipeline moves on, so any exception can be tied to a step.
    current_step = EnrichmentStep.ENRICHING
    try:
        # Step 1: Generate rich context using LLM (unchanged from original)
        logger.info("Step 1: Generating rich context with LLM")
        async with LLMAPIClient() as api_client:
            context = await api_client.generate_context(service_data)
            logger.success(f"Context generated: {len(context)} characters")

            # Step 2: Create per-example points (dense + sparse vectors)
            current_step = EnrichmentStep.EMBEDDING
            logger.info(
                f"Step 2: Creating per-example embeddings for "
                f"{len(service_data.examples)} examples"
            )
            enriched_points: List[EnrichedService] = []

            for i, example in enumerate(service_data.examples):
                logger.info(
                    f"  Creating embeddings for example {i + 1}/{len(service_data.examples)}: "
                    f"'{example[:80]}...'"
                    if len(example) > 80
                    else f"  Creating embeddings for example {i + 1}/{len(service_data.examples)}: "
                    f"'{example}'"
                )

                # Dense: embed the individual example
                dense_embedding = await api_client.create_embedding(example)

                # Sparse: BM25-style term frequencies for the example
                sparse_vec = compute_sparse_vector(example)

                enriched_points.append(
                    EnrichedService(
                        id=service_data.service_id,
                        name=service_data.name,
                        description=service_data.description,
                        examples=service_data.examples,
                        entities=service_data.entities,
                        context=context,
                        embedding=dense_embedding,
                        sparse_indices=sparse_vec.indices,
                        sparse_values=sparse_vec.values,
                        example_text=example,
                        point_type="example",
                        is_common=service_data.is_common or False,
                    )
                )

            # Step 3: Create summary point (combined name + description + context)
            logger.info("Step 3: Creating summary embedding")
            combined_text_parts = [
                f"Service Name: {service_data.name}",
                f"Description: {service_data.description}",
            ]

            if service_data.examples:
                combined_text_parts.append(
                    f"Example Queries: {' | '.join(service_data.examples)}"
                )

            if service_data.entities:
                combined_text_parts.append(
                    f"Required Entities: {', '.join(service_data.entities)}"
                )

            combined_text_parts.append(f"Enriched Context: {context}")
            combined_text = "\n".join(combined_text_parts)

            summary_embedding = await api_client.create_embedding(combined_text)
            summary_sparse = compute_sparse_vector(combined_text)

            enriched_points.append(
                EnrichedService(
                    id=service_data.service_id,
                    name=service_data.name,
                    description=service_data.description,
                    examples=service_data.examples,
                    entities=service_data.entities,
                    context=context,
                    embedding=summary_embedding,
                    sparse_indices=summary_sparse.indices,
                    sparse_values=summary_sparse.values,
                    example_text=None,
                    point_type="summary",
                    is_common=service_data.is_common or False,
                )
            )

        # Step 4: Delete existing points for this service (idempotent update)
        logger.info("Step 4: Removing existing points for idempotent update")
        current_step = EnrichmentStep.QDRANT_DELETE
        qdrant = QdrantManager()
        try:
            qdrant.connect()
            qdrant.ensure_collection()

            # Delete old points before inserting new ones
            deleted = qdrant.delete_service_points(service_data.service_id)
            if not deleted:
                logger.error(
                    f"Failed to delete existing points for service_id={service_data.service_id}; "
                    "aborting upsert to avoid stale data."
                )
                return EnrichmentResult(
                    success=False,
                    service_id=service_data.service_id,
                    message="Qdrant delete failed before upsert",
                    context_length=None,
                    embedding_dimension=None,
                    error=qdrant.last_error or "delete_service_points returned False",
                    failed_step=EnrichmentStep.QDRANT_DELETE,
                )

            # Step 5: Bulk upsert all points (examples + summary)
            logger.info(
                f"Step 5: Storing {len(enriched_points)} points in Qdrant "
                f"({len(service_data.examples)} examples + 1 summary)"
            )
            current_step = EnrichmentStep.QDRANT_UPSERT
            success = qdrant.upsert_service_points(enriched_points)
        finally:
            qdrant.close()

        if success:
            return EnrichmentResult(
                success=True,
                service_id=service_data.service_id,
                message=(
                    f"Service '{service_data.name}' enriched and indexed successfully "
                    f"({len(enriched_points)} points: "
                    f"{len(service_data.examples)} examples + 1 summary)"
                ),
                context_length=len(context),
                embedding_dimension=len(summary_embedding),
                error=None,
            )
        else:
            return EnrichmentResult(
                success=False,
                service_id=service_data.service_id,
                message="Failed to store in Qdrant",
                context_length=None,
                embedding_dimension=None,
                error=(
                    f"{qdrant.last_error or 'upsert_service_points returned False'}. "
                    "The old index for this service was removed and the new one "
                    "was not written."
                ),
                failed_step=EnrichmentStep.QDRANT_UPSERT,
            )

    except Exception as e:
        error = _describe_error(e)
        if current_step == EnrichmentStep.QDRANT_UPSERT:
            error += (
                " The old index for this service was removed and the new one "
                "was not written."
            )
        logger.error(f"Enrichment pipeline failed at step {current_step}: {error}")
        return EnrichmentResult(
            success=False,
            service_id=service_data.service_id,
            message=f"Enrichment pipeline failed at step {current_step}",
            context_length=None,
            embedding_dimension=None,
            error=error,
            failed_step=current_step,
        )


def _describe_error(e: BaseException) -> str:
    """Return a non-empty error message (some exceptions have an empty str())."""
    text = str(e).strip()
    return f"{type(e).__name__}: {text}" if text else type(e).__name__


def write_result_file(path: str, result: dict) -> None:
    """Write the result JSON atomically, so the shell never reads a half file.

    Errors are logged and swallowed: a missing result file is handled by the
    shell script (it falls back to the log tail), and must never change the
    enrichment's exit code.
    """
    try:
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to write result file '{path}': {e}")


def _find_result_file_arg(argv: Sequence[str]) -> Optional[str]:
    """Pick --result-file out of raw argv.

    Used so the result file can still be written when argparse or the
    ServiceData validation fails (before parse_arguments returns).
    """
    for i, arg in enumerate(argv):
        if arg == "--result-file" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--result-file="):
            return arg.split("=", 1)[1]
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for service enrichment.

    Whatever happens, if --result-file was given, a JSON file
    ``{success, step, error, message, attemptNo}`` is written on the way out
    (``finally``), so service_enrichment.sh can report the exact outcome in its
    status callback.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    result_file = _find_result_file_arg(argv)
    result: Optional[EnrichmentResult] = None
    attempt_no: Optional[int] = None
    fatal_error: Optional[str] = None
    exit_code = 1

    logger.info("Starting service data enrichment pipeline")

    try:
        # Parse arguments
        service_data, result_file, attempt_no = parse_arguments(argv)
        logger.info(f"Service ID: {service_data.service_id}")
        logger.info(f"Service Name: {service_data.name}")
        logger.info(f"Attempt No: {attempt_no}")
        logger.info(f"Examples: {len(service_data.examples)} provided")
        logger.info(f"Entities: {len(service_data.entities)} provided")

        # Run enrichment pipeline
        result = asyncio.run(enrich_service(service_data))

        # Log results
        if result.success:
            logger.success("Enrichment completed successfully")
            logger.info(f"Service: {result.service_id}")
            logger.info(f"Message: {result.message}")
            logger.info(f"Context Length: {result.context_length} characters")
            logger.info(f"Embedding Dimension: {result.embedding_dimension}")
            exit_code = 0
        else:
            logger.error("Enrichment failed")
            logger.error(f"Service: {result.service_id}")
            logger.error(f"Step: {result.failed_step}")
            logger.error(f"Message: {result.message}")
            logger.error(f"Error: {result.error}")
            exit_code = 1

    except SystemExit as e:
        # argparse exits with code 2 on bad arguments
        fatal_error = f"Invalid arguments (exit code {e.code})"
        exit_code = e.code if isinstance(e.code, int) and e.code != 0 else 1
        logger.error(f"Fatal error: {fatal_error}")
    except Exception as e:
        fatal_error = _describe_error(e)
        exit_code = 1
        logger.error(f"Fatal error: {fatal_error}")
    finally:
        if result_file:
            if result is not None:
                payload = {
                    "success": result.success,
                    "step": result.failed_step,
                    "error": result.error,
                    "message": result.message,
                    "attemptNo": attempt_no,
                }
            else:
                payload = {
                    "success": False,
                    "step": None,
                    "error": fatal_error
                    or "Enrichment exited before producing a result",
                    "message": "Enrichment failed before the pipeline started",
                    "attemptNo": attempt_no,
                }
            write_result_file(result_file, payload)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

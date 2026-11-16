"""API Endpoint Indexer Pipeline.

Receives raw API EndpointData, enriches it with LLM-generated context,
creates hybrid embeddings (dense + sparse), and stores the result in Qdrant
api_tool_collection as multiple points per endpoint.

Multi-point indexing strategy:
    - One 'example' point per example query extracted from the LLM context.
      Each query is embedded individually so its vector sits in the correct
      language region of the embedding space, enabling accurate short-query matching.
    - One 'summary' point containing the combined name + description + enriched context.
      This handles broad/paraphrased queries that don't match any single example.

Pipeline steps:
    1. Build LLM prompt from endpoint name, description, and params
    2. Generate context via LLMAPIClient.generate_context()
    3. Parse example query lines from the returned context
    4. Create dense + sparse embeddings per example query (example points)
    5. Create dense + sparse embedding for combined summary text (summary point)
    6. Delete all existing Qdrant points for this endpoint (filter-based, idempotent)
    7. Upsert all points to api_tool_collection
    8. Return IndexingResult
"""

import os
import re
import sys
import json
import asyncio
import argparse
import tempfile
from typing import List, Optional, Sequence, Tuple

from src.loki_logger import LokiLogger

from api_tool_indexer.constants import ApiToolIndexerConstants
from api_tool_indexer.models import (
    EndpointData,
    EnrichedEndpoint,
    IndexingResult,
    IndexingStep,
)
from api_tool_indexer.qdrant_manager import ApiToolQdrantManager

# Reuse LLMAPIClient from intent_data_enrichment.
from intent_data_enrichment.api_client import LLMAPIClient

logger = LokiLogger(service_name="api-tool-calling")
try:
    from tool_classifier.sparse_encoder import compute_sparse_vector
except ImportError:
    try:
        from src.tool_classifier.sparse_encoder import compute_sparse_vector
    except ImportError:
        logger.warning(
            "Could not import sparse_encoder from tool_classifier, "
            "attempting direct import"
        )
        import importlib.util

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


def _build_params_summary(params: list) -> str:
    """Build a readable one-liner of params for LLM prompt and embedding.

    Example output:
        'countryIsoCode (string, required): ISO country code; validFrom (date, required): Start date'

    Args:
        params: List of param dicts with name, type, required, description keys.

    Returns:
        Empty string if no params, otherwise a semicolon-separated summary.
    """
    if not params:
        return "No parameters required."

    parts: List[str] = []
    for p in params:
        name = p.get("name", "unknown")
        ptype = p.get("type", "string")
        required = "required" if p.get("required", False) else "optional"
        description = p.get("description", "")
        parts.append(f"{name} ({ptype}, {required}): {description}")

    return "; ".join(parts)


async def _generate_context_for_endpoint(
    api_client: LLMAPIClient,
    endpoint_data: EndpointData,
) -> str:
    """Generate a rich LLM context for an API endpoint.

    Args:
        api_client: Initialized LLMAPIClient.
        endpoint_data: Raw endpoint data from mock_endpoints table.

    Returns:
        LLM-generated enriched context string.

    Raises:
        RuntimeError: If context generation fails after all retries.
    """
    params_summary = _build_params_summary(endpoint_data.params)

    logger.info(f"params_summary : {params_summary}")

    # Escape braces in the URL to prevent str.format() from treating path
    # parameter templates like {id} as format placeholders (KeyError).
    safe_url = endpoint_data.url.replace("{", "{{").replace("}", "}}")
    full_endpoint_info = (
        f"Endpoint: {endpoint_data.name}\n"
        f"Method: {endpoint_data.method}\n"
        f"URL: {safe_url}\n"
        f"Description: {endpoint_data.description}\n"
        f"Parameters: {params_summary}"
    )

    context_prompt = ApiToolIndexerConstants.CONTEXT_TEMPLATE.format(
        full_endpoint_info=full_endpoint_info,
        name=endpoint_data.name,
        description=endpoint_data.description,
        params_summary=params_summary,
        example_count=ApiToolIndexerConstants.EXAMPLE_QUERY_COUNT,
    )

    logger.debug(
        f"Generated context prompt for endpoint '{endpoint_data.endpoint_id}': "
        f"{len(context_prompt)} chars"
    )

    # context_type="api_tool" makes context_manager use API_TOOL_CONTEXT_PROMPT,
    # which passes chunk_prompt through unmodified so CHUNK_CONTEXT_PROMPT cannot
    # override the instructions in CONTEXT_TEMPLATE (e.g. example query generation).
    request_data = {
        "document_prompt": "",
        "chunk_prompt": context_prompt,
        "environment": api_client.environment,
        "use_cache": False,
        "connection_id": api_client.connection_id,
        "context_type": "api_tool",
    }

    last_error = None
    for attempt in range(api_client.max_retries):
        try:
            logger.info(
                f"Generating context for endpoint '{endpoint_data.endpoint_id}' "
                f"(attempt {attempt + 1}/{api_client.max_retries})"
            )

            if not api_client.session:
                raise RuntimeError("HTTP session not initialized")

            response = await api_client.session.post(
                f"{api_client.api_base_url}/generate-context", json=request_data
            )
            response.raise_for_status()
            result = response.json()

            context = result.get("context", "").strip()

            _preview = context[:200].replace("\n", "\\n") + (
                "..." if len(context) > 200 else ""
            )
            logger.debug(f"context preview: {_preview}")

            if not context:
                raise ValueError("Empty context returned from API")

            logger.success(
                f"Context generated for endpoint '{endpoint_data.endpoint_id}': "
                f"{len(context)} characters"
            )
            return context

        except Exception as e:
            last_error = e
            logger.warning(
                f"Context generation attempt {attempt + 1} failed for "
                f"'{endpoint_data.endpoint_id}': {e}"
            )
            if attempt < api_client.max_retries - 1:
                delay = api_client.retry_delay_base**attempt
                logger.info(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)

    error_msg = (
        f"Context generation failed for '{endpoint_data.endpoint_id}' "
        f"after {api_client.max_retries} attempts: {last_error}"
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg)


_EXAMPLE_SECTION_HEADER = re.compile(r"^example queries\s*:", re.IGNORECASE)


def _parse_example_queries(context: str) -> List[str]:
    """Extract example query lines from the LLM-generated context.

    Scans for the 'Example queries:' section header and collects every
    subsequent '- ' line until the section ends.

    Args:
        context: Raw LLM-generated context string from generate_context().

    Returns:
        List of example query strings, deduplicated and preserving order.
    """
    examples: List[str] = []
    in_section = False

    for line in context.splitlines():
        stripped = line.strip()
        if _EXAMPLE_SECTION_HEADER.match(stripped):
            in_section = True
            continue
        if in_section:
            if stripped.startswith("- "):
                examples.append(stripped[2:].strip())
            elif stripped and not stripped.startswith("#"):
                # Non-empty, non-comment line that isn't a list item ends the section
                in_section = False

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for ex in examples:
        if ex and ex not in seen:
            seen.add(ex)
            unique.append(ex)
    return unique


async def index_endpoint(endpoint_data: EndpointData) -> IndexingResult:
    """Index one API endpoint into Qdrant api_tool_collection.

    Creates multiple points per endpoint:
    - One 'example' point per parsed example query, embedded from that
      individual sentence so the vector sits in the correct language region.
    - One 'summary' point embedded from the combined name + description + context.

    Args:
        endpoint_data: Raw endpoint data from mock_endpoints table.

    Returns:
        IndexingResult indicating success or failure with details. On failure,
        ``failed_step`` names the step that failed (see IndexingStep).
    """
    endpoint_id = endpoint_data.endpoint_id
    # Updated as the pipeline moves on, so any exception can be tied to a step.
    current_step = IndexingStep.ENRICHING
    logger.info(
        f"Starting indexing pipeline for endpoint '{endpoint_id}' "
        f"(name='{endpoint_data.name}')"
    )

    try:
        async with LLMAPIClient(
            api_base_url=ApiToolIndexerConstants.DEFAULT_API_BASE_URL,
            environment=ApiToolIndexerConstants.DEFAULT_ENVIRONMENT,
            connection_id=ApiToolIndexerConstants.DEFAULT_CONNECTION_ID,
            max_retries=ApiToolIndexerConstants.MAX_RETRIES,
            retry_delay_base=ApiToolIndexerConstants.RETRY_DELAY_BASE,
            timeout=ApiToolIndexerConstants.REQUEST_TIMEOUT,
        ) as api_client:
            # Step 1: Generate LLM enriched context (prose + example queries)
            logger.info("Step 1/4: Generating LLM enriched context")
            enriched_context = await _generate_context_for_endpoint(
                api_client, endpoint_data
            )

            # Step 2: Parse example query lines from the context
            example_queries = _parse_example_queries(enriched_context)
            if not example_queries:
                logger.warning(
                    f"No example queries parsed from context for endpoint '{endpoint_id}'. "
                    "The LLM output may not contain an 'Example queries:' section. "
                    "Only a summary point will be indexed — search accuracy may be reduced."
                )
            else:
                logger.info(
                    f"Step 2/4: Parsed {len(example_queries)} example queries from context"
                )

            # Step 3: Embed each example query individually → example points
            current_step = IndexingStep.EMBEDDING
            logger.info(
                f"Step 3/4: Creating embeddings for {len(example_queries)} example points"
            )
            enriched_points: List[EnrichedEndpoint] = []

            for i, example in enumerate(example_queries):
                logger.debug(
                    f"  Embedding example {i + 1}/{len(example_queries)}: "
                    f"'{example[:80]}{'...' if len(example) > 80 else ''}'"
                )
                ex_embedding = await api_client.create_embedding(example)
                ex_sparse = compute_sparse_vector(example)
                enriched_points.append(
                    EnrichedEndpoint(
                        endpoint_id=endpoint_id,
                        name=endpoint_data.name,
                        description=endpoint_data.description,
                        url=endpoint_data.url,
                        method=endpoint_data.method,
                        params=endpoint_data.params,
                        enriched_context=enriched_context,
                        service_id=endpoint_data.service_id,
                        cacheable=endpoint_data.cacheable,
                        cache_ttl_seconds=endpoint_data.cache_ttl_seconds,
                        point_type="example",
                        example_text=example,
                        embedding=ex_embedding,
                        sparse_indices=ex_sparse.indices,
                        sparse_values=ex_sparse.values,
                    )
                )

            # Step 4: Embed combined summary text → summary point
            logger.info("Step 4/4: Creating summary point embedding")
            params_summary = _build_params_summary(endpoint_data.params)
            summary_text = (
                f"{endpoint_data.name}. "
                f"{endpoint_data.description}. "
                f"{enriched_context}. "
                f"Parameters: {params_summary}"
            )
            summary_embedding = await api_client.create_embedding(summary_text)

        # Sparse vectors are CPU-bound — computed after the HTTP session closes
        summary_sparse = compute_sparse_vector(summary_text)
        enriched_points.append(
            EnrichedEndpoint(
                endpoint_id=endpoint_id,
                name=endpoint_data.name,
                description=endpoint_data.description,
                url=endpoint_data.url,
                method=endpoint_data.method,
                params=endpoint_data.params,
                enriched_context=enriched_context,
                service_id=endpoint_data.service_id,
                cacheable=endpoint_data.cacheable,
                cache_ttl_seconds=endpoint_data.cache_ttl_seconds,
                point_type="summary",
                embedding=summary_embedding,
                sparse_indices=summary_sparse.indices,
                sparse_values=summary_sparse.values,
            )
        )

        # Qdrant operations — separate block so the connection is always closed
        current_step = IndexingStep.QDRANT_DELETE
        qdrant = ApiToolQdrantManager()
        try:
            qdrant.connect()
            qdrant.ensure_collection()

            # Delete all existing points for this endpoint (filter-based, idempotent)
            deleted = qdrant.delete_endpoint_points(endpoint_id)
            if not deleted:
                logger.error(
                    f"Failed to delete existing points for endpoint '{endpoint_id}'. "
                    "Aborting upsert to prevent stale data."
                )
                return IndexingResult(
                    success=False,
                    endpoint_id=endpoint_id,
                    message="Qdrant delete failed before upsert",
                    error="delete_endpoint_points returned False",
                    failed_step=IndexingStep.QDRANT_DELETE,
                )

            current_step = IndexingStep.QDRANT_UPSERT
            upserted = qdrant.upsert_endpoint_points(enriched_points)
        finally:
            qdrant.close()

        n_examples = len(example_queries)
        if upserted:
            logger.success(
                f"Endpoint '{endpoint_id}' (name='{endpoint_data.name}') indexed successfully "
                f"({n_examples} example points + 1 summary point)"
            )
            return IndexingResult(
                success=True,
                endpoint_id=endpoint_id,
                message=(
                    f"Endpoint '{endpoint_data.name}' indexed successfully into "
                    f"api_tool_collection "
                    f"({n_examples} example points + 1 summary point)"
                ),
            )
        else:
            return IndexingResult(
                success=False,
                endpoint_id=endpoint_id,
                message="Qdrant upsert failed",
                error=(
                    "upsert_endpoint_points returned False. The old index for this "
                    "endpoint was removed and the new one was not written."
                ),
                failed_step=IndexingStep.QDRANT_UPSERT,
            )

    except Exception as e:
        error = _describe_error(e)
        if current_step == IndexingStep.QDRANT_UPSERT:
            error += (
                " The old index for this endpoint was removed and the new one "
                "was not written."
            )
        logger.error(
            f"Indexing pipeline failed for endpoint '{endpoint_id}' "
            f"at step {current_step}: {error}"
        )
        return IndexingResult(
            success=False,
            endpoint_id=endpoint_id,
            message=f"Indexing pipeline failed at step {current_step}",
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
    indexer's exit code.
    """
    try:
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        os.replace(tmp_path, path)
    except Exception as e:
        logger.error(f"Failed to write result file '{path}': {e}")


def parse_arguments(
    argv: Optional[Sequence[str]] = None,
) -> Tuple[EndpointData, Optional[str], Optional[int]]:
    """Parse command line arguments.

    Returns:
        (EndpointData, result_file path or None, attempt_no or None)
    """
    parser = argparse.ArgumentParser(description="API Tool Indexing")
    parser.add_argument("--endpoint-id", type=str, required=True, help="Endpoint ID")
    parser.add_argument("--name", type=str, required=True, help="Endpoint name")
    parser.add_argument(
        "--description", type=str, required=True, help="Endpoint description"
    )
    parser.add_argument("--url", type=str, required=True, help="Endpoint URL")
    parser.add_argument("--service-id", type=str, default="", help="Parent service ID")
    parser.add_argument("--method", type=str, default="GET", help="HTTP method")
    parser.add_argument("--visibility", type=str, default="public", help="Visibility")
    parser.add_argument(
        "--type", type=str, default="custom_endpoint", help="Endpoint type"
    )
    parser.add_argument("--params-file", type=str, help="Path to params JSON file")
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

    # Read and parse JSON array from file
    params = []
    if args.params_file:
        try:
            with open(args.params_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    params = json.loads(content)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to read/parse params file: {e}")

    endpoint_data = EndpointData(
        endpoint_id=args.endpoint_id,
        service_id=args.service_id,
        name=args.name,
        description=args.description,
        method=args.method,
        url=args.url,
        visibility=args.visibility,
        type=args.type,
        params=params,
    )
    return endpoint_data, args.result_file, args.attempt_no


def _find_result_file_arg(argv: Sequence[str]) -> Optional[str]:
    """Pick --result-file out of raw argv.

    Used so the result file can still be written when argparse or the
    EndpointData validation fails (before parse_arguments returns).
    """
    for i, arg in enumerate(argv):
        if arg == "--result-file" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--result-file="):
            return arg.split("=", 1)[1]
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point for API tool indexing via CLI.

    Whatever happens, if --result-file was given, a JSON file
    ``{success, step, error, message}`` is written on the way out (``finally``),
    so api_tool_indexer.sh can report the exact outcome in its status callback.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    result_file = _find_result_file_arg(argv)
    result: Optional[IndexingResult] = None
    attempt_no: Optional[int] = None
    fatal_error: Optional[str] = None
    exit_code = 1

    logger.info("Starting API Tool indexing pipeline...")

    try:
        # Parse arguments
        endpoint_data, result_file, attempt_no = parse_arguments(argv)
        logger.info(f"Endpoint ID: {endpoint_data.endpoint_id}")
        logger.info(f"Endpoint Name: {endpoint_data.name}")
        logger.info(f"Attempt No: {attempt_no}")
        logger.info(f"Params count: {len(endpoint_data.params)}")

        # Run indexing pipeline
        result = asyncio.run(index_endpoint(endpoint_data))

        # Log results
        if result.success:
            logger.success("Indexing completed successfully")
            logger.info(f"Endpoint: {result.endpoint_id}")
            logger.info(f"Message: {result.message}")
            exit_code = 0
        else:
            logger.error("Indexing failed")
            logger.error(f"Endpoint: {result.endpoint_id}")
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
                    "error": fatal_error or "Indexer exited before producing a result",
                    "message": "Indexer failed before the pipeline started",
                    "attemptNo": attempt_no,
                }
            write_result_file(result_file, payload)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

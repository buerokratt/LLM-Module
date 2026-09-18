"""Service workflow executor - Layer 1: External service/API calls."""

import json
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Union

import dspy
import httpx
from langfuse import observe
from src.loki_logger import LokiLogger

from llm_orchestrator_config.llm_manager import LLMManager
from src.guardrails.nemo_rails_adapter import NeMoRailsAdapter

from src.utils.conversation_history_helpers import get_conversation_history
from src.utils.conversation_history_store import ConversationHistoryStore
from src.utils.cost_utils import get_lm_usage_since
from src.utils.observation_utils import update_observation_safe

from models.request_models import (
    ChoiceButton,
    OrchestrationRequest,
    OrchestrationResponse,
    TestOrchestrationResponse,
)
from tool_classifier.base_workflow import BaseWorkflow
from tool_classifier.constants import (
    MAX_SERVICES_FOR_LLM_CONTEXT,
    QDRANT_COLLECTION,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_TIMEOUT,
    RAG_SEARCH_RUUTER_PUBLIC,
    RUUTER_COMMON_SERVICE_BASE_URL,
    RUUTER_SERVICE_BASE_URL,
    SEMANTIC_SEARCH_THRESHOLD,
    SEMANTIC_SEARCH_TOP_K,
    SERVICE_CALL_TIMEOUT,
    SERVICE_COUNT_THRESHOLD,
    SERVICE_DISCOVERY_TIMEOUT,
    SERVICE_STEP_PREFIXES,
)
from tool_classifier.intent_detector import IntentDetectionModule
import time

# Initialize Loki logger
logger = LokiLogger(service_name="service-workflow")

SERVICE_INTENT_DETECTION_METRIC = "service.intent_detection"


class LLMServiceProtocol(Protocol):
    """Protocol defining interface for LLM service embedding operations."""

    def create_embeddings_for_indexer(
        self,
        texts: List[str],
        environment: str = "production",
        connection_id: Optional[str] = None,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """Create embeddings for text inputs using the configured embedding model.

        Args:
            texts: List of text strings to embed
            environment: Environment for model resolution
            connection_id: Optional connection ID for service selection
            batch_size: Number of texts to process in each batch

        Returns:
            Dictionary containing embeddings list and metadata
        """
        ...

    def format_sse(
        self,
        chat_id: str,
        content: str,
        buttons: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Format content as SSE message.

        Args:
            chat_id: Chat/channel identifier
            content: Content to send (token, "END", error message, etc.)
            buttons: Optional list of choice button dicts for MCQ step responses

        Returns:
            SSE-formatted string: "data: {json}\\n\\n"
        """
        ...

    def log_costs(self, costs_metric: Dict[str, Dict[str, Any]]) -> None:
        """Log cost information for tracking.

        Args:
            costs_metric: Dictionary of costs per component
        """
        ...

    def _initialize_service_components(
        self, request: OrchestrationRequest
    ) -> Dict[str, Any]:
        """Initialize and return service components dictionary."""
        ...

    async def handle_output_guardrails(
        self,
        guardrails_adapter: Optional[NeMoRailsAdapter],
        generated_response: Union[OrchestrationResponse, TestOrchestrationResponse],
        request: OrchestrationRequest,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> Union[OrchestrationResponse, TestOrchestrationResponse]:
        """Apply output guardrails to the generated response."""
        ...

    async def store_streaming_inference(
        self,
        request: OrchestrationRequest,
        final_answer: str,
    ) -> None:
        """Store streaming inference data for production/testing environments."""
        ...


class ServiceWorkflowExecutor(BaseWorkflow):
    """Executes external service calls via Ruuter endpoints (Layer 1)."""

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        orchestration_service: Optional[LLMServiceProtocol] = None,
    ) -> None:
        """Initialize service workflow executor."""
        self.llm_manager = llm_manager
        self.orchestration_service = orchestration_service

    def _get_conversation_history_store(self) -> Optional[ConversationHistoryStore]:
        """Return the conversation history store from the orchestration service, or None."""
        if self.orchestration_service is None:
            return None
        return getattr(self.orchestration_service, "conversation_history_store", None)

    async def _semantic_search_services(
        self,
        query: str,
        request: OrchestrationRequest,
        chat_id: str,
        top_k: int = SEMANTIC_SEARCH_TOP_K,
    ) -> Optional[List[Dict[str, Any]]]:
        """Search services using semantic search via Qdrant.

        Creates a new httpx.AsyncClient per request to ensure proper resource cleanup.
        This is safe and efficient since semantic search is infrequent (only when many services exist).
        """
        if not self.orchestration_service:
            logger.error(
                f"[{chat_id}] Semantic search unavailable: orchestration service not provided"
            )
            return None

        try:
            embedding_result = self.orchestration_service.create_embeddings_for_indexer(
                texts=[query],
                environment=request.environment,
                connection_id=request.connection_id,
                batch_size=1,
            )

            embeddings = embedding_result.get("embeddings", [])
            if not embeddings or len(embeddings) == 0:
                logger.error(f"[{chat_id}] No embedding returned for query")
                return None

            query_embedding = embeddings[0]

            qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
            async with httpx.AsyncClient(
                base_url=qdrant_url, timeout=QDRANT_TIMEOUT
            ) as client:
                try:
                    collection_info = await client.get(
                        f"/collections/{QDRANT_COLLECTION}"
                    )
                    if collection_info.status_code == 200:
                        info = collection_info.json()
                        points_count = info.get("result", {}).get("points_count", 0)
                        if points_count == 0:
                            logger.error(f"[{chat_id}] Collection is empty")
                            return None
                except Exception as e:
                    logger.warning(f"[{chat_id}] Could not verify collection: {e}")

                search_payload = {
                    "vector": query_embedding,
                    "limit": top_k,
                    "score_threshold": SEMANTIC_SEARCH_THRESHOLD,
                    "with_payload": True,
                }

                response = await client.post(
                    f"/collections/{QDRANT_COLLECTION}/points/search",
                    json=search_payload,
                )

                if response.status_code != 200:
                    logger.error(
                        f"[{chat_id}] Qdrant search failed: HTTP {response.status_code}"
                    )
                    return None

                search_results = response.json()
                points = search_results.get("result", [])

                if len(points) == 0:
                    logger.warning(
                        f"[{chat_id}] No services matched (threshold={SEMANTIC_SEARCH_THRESHOLD})"
                    )
                    return None

                services: List[Dict[str, Any]] = []
                for point in points:
                    payload = point.get("payload", {})
                    score = float(point.get("score", 0))

                    service = {
                        "serviceId": payload.get("service_id"),
                        "service_id": payload.get("service_id"),
                        "name": payload.get("name"),
                        "description": payload.get("description"),
                        "examples": payload.get("examples", []),
                        "entities": payload.get("entities", []),
                        "similarity_score": score,
                    }
                    services.append(service)

                logger.info(
                    f"[{chat_id}] Found {len(services)} services via semantic search"
                )
                return services

        except Exception as e:
            logger.error(f"[{chat_id}] Semantic search failed: {e}", exc_info=True)
            return None

    async def _call_service_discovery(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Call Ruuter endpoint to get services for intent detection."""
        endpoint = f"{RAG_SEARCH_RUUTER_PUBLIC}/services/get-services"

        try:
            async with httpx.AsyncClient(timeout=SERVICE_DISCOVERY_TIMEOUT) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                data = response.json()
                return data
        except httpx.TimeoutException:
            logger.error(
                f"[{chat_id}] Service discovery timeout after {SERVICE_DISCOVERY_TIMEOUT}s"
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[{chat_id}] Service discovery HTTP error: {e.response.status_code}"
            )
            return None
        except Exception as e:
            logger.error(f"[{chat_id}] Service discovery failed: {e}", exc_info=True)
            return None

    @observe(name="service_intent_detection_orchestration", as_type="generation")
    async def _detect_service_intent(
        self,
        user_query: str,
        services: List[Dict[str, Any]],
        conversation_history: List[Any],
        chat_id: str,
        conversation_summary: Optional[str] = None,
    ) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Use DSPy + LLMManager to detect service intent and extract entities.

        Args:
            user_query: The user's query string.
            services: List of available service dicts.
            conversation_history: Recent conversation turns (``ConversationItem`` objects).
            chat_id: Chat identifier for logging.
            conversation_summary: Optional summary of earlier conversation rounds
                evicted from Redis.  When provided it is prepended to the history
                passed to the intent detector so the LLM has additional context
                without a separate summarisation call.

        Returns:
            Tuple of (intent_result, usage_info):
                - intent_result: Intent detection result dict (or None on error)
                - usage_info: Cost and token usage information
        """
        try:
            if self.llm_manager:
                self.llm_manager.ensure_global_config()
            else:
                logger.error(f"[{chat_id}] LLM Manager not available")
                return None, {}

            lm = dspy.settings.lm
            history_length_before = (
                len(lm.history) if lm and hasattr(lm, "history") else 0
            )

            intent_module = IntentDetectionModule()
            history_dicts: List[Dict[str, str]] = []
            if conversation_summary:
                history_dicts.append(
                    {
                        "authorRole": "system",
                        "message": f"Summary of earlier conversation: {conversation_summary}",
                    }
                )
            history_dicts.extend(
                [
                    {"authorRole": msg.authorRole, "message": msg.message}
                    for msg in conversation_history
                    if hasattr(msg, "authorRole") and hasattr(msg, "message")
                ]
            )

            with self.llm_manager.use_task_local():
                intent_result = intent_module.forward(
                    user_query=user_query,
                    services=services,
                    conversation_history=history_dicts,
                )

            usage_info = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "chat_id": chat_id,
                    "query": user_query,
                    "services_count": len(services),
                },
                output_data={
                    "matched_service_id": (
                        intent_result.get("matched_service_id")
                        if intent_result
                        else None
                    ),
                    "confidence": intent_result.get("confidence", 0.0)
                    if intent_result
                    else 0.0,
                },
                metadata={"usage": usage_info},
            )

            return intent_result, usage_info

        except Exception as e:
            logger.error(f"[{chat_id}] Intent detection failed: {e}", exc_info=True)
            update_observation_safe(
                input_data={
                    "chat_id": chat_id,
                    "query": user_query,
                    "services_count": len(services),
                },
                output_data={"matched_service_id": None, "error": str(e)},
                metadata={"usage": {}},
            )
            return None, {}

    def _validate_detected_service(
        self,
        matched_service_id: str,
        services: List[Dict[str, Any]],
        chat_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate that detected service exists in active services list."""
        for service in services:
            service_id = service.get("serviceId", service.get("service_id"))
            if service_id == matched_service_id:
                return service

        logger.warning(
            f"[{chat_id}] Service validation failed: '{matched_service_id}' not found"
        )
        return None

    async def _process_intent_detection(
        self,
        services: List[Dict[str, Any]],
        request: OrchestrationRequest,
        chat_id: str,
        context: Dict[str, Any],
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> None:
        """Detect intent, validate service, and populate context.

        This helper method encapsulates the common logic of:
        1. Calling intent detection (LLM)
        2. Tracking costs
        3. Validating matched service
        4. Populating context with service metadata

        Args:
            services: List of services to match against
            request: Orchestration request
            chat_id: Chat ID for logging
            context: Context dict to populate with results
            costs_metric: Dictionary to track LLM costs
        """
        conversation_history, conversation_summary = await get_conversation_history(
            chat_id=request.chatId,
            store=self._get_conversation_history_store(),
            fallback=request.conversationHistory,
        )
        intent_result, intent_usage = await self._detect_service_intent(
            user_query=request.message,
            services=services,
            conversation_history=conversation_history,
            chat_id=chat_id,
            conversation_summary=conversation_summary,
        )
        costs_metric["intent_detection"] = intent_usage

        if intent_result and intent_result.get("matched_service_id"):
            service_id = intent_result["matched_service_id"]
            logger.info(f"[{chat_id}] Matched: {service_id}")

            validated_service = self._validate_detected_service(
                matched_service_id=service_id,
                services=services,
                chat_id=chat_id,
            )

            if validated_service:
                context["service_id"] = service_id
                context["confidence"] = intent_result.get("confidence", 0.0)
                context["entities"] = intent_result.get("entities", {})
                context["service_data"] = validated_service

    def _extract_service_metadata(
        self, context: Dict[str, Any], chat_id: str
    ) -> Optional[Dict[str, Any]]:
        """Extract service and entity metadata from context."""
        service_id = context.get("service_id")
        if not service_id:
            logger.error(f"[{chat_id}] Missing service_id in context")
            return None

        service_data = context.get("service_data")
        if not service_data:
            logger.error(f"[{chat_id}] Missing service_data in context")
            return None

        entities_dict = context.get("entities", {})
        entity_schema = service_data.get("entities", []) or []
        service_name = service_data.get("name", service_id)
        ruuter_type = service_data.get("ruuter_type", "POST")
        is_common = bool(service_data.get("is_common", False))

        return {
            "service_id": service_id,
            "service_name": service_name,
            "entities_dict": entities_dict,
            "entity_schema": entity_schema,
            "ruuter_type": ruuter_type,
            "is_common": is_common,
            "service_data": service_data,
        }

    def _validate_entities(
        self,
        extracted_entities: Dict[str, str],
        service_schema: List[str],
        service_name: str,
        chat_id: str,
    ) -> Dict[str, Any]:
        """
        Validate extracted entities against service schema.

        Args:
            extracted_entities: Entity key-value pairs from LLM
            service_schema: Expected entity keys from database
            service_name: Service name for logging
            chat_id: For logging

        Returns:
            Dict with validation results:
            - is_valid: Overall validation status
            - missing_entities: List of schema entities not extracted
            - extra_entities: List of extracted entities not in schema
            - validation_errors: List of error messages
        """
        missing_entities = []
        extra_entities = []
        validation_errors = []

        # Check for missing entities (in schema but not extracted)
        for schema_key in service_schema:
            if schema_key not in extracted_entities:
                missing_entities.append(schema_key)
            elif extracted_entities[schema_key] == "":
                # Entity extracted but value is empty
                validation_errors.append(f"Entity '{schema_key}' has empty value")

        # Check for extra entities (extracted but not in schema)
        extra_entities = [
            entity_key
            for entity_key in extracted_entities
            if entity_key not in service_schema
        ]

        is_valid = True

        return {
            "is_valid": is_valid,
            "missing_entities": missing_entities,
            "extra_entities": extra_entities,
            "validation_errors": validation_errors,
        }

    def _transform_entities_to_array(
        self, entities_dict: Dict[str, str], entity_order: List[str]
    ) -> List[str]:
        """Transform entity dictionary to ordered array based on service schema."""
        if not entity_order:
            return []
        return [entities_dict.get(key, "") for key in entity_order]

    _INVISIBLE_CHAR_TABLE = str.maketrans(
        "", "", "\u2060\u200b\u200c\u200d\ufeff\u00ad\u200e\u200f"
    )

    @staticmethod
    def _parse_service_prefix(
        payload: str,
    ) -> Optional[tuple[str, str]]:
        """Parse a ``#service`` or ``#common_service`` button-payload into an ``(http_method, url)`` tuple.

        Extracts the HTTP method and resource path from payloads of the form
        ``"#service, /POST/services/active/<step_name>"`` and appends the path to
        the appropriate base URL (``RUUTER_COMMON_SERVICE_BASE_URL`` for
        ``#common_service`` prefixes, ``RUUTER_SERVICE_BASE_URL`` otherwise).
        Returns ``None`` for any malformed input.
        """
        stripped = payload.strip()

        # Identify and remove the prefix
        matched_prefix: Optional[str] = None
        for prefix in SERVICE_STEP_PREFIXES:
            if stripped.startswith(prefix):
                matched_prefix = prefix
                break

        if matched_prefix is None:
            return None

        # Remainder after prefix, e.g. " /POST/services/active/foo"
        remainder = stripped[len(matched_prefix) :].strip()

        # Must start with '/' followed by the HTTP method
        if not remainder.startswith("/"):
            return None

        # Split into segments: ['', 'POST', 'services', 'active', 'foo']
        segments = remainder.split("/")
        # segments[0] == '' (empty before leading /);
        # segments[1] == HTTP method; segments[2:] == resource path parts
        if len(segments) < 3:  # noqa: PLR2004
            return None

        http_method = segments[1].upper()
        if not http_method.isalpha():
            return None

        resource_path = "/" + "/".join(segments[2:])
        base_url = (
            RUUTER_COMMON_SERVICE_BASE_URL
            if matched_prefix.startswith("#common_service")
            else RUUTER_SERVICE_BASE_URL
        )
        full_url = f"{base_url}{resource_path}"

        return (http_method, full_url)

    def _construct_service_endpoint(
        self, service_name: str, chat_id: str, is_common: bool = False
    ) -> str:
        """Construct the full service endpoint URL for Ruuter.

        Args:
            service_name: Name of the service to call.
            chat_id: Chat ID for logging.
            is_common: When True, routes to the common-service Ruuter base URL.
        """
        clean_name = (
            service_name.strip().translate(self._INVISIBLE_CHAR_TABLE).replace(" ", "_")
        )
        base_url = (
            RUUTER_COMMON_SERVICE_BASE_URL if is_common else RUUTER_SERVICE_BASE_URL
        )
        service_type = "common" if is_common else "regular"
        logger.debug(
            f"[{chat_id}] Routing to {service_type} service base URL: {base_url}"
        )
        return f"{base_url}/services/active/{clean_name}"

    async def _call_service_endpoint(
        self,
        endpoint_url: str,
        http_method: str,
        entities_array: List[str],
        chat_id: str,
        author_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Call the Ruuter active service endpoint and extract response content.

        Args:
            endpoint_url: Full URL of the active service endpoint
            http_method: HTTP method (POST/GET)
            entities_array: Ordered entity values for the service
            chat_id: Chat session ID
            author_id: Author/user ID

        Returns:
            Dict with "content" (str) and "buttons" (List[Dict]) keys, or None on failure.
        """
        payload = {
            "chatId": chat_id,
            "authorId": author_id,
            "input": entities_array,
        }

        try:
            async with httpx.AsyncClient(timeout=SERVICE_CALL_TIMEOUT) as client:
                if http_method.upper() == "POST":
                    response = await client.post(endpoint_url, json=payload)
                else:
                    response = await client.get(endpoint_url, params=payload)

                response.raise_for_status()
                data = response.json()

                # Ruuter wraps the DSL return value in {"response": ...}
                # The inner value is the DMapper array from bot_responses_to_messages
                if isinstance(data, dict) and "response" in data:
                    data = data["response"]
                # "buttons" is a JSON-encoded string, not a sub-array.
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    content = item.get("content", "")
                    raw_buttons = item.get("buttons", "[]") or "[]"

                    if isinstance(raw_buttons, str):
                        try:
                            buttons = json.loads(raw_buttons)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{chat_id}] Failed to parse buttons JSON string: {raw_buttons}"
                            )
                            buttons = []
                    elif isinstance(raw_buttons, list):
                        buttons = raw_buttons
                    else:
                        buttons = []

                    if not content:
                        logger.warning(
                            f"[{chat_id}] Service response missing 'content' field"
                        )
                    logger.info(
                        f"[{chat_id}] Service endpoint returned content "
                        f"({len(content)} chars, {len(buttons)} buttons)"
                    )
                    logger.debug(f"[{chat_id}] Parsed buttons: {buttons}")
                    return {"content": content, "buttons": buttons}

                logger.warning(
                    f"[{chat_id}] Unexpected service response format: {type(data)}"
                )
                return None

        except httpx.TimeoutException:
            logger.error(
                f"[{chat_id}] Service endpoint timeout after {SERVICE_CALL_TIMEOUT}s: "
                f"{endpoint_url}"
            )
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                f"[{chat_id}] Service endpoint HTTP error: "
                f"{e.response.status_code} for {endpoint_url}"
            )
            return None
        except Exception as e:
            logger.error(
                f"[{chat_id}] Service endpoint call failed: {e}",
                exc_info=True,
            )
            return None

    async def _log_request_details(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        mode: str,
        costs_metric: Dict[str, Dict[str, Any]],
    ) -> None:
        """Log request details and perform service discovery.

        Args:
            request: The orchestration request
            context: Workflow context dictionary
            mode: Execution mode ("streaming" or "non-streaming")
            costs_metric: Dictionary to accumulate cost tracking information
        """
        chat_id = request.chatId
        logger.info(f"[{chat_id}] SERVICE WORKFLOW ({mode}): {request.message}")

        discovery_result = await self._call_service_discovery(chat_id)

        if discovery_result:
            response_data = discovery_result.get("response", {})
            use_semantic = response_data.get("use_semantic_search", False)
            service_count = response_data.get("service_count", 0)

            if isinstance(service_count, str):
                try:
                    service_count = int(service_count)
                except (ValueError, TypeError):
                    service_count = 0

            services_from_ruuter = response_data.get("services", [])

            if service_count > SERVICE_COUNT_THRESHOLD:
                use_semantic = True

            if use_semantic:
                services = await self._semantic_search_services(
                    query=request.message,
                    request=request,
                    chat_id=chat_id,
                    top_k=SEMANTIC_SEARCH_TOP_K,
                )

                if not services:
                    logger.warning(f"[{chat_id}] Semantic search failed")

                    if services_from_ruuter:
                        services = services_from_ruuter
                    elif service_count <= MAX_SERVICES_FOR_LLM_CONTEXT:
                        fallback_result = await self._call_service_discovery(chat_id)
                        if fallback_result:
                            fallback_data = fallback_result.get("response", {})
                            services = fallback_data.get("services", [])
                        else:
                            services = []
                    else:
                        logger.error(f"[{chat_id}] Too many services ({service_count})")
                        services = []

                if services:
                    await self._process_intent_detection(
                        services=services,
                        request=request,
                        chat_id=chat_id,
                        context=context,
                        costs_metric=costs_metric,
                    )
            else:
                services = response_data.get("services", [])

                if services:
                    await self._process_intent_detection(
                        services=services,
                        request=request,
                        chat_id=chat_id,
                        context=context,
                        costs_metric=costs_metric,
                    )
        else:
            logger.warning(f"[{chat_id}] Service discovery failed")

    @observe(name="service_workflow_execute_async", as_type="span")
    async def execute_async(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[OrchestrationResponse]:
        """Execute service workflow in non-streaming mode.

        Uses classification metadata from hybrid search:
        - needs_llm_confirmation=False: Skip discovery + intent detection, use matched service
        - needs_llm_confirmation=True: Run LLM intent detection on candidate services only
        - No metadata: Fall back to original discovery flow

        Args:
            request: Orchestration request
            context: Workflow context
            time_metric: Optional timing dictionary for unified tracking
        """

        chat_id = request.chatId

        costs_metric: Dict[str, Dict[str, Any]] = {}
        if time_metric is None:
            time_metric = {}

        needs_llm_confirmation = context.get("needs_llm_confirmation")

        if needs_llm_confirmation is False:
            matched_service_name = context.get("matched_service_name")
            cosine_score = context.get("cosine_score", 0.0)

            logger.info(
                f"[{chat_id}] High-confidence service match: "
                f"{matched_service_name} (score={cosine_score:.4f})"
            )

            top_results = context.get("top_results", [])
            if top_results:
                matched = top_results[0]

                start_time = time.time()
                await self._process_intent_detection(
                    services=[matched],
                    request=request,
                    chat_id=chat_id,
                    context=context,
                    costs_metric=costs_metric,
                )
                time_metric[SERVICE_INTENT_DETECTION_METRIC] = time.time() - start_time

                if not context.get("service_data"):
                    context["service_id"] = matched.get("service_id")
                    context["service_data"] = matched

        elif needs_llm_confirmation is True:
            top_results = context.get("top_results", [])
            logger.info(
                f"[{chat_id}] Ambiguous match: "
                f"running intent detection on {len(top_results)} candidates"
            )

            start_time = time.time()
            if top_results:
                await self._process_intent_detection(
                    services=top_results,
                    request=request,
                    chat_id=chat_id,
                    context=context,
                    costs_metric=costs_metric,
                )
            time_metric[SERVICE_INTENT_DETECTION_METRIC] = time.time() - start_time

        else:
            start_time = time.time()
            await self._log_request_details(
                request, context, mode="non-streaming", costs_metric=costs_metric
            )
            time_metric["service.discovery"] = time.time() - start_time

        if not context.get("service_id"):
            logger.info(f"[{chat_id}] No service matched, falling back")
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "fallback_to_rag"},
                metadata={"costs": costs_metric},
            )
            return None

        start_time = time.time()
        service_metadata = self._extract_service_metadata(context, chat_id)
        if not service_metadata:
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "missing_service_metadata"},
                metadata={"costs": costs_metric},
            )
            return None

        logger.info(
            f"[{chat_id}] Service: {service_metadata['service_name']}, "
            f"entities: {service_metadata['entities_dict']}"
        )

        validation_result = self._validate_entities(
            extracted_entities=service_metadata["entities_dict"],
            service_schema=service_metadata["entity_schema"],
            service_name=service_metadata["service_name"],
            chat_id=chat_id,
        )
        time_metric["service.entity_validation"] = time.time() - start_time

        if validation_result["missing_entities"]:
            logger.warning(
                f"[{chat_id}] Missing entities: {validation_result['missing_entities']}"
            )

        entities_array = self._transform_entities_to_array(
            entities_dict=service_metadata["entities_dict"],
            entity_order=service_metadata["entity_schema"],
        )

        context["entities_array"] = entities_array
        context["validation_result"] = validation_result

        endpoint_url = self._construct_service_endpoint(
            service_name=service_metadata["service_name"],
            chat_id=chat_id,
            is_common=service_metadata["is_common"],
        )
        context["endpoint_url"] = endpoint_url
        context["http_method"] = service_metadata["ruuter_type"]

        start_time = time.time()
        service_result = await self._call_service_endpoint(
            endpoint_url=endpoint_url,
            http_method=service_metadata["ruuter_type"],
            entities_array=entities_array,
            chat_id=chat_id,
            author_id=request.authorId,
        )
        time_metric["service.endpoint_call"] = time.time() - start_time

        if self.orchestration_service:
            self.orchestration_service.log_costs(costs_metric)

        if service_result is None:
            logger.warning(f"[{chat_id}] Service endpoint call failed, falling back")
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "fallback_to_rag"},
                metadata={"costs": costs_metric},
            )
            return None

        update_observation_safe(
            input_data={"chat_id": chat_id, "query": request.message},
            output_data={
                "workflow_result": "service_response",
                "service_id": context.get("service_id"),
            },
            metadata={"costs": costs_metric},
        )

        service_content = service_result["content"]
        service_buttons = service_result["buttons"]
        buttons_list = [
            ChoiceButton(**b)
            for b in service_buttons
            if "title" in b and "payload" in b
        ]

        return OrchestrationResponse(
            chatId=request.chatId,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=service_content,
            buttons=buttons_list if buttons_list else None,
        )

    @observe(
        name="service_workflow_execute_streaming",
        as_type="span",
        capture_output=False,
    )
    async def execute_streaming(
        self,
        request: OrchestrationRequest,
        context: Dict[str, Any],
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """Execute service workflow in streaming mode.

        Uses classification metadata from hybrid search (same as execute_async).

        Args:
            request: Orchestration request
            context: Workflow context
            time_metric: Optional timing dictionary for unified tracking
        """

        chat_id = request.chatId

        costs_metric: Dict[str, Dict[str, Any]] = {}
        if time_metric is None:
            time_metric = {}

        needs_llm_confirmation = context.get("needs_llm_confirmation")

        if needs_llm_confirmation is False:
            matched_service_name = context.get("matched_service_name")
            cosine_score = context.get("cosine_score", 0.0)

            logger.info(
                f"[{chat_id}] High-confidence service match: "
                f"{matched_service_name} (score={cosine_score:.4f})"
            )

            top_results = context.get("top_results", [])
            if top_results:
                matched = top_results[0]

                start_time = time.time()
                await self._process_intent_detection(
                    services=[matched],
                    request=request,
                    chat_id=chat_id,
                    context=context,
                    costs_metric=costs_metric,
                )
                time_metric[SERVICE_INTENT_DETECTION_METRIC] = time.time() - start_time

                if not context.get("service_data"):
                    context["service_id"] = matched.get("service_id")
                    context["service_data"] = matched

        elif needs_llm_confirmation is True:
            top_results = context.get("top_results", [])
            logger.info(
                f"[{chat_id}] Ambiguous match: "
                f"running intent detection on {len(top_results)} candidates"
            )

            start_time = time.time()
            if top_results:
                await self._process_intent_detection(
                    services=top_results,
                    request=request,
                    chat_id=chat_id,
                    context=context,
                    costs_metric=costs_metric,
                )
            time_metric[SERVICE_INTENT_DETECTION_METRIC] = time.time() - start_time

        else:
            start_time = time.time()
            await self._log_request_details(
                request, context, mode="streaming", costs_metric=costs_metric
            )
            time_metric["service.discovery"] = time.time() - start_time

        if not context.get("service_id"):
            logger.info(f"[{chat_id}] No service matched, falling back")
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "fallback_to_rag"},
                metadata={"costs": costs_metric},
            )
            return None

        service_metadata = self._extract_service_metadata(context, chat_id)
        if not service_metadata:
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "missing_service_metadata"},
                metadata={"costs": costs_metric},
            )
            return None

        logger.info(
            f"[{chat_id}] Service: {service_metadata['service_name']}, "
            f"entities: {service_metadata['entities_dict']}"
        )

        validation_result = self._validate_entities(
            extracted_entities=service_metadata["entities_dict"],
            service_schema=service_metadata["entity_schema"],
            service_name=service_metadata["service_name"],
            chat_id=chat_id,
        )

        if validation_result["missing_entities"]:
            logger.warning(
                f"[{chat_id}] Missing entities: {validation_result['missing_entities']}"
            )

        entities_array = self._transform_entities_to_array(
            entities_dict=service_metadata["entities_dict"],
            entity_order=service_metadata["entity_schema"],
        )

        context["entities_array"] = entities_array
        context["validation_result"] = validation_result

        endpoint_url = self._construct_service_endpoint(
            service_name=service_metadata["service_name"],
            chat_id=chat_id,
            is_common=service_metadata["is_common"],
        )
        context["endpoint_url"] = endpoint_url
        context["http_method"] = service_metadata["ruuter_type"]

        service_result = await self._call_service_endpoint(
            endpoint_url=endpoint_url,
            http_method=service_metadata["ruuter_type"],
            entities_array=entities_array,
            chat_id=chat_id,
            author_id=request.authorId,
        )

        if service_result is None:
            logger.warning(f"[{chat_id}] Service endpoint call failed, falling back")
            update_observation_safe(
                input_data={"chat_id": chat_id, "query": request.message},
                output_data={"workflow_result": "fallback_to_rag"},
                metadata={"costs": costs_metric},
            )
            return None

        if self.orchestration_service is None:
            raise RuntimeError("Orchestration service not initialized for streaming")

        orchestration_service = self.orchestration_service
        service_content = service_result["content"]
        service_buttons = service_result["buttons"]

        async def service_stream() -> AsyncIterator[str]:
            yield orchestration_service.format_sse(
                chat_id, service_content, service_buttons or None
            )
            await orchestration_service.store_streaming_inference(
                request, service_content
            )
            yield orchestration_service.format_sse(chat_id, "END")
            orchestration_service.log_costs(costs_metric)

        update_observation_safe(
            input_data={"chat_id": chat_id, "query": request.message},
            output_data={
                "workflow_result": "service_stream",
                "service_id": context.get("service_id"),
            },
            metadata={"costs": costs_metric},
        )
        return service_stream()

    async def execute_direct_step(
        self,
        request: OrchestrationRequest,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[OrchestrationResponse]:
        """Execute a direct service step from a #service button payload.

        Bypasses discovery, intent detection, and entity extraction entirely.
        The endpoint URL and HTTP method are parsed directly from the payload
        string embedded in the button click.

        Args:
            request: Orchestration request whose message is a #service payload.
            time_metric: Optional timing dictionary for unified tracking.

        Returns:
            OrchestrationResponse with content and buttons, or None on failure.
        """
        chat_id = request.chatId
        if time_metric is None:
            time_metric = {}

        parsed = self._parse_service_prefix(request.message)
        if parsed is None:
            logger.warning(
                f"[{chat_id}] Failed to parse #service prefix: {request.message}"
            )
            return None

        http_method, endpoint_url = parsed
        logger.info(f"[{chat_id}] DIRECT STEP: {endpoint_url}")

        start_time = time.time()
        service_result = await self._call_service_endpoint(
            endpoint_url=endpoint_url,
            http_method=http_method,
            entities_array=[],
            chat_id=chat_id,
            author_id=request.authorId,
        )
        time_metric["service.direct_step"] = time.time() - start_time

        if service_result is None:
            logger.warning(
                f"[{chat_id}] Direct step endpoint call failed: {endpoint_url}"
            )
            return None

        service_content = service_result["content"]
        service_buttons = service_result["buttons"]
        buttons_list = [
            ChoiceButton(**b)
            for b in service_buttons
            if "title" in b and "payload" in b
        ]

        return OrchestrationResponse(
            chatId=chat_id,
            llmServiceActive=True,
            questionOutOfLLMScope=False,
            inputGuardFailed=False,
            content=service_content,
            buttons=buttons_list if buttons_list else None,
        )

    async def execute_direct_step_streaming(
        self,
        request: OrchestrationRequest,
        time_metric: Optional[Dict[str, float]] = None,
    ) -> Optional[AsyncIterator[str]]:
        """Execute a direct service step and return an SSE stream.

        Same logic as execute_direct_step but wraps the response in an SSE
        async generator suitable for the streaming endpoint.

        Args:
            request: Orchestration request whose message is a #service payload.
            time_metric: Optional timing dictionary for unified tracking.

        Returns:
            AsyncIterator yielding SSE-formatted strings, or None on failure.
        """
        chat_id = request.chatId
        if time_metric is None:
            time_metric = {}

        parsed = self._parse_service_prefix(request.message)
        if parsed is None:
            logger.warning(
                f"[{chat_id}] Failed to parse #service prefix: {request.message}"
            )
            return None

        http_method, endpoint_url = parsed
        logger.info(f"[{chat_id}] DIRECT STEP (stream): {endpoint_url}")

        start_time = time.time()
        service_result = await self._call_service_endpoint(
            endpoint_url=endpoint_url,
            http_method=http_method,
            entities_array=[],
            chat_id=chat_id,
            author_id=request.authorId,
        )
        time_metric["service.direct_step"] = time.time() - start_time

        if service_result is None:
            logger.warning(
                f"[{chat_id}] Direct step endpoint call failed: {endpoint_url}"
            )
            return None

        if self.orchestration_service is None:
            raise RuntimeError("Orchestration service not initialized for streaming")

        orchestration_service = self.orchestration_service
        service_content = service_result["content"]
        service_buttons = service_result["buttons"]

        async def step_stream() -> AsyncIterator[str]:
            yield orchestration_service.format_sse(
                chat_id, service_content, service_buttons or None
            )
            await orchestration_service.store_streaming_inference(
                request, service_content
            )
            yield orchestration_service.format_sse(chat_id, "END")

        return step_stream()

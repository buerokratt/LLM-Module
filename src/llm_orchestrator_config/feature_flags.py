"""Feature flags for tool classifier system."""

import os
from src.loki_logger import LokiLogger

# Initialize Loki logger
logger = LokiLogger(service_name="feature-flags")


class FeatureFlags:
    """
    Feature flags for controlling tool classifier and workflow behavior.

    These flags enable safe deployment and gradual rollout of the multi-workflow
    system. They can be controlled via environment variables.

    Deployment Strategy:
    1. Start with TOOL_CLASSIFIER_ENABLED=false (use existing RAG only)
    2. Enable classifier with all workflows disabled for testing
    3. Enable workflows one at a time (SERVICE → CONTEXT → etc.)
    4. Monitor and rollback if issues occur

    Environment Variables:
    - TOOL_CLASSIFIER_ENABLED: Master switch for classifier (default: false)
    - SERVICE_WORKFLOW_ENABLED: Enable Layer 1 service workflow (default: true)
    - API_TOOL_CALLING_WORKFLOW_ENABLED: Enable Layer 2 API tool calling workflow (default: true)
    - CONTEXT_WORKFLOW_ENABLED: Enable Layer 3 context workflow (default: true)
    - MULTI_INTENT_ENABLED: Enable parallel multi-intent path in ATC (default: true)
    """

    # Master switch for tool classifier
    # When False: Uses existing RAG-only pipeline (backward compatibility)
    # When True: Routes through tool classifier
    TOOL_CLASSIFIER_ENABLED = (
        os.getenv("TOOL_CLASSIFIER_ENABLED", "false").lower() == "true"
    )

    # Individual workflow toggles
    # These only take effect when TOOL_CLASSIFIER_ENABLED=true
    SERVICE_WORKFLOW_ENABLED = (
        os.getenv("SERVICE_WORKFLOW_ENABLED", "true").lower() == "true"
    )
    API_TOOL_CALLING_WORKFLOW_ENABLED = (
        os.getenv("API_TOOL_CALLING_WORKFLOW_ENABLED", "true").lower() == "true"
    )
    CONTEXT_WORKFLOW_ENABLED = (
        os.getenv("CONTEXT_WORKFLOW_ENABLED", "true").lower() == "true"
    )

    # Multi-intent (parallel multi-API) path
    # When False: ambiguous-band ATC results go straight to the existing single-endpoint path
    # When True: IntentDecomposer runs on ambiguous-band results; parallel endpoint searches
    #            are attempted when the query is detected as multi-intent
    MULTI_INTENT_ENABLED = os.getenv("MULTI_INTENT_ENABLED", "true").lower() == "true"

    # ATC Response Cache — two-tier Redis cache for API Tool Calling responses
    # When True: successful ATC API responses are cached and follow-up queries are
    #            served from cache where possible (L1 exact hit, L2 follow-up routing)
    # When False: all cache reads and writes are skipped; normal ATC flow unchanged
    ATC_RESPONSE_CACHE_ENABLED: bool = (
        os.getenv("ATC_RESPONSE_CACHE_ENABLED", "true").lower() == "true"
    )

    # RAG and OOD workflows are always enabled (no flags)
    # RAG is the core fallback, OOD is the final safety net

    # Safety: Fallback to RAG if tool classifier encounters errors
    # This ensures service continues working even if classifier fails
    FALLBACK_TO_RAG_ON_ERROR = True

    @classmethod
    def log_configuration(cls) -> None:
        """Log current feature flag configuration (useful for debugging)."""
        logger.info("Tool Classifier Feature Flags:")
        logger.info(f"  TOOL_CLASSIFIER_ENABLED: {cls.TOOL_CLASSIFIER_ENABLED}")
        if cls.TOOL_CLASSIFIER_ENABLED:
            logger.info(f"  SERVICE_WORKFLOW_ENABLED: {cls.SERVICE_WORKFLOW_ENABLED}")
            logger.info(
                f"  API_TOOL_CALLING_WORKFLOW_ENABLED: {cls.API_TOOL_CALLING_WORKFLOW_ENABLED}"
            )
            logger.info(f"  CONTEXT_WORKFLOW_ENABLED: {cls.CONTEXT_WORKFLOW_ENABLED}")
            logger.info(f"  MULTI_INTENT_ENABLED: {cls.MULTI_INTENT_ENABLED}")
            logger.info(
                f"  ATC_RESPONSE_CACHE_ENABLED: {cls.ATC_RESPONSE_CACHE_ENABLED}"
            )
            logger.info(f"  FALLBACK_TO_RAG_ON_ERROR: {cls.FALLBACK_TO_RAG_ON_ERROR}")
        else:
            logger.info("  (Classifier disabled - using RAG-only pipeline)")

    @classmethod
    def is_workflow_enabled(cls, workflow_name: str) -> bool:
        """
        Check if a specific workflow is enabled.

        Args:
            workflow_name: Name of workflow ("service", "api_tool_calling", "context", "rag", "ood")

        Returns:
            True if workflow is enabled and classifier is enabled
        """
        if not cls.TOOL_CLASSIFIER_ENABLED:
            return False

        workflow_flags = {
            "service": cls.SERVICE_WORKFLOW_ENABLED,
            "api_tool_calling": cls.API_TOOL_CALLING_WORKFLOW_ENABLED,
            "context": cls.CONTEXT_WORKFLOW_ENABLED,
            "rag": True,  # Always enabled
            "ood": True,  # Always enabled
        }

        return workflow_flags.get(workflow_name.lower(), False)

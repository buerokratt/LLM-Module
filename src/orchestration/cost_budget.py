"""Cost logging, connection validation, budget tracking, and inference storage.

Extracted from LLMOrchestrationService (Groups I and J).
All functions take explicit arguments — no service state needed.
"""

from typing import Any, Dict, Optional

from src.loki_logger import LokiLogger
from src.llm_orchestrator_config.llm_ochestrator_constants import (
    BUDGET_EXCEEDED_MESSAGES,
    CONNECTION_INACTIVE_MESSAGES,
    get_localized_message,
)
from src.utils.cost_utils import calculate_total_costs
from src.utils.budget_tracker import get_budget_tracker

logger = LokiLogger(service_name="llm-orchestration-service")


def log_costs(costs_metric: Dict[str, Dict[str, Any]]) -> None:
    """Log cost breakdown for all components and current module versions."""
    try:
        if not costs_metric:
            return

        total_costs = calculate_total_costs(costs_metric)

        logger.info("LLM USAGE COSTS BREAKDOWN:")
        for component, costs in costs_metric.items():
            logger.info(
                f"  {component:20s}: ${costs.get('total_cost', 0):.6f} "
                f"({costs.get('num_calls', 0)} calls, "
                f"{costs.get('total_tokens', 0)} tokens)"
            )
        logger.info(
            f"  {'TOTAL':20s}: ${total_costs['total_cost']:.6f} "
            f"({total_costs['total_calls']} calls, "
            f"{total_costs['total_tokens']} tokens)"
        )

        logger.info("\nMODULE VERSIONS IN USE:")
        try:
            from src.optimization.optimized_module_loader import get_module_loader
            from src.guardrails.optimized_guardrails_loader import get_guardrails_loader

            loader = get_module_loader()
            guardrails_loader = get_guardrails_loader()

            refiner_meta = loader.get_module_metadata("refiner")
            logger.info(
                f"  Refiner: {refiner_meta.get('version', 'unknown')} "
                f"({'optimized' if refiner_meta.get('optimized') else 'base'})"
            )

            generator_meta = loader.get_module_metadata("generator")
            logger.info(
                f"  Generator: {generator_meta.get('version', 'unknown')} "
                f"({'optimized' if generator_meta.get('optimized') else 'base'})"
            )

            _, guardrails_meta = guardrails_loader.get_optimized_config_path()
            logger.info(
                f"  Guardrails: {guardrails_meta.get('version', 'unknown')} "
                f"({'optimized' if guardrails_meta.get('optimized') else 'base'})"
            )
        except Exception as version_error:
            logger.debug(f"Could not log module versions: {str(version_error)}")

    except Exception as e:
        logger.warning(f"Failed to log costs: {str(e)}")


def validate_connection_before_processing(
    vault_uuid: Optional[str],
    environment: str,
    detected_language: str = "en",
) -> tuple[bool, Optional[str]]:
    """Validate connection status and budget before processing a request.

    Checks:
    1. Connection status is 'active'
    2. Used budget has not exceeded the stop-budget threshold

    Returns:
        (True, None) if the connection is valid and within budget.
        (False, error_message) if the request should be blocked.
    """
    if not vault_uuid:
        logger.debug("No vault_uuid provided for pre-request validation, skipping")
        return (True, None)

    try:
        from src.utils.connection_id_fetcher import get_connection_id_fetcher

        fetcher = get_connection_id_fetcher()
        conn_data = fetcher.fetch_connection_budget_status_sync(vault_uuid)

        if conn_data is None:
            logger.warning(
                f"Connection not found for vault_uuid={vault_uuid} "
                f"during pre-request validation, allowing through"
            )
            return (True, None)

        status = conn_data.get("connectionStatus", "active")
        if status != "active":
            logger.warning(
                f"[Budget Gate] Connection vault_uuid={vault_uuid} is '{status}' "
                f"— blocking request (environment={environment})"
            )
            msg = get_localized_message(CONNECTION_INACTIVE_MESSAGES, detected_language)
            return (False, msg)

        used_budget = float(conn_data.get("usedBudget", 0) or 0)
        monthly_budget = float(conn_data.get("monthlyBudget", 0) or 0)
        stop_threshold = float(conn_data.get("stopBudgetThreshold", 0) or 0)

        if stop_threshold > 0 and monthly_budget > 0:
            threshold_amount = (monthly_budget / 100) * stop_threshold
            if used_budget >= threshold_amount:
                logger.warning(
                    f"[Budget Gate] Connection vault_uuid={vault_uuid} "
                    f"budget exceeded: used={used_budget:.4f}, "
                    f"threshold={threshold_amount:.4f} "
                    f"({stop_threshold}% of {monthly_budget}) "
                    f"— blocking request"
                )
                msg = get_localized_message(BUDGET_EXCEEDED_MESSAGES, detected_language)
                return (False, msg)

        logger.debug(
            f"[Budget Gate] Connection vault_uuid={vault_uuid} "
            f"validated: status={status}, "
            f"used_budget={used_budget:.4f}/{monthly_budget:.4f}"
        )
        return (True, None)

    except Exception as e:
        logger.error(
            f"Error during pre-request connection validation "
            f"for vault_uuid={vault_uuid}: {e}"
        )
        return (True, None)


def update_connection_budget(
    vault_uuid: Optional[str],
    costs_metric: Dict[str, Dict[str, Any]],
) -> None:
    """Update the budget for an LLM connection based on usage costs."""
    try:
        budget_tracker = get_budget_tracker()
        result = budget_tracker.update_budget_from_costs(vault_uuid, costs_metric)

        if result.get("success"):
            if result.get("budget_exceeded"):
                logger.warning(
                    f"Budget threshold exceeded for vault_uuid={vault_uuid}. "
                    "Connection may have been deactivated."
                )
            else:
                logger.debug(f"Budget updated successfully for vault_uuid={vault_uuid}")
        else:
            reason = result.get("reason", "unknown")
            if reason not in ["no_vault_uuid", "zero_or_negative_cost"]:
                logger.warning(
                    f"Failed to update budget for vault_uuid={vault_uuid}. "
                    f"Reason: {reason}"
                )

    except Exception as e:
        logger.error(f"Error updating budget: {str(e)}")

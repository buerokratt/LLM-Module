"""Budget tracking utility for LLM connection usage."""

from typing import Optional, Dict, Any, cast, List
from src.loki_logger import LokiLogger
import requests

from ..llm_orchestrator_config.llm_ochestrator_constants import RAG_SEARCH_RESQL

# Initialize Loki logger
logger = LokiLogger(service_name="budget-tracker")


class BudgetTracker:
    """Handles budget updates for LLM connections using vault_uuid."""

    def __init__(self) -> None:
        """Initialize the budget tracker with Resql endpoint."""
        # Use Resql directly for budget updates
        self.resql_base = RAG_SEARCH_RESQL
        self.update_endpoint = f"{self.resql_base}/update-llm-connection-used-budget"

        self.timeout = 5  # seconds

    def _make_budget_update_request(
        self, vault_uuid: str, usage_cost: float
    ) -> Dict[str, Any]:
        """
        Make the actual budget update API request.

        Args:
            vault_uuid: The vault UUID identifying the connection
            usage_cost: The cost to add

        Returns:
            Dictionary containing the response or error
        """
        payload = {"vault_uuid": vault_uuid, "usage": usage_cost}
        logger.info(f"Updating budget for vault_uuid={vault_uuid}, usage={usage_cost}")

        response = requests.post(
            self.update_endpoint, json=payload, timeout=self.timeout
        )

        if response.status_code == 200:
            response_data: Any = response.json()

            # Resql returns a list, so get the first item
            data: Any
            if isinstance(response_data, list):
                typed_list = cast(List[Any], response_data)
                if len(typed_list) > 0:
                    data = typed_list[0]
                else:
                    data = {}  # Empty dict if list is empty
            else:
                data = response_data

            # Check if budget was exceeded
            budget_exceeded: bool = False
            if isinstance(data, dict):
                budget_exceeded_value = cast(Dict[str, Any], data).get(
                    "budgetExceeded", False
                )
                budget_exceeded = bool(budget_exceeded_value)

            if budget_exceeded:
                logger.warning(
                    f"Budget threshold exceeded for vault_uuid={vault_uuid}. "
                    f"Connection may have been deactivated."
                )

            return {
                "success": True,
                "data": data,
                "budget_exceeded": budget_exceeded,
            }
        else:
            logger.error(
                f"Failed to update budget for vault_uuid={vault_uuid}. "
                f"Status: {response.status_code}, Response: {response.text}"
            )
            return {
                "success": False,
                "reason": "api_error",
                "status_code": response.status_code,
                "error_message": response.text,
            }

    def update_budget(
        self, connection_id: Optional[str], usage_cost: float
    ) -> Dict[str, Any]:
        """
        Update the used budget for an LLM connection.

        Args:
            connection_id: The vault_uuid identifying the LLM connection
            usage_cost: The cost to add to the used budget

        Returns:
            Dictionary containing the response from the update endpoint
            or an error indicator if the update failed
        """
        # Validate connection_id (vault_uuid) is provided
        if not connection_id:
            logger.debug(
                "No connection_id (vault_uuid) provided, skipping budget update"
            )
            return {
                "success": False,
                "reason": "no_connection_id",
                "connection_id": connection_id,
            }

        # Skip if usage cost is 0 or negative
        if usage_cost <= 0:
            logger.debug(f"Usage cost is {usage_cost}, skipping budget update")
            return {"success": False, "reason": "zero_or_negative_cost"}

        try:
            return self._make_budget_update_request(connection_id, usage_cost)

        except requests.exceptions.Timeout:
            logger.error(
                f"Timeout while updating budget for vault_uuid={connection_id}"
            )
            return {"success": False, "reason": "timeout"}

        except requests.exceptions.RequestException as e:
            logger.error(
                f"Request error while updating budget for vault_uuid={connection_id}: {str(e)}"
            )
            return {"success": False, "reason": "request_error", "error": str(e)}

        except Exception as e:
            logger.error(
                f"Unexpected error while updating budget for vault_uuid={connection_id}: {str(e)}"
            )
            return {"success": False, "reason": "unexpected_error", "error": str(e)}

    def update_budget_from_costs(
        self, connection_id: Optional[str], costs_metric: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Update budget from a costs dictionary containing component costs.

        Args:
            connection_id: The LLM connection ID (optional)
            costs_metric: Dictionary of component costs with total_cost values

        Returns:
            Dictionary containing the response from the update endpoint
        """
        # Calculate total cost from all components
        total_cost = 0.0
        for component_costs in costs_metric.values():
            total_cost += component_costs.get("total_cost", 0.0)

        logger.debug(
            f"Total cost calculated from components: ${total_cost:.6f} "
            f"(components: {list(costs_metric.keys())})"
        )

        return self.update_budget(connection_id, total_cost)


# Singleton instance
_budget_tracker_instance: Optional[BudgetTracker] = None


def get_budget_tracker() -> BudgetTracker:
    """Get or create the singleton budget tracker instance."""
    global _budget_tracker_instance
    if _budget_tracker_instance is None:
        _budget_tracker_instance = BudgetTracker()
    return _budget_tracker_instance

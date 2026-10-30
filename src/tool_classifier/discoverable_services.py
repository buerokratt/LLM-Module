from typing import Any, Dict, List, Set

import httpx

from src.loki_logger import LokiLogger
from llm_orchestrator_config.llm_ochestrator_constants import RAG_SEARCH_RESQL
from tool_classifier.constants import SERVICE_DISCOVERY_TIMEOUT

logger = LokiLogger(service_name="discoverable-services")


async def fetch_discoverable_service_ids() -> Set[str]:
    try:
        async with httpx.AsyncClient(timeout=SERVICE_DISCOVERY_TIMEOUT) as client:
            response = await client.post(
                f"{RAG_SEARCH_RESQL}/get-discoverable-service-ids", json={}
            )

        if response.status_code != 200:
            logger.error(
                f"Failed to fetch discoverable services: HTTP {response.status_code}"
            )
            return set()

        rows = response.json()
        if not isinstance(rows, list):
            return set()

        return {
            str(service_id)
            for row in rows
            if isinstance(row, dict)
            and (service_id := row.get("serviceId", row.get("service_id")))
        }
    except Exception as e:
        logger.error(f"Failed to fetch discoverable services: {e}", exc_info=True)
        return set()


def keep_discoverable(
    services: List[Dict[str, Any]], discoverable_ids: Set[str]
) -> List[Dict[str, Any]]:
    filtered = [
        service
        for service in services
        if str(service.get("service_id", service.get("serviceId"))) in discoverable_ids
    ]

    if len(filtered) < len(services):
        logger.info(
            f"Excluded {len(services) - len(filtered)} non-discoverable services "
            f"from {len(services)} search results"
        )

    return filtered


async def filter_discoverable_services(
    services: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not services:
        return services
    return keep_discoverable(services, await fetch_discoverable_service_ids())

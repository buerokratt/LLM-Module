"""
Production Inference Data Storage Utility

This module provides functionality to store production inference results
to the Ruuter endpoint for analytics and monitoring purposes.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from src.loki_logger import LokiLogger
import requests
import aiohttp

from src.llm_orchestrator_config.llm_ochestrator_constants import (
    RAG_SEARCH_RUUTER_PUBLIC,
)

# Initialize Loki logger
logger = LokiLogger(service_name="production-store")


class ProductionInferenceStore:
    """
    Service for storing production inference results via Ruuter endpoint.
    """

    def __init__(self) -> None:
        """Initialize the production inference store with Ruuter configuration."""
        self.store_endpoint = f"{RAG_SEARCH_RUUTER_PUBLIC}/inference/results/store"
        self.timeout = 10  # seconds

    def _create_payload(
        self,
        chat_id: str,
        user_question: str,
        refined_questions: List[str],
        conversation_history: List[Dict[str, str]],
        ranked_chunks: List[Dict[str, Any]],
        embedding_scores: List[float],
        final_answer: str,
        environment: str,
        vault_uuid: Optional[str],
    ) -> Dict[str, Any]:
        """Create the payload for storing inference results."""
        return {
            "chat_id": chat_id,
            "user_question": user_question,
            "refined_questions": json.dumps(refined_questions),
            "conversation_history": json.dumps(conversation_history),
            "ranked_chunks": json.dumps(ranked_chunks),
            "embedding_scores": json.dumps(embedding_scores),
            "final_answer": final_answer,
            "environment": environment,
            "vault_uuid": vault_uuid,
            "created_at": datetime.now().isoformat(),
        }

    def _handle_response_data(
        self, response_data: dict[str, Any] | list[Any], chat_id: str, environment: str
    ) -> Dict[str, Any]:
        """Handle and validate response data from the API."""
        # Handle nested response structure from Ruuter: {"response": {"data": {...}}}
        if isinstance(response_data, dict) and "response" in response_data:
            nested_data = response_data.get("response", {})
            if isinstance(nested_data, dict) and "data" in nested_data:
                actual_data = nested_data.get("data")
                if actual_data:
                    logger.info(
                        f"Successfully stored inference result for chat_id: {chat_id}, environment: {environment}"
                    )
                    return {
                        "success": True,
                        "data": actual_data,
                        "error": None,
                    }

        # Fallback: handle simple list format for backward compatibility
        if isinstance(response_data, list) and len(response_data) > 0:
            logger.info(
                f"Successfully stored inference result for chat_id: {chat_id}, environment: {environment}"
            )
            return {
                "success": True,
                "data": response_data[0],  # Return first item
                "error": None,
            }

        # Neither format matched - log warning
        logger.warning(
            f"Failed to store inference result for chat_id: {chat_id}, environment: {environment} - "
            f"Empty or invalid response: {response_data}"
        )
        return {
            "success": False,
            "data": None,
            "error": "Empty or invalid response from server",
        }

    def store_inference_result(
        self,
        chat_id: str,
        user_question: str,
        refined_questions: List[str],
        conversation_history: List[Dict[str, str]],
        ranked_chunks: List[Dict[str, Any]],
        embedding_scores: List[float],
        final_answer: str,
        environment: str,
        vault_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store production inference result with comprehensive data.

        Args:
            chat_id: Chat ID for this conversation
            user_question: User's raw question/input
            refined_questions: List of refined questions (LLM-generated)
            conversation_history: Prior messages array of {role, content}
            ranked_chunks: Retrieved chunks ranked with metadata
            embedding_scores: Distance scores for each chunk
            final_answer: LLM's final generated answer
            environment: Deployment environment (production/testing)
            vault_uuid: LLM connection vault UUID (used to look up connection_id in DB)

        Returns:
            Dict containing:
                - success (bool): Whether storage was successful
                - data (Optional[Dict]): Response data from server
                - error (Optional[str]): Error message if failed
        """
        try:
            # Prepare the request payload
            payload = self._create_payload(
                chat_id,
                user_question,
                refined_questions,
                conversation_history,
                ranked_chunks,
                embedding_scores,
                final_answer,
                environment,
                vault_uuid,
            )

            logger.debug(
                f"Storing inference result for chat_id: {chat_id}, environment: {environment}"
            )

            # Make the HTTP POST request to Ruuter endpoint
            response = requests.post(
                self.store_endpoint,
                json=payload,
                timeout=self.timeout,
            )

            # Check if the request was successful
            if response.status_code == 200:
                response_data = response.json()
                return self._handle_response_data(response_data, chat_id, environment)
            else:
                error_msg = (
                    f"Failed to store production inference result. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                logger.error(error_msg)
                return {
                    "success": False,
                    "data": None,
                    "error": error_msg,
                }

        except requests.exceptions.Timeout:
            error_msg = f"Timeout while storing production inference result for chat_id: {chat_id}"
            logger.error(error_msg)
            return {
                "success": False,
                "data": None,
                "error": error_msg,
            }
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"Request error while storing production inference result: {str(e)}"
            )
            logger.error(error_msg)
            return {
                "success": False,
                "data": None,
                "error": error_msg,
            }
        except Exception as e:
            error_msg = (
                f"Unexpected error while storing production inference result: {str(e)}"
            )
            logger.error(error_msg)
            return {
                "success": False,
                "data": None,
                "error": error_msg,
            }

    async def store_inference_result_async(
        self,
        chat_id: str,
        user_question: str,
        refined_questions: List[str],
        conversation_history: List[Dict[str, str]],
        ranked_chunks: List[Dict[str, Any]],
        embedding_scores: List[float],
        final_answer: str,
        environment: str = "production",
        vault_uuid: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async version of store_inference_result for streaming pipelines.

        Args:
            chat_id: Chat ID for this conversation
            user_question: User's raw question/input
            refined_questions: List of refined questions (LLM-generated)
            conversation_history: Prior messages array of {role, content}
            ranked_chunks: Retrieved chunks ranked with metadata
            embedding_scores: Distance scores for each chunk
            final_answer: LLM's final generated answer
            environment: Deployment environment (production/testing)
            vault_uuid: LLM connection vault UUID (used to look up connection_id in DB)

        Returns:
            Dict containing:
                - success (bool): Whether storage was successful
                - data (Optional[Dict]): Response data from server
                - error (Optional[str]): Error message if failed
        """
        try:
            # Prepare the request payload
            payload = self._create_payload(
                chat_id,
                user_question,
                refined_questions,
                conversation_history,
                ranked_chunks,
                embedding_scores,
                final_answer,
                environment,
                vault_uuid,
            )

            logger.debug(
                f"Storing inference result (async) for chat_id: {chat_id}, environment: {environment}"
            )

            # Make the async HTTP POST request to Ruuter endpoint
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.store_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    # Check if the request was successful
                    if response.status == 200:
                        response_data = await response.json()
                        return self._handle_response_data(
                            response_data, chat_id, environment
                        )
                    else:
                        response_text = await response.text()
                        error_msg = (
                            f"Failed to store production inference result (async). "
                            f"Status: {response.status}, Response: {response_text}"
                        )
                        logger.error(error_msg)
                        return {
                            "success": False,
                            "data": None,
                            "error": error_msg,
                        }

        except Exception as e:
            error_msg = (
                f"Error while storing production inference result (async): {str(e)}"
            )
            logger.error(error_msg)
            return {
                "success": False,
                "data": None,
                "error": error_msg,
            }


# Singleton instance for reuse across the application
_production_store_instance: Optional[ProductionInferenceStore] = None


def get_production_store() -> ProductionInferenceStore:
    """
    Get or create the singleton ProductionInferenceStore instance.

    Returns:
        ProductionInferenceStore: The singleton instance
    """
    global _production_store_instance
    if _production_store_instance is None:
        _production_store_instance = ProductionInferenceStore()
    return _production_store_instance

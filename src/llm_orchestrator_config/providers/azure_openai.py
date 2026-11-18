"""Azure OpenAI provider implementation."""

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple

import dspy

from src.loki_logger import LokiLogger

from llm_orchestrator_config.config.schema import AzureOpenAIConfig
from llm_orchestrator_config.providers.base import BaseLLMProvider
from llm_orchestrator_config.exceptions import (
    InvalidConfigurationError,
    ProviderInitializationError,
)

logger = LokiLogger(service_name="azure-openai-provider")

# API constraints, not defaults - defaults live on AzureOpenAIConfig.
# Matched on the configured model key, never the operator-chosen deployment name.
_REASONING_FAMILIES: Tuple[Tuple[Pattern[str], str], ...] = (
    (re.compile(r"^gpt-5"), "gpt5_series/"),
    (re.compile(r"^o[1345](?:[-.]|$)"), "o_series/"),
)

# DSPy anchors and LiteLLM substring-matches the deployment name, not the model key.
_DEPLOYMENT_REASONING_PREFIX: Pattern[str] = re.compile(r"^(?:o[1345]|gpt-5)")
_DEPLOYMENT_REASONING_SUBSTRINGS: Tuple[str, ...] = ("gpt-5", "o1", "o3", "o4")

# Any other temperature is a 400.
REASONING_TEMPERATURE = 1.0

# DSPy refuses to build a reasoning LM below this.
REASONING_MIN_MAX_TOKENS = 16000

VALID_REASONING_EFFORTS = frozenset({"low", "medium", "high"})

# "minimal" is gpt-5 only; the o-series rejects it.
GPT5_ONLY_REASONING_EFFORTS = frozenset({"minimal"})
GPT5_ROUTE = "gpt5_series/"

DEFAULT_REASONING_EFFORT: str = AzureOpenAIConfig.model_fields[
    "reasoning_effort"
].default


def strip_reasoning_route(model: str) -> str:
    """Drop the LiteLLM route prefix, which no cost map is keyed by."""
    for _, route in _REASONING_FAMILIES:
        model = model.replace(route, "")
    return model


def is_reasoning_lm(lm: dspy.LM) -> bool:
    """Whether a built DSPy LM targets an Azure reasoning model.

    Callers that override parameters per request need this, because the
    constraints are enforced when the LM is built, not when it is called.
    """
    model = str(getattr(lm, "model", "") or "")
    if any(route in model for _, route in _REASONING_FAMILIES):
        return True
    return "reasoning_effort" in (getattr(lm, "kwargs", None) or {})


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI provider implementation using DSPY."""

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return "Azure OpenAI"

    def get_required_config_fields(self) -> List[str]:
        """Return list of required configuration fields."""
        return [
            "enabled",
            "model",
            "endpoint",
            "api_key",
            "deployment_name",
            "api_version",
        ]

    def is_reasoning_model(self) -> bool:
        """Whether the configured model belongs to a reasoning family."""
        return self._reasoning_route() is not None

    def _reasoning_route(self) -> Optional[str]:
        """Return the LiteLLM route prefix for the configured reasoning family."""
        model = str(self.config.get("model", "")).lower()
        for pattern, route in _REASONING_FAMILIES:
            if pattern.match(model):
                return route
        return None

    def _build_model_string(self) -> str:
        """Build the DSPy model string for the configured deployment."""
        route = self._reasoning_route() or ""
        # LiteLLM strips the prefix; it only selects the parameter translation.
        return f"azure/{route}{self.config['deployment_name']}"

    def _validate_deployment_name(self) -> None:
        """Reject deployment names downstream layers would misread as reasoning models."""
        if self.is_reasoning_model():
            return
        deployment = str(self.config.get("deployment_name", "")).lower()
        prefix_match = _DEPLOYMENT_REASONING_PREFIX.match(deployment)
        marker = prefix_match.group(0) if prefix_match else None
        if marker is None:
            marker = next(
                (m for m in _DEPLOYMENT_REASONING_SUBSTRINGS if m in deployment), None
            )
        if marker is None:
            return
        raise InvalidConfigurationError(
            f"Azure deployment '{self.config['deployment_name']}' contains "
            f"'{marker}', so DSPy and LiteLLM infer a reasoning model, but "
            f"'{self.config.get('model')}' is a standard chat model. Rename the "
            f"deployment, or point this connection at the reasoning model."
        )

    def _accepted_reasoning_efforts(self) -> frozenset[str]:
        """Return the efforts the configured reasoning family accepts."""
        if self._reasoning_route() == GPT5_ROUTE:
            return VALID_REASONING_EFFORTS | GPT5_ONLY_REASONING_EFFORTS
        return VALID_REASONING_EFFORTS

    def _resolve_reasoning_effort(self, warn: bool = True) -> str:
        """Return the configured reasoning effort, or the default if unusable."""
        effort = self.config["reasoning_effort"]
        if effort not in self._accepted_reasoning_efforts():
            if warn:
                logger.warning(
                    f"Unsupported reasoning_effort '{effort}' for model "
                    f"{self.config.get('model')} - falling back to "
                    f"'{DEFAULT_REASONING_EFFORT}'"
                )
            return DEFAULT_REASONING_EFFORT
        return str(effort)

    def _build_generation_params(self, warn: bool = True) -> Dict[str, Any]:
        """Build the generation parameters the configured model family accepts.

        Read-only callers pass warn=False; this runs per request via
        get_model_info, and the overrides are a config fact, not a new event.
        """
        if not self.is_reasoning_model():
            return {
                "temperature": self.config["temperature"],
                "max_tokens": self.config["max_tokens"],
            }

        configured_temperature = self.config["temperature"]
        if warn and float(configured_temperature) != REASONING_TEMPERATURE:
            logger.warning(
                f"Model {self.config.get('model')} accepts only temperature="
                f"{REASONING_TEMPERATURE} - ignoring configured value "
                f"{configured_temperature}"
            )

        configured_max_tokens = int(self.config["max_tokens"])
        max_tokens = max(configured_max_tokens, REASONING_MIN_MAX_TOKENS)
        if warn and max_tokens != configured_max_tokens:
            logger.warning(
                f"Raised max_tokens from {configured_max_tokens} to {max_tokens} "
                f"for reasoning model {self.config.get('model')}"
            )

        # max_tokens is renamed to max_completion_tokens downstream.
        return {
            "temperature": REASONING_TEMPERATURE,
            "max_tokens": max_tokens,
            "reasoning_effort": self._resolve_reasoning_effort(warn),
        }

    def initialize(self) -> None:
        """Initialize the Azure OpenAI provider.

        Raises:
            ProviderInitializationError: If initialization fails.
        """
        try:
            self.validate_config()
            self._validate_deployment_name()

            # Initialize DSPY LM client with proper Azure OpenAI configuration
            self._client = dspy.LM(
                model=self._build_model_string(),
                model_type=self.config["model_type"],
                cache=self.config["cache"],
                callbacks=None,
                num_retries=self.config["num_retries"],
                api_key=self.config["api_key"],
                api_base=self.config["endpoint"],
                api_version=self.config["api_version"],
                **self._build_generation_params(),
            )

            self._initialized = True

        except Exception as e:
            raise ProviderInitializationError(
                f"Failed to initialize {self.provider_name} provider: {e}"
            ) from e

    def get_dspy_client(self) -> dspy.LM:
        """Return DSPY-compatible client.

        Returns:
            DSPY LM client instance.

        Raises:
            RuntimeError: If the provider is not initialized.
        """
        self._ensure_initialized()

        if self._client is None:
            raise RuntimeError("Client is not initialized")

        return self._client

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model.

        Returns:
            Dictionary containing model information.
        """
        base_info = super().get_model_info()
        base_info.update(
            {
                "endpoint": self.config.get("endpoint", ""),
                "deployment_name": self.config.get("deployment_name", ""),
                "api_version": self.config.get("api_version", ""),
                "is_reasoning_model": self.is_reasoning_model(),
            }
        )
        if self.is_reasoning_model():
            # Report what the client was built with, not what was configured.
            params = self._build_generation_params(warn=False)
            base_info["reasoning_effort"] = params["reasoning_effort"]
            base_info["temperature"] = params["temperature"]
            base_info["max_tokens"] = params["max_tokens"]
        return base_info

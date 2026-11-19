"""Cost must be looked up by the configured model key, not the Azure deployment name."""

from typing import Any, Dict

import litellm
import pytest

from llm_orchestrator_config.config.schema import AzureOpenAIConfig
from llm_orchestrator_config.providers.azure_openai import (
    AzureOpenAIProvider,
    pricing_model_for,
)
from src.utils.cost_utils import _extract_cost_with_fallback

# Test prices, registered so the tests don't depend on LiteLLM's downloaded price list.
INPUT_COST = 0.000001
OUTPUT_COST = 0.000002


@pytest.fixture(autouse=True)
def _register_test_prices() -> None:
    litellm.register_model(
        {
            model: {
                "input_cost_per_token": INPUT_COST,
                "output_cost_per_token": OUTPUT_COST,
                "litellm_provider": "azure",
                "mode": "chat",
            }
            for model in ("azure/gpt-5.4", "azure/gpt-4.1")
        }
    )


def _initialized_provider(model: str, deployment_name: str) -> AzureOpenAIProvider:
    config = AzureOpenAIConfig(
        enabled=True,
        model=model,
        endpoint="https://example.openai.azure.com",
        api_key="test-key",
        deployment_name=deployment_name,
    )
    provider = AzureOpenAIProvider(config.to_dict())
    provider.initialize()
    return provider


def _history_item(lm_model: str) -> Dict[str, Any]:
    # A DSPy history entry without cost, as for streaming calls.
    return {
        "model": lm_model,
        "cost": None,
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }


@pytest.mark.parametrize(
    ("model", "deployment_name"),
    [("gpt-5.4", "bk-chat-prod"), ("gpt-4.1", "bk-gpt41-prod")],
)
def test_pricing_model_uses_model_key(model: str, deployment_name: str) -> None:
    provider = _initialized_provider(model, deployment_name)
    lm = provider.get_dspy_client()

    assert lm.kwargs["base_model"] == f"azure/{model}"
    assert pricing_model_for(lm.model) == f"azure/{model}"


@pytest.mark.parametrize(
    ("model", "deployment_name"),
    [("gpt-5.4", "bk-chat-prod"), ("gpt-4.1", "bk-gpt41-prod")],
)
def test_fallback_cost_ignores_deployment_name(
    model: str, deployment_name: str
) -> None:
    lm = _initialized_provider(model, deployment_name).get_dspy_client()
    item = _history_item(lm.model)

    cost, source = _extract_cost_with_fallback(item, item["usage"])

    assert source == "litellm.cost_per_token"
    assert cost == pytest.approx(100 * INPUT_COST + 50 * OUTPUT_COST)


def test_response_cost_uses_model_key_for_custom_deployment() -> None:
    lm = _initialized_provider("gpt-5.4", "bk-chat-prod").get_dspy_client()

    response = litellm.completion(
        model=lm.model,
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
        **{k: v for k, v in lm.kwargs.items() if k != "max_tokens"},
    )

    assert response._hidden_params.get("response_cost") == pytest.approx(
        response.usage.prompt_tokens * INPUT_COST
        + response.usage.completion_tokens * OUTPUT_COST
    )


def test_unknown_lm_keeps_previous_behaviour() -> None:
    assert pricing_model_for("azure/gpt5_series/never-built") == "azure/never-built"

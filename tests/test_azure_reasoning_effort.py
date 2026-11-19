"""Tests for reasoning_effort resolution in the Azure OpenAI provider."""

from typing import Any, Dict, Optional

import pytest

from llm_orchestrator_config.config.schema import AzureOpenAIConfig
from llm_orchestrator_config.providers.azure_openai import AzureOpenAIProvider


def _provider(
    model: str, reasoning_effort: Optional[str] = None
) -> AzureOpenAIProvider:
    config: Dict[str, Any] = {
        "enabled": True,
        "model": model,
        "endpoint": "https://example.openai.azure.com",
        "api_key": "test-key",
        "deployment_name": "my-deployment",
    }
    if reasoning_effort is not None:
        config["reasoning_effort"] = reasoning_effort
    return AzureOpenAIProvider(AzureOpenAIConfig(**config).to_dict())


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-5.4", "none"),
        ("gpt-5.2", "none"),
        ("gpt-5.1", "none"),
        ("gpt-5", "medium"),
        ("o3", "medium"),
    ],
)
def test_unset_effort_uses_model_default(model: str, expected: str) -> None:
    assert _provider(model)._resolve_reasoning_effort(warn=False) == expected


@pytest.mark.parametrize(
    ("model", "effort"),
    [
        ("gpt-5.4", "none"),
        ("gpt-5.4", "low"),
        ("gpt-5.4", "high"),
        ("gpt-5.4", "xhigh"),
        ("gpt-5.2", "none"),
        ("gpt-5.2", "xhigh"),
        ("gpt-5", "minimal"),
        ("o3", "high"),
    ],
)
def test_supported_effort_is_kept(model: str, effort: str) -> None:
    assert _provider(model, effort)._resolve_reasoning_effort(warn=False) == effort


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gpt-5.4", "minimal", "none"),  # minimal is original gpt-5 only
        ("gpt-5.1", "xhigh", "none"),  # xhigh is gpt-5.2+ only
        ("gpt-5", "none", "medium"),  # none is gpt-5.1+ only
        ("o3", "none", "medium"),
        ("gpt-5.4", "extreme", "none"),
    ],
)
def test_unsupported_effort_falls_back_to_model_default(
    model: str, effort: str, expected: str
) -> None:
    assert _provider(model, effort)._resolve_reasoning_effort(warn=False) == expected


def test_reasoning_params_include_resolved_effort() -> None:
    params = _provider("gpt-5.4")._build_generation_params(warn=False)

    assert params["reasoning_effort"] == "none"
    assert params["temperature"] == 1.0
    assert params["max_tokens"] >= 16000


def test_standard_chat_model_gets_no_reasoning_effort() -> None:
    params = _provider("gpt-4.1")._build_generation_params(warn=False)

    assert "reasoning_effort" not in params

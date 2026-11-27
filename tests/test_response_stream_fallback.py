"""Streaming generation sends the final answer when no answer tokens streamed.

Under concurrent load Azure answered 429; DSPy retried the call with its JSON
adapter, whose output the answer StreamListener cannot parse. Nothing streamed
although the final Prediction had the answer, and the user got an empty reply
(only the references block).
"""

from typing import Any, AsyncIterator, List

import dspy
import pytest
from dspy.streaming import StreamResponse

from src.response_generator.response_generate import ResponseGeneratorAgent

CHUNKS = [{"text": "DigiDoc is the Estonian digital signature format.", "meta": {}}]
ANSWER = "DigiDoc is the Estonian format for digitally signed documents."


def _agent(stream_items: List[Any]) -> ResponseGeneratorAgent:
    """An agent whose streamified predictor yields ``stream_items``."""

    async def stream(**_: Any) -> AsyncIterator[Any]:
        for item in stream_items:
            yield item

    agent = object.__new__(ResponseGeneratorAgent)
    agent._custom_instructions_prefix = ""
    agent._stream_predictor = lambda **kwargs: stream(**kwargs)
    return agent


async def _collect(agent: ResponseGeneratorAgent) -> List[str]:
    return [token async for token in agent.stream_response("what is digidoc?", CHUNKS)]


def _token(text: str) -> StreamResponse:
    return StreamResponse("predict", "answer", text, False)


@pytest.mark.asyncio
async def test_final_answer_is_sent_when_no_tokens_streamed() -> None:
    agent = _agent([dspy.Prediction(answer=ANSWER, questionOutOfLLMScope=False)])

    assert await _collect(agent) == [ANSWER]


@pytest.mark.asyncio
async def test_streamed_tokens_are_not_repeated_by_the_final_answer() -> None:
    agent = _agent(
        [
            _token("DigiDoc is the Estonian format "),
            _token("for digitally signed documents."),
            dspy.Prediction(answer=ANSWER, questionOutOfLLMScope=False),
        ]
    )

    assert await _collect(agent) == [
        "DigiDoc is the Estonian format ",
        "for digitally signed documents.",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prediction",
    [dspy.Prediction(questionOutOfLLMScope=True), dspy.Prediction(answer="  ")],
)
async def test_nothing_is_sent_without_an_answer(prediction: dspy.Prediction) -> None:
    assert await _collect(_agent([prediction])) == []

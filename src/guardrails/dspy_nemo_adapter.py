"""
Native DSPy + NeMo Guardrails LLM adapter with proper streaming support.
Follows both NeMo's official custom LLM provider pattern and DSPy's native architecture.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, cast, Iterator, AsyncIterator
import asyncio
import dspy
from src.loki_logger import LokiLogger
from langfuse import observe
from src.utils.observation_utils import update_observation_safe

from langchain_core.callbacks.manager import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from src.guardrails.guardrails_llm_configs import TEMPERATURE, MAX_TOKENS, MODEL_NAME
from src.utils.cost_utils import get_lm_usage_since

# Initialize Loki logger
logger = LokiLogger(service_name="dspy-nemo-adapter")


class DSPyNeMoLLM(LLM):
    """
    Production-ready custom LLM provider for NeMo Guardrails using DSPy.

    This implementation properly integrates:
    - Native DSPy LM calls (via dspy.settings.lm)
    - NeMo Guardrails LangChain BaseLanguageModel interface
    - Token-level streaming via LiteLLM (DSPy's underlying engine)

    Architecture:
    - DSPy uses LiteLLM internally for all LM operations
    - When stream=True is passed to DSPy LM, it delegates to LiteLLM's streaming
    - This is the proper way to stream with DSPy until dspy.streamify is fully integrated

    Note: dspy.streamify() is designed for DSPy *modules* (Predict, ChainOfThought, etc.)
    not for raw LM calls. Since NeMo calls the LLM directly via LangChain interface,
    this use the lower-level streaming that DSPy's LM provides through LiteLLM.
    """

    model_name: str = MODEL_NAME
    temperature: float = TEMPERATURE
    max_tokens: int = MAX_TOKENS
    streaming: bool = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        logger.info(
            f"Initialized DSPyNeMoLLM adapter "
            f"(model={self.model_name}, temp={self.temperature})"
        )

    @property
    def _llm_type(self) -> str:
        return "dspy-custom"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
        }

    def _get_dspy_lm(self) -> dspy.LM:
        """
        Get the active DSPy LM from settings.

        This is the proper way to access DSPy's LM according to official docs.
        The LM is configured via dspy.configure(lm=...) or dspy.settings.lm
        """
        lm = dspy.settings.lm
        if lm is None:
            raise RuntimeError(
                "No DSPy LM configured. Please configure dspy.settings.lm first."
            )
        return lm

    def _extract_text_from_response(self, response: Union[str, List[Any], Any]) -> str:
        """
        Extract text from non-streaming DSPy response.

        DSPy LM returns various response formats depending on the provider.
        This handles the common cases.
        """
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, list) and len(cast(List[Any], response)) > 0:
            return str(cast(List[Any], response)[0]).strip()
        if not isinstance(response, list):
            return str(response).strip()
        return ""

    def _extract_chunk_text(self, chunk: Any) -> str:
        """
        Extract text from a streaming chunk.

        When DSPy's LM streams (via LiteLLM), it returns chunks in various formats
        depending on the provider. This handles OpenAI-style objects and dicts.

        Reference: DSPy delegates to LiteLLM for streaming, which uses provider-specific
        streaming formats (OpenAI, Anthropic, etc.)
        """
        # Case 1: Raw string
        if isinstance(chunk, str):
            return chunk

        # Case 2: Object with choices (OpenAI style)
        if hasattr(chunk, "choices") and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                return delta.content

        # Case 3: Dict style
        if isinstance(chunk, dict) and "choices" in chunk:
            choices = chunk["choices"]
            if choices and len(choices) > 0:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    return content

        return ""

    @observe(name="nemo_guardrail_llm_call", as_type="generation")
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous non-streaming call.

        This is the standard path for NeMo Guardrails when streaming is disabled.
        Call DSPy's LM directly with the prompt.
        """
        history_length_before = 0
        lm_for_usage = dspy.settings.lm
        if lm_for_usage and hasattr(lm_for_usage, "history"):
            history_length_before = len(lm_for_usage.history)

        try:
            lm = self._get_dspy_lm()

            # Prepare kwargs
            call_kwargs = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if stop:
                call_kwargs["stop"] = stop

            # DSPy LM call - returns text directly
            response = lm(prompt, **call_kwargs)
            response_text = self._extract_text_from_response(response)

            usage_info = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "prompt_preview": prompt[:500],
                    "max_tokens": call_kwargs["max_tokens"],
                    "temperature": call_kwargs["temperature"],
                },
                output_data={"response_preview": response_text[:500]},
                metadata={
                    "model": self.model_name,
                    "usage": usage_info,
                    "num_calls": usage_info.get("num_calls", 0),
                    "streaming": False,
                    "guardrail_provider": "nemo",
                },
            )

            return response_text

        except Exception as e:
            logger.error(f"Error in DSPyNeMoLLM._call: {str(e)}")
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    @observe(name="nemo_guardrail_llm_call", as_type="generation")
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async non-streaming call (Required by NeMo).

        Uses asyncio.to_thread to prevent blocking the event loop.
        This is critical because DSPy's LM is synchronous and makes network calls.
        """
        history_length_before = 0
        lm_for_usage = dspy.settings.lm
        if lm_for_usage and hasattr(lm_for_usage, "history"):
            history_length_before = len(lm_for_usage.history)

        try:
            lm = self._get_dspy_lm()

            # Prepare kwargs
            call_kwargs = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if stop:
                call_kwargs["stop"] = stop

            # Run in thread to avoid blocking
            response = await asyncio.to_thread(lm, prompt, **call_kwargs)
            response_text = self._extract_text_from_response(response)

            usage_info = get_lm_usage_since(history_length_before)
            update_observation_safe(
                input_data={
                    "prompt_preview": prompt[:500],
                    "max_tokens": call_kwargs["max_tokens"],
                    "temperature": call_kwargs["temperature"],
                },
                output_data={"response_preview": response_text[:500]},
                metadata={
                    "model": self.model_name,
                    "usage": usage_info,
                    "num_calls": usage_info.get("num_calls", 0),
                    "streaming": False,
                    "guardrail_provider": "nemo",
                },
            )

            return response_text

        except Exception as e:
            logger.error(f"Error in DSPyNeMoLLM._acall: {str(e)}")
            raise RuntimeError(f"Async LLM generation failed: {str(e)}") from e

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Synchronous streaming via DSPy's native streaming support.

        How this works:
        1. DSPy's LM accepts stream=True parameter
        2. DSPy delegates to LiteLLM which handles provider-specific streaming
        3. LiteLLM returns an iterator of chunks
        4. extract text from each chunk and yield it

        This is the proper low-level streaming approach when not using dspy.streamify(),
        which is designed for higher-level DSPy modules.

        """
        try:
            lm = self._get_dspy_lm()

            # Prepare kwargs with streaming enabled
            call_kwargs = {
                "stream": True,  # This triggers LiteLLM streaming
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            if stop:
                call_kwargs["stop"] = stop

            # Get streaming generator from DSPy LM
            # DSPy's LM will call LiteLLM with stream=True
            stream_generator = lm(prompt, **call_kwargs)

            # Yield tokens as they arrive
            for chunk in stream_generator:
                token = self._extract_chunk_text(chunk)
                if token:
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                    yield GenerationChunk(text=token)

        except Exception as e:
            logger.error(f"Error in DSPyNeMoLLM._stream: {str(e)}")
            raise RuntimeError(f"Streaming failed: {str(e)}") from e

    async def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """
        Async streaming using Threaded Producer / Async Consumer pattern.

        Why this pattern:
        - DSPy's LM is synchronous (calls LiteLLM synchronously)
        - Streaming involves blocking network I/O in the iterator
        - MUST run the synchronous generator in a thread
        - Use a queue to safely pass chunks to the async consumer

        This pattern prevents blocking the event loop while maintaining
        proper async semantics for NeMo Guardrails.
        """
        try:
            lm = self._get_dspy_lm()
        except Exception as e:
            logger.error(f"Error getting DSPy LM: {str(e)}")
            raise RuntimeError(f"Failed to get DSPy LM: {str(e)}") from e

        # Setup queue and event loop
        queue: asyncio.Queue[Union[Any, Exception, None]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Sentinel to mark end of stream
        sentinel = object()

        def producer() -> None:
            """
            Synchronous producer running in a thread.
            Calls DSPy's LM with stream=True and pushes chunks to queue.
            """
            try:
                # Prepare kwargs with streaming
                call_kwargs = {
                    "stream": True,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                if stop:
                    call_kwargs["stop"] = stop

                # Get streaming generator
                stream_generator = lm(prompt, **call_kwargs)

                # Push chunks to queue
                for chunk in stream_generator:
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)

                # Signal completion
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

            except Exception as e:
                # Pass exception to async consumer
                loop.call_soon_threadsafe(queue.put_nowait, e)

        # Start producer in thread pool
        loop.run_in_executor(None, producer)

        # Async consumer - yield tokens as they arrive
        try:
            while True:
                # Wait for next chunk (non-blocking)
                chunk = await queue.get()

                # Check for completion
                if chunk is sentinel:
                    break

                # Check for errors from producer
                if isinstance(chunk, Exception):
                    raise chunk

                # Extract and yield token
                token = self._extract_chunk_text(chunk)
                if token:
                    if run_manager:
                        await run_manager.on_llm_new_token(token)
                    yield GenerationChunk(text=token)

        except Exception as e:
            logger.error(f"Error in DSPyNeMoLLM._astream: {str(e)}")
            raise RuntimeError(f"Async streaming failed: {str(e)}") from e


class DSPyLLMProviderFactory:
    """
    Factory for NeMo Guardrails registration.

    NeMo requires a callable factory that returns an LLM instance.
    """

    def __call__(self, config: Optional[Dict[str, Any]] = None) -> DSPyNeMoLLM:
        """Create and return a DSPyNeMoLLM instance."""
        if config is None:
            config = {}
        return DSPyNeMoLLM(**config)

    # Placeholder methods required by some versions of NeMo validation
    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Factory class - use DSPyNeMoLLM instance")

    async def _acall(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Factory class - use DSPyNeMoLLM instance")

    @property
    def _llm_type(self) -> str:
        return "dspy-custom"

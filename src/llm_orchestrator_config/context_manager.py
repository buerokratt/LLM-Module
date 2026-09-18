"""Context Generation Manager using Anthropic methodology."""

from typing import Any, Dict, Optional

from src.loki_logger import LokiLogger
from src.llm_orchestrator_config.llm_manager import LLMManager
from src.models.request_models import ContextGenerationRequest
from langfuse import observe

# Initialize Loki logger
logger = LokiLogger(service_name="context-manager")


class ContextGenerationManager:
    """Manager for context generation with Anthropic methodology."""

    # Anthropic's exact prompt templates from their research
    DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>"""

    CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

    # Used by the API Tool Calling indexer — passes the caller's prompt through
    # unmodified so CHUNK_CONTEXT_PROMPT cannot override the caller's own instructions.
    API_TOOL_CONTEXT_PROMPT = """{chunk_content}"""

    def __init__(self, llm_manager: LLMManager) -> None:
        """Initialize context generation manager."""
        self.llm_manager = llm_manager
        # Cache structure prepared for future prompt caching implementation
        self._cache: Dict[str, Any] = {}

    @observe(name="generate_context_with_caching", as_type="generation")
    def generate_context_with_caching(
        self, request: ContextGenerationRequest
    ) -> Dict[str, Any]:
        """Generate context using Anthropic methodology with caching structure."""
        try:
            # Resolve model from LLM manager configuration
            model_info = self._resolve_model_for_request(request)
            logger.info(f"Generating context using model: {model_info['model']}")

            # Prepare the full prompt using Anthropic's format
            full_prompt = self._prepare_anthropic_prompt(
                request.document_prompt, request.chunk_prompt, request.context_type
            )

            # For now, call LLM directly (caching structure ready for future)
            # Implement actual prompt caching when ready
            response = self._call_llm_for_context(
                prompt=full_prompt,
                model=model_info["model"],
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                connection_id=request.connection_id,
            )

            # Extract and format response
            usage_metrics = self._extract_usage_metrics(response)

            return {
                "context": response.content.strip(),
                "usage": usage_metrics["usage"],
                "cache_performance": usage_metrics["cache_performance"],
                "model_used": model_info["model"],
            }

        except Exception as e:
            logger.error(f"Context generation failed: {e}")
            raise

    def _resolve_model_for_request(
        self, request: ContextGenerationRequest
    ) -> Dict[str, str]:
        """Resolve model information from LLM configuration based on request.

        Args:
            request: Context generation request with environment and connection_id

        Returns:
            Dictionary with model and provider information
        """
        try:
            # Get the current LLM configuration
            config = self.llm_manager.get_configuration()

            if not config:
                raise RuntimeError("LLM configuration not loaded")

            # Use the default provider from configuration
            default_provider = config.default_provider.value
            provider_config = config.providers.get(default_provider)

            if not provider_config or not provider_config.enabled:
                raise RuntimeError(
                    f"Default provider {default_provider} is not available or enabled"
                )

            return {"provider": default_provider, "model": provider_config.model}

        except Exception as e:
            logger.error(f"Failed to resolve model for context generation: {e}")
            raise RuntimeError(f"Model resolution failed: {e}") from e

    def _prepare_anthropic_prompt(
        self,
        document_prompt: str,
        chunk_prompt: str,
        context_type: str = "chunk",
    ) -> str:
        """Prepare the LLM prompt based on context_type.

        - 'api_tool': returns chunk_prompt as-is using API_TOOL_CONTEXT_PROMPT so the
          caller's own instructions (including example-query generation) are not
          overridden by CHUNK_CONTEXT_PROMPT's closing directive.
        - 'chunk' (default): uses the Anthropic RAG format (document + chunk sections).
        """
        if context_type == "api_tool":
            return self.API_TOOL_CONTEXT_PROMPT.format(chunk_content=chunk_prompt)

        # Format document section
        document_section = self.DOCUMENT_CONTEXT_PROMPT.format(
            doc_content=document_prompt
        )

        # Format chunk section
        chunk_section = self.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_prompt)

        # Combine using Anthropic's methodology
        return f"{document_section}\n\n{chunk_section}"

    def _call_llm_for_context(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        connection_id: Optional[str] = None,
    ) -> Any:
        """Call LLM for context generation."""
        # Acknowledge unused parameters for future implementation
        _ = max_tokens, temperature, connection_id

        # Configure DSPy for this call
        self.llm_manager.ensure_global_config()

        # Use DSPy to make the LLM call
        import dspy

        # Create a simple DSPy signature for context generation
        class ContextGeneration(dspy.Signature):
            """Generate succinct context for a chunk within a document."""

            prompt = dspy.InputField()
            context = dspy.OutputField()

        # Use DSPy Predict to generate context
        context_generator = dspy.Predict(ContextGeneration)
        result = context_generator(prompt=prompt)

        # Return a response object with the expected structure
        class MockResponse:
            def __init__(self, content: str, model: str) -> None:
                self.content = content
                self.model = model
                self.usage = MockUsage(content, prompt)

        class MockUsage:
            def __init__(self, content: str, prompt: str) -> None:
                self.input_tokens = int(len(prompt.split()) * 1.3)  # Rough estimate
                self.output_tokens = int(len(content.split()) * 1.3)

        return MockResponse(str(result.context), model)

    def _extract_usage_metrics(self, response: Any) -> Dict[str, Any]:
        """Extract token usage and caching metrics."""
        # Extract basic usage info
        usage = getattr(response, "usage", {})

        # Prepare cache performance metrics (ready for future implementation)
        cache_performance = {
            "cache_hit": False,
            "cache_tokens_read": 0,
            "cache_tokens_written": 0,
            "cache_savings_percentage": 0.0,
        }

        # Format usage metrics
        formatted_usage = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "total_tokens": getattr(usage, "input_tokens", 0)
            + getattr(usage, "output_tokens", 0),
        }

        return {"usage": formatted_usage, "cache_performance": cache_performance}

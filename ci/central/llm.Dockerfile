FROM python:3.12.10-slim AS build

RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=ghcr.io/astral-sh/uv@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 /uv /uvx /bin/

COPY . /app

WORKDIR /app

# Set Python path to include src directory
ENV PYTHONPATH=/app/src

RUN uv sync --locked --python /usr/local/bin/python3.12

ENV PYTHONPATH=/app/src:/app HOME=/tmp XDG_CACHE_HOME=/tmp/.cache LLM_GENERATION_ENABLED=false
FROM build AS test
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN --network=none /app/.venv/bin/python -m pytest -q tests/test_agentic_loop.py tests/test_context_workflow.py tests/test_history_integration.py tests/test_message_constants.py tests/test_param_extractor.py tests/test_retrieval_relevance_gate.py
FROM build AS runtime
EXPOSE 8100
CMD ["/app/.venv/bin/python3", "-m", "uvicorn", "src.llm_orchestration_service_api:app", "--host", "0.0.0.0", "--port", "8100"]

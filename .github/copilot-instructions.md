# BYK-RAG Module - Copilot Instructions

## Project Overview

BYK-RAG is a Retrieval-Augmented Generation module for Estonian government digital services (Bürokratt ecosystem). It provides secure, multilingual AI-powered responses by integrating multiple LLM providers, contextual retrieval, and guardrails.

## Build, Test, and Lint Commands

### Environment Setup
```bash
# Install Python 3.12.10 and create virtual environment
uv python install 3.12.10
uv sync --frozen

# Install pre-commit hooks
uv run pre-commit install
```

### Running Services
```bash
# Always use uv run for Python scripts (whether venv is activated or not)
uv run python <script.py>

# Start all services with Docker Compose
docker compose up

# Run FastAPI orchestration service locally
uv run uvicorn src.llm_orchestration_service_api:app --reload
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_query_validator.py -v

# Run integration tests (requires Docker and secrets)
uv run pytest tests/integration_tests/ -v --tb=short --log-cli-level=INFO

# Run deepeval tests
uv run pytest tests/deepeval_tests/standard_tests.py -v --tb=short
```

### Linting and Formatting
```bash
# Check code formatting (does NOT modify files)
uv run ruff format --check

# Apply code formatting (SAFE - layout only, no logic changes)
uv run ruff format

# Check linting issues (manual fixes required)
uv run ruff check .

# Get explanation for specific lint rule
uv run ruff rule <rule-code>  # e.g., ANN204

# NEVER use ruff check --fix (can alter logic/control flow)
```

### Type Checking
```bash
# Run Pyright type checker (runs on src/ only, not tests/)
uv run pyright
```

### Pre-commit Hooks
```bash
# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

## Architecture

### Core Components

1. **LLM Orchestration Service** (`src/llm_orchestration_service.py`)
   - Facade and entry point for RAG orchestration (`process_orchestration_request`, `stream_orchestration_response`)
   - Coordinates prompt refinement, retrieval, generation, and guardrails
   - Integrates with Langfuse for observability
   - Pipeline logic lives in `src/orchestration/`; the facade's private methods delegate there:
     - `rag_pipeline.py`: RAG streaming and non-streaming pipelines
     - `request_preamble.py`: query validation, connection and budget checks before routing
     - `guardrails_coordinator.py`: input/output guardrail checks
     - `component_factory.py`: LLM manager, guardrails, retriever, and response generator setup
     - `cost_budget.py`, `prompt_refinement.py`, `inference_storage.py`, `response_builders.py`
     - `constants.py`: shared constants, including `_HISTORY_EXCLUDED_MESSAGES` (localized error messages that must never be saved to conversation history)

2. **FastAPI Application** (`src/llm_orchestration_service_api.py`)
   - HTTP API layer exposing `/orchestrate` endpoint
   - Handles streaming responses and rate limiting
   - Request/response validation via Pydantic models

3. **Contextual Retrieval** (`src/contextual_retrieval/`)
   - Implements Anthropic's Contextual Retrieval methodology
   - Hybrid search: Vector (semantic) + BM25 (lexical) with RRF fusion
   - Multi-query expansion (6 refined queries per user query)
   - Qdrant vector database integration

4. **Prompt Refinement** (`src/prompt_refine_manager/`)
   - DSPy-based query expansion
   - Generates 5 refined variations + original query

5. **Response Generation** (`src/response_generator/`)
   - DSPy-based response synthesis
   - Supports streaming via SSE (Server-Sent Events)
   - Uses top-K retrieved chunks (default: 10)

6. **Guardrails** (`src/guardrails/`)
   - NeMo Guardrails integration with DSPy
   - Input guardrails (pre-refinement) and output guardrails (post-generation)
   - Blocks out-of-scope queries and harmful content

7. **LLM Manager** (`src/llm_orchestrator_config/llm_manager.py`)
   - Multi-provider support: AWS Bedrock, Azure OpenAI, Google Cloud, OpenAI, Anthropic
   - HashiCorp Vault integration for secret management
   - RSA-2048 encrypted credentials storage

8. **Vector Indexer** (`src/vector_indexer/`)
   - Qdrant collection management
   - Embedding generation and indexing
   - BM25 index creation

### Supporting Services (Docker Compose)

- **Ruuter** (Public/Private): API gateway and routing
- **DataMapper**: Data transformation layer
- **Resql**: PostgreSQL query builder
- **CronManager**: Scheduled jobs (knowledge base sync)
- **Qdrant**: Vector database
- **MinIO**: S3-compatible object storage
- **HashiCorp Vault**: Secret management
- **Grafana Loki**: Log aggregation
- **Langfuse**: LLM observability dashboard

### Key Data Flow

```
User Query
  ↓
Input Guardrails (NeMo Rails)
  ↓
Prompt Refinement (DSPy) → 6 queries
  ↓
Parallel Hybrid Search (each query)
  ├─→ Semantic Search (Qdrant, top-40 per query, threshold ≥0.4)
  └─→ BM25 Search (top-40 per query)
  ↓
RRF Fusion → Top-K chunks (10 default)
  ↓
Response Generation (DSPy)
  ↓
Output Guardrails (NeMo Rails)
  ↓
Response to User (JSON or SSE stream)
```

## Key Conventions

### Dependency Management

- **ALWAYS use `uv add <package>`** to add dependencies (never `pip install`)
- **ALWAYS commit both `pyproject.toml` AND `uv.lock`** together
- Use bounded version ranges: `uv add "package>=x.y,<x.(y+1)"`
- After adding/removing deps: `uv sync --reinstall`
- **NEVER edit `uv.lock` manually** or use `requirements.txt`

### Python Execution

```bash
# Correct
uv run python app.py
uv run pytest
uv run pyright

# Wrong (bypasses uv's environment management)
python3 app.py
pytest
```

### Type Safety

- **Pyright in `standard` mode** (configured in `pyproject.toml`)
- Type checks enforced by CI, but **NOT on test files** (src/ only)
- **Runtime validation at system boundaries**: FastAPI endpoints use Pydantic models
- Prefer type inference over explicit annotations where clear
- Third-party libraries without stubs treated as `Any`

### Linting Rules (Ruff)

Selected categories (see `pyproject.toml` for full config):
- **E4, E7, E9**: Pycodestyle errors (imports, indentation, syntax)
- **F**: Pyflakes (undefined names, unused imports)
- **B**: Flake8-bugbear (mutable defaults, exception handling)
- **T20**: Flake8-print (flags `print()` statements)
- **N**: PEP8-naming conventions
- **ANN**: Flake8-annotations (type annotation discipline)
- **ERA**: Eradicate (no commented-out code)
- **PERF**: Perflint (performance anti-patterns)

**Fixing linting issues:**
- ALWAYS fix manually (never use `ruff check --fix`)
- Use `uv run ruff rule <rule-code>` for explanations
- Autofixes can alter control flow/logic unintentionally

### Formatting (Ruff Formatter)

- Double quotes for strings
- Spaces for indentation (4 spaces)
- Respects magic trailing commas
- Auto-detects line endings (LF/CRLF)
- Does NOT reformat docstring code blocks
- `uv run ruff format` is SAFE (layout only, no logic changes)

### DSPy Usage

- Used for prompt refinement (multi-query expansion) and response generation
- Custom LLM adapters integrate DSPy with NeMo Guardrails
- Optimization modules under `src/optimization/` for tuning prompts/metrics
- Models loaded via `optimized_module_loader.py` for compiled DSPy modules

### HashiCorp Vault Integration

- Secrets stored at `secret/users/<user>/<connection_id>/`
- Each connection has `provider`, `environment`, and provider-specific keys
- RSA-2048 encryption layer BEFORE Vault storage
- GUI encrypts with public key; CronManager decrypts with private key
- Vault unavailable = graceful degradation (fail securely)

### Logging

- **loguru** for application logging
- Grafana Loki integration for centralized logs
- Use `logger.info()`, `logger.warning()`, `logger.error()` (NOT `print()`)
- Loki logger available at `grafana-configs/loki_logger.py`

### Streaming Responses

- Implemented via Server-Sent Events (SSE) in FastAPI
- `StreamConfig` and `stream_manager` coordinate streaming state
- `stream_response_native()` in response_generator yields tokens
- Timeout handling via `stream_timeout` utility
- Environment-gated: check `STREAMING_ALLOWED_ENVS`

### Configuration Loading

- `PromptConfigurationLoader` fetches prompt configs from Ruuter endpoint
- Cache TTL: `PROMPT_CONFIG_CACHE_TTL`
- Custom prompts per user/organization (stored in Vault/database)
- Fallback to defaults if Ruuter unavailable

### Error Handling

- `generate_error_id()` creates unique error IDs for tracking
- `log_error_with_context()` for structured error logging
- Localized error messages via `get_localized_message()` (multilingual support)
- Predefined message constants in `llm_orchestrator_constants.py`

### Testing Conventions

- Test files under `tests/` (unit, integration, deepeval)
- Integration tests use `testcontainers` for Docker orchestration
- Secrets required for integration tests (Azure OpenAI keys, etc.)
- Mock data in `tests/mocks/` and `tests/data/`

### CI/CD Checks

1. **uv-env-check**: Lockfile vs. pyproject.toml consistency
2. **pyright-type-check**: Type checking on src/ (strict mode)
3. **ruff-format-check**: Code formatting compliance
4. **ruff-lint-check**: Linting standards
5. **pytest-integration-check**: Full integration tests (requires secrets)
6. **deepeval-tests**: LLM evaluation metrics
7. **gitleaks-check**: Secret detection (pre-commit + CI)

### Pre-commit Hooks

Configured in `.pre-commit-config.yaml`:
- **gitleaks**: Secret scanning
- **uv-lock**: Ensures lockfile consistency

### Constants and Thresholds

Key retrieval constants (`src/vector_indexer/constants.py` and contextual retrieval):
- **Semantic search top-K**: 40 per query
- **Semantic threshold**: 0.4 (cosine similarity ≥0.4 = 50-60% alignment)
- **BM25 top-K**: 40 per query
- **Response generation top-K**: 10 chunks (after RRF fusion)
- **Query refinement count**: 5 variations + original = 6 total
- **Search timeout**: 2 seconds per query

### Docker and Services

- Use `docker compose` (not `docker-compose`)
- Services communicate via `bykstack` network
- Shared volumes: `shared-volume`, `cron_data`
- Vault agent containers per service (llm, gui, cron)
- Resource limits: CPU and memory constraints defined in docker-compose.yml

## Important Notes

- **Python version pinned to 3.12.10** (see `pyproject.toml` and `.python-version`)
- **Line length: 88** (Black-compatible, enforced by Ruff)
- **No print() statements** in production code (use loguru logger)
- **Pydantic for runtime validation** at API boundaries (FastAPI endpoints)
- **Langfuse tracing** for observability (public/secret keys from Vault)
- **Rate limiting** via `RateLimiter` utility (token and request budgets)
- **Cost tracking** via `calculate_total_costs()` and budget tracker
- **Language detection** for multilingual support (Estonian primary)

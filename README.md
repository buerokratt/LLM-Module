# LLM Module

The **LLM Module** is the LLM orchestration component of the
[Bürokratt](https://github.com/buerokratt) ecosystem, providing reliable, multilingual, and compliant
AI-powered responses for Estonian government digital services. It is a **multi-workflow orchestrator**:
a tool classifier inspects every user query and routes it to the most appropriate workflow — answering
from the knowledge base, calling backend services and APIs, handling conversation, or declining
gracefully when a request is out of scope.

## Overview

Rather than treating every request as a single retrieval problem, the LLM Module classifies intent and
dispatches to one of several specialised workflows. Retrieval-Augmented Generation (RAG) is **one** of
these workflows — alongside Service, Context, API-Tool, and Out-of-Domain handling. All workflows run
over configurable, multi-provider LLMs, are protected by safety guardrails, and are fully traced for
cost and quality.

### Key Features

- **Multi-workflow orchestration** — a tool classifier routes each query to the Service, Context,
  **RAG**, API-Tool, or Out-of-Domain (OOD) workflow using hybrid dense + sparse (BM25) search.
- **Configurable LLM providers** — Azure OpenAI, AWS Bedrock, Google Cloud, OpenAI, Anthropic, and
  self-hosted models. Admins create "connections" and switch providers/models without downtime.
- **Grounded, cited answers** — RAG responses are restricted to Central Knowledge Base content, with
  clear citations and an "I don't know" fallback when confidence is low.
- **Agentic API tool calling** — decomposes multi-intent queries and executes multi-endpoint API
  workflows to fulfil actionable requests.
- **Secure credential management** — provider credentials stored in HashiCorp Vault with an additional
  RSA-2048 encryption layer.
- **Safety guardrails** — NeMo Guardrails check input and output content with cost tracking.
- **Observability** — Langfuse for traces/cost analytics and Grafana/Loki for logs.

## Architecture

The LLM Module sits behind the Ruuter API gateway and orchestrates retrieval, generation, and tool
calling over a set of supporting data stores (Qdrant, Redis, PostgreSQL/ClickHouse, MinIO) and
HashiCorp Vault for secrets.

![LLM Module — System Context](./docs/images/LLM%20Module%20Context%20Diagram%20(Current).png)

For the full picture — container and component (C4) diagrams plus the request lifecycle — see
**[docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)**.

## Tech Stack

| Concern | Technology |
| --- | --- |
| Language / runtime | Python 3.12.10 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| API framework | FastAPI + Uvicorn (port `8100`) |
| LLM pipelines | DSPy |
| Safety | NeMo Guardrails |
| Vector database | Qdrant |
| Sessions & history | Redis |
| Analytics store | PostgreSQL + ClickHouse (Langfuse) |
| Object storage | MinIO (S3-compatible) |
| Secrets | HashiCorp Vault |
| Observability | Langfuse, Grafana, Loki |
| API gateway | Ruuter |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (for local development outside containers)

### Run the stack

```bash
# Start the full stack (orchestration service + data stores + tooling)
docker compose up -d

# Check the orchestration service health
curl http://localhost:8100/health
```

### Environment configuration

Configuration is supplied through environment files at the repository root:

| File | Scope |
| --- | --- |
| `.env` | Shared infrastructure (storage, databases, Redis, Vault, feature flags) |
| `.env.llm_orchestration_service` | LLM Orchestration Service |
| `.env.gui` | Admin GUI |
| `.env.notification` | Notification server |

Feature flags such as `TOOL_CLASSIFIER_ENABLED`, `SERVICE_WORKFLOW_ENABLED`,
`CONTEXT_WORKFLOW_ENABLED`, and `API_TOOL_CALLING_WORKFLOW_ENABLED` toggle individual workflows.

## Components

The orchestration service lives under `src/`. The main packages:

| Component | Responsibility | Path |
| --- | --- | --- |
| Orchestration service & API | FastAPI app coordinating the pipeline | `src/llm_orchestration_service_api.py`, `src/llm_orchestration_service.py` |
| Tool classifier & workflows | Intent detection and per-workflow execution (Service / Context / RAG / API-Tool / OOD) | `src/tool_classifier/` |
| Contextual retrieval | Hybrid semantic + BM25 search with RRF fusion | `src/contextual_retrieval/` |
| Vector indexer | Document ingestion and embedding into Qdrant | `src/vector_indexer/` |
| API tool indexer | Indexing of API endpoints for tool calling | `src/api_tool_indexer/` |
| Intent data enrichment | LLM-generated enrichment of intent data | `src/intent_data_enrichment/` |
| Response generator | Grounded answer generation with citations | `src/response_generator/` |
| Prompt refiner | Query refinement for better retrieval | `src/prompt_refine_manager/` |
| Guardrails | NeMo Guardrails input/output safety | `src/guardrails/` |
| LLM configuration | Provider management, Vault credentials, feature flags | `src/llm_orchestrator_config/` |
| Optimization | DSPy-based tuning of guardrails, refiner, generator | `src/optimization/` |
| Utilities | Redis, sessions, rate limiting, streaming, cost, logging | `src/utils/` |

## Core Workflows

The tool classifier routes each query to one of:

- **Service** — maps the query to a backend service/intent ([docs](./docs/TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md))
- **Context** — greetings and conversation handling with Redis-backed history ([docs](./docs/CONTEXT_WORKFLOW_GREETING_DETECTION.md))
- **RAG** — contextual retrieval then grounded generation ([docs](./docs/CONTEXTUAL_RETRIEVAL_FLOW.md))
- **API-Tool** — agentic multi-endpoint API calling ([docs](./docs/API_TOOL_CALLING.md))
- **OOD** — graceful fallback for out-of-domain queries

Classification itself uses hybrid dense + sparse search — see
[docs/HYBRID_SEARCH_CLASSIFICATION.md](./docs/HYBRID_SEARCH_CLASSIFICATION.md).

## Documentation

The full documentation catalogue lives in **[docs/README.md](./docs/README.md)**, covering
architecture, retrieval, workflows, configuration/secrets, sessions, and the API reference.

## Development

```bash
# Install the pinned Python and dependencies
uv python install 3.12.10
uv sync --frozen

# Install pre-commit hooks
uv run pre-commit install

# Run the test suite
uv run pytest tests/ -v
```

See **[CONTRIBUTING.md](./CONTRIBUTING.md)** for the full development workflow, tooling, and CI checks.

## Deployment

Several Docker Compose variants are provided for different environments:

| File | Purpose |
| --- | --- |
| `docker-compose.yml` | Default full stack |
| `docker-compose-ec2.yml` | AWS EC2 deployment variant |
| `docker-compose-test.yml` | Integration testing |
| `docker-compose-eval.yml` | Evaluation / benchmarking |

Kubernetes manifests and Helm charts are under [`kubernetes/`](./kubernetes), including
[`LANGFUSE_SETUP.md`](./kubernetes/LANGFUSE_SETUP.md) and
[`CONTAINER_REGISTRY_SETUP.md`](./kubernetes/CONTAINER_REGISTRY_SETUP.md).

## API Reference

HTTP endpoints for LLM connections, inference results, and chatbot inquiries are documented in
**[docs/API_REFERENCE.md](./docs/API_REFERENCE.md)**.

## Configuration

- **Environment files** — see [Environment configuration](#environment-configuration) above.
- **Feature flags** — `src/llm_orchestrator_config/feature_flags.py`.
- **Service configuration** — YAML configs under each module's `config/` directory (e.g. LLM
  providers, contextual retrieval parameters, indexer settings, guardrails policies).
- **Secrets** — managed in HashiCorp Vault; see
  [docs/LLM_CONFIG_VAULT_INTEGRATION.md](./docs/LLM_CONFIG_VAULT_INTEGRATION.md) and
  [docs/VAULT_SETUP_AND_USAGE.md](./docs/VAULT_SETUP_AND_USAGE.md).

### Storing Langfuse secrets

Generate API keys in the Langfuse UI (**Settings → Project → API Keys**), then store them in Vault.

For Docker Compose deployments, use the [`store-langfuse-secrets.sh`](./store-langfuse-secrets.sh)
script:

```bash
# Copy the script into the vault container
docker cp store-langfuse-secrets.sh vault:/tmp/store-langfuse-secrets.sh

# Run it with your Langfuse keys
docker exec -e LANGFUSE_INIT_PROJECT_PUBLIC_KEY=<your public key> \
            -e LANGFUSE_INIT_PROJECT_SECRET_KEY=<your secret key> \
            vault sh -c "chmod +x /tmp/store-langfuse-secrets.sh && /tmp/store-langfuse-secrets.sh"
```

For Kubernetes, see [kubernetes/LANGFUSE_SETUP.md](./kubernetes/LANGFUSE_SETUP.md).

## Troubleshooting

| Symptom | Where to look |
| --- | --- |
| Service unhealthy | `curl http://localhost:8100/health`; `docker compose ps`; `docker compose logs llm-orchestration-service` |
| Vault / credential errors | [docs/VAULT_SETUP_AND_USAGE.md](./docs/VAULT_SETUP_AND_USAGE.md) and [docs/VAULT_SECURITY_ARCHITECTURE.md](./docs/VAULT_SECURITY_ARCHITECTURE.md) |
| Missing conversation history | Redis connectivity (`REDIS_HOST`/`REDIS_AUTH`); [docs/REDIS_SESSION_STORE.md](./docs/REDIS_SESSION_STORE.md) |
| Retrieval returns nothing | Qdrant availability (port `6333`) and that indexing has run |
| Logs & traces | Grafana/Loki for logs; Langfuse for request traces and cost |

## License

This project is licensed under the terms in the [LICENSE](./LICENSE) file.

## Links

- [Bürokratt project](https://github.com/buerokratt)
- [Architecture](./docs/ARCHITECTURE.md)
- [Documentation index](./docs/README.md)
- [API reference](./docs/API_REFERENCE.md)
- [Contributing](./CONTRIBUTING.md)

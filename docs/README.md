# Documentation Index

Welcome to the documentation for the **LLM Module** — the LLM orchestration component of the
Bürokratt / Estonian Government AI assistant. This index catalogues every document under `docs/`,
grouped by topic. Start with the [project README](../README.md) for the big picture, then dive into
the areas below.

> New to the system? Read [ARCHITECTURE.md](./ARCHITECTURE.md) first — it walks the C4 diagrams from
> system context down to the internal components.

---

## Architecture & Design

| Document | What it covers |
| --- | --- |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | C4 model walkthrough (context → containers → components) with links into every detailed flow. The recommended starting point. |

## Retrieval & Search

| Document | What it covers |
| --- | --- |
| [CONTEXTUAL_RETRIEVAL_FLOW.md](./CONTEXTUAL_RETRIEVAL_FLOW.md) | The RAG workflow in depth: multi-query expansion, hybrid (semantic + BM25) search, RRF rank fusion, thresholds, and quality testing. |
| [HYBRID_SEARCH_CLASSIFICATION.md](./HYBRID_SEARCH_CLASSIFICATION.md) | Tool-classifier architecture using per-example dense (3072-dim) + sparse (BM25) vectors in Qdrant; offline indexing and query-time classification. |

## Tool Classification & Workflows

| Document | What it covers |
| --- | --- |
| [TOOL_CLASSIFIER.md](./TOOL_CLASSIFIER.md) | High-level overview of the classifier: routing model, key functions, and configuration. Start here, then read the per-workflow docs below. |
| [TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md](./TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md) | Service-workflow architecture: high-confidence vs ambiguous routes and service-discovery logic. |
| [CONTEXT_WORKFLOW_GREETING_DETECTION.md](./CONTEXT_WORKFLOW_GREETING_DETECTION.md) | The Context workflow: greeting detection and Redis-backed conversation history with incremental summaries. |
| [API_TOOL_CALLING.md](./API_TOOL_CALLING.md) | The agentic API-Tool workflow end to end: indexing, multi-intent decomposition, multi-endpoint agentic loop, calling, and response formatting. |
| [TESTPRODUCTIONLLM_SERVICE_WORKFLOW.md](./TESTPRODUCTIONLLM_SERVICE_WORKFLOW.md) | Service workflow + streaming via the TestProductionLLM page (three-hop SSE relay). |

## Configuration & Secrets

| Document | What it covers |
| --- | --- |
| [LLM_CONFIG_VAULT_INTEGRATION.md](./LLM_CONFIG_VAULT_INTEGRATION.md) | HashiCorp Vault integration for LLM credentials: KV v2 layout, dev setup, production HA, and migration from `.env`. |
| [VAULT_SETUP_AND_USAGE.md](./VAULT_SETUP_AND_USAGE.md) | Operational guide: dual-network topology, vault-agents, bootstrap flow, AppRole auth, and credential reconciliation. |
| [VAULT_SECURITY_ARCHITECTURE.md](./VAULT_SECURITY_ARCHITECTURE.md) | Security model: threat model, network isolation, AppRole authentication, and the per-policy access-control matrix. |
| [CONNECTION_SWAP_FLOW.md](./CONNECTION_SWAP_FLOW.md) | UUID-based Vault path design enabling zero-I/O environment swaps (promote/demote) between LLM connections. |
| [CUSTOM_PROMPT_CONFIGURATION.md](./CUSTOM_PROMPT_CONFIGURATION.md) | Admin-facing prompt management: database → Ruuter DSL → Python loader with TTL cache and invalidation. |

## Data & Sessions

| Document | What it covers |
| --- | --- |
| [REDIS_SESSION_STORE.md](./REDIS_SESSION_STORE.md) | Redis session-store usage: CRUD for agentic-loop state, TTL behaviour, and the async API. |

## API Reference

| Document | What it covers |
| --- | --- |
| [API_REFERENCE.md](./API_REFERENCE.md) | HTTP API reference: LLM Connections management, Inference Results storage/retrieval, and the chatbot Inquiry endpoint. |

---

## Related resources

- [Project README](../README.md) — overview, quick start, and component map.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — development environment, tooling, and CI checks.
- Architecture diagrams (source images): [`images/`](./images).

# Architecture

This page describes how the **LLM Module** fits together, using the
[C4 model](https://c4model.com/) to move from a high-level system view down to the internal
components of the LLM Orchestration Service. Each level links out to the detailed flow documents
that explain the behaviour in depth.

> The LLM Module is a **multi-workflow orchestrator** for the Bürokratt / Estonian Government AI
> assistant. A tool classifier inspects every user query and routes it to the most appropriate
> workflow — **Service**, **Context**, **RAG**, **API-Tool**, or **Out-of-Domain (OOD)**.
---

## Level 1 — System Context

The context diagram shows the LLM Module as a single system, the people who use it, and the external
systems it depends on (LLM providers, the Central Knowledge Base, observability tooling).

![LLM Module — C4 System Context Diagram](./images/LLM%20Module%20Context%20Diagram%20(Current).png)

**Key relationships**

- **End users / chatbot** send natural-language queries and receive grounded, cited answers.
- **Administrators** configure LLM connections, prompts, budgets, and view analytics.
- **LLM providers** (Azure OpenAI, AWS Bedrock, OpenAI, Anthropic, Google Cloud, self-hosted) supply
  chat and embedding models, selected per connection.
- **Central Knowledge Base (CKB)** provides the source content that is indexed for retrieval.
- **Observability** (Langfuse, Grafana/Loki) captures traces, costs, and logs.

---

## Level 2 — Containers

The container diagram zooms into the deployable units of the system and the data stores they rely on.

![LLM Module — C4 Container Diagram](./images/LLM%20Module%20App%20Diagram%20(Current).png)

**Containers & data stores**

| Container / Store | Role |
| --- | --- |
| **GUI** | Admin web interface for connections, prompts, budgets, and analytics. |
| **Ruuter (public/private)** | API gateway that routes and authorises requests to backend services. |
| **LLM Orchestration Service** | FastAPI service (port `8100`) — the core that runs the workflows. |
| **Notification Server** | Node service pushing real-time updates (e.g. cost alerts, streaming relay). |
| **Qdrant** | Vector database for knowledge-base and API-tool embeddings. |
| **Redis** | Conversation history, session state, and rate-limit counters. |
| **PostgreSQL + ClickHouse** | Relational + columnar stores backing Langfuse analytics. |
| **MinIO (S3)** | Object storage for datasets and documents. |
| **HashiCorp Vault** | Encrypted storage for LLM provider credentials. |

For how requests traverse the gateway and the orchestration service, see
[API_REFERENCE.md](./API_REFERENCE.md).

---

## Level 3 — Components (LLM Orchestration Service)

The component diagram opens up the LLM Orchestration Service to show its internal building blocks: the
tool classifier, the per-workflow executors, the contextual retriever, response generation, and
guardrails.

![LLM Orchestration Service — C4 Component Diagram](./images/LLM%20Orchestration%20Service%20Component%20Diagram%20(Current).png)

### Request lifecycle (high level)

1. **Validation & safety** — input is sanitised and checked against guardrails.
2. **Tool classification** — hybrid dense + sparse (BM25) search over indexed examples routes the
   query to a workflow. See [HYBRID_SEARCH_CLASSIFICATION.md](./HYBRID_SEARCH_CLASSIFICATION.md).
3. **Workflow execution** — one of:
   - **Service** — maps the query to a backend service/intent.
     See [TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md](./TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md).
   - **Context** — greeting/conversation handling with Redis-backed history.
     See [CONTEXT_WORKFLOW_GREETING_DETECTION.md](./CONTEXT_WORKFLOW_GREETING_DETECTION.md).
   - **RAG** — contextual retrieval (hybrid search + RRF fusion) then grounded generation.
     See [CONTEXTUAL_RETRIEVAL_FLOW.md](./CONTEXTUAL_RETRIEVAL_FLOW.md).
   - **API-Tool** — agentic multi-endpoint API calling.
     See [API_TOOL_CALLING.md](./API_TOOL_CALLING.md).
   - **OOD** — graceful fallback for out-of-domain queries.
4. **Generation & guardrails** — the response is generated with citations and re-checked before return.
5. **Observability** — the full trace and cost are recorded in Langfuse; logs go to Loki.

### Component → documentation map

| Concern | Detailed doc |
| --- | --- |
| Tool classifier overview | [TOOL_CLASSIFIER.md](./TOOL_CLASSIFIER.md) |
| Hybrid search & intent enrichment | [HYBRID_SEARCH_CLASSIFICATION.md](./HYBRID_SEARCH_CLASSIFICATION.md) |
| Contextual retrieval (RAG) | [CONTEXTUAL_RETRIEVAL_FLOW.md](./CONTEXTUAL_RETRIEVAL_FLOW.md) |
| API tool calling | [API_TOOL_CALLING.md](./API_TOOL_CALLING.md) |
| Service workflow (UI trace) | [TESTPRODUCTIONLLM_SERVICE_WORKFLOW.md](./TESTPRODUCTIONLLM_SERVICE_WORKFLOW.md) |
| Conversation history & sessions | [REDIS_SESSION_STORE.md](./REDIS_SESSION_STORE.md), [CONTEXT_WORKFLOW_GREETING_DETECTION.md](./CONTEXT_WORKFLOW_GREETING_DETECTION.md) |
| LLM credentials & Vault | [LLM_CONFIG_VAULT_INTEGRATION.md](./LLM_CONFIG_VAULT_INTEGRATION.md), [VAULT_SETUP_AND_USAGE.md](./VAULT_SETUP_AND_USAGE.md), [VAULT_SECURITY_ARCHITECTURE.md](./VAULT_SECURITY_ARCHITECTURE.md) |
| Connection swapping | [CONNECTION_SWAP_FLOW.md](./CONNECTION_SWAP_FLOW.md) |
| Prompt configuration | [CUSTOM_PROMPT_CONFIGURATION.md](./CUSTOM_PROMPT_CONFIGURATION.md) |

---

For the full catalogue of documentation, see the [Documentation Index](./README.md).
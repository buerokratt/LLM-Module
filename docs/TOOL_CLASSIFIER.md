# Tool Classifier

## Overview

The **Tool Classifier** is the entry router of the LLM Module. For every incoming query it inspects the
message (and conversation state) and dispatches it to exactly one **workflow** that knows how to answer.
Retrieval-Augmented Generation (RAG) is just one of those workflows.

This document gives a high-level understanding of the classifier itself — its routing model, key
functions, and configuration. The **behaviour of each individual workflow is documented separately**;
see [Related documentation](#related-documentation).

Source: [`src/tool_classifier/`](../src/tool_classifier/). The classifier only runs when
`TOOL_CLASSIFIER_ENABLED=true`; otherwise the service uses the RAG-only pipeline (backward compatible).

---

## Routing model

Workflows are evaluated as a **layer-wise chain** (Strategy pattern). Each workflow either handles the
query or returns `None` to fall through to the next layer. The order is defined by
`WORKFLOW_LAYER_ORDER` in [`enums.py`](../src/tool_classifier/enums.py):

```
User query
    │
    ├─ active API-tool session for this chat_id?  ──► short-circuit to API_TOOL_CALLING
    │
    ▼
Layer 1: SERVICE            → external Bürokratt service calls
    ↓ (None)
Layer 2: API_TOOL_CALLING   → agentic external API tool calling
    ↓ (None)
Layer 3: CONTEXT            → greetings + conversation-history answers
    ↓ (None)
Layer 4: RAG                → knowledge-base retrieval + generation
    ↓ (None)
Layer 5: OOD                → out-of-domain fallback (always answers)
```

If the classifier errors at any point, it falls back to RAG (`FALLBACK_TO_RAG_ON_ERROR = True`).

---

## How classification works

`classify()` uses a **two-step search** to decide the workflow:

1. **Dense search** — cosine similarity against the service collection for a relevance check.
2. **Hybrid search** — dense + sparse (BM25) vectors fused with Reciprocal Rank Fusion (RRF) to
   identify the best-matching service.

High-confidence matches route straight to `SERVICE`; if no service matches, an API-tool search may route
to `API_TOOL_CALLING`; otherwise the query falls through to `CONTEXT` / `RAG`. The full scoring scheme
(thresholds, score-gap logic, sparse encoding) is documented in
[HYBRID_SEARCH_CLASSIFICATION.md](./HYBRID_SEARCH_CLASSIFICATION.md).

---

## Key functions

### `ToolClassifier` ([`classifier.py`](../src/tool_classifier/classifier.py))

| Method | Purpose |
| --- | --- |
| `classify(query, conversation_history, language, request=None)` | Runs the two-step search and returns a `ClassificationResult` indicating the target workflow. Also handles the active-session short-circuit and intent-switch detection. |
| `route_to_workflow(classification, request, is_streaming, ...)` | Executes the chosen workflow with layer-wise fallback. Returns an `OrchestrationResponse` (non-streaming) or an SSE `AsyncIterator[str]` (streaming). |
| `aclose()` | Releases the shared Qdrant `httpx` client. |

Internal helpers (not part of the public surface): `_dense_search()`, `_hybrid_search()`,
`_try_api_tool_classification()`, and `_execute_with_fallback_async/streaming()`.

### `BaseWorkflow` ([`base_workflow.py`](../src/tool_classifier/base_workflow.py))

Every workflow executor inherits this contract:

| Method | Purpose |
| --- | --- |
| `execute_async(request, context, time_metric=None)` | Non-streaming execution (`/orchestrate`, `/orchestrate/test`). Returns a response, or `None` to fall back to the next layer. |
| `execute_streaming(request, context, time_metric=None)` | Streaming execution (`/orchestrate/stream`). Returns an SSE `AsyncIterator[str]`, or `None` to fall back. |

The **return-`None` fallback** is the mechanism that powers the layer chain.

### `ClassificationResult` ([`models.py`](../src/tool_classifier/models.py))

| Field | Type | Description |
| --- | --- | --- |
| `workflow` | `WorkflowType` | Which workflow should handle the query. |
| `confidence` | `float` (0.0–1.0) | Confidence in the classification. |
| `metadata` | `dict` | Workflow-specific data passed to the executor (e.g. matched service/endpoint). |
| `reasoning` | `str \| None` | Human-readable explanation of the decision. |

---

## Workflow executors

Each `WorkflowType` maps to one executor under
[`src/tool_classifier/workflows/`](../src/tool_classifier/workflows/). Detailed behaviour lives in the
linked docs.

| Workflow | Executor | Detailed documentation |
| --- | --- | --- |
| `SERVICE` | `service_workflow.py` | [TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md](./TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md) |
| `API_TOOL_CALLING` | `api_tool_workflow.py` | [API_TOOL_CALLING.md](./API_TOOL_CALLING.md) |
| `CONTEXT` | `context_workflow.py` | [CONTEXT_WORKFLOW_GREETING_DETECTION.md](./CONTEXT_WORKFLOW_GREETING_DETECTION.md) |
| `RAG` | `rag_workflow.py` | [CONTEXTUAL_RETRIEVAL_FLOW.md](./CONTEXTUAL_RETRIEVAL_FLOW.md) |
| `OOD` | `ood_workflow.py` | — (fixed out-of-domain response) |

---

## Configuration

### Feature flags ([`feature_flags.py`](../src/llm_orchestrator_config/feature_flags.py))

All are environment variables read at startup.

| Flag | Default | Effect |
| --- | --- | --- |
| `TOOL_CLASSIFIER_ENABLED` | `false` | Master switch. When `false`, the service uses the RAG-only pipeline. |
| `SERVICE_WORKFLOW_ENABLED` | `true` | Enables Layer 1 (Service). |
| `API_TOOL_CALLING_WORKFLOW_ENABLED` | `true` | Enables Layer 2 (API tool calling). |
| `CONTEXT_WORKFLOW_ENABLED` | `true` | Enables Layer 3 (Context). |
| `MULTI_INTENT_ENABLED` | `true` | Enables the parallel multi-intent path (IntentDecomposer) in API tool calling. |
| `ATC_RESPONSE_CACHE_ENABLED` | `true` | Enables the two-tier Redis response cache for API tool calling. |
| `FALLBACK_TO_RAG_ON_ERROR` | `true` (constant) | Routes to RAG if the classifier raises. |

> RAG and OOD have no flags — RAG is the core fallback and OOD is the final safety net.

### Classification constants ([`constants.py`](../src/tool_classifier/constants.py))

| Constant | Value | Purpose |
| --- | --- | --- |
| `QDRANT_COLLECTION` | `intent_collections` | Qdrant collection for Bürokratt services. |
| `API_TOOL_COLLECTION` | `api_tool_collection` | Qdrant collection for registered API tool endpoints. |
| `DENSE_MIN_THRESHOLD` | `0.5` | Below this cosine → not a service match. |
| `DENSE_HIGH_CONFIDENCE_THRESHOLD` | `0.55` | At/above (with gap) → SERVICE without LLM confirmation. |
| `DENSE_SCORE_GAP_THRESHOLD` | `0.05` | Required lead of the top service over the runner-up. |
| `API_TOOL_MIN_THRESHOLD` | `0.40` | Below this → no API-tool match. |
| `API_TOOL_HIGH_CONFIDENCE_THRESHOLD` | `0.60` | At/above → API-tool single-path immediately. |
| `API_TOOL_INTENT_SWITCH_THRESHOLD` | `0.50` | Min cosine for a new endpoint to interrupt an active session. |

See [HYBRID_SEARCH_CLASSIFICATION.md](./HYBRID_SEARCH_CLASSIFICATION.md) for how these thresholds combine,
and [API_TOOL_CALLING.md](./API_TOOL_CALLING.md) for the multi-intent / caching constants.

---

## Related documentation

- [Hybrid Search Classification](./HYBRID_SEARCH_CLASSIFICATION.md) — scoring, sparse encoding, intent enrichment.
- [Service Workflow](./TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md) — Layer 1 in depth.
- [API Tool Calling](./API_TOOL_CALLING.md) — Layer 2 agentic loop, multi-intent, response cache.
- [Context Workflow](./CONTEXT_WORKFLOW_GREETING_DETECTION.md) — Layer 3 greetings & history.
- [Contextual Retrieval Flow](./CONTEXTUAL_RETRIEVAL_FLOW.md) — Layer 4 RAG.
- [Architecture](./ARCHITECTURE.md) · [Documentation Index](./README.md)

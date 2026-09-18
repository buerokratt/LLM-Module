# Context Workflow: Greeting Detection and Conversation History Analysis

## Overview

The **Context Workflow (Layer 3)** intercepts user queries that can be answered without searching the knowledge base. It handles two categories:

1. **Greetings** — Detects and responds to social exchanges (hello, goodbye, thanks) in multiple languages
2. **Conversation history references** — Answers follow-up questions that refer to information already discussed in the session

Conversation history is now sourced from a **Redis-backed store** (canonical source) rather than the GUI-provided `request.conversationHistory`. The store retains the most recent 10 rounds per session and maintains an incremental summary of evicted older rounds, enabling context detection to cover the full conversation lifetime.

When the context workflow can answer, a response is returned immediately, bypassing the RAG pipeline entirely. When it cannot answer, the query falls through to the RAG workflow (Layer 4).

---

## Architecture

### Position in the Classifier Chain

```
User Query
    ↓
Layer 1: SERVICE          → External API calls
    ↓ (cannot handle)
Layer 2: API_TOOL_CALLING → Agentic API tool execution
    ↓ (cannot handle)
Layer 3: CONTEXT          → Greetings + conversation history  ←── This document
    ↓ (cannot handle)
Layer 4: RAG              → Knowledge base retrieval
    ↓ (cannot handle)
Layer 5: OOD              → Out-of-domain fallback
```

> **Note**: The classifier also checks for an **active API tool session** (`chatId` present in `session_store`) before any layer evaluation. If found, the request short-circuits directly to the `API_TOOL_CALLING` workflow to continue parameter collection — the context workflow is never reached in this path.

### Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `ContextAnalyzer` | `src/tool_classifier/context_analyzer.py` | LLM-based greeting detection, context analysis, summary generation |
| `ContextWorkflowExecutor` | `src/tool_classifier/workflows/context_workflow.py` | Orchestrates the workflow, history fetching, streaming/non-streaming |
| `ToolClassifier` | `src/tool_classifier/classifier.py` | Invokes `ContextAnalyzer` during classification and routes to `ContextWorkflowExecutor` |
| `FeatureFlags.CONTEXT_WORKFLOW_ENABLED` | `src/llm_orchestrator_config/feature_flags.py` | Guards the context workflow; if `False`, the layer is skipped in the fallback chain and the request proceeds directly to RAG |
| `ConversationHistoryStore` | `src/utils/conversation_history_store.py` | Redis CRUD store for per-session rounds and incremental summary |
| `conversation_summary_generator` | `src/utils/conversation_summary_generator.py` | Factory for the incremental summarizer callable injected into the store |
| `redis_client` | `src/utils/redis_client.py` | Singleton async Redis client (db=1, TLS-capable) |
| `greeting_constants.py` | `src/tool_classifier/greeting_constants.py` | Static greeting response templates for Estonian and English |

---

## Full Request Flow

```
User Query + Conversation History
    ↓
ToolClassifier.classify()
    ├─ Pre-check: active API tool session for chatId?
    │      └─ Yes → short-circuit to API_TOOL_CALLING (context skipped)
    │
    ├─ Layer 1 (SERVICE): Embedding-based intent routing
    │      └─ If no service tool matches → try Layer 2
    │
    ├─ Layer 2 (API_TOOL_CALLING): Semantic search in api_tool_collection
    │      └─ If no API tool matches (or feature disabled) → route to CONTEXT workflow
    │
    └─ ClassificationResult(workflow=CONTEXT)

ToolClassifier.route_to_workflow()
    ├─ Non-streaming → ContextWorkflowExecutor.execute_async()
    │      ├─ _build_history() → ConversationHistoryStore.get_context() [Redis, with fallback]
    │      ├─ Phase 1: _detect() → context_analyzer.detect_context_with_summary_fallback()
    │      │      ├─ Step 1: detect_context() on last 10 turns
    │      │      ├─ Step 2 (if needed): use pre_computed_summary from Redis OR generate summary
    │      │      └─ Step 3 (if needed): _analyze_from_summary() on summary
    │      ├─ If greeting → return greeting OrchestrationResponse (static template)
    │      ├─ If can_answer → _generate_response_async() → context_analyzer.generate_context_response()
    │      └─ Otherwise → return None (RAG fallback)
    │
    └─ Streaming → ContextWorkflowExecutor.execute_streaming()
           ├─ _build_history() → ConversationHistoryStore.get_context() [Redis, with fallback]
           ├─ Phase 1: _detect() → context_analyzer.detect_context_with_summary_fallback()
           ├─ If greeting → _stream_greeting() async generator (static template)
           ├─ If can_answer → _create_history_stream() → context_analyzer.stream_context_response()
           └─ Otherwise → return None (RAG fallback)
```

---

## Redis-Backed Conversation History

### Overview

`ConversationHistoryStore` is a Redis-backed CRUD store (db=1) that holds per-session conversation data. It is the **canonical source of truth** for conversation history, replacing the GUI-provided `request.conversationHistory` when available.

### Key Layout

| Redis Key | Content | TTL |
|-----------|---------|-----|
| `conv:{chat_id}` | JSON list of up to 10 `ConversationRound` objects | 30 minutes (sliding) |
| `conv:summary:{chat_id}` | Plain-text incremental summary of evicted rounds | 30 minutes (sliding) |

Both keys share a sliding TTL: every write resets the expiry on both keys to keep them in sync.

### History Capping and Eviction

The store caps history at **10 rounds** (`_MAX_ROUNDS`). When appending a new round causes the count to exceed 10, the oldest rounds are trimmed. Trimmed (evicted) rounds are passed to an optional `summarizer` callable as a fire-and-forget background `asyncio.Task`, which merges them into the running summary using `IncrementalSummarySignature`.

### `_build_history()` — History Resolution in the Workflow

`ContextWorkflowExecutor._build_history()` resolves the history and pre-computed summary to pass to Phase 1:

1. If `ConversationHistoryStore` is wired in, call `get_context(chat_id)` to retrieve rounds and the optional Redis summary.
2. If rounds are present, flatten them into `{"authorRole", "message", "timestamp"}` dicts and return `(history, summary)`.
3. If the store is absent, raises, or returns no rounds → fall back to `request.conversationHistory` with `summary=None`.

The returned `pre_computed_summary` is forwarded to `detect_context_with_summary_fallback()` to skip an expensive LLM summarisation step when Redis already has one.

### Optimistic Locking

`save_round()` uses Redis `WATCH`/`MULTI`/`EXEC` (optimistic locking) to detect concurrent writes and retries up to 3 times on conflict.

---

## Phase 1: Detection (Classify Only)

### Three-Step Detection Flow

Every query is processed by `detect_context_with_summary_fallback()`, which implements a three-step detection pipeline:

**Step 1 — Recent turns check (`detect_context`)**

Runs `ContextDetectionSignature` via `dspy.ChainOfThought` against the **most recent 10 conversation turns**. This phase **does not generate an answer** — it only classifies the query and extracts a relevant context snippet for Phase 2.

**Step 2 — Summary path (triggered when Step 1 cannot answer)**

Triggered when `can_answer_from_context=False` AND one of the following is true:
- Total history exceeds 10 turns (older turns exist), OR
- Redis has a `pre_computed_summary` (covers evicted rounds beyond the current active window)

Two sub-paths:
- **Redis path**: Pre-computed summary is available → used directly (no LLM call, zero cost).
- **On-demand path**: No pre-computed summary → older turns (beyond last 10) are summarised via `ConversationSummarySignature`.

**Step 3 — Summary analysis (`_analyze_from_summary`)**

Runs `SummaryAnalysisSignature` against the summary string to determine if the query can be answered from it. If so, the summary-derived answer is returned as `context_snippet` (with `answered_from_summary=True`) for Phase 2 generation.

### LLM Output Format

`ContextDetectionSignature` returns a JSON object parsed into `ContextDetectionResult`:

```json
{
  "is_greeting": false,
  "greeting_type": "hello",
  "can_answer_from_context": true,
  "reasoning": "User is asking about tax rate discussed earlier",
  "context_snippet": "Bot confirmed the flat rate is 20%, applying equally to all income brackets."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_greeting` | `bool` | Whether the query is a greeting |
| `greeting_type` | `str` | One of `hello`, `goodbye`, `thanks`, `casual` (relevant when `is_greeting=True`) |
| `can_answer_from_context` | `bool` | Whether the query can be answered from conversation history |
| `reasoning` | `str` | Brief explanation of the detection decision |
| `context_snippet` | `str \| null` | Relevant excerpt from history or summary for Phase 2, or `null` |
| `answered_from_summary` | `bool` | `True` when the answer was derived from the summary path (internal, default `False`) |

### Decision After Phase 1

```
is_greeting=True                              → Phase 2: return greeting response (static template, no LLM)
can_answer_from_context=True AND snippet set  → Phase 2: generate answer from snippet
Otherwise (all steps exhausted)               → Fall back to RAG
```

---

## Phase 2: Response Generation

### Non-Streaming (`_generate_response_async`)

Calls `generate_context_response(query, context_snippet)` which uses `ContextResponseGenerationSignature` to produce a complete answer in a single LLM call. Output guardrails are applied before returning the `OrchestrationResponse`.

### Streaming (`_create_history_stream` → `stream_context_response`)

Calls `stream_context_response(query, context_snippet)` which uses DSPy native streaming (`dspy.streamify`) with `ContextResponseGenerationSignature`. A fresh `StreamListener` is created per call to avoid stale state. Tokens are yielded in real time and passed through NeMo Guardrails before being SSE-formatted.

**Fallback chain inside `stream_context_response`:**
1. DSPy `streamify` → yield `StreamResponse` tokens as they arrive.
2. If no stream tokens received but the final `Prediction` has an answer, yield it in word-group chunks.
3. If that is also empty, call `generate_context_response()` directly and yield its result in word-group chunks.

---

## Greeting Detection

### Supported Languages

| Language | Code |
|----------|------|
| Estonian | `et` |
| English | `en` |

### Supported Greeting Types

| Type | Estonian Examples | English Examples |
|------|-------------------|-----------------|
| `hello` | Tere, Hei, Tervist, Moi | Hello, Hi, Hey, Good morning |
| `goodbye` | Nägemist, Tšau | Bye, Goodbye, See you, Good night |
| `thanks` | Tänan, Aitäh, Tänud | Thank you, Thanks |
| `casual` | Tere, Tervist | Hey |

### Greeting Response Generation

Greeting detection is handled in **Phase 1 (`detect_context`)**, where the LLM classifies whether the query is a greeting, identifies the `greeting_type`, and sets `is_greeting=True`. A message is only treated as a greeting if it contains **nothing beyond the greeting itself** — a greeting combined with a question is routed to RAG instead.

In **Phase 2**, `ContextWorkflowExecutor` calls `get_greeting_response(greeting_type=..., language=...)`, which returns a static template from `greeting_constants.py`. The language is determined by `detect_language()` on the user query. No LLM call is made for greeting responses.

**Greeting response templates (`greeting_constants.py`):**

```python
GREETINGS_ET = {
    "hello": "Tere! Kuidas ma saan sind aidata?",
    "goodbye": "Nägemist! Head päeva!",
    "thanks": "Palun! Kui on veel küsimusi, küsi julgelt.",
    "casual": "Tere! Mida ma saan sinu jaoks teha?",
}

GREETINGS_EN = {
    "hello": "Hello! How can I help you?",
    "goodbye": "Goodbye! Have a great day!",
    "thanks": "You're welcome! Feel free to ask if you have more questions.",
    "casual": "Hey! What can I do for you?",
}
```

---

## Streaming Support

The context workflow supports both response modes:

### Non-Streaming (`execute_async`)

Returns a complete `OrchestrationResponse` object with the answer as a single string. Output guardrails are applied before the response is returned. If a `pre_computed_analysis_result` is present in the classifier context, Phase 1 is skipped entirely (reuses the already-computed detection).

### Streaming (`execute_streaming`)

Returns an `AsyncIterator[str]` that yields SSE (Server-Sent Events) chunks.

**Greeting responses** are yielded as a single SSE chunk followed by `END`.

**History responses** use DSPy native streaming (`dspy.streamify`) with `ContextResponseGenerationSignature`. Tokens are emitted in real time as they arrive from the LLM, then passed through NeMo Guardrails (`stream_with_guardrails`) before being SSE-formatted. If a guardrail violation is detected in a chunk, streaming stops and the violation message is sent instead.

### Conversation History Persistence (Streaming)

After streaming completes, `llm_orchestration_service.py` saves a `ConversationRound` to `ConversationHistoryStore` — but **only for non-RAG workflows** (SERVICE, API_TOOL_CALLING, CONTEXT). RAG has its own internal save hook inside `_stream_rag_pipeline` and does not go through this path. The accumulated content is filtered to exclude SSE control tokens (`END`) and predefined excluded messages before saving.

**SSE Format:**
```
data: {"chatId": "abc123", "payload": {"content": "Tere! Kuidas ma"}, "timestamp": "...", "sentTo": []}

data: {"chatId": "abc123", "payload": {"content": " saan sind aidata?"}, "timestamp": "...", "sentTo": []}

data: {"chatId": "abc123", "payload": {"content": "END"}, "timestamp": "...", "sentTo": []}
```

---

## Cost Tracking

LLM token usage and cost is tracked via `get_lm_usage_since()` and stored in `costs_metric` within the workflow executor. Costs are logged via `orchestration_service.log_costs()` at the end of each execution path.

Two cost keys are tracked separately. When the summary fallback path is taken, its LLM calls are **merged into** `context_detection`:

```python
costs_metric = {
    "context_detection": {
        # Phase 1: detect_context() + optional summary generation + summary analysis
        # All summary-path costs are merged here via _merge_cost_dicts()
        "total_cost": 0.0012,
        "total_tokens": 180,
        "total_prompt_tokens": 150,
        "total_completion_tokens": 30,
        "num_calls": 1,
    },
    "context_response": {
        # Phase 2: generate_context_response() or stream_context_response()
        "total_cost": 0.003,
        "total_tokens": 140,
        "total_prompt_tokens": 100,
        "total_completion_tokens": 40,
        "num_calls": 1,
    },
}
```

Greeting responses skip Phase 2, so only `"context_detection"` cost is populated. When the Redis pre-computed summary is used, the summary-generation cost is zero (no LLM call).

---

## Error Handling and Fallback

| Failure Point | Behaviour |
|---------------|-----------|
| Redis unavailable (`ConversationHistoryStore`) | Logged as warning → falls back to `request.conversationHistory` |
| Redis fetch raises exception | Logged as warning → falls back to `request.conversationHistory` |
| Phase 1 LLM call raises exception | `can_answer_from_context=False` → falls back to RAG |
| Phase 1 returns invalid JSON | Logged as warning, all flags default to `False` → falls back to RAG |
| Summary generation (on-demand) fails | Logged as error → summary path skipped → falls back to RAG |
| Summary analysis returns no answer | Logged as info → falls back to RAG |
| Phase 2 LLM call raises exception | Logged as error, `_generate_response_async` returns `None` → falls back to RAG |
| Phase 2 returns empty answer | Logged as warning → falls back to RAG |
| All Phase 2 streaming fallbacks exhausted | Logged as error → empty response |
| Output guardrails fail | Logged as warning, response returned without guardrail check |
| Guardrail violation in streaming | `OUTPUT_GUARDRAIL_VIOLATION_MESSAGE` sent, stream terminated |
| `orchestration_service` unavailable | History streaming skipped → `None` returned → RAG fallback |
| `guardrails_adapter` not a `NeMoRailsAdapter` | Logged as warning → cannot stream → RAG fallback |
| Any unhandled exception in executor | Error logged, `execute_async/execute_streaming` returns `None` → RAG fallback via classifier |

---

## Logging

Key log entries emitted during a request:

| Level | Message | When |
|-------|---------|------|
| `INFO` | `CONTEXT WORKFLOW (NON-STREAMING) \| Query: '...'` | `execute_async()` entry |
| `INFO` | `CONTEXT WORKFLOW (STREAMING) \| Query: '...'` | `execute_streaming()` entry |
| `DEBUG` | `[chatId] Using Redis history: N rounds, summary=present\|absent` | Redis history fetched successfully |
| `WARNING` | `[chatId] Redis history fetch failed, falling back to request history: ...` | Redis read error |
| `INFO` | `CONTEXT DETECTOR: Phase 1 \| Query: '...' \| History: N turns` | `detect_context()` entry |
| `INFO` | `DETECTION RESULT \| Greeting: ... \| Can Answer: ... \| Has snippet: ...` | Phase 1 LLM response parsed |
| `INFO` | `Detection cost \| Total: $... \| Tokens: N` | After Phase 1 cost tracked |
| `INFO` | `Pre-computed summary available \| Skipping LLM summary generation, using Redis summary directly` | Redis summary reused |
| `INFO` | `History has N turns (> 10) \| Cannot answer from recent 10 \| Attempting summary-based detection` | On-demand summary path triggered |
| `INFO` | `DETECTION: Can answer from summary \| Reasoning: ...` | Summary path answered query |
| `INFO` | `Cannot answer from summary either \| Falling back to RAG` | Summary path failed |
| `INFO` | `Detection: greeting=... can_answer=...` | After `_detect()` returns in executor |
| `INFO` | `CONTEXT GENERATOR: Phase 2 non-streaming \| Query: '...'` | `generate_context_response()` entry |
| `INFO` | `CONTEXT GENERATOR: Phase 2 streaming \| Query: '...'` | `stream_context_response()` entry |
| `INFO` | `Context response streaming complete (final Prediction received)` | DSPy streaming finished |
| `WARNING` | `Stream tokens not received — yielding answer from final Prediction in chunks.` | Streaming fallback 1 |
| `WARNING` | `No answer from streamify — falling back to generate_context_response.` | Streaming fallback 2 |
| `WARNING` | `[chatId] Phase 2 empty answer — fallback to RAG` | Phase 2 returned no content |
| `WARNING` | `[chatId] Guardrails violation in context streaming` | Violation detected mid-stream |
| `WARNING` | `[chatId] Cannot answer from context — falling back to RAG` | Neither phase could answer |

---

## Data Models

### `ConversationRound` (Redis storage unit)

```python
class ConversationRound(BaseModel):
    user_message: str  # The user's message text
    bot_message: str  # The bot's response text
    timestamp: float  # Unix timestamp of the round
```

### `ConversationHistoryState` (Redis fetch result)

```python
class ConversationHistoryState(BaseModel):
    chat_id: str  # Unique conversation identifier
    rounds: list[ConversationRound]  # Ordered rounds (newest last), capped at 10
    summary: Optional[str]  # Incremental summary of evicted older rounds
```

### `ContextDetectionResult` (Phase 1 output)

```python
class ContextDetectionResult(BaseModel):
    is_greeting: bool  # True if query is a greeting
    greeting_type: str  # "hello" | "goodbye" | "thanks" | "casual"
    can_answer_from_context: (
        bool  # True if query can be answered from history or summary
    )
    reasoning: str  # LLM's brief explanation
    answered_from_summary: bool  # True when answer derived from summary path
    context_snippet: Optional[str]  # Relevant excerpt for Phase 2 generation, or None
```

### `ContextDetectionSignature` (DSPy — Phase 1, recent turns)

| Field | Type | Description |
|-------|------|-------------|
| `conversation_history` | Input | Last 10 turns formatted as JSON |
| `user_query` | Input | Current user query |
| `detection_result` | Output | JSON with `is_greeting`, `greeting_type`, `can_answer_from_context`, `reasoning`, `context_snippet` |

> Detection only — **no answer generated here**.

### `ConversationSummarySignature` (DSPy — on-demand summary generation)

| Field | Type | Description |
|-------|------|-------------|
| `conversation_history` | Input | JSON of older turns to summarize |
| `summary` | Output | Concise summary preserving key facts, names, numbers, dates |

### `IncrementalSummarySignature` (DSPy — background eviction summary)

| Field | Type | Description |
|-------|------|-------------|
| `existing_summary` | Input | Current summary (may be empty for first eviction) |
| `new_rounds` | Input | JSON array of just-evicted rounds |
| `updated_summary` | Output | Merged summary incorporating new rounds |

### `SummaryAnalysisSignature` (DSPy — Phase 1, summary path)

| Field | Type | Description |
|-------|------|-------------|
| `conversation_summary` | Input | Summary of earlier conversation |
| `user_query` | Input | Current user query |
| `analysis_result` | Output | JSON with `can_answer_from_context`, `answer`, `reasoning` |

### `ContextResponseGenerationSignature` (DSPy — Phase 2)

| Field | Type | Description |
|-------|------|-------------|
| `context_snippet` | Input | Relevant excerpt from Phase 1 (or summary-derived answer) |
| `user_query` | Input | Current user query |
| `answer` | Output | Natural language response in the same language as the query |

---

## Decision Summary Table

| Scenario | Phase 1 LLM Calls | Phase 2 LLM Calls | Outcome |
|----------|--------------------|--------------------|---------|
| Greeting detected | 1 (`detect_context`) | 0 (static template) | Context responds (greeting) |
| Follow-up answerable from last 10 turns | 1 (`detect_context`) | 1 (`generate_context_response` or `stream_context_response`) | Context responds |
| Cannot answer from 10 turns; Redis summary answers | 1 + 1 (`detect_context` + `_analyze_from_summary`) | 1 | Context responds (summary path) |
| Cannot answer from 10 turns; Redis summary reused (no new LLM call) | 1 + 1 (`detect_context` + `_analyze_from_summary`; 0 for summary gen) | 1 | Context responds (Redis summary path) |
| Cannot answer from 10 turns; on-demand summary answers | 1 + 1 + 1 (`detect_context` + `_generate_conversation_summary` + `_analyze_from_summary`) | 1 | Context responds (on-demand summary path) |
| Cannot answer from any path | 1–3 (all detection steps) | 0 | Falls back to RAG |
| Phase 1 LLM error / JSON parse failure | — | 0 | Falls back to RAG |
| Phase 2 LLM error or empty answer | 1–3 | — | Falls back to RAG |
| Redis unavailable | 0 (fallback to request history) | varies | Proceeds normally with request history |

---

## File Reference

| File | Purpose |
|------|---------|
| `src/tool_classifier/context_analyzer.py` | Core LLM analysis logic (detection, summary generation, response generation) |
| `src/tool_classifier/workflows/context_workflow.py` | Workflow executor (history fetching, streaming + non-streaming) |
| `src/tool_classifier/classifier.py` | Classification layer that invokes context analysis |
| `src/tool_classifier/greeting_constants.py` | Static greeting response templates (ET/EN) |
| `src/utils/conversation_history_store.py` | Redis CRUD store for rounds and incremental summary |
| `src/utils/conversation_summary_generator.py` | Factory for the incremental summarizer callable |
| `src/utils/redis_client.py` | Singleton async Redis client (db=1, TLS-capable) |
| `src/models/conversation_history_models.py` | Pydantic models: `ConversationRound`, `ConversationHistoryState` |
| `tests/test_context_analyzer.py` | Unit tests for `ContextAnalyzer` |
| `tests/test_context_workflow.py` | Unit tests for `ContextWorkflowExecutor` |
| `tests/test_context_workflow_integration.py` | Integration tests for the full classify → route → execute chain |
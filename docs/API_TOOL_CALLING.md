# API Tool Calling — Architecture & Implementation



## Overview

API Tool Calling enables the LLM module to discover and invoke external API endpoints
in response to user queries. Endpoints are registered, semantically indexed in Qdrant,
and retrieved at query time using hybrid search. Once matched, a multi-turn agentic
loop collects all required parameters from the user before the API call is made.


| Component | What it does | Status |
|---|---|---|
| **Indexing pipeline** | Takes an endpoint definition → enriches it with LLM context → stores hybrid vectors in Qdrant |  Complete |
| **Tool classifier** | At query time, routes to the best matching endpoint via hybrid search + LLM disambiguation |  Complete |
| **Multi-intent detection** | Score-band gate triggers `IntentDecomposer` (DSPy) to decompose a multi-intent query into focused sub-queries; each sub-query is matched in parallel via `asyncio.gather` |  Phase 1 & 2 Complete |
| **Agentic loop** | Multi-turn parameter collection with session persistence, language-aware clarifying questions, param correction, continuation prompt, and intent-switch detection |  Complete |
| **API caller** | Execute collected params against the real API endpoint, with circuit-breaker protection and localized error handling |  Complete |
| **Response formatter** | Convert raw API JSON into a natural-language answer via DSPy, streamed token-by-token to the GUI |  Complete |
| **Multi-endpoint loop** | Merges param schemas for all parallel endpoints; collects params across turns with a single deduplicated clarifying question per turn; distributes values back per endpoint |  Phase 3 Complete |
| **Parallel API caller** | Fires all completed endpoint calls concurrently via `asyncio.gather` with batch timeout and partial-failure handling |  Phase 4 Complete |
| **Multi-response formatter** | DSPy module that synthesises N API results into a single coherent natural-language answer; supports streaming and blocking execution |  Phase 5 Complete |
| **Full wiring** | `APIToolWorkflowExecutor` routes parallel sessions through `MultiEndpointAgenticLoop` → `MultiAPICaller` → `MultiResponseFormatterModule` with output guardrails |  Phase 6 Complete |
| **ATC Response Cache** | Two-tier Redis cache (L1 exact-match + L2 follow-up context) that eliminates redundant API calls and enables intelligent follow-up handling without re-running the agentic loop |  Complete |

---

## System Components

```
Ruuter DSL  (/api-tools/index)
        ↓  HTTP POST
CronManager  (api_tool_indexer job)
        ↓  exec
api_tool_indexer.sh  (bash)
        ↓  python3
main_indexer.py  (indexing pipeline)
        ↓  upsert
api_tool_collection  (Qdrant)
        ↑  query at runtime
APISemanticSearcher  (src/tool_classifier/api_semantic_searcher.py)
        ↑  called by
ToolClassifier._try_api_tool_classification()
        │
        ├─ score ≥ HIGH_CONFIDENCE → single path (unchanged)
        │
        └─ score in ambiguous band → IntentDecomposer (DSPy)
               │
               ├─ mode=single  → top candidate, existing path
               └─ mode=parallel → asyncio.gather(search per sub-query)
                      ↓  ClassificationResult(execution_mode=parallel, matched_endpoints=[...])
        ↓  ClassificationResult(workflow=API_TOOL_CALLING)
APIToolWorkflowExecutor  (src/tool_classifier/workflows/api_tool_workflow.py)
        │
        ├─ execution_mode=single  → AgenticLoop
        └─ execution_mode=parallel → MultiEndpointAgenticLoop
               ↓  merged schema; one clarifying question per turn
               ↓  distributes extracted values back per endpoint
APIToolSessionStore  (Redis, keyed by chat_id, 30-min TTL)
        ↓  all endpoints completed
        ├─ single  → APICaller
        └─ parallel → MultiAPICaller → asyncio.gather per endpoint
               ↓  batch timeout: MULTI_API_BATCH_TIMEOUT (30 s)
               ↓  raw JSON responses (partial-failure safe)
        ├─ single  → APIResponseFormatterModule
        └─ parallel → MultiResponseFormatterModule (DSPy)
               ↓  buffer-first guardrails validation
               ↓  SSE token stream
User (GUI)
```

---

## Part 1 — Indexing Pipeline

### Trigger: `POST /rag-search/api-tools/index`

Defined in [DSL/Ruuter.public/rag-search/POST/api-tools/index.yml](../DSL/Ruuter.public/rag-search/POST/api-tools/index.yml).

**Request body** (sent from Postman while no UI exists):

```json
{
  "endpointId": "a3f7c2d1-84e6-4b19-92f3-d51c7e890ab2",
  "serviceId": "",
  "name": "get_national_holidays",
  "description": "Fetch national holidays for a specific country to see when they have public days off.",
  "method": "GET",
  "url": "https://openholidaysapi.org/PublicHolidays",
  "visibility": "public",
  "type": "custom_endpoint",
  "params": [
    {"name": "countryIsoCode", "type": "string", "required": true,  "description": "The 2-letter ISO country code (e.g., EE for Estonia, DE for Germany)"},
    {"name": "languageIsoCode", "type": "string", "required": false, "description": "The 2-letter ISO language code (e.g., ET, EN)"},
    {"name": "validFrom",       "type": "date",   "required": false, "description": "Start date for the holiday search (YYYY-MM-DD)"},
    {"name": "validTo",         "type": "date",   "required": false, "description": "End date for the holiday search (YYYY-MM-DD)"}
  ]
}
```

Ruuter URL-encodes `params` (JSON array → `encodeURIComponent`) and forwards everything to
CronManager via:

```
POST http://cron-manager:8080/execute/api_tool_indexer/index_endpoint
     ?endpoint_id=...&name=...&params=%5B...%5D
```

---

### CronManager Job: `api_tool_indexer`

Defined in [DSL/CronManager/DSL/api_tool_indexer.yml](../DSL/CronManager/DSL/api_tool_indexer.yml).

```yaml
index_endpoint:
  trigger: off         # Not scheduled — on-demand only
  type: exec
  command: "/app/scripts/api_tool_indexer.sh"
  allowedEnvs: ['endpoint_id', 'service_id', 'name', 'description',
                 'method', 'url', 'visibility', 'type', 'params']
```

`trigger: off` means this job is never run on a schedule — it only executes when
CronManager receives an HTTP `POST /execute/api_tool_indexer/index_endpoint` call.
The query params from Ruuter are injected as environment variables for the shell script.

---

### Shell Script: `api_tool_indexer.sh`

Defined in [DSL/CronManager/script/api_tool_indexer.sh](../DSL/CronManager/script/api_tool_indexer.sh).

**What it does (in order):**

1. Validates required env vars (`endpoint_id`, `name`, `description`, `url`)
2. Activates the pre-built Python venv at `/app/python_virtual_env`
3. Installs required packages via `uv pip install` (`httpx`, `pydantic`, `qdrant-client`, `loguru`)
4. Sets `PYTHONPATH` to include `/app/src`
5. URL-decodes the `params` env var back to a JSON array and writes it to a temp file
6. Invokes `main_indexer.py` with CLI args (avoids shell parsing issues with JSON)
7. Cleans up the temp file; exits with the Python exit code



---

### Python Indexing Pipeline: `main_indexer.py`

Defined in [src/api_tool_indexer/main_indexer.py](../src/api_tool_indexer/main_indexer.py).

**Entry point**: `index_endpoint(endpoint_data: EndpointData) → IndexingResult`

The pipeline runs 5 sequential steps:

#### Step 1 — LLM Context Generation

Builds a structured prompt from the endpoint's name, description, method, URL, and
params, then calls the internal `/generate-context` endpoint via `LLMAPIClient`.

The `CONTEXT_TEMPLATE` (in `constants.py`) instructs the LLM to generate a rich
semantic description covering:
- What the user wants to accomplish by calling this endpoint
- Key terms and synonyms
- Related concepts and use cases
- Common natural language phrasings
- Response is in the **same language as the description** (Estonian / English / Russian)

**`embed_text`** is then assembled as:

```
{name}. {description}. {enriched_context}. Parameters: {params_summary}
```

where `params_summary` is a semicolon-separated one-liner like:
```
countryIsoCode (string, required): ISO country code; validFrom (date, optional): Start date
```

#### Step 2 — Dense Embedding

`embed_text` is sent to Azure OpenAI `text-embedding-3-large` via `LLMAPIClient.create_embedding()`.
Returns a **3072-dimensional float vector** (cosine similarity space).

#### Step 3 — Sparse (BM25) Vector

`embed_text` is tokenised and hashed using `compute_sparse_vector()` from
[src/tool_classifier/sparse_encoder.py](../src/tool_classifier/sparse_encoder.py)
(shared with the tool classifier).

```
tokens = regex word-split of lowercase embed_text
index  = MD5(token)[:4 bytes] % 50_000  (hash to vocab space)
value  = term frequency (collisions are accumulated)
```

Returns `SparseVector(indices=[...], values=[...])` — sorted for consistency.

#### Step 4 — Delete Existing Qdrant Point (idempotent)

`ApiToolQdrantManager.delete_endpoint_point(endpoint_id)` filters by `endpoint_id`
field in the payload and deletes the point before upserting. This ensures re-indexing
the same endpoint never creates duplicates.

#### Step 5 — Upsert to Qdrant

A `PointStruct` is built:

```python
PointStruct(
    id=endpoint_id,  # UUID used directly as Qdrant point ID
    vector={
        "dense": [v1, v2, ..., v3072],
        "sparse": {"indices": [...], "values": [...]},
    },
    payload={  # Stored metadata — no extra DB lookup needed
        "endpoint_id": "...",
        "name": "get_national_holidays",
        "description": "...",
        "url": "https://openholidaysapi.org/PublicHolidays",
        "method": "GET",
        "params": [...],
        "enriched_context": "...",
        "service_id": "...",
    },
)
```

Upserted into the `api_tool_collection` Qdrant collection.

---

### Qdrant Collection: `api_tool_collection`

Created automatically on first run by `ApiToolQdrantManager.ensure_collection()`.

```
Collection:  api_tool_collection
Vectors:
  "dense"  → VectorParams(size=3072, distance=COSINE)
  "sparse" → SparseVectorParams(index=SparseIndexParams(on_disk=False))
```

One point per endpoint. The full `EnrichedEndpoint` payload is stored so the agentic
loop can execute the API call without an additional database round-trip.

---

### Data Models

**`EndpointData`** — input to the pipeline (from Postman / DB):

| Field | Type | Required | Description |
|---|---|---|---|
| `endpoint_id` | UUID |  | Unique identifier |
| `name` | str |  | snake_case function name |
| `description` | str |  | Human-readable purpose |
| `url` | str |  | Full target API URL |
| `method` | str |  | `GET` or `POST` |
| `params` | List[Dict] | | Parameter schema `[{name, type, required, description}]` |
| `service_id` | UUID | | Parent service group |
| `visibility` | str | | `public` or `private` (default: `public`) |
| `type` | str | | Endpoint type (default: `custom_endpoint`) |

**`ParamSchema`** — schema for each param:

| Field | Type | Description |
|---|---|---|
| `name` | str | Parameter name |
| `type` | str | `string`, `date`, `datetime`, `integer`, `boolean`, `number` |
| `required` | bool | Whether the caller must supply this param |
| `description` | str | Human-readable description |

> **`datetime` type:** normalised to `YYYY-MM-DDTHH:MM:SSZ` by `ParamExtractionModule._validate_param_type()`. Useful for APIs that require ISO 8601 datetime strings (e.g. electricity price endpoints).

---


## Part 2 — Tool Classifier (Query-Time)

### Overview

At query time, `ToolClassifier` in [src/tool_classifier/classifier.py](../src/tool_classifier/classifier.py) the layer by layer execution happens 


1. **Service search** → `intent_collections` (Qdrant) — existing Bürokratt services
2. **API Tool search** → `api_tool_collection` (Qdrant) — registered API tool endpoints

API tool search (`_try_api_tool_classification`) is triggered when:
- `SERVICE_WORKFLOW_ENABLED=false` (service workflow disabled globally)
- Dense service search returns no results
- Service cosine score falls below `DENSE_MIN_THRESHOLD`

It is **always** tried before falling back to Context/RAG.

---

### Component: `APISemanticSearcher`

Defined in [src/tool_classifier/api_semantic_searcher.py](../src/tool_classifier/api_semantic_searcher.py).

Instantiated once in `ToolClassifier.__init__()` and reuses the shared Qdrant `httpx.AsyncClient`.

**Constructor:**

```python
APISemanticSearcher(
    embedding_service=orchestration_service,  # generates dense embeddings
    qdrant_client=self._qdrant_client,  # shared connection pool
    disambiguator=None,  # optional: inject for testing
)
```

**Key constants** (from `constants.py`):

| Constant | Value | Purpose |
|---|---|---|
| `API_TOOL_COLLECTION` | `api_tool_collection` | Qdrant collection name |
| `API_TOOL_SEARCH_TOP_K` | `5` | Max hybrid results |
| `API_TOOL_MIN_THRESHOLD` | cosine threshold | Below this → no match |
| `API_TOOL_HIGH_CONFIDENCE_THRESHOLD` | cosine threshold | Above this → high confidence |
| `API_TOOL_SCORE_GAP_THRESHOLD` | gap threshold | Minimum lead over runner-up |

---

### Search Flow: `APISemanticSearcher.search()`

```
User query
    │
    ├─ precomputed_embedding provided? → reuse it (no extra API call)
    └─ otherwise → generate dense embedding via embedding_service
    │
    ▼
Step 1: Dense search (api_tool_collection)
    → Real cosine similarity scores per endpoint
    │
    ├─ No results → return []
    ├─ top_cosine < API_TOOL_MIN_THRESHOLD → return []
    └─ continue
    │
    ▼
Step 2: Hybrid search (dense + sparse/BM25 + RRF)
    → Best-ranked results by RRF fusion score
    │  Falls back to dense results if hybrid returns nothing
    │
    ▼
Step 3: Annotate confidence for each hybrid result
    │
    │  cosine lookup: dense_cosine_map[endpoint_id]
    │             └─ fallback: point["cosine_score"] (sparse-driven result)
    │             └─ skip if neither available
    │
    │  effective_gap = this_cosine − best_other_cosine_in_dense
    │
    ├─ i==0 AND cosine ≥ HIGH_THRESHOLD AND effective_gap ≥ GAP_THRESHOLD → "high"
    ├─ cosine ≥ MIN_THRESHOLD → "medium"
    └─ else → skip
    │
    ▼
Step 4: Resolve to exactly one result
    ├─ high-confidence result exists → return immediately
    ├─ single medium + large gap → return directly
    └─ multiple medium OR small gap → LLM disambiguation
           │
           └─ EndpointDisambiguatorModule (DSPy + asyncio.to_thread)
                  → picks winner or returns None
                  → None means no match → return []
```

---

### Embedding Reuse

When `ToolClassifier.classify()` already generated a dense embedding for the service
search, it passes it as `precomputed_embedding` to `_try_api_tool_classification`:

```python
api_tool_result = await self._try_api_tool_classification(
    query, request, precomputed_embedding=query_embedding
)
```

`APISemanticSearcher.search()` skips the embedding step entirely when this is provided,
saving one embedding API call per request.

---

### LLM Disambiguation: `EndpointDisambiguatorModule`

Used when multiple medium-confidence endpoints score similarly and no clear winner
can be determined from cosine scores alone.

- DSPy `Predict` module with `EndpointDisambiguationSignature`
- Inputs: `user_query` + `candidates` (JSON list of `{endpoint_id, name, description, cosine_score}`)
- Output: `best_endpoint_id` — the winning `endpoint_id`, or `"none"` if no match
- Run via `asyncio.to_thread()` to avoid blocking the async event loop
- Understands Estonian, Russian, and English queries

---

### Feature Flag

API tool calling is gated by `FeatureFlags.API_TOOL_CALLING_WORKFLOW_ENABLED`.
When `false`, `_try_api_tool_classification` returns `None` immediately without
touching Qdrant.

---

### Component: `APIToolWorkflowExecutor`

Defined in [src/tool_classifier/workflows/api_tool_workflow.py](../src/tool_classifier/workflows/api_tool_workflow.py).

Handles `WorkflowType.API_TOOL_CALLING` after `ToolClassifier.classify()` has set
`matched_endpoint` in the context dict.

**Responsibilities:**

- **Turn 1 (new session):** reads `context["matched_endpoint"]`, creates a new
  `APIToolSession` in Redis, runs the first agentic loop turn.
- **Turn 2-N (resume):** loads the existing session from Redis, runs the next turn.
- **Fast path:** if the endpoint has no required params, immediately calls the API
  without starting a session.
- **Clarifying question:** when params are still missing, streams the LLM-generated
  question token-by-token via SSE. Each token is one `format_sse` frame; the stream
  ends with an `END` frame.
- **API call:** when all params are collected, calls `APICaller.call()` then streams
  the natural-language answer from `APIResponseFormatterModule.stream_forward()`
  token-by-token via SSE.
- **Max turns:** deletes the session and returns `None` to trigger RAG fallback.

**Streaming architecture:**

Both clarifying questions and final responses are streamed token-by-token.
`_compute_loop_step()` is the single source of truth — it returns a `_LoopStep`
tagged as `"question"`, `"api_call"`, or `"fallback"`. `execute_streaming()` then
handles each case:

```
"question"  → iterate step.question_tokens (real DSPy tokens)
              → yield format_sse(chat_id, token) per token → yield END

"api_call"  → APICaller.call() [blocking HTTP]
              → async for token in APIResponseFormatterModule.stream_forward()
              → yield format_sse(chat_id, token) per token → yield END
```

---

## Part 3 — Agentic Loop (Multi-Turn Parameter Collection)

### Overview

Defined in [src/tool_classifier/agentic_loop.py](../src/tool_classifier/agentic_loop.py).

`AgenticLoop` is **stateless** — it carries no internal state between HTTP requests.
All state is passed in as arguments (loaded from Redis by the workflow executor before
calling `run_turn`) and saved back to Redis inside `run_turn` before returning.

### Session Model: `APIToolSession`

Defined in [src/models/session_models.py](../src/models/session_models.py).

Stored in Redis keyed by `chat_id` with a **30-minute sliding TTL**.

| Field | Type | Description |
|---|---|---|
| `chat_id` | str | Unique conversation identifier |
| `state` | str | Current state (`collecting_params`, etc.) |
| `selected_endpoint` | dict | Full endpoint payload from Qdrant |
| `collected_params` | dict | Parameters collected so far |
| `turn_count` | int | Number of turns elapsed |
| `max_turns` | int | Max turns before fallback (default: 5) |
| `awaiting_continuation` | bool | True when continuation prompt has been shown |
| `detected_language` | str | Language from first message (`en`, `et`, `ru`) — persisted so all clarifying questions use the same language |
| `original_query` | str | The user’s first message that triggered the session — preserved across turns so the response formatter always receives the full original intent, not just the last short follow-up (e.g. `"from 2026-04-01 to 2026-04-30"`) |

### Turn Flow

```
APIToolWorkflowExecutor._run()
    │
    ├─ Load session from Redis (or create new)
    │
    └─ AgenticLoop.run_turn(
           user_message, conversation_history,
           params_schema, collected_params,
           turn_count, max_turns, awaiting_continuation,
           session_language
       )
           │
           ├─ AWAITING_CONTINUATION_DECISION?
           │     yes → parse yes/no from user_message
           │             yes → clear flag, continue collecting
           │             no  → return MAX_TURNS_REACHED (RAG fallback)
           │
           ├─ ParamExtractionModule.forward()
           │     → DSPy extracts params from user_message + conversation_history
           │     → uses session_language for all questions
           │     → new values OVERWRITE old (allows corrections)
           │
           ├─ All required params present? → COMPLETED
           │
           ├─ turn_count reached CONTINUATION_TURN (default: 3)?
           │     → set awaiting_continuation=True
           │     → return AWAITING_CONTINUATION_DECISION
           │     → question = localized CONTINUATION_QUESTION (EN/ET/RU)
           │
           └─ else → generate clarifying question for next missing param
                   → return NEEDS_INPUT
           │
           └─ Save updated session to Redis
```

### Key Behaviours

**Language persistence:**
The language is detected once from the user's first message and stored in
`APIToolSession.detected_language`. All subsequent clarifying questions and the
continuation prompt are generated in that language, even when follow-up replies
like "yes" or "2026-01-01" are too short to re-detect reliably.

Supported: `en` (default), `et` (Estonian), `ru` (Russian).

**Parameter correction:**
If the user says "No, use Russia instead of Estonia", the extractor overwrites the
previously collected `countryIsoCode` value. There is no guard preventing
re-extraction of already-collected params — new values always win.

**Continuation prompt:**
After `CONTINUATION_TURN` turns without completing, the loop asks the user whether
to continue. If the user says no (or anything not in the yes-list), the session is
abandoned and the request falls back to the RAG workflow.

Localized continuation questions are defined in
[src/tool_classifier/constants.py](../src/tool_classifier/constants.py):
`CONTINUATION_QUESTION`, `CONTINUATION_QUESTION_ET`, `CONTINUATION_QUESTION_RU`.

**Constants** (in `src/tool_classifier/constants.py`):

| Constant | Value | Description |
|---|---|---|
| `CONTINUATION_TURN` | `3` | Turn at which the continuation prompt is shown |

---

## Part 4 — API Caller & Response Formatter

### Component: `APICaller`

Defined in [src/tool_classifier/api_caller.py](../src/tool_classifier/api_caller.py).

Executes the external HTTP request once all required parameters have been collected
by the agentic loop.

**Supported methods:** `GET` (params → query string) and `POST` (params → JSON body).

**Timeout:** `API_CALL_TIMEOUT` seconds (from `constants.py`). Overridable per-call.

**Return type:** `APICallResult`

| Field | Type | Description |
|---|---|---|
| `success` | bool | `True` for 2xx responses |
| `status_code` | int | HTTP status code; `0` for network/timeout/circuit-breaker failures |
| `response_data` | Any | Parsed JSON on success; raw parsed error body on 4xx; empty string on all other failures |
| `error` | str \| None | Localized user-facing error message on failure; `None` on success |

**Error handling:**

| Failure type | `status_code` | `response_data` | `error` field |
|---|---|---|---|
| 4xx (client error, e.g. bad params) | actual code | Raw parsed body (preserved for agentic loop re-prompting) | Localized `CLIENT_ERROR_MESSAGES` |
| 5xx (server error) | actual code | `""` | Localized `SERVICE_UNAVAILABLE_MESSAGES` |
| Timeout | `0` | `""` | Localized `SERVICE_TIMEOUT_MESSAGES` |
| Network error | `0` | `""` | Localized `SERVICE_TIMEOUT_MESSAGES` |
| Redirect not followed | `3xx` | `""` | Localized `REDIRECT_NOT_FOLLOWED_MESSAGES` |
| Circuit breaker open | `0` | `""` | Localized `CIRCUIT_BREAKER_OPEN_MESSAGES` |

4xx responses do **not** trip the circuit breaker — they indicate bad input, not a
server outage. The agentic loop can re-prompt the user for corrected values.

**Language-aware errors:** all error messages are localized using `session.detected_language`
(`et`, `en`, `ru`). The message constants are defined in
[src/tool_classifier/constants.py](../src/tool_classifier/constants.py).

---

### Component: `CircuitBreaker`

Part of `api_caller.py`. One breaker instance per URL, shared across requests for the
lifetime of the `APICaller` instance.

```
CLOSED  → OPEN:      after CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive server/network failures
OPEN    → HALF_OPEN: after CIRCUIT_BREAKER_COOLDOWN_SECONDS
HALF_OPEN → CLOSED:  on first successful probe call
HALF_OPEN → OPEN:    on first failed probe call
```

When OPEN, `call()` returns immediately without making an HTTP request.

**Constants** (in `src/tool_classifier/constants.py`):

| Constant | Description |
|---|---|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | Consecutive failures before opening |
| `CIRCUIT_BREAKER_COOLDOWN_SECONDS` | Seconds to wait before probing |

---

### Component: `APIResponseFormatterModule`

Defined in [src/tool_classifier/api_response_formatter.py](../src/tool_classifier/api_response_formatter.py).

Converts the raw API JSON response into a natural-language answer using DSPy.
Supports both blocking (`forward`) and streaming (`stream_forward`) execution.

**DSPy Signature:** `APIResponseFormatterSignature`

| Input field | Description |
|---|---|
| `user_query` | The user's original question |
| `api_response` | Raw API JSON as a string (truncated to `_MAX_RESPONSE_BYTES` = 50 KB) |
| `endpoint_description` | Short description of what the endpoint does |
| `response_language` | `"English"`, `"Estonian"`, or `"Russian"` — derived from `detected_language` |

| Output field | Description |
|---|---|
| `formatted_answer` | Clean natural-language answer, no raw JSON or markdown headers |



## Part 5 — Session Management & Intent Switch Detection

### `APIToolSessionStore`

Defined in [src/utils/api_tool_session_store.py](../src/utils/api_tool_session_store.py).

Redis-backed store. Key format: `session:{chat_id}`. TTL resets on every `update()`.

Operations: `save()`, `get()`, `update()`, `delete()`.

### Session Lifecycle

```
Turn 1: new query matches API tool endpoint
    → session CREATED (state=collecting_params)
    → clarifying question returned

Turn 2-N: user replies
    → session LOADED → loop runs → session UPDATED

Final turn: all params collected
    → session DELETED
    → completed JSON returned

OR: max turns reached / user says "no" to continuation
    → session DELETED
    → None returned → RAG fallback
```

### Intent Switch Detection

Defined in `ToolClassifier.classify()` — the session-resume short-circuit block.

Before resuming an active session, the classifier runs `_try_api_tool_classification()`
on the new message. If it matches a **different** endpoint with sufficient confidence,
the old session is abandoned and the new query starts fresh:

```python
new_api_match = await self._try_api_tool_classification(query, request)
if (
    new_api_match is not None
    and new_api_match.metadata["matched_endpoint"]["name"] != endpoint_name
):
    await session_store.delete(request.chatId)
    return new_api_match  # start new session for different endpoint
```

### Test Endpoint Behaviour (`/orchestrate/test`)

The test endpoint hardcodes `chatId="test-session"` for all requests. Because every
test user shares this ID, any incomplete session would be resumed by the next
unrelated test query.

**Fix:** the test endpoint deletes `"test-session"` from Redis at the **start** of
every request, before classification runs. This makes each test query a fresh
single-turn request.

**Trade-off:** multi-turn API tool flows cannot be tested via the test-LLM page.
The session is wiped before turn 2 can use it. To test multi-turn flows, use the
production `/orchestrate/stream` endpoint (which uses unique `chatId` per tab) or
the integration test script.

---

### End-to-End Flow (Query Time)

```
Turn 1 — User: "What are the public holidays in Estonia?"
    │
    ▼
ToolClassifier.classify()
    │
    ├─ No active session in Redis for this chat_id
    ├─ Dense search (intent_collections) → low cosine → below threshold
    └─ _try_api_tool_classification()
           └─ APISemanticSearcher.search()
                  ├─ Dense: get_public_holidays cosine=0.87 → high confidence
                  └─ return [APIToolSearchResult(name="get_public_holidays", ...)]
           └─ ClassificationResult(workflow=API_TOOL_CALLING, metadata={matched_endpoint: {...}})
    │
    ▼
APIToolWorkflowExecutor._run()
    ├─ No existing session → create new APIToolSession (turn_count=0, language=en, original_query="What are the public holidays in Estonia?")
    └─ AgenticLoop.run_turn(turn_count=0, history=[])
           ├─ ParamExtractionModule: no params in "What are the public holidays in Estonia?"
           │   but countryIsoCode=EE can be inferred → extracted
           ├─ Missing: validFrom, validTo
           └─ NEEDS_INPUT → "Which date range would you like? (validFrom, validTo)"
    │
    Session saved to Redis
    ▼
Bot: "Which date range would you like? Please provide validFrom and validTo (YYYY-MM-DD)."

---

Turn 2 — User: "This year, 2026-01-01 to 2026-12-31"
    │
    ▼
ToolClassifier.classify()
    ├─ Active session found for chat_id → run intent-switch check
    ├─ _try_api_tool_classification("This year, 2026-01-01 to 2026-12-31")
    │   → cosine=0.12 < threshold → no new API tool match
    └─ Same endpoint → resume session → ClassificationResult(reason=active_session_resume)
    │
    ▼
APIToolWorkflowExecutor._run()
    └─ AgenticLoop.run_turn(turn_count=1, collected_params={countryIsoCode: "EE"})
           ├─ ParamExtractionModule: extracts validFrom=2026-01-01, validTo=2026-12-31
           ├─ All required params present
           └─ COMPLETED
    │
    Session DELETED from Redis
    ▼
APIToolWorkflowExecutor._stream_api_and_format()
    ├─ user_query = session.original_query → "What are the public holidays in Estonia?"
    ├─ APICaller.call(GET https://openholidaysapi.org/PublicHolidays, params={countryIsoCode,validFrom,validTo})
    │   → status=200, response_data=[{"name": "New Year's Day", ...}, ...]
    └─ APIResponseFormatterModule.stream_forward(user_query, api_response, description, language="en")
           → DSPy StreamResponse tokens yielded one by one
           → format_sse(chat_id, "Here are the public holidays ") ...
           → format_sse(chat_id, "END")
    │
    ▼
Bot: "Here are the public holidays in Estonia for 2026:\n- New Year's Day (1 Jan)\n- ..."  ← streamed token-by-token
```

---

## Part 6 — Integration Testing

### Test Script

Defined in [tests/api_tool_eval/integration_test_agentic_loop.py](../tests/api_tool_eval/integration_test_agentic_loop.py).

Runs end-to-end against the live service at `http://localhost:8100` via `/orchestrate`.
Each scenario uses a unique `chatId` (UUID) so sessions are fully isolated.

```bash
uv run --no-project --with requests python tests/api_tool_eval/integration_test_agentic_loop.py \
  --no-fail-fast \
  --output tests/api_tool_eval/integration-results.json
```

### Covered Scenarios

| # | Scenario | Turns | What it validates |
|---|---|---|---|
| 1 | Single-turn complete | 1 | Vehicle tax with plate number in first message → immediate API call + formatted response |
| 2 | Multi-turn EN | 2 | Public holidays, country extracted turn 1, dates provided turn 2 → API call + formatted response |
| 3 | Multi-turn ET | 2 | School holidays in Estonian → language-aware classification + Estonian response |
| 4 | No-params fast path | 1 | Parliament votings endpoint has no required params → immediate API call without session |
| 5 | Address search | 2 | Two-turn address lookup |
| 6 | Electricity prices | 2 | `datetime` params across two turns |
| 7 | Session isolation | 2 | Two different chat IDs — no param leak between sessions |
| 8 | AWAITING_CONTINUATION → yes | 4+ | User says “yes” at continuation prompt → loop resumes → API call on completion |
| 9 | MAX_TURNS_REACHED | 5+ | User never provides params → falls back to RAG |

---

## Part 7 — Multi-Intent Handling

> **Status:** All phases complete. Phase 1 (intent detection + parallel search), Phase 2 (session model extension), Phase 3 (multi-endpoint agentic loop), Phase 4 (parallel API caller), Phase 5 (multi-response formatter), and Phase 6 (full wiring in workflow executor) are production-ready.

### Overview

A multi-intent query like *"What are the public holidays in Estonia and what is the current electricity price?"* produces a **diluted embedding** — the dense vector sits between two endpoints rather than close to either one. The cosine score lands in the ambiguous band (≥ `API_TOOL_MIN_THRESHOLD`, < `API_TOOL_HIGH_CONFIDENCE_THRESHOLD`) rather than producing a clean high-confidence hit.

Phase 1 adds a **score-band gate** that intercepts these ambiguous results and passes the raw query to `IntentDecomposer`. If two or more distinct intents are detected, sub-queries are searched in parallel and the results are stored in the session for eventual multi-endpoint execution.

---

### Phase 1: Score-Band Gate + Intent Decomposer

#### Gate Logic (`classifier.py` — `_try_api_tool_classification`)

```
cosine ≥ HIGH_CONFIDENCE_THRESHOLD           → single path, existing code unchanged
cosine in [MIN_THRESHOLD, HIGH_CONFIDENCE)   → ambiguous band → IntentDecomposer
cosine < MIN_THRESHOLD                       → no match → RAG fallback
```

The gate only fires when:
- `FeatureFlags.MULTI_INTENT_ENABLED = true` (env: `MULTI_INTENT_ENABLED`, default `true`)
- The top result was **not already LLM-validated** by the disambiguator (`llm_validated=False`)

The `llm_validated` flag on `APIToolSearchResult` prevents double LLM calls: when the disambiguator has already selected a winner it sets `llm_validated=True`, so the IntentDecomposer gate is skipped for that result.

#### Disambiguator Edge Case Fix

When `APISemanticSearcher` runs LLM disambiguation and the disambiguator **rejects all candidates** (returns `winner_id=None`):

- **Old behaviour:** return `[]` → classified as RAG/CONTEXT even when the query was multi-intent
- **New behaviour:** if there were multiple medium-confidence candidates, return the top cosine result *without* `llm_validated=True` so the IntentDecomposer gate can run

This is the key fix that allows multi-intent queries to reach `IntentDecomposer` instead of falling through to RAG.

#### `IntentDecomposerModule` (`src/tool_classifier/intent_decomposer.py`)

DSPy module — receives the **raw user query**

| Output | Description |
|---|---|
| `mode` | `"single"` or `"parallel"` |
| `sub_queries` | List of focused sub-queries when `mode=parallel`; empty for single |

**Conservative by design:** returns `"single"` on any failure or ambiguity — never forces a parallel path.

Run asynchronously via `asyncio.to_thread(self, user_query)` (calls `__call__`, not `forward`, to avoid a DSPy warning).

Sub-query count is capped at `MULTI_API_MAX_ENDPOINTS = 3`.

#### Parallel Sub-Query Search

When `mode=parallel`, each sub-query is independently searched against `api_tool_collection` using `asyncio.gather`:

```python
results = await asyncio.gather(
    *[self._api_searcher.search(q, ...) for q in sub_queries]
)
```

Each search generates its own focused embedding — no dilution from the combined query.

Results are **deduplicated by endpoint name** (a single endpoint matched by two sub-queries counts once). If fewer than 2 distinct endpoints are found after dedup, the parallel result is discarded and the classifier falls back to the single path.

#### `ExecutionMode` Enum (`src/tool_classifier/enums.py`)

```python
class ExecutionMode(str, Enum):
    SINGLE = "single"
    PARALLEL = "parallel"
```

Inherits from `str` so values serialize cleanly to JSON. Always compare against the enum member (e.g. `== ExecutionMode.PARALLEL`), not a string literal.

#### `ClassificationResult` metadata for parallel mode

| Key | Type | Description |
|---|---|---|
| `execution_mode` | `ExecutionMode` | `SINGLE` or `PARALLEL` |
| `matched_endpoint` | dict | Set for single mode |
| `matched_endpoints` | list[dict] | Set for parallel mode — all deduplicated endpoints |

---

### Phase 2: Session Model Extension

Defined in [src/models/session_models.py](../src/models/session_models.py).

#### `EndpointSessionState` (new model)

Tracks per-endpoint collection state within a parallel session:

| Field | Type | Default | Description |
|---|---|---|---|
| `endpoint` | dict | required | Full endpoint payload |
| `collected_params` | dict | `{}` | Params gathered for this endpoint so far |
| `completed` | bool | `False` | Flipped to `True` when all required params for this endpoint have been collected; API execution happens later, gated on `AgenticLoopStatus.COMPLETED` at the loop level |

#### Extended `APIToolSession` fields

Three new fields added — all optional with safe defaults so **existing Redis sessions deserialize without error**:

| Field | Type | Default | Description |
|---|---|---|---|
| `execution_mode` | str | `"single"` | `"single"` or `"parallel"` — drives which loop handles this session |
| `parallel_endpoints` | list[EndpointSessionState] | `[]` | One entry per matched endpoint; empty in single mode |
| `active_endpoint_index` | int | `0` | Reserved for future sequential endpoint-tracking logic; not currently read or written by the parallel workflow |

The existing single-mode fields (`selected_endpoint`, `collected_params`, `turn_count`, etc.) are **completely unchanged**. Single-mode sessions have `execution_mode="single"` and `parallel_endpoints=[]`.

#### Session creation in parallel mode (`api_tool_workflow.py`)

When `context["execution_mode"] == ExecutionMode.PARALLEL`, the workflow captures all matched endpoints from context and populates `parallel_endpoints`:

```python
APIToolSession(
    ...
    execution_mode="parallel",
    parallel_endpoints=[
        EndpointSessionState(endpoint=e) for e in all_matched
    ],
)
```



---

### Phase 3: Multi-Endpoint Agentic Loop

Defined in [src/tool_classifier/multi_agentic_loop.py](../src/tool_classifier/multi_agentic_loop.py).

`MultiEndpointAgenticLoop` replaces `AgenticLoop` for parallel sessions. It is stateless between HTTP requests — all mutable state is held in the `EndpointSessionState` list passed in on each call.

**Key behaviours:**

- **Merged schema:** All endpoint `params` schemas are merged and deduplicated by param name. A param shared across two endpoints is asked once and applied to both.
- **One question per turn:** The loop generates a single clarifying question covering the next highest-priority missing param across all endpoints.
- **Per-endpoint distribution:** After extraction, `_distribute_params()` copies each extracted value to every endpoint whose schema includes that param name.
- **Completion tracking:** An endpoint is marked `completed=True` in its `EndpointSessionState` once all its required params are present. The loop returns `COMPLETED` only when every endpoint is completed.
- **Turn limit:** Determined by `_compute_turn_limits(num_endpoints)`:
  - Multi-intent (`num_endpoints > 1`): fixed `MULTI_INTENT_MAX_TURNS` (6).
  - Single-intent (`num_endpoints == 1`): `min(3 × num_endpoints, MULTI_API_MAX_TURNS)` — scales with endpoint count, capped at 9.
- **Continuation threshold:** Also from `_compute_turn_limits`:
  - Multi-intent: fixed `MULTI_INTENT_CONTINUATION_TURN` (4).
  - Single-intent: `num_endpoints + 1`.

**`stream_run_turn` signature:**

```python
await multi_loop.stream_run_turn(
    chat_id=chat_id,
    user_message=request.message,
    conversation_history=conversation_history,
    endpoint_states=session.parallel_endpoints,  # list[EndpointSessionState]
    turn_count=session.turn_count,
    awaiting_continuation=session.awaiting_continuation,
    session_language=effective_session_language,
)
```

Returns `(AgenticLoopResult, list[str])` — the result and pre-tokenised question tokens.

---

### Phase 4: Parallel API Caller

Defined in [src/tool_classifier/multi_api_caller.py](../src/tool_classifier/multi_api_caller.py).

`MultiAPICaller` wraps `APICaller` and fires all endpoint calls concurrently via `asyncio.gather`.

**Key design decisions:**

- Reuses the shared `APICaller` instance so per-URL circuit breaker state is preserved across single and batch invocations.
- Expects a `"call_params"` key on each endpoint dict (distinct from the `"params"` schema list) to prevent the schema descriptor from being forwarded to the HTTP call.
- A `MULTI_API_BATCH_TIMEOUT` (30 s) caps total wall-clock time. Pending tasks are cancelled on timeout and replaced with failure results — the caller always receives a fully-populated `MultiAPICallResult`.
- Results are returned in the same order as the input endpoint list.
- Partial failure is safe — a failed endpoint produces `APICallResult(success=False, error=<localized message>)` without affecting other endpoints.

**Usage:**

```python
call_payloads = [
    {**state.endpoint, "call_params": state.collected_params}
    for state in parallel_endpoints
]
multi_result = await MultiAPICaller(api_caller).call_all(
    call_payloads, language=detected_language
)
```

---

### Phase 5: Multi-Response Formatter

Defined in [src/tool_classifier/multi_response_formatter.py](../src/tool_classifier/multi_response_formatter.py).

`MultiResponseFormatterModule` is a DSPy module that synthesises N API results into one unified natural-language answer.

**DSPy Signature:** `MultiResponseFormatterSignature`

| Input field | Description |
|---|---|
| `user_query` | The user's original first-turn question |
| `api_results_block` | Formatted block of all API results (name, description, data) |
| `num_results` | Number of results being synthesised |
| `response_language` | `"English"`, `"Estonian"`, or `"Russian"` |
| `custom_instructions` | Optional operator prompt overrides |

| Output field | Description |
|---|---|
| `unified_answer` | Single coherent natural-language answer covering all endpoints |

**Rules enforced by signature:**
- Always write in `response_language` regardless of API data language.
- Address every result — do not silently omit any endpoint.
- Gracefully acknowledge failed or empty results without dwelling on them.
- No raw JSON, no markdown headers, no follow-up invitation sentences.

**Max input size:** 100 KB across all results combined (`_MAX_TOTAL_RESPONSE_BYTES`).

**Methods:** `forward(user_query, api_results, detected_language)` (blocking) and `stream_forward_multi(user_query, api_results, detected_language)` (async token iterator).

---

### Phase 6: Full Wiring in `APIToolWorkflowExecutor`

Defined in [src/tool_classifier/workflows/api_tool_workflow.py](../src/tool_classifier/workflows/api_tool_workflow.py).

`_LoopStep` now has four possible `kind` values:

| `kind` | Meaning |
|---|---|
| `"api_call"` | Single endpoint; all params collected; call API and format |
| `"multi_api_call"` | Parallel endpoints; call all APIs concurrently and merge results |
| `"question"` | Agentic loop needs more input; return question to user |
| `"fallback"` | Nothing to do; caller falls back to RAG |

**Parallel fast-path:** When all matched endpoints have no required params, `_compute_loop_step` skips session creation and returns a `"multi_api_call"` step immediately.

**Streaming path (`_stream_multi_api_and_format`):**
1. Builds `call_params`-keyed payloads for each `EndpointSessionState`.
2. Calls `MultiAPICaller.call_all()` — all HTTP calls fire concurrently.
3. Collects tokens from `MultiResponseFormatterModule.stream_forward_multi()` into a buffer.
4. Runs output guardrails on the full buffered response before yielding any token to the client.
5. Yields tokens one-by-one via `format_sse`, then yields `format_sse(chat_id, "END")`.

**Blocking path (`_execute_multi_api_and_format`):**
Same API call steps, then `asyncio.to_thread(formatter.forward, ...)` for the synthesis step.

---

### Constants and Feature Flags

Defined in [src/tool_classifier/constants.py](../src/tool_classifier/constants.py) and [src/llm_orchestrator_config/feature_flags.py](../src/llm_orchestrator_config/feature_flags.py):

| Name | Value | Description |
|---|---|---|
| `MULTI_INTENT_ENABLED` | `true` (env override) | Feature flag — set `MULTI_INTENT_ENABLED=false` to disable the IntentDecomposer gate globally |
| `MULTI_API_MAX_ENDPOINTS` | `3` | Hard cap on parallel sub-queries per request |
| `MULTI_API_MAX_TURNS` | `9` | Absolute cap on turns for any parallel session (`min(3×N, 9)` per session) |
| `MULTI_API_BATCH_TIMEOUT` | `30` | Seconds before the parallel HTTP batch is cancelled and partial results returned |
| `API_TOOL_HIGH_CONFIDENCE_THRESHOLD` | `0.60` | Cosine score above which single-path is taken immediately |
| `API_TOOL_MIN_THRESHOLD` | `0.40` | Minimum score for any match (below → RAG) |
| `API_TOOL_INTENT_SWITCH_THRESHOLD` | `0.50` | Minimum cosine for the new match to trigger intent-switch detection |

---

### Multi-Intent End-to-End Flow

```
Turn 1 — User: "Can you find an address for me and also calculate my vehicle tax?"
    │
    ▼
APISemanticSearcher.search()
    → top result: search_address, cosine=0.54 (ambiguous band)
    → disambiguator rejects both candidates (multi-intent dilution)
    → returns top candidate WITHOUT llm_validated=True
    │
    ▼
_try_api_tool_classification() — gate fires
    → MULTI_INTENT_ENABLED=true AND not llm_validated
    → IntentDecomposer (DSPy, asyncio.to_thread)
          → mode=parallel
          → sub_queries=["address lookup and location search", "vehicle tax calculation"]
    │
    ▼
asyncio.gather(
    search("address lookup and location search") → search_address,       cosine=0.82
    search("vehicle tax calculation")           → get_vehicle_tax_info, cosine=0.79
)
    → 2 distinct endpoints after dedup → parallel path confirmed
    │
    ▼
ClassificationResult(
    workflow=API_TOOL_CALLING,
    metadata={
        execution_mode: ExecutionMode.PARALLEL,
        matched_endpoints: [search_address, get_vehicle_tax_info]
    }
)
    │
    ▼
APIToolWorkflowExecutor._compute_loop_step()
    → no existing session → create new:
        APIToolSession(
            execution_mode="parallel",
            selected_endpoint=search_address,        # first endpoint
            original_query="Can you find an address...",
            parallel_endpoints=[
                EndpointSessionState(endpoint=search_address,       collected_params={}),
                EndpointSessionState(endpoint=get_vehicle_tax_info, collected_params={}),
            ],
            turn_count=0,
            max_turns=6,                             # min(3×2, 9)
        )
    → MultiEndpointAgenticLoop.stream_run_turn(endpoint_states=[...], turn_count=0)
          → merged schema: {address, regNr, calculationYear}  (deduped across both endpoints)
          → nothing extracted from turn-1 message (intent query, not param values)
          → missing: [address, regNr, calculationYear]
          → NEEDS_INPUT → clarifying question
    │
    Session saved to Redis, _LoopStep(kind="question")
    ▼
Bot: "To help you, I need a few details: the address you'd like to look up,
      your vehicle registration number (regNr), and the calculation year for the vehicle tax."

───────────────────────────────────────────────────────────────

Turn 2 — User: "123ABC"
    │
    ▼
ToolClassifier.classify()
    → Active session found → intent-switch check
    → "123ABC" cosine < API_TOOL_INTENT_SWITCH_THRESHOLD → no switch
    → ClassificationResult(reason=active_session_resume)
    │
    ▼
MultiEndpointAgenticLoop.stream_run_turn(turn_count=1)
    → ParamExtractionModule extracts regNr="123ABC"
    → _distribute_params: regNr → get_vehicle_tax_info.collected_params
    → still missing: [address, calculationYear]
    → NEEDS_INPUT → clarifying question covers ALL remaining missing params
    ▼
Bot: "Got it! I still need two more things: the address you'd like to look up,
      and the calculation year for the vehicle tax."

───────────────────────────────────────────────────────────────

Turn 3 — User: "Viru tn 4, Tallinn and year 2026"
    │
    ▼
MultiEndpointAgenticLoop.stream_run_turn(turn_count=2)
    → ParamExtractionModule extracts address="Viru tn 4, Tallinn", calculationYear="2026"
    → _distribute_params:
          address         → search_address.collected_params       → completed=True
          calculationYear → get_vehicle_tax_info.collected_params
    → get_vehicle_tax_info now has [regNr, calculationYear] → completed=True
    → ALL endpoints completed → AgenticLoopStatus.COMPLETED
    │
    Session DELETED from Redis
    _LoopStep(kind="multi_api_call", parallel_endpoints=[...])
    ▼
APIToolWorkflowExecutor._stream_multi_api_and_format()
    │
    ├─ Build call_payloads:
    │     [{...search_address,       call_params: {address: "Viru tn 4, Tallinn"}},
    │      {...get_vehicle_tax_info, call_params: {regNr: "123ABC", calculationYear: "2026"}}]
    │
    ├─ MultiAPICaller.call_all(payloads, language="en")
    │     asyncio.gather(
    │         GET /address-search?address=Viru+tn+4%2C+Tallinn  → 200 OK, address JSON
    │         GET /vehicle-tax?regNr=123ABC&year=2026           → 200 OK, tax JSON
    │     )  # batch_timeout=30 s; 2/2 succeeded
    │
    ├─ MultiResponseFormatterModule.stream_forward_multi(
    │       user_query=session.original_query,      # full first-turn message
    │       api_results=[("search_address", ..., address_data),
    │                    ("get_vehicle_tax_info", ..., tax_data)],
    │       detected_language="en"
    │   )
    │     → DSPy streams unified answer tokens
    │     → buffer-first: collect all tokens
    │
    ├─ Output guardrails on full buffered response → passed
    │
    └─ yield format_sse(chat_id, token) per token → yield format_sse(chat_id, "END")
    ▼
Bot: "Here's what I found: The address Viru tn 4 is located in Tallinn city centre
      (full address: Viru tn 4, 10111 Tallinn). For vehicle 123ABC, the estimated
      vehicle tax for 2026 is €127.40."  ← streamed token-by-token
```

---




## Part 8 — ATC Response Cache

### Overview

The ATC Response Cache is a two-tier Redis cache that sits inside `_compute_loop_step()` in
[src/tool_classifier/workflows/api_tool_workflow.py](../src/tool_classifier/workflows/api_tool_workflow.py).
It is checked on every **new request** (no active session) before the agentic loop is created.

Goal: avoid redundant API calls and agentic loop turns when the user is repeating or
following up on a query that was already answered in the same conversation.

Gated by `FeatureFlags.ATC_RESPONSE_CACHE_ENABLED` (`ATC_RESPONSE_CACHE_ENABLED` env var, default `true`).
Setting it to `false` disables all cache reads and writes without touching any other ATC logic.

---

### Cache Architecture — Two Tiers

#### Tier 1 — L1 Exact Response Cache

```
Key:   atc:cache:{chat_id}:{api_name}:{param_hash}
Value: raw API response JSON (dict or list)
TTL:   per-endpoint cache_ttl_seconds  OR  ATC_CACHE_DEFAULT_TTL_SECONDS (30 min)
```

Answers the question: *Has this exact conversation called this exact endpoint with these exact params before?*

`param_hash` is a 16-character hex digest of the **normalised, sorted** param dict:
- String values are stripped of whitespace
- Purely numeric strings (`"2026"`) are cast to `int` before hashing
- All-alpha strings (enum-like, e.g. `"GET"`, `"EE"`) are lowercased
- Keys are sorted so order does not matter

This means `{year: "2026", country: "EE"}` and `{country: "ee", year: 2026}` produce
the **same hash** and hit the same cache entry.

#### Tier 2 — L2 Last Call Context

```
Key:   atc:last:{chat_id}
Value: JSON list[LastCallContext]
TTL:   ATC_LAST_CALL_TTL_SECONDS (30 min, sliding — reset on every write)
```

Answers the question: *What was the last API call made in this conversation?*

Stores a full `LastCallContext` per succeeded endpoint. Single-intent calls write a
one-element list; multi-intent parallel calls write one entry per succeeded endpoint.
The follow-up detector searches this list by `api_name` to find the relevant prior call.

---

### Data Model: `LastCallContext`

Defined in [src/models/session_models.py](../src/models/session_models.py).

| Field | Type | Description |
|---|---|---|
| `api_name` | str | Endpoint name (snake_case) that was called |
| `endpoint` | dict | Full endpoint payload from Qdrant (params schema, URL, method, etc.) |
| `collected_params` | dict | Parameter values that were passed to the API call |
| `raw_response` | Any | Parsed API JSON (dict or list) as returned by `APICaller` |
| `original_query` | str | User's first-turn query that triggered this API call |
| `timestamp` | float | Unix timestamp of the call (for staleness reference) |

---


### Cache Write Points

L1 and L2 are written **after** every successful API call, as a background
`asyncio.create_task` so they never delay the user-facing response:

**Single-intent (`_execute_api_and_format`)**
After `api_result.success == True` and before the formatter:
```
set_l1(chat_id, endpoint["name"], collected_params, response_data, ttl)
set_l2(chat_id, [LastCallContext(...)])
```

**Multi-intent (`_execute_multi_api_and_format` / `_stream_multi_api_and_format`)**
After `multi_result` is received, one write per succeeded endpoint:
```
for each (endpoint_state, result) where result.success and endpoint.cacheable:
    set_l1(chat_id, endpoint["name"], endpoint_state.collected_params, result.response_data, ttl)
    append LastCallContext to contexts_list
set_l2(chat_id, contexts_list)   ← one write for all endpoints
```

---

### Cache Read Logic in `_compute_loop_step`

The cache block runs only when:
- No active Redis session exists (fresh request, not mid-loop)
- Not a parallel multi-intent query (`not all_matched`)
- `endpoint.cacheable == True`
- `ATC_RESPONSE_CACHE_ENABLED == True`

```
New request → no session → endpoint resolved
       │
       ▼
  ── L1 check ──────────────────────────────────────────────────────────
  get_l1(chat_id, endpoint["name"], pre_extracted_params)
       │
       ├─ HIT  → _LoopStep(kind="cached_response", cache_source="L1")
       │          formatter receives cached raw response — no API call, no loop
       │
       └─ MISS → continue to L2
       │
  ── L2 check ──────────────────────────────────────────────────────────
  get_l2(chat_id) → find entry where api_name == endpoint["name"]
       │
       ├─ No match → fall through to normal agentic loop
       │
       └─ Match found → FollowUpDetectorModule (DSPy via asyncio.to_thread)
              │
              │  Inputs: user_query, previous_query, previous_params, params_schema
              │
              ├─ "response_question"
              │     → _LoopStep(kind="cached_response", cache_source="L2",
              │                  cached_raw_response=matching.raw_response)
              │       no API call; formatter answers from the previous response
              │
              ├─ "param_update"
              │     merged = {**matching.collected_params, **updated_params}
              │     missing = _missing_required_params(schema, merged)
              │
              │     missing == []
              │       ├─ hashes equal (params unchanged)
              │       │     → try L1 with matching.collected_params
              │       │       hit  → cached_response (L1)
              │       │       miss → cached_response (L2 raw_response)
              │       └─ hashes differ (genuinely new params)
              │             → _LoopStep(kind="api_call", collected_params=merged)
              │               API called directly — entire agentic loop skipped
              │
              │     missing != []
              │       → context["seeded_params"] = merged
              │         fall through to agentic loop — only asks for gaps
              │
              └─ "new_intent"
                    → ignore L2; fall through to normal agentic loop

  On any FollowUpDetectorModule exception → fall through to normal loop (fail-open)
```

---

### Component: `FollowUpDetectorModule`

Defined in [src/tool_classifier/follow_up_detector.py](../src/tool_classifier/follow_up_detector.py).

DSPy `Predict` module that classifies the relationship between the new user query
and the previous API call. Run via `asyncio.to_thread` to avoid blocking the event loop.

**Inputs:**

| Input | Description |
|---|---|
| `user_query` | The new user message |
| `previous_query` | The user's original question that triggered the last API call |
| `previous_params` | JSON of param values from the last call |
| `params_schema` | JSON of the endpoint's param schema |

**Output — three possible values for `follow_up_type`:**

| Value | Meaning | Action |
|---|---|---|
| `response_question` | User is asking about the data already returned | Pass L2 `raw_response` to formatter; no API call |
| `param_update` | User wants the same endpoint with different/additional params | Merge new params into previous; go to API directly if complete, else seed the loop |
| `new_intent` | Completely unrelated query | Ignore L2; run normal agentic loop from scratch |

**Security:** `updated_params` from the LLM is validated against the endpoint's param
schema — keys not in the schema are silently dropped to prevent injection.

**Fail-open:** any exception returns `{follow_up_type: "new_intent", updated_params: {}}`
so the user is never blocked.

---

### Param Seeding

When the L2 `param_update` path finds that merged params are still incomplete,
it sets `context["seeded_params"] = merged` before falling through to the agentic loop.

`AgenticLoop.run_turn()` and `stream_run_turn()` accept an optional `seeded_params` argument.
On turn 0, the seeds are merged into `collected_params` **before** any extraction runs:

```python
if turn_count == 0 and seeded_params:
    collected_params = {**seeded_params, **collected_params}
```

The merge order means existing `collected_params` win — seeds cannot overwrite values
that were already explicitly provided. The seeds are also stored directly in the new
Redis session (`APIToolSession.collected_params = seeded_params`) so they survive
across HTTP requests.

**Effect:** the agentic loop starts with inherited values already populated and only
generates a question for the genuinely missing params.

---

### L2 Invalidation on Intent Switch

When an intent switch is detected in `ToolClassifier.classify()` (user mid-session for
endpoint A sends a message that strongly matches endpoint B), both the session and the
L2 key are cleaned up:

```python
await session_store.delete(request.chatId)  # existing behaviour
if FeatureFlags.ATC_RESPONSE_CACHE_ENABLED:
    await ATCCacheStore().invalidate_l2(request.chatId)
```

`invalidate_l2` deletes only the `atc:last:{chat_id}` key. L1 keys are **not** deleted —
they are param-hash-scoped and expire on their own TTL. Deleting L1 would provide no
safety benefit and would waste valid cached data.

---

### Cache Constants and Feature Flag

Defined in [src/tool_classifier/constants.py](../src/tool_classifier/constants.py) and
[src/llm_orchestrator_config/feature_flags.py](../src/llm_orchestrator_config/feature_flags.py):

| Name | Value | Description |
|---|---|---|
| `ATC_CACHE_KEY_PREFIX` | `atc:cache` | Redis key prefix for L1 entries |
| `ATC_LAST_CALL_KEY_PREFIX` | `atc:last` | Redis key prefix for L2 entries |
| `ATC_CACHE_DEFAULT_TTL_SECONDS` | `1800` | Default L1 TTL (30 min); overridable per endpoint via `cache_ttl_seconds` |
| `ATC_LAST_CALL_TTL_SECONDS` | `1800` | L2 TTL (30 min, sliding) |
| `ATC_RESPONSE_CACHE_ENABLED` | `true` (env) | Master kill-switch — disables all reads and writes when `false` |

---

### Cache Component Reference

| Class / File | Responsibility |
|---|---|
| `ATCCacheStore` ([src/utils/atc_cache_store.py](../src/utils/atc_cache_store.py)) | All Redis operations for L1 and L2; param normalisation and hashing |
| `FollowUpDetectorModule` ([src/tool_classifier/follow_up_detector.py](../src/tool_classifier/follow_up_detector.py)) | DSPy classifier for follow-up type detection |
| `LastCallContext` ([src/models/session_models.py](../src/models/session_models.py)) | Pydantic model stored in L2 |
| `_compute_loop_step` ([src/tool_classifier/workflows/api_tool_workflow.py](../src/tool_classifier/workflows/api_tool_workflow.py)) | Where L1 + L2 are read and routing decisions are made |
| `_execute_api_and_format` / `_stream_api_and_format` | Where L1 + L2 are written after single-intent calls |
| `_execute_multi_api_and_format` / `_stream_multi_api_and_format` | Where L1 + L2 are written after parallel calls |
| `ToolClassifier.classify` ([src/tool_classifier/classifier.py](../src/tool_classifier/classifier.py)) | L2 invalidation on intent switch |

---

### End-to-End Cache Example

```
Turn 1 — "What are public holidays in Estonia in 2026?"
    Agentic loop collects {countryIsoCode:"EE", validFrom:"2026-01-01", validTo:"2026-12-31"}
    API called → 12 holidays returned
    L1 written: atc:cache:{id}:get_national_holidays:{hash({EE,2026-01-01,2026-12-31})}
    L2 written: atc:last:{id} = [LastCallContext{api_name="get_national_holidays", ...}]

Turn 2 — "Same for Latvia?"
    No session → L1 miss (country changed) → L2 hit
    FollowUpDetectorModule → param_update, updated_params={countryIsoCode:"LV"}
    merged = {countryIsoCode:"LV", validFrom:"2026-01-01", validTo:"2026-12-31"}
    no missing params + hashes differ → api_call step
    API called directly — zero agentic loop turns
    L1 + L2 updated with new result

Turn 3 — "Which of those is a bank holiday?"
    No session → L1 miss → L2 hit
    FollowUpDetectorModule → response_question
    Formatter receives Latvia raw_response from L2 → answers from cached data
    No API call, no loop

Turn 4 — "What is the weather in Tallinn?"
    Classifier: get_weather matched (different endpoint)
    Intent switch → session_store.delete + invalidate_l2
    Fresh session for get_weather starts with empty L1 and L2
```

---
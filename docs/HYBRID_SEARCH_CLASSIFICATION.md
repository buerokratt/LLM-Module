# Hybrid Search Classification & Intent Data Enrichment

> Updated architecture for the Tool Classifier using hybrid search (dense + sparse + RRF) with per-example indexing.  
> Replaces the single-embedding approach documented in `TOOL_CLASSIFIER_AND_SERVICE_WORKFLOW.md`.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Intent Data Enrichment (Indexing)](#intent-data-enrichment-indexing)
3. [Classification Flow (Query Time)](#classification-flow-query-time)
4. [Intent Detection & Entity Extraction](#intent-detection--entity-extraction)
5. [Thresholds & Configuration](#thresholds--configuration)

---

## Architecture Overview

The system has two phases:

1. **Indexing (offline):** For each service, create multiple Qdrant points with dense + sparse vectors
2. **Classification (query time):** Two-step search to route queries — dense for relevance, hybrid for service identification

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INDEXING (Offline)                               │
│                                                                     │
│  service_enrichment.sh → main_enrichment.py                        │
│    ├─ LLM context generation                                       │
│    ├─ Per-example: dense embedding + sparse BM25 vector             │
│    ├─ Summary: dense embedding + sparse BM25 vector                 │
│    └─ Qdrant upsert (N examples + 1 summary = N+1 points)         │
├─────────────────────────────────────────────────────────────────────┤
│                  CLASSIFICATION (Query Time)                        │
│                                                                     │
│  User Query                                                         │
│    ├─ Step 1: Dense search → cosine similarity (relevance check)   │
│    ├─ Step 2: Hybrid search → RRF fusion (service identification)  │
│    └─ Route: HIGH-CONFIDENCE / AMBIGUOUS / CONTEXT-RAG             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Intent Data Enrichment (Indexing)

### Source Files

| File | Role |
|------|------|
| `DSL/CronManager/script/service_enrichment.sh` | Entry point — sets environment, runs Python script |
| `src/intent_data_enrichment/main_enrichment.py` | Orchestrates per-example and summary point creation |
| `src/intent_data_enrichment/qdrant_manager.py` | Qdrant collection management, upsert, and deletion |
| `src/intent_data_enrichment/api_client.py` | LLM API calls (context generation, embeddings) |
| `src/intent_data_enrichment/models.py` | `ServiceData`, `EnrichedService`, `EnrichmentResult` data models |
| `src/intent_data_enrichment/constants.py` | `EnrichmentConstants` — API URLs, Qdrant config, vector sizes, LLM prompt template |
| `src/tool_classifier/sparse_encoder.py` | BM25-style sparse vector computation |

### What Changed: Single Embedding → Per-Example Indexing

**Before (old):** One point per service from concatenated text.

**After (new):** N+1 points per service — one per example query, plus one summary.

Example for a service with 3 examples:
```
Service "Valuutakursid" → 4 Qdrant points

  Point 0 (example): "Mis suhe on euro ja usd vahel"
    dense:  3072-dim embedding of this exact text
    sparse: BM25 vector → {euro: 1.0, usd: 1.0, suhe: 1.0, ...}

  Point 1 (example): "Mis on euro ja btc vahetuskurss?"
    dense:  3072-dim embedding of this exact text
    sparse: BM25 vector → {euro: 1.0, btc: 1.0, vahetuskurss: 1.0, ...}

  Point 2 (example): "euro ja gbp vaheline kurss"
    dense:  3072-dim embedding of this exact text
    sparse: BM25 vector → {euro: 1.0, gbp: 1.0, kurss: 1.0, ...}

  Point 3 (summary): "Service Name: Valuutakursid\nDescription: ...\nExample Queries: ...\nRequired Entities: ...\nEnriched Context: ..."
    dense:  3072-dim embedding of combined text
    sparse: BM25 vector of combined text
```

### Why Per-Example Indexing?

- Each example gets its own embedding, matching diverse user phrasings better
- Short example queries aren't diluted by long descriptions
- More examples = wider coverage "net" for query matching
- Sparse vectors enable keyword matching ("EUR", "USD") alongside semantic search

### Dense vs Sparse Vectors

| Type | Generation | Strength |
|------|-----------|----------|
| **Dense** (3072-dim) | `text-embedding-3-large` via Azure OpenAI | Semantic similarity — matches paraphrases, cross-language |
| **Sparse** (BM25) | Term frequency hashing (`sparse_encoder.py`) | Keyword overlap — exact token matching ("EUR", "USD", "THB") |

### Sparse Vector Generation

```python
# sparse_encoder.py
SPARSE_VOCAB_SIZE = 50_000

text = "Mis suhe on euro ja usd vahel"
tokens = re.findall(r"\w+", text.lower())  # ["mis", "suhe", "on", "euro", ...]
# Each token → MD5 hash (first 4 bytes) to index in [0, SPARSE_VOCAB_SIZE), value = term frequency
# Collisions are handled by summing values at the same index
# Output: SparseVector(indices=[hash("mis"), hash("euro"), ...], values=[1.0, 1.0, ...])
```

### Qdrant Collection Schema

```python
# Collection: "intent_collections"
vectors_config = {"dense": VectorParams(size=3072, distance=Distance.COSINE)}
sparse_vectors_config = {
    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
}
```

Each point payload:
```json
{
  "service_id": "common_service_exchange_rate",
  "name": "Valuutakursid",
  "description": "Kasutaja soovib infot valuutade kohta",
  "examples": ["Mis suhe on euro ja usd vahel", "..."],
  "entities": ["currency_from", "currency_to"],
  "context": "LLM-generated enriched context...",
  "point_type": "example",
  "example_text": "Mis suhe on euro ja usd vahel"
}
```

### Enrichment Pipeline Flow

```
service_enrichment.sh
  │
  ├─ Parse args: service_id, name, description, examples, entities
  │
  ├─ Step 1: LLM context generation (enriched description)
  │
  ├─ Step 2: For each example query:
  │    ├─ Generate dense embedding (text-embedding-3-large)
  │    └─ Generate sparse vector (BM25 term hashing)
  │
  ├─ Step 3: Summary point (name + description + examples + entities + LLM context):
  │    ├─ Generate dense embedding
  │    └─ Generate sparse vector
  │
  ├─ Step 4: Delete existing points for this service (idempotent)
  │
  └─ Step 5: Bulk upsert N+1 points to Qdrant
```

### Summary Point Combined Text Format

The summary point embeds a structured concatenation:
```
Service Name: {name}
Description: {description}
Example Queries: {example1} | {example2} | ...
Required Entities: {entity1}, {entity2}, ...
Enriched Context: {LLM-generated context}
```

### Service Deletion

When a service is deactivated, all its points are removed:
```python
qdrant_manager.delete_service_points(service_id)
# Uses payload filter: {"service_id": service_id}
```

---

## Classification Flow (Query Time)

### Source Files

| File | Role |
|------|------|
| `src/tool_classifier/classifier.py` | Two-step search + routing decisions |
| `src/tool_classifier/constants.py` | All thresholds and configuration |
| `src/tool_classifier/sparse_encoder.py` | Query sparse vector generation |
| `src/tool_classifier/workflows/service_workflow.py` | Service execution with 3 routing paths |

### Step 1: Dense Search — "Is This a Service Query?"

Queries Qdrant using only the dense vector to get **actual cosine similarity scores** (0.0 – 1.0).

```python
# classifier.py → _dense_search()
POST / collections / intent_collections / points / query
{
    "query": [0.023, -0.041, ...],  # 3072-dim dense vector
    "using": "dense",
    "limit": 6,  # DENSE_SEARCH_TOP_K * 2 (3 * 2 = 6, allows dedup)
    "with_payload": true,
}
```

Results are deduplicated by `service_id` (best score per service), returning up to `DENSE_SEARCH_TOP_K` (3) unique services.

**Why not use RRF scores?**  
Qdrant's RRF uses `1/(1+rank)`, producing fixed scores (0.50, 0.33, 0.25) regardless of actual relevance. A perfect match and a random query both get 0.50 for rank 1. Cosine similarity reflects true semantic closeness.

### Step 2: Hybrid Search — "Which Service?"

Only runs if cosine ≥ `DENSE_MIN_THRESHOLD`. Combines dense + sparse search with RRF fusion.
Sparse prefetch is only included if the query produces a non-empty sparse vector.

```python
# classifier.py → _hybrid_search()
# First checks collection exists and has data (points_count > 0)
POST / collections / intent_collections / points / query
{
    "prefetch": [
        {"query": dense_vector, "using": "dense", "limit": 10},
        {"query": {"indices": [...], "values": [...]}, "using": "sparse", "limit": 10},
    ],
    "query": {"fusion": "rrf"},
    "limit": 5,
    "with_payload": true,
}
```

> **Note:** Prefetch limit is `HYBRID_SEARCH_TOP_K * 2` (5 * 2 = 10). The sparse prefetch is conditionally added only when `sparse_vector.is_empty()` is False.

Hybrid results are also deduplicated by `service_id` (best RRF score per service).

### Routing Decision

```
Dense cosine score + gap
        │
        ├─ cosine < 0.38              → PATH 1: Skip SERVICE → CONTEXT/RAG
        │
        ├─ cosine ≥ 0.40 AND          → PATH 2: HIGH-CONFIDENCE SERVICE
        │  gap ≥ 0.05                     (skip discovery, intent detection on matched service only)
        │
        └─ else (0.38 ≤ cosine < 0.40 → PATH 3: AMBIGUOUS SERVICE
           OR gap < 0.05)                 (LLM intent detection on candidates)
```

### Path 1: Non-Service Query → CONTEXT/RAG

Top cosine score below minimum threshold. The query has no meaningful similarity to any indexed service.

```
Query: "Miks ID-kaart ei tööta e-teenustes?"
Dense: top cosine=0.29 → below 0.38 → skip SERVICE
→ Routes directly to CONTEXT → RAG (saves ~50-300ms by skipping hybrid search)
```

### Path 2: HIGH-CONFIDENCE Service Match

One service clearly stands out with high cosine and large gap to second result.

```
Query: "Palju saan 1 EUR eest THBdes?"
Dense: Valuutakursid (cosine=0.5511), gap=0.2371
→ 0.5511 ≥ 0.40 AND 0.2371 ≥ 0.05 → HIGH-CONFIDENCE
→ Skips service discovery
→ Runs intent detection + entity extraction on matched service only
→ Entities: {currency_from: EUR, currency_to: THB}
→ Validation: PASSED ✓
→ Calls service endpoint → Returns response
```

### Path 3: AMBIGUOUS Service Match → LLM Confirmation

Multiple services score similarly or cosine is in the medium range.

```
Query: "Mis on täna ilm?"
Dense: Ilmapäring (cosine=0.39), gap=0.03
→ 0.39 ≥ 0.38 but 0.39 < 0.40 → AMBIGUOUS
→ Runs LLM Intent Detection on top 3 candidates
→ LLM confirms or rejects → falls back to RAG if rejected
```

> **Note:** With the current threshold (0.38), the AMBIGUOUS zone (0.38–0.40) is intentionally narrow.
> Most queries resolve cleanly to either NON-SERVICE (<0.38) or HIGH-CONFIDENCE (≥0.40 with gap).

### Fallback Chain

Each workflow returns a response or `None` (fallback to next):

```
SERVICE (Layer 1)  →  CONTEXT (Layer 2)  →  RAG (Layer 3)  →  OOD (Layer 4)
```

---

## Intent Detection & Entity Extraction

### When Does It Run?

| Path | Intent Detection | Entity Extraction |
|------|-----------------|-------------------|
| HIGH-CONFIDENCE | On 1 service (matched) | Yes — from LLM output |
| AMBIGUOUS | On top candidates (from `top_results`) | Yes — if LLM matches |
| Non-service | Not run | Not run |

### Intent Detection Module (DSPy)

**File:** `src/tool_classifier/intent_detector.py`

The DSPy `IntentDetectionModule` uses `dspy.Predict` (direct prediction) and receives:
- User query
- Candidate services (formatted as JSON with service_id, name, description, required_entities, top 3 examples)
- Conversation history (last 3 turns, formatted as `{authorRole}: {message}`)

It returns:
```json
{
    "matched_service_id": "common_service_exchange_rate",
    "confidence": 0.92,
    "entities": {
        "currency_from": "EUR",
        "currency_to": "THB"
    },
    "reasoning": "User wants EUR to THB exchange rate"
}
```

### Entity Validation

**File:** `src/tool_classifier/workflows/service_workflow.py` → `_validate_entities()`

Extracted entities are validated against the service's schema:

```
Schema:    ["currency_from", "currency_to"]
Extracted: {"currency_from": "EUR", "currency_to": "THB"}
Result:    PASSED ✓
```

- **Missing entities** → sent as empty strings (service validates)
- **Extra entities** → ignored
- **Validation is lenient** — always proceeds, lets the service endpoint validate

### Entity Transformation

Entities dict → ordered array matching service schema:

```python
# Schema: ["currency_from", "currency_to"]
# Dict:   {"currency_from": "EUR", "currency_to": "THB"}
# Array:  ["EUR", "THB"]
```

### Service Endpoint Call

After entity validation and transformation, the workflow calls the Ruuter active service endpoint:

```python
# Endpoint: {RUUTER_SERVICE_BASE_URL}/services/active/{clean_service_name}
# Payload: {"chatId": "...", "authorId": "...", "input": ["EUR", "THB"]}
# Response: {"response": [{"content": "..."}]} → extracts content string
```

In streaming mode, the service content is wrapped as SSE events and streamed to the client.

---

## Thresholds & Configuration

All defined in `src/tool_classifier/constants.py`.

### Classification Thresholds

| Constant | Value | Description |
|----------|-------|-------------|
| `DENSE_MIN_THRESHOLD` | `0.38` | Minimum cosine to consider any service match. Below → skip SERVICE entirely. Empirically tuned: SERVICE queries score ≥ 0.49, RAG queries ≤ 0.35 — threshold sits in the 0.134 natural gap between the two distributions. |
| `DENSE_HIGH_CONFIDENCE_THRESHOLD` | `0.40` | Cosine for HIGH-CONFIDENCE path. Service queries with correct match score ≥ 0.49 (observed range: 0.49–1.00). Non-service score 0.27–0.35. |
| `DENSE_SCORE_GAP_THRESHOLD` | `0.05` | Required gap between top two services. Prevents false positives when multiple services score similarly. Service gaps: 0.15–0.75, non-service gaps: 0.001–0.029. |

### Search Configuration

| Constant | Value | Description |
|----------|-------|-------------|
| `DENSE_SEARCH_TOP_K` | `3` | Unique services from dense search |
| `HYBRID_SEARCH_TOP_K` | `5` | Results from hybrid RRF search |

### Observed Score Distributions

Based on empirical testing with 42 Estonian queries (20 SERVICE, 22 RAG):

| Metric | Service Query (n=20) | Non-Service / RAG Query (n=22) |
|--------|:--------------------:|:------------------------------:|
| Top cosine range | **0.49 – 1.00** | 0.27 – 0.35 |
| Top cosine mean | **0.77** | 0.30 |
| Cosine gap range | **0.15 – 0.75** | 0.001 – 0.029 |
| Cosine gap mean | **0.31** | 0.010 |
| Decision | HIGH-CONFIDENCE (100%) | NON-SERVICE (100%) |

> **Separation gap:** The lowest SERVICE cosine (0.49) and highest RAG cosine (0.35) are separated by **0.134** — a clean margin with no overlap. The threshold at 0.38 sits centrally in this gap.

### Performance by Path

| Path | Latency | LLM Calls | Cost |
|------|:-------:|:---------:|:----:|
| Non-service (below threshold) | ~50ms | 0 | $0 |
| HIGH-CONFIDENCE service | ~100ms | 1 | ~$0.002 |
| AMBIGUOUS service | ~3.5s | 1-2 | ~$0.002–0.004 |
| Legacy (no classifier) | ~4.0s | 2+ | ~$0.004+ |

> **Note:** Latencies above are classification time only (embedding + Qdrant search), excluding the downstream service call or RAG pipeline.

### Tuning Recommendations

- **Adding more services:** Score distributions improve naturally — service queries score higher, non-service score lower.
- **Adding more examples per service:** Diverse phrasings expand the embedding coverage. Aim for 5-8 examples per service covering formal + informal + different word orders.
- **Adjusting thresholds:** Monitor the logs (`Dense search: top=... cosine=...`) and adjust if real-world scores differ from test data.

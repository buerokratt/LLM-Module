# Tool Classifier and Service Workflow Architecture

## Overview

The Tool Classifier implements a **layer-wise fallback architecture** that routes user queries to the most appropriate workflow handler. The Service Workflow (Layer 1) handles external API/service calls with intelligent intent detection and entity extraction.

---

## Tool Classifier - Layer Architecture

### Design Pattern: Chain of Responsibility

The classifier tries each layer sequentially. If a layer returns `None`, it falls back to the next layer:

```
Layer 1: SERVICE  → External API calls (currency, weather, etc.)
Layer 2: CONTEXT  → Greetings, conversation history queries
Layer 3: RAG      → Knowledge base retrieval (documents, regulations)
Layer 4: OOD      → Out-of-domain fallback (polite rejection)
```

### Layer Execution Flow

```python
# Non-streaming mode
classification = await classifier.classify(query, history, language)
response = await classifier.route_to_workflow(
    classification, request, is_streaming=False
)

# Streaming mode
classification = await classifier.classify(query, history, language)
stream = await classifier.route_to_workflow(classification, request, is_streaming=True)
async for sse_chunk in stream:
    yield sse_chunk
```

### Fallback Chain

Each workflow's `execute_async()` or `execute_streaming()` can return:
- **OrchestrationResponse / AsyncIterator[str]**: Layer handled the query successfully
- **None**: Layer cannot handle → Fallback to next layer

Example cascading:
```
Query: "What is VAT rate?"
└─ SERVICE (Layer 1) → No matching service → Returns None
   └─ CONTEXT (Layer 2) → Not a greeting → Returns None
      └─ RAG (Layer 3) → Found in docs → Returns response ✓
```

---

## Service Workflow (Layer 1) - Detailed Architecture

### Purpose
Handle queries that require calling external services/APIs:
- Currency conversion: "How much is 100 EUR in USD?"
- Weather services: "What's the temperature in Tallinn?"
- Custom Ruuter endpoints: Any service registered in database

### High-Level Flow

The service workflow has **3 routing paths** based on classification metadata from hybrid search:

```
Classification Result (from classifier.py)
│
├─ needs_llm_confirmation = False (HIGH-CONFIDENCE)
│    → Skip discovery, run intent detection on matched service only
│
├─ needs_llm_confirmation = True (AMBIGUOUS)
│    → Run LLM intent detection on top candidate services
│
└─ No metadata (LEGACY / fallback)
     → Full service discovery + optional semantic search + intent detection
```

Each path then continues through:
```
1. Entity Extraction (from LLM output)
↓
2. Entity Validation (against service schema)
↓
3. Entity Transformation (Dict → Ordered Array)
↓
4. Service Endpoint Construction
↓
5. Service Call (Ruuter endpoint invocation)
```

---

## Service Discovery (Legacy Path)

### Method: `_call_service_discovery()`

Calls Ruuter public endpoint to fetch available services:

```python
GET {RAG_SEARCH_RUUTER_PUBLIC}/services/get-services
# Default: http://ruuter-public:8086/rag-search/services/get-services
```

**Response Structure:**
```json
{
  "response": {
    "service_count": 15,
    "use_semantic_search": true,
    "services": [
      {
        "serviceId": "currency_conversion_eur",
        "name": "Currency Conversion (EUR Base)",
        "description": "Convert EUR to other currencies",
        "ruuterType": "POST",
        "ruuterUrl": "/currency/convert",
        "entities": ["target_currency"],
        "examples": [
          "How much is 100 EUR in USD?",
          "Convert EUR to JPY"
        ]
      }
    ]
  }
}
```

### Service Count Threshold Logic

```python
SERVICE_COUNT_THRESHOLD = 10

if service_count <= 10:
    # Few services → Use all services for LLM intent detection
    services = response["services"]
    
elif service_count > 10:
    # Many services → Use semantic search to narrow down
    services = await _semantic_search_services(query, top_k=10)
```

---

## Semantic Search (When Many Services)

### Method: `_semantic_search_services()`

Uses Qdrant vector database to find relevant services:

```python
# 1. Generate embedding for user query
embedding = orchestration_service.create_embeddings_for_indexer([query])

# 2. Search Qdrant collection
search_payload = {
    "vector": query_embedding,
    "limit": 10,  # Top 10 services (SEMANTIC_SEARCH_TOP_K)
    "score_threshold": 0.2,  # Minimum similarity (SEMANTIC_SEARCH_THRESHOLD)
    "with_payload": True,
}

response = qdrant_client.post(
    f"/collections/{QDRANT_COLLECTION}/points/search", json=search_payload
)
```

**Returns:** Top-K most semantically relevant services for intent detection

---

## Intent Detection (LLM-Based)

### Method: `_detect_service_intent()`

Uses **DSPy + LLM** to intelligently match user query to a specific service and extract entities.

### DSPy Module: `IntentDetectionModule`

**Purpose:** Analyze user query against available services and extract structured information

**Signature:**
```python
class ServiceIntentDetector(dspy.Signature):
    # Inputs
    user_query: str  # "How much is 100 EUR in USD?"
    available_services: str  # JSON of service definitions
    conversation_context: str  # Recent 3 conversation turns

    # Output
    intent_result: str  # JSON: {matched_service_id, confidence, entities, reasoning}
```

### LLM Call Flow

```python
# 1. Prepare service context
services_formatted = [
    {
        "service_id": "currency_conversion_eur",
        "name": "Currency Conversion",
        "description": "Convert EUR to other currencies",
        "required_entities": ["target_currency"],
        "examples": ["How much is EUR in USD?", "Convert EUR to JPY"],  # Top 3 examples
    }
]

# 2. Prepare conversation context (last 3 turns)
conversation_context = """
end_user: Hello
backoffice_user: Hi! How can I help?
end_user: How much is 100 EUR in USD?
"""

# 3. Call DSPy module (uses dspy.Predict, not ChainOfThought)
with self.llm_manager.use_task_local():
    intent_result = intent_module.forward(
        user_query="How much is 100 EUR in USD?",
        services=services_formatted,
        conversation_history=conversation_history,
    )
```

### LLM Output Format

The LLM returns structured JSON:

```json
{
  "matched_service_id": "currency_conversion_eur",
  "confidence": 0.95,
  "entities": {
    "target_currency": "USD"
  },
  "reasoning": "User wants to convert EUR to USD, matches currency conversion service"
}
```

### Confidence Threshold

```python
if matched_service_id is None or confidence < 0.7:
    # Low confidence → Service workflow returns None → Fallback to Context/RAG
    return None
```

### Cost Tracking

Intent detection is an LLM call, so costs are tracked:

```python
# Before LLM call
history_length_before = len(dspy.settings.lm.history)

# Call intent detector
intent_result = intent_module.forward(...)

# After LLM call
usage_info = get_lm_usage_since(history_length_before)
costs_metric["intent_detection"] = usage_info

# Later: orchestration_service.log_costs(costs_metric)
```

---

## Entity Extraction

### From LLM Output

The LLM extracts entities directly from the user query:

**User Query:** `"Palju saan 1 EUR eest THBdes?"`  
(Estonian: "How much do I get for 1 EUR in THB?")

**LLM Extraction:**
```json
{
  "entities": {
    "target_currency": "THB"
  }
}
```

### Entity Format

Entities are extracted as **key-value pairs** where:
- **Key**: Entity name defined in service schema (`target_currency`)
- **Value**: Extracted value from user query (`"THB"`)

### Multi-Entity Example

**Service Schema:**
```json
{
  "serviceId": "weather_forecast",
  "entities": ["location", "date"]
}
```

**User Query:** "What's the weather in Tallinn tomorrow?"

**LLM Extraction:**
```json
{
  "entities": {
    "location": "Tallinn",
    "date": "tomorrow"
  }
}
```

---

## Entity Validation

### Method: `_validate_entities()`

Validates extracted entities against the service's expected schema.

### Validation Checks

#### 1. Missing Entities
Entities required by schema but not extracted by LLM:

```python
service_schema = ["target_currency", "amount"]
extracted = {"target_currency": "USD"}

# Missing: "amount"
missing_entities = ["amount"]
```

**Strategy:** Send empty string for missing entities (let service validate)

#### 2. Extra Entities
Entities extracted but not in service schema:

```python
service_schema = ["target_currency"]
extracted = {"target_currency": "USD", "random_field": "value"}

# Extra: "random_field"
extra_entities = ["random_field"]
```

**Strategy:** Ignore extra entities (not sent to service)

#### 3. Empty Values
Entities extracted but with empty values:

```python
extracted = {"target_currency": ""}

validation_errors = ["Entity 'target_currency' has empty value"]
```

**Strategy:** Log warning, proceed anyway (service validates)

### Validation Result

```python
{
    "is_valid": True,  # Always true (lenient validation)
    "missing_entities": ["amount"],  # Will send empty strings
    "extra_entities": ["random_field"],  # Will be ignored
    "validation_errors": [  # Warnings only
        "Entity 'amount' has empty value"
    ],
}
```

### Validation Philosophy

**Lenient Approach:**
- Always returns `is_valid: True`
- Proceeds with partial entities
- Service endpoint validates required parameters
- Avoids false negatives from over-strict validation

---

## Entity Transformation

### Method: `_transform_entities_to_array()`

Transforms entity dictionary to **ordered array** matching service schema order.

### Why Ordered Array?

Ruuter services expect parameters in specific order:
```python
# Service schema defines order
entities_schema = ["target_currency", "source_currency", "amount"]

# LLM extraction (unordered dict)
entities_dict = {"amount": "100", "target_currency": "USD", "source_currency": "EUR"}

# Transform to ordered array
entities_array = ["USD", "EUR", "100"]
#                  ↑      ↑      ↑
#                  [0]    [1]    [2]  (matches schema order)
```

### Transformation Logic

```python
def _transform_entities_to_array(
    self, entities_dict: Dict[str, str], entity_order: List[str]
) -> List[str]:
    """Transform entity dict to ordered array."""
    if not entity_order:
        return []
    return [entities_dict.get(key, "") for key in entity_order]
```

### Example

**Service Schema:**
```json
["target_currency", "base_currency", "amount"]
```

**Extracted Entities:**
```json
{
  "target_currency": "JPY",
  "amount": "500"
}
```

**Transformed Array:**
```python
["JPY", "", "500"]
#        ↑
#   Missing "base_currency" → empty string
```

---

## Service Call (Step 7 — Implemented)

### Endpoint Construction

```python
def _construct_service_endpoint(self, service_name: str, chat_id: str) -> str:
    # Clean service name: strip whitespace, remove invisible Unicode chars, replace spaces with _
    clean_name = service_name.strip().translate(INVISIBLE_CHAR_TABLE).replace(" ", "_")
    return f"{RUUTER_SERVICE_BASE_URL}/services/active/{clean_name}"
    # Example: "http://ruuter-public:8086/services/services/active/Currency_Conversion"
```

### Payload Construction

```python
payload = {
    "chatId": chat_id,
    "authorId": author_id,
    "input": entities_array,  # ["USD", "EUR", "100"]
}
```

### HTTP Call

```python
async def _call_service_endpoint(
    self, endpoint_url, http_method, entities_array, chat_id, author_id
) -> Optional[str]:
    async with httpx.AsyncClient(timeout=SERVICE_CALL_TIMEOUT) as client:
        if http_method.upper() == "POST":
            response = await client.post(endpoint_url, json=payload)
        else:
            response = await client.get(endpoint_url, params=payload)

        response.raise_for_status()
        data = response.json()

        # Ruuter wraps the DSL return value in {"response": ...}
        if isinstance(data, dict) and "response" in data:
            data = data["response"]

        # DMapper returns a JSON array; each item has a "content" field
        if isinstance(data, list) and len(data) > 0:
            content = data[0].get("content", "")
            return content if content else None
```

### Streaming Mode

In streaming mode, the service content is wrapped as SSE events:

```python
async def service_stream() -> AsyncIterator[str]:
    yield orchestration_service.format_sse(chat_id, service_content)
    yield orchestration_service.format_sse(chat_id, "END")
    orchestration_service.log_costs(costs_metric)
```

---

## Complete Example Flow

### User Query
```
"Palju saan 1 EUR eest THBdes?"
(How much do I get for 1 EUR in THB?)
```

### Step-by-Step Execution

#### 1. Classification (Hybrid Search)
```python
# Dense search finds best service match
# cosine=0.5511, gap=0.2371
# → HIGH-CONFIDENCE path (needs_llm_confirmation=False)
```

#### 2. Intent Detection (LLM Call on matched service only)
```json
{
  "matched_service_id": "currency_conversion_eur",
  "confidence": 0.92,
  "entities": {
    "target_currency": "THB"
  },
  "reasoning": "User wants to convert EUR to THB"
}
```

#### 3. Entity Extraction
```python
entities_dict = {"target_currency": "THB"}
```

#### 4. Entity Validation
```python
validation_result = {
    "is_valid": True,
    "missing_entities": [],
    "extra_entities": [],
    "validation_errors": [],
}
```

#### 5. Entity Transformation
```python
# Schema: ["target_currency"]
# Dict: {"target_currency": "THB"}
# Array: ["THB"]
entities_array = ["THB"]
```

#### 6. Service Call
```python
endpoint_url = "http://ruuter-public:8086/services/services/active/Currency_Conversion"
response = await _call_service_endpoint(
    endpoint_url=endpoint_url,
    http_method="POST",
    entities_array=["THB"],
    chat_id="...",
    author_id="...",
)
# Returns content string from Ruuter response
```

---

## Cost Tracking

Service workflow tracks LLM costs following the RAG workflow pattern:

```python
# Create costs dict at workflow level
costs_metric: Dict[str, Dict[str, Any]] = {}

# Intent detection captures costs
intent_result, intent_usage = await _detect_service_intent(...)
costs_metric["intent_detection"] = intent_usage

# Log costs after workflow completes
orchestration_service.log_costs(costs_metric)
```

**Cost Breakdown Logged:**
```
LLM USAGE COSTS BREAKDOWN:
  intent_detection    : $0.000120 (1 calls, 450 tokens)
```

---

## Fallback Behavior

### When Service Workflow Returns None

```python
# Scenario 1: No service_id in context after intent detection
if not context.get("service_id"):
    return None  # Fallback to CONTEXT layer

# Scenario 2: Service metadata extraction failed
if not service_metadata:
    return None  # Fallback to CONTEXT layer

# Scenario 3: Service endpoint call failed
if service_content is None:
    return None  # Fallback to CONTEXT layer
```

### Fallback Chain Result

```
Query: "What is VAT?"
└─ SERVICE → No service matches "VAT information" → None
   └─ CONTEXT → Not a greeting → None
      └─ RAG → Found in knowledge base → Response ✓
```

---

## Configuration Constants

```python
# Ruuter service configuration
RUUTER_BASE_URL = "http://ruuter-private:8086"
RUUTER_SERVICE_BASE_URL = "http://ruuter-public:8086/services"
RAG_SEARCH_RUUTER_PUBLIC = "http://ruuter-public:8086/rag-search"

# Service call timeouts
SERVICE_CALL_TIMEOUT = 10  # seconds for external service calls
SERVICE_DISCOVERY_TIMEOUT = 10.0  # seconds for service discovery

# Service selection thresholds
SERVICE_COUNT_THRESHOLD = 10  # Switch to semantic search if exceeded
MAX_SERVICES_FOR_LLM_CONTEXT = 50  # Max services to pass to LLM

# Semantic search
QDRANT_COLLECTION = "intent_collections"
SEMANTIC_SEARCH_TOP_K = 10  # Top 10 relevant services
SEMANTIC_SEARCH_THRESHOLD = 0.2  # Minimum similarity score
QDRANT_TIMEOUT = 10.0  # seconds

# Hybrid search classification (see HYBRID_SEARCH_CLASSIFICATION.md)
DENSE_MIN_THRESHOLD = 0.38  # Minimum cosine to consider service match
DENSE_HIGH_CONFIDENCE_THRESHOLD = 0.40  # Cosine for high-confidence path
DENSE_SCORE_GAP_THRESHOLD = 0.05  # Required gap between top two services
DENSE_SEARCH_TOP_K = 3  # Unique services from dense search
HYBRID_SEARCH_TOP_K = 5  # Results from hybrid RRF search
```

---

## Key Design Decisions

### 1. **Lenient Entity Validation**
- Proceeds with partial entities
- Service validates required parameters
- Reduces false negatives

### 2. **Ordered Entity Arrays**
- Ruuter services expect positional parameters
- Schema defines canonical order
- Missing entities → empty strings

### 3. **Three Routing Paths**
- **High-confidence**: Hybrid search matched → skip discovery, intent on 1 service
- **Ambiguous**: Moderate match → intent detection on top candidates
- **Legacy**: No classification metadata → full discovery flow

### 4. **LLM-Based Intent Detection**
- Uses DSPy `dspy.Predict` (not ChainOfThought) for direct prediction
- Intelligent service matching
- Natural language understanding
- Multilingual support (Estonian, English, Russian)

### 5. **Cost Tracking**
- Follows RAG workflow pattern
- Tracks intent detection LLM costs
- Integrated with budget system

### 6. **Implemented Service Call**
- Calls Ruuter active service endpoint via httpx
- Handles POST and GET methods
- Parses DMapper response format (`{"response": [{"content": "..."}]}`)
- Cleans service name (invisible chars, whitespace → underscore)

---

## Summary

The Tool Classifier's layer architecture enables intelligent query routing with graceful fallbacks. The Service Workflow (Layer 1) uses **hybrid search classification** (dense + sparse + RRF) to route queries into 3 paths: high-confidence (skip discovery), ambiguous (LLM confirmation on candidates), or legacy (full discovery). It then uses **LLM-based intent detection** (DSPy Predict) to match user queries to external services, extract entities, validate them against service schemas, transform to ordered arrays, and **call the Ruuter active service endpoint** — all while maintaining comprehensive cost tracking and seamless integration with the broader RAG pipeline.

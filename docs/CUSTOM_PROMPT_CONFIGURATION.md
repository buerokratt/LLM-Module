# Custom Prompt Configuration Flow

## Overview

The custom prompt configuration system allows admins to configure a single organisation-level prompt via the UI that automatically applies to user-facing answer generation. Changes are cached with a 5-minute TTL and can be immediately refreshed when updated.

The same configured prompt is consumed by **two** workflows — the **RAG workflow** and the **API Tool Calling workflow** — through one shared `PromptConfigurationLoader`. The **Context workflow** does **not** apply custom prompts (greetings use static templates and history answers use their own signature). See [Where Custom Prompts Are Applied (by Workflow)](#where-custom-prompts-are-applied-by-workflow).

---

## Where Custom Prompts Are Applied (by Workflow)

All consumers read the same prompt from the shared `PromptConfigurationLoader`
(`src/utils/prompt_config_loader.py`, 5-minute TTL cache). They differ in **how** they inject it.

| Workflow | Custom prompt applied? | Where / how |
|---|---|---|
| **RAG** |  Yes | `ResponseGeneratorAgent` — the prompt is wrapped as `[SYSTEM INSTRUCTIONS]…[USER QUESTION]` and appended to the question for both streaming and non-streaming generation. |
| **API Tool Calling** |  Yes | `APIToolWorkflowExecutor._get_custom_instructions()` loads the raw prompt and passes it into parameter extraction and response formatting (see below). |
| **Context** |  No | `context_workflow.py` / `context_analyzer.py` do not load or apply the custom prompt. Greetings return static templates; history answers use `ContextResponseGenerationSignature` without injection. |
| **Service** |  No | The response is pre-formed text from Ruuter/DMapper — there is no LLM generation step to steer. |
| **OOD** |  No | Fixed localized out-of-scope message. |

### RAG workflow

- Source: [`src/llm_orchestration_service.py`](../src/llm_orchestration_service.py) → `_get_custom_instructions_for_response_generation()` builds the prefix
  `"[SYSTEM INSTRUCTIONS]\n{prompt}\n\n[USER QUESTION]\n"` and passes it as
  `ResponseGeneratorAgent(custom_instructions_prefix=…)`.
- Application: [`src/response_generator/response_generate.py`](../src/response_generator/response_generate.py) applies it as
  `augmented_question = f"{question}\n\n{custom_instructions_prefix}"` in both `forward()` and
  `stream_response()`.
- **Note:** despite the name `custom_instructions_prefix`, the string is **appended after** the
  question (the wrapper text itself carries the `[USER QUESTION]` marker). It is **not** applied to
  `PromptRefinerAgent`, which only optimises the query for retrieval.

### API Tool Calling workflow

- Source: [`src/tool_classifier/workflows/api_tool_workflow.py`](../src/tool_classifier/workflows/api_tool_workflow.py) → `_get_custom_instructions()` reads the **same** `prompt_config_loader`
  (via `asyncio.to_thread`, fail-open to `""`). Unlike the RAG path, it passes the **raw** prompt
  (no `[SYSTEM INSTRUCTIONS]` wrapper).
- It is injected as a dedicated DSPy `custom_instructions` input field into:
  - `ParamExtractionModule` ([`param_extractor.py`](../src/tool_classifier/param_extractor.py)) — steers how parameters are extracted from the user.
  - `APIResponseFormatterModule` ([`api_response_formatter.py`](../src/tool_classifier/api_response_formatter.py)) — single-endpoint natural-language answer.
  - `MultiResponseFormatterModule` ([`multi_response_formatter.py`](../src/tool_classifier/multi_response_formatter.py)) — multi-endpoint synthesis.
- In the formatter/extractor signatures, a non-empty `custom_instructions` is followed with
  **HIGHEST PRIORITY**, overriding defaults such as language policy, tone, and formatting.
- Additionally, the workflow derives the response language from the prompt via
  `_language_from_custom_instructions()` and merges any Redis conversation summary into the same
  `custom_instructions` string before extraction.

---

## Architecture Components

### 1. **Database Layer**
- **Table**: `rag_search.prompt_configuration`
- **Columns**: `id` (BIGINT), `prompt` (TEXT)
- Stores the custom prompt text configured by admins

### 2. **Ruuter DSL Endpoints**
- **Get Prompt**: `DSL/Ruuter.public/rag-search/POST/llm-connections/prompts/get-prompt.yml`
  - Fetches prompt from database via Resql
  - Returns prompt data or empty object

- **Save Prompt**: `DSL/Ruuter.private/rag-search/POST/prompt-configuration/save.yml`
  - Updates/inserts prompt in database
  - Automatically triggers cache refresh after save

### 3. **Python Components**
- **PromptConfigurationLoader** (`src/utils/prompt_config_loader.py`)
  - HTTP client to fetch prompts via Ruuter
  - 5-minute TTL cache with thread safety
  - Retry logic (3 attempts, exponential backoff)
  - Force refresh capability

- **LLMOrchestrationService** (`src/llm_orchestration_service.py`)
  - Initializes loader at startup
  - Formats custom instructions with wrapper tags
  - Passes to ResponseGeneratorAgent

- **ResponseGeneratorAgent** (`src/response_generator/response_generate.py`)
  - Accepts `custom_instructions_prefix` parameter
  - Appends custom instructions after the user question
  - Applied in both streaming and non-streaming modes

### 4. **API Endpoints**
- **`POST /orchestrate`** - Standard request flow
- **`POST /orchestrate/test`** - Test request flow
- **`POST /orchestrate/stream`** - Streaming request flow
- **`POST /prompt-config/refresh`** - Force cache refresh

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ ADMIN UPDATES PROMPT IN UI                                      │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Ruuter: save.yml                                                │
│  1. Update/Insert in PostgreSQL                                 │
│  2. Call POST /prompt-config/refresh                            │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI: /prompt-config/refresh                                 │
│  - PromptConfigurationLoader.force_refresh()                    │
│  - Invalidates cache immediately                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Cache Updated - Ready for Next Request                          │
└─────────────────────────────────────────────────────────────────┘

╔═════════════════════════════════════════════════════════════════╗
║ USER SENDS MESSAGE                                              ║
╚════════════════┬════════════════════════════════════════════════╝
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ FastAPI: /orchestrate, /orchestrate/test, or /orchestrate/stream│
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ LLMOrchestrationService.process_orchestration_request()         │
│ or stream_orchestration_response()                              │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ _initialize_service_components()                                │
│  ↓                                                               │
│ _safe_initialize_response_generator()                           │
│  ↓                                                               │
│ _initialize_response_generator()                                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ _get_custom_instructions_for_response_generation()              │
│  ↓                                                               │
│ prompt_config_loader.get_custom_instructions()                  │
│  - Returns from cache if valid (< 5 min old)                    │
│  - OR fetches via Ruuter if expired                             │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Format custom instructions:                                     │
│ "[SYSTEM INSTRUCTIONS]\n{prompt}\n\n[USER QUESTION]\n"          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ResponseGeneratorAgent(custom_instructions_prefix=prefix)       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ ResponseGeneratorAgent.forward() or stream_response()          │
│  - Appends custom_instructions_prefix after user question       │
│  - Modified question = "{user_question}{prefix}"                │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ DSPy Predictor receives modified question                       │
│  - Custom instructions guide response generation                │
│  - LLM follows configured rules (language, tone, format, etc.)  │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ Response returned to user                                       │
│  - Follows custom prompt configuration                          │
│  - Language policy applied                                      │
│  - Formatting rules applied                                     │
│  - Safety guidelines applied                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Step-by-Step Flow

### **Startup Phase**
1. **Service Initialization** (`LLMOrchestrationService.__init__`)
   - Creates `PromptConfigurationLoader` instance
   - Warms up cache by calling `get_custom_instructions()`
   - Logs success: "Custom prompt configuration loaded at startup (X chars)"
   - Logs if not found: "ℹNo custom prompt configuration found - using defaults"

### **Admin Updates Prompt**
1. **UI Save Action**
   - Admin edits prompt text in UI
   - Submits save request

2. **Ruuter Processing** (`save.yml`)
   - Checks if prompt exists in database
   - Updates existing or inserts new prompt
   - Calls `POST /prompt-config/refresh` endpoint

3. **Cache Invalidation** (`/prompt-config/refresh`)
   - `force_refresh()` clears cache immediately
   - Fetches new prompt from Ruuter
   - Returns success status with prompt length and content hash (no preview for security)

### **User Request Processing**
1. **Request Received** (Any of 3 endpoints)
   - `/orchestrate/test` - Test Sresponse
   - `/orchestrate/stream` - Streaming response

2. **Service Components Initialization**
   - LLM Manager initialized
   - Contextual Retriever initialized
   - **Response Generator initialized** ← Custom prompt applied here

3. **Custom Instructions Loading**
   ```python
   custom_prefix = self._get_custom_instructions_for_response_generation()
   # Returns: "[SYSTEM INSTRUCTIONS]\n{prompt}\n\n[USER QUESTION]\n"
   ```

4. **Response Generator Creation**
   ```python
   ResponseGeneratorAgent(custom_instructions_prefix=custom_prefix)
   ```

5. **Question Modification**
   ```python
   # In forward() or stream_response()
   modified_question = f"{user_question}{custom_instructions_prefix}"
   ```

6. **LLM Processing**
   - DSPy predictor receives modified question
   - Custom instructions guide response behavior
   - Response generated following configured rules

---

## Cache Behavior

### **TTL Cache (5 minutes)**
- **Cache Hit**: Returns immediately from memory (fast)
- **Cache Miss**: Fetches via HTTP from Ruuter (slower, ~100-500ms)
- **Stale Fallback**: If fetch fails, returns last known good value

### **Force Refresh**
- Triggered by admin save action
- Bypasses cache TTL
- Ensures immediate propagation of changes

### **Thread Safety**
- Uses `threading.Lock()` for concurrent requests
- Single fetch for multiple simultaneous requests
- Cache shared across all requests

---

## Configuration

### **Constants** (`src/llm_orchestrator_config/llm_ochestrator_constants.py`)
```python
RUUTER_PROMPT_CONFIG_ENDPOINT = (
    "http://ruuter-public:8086/rag-search/llm-connections/prompts/get-prompt"
)
PROMPT_CONFIG_CACHE_TTL = 300  # 5 minutes cache
```

### **Environment Variables** (`constants.ini`)
```ini
RAG_SEARCH_PROMPT_REFRESH=http://llm-orchestration-service:8100/prompt-config/refresh
```

---

## Testing

### **1. Insert Test Prompt**
```sql
INSERT INTO rag_search.prompt_configuration (id, prompt)
VALUES (1, 'Always respond in Estonian language. Be professional and concise.')
ON CONFLICT (id) DO UPDATE SET prompt = EXCLUDED.prompt;
```

### **2. Test via API**
```bash
curl -X POST http://localhost:8100/orchestrate/test \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is artificial intelligence?",
    "environment": "development",
    "connectionId": 1
  }'
```

### **3. Update Prompt**
```sql
UPDATE rag_search.prompt_configuration
SET prompt = 'Provide concise answers using bullet points. Be helpful and clear.'
WHERE id = 1;
```

### **4. Verify Immediate Refresh**
- Check logs for: "Prompt configuration cache refreshed successfully"
- Test same question - response format should change immediately

### **5. Check Cache Status**
```bash
# Manual refresh (optional)
curl -X POST http://localhost:8100/prompt-config/refresh
```

**Response:**
```json
{
  "refreshed": true,
  "message": "Prompt configuration refreshed successfully",
  "prompt_length": 245,
  "content_hash": "a3f5b8c9e1d2f4a6"
}
```
**Note:** For security, the endpoint returns only the prompt length and a SHA-256 hash (not the actual prompt content).

---

## Key Features

 **TTL Caching** - 5-minute cache reduces database calls  
 **Immediate Updates** - Admin changes trigger instant refresh  
 **Graceful Degradation** - If refresh fails, TTL cache continues working  
 **Thread-Safe** - Multiple concurrent requests handled safely  
 **Retry Logic** - 3 attempts with exponential backoff for HTTP failures  
 **Instruction Appending** - Custom instructions appended to the question without modifying the DSPy signature  
 **Applied Consistently** - Works across all 3 orchestration endpoints  
 **Applied to RAG & API Tool Calling** - In RAG, only the ResponseGenerator (not the PromptRefiner); in API Tool Calling, the param extractor and response formatters. Not applied to the Context workflow.

---

## Example

**Database Prompt:**
```
Always respond in Estonian language. Be professional and concise. 
When answering, prioritize accuracy and cite sources when available.
```

**What DSPy Receives:**
```
[SYSTEM INSTRUCTIONS]
Always respond in Estonian language. Be professional and concise. 
When answering, prioritize accuracy and cite sources when available.

[USER QUESTION]
What is DigiDoc and how can I use it?

Context: [retrieved documentation chunks...]
```

**Expected Response:**
- In Estonian language 
- Professional tone 
- Concise format 
- Citations included 

---

## Files Modified

| File | Purpose |
|------|---------|
| `src/utils/prompt_config_loader.py` | HTTP loader with caching and retry |
| `src/llm_orchestration_service.py` | Initialize loader, format instructions |
| `src/llm_orchestration_service_api.py` | Refresh endpoint |
| `src/response_generator/response_generate.py` | Accept and apply custom prefix |
| `DSL/Ruuter.public/rag-search/POST/llm-connections/prompts/get-prompt.yml` | Fetch prompt endpoint |
| `DSL/Ruuter.private/rag-search/POST/prompt-configuration/save.yml` | Save with refresh trigger |
| `src/llm_orchestrator_config/llm_ochestrator_constants.py` | Configuration constants |
| `constants.ini` | Refresh endpoint URL |

---

## Troubleshooting

### **Prompt Not Applied**
- Check logs for: "Custom prompt configuration loaded at startup"
- Verify database has prompt: `SELECT * FROM rag_search.prompt_configuration;`
- Test refresh endpoint: `curl -X POST http://localhost:8100/prompt-config/refresh`

### **Cache Not Refreshing**
- Check Ruuter save.yml calls refresh endpoint
- Verify `RAG_SEARCH_PROMPT_REFRESH` constant in constants.ini
- Check logs for refresh success/failure

### **Empty Prompt**
- Check Ruuter endpoint returns correct format
- Verify response unwrapping logic in loader
- Check logs for "No prompt configuration found in database; caching empty result"

---

## Notes

- In the **RAG** path, custom prompts apply **only to `ResponseGeneratorAgent`** (not `PromptRefinerAgent`).
  The wrapped instructions are appended to the question rather than modifying the DSPy signature.
- The **API Tool Calling** workflow consumes the **same** configured prompt (via the shared loader) and
  injects the raw text as a dedicated `custom_instructions` DSPy input field on the parameter extractor
  and the response formatters, where it is followed with highest priority.
- The **Context** and **Service** workflows do not apply custom prompts (see the per-workflow section).

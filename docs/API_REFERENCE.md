# API Reference

This document is the consolidated HTTP API reference for the LLM Module. It covers the **LLM
Connections** management endpoints, the **Inference Results** storage/retrieval endpoints, and the
chatbot **Inquiry** endpoint exposed to the LLM Orchestration Service.

> Routing note: the public-facing paths below are served through the Ruuter API gateway
> (`ruuter-private` / `ruuter-public`), which proxies to the LLM Orchestration Service. See
> [ARCHITECTURE.md](./ARCHITECTURE.md) for how requests flow through the system.

## Contents

- [LLM Connections API](#llm-connections-api-endpoints)
- [Inference Results API](#inference-results-api-endpoints)

---

## LLM Connections API Endpoints

### Base URL
```
/ruuter-private/llm/connections
```

---

## 1. Create LLM Connection

### Endpoint
```http
POST /ruuter-private/llm/connections/create
```

### Request Body
```json
{
  "llmPlatform": "OpenAI",
  "llmModel": "GPT-4o",
  "embeddingPlatform": "OpenAI",
  "embeddingModel": "text-embedding-3-small",
  "monthlyBudget": 1000.00,
  "deploymentEnvironment": "Testing",
  // Azure credentials (optional)
  "deploymentName": "my-deployment",
  "targetUri": "https://my-endpoint.azure.com",
  "apiKey": "azure-api-key",
  // AWS Bedrock credentials (optional)
  "secretKey": "aws-secret-key",
  "accessKey": "aws-access-key",
  // Embedding model credentials (optional)
  "embeddingModelApiKey": "embedding-api-key"
}
```

### Response (201 Created)
```json
{
  "id": 1,
  "llmPlatform": "OpenAI",
  "llmModel": "GPT-4o",
  "embeddingPlatform": "OpenAI",
  "embeddingModel": "text-embedding-3-small",
  "monthlyBudget": 1000.00,
  "usedBudget": 0.00,
  "deploymentEnvironment": "Testing",
  "status": "active",
  "createdAt": "2025-09-02T10:15:30.000Z",
  // Azure credentials (if provided)
  "deploymentName": "my-deployment",
  "targetUri": "https://my-endpoint.azure.com",
  "apiKey": "azure-api-key",
  // AWS Bedrock credentials (if provided)  
  "secretKey": "aws-secret-key",
  "accessKey": "aws-access-key",
  // Embedding model credentials (if provided)
  "embeddingModelApiKey": "embedding-api-key"
}
```

---

## 2. Update LLM Connection

### Endpoint
```http
POST /ruuter-private/llm/connections/update
```

### Request Body
```json
{
  "connectionId": 1,
  "llmPlatform": "Azure AI",
  "llmModel": "GPT-4o-mini",
  "embeddingPlatform": "Azure AI", 
  "embeddingModel": "text-embedding-ada-002",
  "monthlyBudget": 2000.00,
  "deploymentEnvironment": "Production",
  // Azure credentials (optional)
  "deploymentName": "updated-deployment",
  "targetUri": "https://updated-endpoint.azure.com",
  "apiKey": "updated-azure-api-key",
  // AWS Bedrock credentials (optional)
  "secretKey": "updated-aws-secret-key",
  "accessKey": "updated-aws-access-key",
  // Embedding model credentials (optional)
  "embeddingModelApiKey": "updated-embedding-api-key"
}
```

### Response (200 OK)
```json
{
  "id": 1,
  "llmPlatform": "Azure AI",
  "llmModel": "GPT-4o-mini",
  "embeddingPlatform": "Azure AI",
  "embeddingModel": "text-embedding-ada-002",
  "monthlyBudget": 2000.00,
  "usedBudget": 150.75,
  "deploymentEnvironment": "Production",
  "status": "active",
  "createdAt": "2025-09-02T10:15:30.000Z",
  // Azure credentials (if provided)
  "deploymentName": "updated-deployment",
  "targetUri": "https://updated-endpoint.azure.com",
  "apiKey": "updated-azure-api-key",
  // AWS Bedrock credentials (if provided)  
  "secretKey": "updated-aws-secret-key",
  "accessKey": "updated-aws-access-key",
  // Embedding model credentials (if provided)
  "embeddingModelApiKey": "updated-embedding-api-key"
}
```

---

## 3. Get LLM Connections (Paginated List)

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/list
```

### Request Body
```json
{
  "page": 1,
  "page_size": 10,
  "sorting": "created_at desc"
}
```

### Request Parameters
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `page` | number | No | Page number (1-based) | 1 |
| `page_size` | number | No | Number of items per page | 10 |
| `sorting` | string | No | Sorting criteria | "created_at desc" |

### Sorting Options
- `llm_platform asc/desc`
- `llm_model asc/desc`
- `embedding_platform asc/desc`
- `embedding_model asc/desc`
- `monthly_budget asc/desc`
- `environment asc/desc`
- `status asc/desc`
- `created_at asc/desc`
- `updated_at asc/desc`

### Response (200 OK)
```json
[
  {
    "id": 1,
    "llmPlatform": "OpenAI",
    "llmModel": "GPT-4o",
    "embeddingPlatform": "OpenAI",
    "embeddingModel": "text-embedding-3-small",
    "monthlyBudget": 1000.00,
    "environment": "Testing",
    "status": "active",
    "createdAt": "2025-09-02T10:15:30.000Z",
    "updatedAt": "2025-09-02T10:15:30.000Z",
    "totalPages": 3
  },
  {
    "id": 2,
    "llmPlatform": "Azure AI",
    "llmModel": "GPT-4o-mini",
    "embeddingPlatform": "Azure AI",
    "embeddingModel": "Ada-200-1",
    "monthlyBudget": 2000.00,
    "environment": "Production",
    "status": "active",
    "createdAt": "2025-09-02T09:30:15.000Z",
    "updatedAt": "2025-09-02T11:00:00.000Z",
    "totalPages": 3
  }
]
```

---

## 4. Get Single LLM Connection

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/get
```

### Request Body
```json
{
  "connection_id": 1
}
```

### Response (200 OK)
```json
{
  "id": 1,
  "llmPlatform": "OpenAI",
  "llmModel": "GPT-4o",
  "embeddingPlatform": "OpenAI",
  "embeddingModel": "text-embedding-3-small",
  "monthlyBudget": 1000.00,
  "environment": "Testing",
  "status": "active",
  "createdAt": "2025-09-02T10:15:30.000Z",
  "updatedAt": "2025-09-02T10:15:30.000Z"
}
```

### Response (404 Not Found)
```json
"error: connection not found"
```

---

## 5. Add New LLM Connection

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/add
```

### Request Body
```json
{
  "llm_platform": "OpenAI",
  "llm_model": "GPT-4o",
  "embedding_platform": "OpenAI",
  "embedding_model": "text-embedding-3-small",
  "monthly_budget": 1000.00,
  "environment": "Testing"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llm_platform` | string | Yes | LLM platform (e.g., "Azure AI", "OpenAI") |
| `llm_model` | string | Yes | LLM model (e.g., "GPT-4o") |
| `embedding_platform` | string | Yes | Embedding platform |
| `embedding_model` | string | Yes | Embedding model |
| `monthly_budget` | number | Yes | Monthly budget amount |
| `environment` | string | Yes | "Testing" or "Production" |

### Response (200 OK)
```json
{
  "id": 3,
  "llm_platform": "OpenAI",
  "llm_model": "GPT-4o",
  "embedding_platform": "OpenAI",
  "embedding_model": "text-embedding-3-small",
  "monthly_budget": 1000.00,
  "environment": "Testing",
  "status": "active",
  "created_at": "2025-09-02T12:00:00.000Z",
  "updated_at": "2025-09-02T12:00:00.000Z"
}
```

### Response (400 Bad Request)
```json
"error: environment must be 'Testing' or 'Production'"
```

---

## 6. Update LLM Connection

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/edit
```

### Request Body
```json
{
  "connection_id": 1,
  "llm_platform": "Azure AI",
  "llm_model": "GPT-4o-mini",
  "embedding_platform": "Azure AI",
  "embedding_model": "Ada-200-1",
  "monthly_budget": 2000.00,
  "environment": "Production"
}
```

### Response (200 OK)
```json
{
  "id": 1,
  "llm_platform": "Azure AI",
  "llm_model": "GPT-4o-mini",
  "embedding_platform": "Azure AI",
  "embedding_model": "Ada-200-1",
  "monthly_budget": 2000.00,
  "environment": "Production",
  "status": "active",
  "created_at": "2025-09-02T10:15:30.000Z",
  "updated_at": "2025-09-02T12:30:00.000Z"
}
```

### Response (404 Not Found)
```json
"error: connection not found"
```

---

## 7. Delete LLM Connection

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/delete
```

### Request Body
```json
{
  "connection_id": 1
}
```

### Response (200 OK)
```json
"LLM connection deleted successfully"
```

### Response (404 Not Found)
```json
"error: connection not found"
```

---

## 4. List All LLM Connections

### Endpoint
```http
GET /ruuter-private/llm/connections/list
```

### Query Parameters (Optional for filtering)
| Parameter | Type | Description |
|-----------|------|-------------|
| `llmPlatform` | `string` | Filter by LLM platform |
| `llmModel` | `string` | Filter by LLM model |
| `deploymentEnvironment` | `string` | Filter by environment (Testing / Production) |
| `pageNumber` | `number` | Page number (1-based) |
| `pageSize` | `number` | Number of items per page |
| `sortBy` | `string` | Field to sort by |
| `sortOrder` | `string` | Sort order: 'asc' or 'desc' |

### Example Request
```http
GET /ruuter-private/llm/connections/list?llmPlatform=OpenAI&deploymentEnvironment=Testing&model=GPT4
```

---

## 5. Get Production LLM Connection (with filters)

### Endpoint
```http
GET /ruuter-private/llm/connections/production
```

### Query Parameters (Optional for filtering)
| Parameter | Type | Description |
|-----------|------|-------------|
| `llmPlatform` | `string` | Filter by LLM platform |
| `llmModel` | `string` | Filter by LLM model |
| `embeddingPlatform` | `string` | Filter by embedding platform |
| `embeddingModel` | `string` | Filter by embedding model |
| `connectionStatus` | `string` | Filter by connection status |
| `sortBy` | `string` | Field to sort by |
| `sortOrder` | `string` | Sort order: 'asc' or 'desc' |

### Example Request
```http
GET /ruuter-private/llm/connections/production?llmPlatform=OpenAI&connectionStatus=active
```

### Response (200 OK)
```json
[
  {
    "id": 1,
    "llmPlatform": "OpenAI",
    "llmModel": "GPT-4o",
    "embeddingPlatform": "OpenAI",
    "embeddingModel": "text-embedding-3-small",
    "monthlyBudget": 1000.00,
    "deploymentEnvironment": "Testing",
    "status": "active",
    "createdAt": "2025-09-02T10:15:30.000Z",
    "updatedAt": "2025-09-02T10:15:30.000Z"
  }
]
```

---

## 5. Get Single LLM Connection

### Endpoint
```http
GET /ruuter-private/llm/connections/overview
```

### Response (200 OK)
```json
{
  "id": 1,
  "llmPlatform": "OpenAI",
  "llmModel": "GPT-4o",
  "embeddingPlatform": "OpenAI",
  "embeddingModel": "text-embedding-3-small",
  "monthlyBudget": 1000.00,
  "deploymentEnvironment": "Testing",
  "status": "active",
  "createdAt": "2025-09-02T10:15:30.000Z",
  "updatedAt": "2025-09-02T10:15:30.000Z"
}
```

---

## 6. Check if LLM Connection Exists

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/exists
```

### Request Body
```json
{ "connection_id": 1 }
```

### Response (200 OK)
```json
"true"
```
or
```json
"false"
```

---

## 7. Update LLM Connection Status

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/update-status
```

### Request Body
```json
{
  "connection_id": 1,
  "connection_status": "inactive"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `connection_id` | number | Yes | LLM connection ID |
| `connection_status` | string | Yes | `"active"` or `"inactive"` |

### Response (200 OK)
Returns the updated connection object.

### Response (400 Bad Request)
```json
"error: connection_status must be 'active' or 'inactive'"
```

### Response (404 Not Found)
```json
"error: connection not found"
```

---

## 8. List LLM Connections — GET (Paginated, with Filters)

### Endpoint
```http
GET /ruuter-private/rag-search/llm-connections/list
```

### Query Parameters
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `pageNumber` | number | No | `1` | Page number (1-based) |
| `pageSize` | number | No | `10` | Items per page (1–100) |
| `sortBy` | string | No | `"created_at"` | Field to sort by |
| `sortOrder` | string | No | `"desc"` | `"asc"` or `"desc"` |
| `llmPlatform` | string | No | `""` | Filter by LLM platform |
| `llmModel` | string | No | `""` | Filter by LLM model |
| `environment` | string | No | `""` | Filter by environment |

### Example Request
```http
GET /ruuter-private/rag-search/llm-connections/list?pageNumber=1&pageSize=10&llmPlatform=OpenAI
```

### Response (200 OK)
```json
[
  {
    "id": 1,
    "llmPlatform": "OpenAI",
    "llmModel": "GPT-4o",
    "embeddingPlatform": "OpenAI",
    "embeddingModel": "text-embedding-3-small",
    "monthlyBudget": 1000.00,
    "environment": "Testing",
    "status": "active",
    "createdAt": "2025-09-02T10:15:30.000Z",
    "updatedAt": "2025-09-02T10:15:30.000Z",
    "totalPages": 3
  }
]
```

### Response (400 Bad Request)
```json
"Page number must be greater than 0"
```

---

## 9. List All LLM Connections — GET (Paginated, with Filters)

### Endpoint
```http
GET /ruuter-private/rag-search/llm-connections/all
```

Same as endpoint 8 above but queries all connections regardless of status. Accepts the same query parameters.

---

## 10. Get Production LLM Connection — GET (with Filters)

### Endpoint
```http
GET /ruuter-private/rag-search/llm-connections/production
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llmPlatform` | string | No | Filter by LLM platform |
| `llmModel` | string | No | Filter by LLM model |
| `embeddingPlatform` | string | No | Filter by embedding platform |
| `embeddingModel` | string | No | Filter by embedding model |
| `connectionStatus` | string | No | Filter by connection status |
| `sortBy` | string | No | Field to sort by (default: `"created_at"`) |
| `sortOrder` | string | No | `"asc"` or `"desc"` (default: `"desc"`) |

### Example Request
```http
GET /ruuter-private/rag-search/llm-connections/production?connectionStatus=active
```

### Response (200 OK)
```json
[
  {
    "id": 1,
    "llmPlatform": "OpenAI",
    "llmModel": "GPT-4o",
    "embeddingPlatform": "OpenAI",
    "embeddingModel": "text-embedding-3-small",
    "monthlyBudget": 1000.00,
    "environment": "Production",
    "status": "active",
    "createdAt": "2025-09-02T10:15:30.000Z",
    "updatedAt": "2025-09-02T10:15:30.000Z"
  }
]
```

---

## 11. Update Used Budget for a Connection

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/cost/update
```

Adds `usage` to the connection's current `used_budget`. If `disconnectOnBudgetExceed` is set and the stop threshold is reached, the connection is automatically deactivated.

### Request Body
```json
{
  "connection_id": 1,
  "usage": 12.50
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `connection_id` | number | Yes | LLM connection ID |
| `usage` | number | Yes | Amount to add to `used_budget` (≥ 0) |

### Response (200 OK) — within budget
```json
{
  "data": { "id": 1, "usedBudget": 162.50, "monthlyBudget": 1000.00 },
  "budgetExceeded": false,
  "message": "Used budget updated successfully",
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (200 OK) — budget exceeded, connection deactivated
```json
{
  "data": { "id": 1, "usedBudget": 1005.00, "status": "inactive" },
  "budgetExceeded": true,
  "message": "Used budget updated successfully. Connection deactivated due to budget threshold exceeded.",
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (400 Bad Request)
```json
"error: connection_id and usage (>= 0) are required"
```

### Response (404 Not Found)
```json
"error: connection not found"
```

---

## 12. Check Budget Usage

### Endpoint
```http
POST /ruuter-private/rag-search/llm-connections/usage/check
```

Returns whether the connection's budget is within the stop threshold, exceeded (not disconnected), or exceeded with disconnection.

### Request Body
```json
{ "connection_id": 1 }
```

### Response (200 OK) — within budget
```json
{
  "isBudgetExceed": false,
  "isLLMConnectionDisconnected": false
}
```

### Response (200 OK) — exceeded, not disconnected
```json
{
  "isBudgetExceed": true,
  "isLLMConnectionDisconnected": false
}
```

### Response (200 OK) — exceeded and disconnected
```json
{
  "isBudgetExceed": true,
  "isLLMConnectionDisconnected": true
}
```

### Response (404 Not Found)
```json
"Connection not found"
```

---

## 13. Check Budget Thresholds for Production Connection

### Endpoint
```http
GET /ruuter-private/rag-search/llm-connections/cost/check
```

Returns warn/stop threshold status for the active production connection.

### Response (200 OK)
```json
{
  "data": {
    "id": 1,
    "monthlyBudget": 1000.00,
    "usedBudget": 620.00,
    "warnBudgetThreshold": 70,
    "stopBudgetThreshold": 90
  },
  "used_budget_percentage": 62.0,
  "exceeded_stop_budget": false,
  "exceeded_warn_budget": false
}
```

### Response (404 Not Found)
```json
"No production LLM connection found"
```

---

## 14. Reset Used Budget for All Connections

### Endpoint
```http
POST /ruuter-public/rag-search/llm-connections/cost/reset
```

Resets `used_budget` to `0` for all LLM connections. Typically called by a scheduled job at the start of each billing period.

### Request Body
None required.

### Response (200 OK)
```json
{
  "message": "Used budget reset to 0 successfully for all connections",
  "totalConnections": "5",
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (500 Internal Server Error)
```json
"error: failed to reset used budget"
```

---
# Inference Results API Endpoints

## Base URL
```
/ruuter-private/inference/results
```

---

## 1. Store Test Inference Result

### Endpoint
```http
POST /ruuter-private/inference/results/test/store
```

### Request Body
```json
{
  "llm_connection_id": 1,
  "user_question": "What are the benefits of using LLMs?",
  "final_answer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation."
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llm_connection_id` | number | Yes | ID of the LLM connection |
| `user_question` | string | Yes | User's raw question/input |
| `final_answer` | string | Yes | LLM's final generated answer |

### Response (200 OK)
```json
{
  "data": {
    "id": 10,
    "llm_connection_id": 1,
    "chat_id": null,
    "user_question": "What are the benefits of using LLMs?",
    "refined_questions": null,
    "conversation_history": null,
    "ranked_chunks": null,
    "embedding_scores": null,
    "final_answer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation.",
    "environment": "testing",
    "created_at": "2025-09-25T12:15:00.000Z"
  },
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (400 Bad Request)
```json
{
  "data": "[]",
  "operationSuccess": false,
  "statusCode": 400
}
```

### Response (404 Not Found)
```json
"error: LLM connection not found"
```

---

## 2. Store Production Inference Result

### Endpoint
```http
POST /ruuter-private/inference/results/production/store
```

### Request Body
```json
{
  "chat_id": "chat-12345",
  "user_question": "What are the benefits of using LLMs?",
  "refined_questions": [
    "How do LLMs improve productivity?",
    "What are practical use cases of LLMs?"
  ],
  "conversation_history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help you?" }
  ],
  "ranked_chunks": [
    { "id": "chunk_1", "content": "LLMs help in summarization", "rank": 1 },
    { "id": "chunk_2", "content": "They improve Q&A systems", "rank": 2 }
  ],
  "embedding_scores": [0.92, 0.85, 0.78],
  "final_answer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation."
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `chat_id` | string | No | Optional chat session ID |
| `user_question` | string | Yes | User's raw question/input |
| `refined_questions` | object | No | List of refined questions (LLM-generated) |
| `conversation_history` | object | No | Prior messages array of {role, content} |
| `ranked_chunks` | object | No | Retrieved chunks ranked with metadata |
| `embedding_scores` | object | No | Distance scores for each chunk |
| `final_answer` | string | Yes | LLM's final generated answer |

### Response (200 OK)
```json
{
  "data": {
    "id": 15,
    "llm_connection_id": null,
    "chat_id": "chat-12345",
    "user_question": "What are the benefits of using LLMs?",
    "refined_questions": [
      "How do LLMs improve productivity?",
      "What are practical use cases of LLMs?"
    ],
    "conversation_history": [
      { "role": "user", "content": "Hello" },
      { "role": "assistant", "content": "Hi! How can I help you?" }
    ],
    "ranked_chunks": [
      { "id": "chunk_1", "content": "LLMs help in summarization", "rank": 1 },
      { "id": "chunk_2", "content": "They improve Q&A systems", "rank": 2 }
    ],
    "embedding_scores": [0.92, 0.85, 0.78],
    "final_answer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation.",
    "environment": "production",
    "created_at": "2025-09-25T12:15:00.000Z"
  },
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (400 Bad Request)
```json
{
  "data": "[]",
  "operationSuccess": false,
  "statusCode": 400
}
```

---

## 3. View/get Inference Result

### Endpoint
```http
POST /ruuter-private/inference/results/test/store
```

### Request Body
```json
{
  "llmConnectionId": 1,
  "userQuestion": "What are the benefits of using LLMs?",
  "finalAnswer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation."
}
```

### Response (201 Created)
```json
{
  "data": {
    "id": 15,
    "llmConnectionId": 1,
    "userQuestion": "What are the benefits of using LLMs?",
    "finalAnswer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation.",
    "environment": "testing",
    "createdAt": "2025-09-25T10:15:30.000Z"
  },
  "operationSuccess": true,
  "statusCode": 200
}
```

## 4. Inquiry from chatbot to llm orchestration service

### Endpoint
```http
POST /ruuter-private/inference/results/production/store
```

### Request Body
```json
{
  "llmConnectionId": 1,
  "chatId": "chat-session-12345",
  "userQuestion": "What are the benefits of using LLMs?",
  "refinedQuestions": [
    "How do LLMs improve productivity?",
    "What are practical use cases of LLMs?"
  ],
  "conversationHistory": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help you?" }
  ],
  "rankedChunks": [
    { "id": "chunk_1", "content": "LLMs help in summarization", "rank": 1 },
    { "id": "chunk_2", "content": "They improve Q&A systems", "rank": 2 }
  ],
  "embeddingScores": {
    "chunk_1": 0.92,
    "chunk_2": 0.85
  },
  "finalAnswer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation."
}
```

### Response (201 Created)
```json
{
  "id": 20,
  "llmConnectionId": 1,
  "chatId": "chat-session-12345",
  "userQuestion": "What are the benefits of using LLMs?",
  "refinedQuestions": [
    "How do LLMs improve productivity?",
    "What are practical use cases of LLMs?"
  ],
  "conversationHistory": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help you?" }
  ],
  "rankedChunks": [
    { "id": "chunk_1", "content": "LLMs help in summarization", "rank": 1 },
    { "id": "chunk_2", "content": "They improve Q&A systems", "rank": 2 }
  ],
  "embeddingScores": {
    "chunk_1": 0.92,
    "chunk_2": 0.85
  },
  "finalAnswer": "LLMs can improve productivity by summarizing large documents, enabling Q&A, and enhancing automation.",
  "environment": "production",
  "createdAt": "2025-09-25T10:15:30.000Z"
}
```

---

## 5. Production Inference

### Endpoint
```http
POST /ruuter-private/rag-search/inference/production
```

Validates the production connection's budget then proxies the request to the LLM Orchestration Service.

### Request Body
```json
{
  "chatId": "chat-session-123",
  "message": "What are the benefits of using LLMs?",
  "authorId": "user-456",
  "conversationHistory": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi! How can I help you?" }
  ],
  "url": "https://example.com/context"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `chatId` | string | Yes | Chat session ID |
| `message` | string | Yes | User message |
| `authorId` | string | Yes | Author ID |
| `conversationHistory` | array | No | Prior `{role, content}` messages |
| `url` | string | No | URL reference |

### Response (200 OK)
Proxied response from the LLM Orchestration Service.

### Response (400 Bad Request) — connection disconnected due to budget
```json
{
  "chatId": "chat-session-123",
  "content": "The LLM connection is currently unavailable. Your request couldn't be processed. Please retry shortly.",
  "status": 400
}
```

### Response (404 Not Found)
```json
"No production connection found"
```

---

## 6. Test Inference

### Endpoint
```http
POST /ruuter-private/rag-search/inference/test
```

Validates a specific connection's budget then calls the LLM Orchestration Service `/test` endpoint.

### Request Body
```json
{
  "connectionId": "1",
  "message": "What are the benefits of using LLMs?"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `connectionId` | string | Yes | Connection ID to test against |
| `message` | string | Yes | User message |

### Response (200 OK)
Proxied response from the LLM Orchestration Service `/test` endpoint.

### Response (400 Bad Request) — connection disconnected due to budget
```json
{
  "connectionId": "1",
  "content": "The LLM connection is currently unavailable. Your request couldn't be processed. Please retry shortly.",
  "status": 400
}
```

### Response (404 Not Found)
```json
"No test connection found"
```

---

## 7. View Inference Result (Mock)

### Endpoint
```http
POST /ruuter-private/rag-search/inference/results/view
```

Returns a mock inference response for testing purposes.

### Request Body
```json
{
  "llmConnectionId": 1,
  "message": "What services are available?"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `llmConnectionId` | number | Yes | LLM connection ID |
| `message` | string | Yes | User message/question |

### Response (200 OK)
```json
{
  "chatId": 10,
  "llmServiceActive": true,
  "questionOutOfLlmScope": true,
  "content": "Random answer with citations\n  - https://gov.ee/sample1,\n  - https://gov.ee/sample1"
}
```

### Response (400 Bad Request)
```json
"llmConnectionId and message are required"
```

---

## 8. Store Inference Result (Public)

### Endpoint
```http
POST /ruuter-public/rag-search/inference/results/store
```

Public variant of the inference result store. Accepts the same fields as the private store endpoints, plus `environment` and `vault_uuid`.

### Request Body
```json
{
  "user_question": "What are the benefits of using LLMs?",
  "final_answer": "LLMs can improve productivity...",
  "chat_id": "chat-12345",
  "environment": "production",
  "vault_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "refined_questions": ["How do LLMs improve productivity?"],
  "conversation_history": [{ "role": "user", "content": "Hello" }],
  "ranked_chunks": [{ "id": "chunk_1", "content": "...", "rank": 1 }],
  "embedding_scores": [0.92, 0.85]
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `user_question` | string | Yes | User's raw question/input |
| `final_answer` | string | Yes | LLM's final generated answer |
| `chat_id` | string | No | Chat session ID |
| `environment` | string | No | Environment identifier |
| `vault_uuid` | string | No | Vault UUID for the LLM connection |
| `refined_questions` | object | No | List of refined questions |
| `conversation_history` | object | No | Prior `{role, content}` messages |
| `ranked_chunks` | object | No | Retrieved chunks ranked with metadata |
| `embedding_scores` | object | No | Distance scores for each chunk |

### Response (200 OK)
```json
{
  "data": { "id": 20, "user_question": "...", "final_answer": "...", "environment": "production" },
  "operationSuccess": true,
  "statusCode": 200
}
```

### Response (400 Bad Request)
```json
{
  "data": "[]",
  "operationSuccess": false,
  "statusCode": 400
}
```

---

# LLM Platforms & Models API Endpoints

## Base URL
```
/ruuter-private/rag-search
```

---

## 1. Get LLM Platforms

### Endpoint
```http
GET /ruuter-private/rag-search/llm/platforms
```

Returns all active LLM platforms.

### Response (200 OK)
```json
[
  { "id": 1, "value": "openai", "label": "OpenAI" },
  { "id": 2, "value": "azure", "label": "Azure AI" },
  { "id": 3, "value": "aws", "label": "AWS Bedrock" }
]
```

---

## 2. Get LLM Models by Platform

### Endpoint
```http
GET /ruuter-private/rag-search/llm/models
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `platform_key` | string | Yes | Platform key to filter models (e.g. `"openai"`) |

### Example Request
```http
GET /ruuter-private/rag-search/llm/models?platform_key=openai
```

### Response (200 OK)
```json
[
  { "id": 1, "value": "gpt-4o", "label": "GPT-4o", "platform_id": 1, "platform_key": "openai", "platform_name": "OpenAI" },
  { "id": 2, "value": "gpt-4o-mini", "label": "GPT-4o-mini", "platform_id": 1, "platform_key": "openai", "platform_name": "OpenAI" }
]
```

---

## 3. Get All LLM Models

### Endpoint
```http
GET /ruuter-private/rag-search/llm/models-list
```

Returns all LLM models with no platform filter.

### Response (200 OK)
```json
[
  { "id": 1, "platform_id": 1, "value": "gpt-4o", "label": "GPT-4o" },
  { "id": 2, "platform_id": 1, "value": "gpt-4o-mini", "label": "GPT-4o-mini" }
]
```

---

## 4. Get Embedding Platforms

### Endpoint
```http
GET /ruuter-private/rag-search/embedding/platforms
```

Returns all active embedding platforms.

### Response (200 OK)
```json
[
  { "id": 1, "value": "openai", "label": "OpenAI" },
  { "id": 2, "value": "azure", "label": "Azure AI" }
]
```

---

## 5. Get Embedding Models by Platform

### Endpoint
```http
GET /ruuter-private/rag-search/embedding/models
```

### Query Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `embedding_platform_key` | string | Yes | Platform key to filter models |

### Example Request
```http
GET /ruuter-private/rag-search/embedding/models?embedding_platform_key=openai
```

### Response (200 OK)
```json
[
  { "id": 1, "value": "text-embedding-3-small", "label": "text-embedding-3-small", "platform_id": 1, "platform_key": "openai", "platform_name": "OpenAI" },
  { "id": 2, "value": "text-embedding-ada-002", "label": "text-embedding-ada-002", "platform_id": 1, "platform_key": "openai", "platform_name": "OpenAI" }
]
```

---

# Prompt Configuration API Endpoints

## Base URL
```
/ruuter-private/rag-search/prompt-configuration
```

---

## 1. Get Prompt Configuration

### Endpoint
```http
GET /ruuter-private/rag-search/prompt-configuration/get
```

Returns the active custom prompt configuration. Returns an empty array if none is configured.

### Response (200 OK)
```json
[
  {
    "id": 1,
    "prompt": "You are a helpful assistant for government services...",
    "created_at": "2025-09-02T10:15:30.000Z",
    "updated_at": "2025-09-02T12:30:00.000Z"
  }
]
```

---

## 2. Save Prompt Configuration

### Endpoint
```http
POST /ruuter-private/rag-search/prompt-configuration/save
```

Upserts the prompt configuration (inserts if none exists, updates otherwise). Also triggers an LLM cache refresh.

### Request Body
```json
{
  "prompt": "You are a helpful assistant for government services. Answer questions accurately and concisely."
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Prompt text to save |

### Response (200 OK)
Returns the saved prompt configuration object.

```json
{
  "id": 1,
  "prompt": "You are a helpful assistant for government services. Answer questions accurately and concisely.",
  "updated_at": "2025-09-25T12:00:00.000Z"
}
```

---

# Vault Secrets API Endpoints

## Base URL
```
/ruuter-private/rag-search/vault/secret
```

---

## 1. Create Vault Secret

### Endpoint
```http
POST /ruuter-private/rag-search/vault/secret/create
```

Stores LLM connection credentials in Vault via CronManager. Supported platforms: `"aws"`, `"azure"`.

### Request Body (AWS)
```json
{
  "vaultUuid": "550e8400-e29b-41d4-a716-446655440000",
  "llmPlatform": "aws",
  "llmModel": ["claude-3-sonnet"],
  "secretKey": "aws-secret-key",
  "accessKey": "aws-access-key",
  "embeddingModel": "amazon.titan-embed-text-v1",
  "embeddingPlatform": "aws",
  "embeddingAccessKey": "embed-access-key",
  "embeddingSecretKey": "embed-secret-key",
  "deploymentEnvironment": "Production"
}
```

### Request Body (Azure)
```json
{
  "vaultUuid": "550e8400-e29b-41d4-a716-446655440000",
  "llmPlatform": "azure",
  "llmModel": ["gpt-4o"],
  "deploymentName": "my-deployment",
  "targetUrl": "https://my-endpoint.azure.com",
  "apiKey": "azure-api-key",
  "embeddingModel": "text-embedding-ada-002",
  "embeddingPlatform": "azure",
  "embeddingDeploymentName": "embed-deployment",
  "embeddingTargetUri": "https://embed-endpoint.azure.com",
  "embeddingAzureApiKey": "embed-azure-api-key",
  "deploymentEnvironment": "Production"
}
```

### Request Parameters
| Parameter | Type | Platform | Description |
|-----------|------|----------|-------------|
| `vaultUuid` | string | Both | Stable UUID for the vault path |
| `llmPlatform` | string | Both | `"aws"` or `"azure"` |
| `llmModel` | array | Both | LLM model identifier(s) |
| `deploymentEnvironment` | string | Both | Deployment environment |
| `embeddingModel` | string | Both | Embedding model identifier |
| `embeddingPlatform` | string | Both | Embedding platform |
| `secretKey` | string | AWS | AWS secret key |
| `accessKey` | string | AWS | AWS access key |
| `embeddingAccessKey` | string | AWS | Embedding AWS access key |
| `embeddingSecretKey` | string | AWS | Embedding AWS secret key |
| `deploymentName` | string | Azure | Azure deployment name |
| `targetUrl` | string | Azure | Azure endpoint URL |
| `apiKey` | string | Azure | Azure API key |
| `embeddingDeploymentName` | string | Azure | Embedding Azure deployment name |
| `embeddingTargetUri` | string | Azure | Embedding Azure endpoint URI |
| `embeddingAzureApiKey` | string | Azure | Embedding Azure API key |

### Response (200 OK) — AWS
```json
"Executed cron manager successfully to store aws secrets"
```

### Response (200 OK) — Azure
```json
"Executed cron manager successfully to store azure secrets"
```

### Response (400 Bad Request)
```json
{
  "message": "Platform not supported",
  "operationSuccessful": false,
  "statusCode": 400
}
```

---

## 2. Delete Vault Secret

### Endpoint
```http
POST /ruuter-private/rag-search/vault/secret/delete
```

Removes LLM connection credentials from Vault via CronManager.

### Request Body
```json
{
  "vaultUuid": "550e8400-e29b-41d4-a716-446655440000",
  "llmPlatform": "azure",
  "llmModel": "gpt-4o",
  "embeddingModel": "text-embedding-ada-002",
  "embeddingPlatform": "azure",
  "deploymentEnvironment": "Production"
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `vaultUuid` | string | Yes | Vault UUID of the connection |
| `llmPlatform` | string | Yes | LLM platform |
| `llmModel` | string | Yes | LLM model identifier |
| `embeddingModel` | string | Yes | Embedding model identifier |
| `embeddingPlatform` | string | Yes | Embedding platform |
| `deploymentEnvironment` | string | Yes | Deployment environment |

### Response (200 OK)
```json
"Executed cron manager successfully to delete secrets from vault"
```

### Response (404 Not Found)
```json
{
  "message": "Connection not found with the provided vaultUuid",
  "operationSuccessful": false,
  "statusCode": 404
}
```

---

# Data Sync & Services API Endpoints

## Base URL
```
/ruuter-public/rag-search
```

---

## 1. Get Services (for Intent Detection)

### Endpoint
```http
GET /ruuter-public/rag-search/services/get-services
```

Returns all active services if the count is ≤ 10. If count > 10, signals the caller to use semantic search instead.

### Response (200 OK) — ≤ 10 services
```json
{
  "use_semantic_search": false,
  "service_count": 5,
  "services": [
    { "id": "svc-1", "name": "Pension Application", "description": "..." }
  ]
}
```

### Response (200 OK) — > 10 services
```json
{
  "use_semantic_search": true,
  "service_count": 23,
  "message": "Service count exceeds threshold - use semantic search"
}
```

---

## 2. Resync Data from KB

### Endpoint
```http
POST /ruuter-public/rag-search/data/update
```

Fetches the latest agency data from CKB, compares the data hash, and if changed triggers vector re-indexing via CronManager.

### Request Body
None required.

### Response (200 OK) — sync initiated
```json
{
  "message": "Data synchronization initiated successfully",
  "operationSuccessful": true
}
```

### Response (200 OK) — already up to date
```json
{
  "success": true,
  "message": "No sync required - data is up to date"
}
```

### Response (400 Bad Request)
```json
{
  "message": "CKB service returned an error - data synchronization aborted",
  "operationSuccessful": false,
  "error": "CKB_ERROR"
}
```

### Response (404 Not Found)
```json
{
  "success": false,
  "message": "Data synchronization failed - CKB agency data not found"
}
```

---

## 3. Trigger API Tool Endpoint Indexing

### Endpoint
```http
POST /ruuter-public/rag-search/api-tools/index
```

Queues an API tool endpoint for vector indexing in Qdrant via CronManager (async).

### Request Body
```json
{
  "endpointId": "ep-001",
  "serviceId": "svc-1",
  "name": "Get Pension Status",
  "description": "Retrieve the current pension application status for a citizen",
  "method": "GET",
  "url": "https://api.example.com/pension/status",
  "visibility": "public",
  "params": [
    { "name": "nationalId", "type": "string", "required": true }
  ]
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `endpointId` | string | Yes | Unique endpoint identifier |
| `name` | string | Yes | Endpoint name |
| `description` | string | Yes | Endpoint description |
| `url` | string | Yes | API URL |
| `serviceId` | string | No | Parent service ID |
| `method` | string | No | HTTP method (default: `"GET"`) |
| `visibility` | string | No | `"public"` or `"private"` (default: `"public"`) |
| `type` | string | No | Endpoint type (default: `"custom_endpoint"`) |
| `params` | array | No | List of parameters |

### Response (200 OK)
```json
{
  "success": true,
  "endpoint_id": "ep-001",
  "message": "API Tool indexing job queued successfully. Processing asynchronously."
}
```

### Response (400 Bad Request)
```json
{
  "success": false,
  "error": "MISSING_REQUIRED_FIELDS",
  "message": "endpointId, name, description, and url are required"
}
```

### Response (500 Internal Server Error)
```json
{
  "success": false,
  "error": "INDEXING_QUEUE_FAILED",
  "message": "Failed to queue indexing job. CronManager may be unavailable."
}
```

---

## 4. Enrich and Index Service

### Endpoint
```http
POST /ruuter-public/rag-search/services/enrich
```

Queues a service for enrichment and Qdrant indexing via CronManager (async).

### Request Body
```json
{
  "service_id": "svc-001",
  "name": "Pension Application",
  "description": "Submit a new pension application for eligible citizens",
  "examples": ["How do I apply for pension?", "Pension eligibility requirements"],
  "entities": ["nationalId", "dateOfBirth"],
  "ruuter_type": "POST",
  "current_state": "active",
  "is_common": false
}
```

### Request Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `service_id` | string | Yes | Unique service identifier |
| `name` | string | Yes | Service name |
| `description` | string | Yes | Service description |
| `examples` | array | No | Example user queries |
| `entities` | array | No | Expected entity names |
| `ruuter_type` | string | No | HTTP method (default: `"GET"`) |
| `current_state` | string | No | `"active"`, `"inactive"`, or `"draft"` (default: `"draft"`) |
| `is_common` | boolean | No | Whether this is a common service (default: `false`) |

### Response (200 OK)
```json
{
  "success": true,
  "service_id": "svc-001",
  "message": "Service enrichment job queued successfully. Processing asynchronously."
}
```

### Response (400 Bad Request)
```json
{
  "success": false,
  "error": "MISSING_REQUIRED_FIELDS",
  "message": "service_id, name, and description are required"
}
```

### Response (500 Internal Server Error)
```json
{
  "success": false,
  "error": "ENRICHMENT_QUEUE_FAILED",
  "message": "Failed to queue enrichment job. CronManager may be unavailable."
}
```

---
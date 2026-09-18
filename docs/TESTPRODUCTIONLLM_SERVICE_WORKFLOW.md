# TestProductionLLM Page — Service Workflow & Streaming Documentation

This document traces the **service workflow** end-to-end through the **TestProductionLLM** UI page. Unlike the TestModel page, this page exclusively uses the **streaming endpoint** (`/orchestrate/stream`). It covers:

1. **Natural-language service detection** — the user sends a free-text query that is classified as a service, with the response delivered via SSE.
2. **MCQ button-click** — the user clicks a choice button, sending the `#service` payload through the same streaming pipeline but short-circuiting all NLU.
3. **Deep dive into the streaming architecture** — how tokens flow from the backend through the notification server to the browser.

---

## Architecture Overview

The TestProductionLLM streaming architecture has **three hops**:

```
┌──────────────────┐                      ┌───────────────────┐                    ┌───────────────────────┐
│  TestProductionLLM│   1. GET /sse/stream │  Notification     │  3. POST /orchestrate│  LLM Orchestration    │
│  (Browser)       │ ◀════════════(SSE)══ │  Server (Node.js) │    /stream           │  Service (Python)     │
│                  │                      │                   │ ──────────────────▷  │                       │
│  index.tsx       │   2. POST /channels/ │                   │ ◀═══(SSE stream)═══ │  llm_orchestration_   │
│  useStreaming     │      orchestrate/    │  streamingService │                      │  service.py           │
│  Response.tsx    │      stream          │  .js              │                      │                       │
└──────────────────┘ ──────────────────▷  └───────────────────┘                    └───────────────────────┘
```

**Why three hops?** The browser opens a persistent SSE connection to the notification server (Node.js), then triggers the stream via a separate POST. The notification server acts as a relay: it calls the Python backend's `/orchestrate/stream` endpoint, reads the SSE response, parses each `data:` line, and re-emits it to the browser through its own SSE connection with a different message format.

---

## Key Files

| Layer | File | Purpose |
|---|---|---|
| **GUI Page** | `GUI/src/pages/TestProductionLLM/index.tsx` | Chat-style UI with message history, input area, MCQ button rendering |
| **Streaming Hook** | `GUI/src/hooks/useStreamingResponse.tsx` | `useStreamingResponse()` — manages SSE connection lifecycle, calls notification server |
| **GUI Service** | `GUI/src/services/inference.ts` | `ChoiceButton` interface (shared with TestModel) |
| **Notification Server** | `notification-server/src/server.js` | Express server: `GET /sse/stream/:channelId` and `POST /channels/:channelId/orchestrate/stream` |
| **Streaming Service** | `notification-server/src/streamingService.js` | `createLLMOrchestrationStreamRequest()` — calls backend, parses SSE, re-emits to browser |
| **API Layer** | `src/llm_orchestration_service_api.py` | `/orchestrate/stream` handler — validates, rate-limits, wraps with timeout |
| **Orchestration** | `src/llm_orchestration_service.py` | `stream_orchestration_response()` — the core streaming pipeline |
| **Service Workflow** | `src/tool_classifier/workflows/service_workflow.py` | `execute_streaming()`, `execute_direct_step_streaming()` — SSE generators for service responses |
| **Models** | `src/models/request_models.py` | `OrchestrationRequest`, `ChoiceButton` |

---

## Flow 1: Natural-Language Service Detection (Streaming)

### 1.1 Frontend — User Sends a Message

The user types a message and presses Send or Enter.

**`TestProductionLLM/index.tsx` → `handleSendMessage()`** (line 47):

1. Adds a user `Message` to state (with id, content, timestamp)
2. Clears input, sets `isLoading=true`
3. Creates a `botMessageId` for the upcoming bot response
4. Builds `conversationHistory` from existing messages
5. Defines callbacks: `onToken`, `onButtons`, `onComplete`, `onError`
6. Calls `startStreaming(userMessageText, streamingOptions, onToken, onComplete, onError, onButtons)`

### 1.2 Streaming Hook — SSE Connection Setup

**`useStreamingResponse.tsx` → `startStreaming()`** (line 52):

This follows a **two-phase protocol**:

#### Phase 1: Open SSE Connection

```ts
const sseUrl = `${notificationNodeUrl}/sse/stream/${channelId}`;
const eventSource = new EventSource(sseUrl);
```

- `channelId` is a unique session-level ID: `channel-<random>` (generated via `useMemo`, line 28)
- The `EventSource` hits `GET /sse/stream/:channelId` on the notification server
- The server registers this connection with a `sender` function that can push messages back

#### Phase 2: Trigger the Stream (after 500ms wait)

```ts
await new Promise(resolve => setTimeout(resolve, 500));  // Wait for SSE to establish

const postUrl = `${notificationNodeUrl}/channels/${channelId}/orchestrate/stream`;
await axios.post(postUrl, { message, options });
```

#### SSE Message Handling

The `eventSource.onmessage` handler processes four message types:

| `data.type` | Action |
|---|---|
| `stream_start` | Sets `isStreaming=true` |
| `stream_chunk` | Calls `onToken(data.content)` — appends token to bot message. If `data.buttons` present → calls `onButtons(data.buttons)` |
| `stream_end` | Closes EventSource, calls `onComplete()` |
| `stream_error` | Closes EventSource, calls `onError(data.error)` |

### 1.3 Notification Server — Relay

**`streamingService.js` → `createLLMOrchestrationStreamRequest()`** (line 11):

1. Finds SSE connections for the channel
2. Constructs the `OrchestrationRequest` payload:

```js
const orchestrationPayload = {
  chatId: channelId,
  message: message,
  authorId: options.authorId || `user-${channelId}`,
  conversationHistory: options.conversationHistory || [],
  url: options.url || "sse-stream-context",
  environment: "production",           // ← hardcoded; streaming is production-only
  connection_id: options.connection_id || connectionId
};
```

3. Calls the Python backend:

```js
const response = await fetch(
  `${LLM_ORCHESTRATOR_URL}/orchestrate/stream`,
  { method: 'POST', body: JSON.stringify(orchestrationPayload) }
);
```

4. Sends `stream_start` to browser
5. Reads the SSE response body as a stream:

```js
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  buffer += decoder.decode(value, { stream: true });
  // Parse "data: {json}\n" lines
  for (const line of lines) {
    const data = JSON.parse(line.slice(6)); // Strip "data: "
    const content = data.payload?.content;
    const buttons = data.payload?.buttons;
    
    if (content === "END") {
      sender({ type: "stream_end", ... });    // → browser
    } else {
      sender({ type: "stream_chunk", content, buttons, ... }); // → browser
    }
  }
}
```

> **Key insight**: The notification server **re-formats** the SSE. The backend emits `data: {"chatId":"...","payload":{"content":"token","buttons":[...]},...}\n\n`, while the notification server emits `{"type":"stream_chunk","content":"token","buttons":[...]}` over its own SSE channel.

### 1.4 Backend API Layer — `/orchestrate/stream`

**`llm_orchestration_service_api.py` → `stream_orchestrated_response()`** (line 415):

1. **Environment check**: Streaming is only available for environments in `STREAMING_ALLOWED_ENVS`
2. **Service initialization check**: Verifies `orchestration_service` is available
3. **Rate limiting**: If enabled, estimates tokens and checks per-user rate limits
4. **Timeout wrapper**:

```python
async def timeout_wrapped_stream():
    async with stream_timeout(StreamConfig.MAX_STREAM_DURATION_SECONDS):
        async for chunk in orchestration_service.stream_orchestration_response(request):
            yield chunk
```

5. Returns `StreamingResponse(timeout_wrapped_stream(), media_type="text/event-stream")`

### 1.5 Orchestration — `stream_orchestration_response()`

**`llm_orchestration_service.py` → `stream_orchestration_response()`** (line 532):

This is an `async generator` that yields SSE-formatted strings.

```
STEP 0:   Language detection → detect_language(request.message)

STEP 0.1: Check request.message.startswith(SERVICE_STEP_PREFIXES)
          → FALSE (natural language) → skip, continue

STEP 0.5: Query validation → validate_query_basic()
          If invalid → yield format_sse(error_msg) + format_sse("END") + return

STEP 1:   StreamManager context (managed_stream for cleanup tracking)

STEP 2:   Component initialization (LLM manager, guardrails)

STEP 3:   Input guardrails check
          If blocked → yield format_sse(violation_msg) + yield format_sse("END") + return

STEP 4:   ToolClassifier.classify() → Classification(workflow=SERVICE)

STEP 5:   route_to_workflow(classification, request, is_streaming=True)
          → ServiceWorkflowExecutor.execute_streaming()
          → returns AsyncIterator[str]

STEP 6:   Yield all SSE chunks from the iterator:
          async for sse_chunk in stream_result:
              yield sse_chunk
```

### 1.6 ServiceWorkflowExecutor — `execute_streaming()`

**`service_workflow.py` → `execute_streaming()`** (line 855):

The service discovery, intent detection, entity extraction, and endpoint call logic is **identical** to `execute_async()` (documented in TestModel flow). The only difference is the response wrapping.

After `_call_service_endpoint()` returns `{"content": str, "buttons": List[Dict]}`:

```python
orchestration_service = self.orchestration_service
service_content = service_result["content"]
service_buttons = service_result["buttons"]


async def service_stream() -> AsyncIterator[str]:
    yield orchestration_service.format_sse(
        chat_id, service_content, service_buttons or None
    )
    yield orchestration_service.format_sse(chat_id, "END")
    orchestration_service.log_costs(costs_metric)


return service_stream()
```

> **Key insight**: For service workflow responses, the stream yields exactly **2 SSE messages**: the complete service response (with content + buttons) and the `END` marker. There is no token-by-token streaming for service endpoints — the entire response arrives in one chunk. This is because the DMapper/Ruuter response is a pre-formed string, not an LLM generation.

### 1.7 `format_sse()` — MCQ Button Inclusion

**`llm_orchestration_service.py` → `format_sse()`** (line 1195):

```python
def format_sse(
    self, chat_id: str, content: str, buttons: Optional[List[Dict[str, Any]]] = None
) -> str:
    inner_payload: Dict[str, Any] = {"content": content}
    if buttons:
        inner_payload["buttons"] = buttons

    payload = {
        "chatId": chat_id,
        "payload": inner_payload,
        "timestamp": str(int(datetime.now().timestamp() * 1000)),
        "sentTo": [],
    }
    return f"data: {json_module.dumps(payload)}\n\n"
```

**Example SSE output for a service with buttons:**
```
data: {"chatId":"channel-abc","payload":{"content":"Which operating system?","buttons":[{"title":"Windows","payload":"#service, /POST/..."},{"title":"Mac","payload":"#service, /POST/..."}]},"timestamp":"1711512000000","sentTo":[]}

data: {"chatId":"channel-abc","payload":{"content":"END"},"timestamp":"1711512000001","sentTo":[]}
```

### 1.8 Frontend — Token Rendering & Button Display

Back in `TestProductionLLM/index.tsx`:

**`onToken` callback** (line 88): Appends the content to the bot message. For service responses, this is the entire answer text in one chunk (not token-by-token).

**`onButtons` callback** (line 120): Attaches buttons to the bot message:

```tsx
const onButtons = (buttons: ChoiceButton[]) => {
  setMessages(prev => {
    const botMsgIndex = prev.findIndex(msg => msg.id === botMessageId);
    if (botMsgIndex === -1) return prev;
    const updated = [...prev];
    updated[botMsgIndex] = { ...updated[botMsgIndex], buttons };
    return updated;
  });
};
```

**Button rendering** (line 286-299): Within each bot message:

```tsx
{!msg.isUser && msg.buttons && msg.buttons.length > 0 && (
  <div className="mcq-buttons">
    {msg.buttons.map((btn) => (
      <Button
        key={btn.payload}
        onClick={() => handleButtonClick(btn.title, btn.payload)}
        disabled={isLoading || isStreaming}
        appearance="secondary"
      >
        {btn.title}
      </Button>
    ))}
  </div>
)}
```

---

## Flow 2: MCQ Button Click (Streaming Direct Step)

### 2.1 Frontend — Button Click

**`handleButtonClick()`** (line 190):

```tsx
const handleButtonClick = async (title: string, payload: string) => {
  if (isLoading || isStreaming) return;

  // Add *title* as user message (not the raw payload)
  const userMessage: Message = {
    id: `user-${Date.now()}`,
    content: title,        // Shows "Windows" in the chat, not "#service, /POST/..."
    isUser: true,
    timestamp: new Date().toISOString(),
  };
  setMessages(prev => [...prev, userMessage]);

  // Start streaming with the *payload* as the message
  await startStreaming(payload, streamingOptions, onToken, onComplete, onError, onButtons);
};
```

> **Note**: Unlike TestModel where the raw payload is shown, TestProductionLLM shows the button **title** to the user and sends the **payload** as the message. This provides a cleaner chat experience.

### 2.2 Same Streaming Path

The payload flows through the same streaming pipeline:

```
Browser → Notification Server → POST /orchestrate/stream → stream_orchestration_response()
```

### 2.3 Orchestration — `#service` Prefix Short-Circuit (Streaming)

**`stream_orchestration_response()`** (line 585-603):

```python
# STEP 0.1: Multi-step service prefix check (bypass NLU pipeline)
if request.message.startswith(SERVICE_STEP_PREFIXES):
    logger.info(f"[{request.chatId}] #service prefix detected - direct step stream")
    executor = self._get_service_workflow_executor()
    step_stream = await executor.execute_direct_step_streaming(
        request=request,
        time_metric=time_metric,
    )
    if step_stream is not None:
        async for chunk in step_stream:
            yield chunk
        log_step_timings(time_metric, request.chatId)
        return
```

This happens **before** the `StreamManager` context, component initialization, guardrails, and classifier. The entire NLU pipeline is bypassed.

### 2.4 ServiceWorkflowExecutor — `execute_direct_step_streaming()`

**`service_workflow.py` → `execute_direct_step_streaming()`** (line 1065):

```python
async def execute_direct_step_streaming(self, request, time_metric=None):
    parsed = self._parse_service_prefix(request.message)
    if parsed is None:
        return None

    http_method, endpoint_url = parsed
    
    service_result = await self._call_service_endpoint(
        endpoint_url, http_method, entities_array=[], chat_id, author_id
    )
    if service_result is None:
        return None

    service_content = service_result["content"]
    service_buttons = service_result["buttons"]

    async def step_stream() -> AsyncIterator[str]:
        yield orchestration_service.format_sse(
            chat_id, service_content, service_buttons or None
        )
        yield orchestration_service.format_sse(chat_id, "END")

    return step_stream()
```

Again, exactly **2 SSE messages** are yielded.

---

## Deep Dive: Streaming Architecture

### End-to-End Token Flow

```
                              ┌─────────────────────────────────────────────────────────────────────────┐
                              │                     Python Backend (FastAPI)                            │
                              │                                                                         │
                              │  stream_orchestration_response()                                        │
                              │  ┌─────────────────────────────────────────┐                           │
                              │  │ For Service Workflows:                  │                           │
                              │  │                                         │                           │
                              │  │  yield format_sse(content, buttons)     │ ──▷ "data: {json}\n\n"   │
                              │  │  yield format_sse("END")                │ ──▷ "data: {json}\n\n"   │
                              │  │                                         │                           │
                              │  │ For RAG Workflows:                      │                           │
                              │  │  yield format_sse(token_1)              │ ──▷ "data: {json}\n\n"   │
                              │  │  yield format_sse(token_2)              │ ──▷ "data: {json}\n\n"   │
                              │  │  ...                                    │                           │
                              │  │  yield format_sse("END")                │ ──▷ "data: {json}\n\n"   │
                              │  └─────────────────────────────────────────┘                           │
                              └────────────────────────────┬──────────────────────────────────────────────┘
                                                           │ HTTP Response (chunked transfer encoding)
                                                           ▼
                              ┌──────────────────────────────────────────────────────────────────────────┐
                              │                    Notification Server (Node.js)                         │
                              │                                                                          │
                              │  streamingService.js                                                     │
                              │  ┌──────────────────────────────────────────────────┐                   │
                              │  │ 1. fetch("/orchestrate/stream", {body: payload}) │                   │
                              │  │ 2. response.body.getReader()                     │                   │
                              │  │ 3. Loop: read() → decode → parse "data:" lines  │                   │
                              │  │                                                  │                   │
                              │  │ For each parsed SSE line:                        │                   │
                              │  │   content = data.payload.content                 │                   │
                              │  │   buttons = data.payload.buttons                 │                   │
                              │  │                                                  │                   │
                              │  │   if content == "END":                           │                   │
                              │  │     sender({type:"stream_end"})                  │                   │
                              │  │   else:                                          │                   │
                              │  │     sender({type:"stream_chunk", content, btns}) │                   │
                              │  └──────────────────────────────────────────────────┘                   │
                              └────────────────────────────┬─────────────────────────────────────────────┘
                                                           │ SSE (EventSource)
                                                           ▼
                              ┌──────────────────────────────────────────────────────────────────────────┐
                              │                        Browser (React)                                   │
                              │                                                                          │
                              │  useStreamingResponse.tsx                                                │
                              │  ┌──────────────────────────────────────────────────┐                   │
                              │  │ eventSource.onmessage = (event) => {             │                   │
                              │  │   data = JSON.parse(event.data)                  │                   │
                              │  │                                                  │                   │
                              │  │   "stream_start" → setIsStreaming(true)           │                   │
                              │  │   "stream_chunk" → onToken(data.content)         │                   │
                              │  │                    onButtons(data.buttons)        │                   │
                              │  │   "stream_end"   → onComplete()                  │                   │
                              │  │   "stream_error" → onError(data.error)           │                   │
                              │  │ }                                                │                   │
                              │  └──────────────────────────────────────────────────┘                   │
                              │                                                                          │
                              │  TestProductionLLM/index.tsx                                             │
                              │  ┌──────────────────────────────────────────────────┐                   │
                              │  │ onToken: Append content to bot message           │                   │
                              │  │ onButtons: Attach buttons array to bot message   │                   │
                              │  │ onComplete: setIsLoading(false)                  │                   │
                              │  └──────────────────────────────────────────────────┘                   │
                              └──────────────────────────────────────────────────────────────────────────┘
```

### SSE Format Comparison

| Stage | Format |
|---|---|
| **Backend → Notification Server** | `data: {"chatId":"ch-1","payload":{"content":"text","buttons":[...]},"timestamp":"...","sentTo":[]}\n\n` |
| **Notification Server → Browser** | `data: {"type":"stream_chunk","content":"text","buttons":[...],"streamId":"ch-1","channelId":"ch-1","isComplete":false}\n` |

### Service vs RAG Streaming Behavior

| Aspect | Service Workflow | RAG Workflow |
|---|---|---|
| **Number of SSE chunks** | Exactly 2 (content+buttons, then END) | Many (token by token, then END) |
| **`buttons` field** | Present when MCQ | Never present |
| **Token granularity** | Full response in one chunk | Individual tokens (~200 chars buffered) |
| **Why?** | DMapper returns pre-formed text | LLM generates token by token |

### Connection Lifecycle

```
Browser                    Notification Server             Backend
   │                            │                             │
   │ GET /sse/stream/ch-1       │                             │
   │ ═══════════(SSE open)═══▷  │ Register connection         │
   │                            │                             │
   │ POST /channels/ch-1/       │                             │
   │   orchestrate/stream       │                             │
   │ ─────────────────────────▷ │                             │
   │                            │ POST /orchestrate/stream    │
   │                            │ ─────────────────────────▷  │
   │                            │                             │ async generator starts
   │                            │                             │ yield SSE chunk
   │      ◁═ stream_start ═══  │ ◁═══ data: {...}\n\n ═════ │
   │      ◁═ stream_chunk ═══  │ ◁═══ parse "data:" line ══  │
   │                            │                             │ yield END
   │      ◁═ stream_end ══════ │ ◁══ data: {...END}\n\n ═══  │
   │                            │                             │
   │ EventSource closes         │ Connection cleaned up       │
```

### Timeout & Error Handling

| Layer | Mechanism | Behavior |
|---|---|---|
| **API Layer** | `stream_timeout(MAX_STREAM_DURATION_SECONDS)` | If the entire generator takes too long, a `StreamTimeoutException` is raised and a timeout SSE message is sent |
| **API Layer** | Rate limiter | Per-user request/token limiting; returns 429 SSE error |
| **Orchestration** | `StreamManager.managed_stream()` | Tracks active streams, guarantees cleanup |
| **Notification Server** | `activeConnections.has(connectionId)` check | Stops reading if client disconnected |
| **Notification Server** | Stale request cleanup | Every 5 minutes, clears requests older than 1 hour |
| **Browser** | `eventSource.onerror` | Closes connection, calls `onError` |
| **Browser** | Component unmount cleanup | Calls `stopStreaming()`, closes EventSource |

---

## Complete MCQ Streaming Sequence

```
User              TestProductionLLM     useStreaming    Notif Server       Backend (stream)        ServiceWorkflow
  │                    │                   │               │                    │                      │
  │ types "keyboard    │                   │               │                    │                      │
  │  not working"      │                   │               │                    │                      │
  ├───────────────────▶│ handleSendMessage │               │                    │                      │
  │                    ├──────────────────▶│ startStreaming │                    │                      │
  │                    │                   │ GET /sse/     │                    │                      │
  │                    │                   │  stream/ch-1  │                    │                      │
  │                    │                   ├──────════════▶│ register SSE      │                      │
  │                    │                   │ (500ms wait)  │                    │                      │
  │                    │                   │ POST /channels│                    │                      │
  │                    │                   │  /ch-1/stream │                    │                      │
  │                    │                   ├──────────────▶│                    │                      │
  │                    │                   │               │ POST /orchestrate/ │                      │
  │                    │                   │               │  stream            │                      │
  │                    │                   │               ├───────────────────▶│                      │
  │                    │                   │               │                    │ startswith(#svc)?→NO │
  │                    │                   │               │                    │ classify()→SERVICE   │
  │                    │                   │               │                    ├─────────────────────▶│
  │                    │                   │               │                    │                      │ execute_streaming()
  │                    │                   │               │                    │                      │ _call_service_endpoint
  │                    │                   │               │                    │◁═ SSE: content+btns ═│ format_sse()
  │                    │                   │               │◁═ parse data: ═════│                      │
  │                    │                   │◁═ stream_chunk│                    │                      │
  │                    │◁═ onToken+onBtns ═│               │                    │◁═ SSE: END ══════════│
  │                    │                   │               │◁═ parse END ═══════│                      │
  │                    │                   │◁═ stream_end ═│                    │                      │
  │◁ render text +     │ onComplete        │               │                    │                      │
  │  [Windows] [Mac]   │                   │               │                    │                      │
  │                    │                   │               │                    │                      │
  │ clicks [Windows]   │                   │               │                    │                      │
  ├───────────────────▶│ handleButtonClick │               │                    │                      │
  │                    │ add "Windows" msg │               │                    │                      │
  │                    ├──────────────────▶│ startStreaming(payload)             │                      │
  │                    │                   │ ═══▷ ═══▷ ═══▷                     │                      │
  │                    │                   │               │ POST /orch/stream  │                      │
  │                    │                   │               ├───────────────────▶│                      │
  │                    │                   │               │                    │ startswith(#svc)?→YES│
  │                    │                   │               │                    │ SKIP everything      │
  │                    │                   │               │                    ├─────────────────────▶│
  │                    │                   │               │                    │                      │ execute_direct_step_
  │                    │                   │               │                    │                      │  streaming()
  │                    │                   │               │                    │                      │ _parse_service_prefix
  │                    │                   │               │                    │                      │ _call_service_endpoint
  │                    │                   │               │                    │◁═ SSE: content+btns ═│
  │                    │                   │               │◁═ parse ═══════════│                      │
  │                    │                   │◁═ stream_chunk│                    │◁═ SSE: END ══════════│
  │                    │                   │◁═ stream_end ═│                    │                      │
  │◁ render next MCQ   │ onToken+onBtns   │               │                    │                      │
  │  or final answer   │                   │               │                    │                      │
```

---

## Key Differences from TestModel Page

| Aspect | TestModel | TestProductionLLM |
|---|---|---|
| **Endpoint** | `/orchestrate/test` (non-streaming) | `/orchestrate/stream` (streaming) |
| **Protocol** | HTTP POST → JSON response | HTTP POST → SSE stream |
| **Relay** | Direct API call (no proxy) | Via Notification Server |
| **Environment** | `testing` (from connection) | `production` (hardcoded) |
| **Chat history** | None (stateless, single request) | Maintains `messages[]` array, sends `conversationHistory` |
| **Button click display** | Shows raw payload as message | Shows button **title** as user message |
| **Response delivery** | Full JSON at once | SSE chunks (but still 1 chunk for services) |
| **LLM connection** | User-selected from dropdown | No selection; uses production config |
| **Service response** | `content` + `buttons` in JSON body | `content` + `buttons` inside SSE `payload` |
| **Chunks/Context** | Shown in collapsible section | Not shown (streaming doesn't return chunks) |

---

## Streaming Payloads Reference

This section documents the **exact payload shape** at each boundary in the three-hop streaming chain for both streaming flows (natural-language service detection and MCQ button click).

---

### Hop 1 — Browser → Notification Server

**Endpoint:** `POST /channels/:channelId/orchestrate/stream`

The browser's `useStreamingResponse.tsx` sends an HTTP POST immediately after the SSE connection is established.

#### Natural-Language Query Payload
```json
{
  "message": "My keyboard is not working",
  "options": {
    "authorId": "user-channel-abc123",
    "conversationHistory": [
      {
        "role": "user",
        "content": "My keyboard is not working"
      }
    ],
    "url": "sse-stream-context",
    "connection_id": "conn-xyz"
  }
}
```

#### MCQ Button-Click Payload
```json
{
  "message": "#service, /POST/dmapper/v2/keyboard/os-select",
  "options": {
    "authorId": "user-channel-abc123",
    "conversationHistory": [
      {
        "role": "user",
        "content": "My keyboard is not working"
      },
      {
        "role": "assistant",
        "content": "Which operating system are you using?"
      },
      {
        "role": "user",
        "content": "Windows"
      }
    ],
    "url": "sse-stream-context",
    "connection_id": "conn-xyz"
  }
}
```

> **Key difference:** For MCQ button clicks, `message` contains the raw `#service, /METHOD/...` payload (not the button title shown to the user). The button title (`"Windows"`) is only stored in the local `messages` state for display purposes.

---

### Hop 2 — Notification Server → Backend

**Endpoint:** `POST /orchestrate/stream` (Python FastAPI)

`streamingService.js` maps the incoming request onto the `OrchestrationRequest` model:

#### Natural-Language Query Payload
```json
{
  "chatId": "channel-abc123",
  "message": "My keyboard is not working",
  "authorId": "user-channel-abc123",
  "conversationHistory": [
    {
      "role": "user",
      "content": "My keyboard is not working"
    }
  ],
  "url": "sse-stream-context",
  "environment": "production",
  "connection_id": "conn-xyz"
}
```

#### MCQ Button-Click Payload
```json
{
  "chatId": "channel-abc123",
  "message": "#service, /POST/dmapper/v2/keyboard/os-select",
  "authorId": "user-channel-abc123",
  "conversationHistory": [
    {
      "role": "user",
      "content": "My keyboard is not working"
    },
    {
      "role": "assistant",
      "content": "Which operating system are you using?"
    },
    {
      "role": "user",
      "content": "Windows"
    }
  ],
  "url": "sse-stream-context",
  "environment": "production",
  "connection_id": "conn-xyz"
}
```

> **`environment` is always `"production"`** for this page — it is hardcoded in `streamingService.js` and is not derived from user-selected connections.

---

### Hop 3 — Backend → Notification Server (SSE Stream)

The Python backend yields SSE-formatted strings. Each line follows the `data: {json}\n\n` format.

#### Message 1 — Service Response (with buttons)

Emitted by `format_sse(chat_id, content, buttons)`:

```
data: {
  "chatId": "channel-abc123",
  "payload": {
    "content": "Which operating system are you using?",
    "buttons": [
      { "title": "Windows", "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=windows" },
      { "title": "Mac",     "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=mac" },
      { "title": "Linux",   "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=linux" }
    ]
  },
  "timestamp": "1711512000000",
  "sentTo": []
}

```

#### Message 1 — Service Response (without buttons / final answer)

```
data: {
  "chatId": "channel-abc123",
  "payload": {
    "content": "To fix your keyboard on Windows, please try the following steps: ..."
  },
  "timestamp": "1711512001000",
  "sentTo": []
}

```

#### Message 2 — END Marker (always the final SSE frame)

Emitted by `format_sse(chat_id, "END")`:

```
data: {
  "chatId": "channel-abc123",
  "payload": {
    "content": "END"
  },
  "timestamp": "1711512001001",
  "sentTo": []
}

```

> **Service workflows always yield exactly 2 SSE frames:** the content frame (Message 1) and the END frame (Message 2). RAG workflows yield many content frames (one per token) before the END frame.

#### Error Frame (e.g. guardrail violation, rate limit, timeout)

```
data: {
  "chatId": "channel-abc123",
  "payload": {
    "content": "I'm sorry, I can only assist with topics related to e-government services."
  },
  "timestamp": "1711512001000",
  "sentTo": []
}

```

Followed immediately by the END frame.

---

### Hop 4 — Notification Server → Browser (SSE Stream)

After parsing each `data:` line from the backend, `streamingService.js` re-emits to the browser via its own SSE channel. The format changes: the outer `chatId`/`timestamp` wrapper is dropped and a `type` discriminator is added.

#### `stream_start` — Sent once, before any backend data

```json
{
  "type": "stream_start",
  "streamId": "channel-abc123",
  "channelId": "channel-abc123",
  "isComplete": false
}
```

#### `stream_chunk` — Sent once per parsed SSE frame (except END)

With buttons (MCQ step):
```json
{
  "type": "stream_chunk",
  "content": "Which operating system are you using?",
  "buttons": [
    { "title": "Windows", "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=windows" },
    { "title": "Mac",     "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=mac" },
    { "title": "Linux",   "payload": "#service, /POST/dmapper/v2/keyboard/os-select?os=linux" }
  ],
  "streamId": "channel-abc123",
  "channelId": "channel-abc123",
  "isComplete": false
}
```

Without buttons (final answer or RAG token):
```json
{
  "type": "stream_chunk",
  "content": "To fix your keyboard on Windows, please try the following steps: ...",
  "streamId": "channel-abc123",
  "channelId": "channel-abc123",
  "isComplete": false
}
```

#### `stream_end` — Sent when backend `payload.content === "END"`

```json
{
  "type": "stream_end",
  "streamId": "channel-abc123",
  "channelId": "channel-abc123",
  "isComplete": true
}
```

#### `stream_error` — Sent on fetch failure or uncaught exception in `streamingService.js`

```json
{
  "type": "stream_error",
  "error": "Failed to connect to LLM orchestration service",
  "streamId": "channel-abc123",
  "channelId": "channel-abc123",
  "isComplete": true
}
```

---

### Payload Shape Summary

| Hop | Direction | Content Key | Envelope |
|---|---|---|---|
| **1** | Browser → Notification Server | `message` (string) | `{ message, options: { authorId, conversationHistory, url, connection_id } }` |
| **2** | Notification Server → Backend | `message` (string) | `{ chatId, message, authorId, conversationHistory, url, environment, connection_id }` |
| **3** | Backend → Notification Server | `payload.content` (string), optional `payload.buttons` | `data: { chatId, payload, timestamp, sentTo }\n\n` |
| **4** | Notification Server → Browser | `content` (string), optional `buttons` | `data: { type, content?, buttons?, streamId, channelId, isComplete }\n` |

---

## Error Handling in Streaming Flow

| Error Scenario | Where Handled | Behavior |
|---|---|---|
| `_parse_service_prefix()` fails | `execute_direct_step_streaming()` | Returns `None` → falls through to normal pipeline |
| Service endpoint timeout/error | `_call_service_endpoint()` | Returns `None` → falls back to RAG |
| StreamTimeout | `timeout_wrapped_stream()` (API layer) | Sends SSE timeout message to client |
| Rate limit exceeded | `stream_orchestrated_response()` (API layer) | Sends SSE error with 429 status |
| Notification server no connections | `createLLMOrchestrationStreamRequest()` | Queues request or returns 404 |
| Browser disconnect mid-stream | `activeConnections.has()` check in loop | Stops reading response body |
| Component unmount during stream | `useEffect` cleanup in hook | Calls `stopStreaming()` → closes EventSource |
| Bot message incomplete on error | `onError` callback in component | Marks message with `hasError: true`, shows error indicator |

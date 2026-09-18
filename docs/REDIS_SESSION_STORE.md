# Redis Session Store — Usage Guide

The `APIToolSessionStore` provides simple async CRUD for persisting agentic loop state
across multiple HTTP requests, keyed by `chat_id` with a 30-minute sliding TTL.

---

## Accessing the Store

The store is available on `app.state` from any FastAPI endpoint or workflow that receives
the FastAPI `Request` object.

```python
session_store = request.app.state.session_store  # APIToolSessionStore | None
```

Always guard against `None` — the service starts even if Redis is down:

```python
if session_store is None:
    # Redis unavailable, handle gracefully (e.g. fall back, log warning)
    ...
```

---

## CRUD Operations

### Create — `save()`

Use `save()` to create a new session at the start of a multi-turn workflow.

```python
from src.models.session_models import APIToolSession

session = APIToolSession(
    chat_id=request.chatId,
    state="collecting_params",
    selected_endpoint={
        "url": "https://api.example.com/weather",
        "method": "GET",
        "params_schema": [{"name": "city", "required": True}],
    },
    collected_params={},
    turn_count=1,
    max_turns=5,
)

await session_store.save(session)
```

`save()` also resets the TTL — so calling it again later also acts as a keep-alive.

---

### Read — `get()`

Use `get()` to load an existing session. Returns `None` if the session does not exist
or has expired.

```python
session = await session_store.get(request.chatId)

if session is None:
    # No active session → this is a fresh conversation
    ...
else:
    print(session.state)  # "collecting_params"
    print(session.collected_params)  # {"city": "Tallinn"}
    print(session.turn_count)  # 2
```

---

### Update — `update()`

Use `update()` for partial changes — only the fields you pass are modified.
All other fields are preserved. The TTL is reset automatically.

```python
# Add a newly collected param and increment the turn counter
updated_session = await session_store.update(
    request.chatId,
    collected_params={"city": "Tallinn"},
    turn_count=session.turn_count + 1,
)

# Change only the state
await session_store.update(request.chatId, state="ready")
```

Returns the updated `APIToolSession`, or `None` if the session was not found.

---

### Delete — `delete()`

Use `delete()` once the workflow completes (all params collected, API called, or user
abandoned the flow).

```python
await session_store.delete(request.chatId)
```

---

### Check Existence — `exists()`

Use `exists()` when you only need to know whether a session is active, without loading it.

```python
if await session_store.exists(request.chatId):
    # Resume existing session
    ...
else:
    # Start a new one
    ...
```

---

## Typical Multi-Turn Pattern

```python
session_store = request.app.state.session_store

# --- Turn 1 ---
# No session yet → detect endpoint, create session, ask for missing params
session = await session_store.get(request.chatId)
if session is None:
    detected_endpoint = ...  # endpoint detected from user query
    await session_store.save(
        APIToolSession(
            chat_id=request.chatId,
            state="collecting_params",
            selected_endpoint=detected_endpoint,
            turn_count=1,
        )
    )
    return "Which city would you like weather for?"

# --- Turn 2+ ---
# Session exists → merge new params, check if ready
session = await session_store.get(request.chatId)

# Guard: abandon if turn limit reached
if session.turn_count >= session.max_turns:
    await session_store.delete(request.chatId)
    return "I was unable to collect all required information. Please try again."

new_params = extract_params_from_message(request.message, session)
merged_params = {**session.collected_params, **new_params}

if all_params_collected(session.selected_endpoint, merged_params):
    # Call the actual API
    result = await call_api(session.selected_endpoint, merged_params)
    await session_store.delete(request.chatId)
    return result
else:
    # Still missing params — update and ask again
    await session_store.update(
        request.chatId,
        collected_params=merged_params,
        turn_count=session.turn_count + 1,
    )
    return ask_for_next_missing_param(session.selected_endpoint, merged_params)
```

---

## Session Schema Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `chat_id` | `str` | required | Unique conversation ID |
| `state` | `str` | required | Current state (`"collecting_params"`, `"ready"`, `"completed"`) |
| `selected_endpoint` | `dict \| None` | `None` | The API endpoint to call |
| `collected_params` | `dict` | `{}` | Parameters gathered so far |
| `turn_count` | `int` | `0` | Number of turns elapsed |
| `max_turns` | `int` | `5` | Abandon session after this many turns |

**Redis key:** `session:{chat_id}` — **TTL:** 30 minutes, sliding (reset on every `save`/`update`)

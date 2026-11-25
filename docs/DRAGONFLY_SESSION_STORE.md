# Dragonfly Session Store — Usage Guide

The `APIToolSessionStore` provides simple async CRUD for persisting agentic loop state
across multiple HTTP requests, keyed by `chat_id` with a 30-minute sliding TTL.

> **Backend:** all stores run on **Dragonfly** (`dragonfly:6379`) and connect through
> `src/utils/dragonfly_client.py`, configured by the `DRAGONFLY_*` settings below. Dragonfly
> speaks the Redis wire protocol, so that module uses the redis-py client library; it is
> the only app module that imports it.

---

## Backend and keyspace

| | Value |
|---|---|
| Server | Dragonfly `v2.0.2` (Compose service and Helm chart `dragonfly`) |
| App database | **db 1** (`DRAGONFLY_SESSION_DB`; an empty value also falls back to 1) |
| Langfuse | **db 0** (its BullMQ job queues); never shared with app keys |
| Optional key prefix | `DRAGONFLY_KEY_PREFIX` (default empty). Only needed if the app ever has to share db 0 with Langfuse (Dragonfly cluster mode). |

## Keys and TTLs

All app keys go through `dragonfly_key()` in `src/utils/dragonfly_client.py`, which prepends
`DRAGONFLY_KEY_PREFIX`.

| Key | Store | Content | TTL |
|---|---|---|---|
| `session:{chat_id}` | `APIToolSessionStore` | API tool-calling session | 30 min, sliding (reset on `save`/`update`) |
| `conv:{chat_id}` | `ConversationHistoryStore` | Last 10 conversation rounds | 30 min, sliding (reset on `save_round`) |
| `conv:summary:{chat_id}` | `ConversationHistoryStore` | Incremental summary of older rounds | 30 min, kept in sync with `conv:` |
| `atc:cache:{chat_id}:{api_name}:{param_hash}` | `ATCCacheStore` | L1 cached API response | 30 min (`ATC_CACHE_DEFAULT_TTL_SECONDS`) |
| `atc:last:{chat_id}` | `ATCCacheStore` | L2 last-call context for follow-ups | 30 min, sliding |

Every key expires, so the store never grows without bound. Dragonfly never evicts
(`noeviction`): if `maxmemory` is reached it refuses writes, which the
`dragonfly-memory-*` alerts warn about.

## Client settings

Set in `.env` / `env.example` and read by `src/utils/dragonfly_client.py`:

| Variable | Default | Purpose |
|---|---|---|
| `DRAGONFLY_HOST` | `dragonfly` | Store hostname |
| `DRAGONFLY_PORT` | `6379` | Store port |
| `DRAGONFLY_PASSWORD` | — | Password (Kubernetes: Secret `dragonfly-secrets`, key `DRAGONFLY_PASSWORD`) |
| `DRAGONFLY_SESSION_DB` | `1` | App database |
| `DRAGONFLY_KEY_PREFIX` | empty | Optional prefix for all app keys |
| `DRAGONFLY_CONNECT_TIMEOUT` | `2` (seconds) | Give up connecting after this long |
| `DRAGONFLY_SOCKET_TIMEOUT` | `3` (seconds) | Give up waiting for a reply after this long |
| `DRAGONFLY_MAX_CONNECTIONS` | `100` | Connection pool limit |
| `DRAGONFLY_TLS_*` | TLS off | Optional TLS and client certificates |

The client also uses `health_check_interval=30` and `retry_on_timeout=True` (one retry).
Invalid or empty numeric values fall back to the defaults with a warning.

## Fail-open behaviour

The stores never let a store problem break a chat. Every method catches errors
(timeouts, lost connections, `OOM` write rejections) and returns a safe fallback:

| Store | Read methods return | Write methods |
|---|---|---|
| `APIToolSessionStore` | `get` / `update` → `None`, `exists` → `False` | `save`, `delete` → log and continue |
| `ConversationHistoryStore` | `get_history` → `[]`, `get_summary` → `None` | `save_round`, `save_summary` → log and continue |
| `ATCCacheStore` | `get_l1` / `get_l2` → `None` (cache miss) | `set_l1`, `set_l2`, `invalidate_l2` → log and continue |

- If the store is unreachable at startup, the service still starts with the stores set
  to `None`; chats work without history, sessions or cache.
- With the timeouts above, a hung store adds at most a few seconds to a request instead
  of blocking it.
- `/health` reports `dragonfly_session_store` as `connected`, `disconnected` or
  `not_configured`, and logs a warning when it is not connected (Grafana alert
  "LLM service lost its Dragonfly connection").
- These behaviours are covered by `tests/test_store_failure_modes.py`.

---

## Accessing the Store

The store is available on `app.state` from any FastAPI endpoint or workflow that receives
the FastAPI `Request` object.

```python
session_store = request.app.state.session_store  # APIToolSessionStore | None
```

Always guard against `None` — the service starts even if Dragonfly is down:

```python
if session_store is None:
    # Dragonfly unavailable, handle gracefully (e.g. fall back, log warning)
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

**Key:** `session:{chat_id}` in db 1 (with `DRAGONFLY_KEY_PREFIX` if set) — **TTL:** 30 minutes, sliding (reset on every `save`/`update`)

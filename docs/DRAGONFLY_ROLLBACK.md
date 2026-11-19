# Dragonfly → Redis Rollback Procedure

How to switch the Redis-compatible store back from **Dragonfly** to **Redis 7** if
Dragonfly causes problems after the migration. The Redis service and Helm chart are kept
(disabled) for one release so this is a configuration switch, not a rebuild.

**Related:** [DRAGONFLY_OPERATIONS.md](./DRAGONFLY_OPERATIONS.md) (runbook) ·
[REDIS_SESSION_STORE.md](./REDIS_SESSION_STORE.md) (keys, settings, fail-open)

---

## When to roll back

Roll back when a Dragonfly problem cannot be fixed quickly, for example:

- Langfuse jobs stop draining ("Langfuse queue backlog not draining" alert) and restarting
  the worker does not help, or the worker logs Lua / `undeclared key` errors
- Dragonfly keeps restarting or being OOM-killed ("Dragonfly restarted" alert)
- The app keeps losing its store ("LLM service lost its session store" alert)

Memory alerts alone (`dragonfly-memory-*`) are not a reason to roll back: raise
`maxmemory` and the memory limit instead (see the runbook).

## What is lost

| Data | Effect |
|---|---|
| App sessions, conversation history, API cache | Lost on the switch (they live ≤30 min anyway). Chats in progress lose their context once; nothing breaks. |
| Langfuse jobs still queued in Dragonfly | Lost unless drained first (step 1). Already-ingested traces are safe in ClickHouse. |
| Dragonfly's own data | Kept in its volume (`dragonfly_data` / PVC `data-dragonfly-0`), so rolling forward again is possible |

---

## Docker Compose (local, EC2)

1. **Drain Langfuse** (skip if Langfuse is already broken):
   ```bash
   docker compose stop langfuse-web
   # wait until no Langfuse jobs are waiting or active:
   PW=$(grep '^REDIS_AUTH=' .env | cut -d= -f2-)
   docker compose exec -T dragonfly redis-cli --no-auth-warning -a "$PW" --scan --pattern 'bull:*:wait'
   docker compose exec -T dragonfly redis-cli --no-auth-warning -a "$PW" --scan --pattern 'bull:*:active'
   docker compose stop langfuse-worker
   ```
   No output from both scans = nothing left to process (empty lists are deleted).

2. **Edit `docker-compose.yml`** (`docker-compose-ec2.yml` on EC2):
   - Uncomment the `redis:` service block (under the "Old Redis service" comment).
   - In the `&langfuse-depends-on` block, change `dragonfly:` to `redis:`
     (keep `condition: service_healthy`).
   - In `llm-orchestration-service` → `depends_on`, change `- dragonfly` to `- redis`.

3. **Edit `.env`:** `REDIS_HOST=redis`. Keep `REDIS_AUTH` as it is (Redis uses the same
   variable for its password).

4. **Stop Dragonfly** (Redis needs port 6379), then start everything:
   ```bash
   docker compose stop dragonfly
   docker compose up -d
   ```
   Do not use `docker compose down -v`: it would delete the `dragonfly_data` volume.

5. **Verify:**
   ```bash
   docker compose ps redis langfuse-worker langfuse-web llm-orchestration-service
   curl -s http://localhost:8100/health
   ```
   `redis` is `healthy`, `/health` shows `"redis_session_store":"connected"`, and a test
   chat produces a trace in Langfuse.

The monitoring jobs (`langfuse_queue_depth`, `dragonfly_metrics`) read `REDIS_HOST` from
`.env`, so they follow the switch automatically.

---

## Kubernetes (Helm)

1. **Drain Langfuse:** scale `langfuse-web` to 0, wait until the queue-depth logs show
   `waiting_total: 0` and `active_total: 0`, then scale `langfuse-worker` to 0.

2. **Change values:**

   | File | Change |
   |---|---|
   | `kubernetes/values.yaml` | `Redis.enabled: true`, `Dragonfly.enabled: false` |
   | `charts/Langfuse-Web/values.yaml`, `charts/Langfuse-Worker/values.yaml` | `REDIS_HOST: "redis"` |
   | `charts/LLM-Orchestration-Service/values.yaml` | `REDIS_HOST: "redis"`; `redisAuth.secretName: "redis-secrets"`, `redisAuth.secretKey: "REDIS_PASSWORD"` |

   Make sure `redis-secrets` (`REDIS_PASSWORD`) and the Langfuse secrets (`REDIS_AUTH`)
   hold the **same** password.

3. **Deploy:** `helm upgrade` (or sync the ArgoCD app) with the changed values.
   The Langfuse init containers wait for `redis:6379` automatically, because they use
   `REDIS_HOST`.

4. **Verify:** the `redis` pod is Running, `/health` shows `connected`, and a test
   chat produces a trace.

The Dragonfly PVC (`data-dragonfly-0`) is not deleted by `helm` and keeps its data.

---

## Rolling forward again

Reverse the same changes (Compose: comment Redis out, `depends_on` back to `dragonfly`,
`REDIS_HOST=dragonfly`, `docker compose stop redis`, `up -d`; Kubernetes: swap the
`enabled` flags and values back). Dragonfly loads its last snapshot from its volume.

## Rehearsal record

| Date | Environment | What was rehearsed | Result |
|---|---|---|---|
| 2026-10-20 | minikube (`df-test`) | `helm uninstall` Dragonfly chart → `helm install` Redis chart; Dragonfly PVC retained | Passed (#743) |

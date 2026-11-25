# Dragonfly Operations Runbook

Day-to-day operation of **Dragonfly**, the in-memory store behind the LLM
service's sessions, conversation history and API response cache (db 1) and Langfuse's
job queues (db 0).

**Related:** [DRAGONFLY_SESSION_STORE.md](./DRAGONFLY_SESSION_STORE.md) (keys, client settings,
fail-open)

---

## Where it is configured

| Environment | Service / chart | Settings |
|---|---|---|
| Local / EC2 | `dragonfly` in `docker-compose.yml` / `docker-compose-ec2.yml` | `.env` (see `env.example`) |
| CI | `dragonfly` in `docker-compose-test.yml` / `docker-compose-eval.yml` | Fixed test values in the files |
| Kubernetes | `kubernetes/charts/Dragonfly` (StatefulSet + PVC) | `values.yaml`; password in Secret `dragonfly-secrets` |

Image: `ghcr.io/dragonflydb/dragonfly:v2.0.2` (Docker Hub only has v1.x).
Langfuse is pinned to `3.225.2`, the version tested on Dragonfly.

## Flags and why they are set

| Flag | Value | Why |
|---|---|---|
| `--requirepass` | `DRAGONFLY_PASSWORD` | Password for every client |
| `--proactor_threads` | `DRAGONFLY_THREADS` / `threads` (**required**) | Without it Dragonfly starts one thread per CPU it sees, which can be the whole host, and run out of memory |
| `--maxmemory` | `DRAGONFLY_MAXMEMORY` / `maxmemory` (**required**) | Data limit. Dragonfly never evicts (`noeviction`): above it, writes are refused |
| `--default_lua_flags=allow-undeclared-keys` | — | "Option A": Langfuse's BullMQ Lua scripts use keys they don't declare. Every script then takes a global lock, which is safe but slower under heavy load |
| `--dir=/data --dbfilename=dump --snapshot_cron=*/15 * * * *` | — | Snapshot every 15 min to the volume, so data survives restarts |

Compose refuses to start and `helm template` fails if the required values are missing.
The CI check `.github/scripts/check_dragonfly_config.py` enforces the same rules and
that Dragonfly and Langfuse images are pinned.

## Sizing rule

1. **Threads:** start with 2. Each thread needs about **256 MiB**.
2. **Memory limit** (`DRAGONFLY_MEM_LIMIT` / `resources.limits.memory`): at least
   `threads × 256 MiB` plus headroom.
3. **maxmemory:** about **80%** of the memory limit (the rest is Dragonfly's own overhead).
4. **CPU request** (Kubernetes) = `threads` (the chart sets this automatically).

Current values: local Compose 2 threads / `1gb` / `1280m`; Kubernetes chart 2 threads /
`800mb` / `1Gi`; CI 2 threads / `512mb`. Resize production from real usage on the
"Dragonfly" dashboard: keep peak memory below ~60% of `maxmemory`.

## Monitoring

| What | Where |
|---|---|
| Dashboard | Grafana → Dashboards → RAG Module → **Dragonfly** (memory %, used vs max, clients, ops/sec, keys, uptime, Langfuse backlog) |
| Raw metrics | Loki: `{service="dragonfly-metrics"}` (every minute, CronManager job `dragonfly_metrics`) |
| Langfuse queue depth | Loki: `{service="langfuse-queue-depth"}` (every minute, CronManager job `langfuse_queue_depth`) |
| Alerts | Grafana → Alerting → Alert rules → *RAG Module Alerts* (`grafana-configs/grafana-alerting.yaml`) |

The CronManager jobs read `DRAGONFLY_HOST`/`DRAGONFLY_PASSWORD` from the mounted `.env`. Langfuse's
own container logs are **not** shipped to Loki, so Langfuse Lua errors only show up
indirectly through the queue-backlog alert.

### Alert responses

| Alert | First steps |
|---|---|
| **Dragonfly memory above 70% / 85%** | Check the dashboard for steady growth vs a spike. Raise `maxmemory` **and** the memory limit together (keep the 80% rule), then restart Dragonfly. At 85% act now: writes fail at 100%. |
| **Dragonfly restarted** | If not a deploy: check for OOM kills (`docker inspect dragonfly --format '{{.State.OOMKilled}}'` / `kubectl describe pod dragonfly-0`) and the Dragonfly logs. Repeated OOM kills mean the memory limit is too low for the thread count: see **Sizing rule**. |
| **Langfuse queue backlog not draining** | Check `langfuse-worker` is running and its logs; check Dragonfly CPU. Restart the worker. If it persists, suspect the global Lua lock under load (see `--default_lua_flags` above) and raise it with the team. |
| **Dragonfly errors in the app** | Check Dragonfly is up and healthy and the password matches. The app keeps answering without history/cache. |
| **Dragonfly update retries exhausted** | Concurrent writes to one chat kept conflicting; usually a client retrying the same chat rapidly or a slow store. Check store latency. |
| **Dragonfly is out of memory** | Dragonfly reached `maxmemory` and refuses writes: raise memory (see above) immediately. |
| **LLM service lost its Dragonfly connection** | `curl localhost:8100/health`; check Dragonfly is running, `DRAGONFLY_HOST`/`DRAGONFLY_PASSWORD`. The app reconnects by itself once Dragonfly is back. |

## Common commands (Compose)

`redis-cli` is the command-line client that ships inside the Dragonfly image.

```bash
PW=$(grep '^DRAGONFLY_PASSWORD=' .env | cut -d= -f2-)
DF="docker compose exec -T dragonfly redis-cli --no-auth-warning -a $PW"

$DF ping                                   # PONG = up and password correct
$DF INFO server | grep dragonfly_version   # confirm it is Dragonfly
$DF INFO memory | grep -E "used_memory_human|maxmemory_human"
$DF INFO keyspace                          # db0 = Langfuse, db1 = app
$DF -n 1 --scan --pattern 'conv:*'         # app keys
$DF TTL <key>                              # remaining lifetime (≤1800)
$DF --scan --pattern 'bull:*:wait'         # Langfuse jobs waiting (none = drained)
```

Kubernetes: the same with `kubectl -n <ns> exec dragonfly-0 -- redis-cli -a <password> ...`.

## Snapshots, backup and restore

- Dragonfly writes a snapshot to `/data` every 15 minutes **and on a clean shutdown**
  (restart, redeploy), and loads it on start ("Load snapshot" in the logs). Only a crash
  can lose up to 15 minutes of data.
- Force a snapshot now: `$DF SAVE`.
- **Backup (Compose):**
  ```bash
  docker run --rm -v dragonfly_data:/data:ro -v "$PWD:/backup" alpine \
    tar czf /backup/dragonfly_data-$(date +%F).tgz -C /data .
  ```
- **Restore:** stop Dragonfly, extract the backup into the volume, start Dragonfly.
- Most data is short-lived (≤30 min TTLs), so backups mainly matter for in-flight
  Langfuse jobs.

## Upgrades

**Langfuse** (Dragonfly support is only "community-supported", so test every version):
1. Change the pinned `3.x.y` tag in a local spike override and run the stack.
2. Check: traces arrive, `bull:*:wait` drains, no Lua / `undeclared key` errors in the
   worker or Dragonfly logs.
3. Update all pins together: `docker-compose*.yml` (4 files) and the Langfuse-Web /
   Langfuse-Worker chart values.

**Dragonfly:** pick a release from GitHub, test the same way plus the minikube chart test
(probes, threads, restart, data persistence), then update the tag in the 4 compose files
and `charts/Dragonfly/values.yaml`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `docker compose` fails: `DRAGONFLY_MAXMEMORY must be set in .env` | `.env` is missing the new settings: copy them from `env.example` |
| `docker compose` fails: `DRAGONFLY_PASSWORD must be set in .env` | `.env` still uses the old `REDIS_*` names: rename them to the `DRAGONFLY_*` names in `env.example` |
| Dragonfly won't start: port 6379 already in use | Another container is using the port: find it with `docker ps`, then `docker stop` and `docker rm` it |
| App keys appear in db0 next to `bull:*` | `DRAGONFLY_SESSION_DB` is set to `0`; it must be `1` (empty also falls back to 1) |
| Langfuse can't connect (auth errors) | Kubernetes: Langfuse's own secrets (`REDIS_AUTH`, Langfuse's setting name) must hold the same value as `DRAGONFLY_PASSWORD` in `dragonfly-secrets` |
| `/health` shows `not_configured` | The store was unreachable at startup; the app is running without sessions. Fix Dragonfly, then restart the LLM service. |

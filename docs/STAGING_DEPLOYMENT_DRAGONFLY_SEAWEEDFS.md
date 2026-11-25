# Staging Deployment Guide: Dragonfly and SeaweedFS Migration

How to deploy the `llm/dragonfly-and-seaweedfs-migration` changes to **staging (Kubernetes / Helm)**
step by step. Each step has a check (a "gate") to pass before moving on, and there's a rollback path.

| | Before | After |
|---|---|---|
| In-memory store | Redis 7 (`redis`) | Dragonfly `v2.0.2` (`dragonfly`, StatefulSet + PVC) |
| Object storage | MinIO (`minio`, ports 9000/9001) | SeaweedFS `4.47` (`seaweedfs`, S3 on 8333, filer on 8888) |
| Langfuse | Floating tag `3` | Pinned `3.225.2` |
| S3-Ferry | `PRE-ALPHA-1.1.1` | `v1.1.3` (reads `S3_DEFAULT_*`) |
| App settings | `REDIS_*` | `DRAGONFLY_*` |

What uses what:

- **Dragonfly db 0:** Langfuse job queues (`bull:*`).
- **Dragonfly db 1:** the LLM service's data: `session:`, `conv:` and `atc:` keys, all with a 30-minute TTL.
- **SeaweedFS bucket `rag-search`:** Langfuse events and media, the DVC cache and `processed-metadata.json`.

**Related:**
- [DRAGONFLY_OPERATIONS.md](./DRAGONFLY_OPERATIONS.md): runbook, flags, sizing and alert responses
- [DRAGONFLY_SESSION_STORE.md](./DRAGONFLY_SESSION_STORE.md): keys and fail-open behaviour

---

## 0. Read first: known gaps

Check these before you schedule the window. Items 1–3 can cause data loss or a broken deployment if ignored.

| # | Gap | What to do |
|---|---|---|
| 1 | **The MinIO PVC is deleted on upgrade.** This branch removes the `minio` chart. `pvc-minio-data` has no `helm.sh/resource-policy: keep`, so `helm upgrade` deletes it along with all objects in it. | Back up the bucket and annotate the PVC before upgrading ([step 2.4](#24-back-up-minio-and-keep-its-pvc)). |
| 2 | **The Redis and MinIO charts no longer exist on this branch.** Plan item 7.6 ("keep the subchart disabled but deployable") can't be followed as written. | Rollback means returning to the previous Helm revision ([section 7](#7-rollback-77)). Keep that revision, the pre-migration secrets and the MinIO backup until decommission ([section 8](#8-decommission-76)). |
| 3 | **Secrets are `<SET_ME>` placeholders in the chart templates**, and the Dragonfly password lives in **three** Secrets that must match (see [section 1](#1-passwords-keys-and-values)). | Set them with `kubectl` after `helm upgrade` ([step 4.1](#41-upgrade-and-set-the-secrets)). **Check them again after every `helm upgrade`**, because Helm can reset them to `<SET_ME>`. |
| 4 | **Dragonfly monitoring isn't set up on Kubernetes.** The Grafana chart doesn't load the Dragonfly dashboard or alert rules (only Compose does). The CronManager jobs `dragonfly_metrics` and `langfuse_queue_depth` can't get the Dragonfly password on Kubernetes, and CronManager clones its scripts from `rootcodelabs/RAG-Module` `v1.0.0`. | Hypercare uses the manual checks and Loki queries in [section 6](#6-hypercare-48-h-75) instead of the "Phase 5" dashboards and alerts. |

---

## 1. Passwords, keys and values

Prepare these before the window and keep them in your secret store. **Never commit real values.**

```bash
export DF_PW=$(openssl rand -hex 24)        # Dragonfly password (hex: no characters that need escaping)
export S3_KEY=$(openssl rand -hex 12)       # SeaweedFS access key
export S3_SECRET=$(openssl rand -hex 24)    # SeaweedFS secret key
```

| What | Where | Value |
|---|---|---|
| Dragonfly password | `dragonfly-secrets` → `DRAGONFLY_PASSWORD` (Dragonfly and the LLM service read it) | `$DF_PW` |
| | `langfuse-web-secrets` → `REDIS_AUTH` | **same** `$DF_PW` |
| | `langfuse-worker-secrets` → `REDIS_AUTH` | **same** `$DF_PW` |
| SeaweedFS S3 login | `--set seaweedfs.auth.accessKey=… --set seaweedfs.auth.secretKey=…` | `$S3_KEY` / `$S3_SECRET` |
| Langfuse S3 keys | `langfuse-web-secrets` **and** `langfuse-worker-secrets` → `LANGFUSE_S3_{EVENT_UPLOAD,MEDIA_UPLOAD,BATCH_EXPORT}_{ACCESS_KEY_ID,SECRET_ACCESS_KEY}` | **same** as the SeaweedFS keys |
| S3-Ferry S3 keys | `s3-ferry-secrets` → `S3_DEFAULT_ACCESS_KEY_ID`, `S3_DEFAULT_SECRET_ACCESS_KEY` | **same** as the SeaweedFS keys |
| Indexer / DVC S3 settings | Wherever staging provides `S3_ENDPOINT_URL`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` | `http://seaweedfs:8333` and the SeaweedFS keys |
| Dragonfly sizing | `Dragonfly.threads`, `Dragonfly.maxmemory`, `Dragonfly.resources.limits.memory` | Defaults 2 / `800mb` / `1Gi`. Keep `maxmemory` at about 80% of the limit (see the [sizing rule](./DRAGONFLY_OPERATIONS.md#sizing-rule)). |
| SeaweedFS volume | `seaweedfs.volumes.data.size` | Default `20Gi`. Must hold the MinIO data plus growth. |

Every pod reads these values only when it starts. After changing any of them, restart the pods
that use them (Dragonfly, `langfuse-*`, `llm-orchestration-service`, `s3-ferry`).

---

## 2. Preparation (the day before)

### 2.1 Build the chart

```bash
helm dependency build ./kubernetes
helm template stg ./kubernetes -f <staging-values.yaml> > /tmp/render.yaml   # must succeed
grep -c "name: \"dragonfly\"" /tmp/render.yaml                               # Dragonfly is in it
```

`helm template` fails if `Dragonfly.threads` or `Dragonfly.maxmemory` is empty. That's intended.

`Chart.yaml` depends on `charts/TIM-database`, which isn't in this repo. Build from wherever
staging's pipeline gets that chart, or `helm dependency build` fails with `TIM-database not found`.

### 2.2 Record the rollback point

```bash
export NS=<staging-namespace> REL=<helm-release>
helm -n $NS history $REL          # note the current REVISION: this is the rollback target
helm -n $NS get values $REL -o yaml > pre-migration-values.yaml
kubectl -n $NS get secret redis-secrets minio-secrets langfuse-web-secrets \
  langfuse-worker-secrets s3-ferry-secrets -o yaml > pre-migration-secrets.yaml
```

Store `pre-migration-secrets.yaml` somewhere safe and outside git: it contains real secrets.
A rollback needs them, because the Langfuse secrets will hold the new Dragonfly and SeaweedFS
values, and `redis-secrets` and `minio-secrets` come back as `<SET_ME>`.

### 2.3 Record a baseline

From the soak or current staging run, note these so hypercare has something to compare against:
- LLM service p99 latency
- Langfuse ingestion lag
- Redis memory use
- the error rate

### 2.4 Back up MinIO and keep its PVC

```bash
# Keep the PVC when the minio chart is removed (Helm checks this annotation on the live object)
kubectl -n $NS annotate pvc pvc-minio-data helm.sh/resource-policy=keep

# Copy the whole bucket to local disk
kubectl -n $NS port-forward svc/minio 9000:9000 &
export RCLONE_CONFIG_MINIO_TYPE=s3 RCLONE_CONFIG_MINIO_PROVIDER=Minio \
       RCLONE_CONFIG_MINIO_ENDPOINT=http://localhost:9000 \
       RCLONE_CONFIG_MINIO_ACCESS_KEY_ID=<MINIO_ROOT_USER> \
       RCLONE_CONFIG_MINIO_SECRET_ACCESS_KEY=<MINIO_ROOT_PASSWORD>
rclone sync minio:rag-search ./minio-backup/rag-search --progress
rclone size minio:rag-search          # note object count and size
rclone size ./minio-backup/rag-search # must match
```

Run the sync again right before the upgrade (step 4.1) to pick up the latest objects.

---

## 3. Maintenance window and drain (7.2, 7.3)

### 3.1 Announce the window (7.2)

Pick a low-traffic window and tell users that **chats in progress may lose their context**.
Sessions and history live in the store with a 30-minute TTL and are not migrated.

### 3.2 Drain Langfuse on the old Redis (7.3)

1. **Pause trace ingestion:** `kubectl -n $NS scale deploy/langfuse-web --replicas=0`
2. **Wait until no jobs are waiting or active.** Repeat until both commands print nothing (BullMQ deletes empty lists):
   ```bash
   RCLI='redis-cli -a "$REDIS_PASSWORD" --no-auth-warning'
   kubectl -n $NS exec deploy/redis -- sh -c "$RCLI --scan --pattern 'bull:*:wait'"
   kubectl -n $NS exec deploy/redis -- sh -c "$RCLI --scan --pattern 'bull:*:active'"
   ```
3. **Stop the worker:** `kubectl -n $NS scale deploy/langfuse-worker --replicas=0`

Traces that were already processed are safe in ClickHouse. Only jobs still in the queue would be lost.

---

## 4. Deploy and gated rollout (7.4)

### 4.1 Upgrade and set the secrets

1. Run the MinIO sync from 2.4 one last time.
2. Upgrade with both Langfuse deployments held at 0, so they only start at their gates:

```bash
helm -n $NS upgrade $REL ./kubernetes -f <staging-values.yaml> --timeout 10m \
  --set seaweedfs.auth.accessKey=$S3_KEY --set seaweedfs.auth.secretKey=$S3_SECRET \
  --set Langfuse-Worker.replicas=0 --set Langfuse-Web.replicas=0
```

`helm upgrade` waits for the SeaweedFS bucket Job (a post-upgrade hook), which retries until
SeaweedFS is up. `--timeout 10m` gives it room on a slow first start.

3. Set the Dragonfly password in all three Secrets, and the S3 keys in the Langfuse and S3-Ferry Secrets:

```bash
kubectl -n $NS patch secret dragonfly-secrets --type merge \
  -p "{\"stringData\":{\"DRAGONFLY_PASSWORD\":\"$DF_PW\"}}"

for s in langfuse-web-secrets langfuse-worker-secrets; do
  kubectl -n $NS patch secret $s --type merge -p "{\"stringData\":{
    \"REDIS_AUTH\":\"$DF_PW\",
    \"LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID\":\"$S3_KEY\",  \"LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY\":\"$S3_SECRET\",
    \"LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID\":\"$S3_KEY\",  \"LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY\":\"$S3_SECRET\",
    \"LANGFUSE_S3_BATCH_EXPORT_ACCESS_KEY_ID\":\"$S3_KEY\", \"LANGFUSE_S3_BATCH_EXPORT_SECRET_ACCESS_KEY\":\"$S3_SECRET\"}}"
done

kubectl -n $NS patch secret s3-ferry-secrets --type merge -p "{\"stringData\":{
  \"S3_DEFAULT_ACCESS_KEY_ID\":\"$S3_KEY\", \"S3_DEFAULT_SECRET_ACCESS_KEY\":\"$S3_SECRET\"}}"
```

Check that none of them still hold `<SET_ME>`:

```bash
for s in dragonfly-secrets langfuse-web-secrets langfuse-worker-secrets s3-ferry-secrets; do
  kubectl -n $NS get secret $s -o json \
    | jq -r --arg s "$s" '.data | to_entries[] | select((.value|@base64d)=="<SET_ME>") | "\($s): \(.key) still <SET_ME>"'
done
```

The only lines it may print are for keys outside this migration that staging already manages,
such as `DATABASE_URL`. If it prints any of the keys set above, patch them again.

4. Restart Dragonfly and S3-Ferry so they pick up the new values:

```bash
kubectl -n $NS rollout restart statefulset/dragonfly deploy/s3-ferry
```

The LLM service has already rolled with its new settings, but it may have started before the
password was set. Gate 4 restarts it.

**Check every gate before moving on. If any gate fails, go to [section 7](#7-rollback-77).**

### Gate 1: Dragonfly is ready and an authenticated PING works

```bash
kubectl -n $NS rollout status statefulset/dragonfly
kubectl -n $NS exec dragonfly-0 -- redis-cli -a "$DF_PW" --no-auth-warning ping             # PONG
kubectl -n $NS exec dragonfly-0 -- redis-cli -a "$DF_PW" --no-auth-warning INFO server | grep dragonfly_version
kubectl -n $NS exec dragonfly-0 -- redis-cli -a "$DF_PW" --no-auth-warning INFO memory | grep -E "used_memory_human|maxmemory_human"
```

The PING uses `$DF_PW` from your shell, so it proves Dragonfly runs with **your** password, not `<SET_ME>`.
If it answers `WRONGPASS`, Dragonfly started before the patch: restart it again.

### Gate 1b: SeaweedFS is ready and the data is restored

```bash
kubectl -n $NS get pods -l app=seaweedfs          # READY 1/1
kubectl -n $NS port-forward svc/seaweedfs 8333:8333 &
export RCLONE_CONFIG_SEAWEEDFS_TYPE=s3 RCLONE_CONFIG_SEAWEEDFS_PROVIDER=SeaweedFS \
       RCLONE_CONFIG_SEAWEEDFS_ENDPOINT=http://localhost:8333 \
       RCLONE_CONFIG_SEAWEEDFS_ACCESS_KEY_ID=$S3_KEY RCLONE_CONFIG_SEAWEEDFS_SECRET_ACCESS_KEY=$S3_SECRET
rclone lsd seaweedfs:                                  # rag-search exists (made by the bucket Job)
rclone copy ./minio-backup/rag-search seaweedfs:rag-search --progress
rclone check ./minio-backup/rag-search seaweedfs:rag-search --one-way   # 0 differences
rclone size seaweedfs:rag-search                       # matches the MinIO count from 2.4
```

The bucket Job deletes itself once it succeeds, so check that the bucket exists rather than looking for the Job.

### Gate 2: Start langfuse-worker, then 5 minutes with no auth or Lua errors

```bash
kubectl -n $NS scale deploy/langfuse-worker --replicas=1
kubectl -n $NS rollout status deploy/langfuse-worker
sleep 300
kubectl -n $NS logs deploy/langfuse-worker --since=5m \
  | grep -iE "NOAUTH|WRONGPASS|invalid password|undeclared|lua|script error" || echo "clean"
```

- Auth errors mean `REDIS_AUTH` in `langfuse-worker-secrets` doesn't match `DRAGONFLY_PASSWORD`. Fix the Secret, then restart the worker.
- Lua or `undeclared` errors mean the `--default_lua_flags` flag is missing; check the Dragonfly pod's args.

### Gate 3: Start langfuse-web, then a test trace appears in the UI

```bash
kubectl -n $NS scale deploy/langfuse-web --replicas=1
kubectl -n $NS rollout status deploy/langfuse-web
```

1. Send one test chat through staging.
2. In the Langfuse UI, the trace appears within about a minute.
3. The queue drains: `kubectl -n $NS exec dragonfly-0 -- redis-cli -a "$DF_PW" --no-auth-warning --scan --pattern 'bull:*:wait'` prints nothing once ingestion has finished.
4. Langfuse wrote to SeaweedFS: `rclone ls seaweedfs:rag-search/langfuse/events/ | tail` shows new objects.

### Gate 4: Restart llm-orchestration-service, then three checks

```bash
kubectl -n $NS rollout restart deploy/llm-orchestration-service
kubectl -n $NS rollout status deploy/llm-orchestration-service
kubectl -n $NS exec deploy/llm-orchestration-service -- curl -s localhost:8100/health
```

1. **Health:** `/health` shows `"dragonfly_session_store":"connected"`. `not_configured` means Dragonfly wasn't reachable at startup: fix it, then restart again.
2. **Sessions and history:** send a test chat that triggers an API tool call, then check that the `session:` and `conv:` keys exist with a TTL of 1800 or less:
   ```bash
   DF="redis-cli -a $DF_PW --no-auth-warning -n 1"
   kubectl -n $NS exec dragonfly-0 -- $DF --scan --pattern 'conv:*'
   kubectl -n $NS exec dragonfly-0 -- $DF --scan --pattern 'session:*'
   kubectl -n $NS exec dragonfly-0 -- $DF TTL conv:<chat_id>
   ```
3. **ATC cache hits on repeat:** ask the same API question again in the same chat. An `atc:cache:<chat_id>:*` key exists, and the second answer comes back without a new upstream API call (check the LLM service logs).

Also check that the app keys are in **db 1 only** (`INFO keyspace`: db0 holds `bull:*`, db1 holds the app keys).

### 4.2 Finish the upgrade

1. Run `helm upgrade` again **without** the `replicas=0` overrides, so the release matches the values.
   **Keep the SeaweedFS keys.** Without them SeaweedFS restarts with `<SET_ME>` keys, and Langfuse and S3-Ferry can no longer log in:
   ```bash
   helm -n $NS upgrade $REL ./kubernetes -f <staging-values.yaml> \
     --set seaweedfs.auth.accessKey=$S3_KEY --set seaweedfs.auth.secretKey=$S3_SECRET
   ```
   Every later `helm upgrade` needs these two `--set` flags too (or put them in a staging values file kept outside git).
2. **Run the `<SET_ME>` check from 4.1 again.** If Helm reset any Secret, patch it again and restart the pods that use it.
3. Confirm the S3-Ferry pod is Running, and that one indexing run (CronManager / DVC) reads and writes `rag-search`.
4. Check that `kubectl -n $NS get pods` shows no CrashLoopBackOff.

---

## 5. Gate summary

| Gate | Pass when | Fail → |
|---|---|---|
| 1 Dragonfly | Pod Ready, `PING` with your password → `PONG` | Re-patch the secret and restart; check sizing / OOM; else rollback |
| 1b SeaweedFS | Pod Ready, bucket exists, `rclone check` 0 differences | Check S3 keys / PVC size; else rollback |
| 2 Worker | No auth/Lua errors for 5 min | Fix `REDIS_AUTH` / Lua flag, restart; else rollback |
| 3 Web | Test trace visible, queue drains, objects land in SeaweedFS | Check the Langfuse S3 keys; else rollback |
| 4 LLM service | `/health` connected, `session:`/`conv:` keys created, ATC cache hit | Restart once; else rollback |

---

## 6. Hypercare (48 h) (7.5)

The "Phase 5" Dragonfly dashboard and alerts are **not** set up on Kubernetes (gap 4). For 48 hours,
use the checks below **at least every few hours, and right after any alert or user report**.
Compare against the baseline from 2.3.

**Langfuse queue lag and Dragonfly memory** (run from a shell with `$DF_PW` set):

```bash
DF0="redis-cli -a $DF_PW --no-auth-warning"
kubectl -n $NS exec dragonfly-0 -- $DF0 INFO memory | grep -E "used_memory_human|maxmemory_human"
kubectl -n $NS exec dragonfly-0 -- $DF0 INFO stats  | grep -E "instantaneous_ops_per_sec"
kubectl -n $NS exec dragonfly-0 -- $DF0 --scan --pattern 'bull:*:wait'     # empty, or empties within minutes
kubectl -n $NS exec dragonfly-0 -- $DF0 INFO keyspace                       # db0 = Langfuse, db1 = app
kubectl -n $NS get pod dragonfly-0                                          # RESTARTS stays 0
kubectl -n $NS top pod dragonfly-0
```

**App-side errors in Grafana → Explore (Loki).** The LLM service sends these logs to Loki on Kubernetes too:

| What | Loki query | Look for |
|---|---|---|
| Store errors | `{service=~"api-tool-session-store\|conversation-history\|atc-cache-store"} \|~ "failed\|Dragonfly unavailable"` | None, or only isolated lines |
| Out of memory | `{service=~"api-tool-session-store\|conversation-history\|atc-cache-store"} \|~ "OOM\|maxmemory"` | None |
| Lost connection | `{service="llm-orchestration-api"} \|= "Dragonfly health check"` | None |
| Update conflicts | `{service=~"api-tool-session-store\|conversation-history"} \|= "retries due to concurrent writes"` | None |

**Thresholds** (the same ones the Compose alerts use):

| Signal | Act when |
|---|---|
| Dragonfly memory | Above 70% of `maxmemory`: plan to raise it. Above 85%: raise `maxmemory` **and** the memory limit now (writes fail at 100%). |
| Langfuse queue | `bull:*:wait` not empty for 15 minutes: check and restart `langfuse-worker`. |
| Restarts | Any unplanned Dragonfly restart: `kubectl describe pod dragonfly-0` (OOMKilled?) |
| App p99 / errors | Worse than the baseline |
| SeaweedFS | Pod not Running, or PVC usage above 80% |

How to respond: [DRAGONFLY_OPERATIONS.md → Alert responses](./DRAGONFLY_OPERATIONS.md#alert-responses).
Memory warnings alone are **not** a reason to roll back.

---

## 7. Rollback (7.7)

**Roll back when:**
- a gate fails and can't be fixed quickly
- the Langfuse queue keeps growing even after restarting the worker
- Dragonfly keeps restarting or being OOM-killed
- the app keeps losing its store

**What's lost:**
- Sessions, history and ATC cache: ≤30 min of data, so chats in progress lose context once.
- Langfuse jobs still queued in Dragonfly: lost unless you drain first.
- Objects written to SeaweedFS after the upgrade: they're not in MinIO. Copy them back (step 1 below) if they matter.

**Steps:**
1. **Optional, while SeaweedFS still runs:** copy new objects back with `rclone copy seaweedfs:rag-search minio-backup-new/` and add them to MinIO after step 4.
2. **Drain Langfuse on Dragonfly** (skip this if Langfuse is already broken): same as 3.2, but run the scans against `dragonfly-0` with `$DF_PW`.
3. **Return to the previous revision:**
   ```bash
   helm -n $NS rollback $REL <REVISION-from-2.2>
   ```
   This brings back the `redis` and `minio` charts, the old settings and the old Langfuse tag.
   - `pvc-minio-data` was kept (step 2.4), so MinIO starts with its old data. If it was lost anyway, restore it with `rclone copy ./minio-backup/rag-search minio:rag-search`.
   - The Dragonfly PVC (`data-dragonfly-0`) isn't deleted by Helm and keeps its data for a later roll-forward.
4. **Restore all secrets** from `pre-migration-secrets.yaml` (`kubectl apply -f`). The Langfuse secrets still hold the Dragonfly password and SeaweedFS keys, and `redis-secrets` and `minio-secrets` come back as `<SET_ME>`. Then restart `redis`, `minio`, `langfuse-*`, `s3-ferry` and `llm-orchestration-service`.
5. **Verify:** the `redis` and `minio` pods are Running, `/health` reports the store as connected, and a test chat produces a Langfuse trace.

Record the rollback and its reason, and fix the cause before trying again.

---

## 8. Decommission (7.6)

This branch already removes the Redis and MinIO charts from the code, and Redis 7 and MinIO stay
pinned in the previous release. "Deployable" therefore means the **previous Helm revision and git tag**.
Keep these until **one stable release** has passed on the new stack:

- the Helm revision from 2.2 (don't let `helm history` drop it)
- `pre-migration-values.yaml` and `pre-migration-secrets.yaml` (stored securely)
- the `pvc-minio-data` PVC and `./minio-backup/`

After that stable release, delete `pvc-minio-data`, the MinIO backup, and the old secrets
(`redis-secrets`, `minio-secrets`).

If the team wants the rollback to be a pure config switch instead, add the `Redis` chart back
with `enabled: false` before deploying to staging. That's a code change, not a deployment step.

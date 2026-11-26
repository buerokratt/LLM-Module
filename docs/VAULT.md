# Vault

How LLM-Module uses the shared central Vault (issue #777). Replaces
`VAULT_SECURITY_ARCHITECTURE.md`, which described the module's own Vault.

---

## Model

The Vault server belongs to the platform team. We never hold its root token or
unseal keys, and no application of ours talks to it directly.

Three **Vault Agents** run as sidecars, one per consuming workload. Each does
two jobs:

- **auto_auth** — reads `role_id` + `secret_id` from a mounted Secret, logs in
  via AppRole, keeps a periodic token renewed, writes it to a sink file.
- **api_proxy** — listens on a local port and forwards requests to Vault with
  `X-Vault-Token` injected.

So application code sends **tokenless** HTTP to `localhost:<agent port>` and the
agent attaches credentials. The policy is attached to the token, not to the
files on disk.

| Workload | Agent port | Role | Token period |
|---|---|---|---|
| GUI | 8202 | `llm-module-gui-service` | 20m |
| CronManager | 8203 | `llm-module-cron-manager-service` | 30m |
| Orchestration | 8201 | `llm-module-llm-orchestration-service` | 1h |

---

## What we store

One KV v2 engine mounted at `secret`, everything under the `llm-module/` prefix.

```
secret/llm-module/
├── encryption/
│   ├── public_key          GUI encrypts with this
│   └── private_key         CronManager decrypts with this
├── llm/connections/{provider}/{vault_uuid}
├── embeddings/connections/{provider}/{vault_uuid}
└── langfuse/config
```

`{provider}` is `aws_bedrock` or `azure_openai`. `{vault_uuid}` is the
connection's `vault_uuid` column in `rag-search-db`.

**The environment is not part of the path.** Promoting a connection from
testing to production is a database update; the secret never moves.

### encryption/{public,private}_key

RSA-2048, fields `key` (PEM), `algorithm: RSA-OAEP`, `key_size`, `key_id`,
`created_at`. **Seeded by central's bootstrap**, not by us — no role we own can
write to `encryption/*`.

### llm/connections and embeddings/connections

Written by CronManager, read by the orchestration service.

| Provider | Fields written |
|---|---|
| `azure_openai` | `connection_id`, `endpoint`, `api_key`, `deployment_name`, `model`, `api_version`, `tags` |
| `aws_bedrock` | `connection_id`, `access_key`, `secret_key`, `model`, `tags` |

Embeddings use the same shape, with the model from `embeddingModel` and a
different hardcoded `api_version`. Writing embeddings is skipped when no
embedding platform is supplied.

### langfuse/config

Fields `public_key`, `secret_key`, `host`. A single global document, read by the
orchestration service at startup to initialise tracing.

Central does **not** seed this, and our orchestration role has `read` only — no
role we own can write to `langfuse/*`. So it is a one-time manual write by
whoever holds a token with write access.

Get the keys first from the Langfuse UI (**Settings → API Keys**), or choose
them up front via `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `..._SECRET_KEY` on a
fresh Langfuse instance.

**Kubernetes.** `host` must be the in-cluster Service URL, which is port `3005`
even though the container listens on `3000`:

```bash
vault kv put secret/llm-module/langfuse/config \
  public_key=pk-lf-... \
  secret_key=sk-lf-... \
  host=http://langfuse-web:3005
```

**Local development.** The compose stack has its own Vault, so the root token is
on its data volume and the script can find it:

```bash
docker cp store-langfuse-secrets.sh vault:/tmp/
docker exec -e LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-... \
            -e LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-... \
            -e VAULT_PATH_PREFIX=llm-module \
            vault sh /tmp/store-langfuse-secrets.sh
```

The script also accepts `VAULT_TOKEN` and `VAULT_ADDR`, which is how to point it
at a Vault whose init material is not ours.

Two things that waste time if missed:

- **Trim trailing whitespace.** The orchestration service puts the public key in
  an HTTP header when exporting traces, so a stray newline makes every export
  fail with `Invalid header value` — while `auth_check()` still passes and the
  client logs that it initialised successfully.
- **Restart the deployment afterwards.** The secret is read once at service
  construction; seeding alone changes nothing.

### AppRole credentials (not KV)

`role_id` and `secret_id`, one Secret per workload, created by central's
bootstrap **into our namespace** via the Kubernetes API:

```
llm-module-gui-approle      llm-module-cron-approle      llm-module-llm-approle
```

Each is mounted read-only at `/vault/approle` in its agent sidecar only — the
application container does not mount it. Secret keys become files, which is
what the agent's HCL expects. They never expire (`secret_id_ttl: "0"`).

These must exist before our pods start; a missing Secret means the pod cannot
mount the volume and will not start.

---

## Who can do what

| Path | GUI | Cron | Orchestration |
|---|---|---|---|
| `encryption/public_key` | read | read | deny |
| `encryption/private_key` | deny | read | deny |
| `llm/connections/*` | deny | CRUD | read, list |
| `embeddings/connections/*` | deny | CRUD | read, list |
| `langfuse/*` | — | — | read, list |

CronManager is the only writer and the only service that can read the private
key, because it is the only thing that must decrypt what the GUI encrypted.
Orchestration is denied `encryption/*` outright.

`list` is a **metadata** operation. Model listing needs `list` on
`secret/metadata/llm-module/...`, not just `read` on `secret/data/...`, and a
missing grant returns an empty list rather than an error.

---

## The path prefix

`llm-module` is configuration, defaulting to empty. It reaches four consumers
that share no configuration mechanism:

| Consumer | Source |
|---|---|
| Orchestration (Python) | `VAULT_PATH_PREFIX` env, else `constants.ini` |
| CronManager (shell) | `constants.ini` only — job scripts run with an empty environment |
| GUI (browser) | `REACT_APP_VAULT_PATH_PREFIX` |
| Bootstrap scripts | `VAULT_PATH_PREFIX` env |

In Kubernetes it is set once as `global.vaultPathPrefix` in the umbrella values
and propagates to every subchart.

**Python applies the prefix in `VaultAgentClient`, nowhere else.** The client is
the single chokepoint for every read, so the Langfuse lookup that bypasses
`SecretResolver` is covered automatically, and resolver paths stay unprefixed —
which keeps `refresh_secret()`'s positional provider parsing (`path_parts[2]`)
correct. Prefixing in both places would produce `llm-module/llm-module/...`.
Guarded by `tests/test_vault_path_prefix.py`.

---

## Access mechanisms

The same logical secret is spelled differently per caller:

| Caller | Transport | Path in code | Adds `secret/` | Adds `data/` |
|---|---|---|---|---|
| Orchestration | `hvac` → agent :8201 | `llm/connections/...` | `hvac` (mount) | `hvac` |
| CronManager | `curl` → agent :8203 | `secret/llm-module/llm/connections/...` | script | `sed` |
| GUI | `fetch` → agent :8202 | `/vault-agent-gui/v1/secret/data/llm-module/encryption/public_key` | literal | literal |

Endpoint resolution: orchestration uses `VAULT_AGENT_LLM_URL` from
`constants.ini` (`VAULT_ADDR` overrides it, which is how test stacks bypass the
agent); CronManager uses `VAULT_AGENT_CRON_URL`; the GUI has no address at all
and calls a path prefix that nginx/vite proxies to its sidecar.

Deletion is two calls — `DELETE secret/data/...` soft-deletes the latest
version, `DELETE secret/metadata/...` destroys all versions. Both are prefixed.

---

## Creating a connection, end to end

```
1. GUI fetches the public key through vault-agent-gui
2. Browser encrypts the API key with RSA-OAEP SHA-256
3. POST to Ruuter → CronManager runs store_secrets_in_vault.sh
4. CronManager reads the PRIVATE key through its agent and decrypts in memory
5. Writes plaintext to secret/llm-module/llm/connections/{provider}/{vault_uuid}
6. The vault_uuid is stored on the connection row in rag-search-db
```

At inference time the orchestration service looks up `vault_uuid` for the
connection, builds `llm/connections/{provider}/{uuid}`, and the client prefixes
it. Results are cached for 5 minutes with a last-known-good fallback.

---

## Local development

docker-compose still runs its **own** Vault, `vault-init` and a shared
credentials volume — there is no central Vault to point at, and compose has no
equivalent of a Kubernetes Secret. Only the path prefix is shared with the
Kubernetes setup.

Set `VAULT_PATH_PREFIX` in **both** `.env` (vault-init policies, keypair, GUI)
and `constants.ini` (orchestration, CronManager). A mismatch grants policies on
paths nothing reads.

Consequence: the credential-delivery half of the migration is not exercised
locally — compose reads `/agent/credentials/<service>_role_id`, production reads
`/vault/approle/role_id`. Only the prefix half is covered.

---

## Verifying

Through an agent, no token needed:

```bash
kubectl -n <ns> exec deploy/gui -c vault-agent-gui -- \
  wget -qO- http://127.0.0.1:8202/v1/secret/data/llm-module/encryption/public_key
```

A `403` from the GUI agent on `private_key` is correct — that is the policy
working. A `404` on a LIST means the grant exists but nothing is stored yet; a
`403` means the grant is missing.

To confirm a connection landed where the reader expects it, compare the
`vault_uuid` on the `rag_search.llm_connections` row with the key under
`secret/llm-module/llm/connections/{provider}/`.

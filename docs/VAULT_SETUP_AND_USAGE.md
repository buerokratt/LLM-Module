# Vault Setup & Usage Guide

A single reference for how HashiCorp Vault is deployed, initialized, and consumed in the
RAG-Module. It covers the topology, the three Vault Agents, the secret layout, and — in
depth — **how each agent renews its token and how secrets are rotated**.

Source files this document describes:

- `docker-compose.yml` — service/topology definition
- `vault/config/vault.hcl` — Vault server config
- `vault-init.sh` — one-time bootstrap + per-restart reconcile
- `vault/agents/{gui,cron,llm}/*.hcl` — the three Vault Agent configs
- `DSL/CronManager/script/store_secrets_in_vault.sh` — writes/rotates secrets
- `DSL/CronManager/script/delete_secrets_from_vault.sh` — deletes secrets

For the security rationale (threat model, defense-in-depth, access matrix) see the
companion `docs/VAULT_SECURITY_ARCHITECTURE.md`. This guide focuses on the *operational*
mechanics.

---

## 1. Topology at a glance

```
                bykstack (application network)                  vault-network (internal: true)
 ┌───────────────────────────────────────────────┐        ┌──────────────────────────────┐
 │  gui ──────────────► vault-agent-gui  :8202 ───┼────────┤                              │
 │  cron-manager ─────► vault-agent-cron :8203 ───┼────────┤        vault  :8200          │
 │  llm-orchestration ► vault-agent-llm  :8201 ───┼────────┤   (Raft storage, KV v2,      │
 │                                                │        │    AppRole auth)             │
 │  vault-init (also on vault-network) ───────────┼────────┤                              │
 └───────────────────────────────────────────────┘        └──────────────────────────────┘
```

- **`vault`** runs only on `vault-network`, which is `internal: true` — it has **no route to
  or from the host or the internet**. Port 8200 is never published.
- **Vault Agents** straddle both networks: they reach `vault` on `vault-network` and are
  reachable by their owning application on `bykstack`.
- **Applications** talk *only* to their agent (`VAULT_ADDR=http://vault-agent-*:820x`) and
  never hold a Vault token themselves. The agent injects the token transparently.

| Service | Agent it uses | Agent address | AppRole | Policy |
|---|---|---|---|---|
| `gui` | `vault-agent-gui` | `:8202` | `gui-service` | `gui-policy` |
| `cron-manager` | `vault-agent-cron` | `:8203` | `cron-manager-service` | `cron-manager-policy` |
| `llm-orchestration-service` | `vault-agent-llm` | `:8201` | `llm-orchestration-service` | `llm-orchestration-policy` |

---

## 2. Vault server (`vault/config/vault.hcl`)

- **Storage:** Raft, single node (`node_id = vault-node-1`, path `/vault/file`, persisted in
  the `vault-data` volume). No `retry_join` — a lone node self-bootstraps; adding a self-
  pointing join was found to cause "Vault is sealed" boot loops.
- **Listener:** `0.0.0.0:8200`, `tls_disable = true` (TLS is terminated at the network
  boundary; the network itself is the isolation layer here). Port `8201` is *not* given its
  own listener because Vault uses it as the internal cluster port automatically.
- **Lease defaults:** `default_lease_ttl = 168h` (7 days), `max_lease_ttl = 720h` (30 days).
  These are *system ceilings*; the per-AppRole token TTLs (below) are much shorter and are
  what actually governs agent renewal cadence.
- `disable_mlock = false`, `ui = false`, JSON logs at INFO.

Vault boots **sealed**. It must be unsealed before any operation — that is `vault-init`'s
first job.

---

## 3. Bootstrap & reconcile (`vault-init.sh`)

`vault-init` is a **run-once-then-exit** container (`restart: "no"`). The agents declare
`depends_on: vault-init: condition: service_completed_successfully`, so they only start
after init has finished cleanly. It runs `su vault -s /bin/sh /vault-init.sh` after creating
and `chown`ing the shared agent directories.

The script has two branches, selected by the presence of `/vault/data/.initialized`.

### 3.1 First-time deployment

1. Wait for `/v1/sys/health` to respond.
2. **Initialize** with Shamir's Secret Sharing: `secret_shares=5`, `secret_threshold=3`.
   The full response (5 unseal keys + root token) is written to
   `/vault/data/unseal-keys.json`.
3. **Unseal** by submitting 3 of the 5 keys.
4. **Enable engines:** KV v2 at `secret/`, and the AppRole auth method.
5. **Create three ACL policies** (see §5).
6. **Create three AppRoles** issuing periodic tokens (see §4 — this is the heart of renewal),
   via the `ensure_approles` helper. The same helper re-runs on subsequent deploys, so AppRole
   config changes land without re-initializing Vault.
7. **Issue credentials:** for each role, fetch the static `role_id` and mint a `secret_id`,
   writing both to `/agent/credentials/<svc>_role_id` and `<svc>_secret_id` (`chmod 640`).
8. **Generate an RSA-2048 keypair** with `openssl` and store it in Vault at
   `secret/encryption/public_key` and `secret/encryption/private_key`
   (algorithm `RSA-OAEP`, with `key_id` and `created_at` metadata).
9. Seed a test LLM secret, then `touch /vault/data/.initialized`.

### 3.2 Subsequent deployment (restart)

1. Check `/v1/sys/seal-status`; if sealed, reload the 3 unseal keys from
   `unseal-keys.json` and unseal.
2. **Reconcile each secret_id** via `reconcile_secret_id`:
   - `ensure_role_id` — make sure the `role_id` file exists (re-fetch from Vault if missing).
   - `validate_secret_id` — attempt an AppRole login with the on-disk `role_id` + `secret_id`.
     If it returns a `client_token`, the credential is still good.
   - **Valid → reuse** the existing `secret_id` (no churn).
   - **Invalid/missing → `mint_secret_id`** writes a fresh one.

This is deliberate: because the AppRoles are created with `secret_id_ttl=0` and
`secret_id_num_uses=0` (non-expiring, unlimited-use), a single long-lived `secret_id`
survives normal restarts instead of being regenerated every boot. The RSA keypair, policies,
and stored secrets are all preserved across restarts.

> **Note on file permissions:** `vault-init.sh` writes credential files with `chmod 640`.
> (The older architecture doc mentions `644`; the script is the source of truth — `640`.)

---

## 4. The three Vault Agents — auth, renewal & rotation

This is the core of the question. All three agents are the same Vault binary
(`hashicorp/vault:1.20.3`) run as `vault agent -config=...`. They differ only in which
credentials they read, which token sink they write, and their listener port.

### 4.1 What an agent config actually does

Example (`vault/agents/llm/agent.hcl`; gui/cron are identical in shape):

```hcl
vault { address = "http://vault:8200"; retry { num_retries = 5 } }

auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path   = "/agent/credentials/llm_role_id"
      secret_id_file_path = "/agent/credentials/llm_secret_id"
      remove_secret_id_file_after_reading = false
    }
  }
  sink "file" { config = { path = "/agent/llm-token/token"; mode = 0640 } }
}

cache { default_lease_duration = "1h" }
listener "tcp" { address = "0.0.0.0:8201"; tls_disable = true }
api_proxy { use_auto_auth_token = true }
```

Three mechanisms are at work:

1. **`auto_auth` (authentication + renewal):** On startup the agent reads `role_id` +
   `secret_id` and calls `POST /v1/auth/approle/login`. Vault returns a **periodic token**
   (the AppRoles set `token_period`, defined in `vault-init.sh`, *not* in the HCL). The agent
   then runs Vault's **auto-auth lifecycle manager**, which **renews the token automatically
   in the background** before each period elapses. A periodic token has **no max-TTL**, so the
   agent renews it indefinitely and — during normal operation — **never has to call
   `approle/login` again**. The agent only re-authenticates (and thus only needs the
   `secret_id` again) if it is **restarted** or if a renewal is missed long enough for the
   token to lapse. `remove_secret_id_file_after_reading = false` keeps the `secret_id` on disk
   so the agent can re-auth after a restart without `vault-init` re-minting.

   > **Why periodic tokens?** An earlier design issued tokens with `token_ttl`/`token_max_ttl`,
   > which forced a full re-login every time `token_max_ttl` was reached. If the `secret_id`
   > had become invalid by then (expiry, clock skew, server re-init), the agent got stuck in an
   > `invalid role or secret ID` 400 backoff loop with no way to self-heal. Periodic tokens
   > remove that re-login from the steady state, so a stale `secret_id` can no longer strand a
   > running agent.
2. **`sink "file"` (token hand-off):** Every time the agent obtains/renews a token it writes
   it to a file (`/agent/<svc>-token/token`, mode `0640`). The compose **health check** for
   each agent is simply `test -f <token> && test -s <token>` — a non-empty token file means
   the agent has authenticated successfully.
3. **`api_proxy { use_auto_auth_token = true }` (transparent injection):** The agent also
   listens as an HTTP proxy on its port. When the application sends a token-less request, the
   agent injects `X-Vault-Token: <current cached token>` and forwards it to `vault:8200`.
   This is why application code never sets `VAULT_TOKEN`.

> **`cache.default_lease_duration` is not the token TTL.** It is the agent's cache lease
> hint. The authoritative token lifetime comes from the AppRole's `token_period` in
> `vault-init.sh`. The per-agent cache hint is set to match the period.

### 4.2 Per-agent renewal parameters

AppRole token settings are created in `vault-init.sh`; all three use
`token_period` (periodic token, **no max-TTL**), `secret_id_ttl=0`, `secret_id_num_uses=0`,
`token_num_uses=0`, `bind_secret_id=true`.

| Agent | AppRole | `token_period` | Proactive renewal (~⅔ of period) | Re-login (`approle/login`) |
|---|---|---|---|---|
| `vault-agent-gui` | `gui-service` | **20m** | ~every 13 min | only on agent restart |
| `vault-agent-cron` | `cron-manager-service` | **30m** | ~every 20 min | only on agent restart |
| `vault-agent-llm` | `llm-orchestration-service` | **1h** | ~every 40 min | only on agent restart |

Reading the lifecycle for, e.g., the LLM agent:

```
T=0       login → periodic token (period 1h)        → written to /agent/llm-token/token
T≈40m     renew-self → period resets to 1h          → token file refreshed
...       renew repeats forever; token never hits a max-TTL
(restart) agent re-runs approle/login with the on-disk secret_id → fresh token
```

The periods are tuned per service (shorter for the GUI, which only reads the public key;
longer for the high-traffic LLM read path), but functionally all three behave the same:
**renew forever, re-login only on restart.**

### 4.3 Two distinct "rotation" concepts — keep them separate

1. **Token rotation (automatic, continuous):** Handled entirely by the agent's `auto_auth`
   loop as described above — the periodic token is renewed indefinitely with no human action
   and no `vault-init` involvement.
2. **`secret_id` rotation (rare):** The `secret_id` is the long-lived credential the agent
   uses to *log in* (at startup/restart only, now that tokens are periodic). It is configured
   non-expiring (`secret_id_ttl=0`, `secret_id_num_uses=0`) and is only replaced by
   `vault-init` on a restart when the existing one fails validation (§3.2). To force rotation,
   delete the `secret_id` file (or invalidate it in Vault) and re-run `vault-init`, then
   restart the agent so it logs in with the freshly minted one.

   > **Operational caveat (learned the hard way):** if a `secret_id` ever does become invalid
   > while an agent is running, the periodic-token design means a *running* agent keeps working
   > (it only renews, never re-logs-in). But a **restarted** agent needs a valid `secret_id` to
   > log in. Recovery is always: re-run `vault-init` (mints a fresh `secret_id` via the §3.2
   > reconcile) → restart the affected agent. See `docs/` runbook / the troubleshooting note
   > below.

### 4.4 Restart behavior

- **Restart an agent:** It re-reads `role_id`/`secret_id` from the (read-only) creds volume
  and re-authenticates. New token, written to the sink. App sees a brief blip.
- **Restart `vault`:** Data persists; `vault-init` (or the existing agent tokens, if still
  valid) handle re-unseal/re-auth. Existing tokens remain valid if not expired.
- **Full `down && up`:** Order is `vault → vault-init → agents → apps`. `vault-init` detects
  the `.initialized` flag, skips first-time setup, reconciles secret_ids, and the agents
  start with validated credentials.

---

## 5. Authorization — policies (who can touch what)

Created in `vault-init.sh`. Paths are KV v2, so data lives under `secret/data/...` and
listing/metadata under `secret/metadata/...`.

| Path | `gui-policy` | `cron-manager-policy` | `llm-orchestration-policy` |
|---|---|---|---|
| `secret/data/encryption/public_key` | **read** | read | — |
| `secret/data/encryption/private_key` | **deny** | **read** | — |
| `secret/data/encryption/*` | — | — | **deny** |
| `secret/data/llm/connections/*` | deny | **create/read/update/delete** | **read, list** |
| `secret/data/embeddings/connections/*` | deny | **create/read/update/delete** | **read, list** |
| `auth/token/lookup-self` | — | read | read |

The intent, by tier:

- **GUI** — can read *only* the public key, to encrypt user-entered credentials in the
  browser before they ever leave it. Everything else is explicitly denied.
- **CronManager** — the only writer. Reads the **private key** to decrypt what the GUI
  encrypted, then writes plaintext credentials into Vault. Full CRUD on connection secrets.
- **LLM Orchestration** — read-only consumer of connection secrets. **Explicitly denied** all
  encryption keys, so a compromise of this hot-path service cannot exfiltrate the private key.

---

## 6. Secret layout (KV v2 under `secret/`)

```
secret/
├── llm/connections/<platform>/<vaultUuid>          ← e.g. aws_bedrock, azure_openai
├── embeddings/connections/<platform>/<vaultUuid>
└── encryption/
    ├── public_key     { key, algorithm: RSA-OAEP, key_size: 2048, key_id, created_at }
    └── private_key    { key, algorithm: RSA-OAEP, key_size: 2048, key_id, created_at }
```

The current write/delete scripts key connection secrets by a stable **`vaultUuid`** as the
final path segment (environment is tracked in the DB, not the path). KV v2 versions every
write, so updating a credential keeps prior versions for audit/rollback.

LLM secret shape (AWS): `{ connection_id, access_key, secret_key, model, tags }`.
Azure: `{ connection_id, endpoint, api_key, deployment_name, model, api_version, tags }`.

---

## 7. Usage flows

### 7.1 Storing / rotating a credential (`store_secrets_in_vault.sh`, via cron-manager)

1. GUI encrypts the raw key with the RSA **public** key and submits it.
2. The cron-manager job runs the script against `vault-agent-cron:8203` (no token — the agent
   injects it).
3. The script **fetches the private key** (`GET secret/data/encryption/private_key`), then
   decrypts each sensitive field in-memory via `decrypt_vault_secrets.py` (RSA-OAEP).
4. It builds the JSON payload with `jq` and `POST`s plaintext to
   `secret/data/<llm|embeddings>/connections/<platform>/<vaultUuid>`. Re-posting the same path
   = a KV v2 version bump = credential rotation.
5. Sensitive shell variables are `unset` immediately after use.

### 7.2 Deleting a credential (`delete_secrets_from_vault.sh`)

`DELETE`s both `secret/data/...` and `secret/metadata/...` for the connection (404 treated as
success), again through `vault-agent-cron` with no explicit token.

### 7.3 Reading a credential (LLM orchestration)

The LLM service issues a token-less `GET http://vault-agent-llm:8201/v1/secret/data/llm/...`.
`vault-agent-llm` injects its cached token, Vault validates it against
`llm-orchestration-policy`, and returns the secret. The service then calls AWS/Azure with it.

---

## 8. Operational notes & known trade-offs

- **Unseal keys + root token sit in the `vault-data` volume** (`unseal-keys.json`). This makes
  auto-unseal on restart trivial but is a **dev/test convenience**. For production, switch to
  auto-unseal backed by a cloud KMS/HSM and remove the keys from the volume.
- **Root token** is used only by `vault-init` and is never injected into app containers. Best
  practice for production is to revoke it after bootstrap and use scoped admin policies.
- **TLS is disabled** on the Vault listener and agent listeners; isolation relies on the
  `internal: true` `vault-network`. Add TLS for any non-local deployment.
- **Audit logging is available but not enabled.** Turn it on with
  `vault audit enable file file_path=/vault/logs/audit.log` (the `./vault/logs` mount already
  exists) for a full request trail.
- **Credential files are world-readable within the shared volume** (mode 640, single owner,
  but all agents mount the same `vault-agent-creds` volume read-only) — isolation is at the
  volume level, not per-file. Fine for this trust boundary; note it if the threat model
  tightens.

---

## 9. Troubleshooting: agents looping on `invalid role or secret ID`

**Symptom:** an agent logs `lifetime watcher done channel triggered, re-authenticating`
followed by repeating `PUT .../auth/approle/login → Code: 400 ... invalid role or secret ID`
with growing backoff. Token *renewals* had been succeeding up to that point.

**Cause:** the agent's `secret_id` became invalid server-side (expiry, clock skew, or a Vault
re-init), and the agent reached a point where it had to do a full `approle/login`. With the
old `token_ttl`/`token_max_ttl` design this happened on every `token_max_ttl` cycle; the
switch to **periodic tokens** (§4) removes re-login from steady state, so a *running* agent no
longer hits this — but a **restarted** agent still needs a valid `secret_id`.

**Recovery:**

```bash
# Mint fresh secret_ids (vault-init's reconcile detects the invalid ones and replaces them)
docker compose up -d --force-recreate vault-init
docker wait vault-init
# Restart the affected agents so they log in with the fresh secret_id
docker compose restart vault-agent-gui vault-agent-cron vault-agent-llm
```

**Confirm root cause (read-only):**

```bash
ROOT=$(docker exec vault sh -c "grep -o '\"root_token\":\"[^\"]*\"' /vault/file/unseal-keys.json | cut -d: -f2 | tr -d '\"'")
docker exec -e VAULT_TOKEN=$ROOT -e VAULT_ADDR=http://127.0.0.1:8200 vault \
  vault read auth/approle/role/gui-service          # expect token_period set, secret_id_ttl=0
echo "host: $(date -u)"; docker exec vault date -u  # check for WSL2/Docker clock drift
```

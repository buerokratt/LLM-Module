# Langfuse Setup

**you can seed secrets in Langfuse-web , Langfuse-worker,clickhouse and database  with .env file values**

## 1. Verify Required Pods

```bash
kubectl get pods -n your-namespace
```

All of the following must be `Running` or `Completed` — Langfuse will not start without them:

| Pod | Purpose |
|---|---|
| `rag-search-db-0` | PostgreSQL (hosts `rag-search` and `langfuse-db`) |
| `minio-*` | Object storage for Langfuse events/media |
| `redis-*` | Queue backend for Langfuse worker |
| `clickhouse-*` | Analytics DB for Langfuse ingestion |
| `langfuse-worker-*` | Must be `Running` before web starts |
| `langfuse-web-*` | UI + runs DB migrations on first boot |
| `vault` | Secret storage |
| `vault-Init`  | unseal vault | 

## 2. Wait for DB Migrations

On first startup, `langfuse-web` runs database migrations — this takes 1–2 minutes. Watch the logs:

```bash
kubectl logs -n your-namespace deployment/langfuse-web -f
```

Do **not** proceed until the pod is fully `Running`.

## 3. Access the Dashboard

```bash
kubectl port-forward -n your-namespace svc/langfuse-web 3005:3005
```

Open **http://localhost:3005**, sign up / log in, then go to **Settings → API Keys → Create new key**.

> Save both keys — the secret key is only shown once.
> - `pk-lf-...` → Public Key  
> - `sk-lf-...` → Secret Key

## 4. Store Keys in Vault

```bash
kubectl cp store-langfuse-secrets.sh rag-module/vault-0:/tmp/store-langfuse-secrets.sh

kubectl exec -n your-namespace vault-0 -- sh -c \
  "LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-YOUR_KEY \
   LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-YOUR_KEY \
   LANGFUSE_HOST=http://langfuse-web:3005 \
   sh /tmp/store-langfuse-secrets.sh"
```

Replace `pk-lf-YOUR_KEY` and `sk-lf-YOUR_KEY` with the actual keys from step 3.

> **Note:** In Kubernetes, the Langfuse-Web service port is `3005` (mapped to container port 3000), so `LANGFUSE_HOST` must be set explicitly. In Docker Compose, the default (`http://langfuse-web:3000`) is used automatically.

The script stores them at `secret/data/langfuse/config` in Vault, where the LLM Orchestration Service reads them.
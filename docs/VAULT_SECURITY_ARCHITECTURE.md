# Vault Security Architecture

## Overview

This document provides a technical deep dive into the HashiCorp Vault security architecture implemented in the RAG-Module. The design follows a defense-in-depth strategy with multiple security layers to protect sensitive credentials used by LLM providers (AWS Bedrock, Azure OpenAI) and embedding services.

### Security Principles

1. **Zero Trust Network**: No service has direct access to Vault server
2. **Least Privilege**: Each service gets only the minimum required permissions
3. **Defense in Depth**: Multiple security layers (network, authentication, authorization)
4. **Secure by Default**: Deny-all policies with explicit allow rules
5. **Credential Isolation**: Secrets never exposed in environment variables or logs

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Docker Network: bykstack                    │
│  (Application Layer - No Direct Vault Access)                       │
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │     GUI      │    │ CronManager  │    │  LLM Service │          │
│  │  (Frontend)  │    │   (Worker)   │    │ Orchestrator │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                    │                   │
│         │ :8202            │ :8203              │ :8201             │
│         ▼                   ▼                    ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ vault-agent  │    │ vault-agent  │    │ vault-agent  │          │
│  │     -gui     │    │    -cron     │    │     -llm     │          │
│  │   (Proxy)    │    │   (Proxy)    │    │   (Proxy)    │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                    │                   │
│         └───────────────────┴────────────────────┘                   │
│                             │                                         │
└─────────────────────────────┼─────────────────────────────────────────┘
                              │
                              │ Secured Connection
                              │
┌─────────────────────────────▼─────────────────────────────────────────┐
│                    Docker Network: vault-network                       │
│                (Internal Only - No External Access)                    │
│                                                                         │
│                      ┌──────────────────┐                              │
│                      │   Vault Server   │                              │
│                      │   (Core Vault)   │                              │
│                      │    Port: 8200    │                              │
│                      └──────────────────┘                              │
│                                                                         │
│                    - KV v2 Secrets Engine                              │
│                    - AppRole Auth Method                               │
│                    - Policy Enforcement                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Network Security & Isolation

### Dual-Network Architecture

The system uses two isolated Docker networks to create a security boundary:

#### 1. **vault-network** (Internal Network)
- **Purpose**: Vault core server isolation
- **Access**: Only Vault server and Vault agents
- **Configuration**: `internal: true` (no external routing)
- **Security Benefit**: Vault server is completely unreachable from outside containers

#### 2. **bykstack** (Application Network)
- **Purpose**: Application services communication
- **Access**: All application containers and Vault agents
- **Security Benefit**: Applications can only reach Vault agents, never Vault directly

### Why This Matters

```
 Without Network Isolation:
   App → Vault (direct access with token)
   Risk: Token compromise = full Vault access

 With Network Isolation:
   App → Vault Agent → Vault
   Benefit: Agent handles auth, app never sees token
```

### Port Exposure Strategy

| Service | Port | Network | Exposed to Host | Purpose |
|---------|------|---------|-----------------|---------|
| Vault Server | 8200 | vault-network |  No | Core secrets storage |
| vault-agent-gui | 8202 | bykstack |  No | GUI proxy |
| vault-agent-cron | 8203 | bykstack |  No | CronManager proxy |
| vault-agent-llm | 8201 | bykstack |  No | LLM service proxy |

**Security Principle**: No Vault-related ports are exposed to the host machine, preventing external attacks.

---

## Authentication Layer

### AppRole Authentication Method

Vault uses **AppRole** authentication - a machine-oriented authentication method designed for automated workflows.

#### How AppRole Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    AppRole Authentication Flow                   │
└─────────────────────────────────────────────────────────────────┘

1. Initialization Phase (vault-init container):
   
   vault-init
      │
      ├─► Creates AppRole: "gui-service"
      ├─► Creates AppRole: "cron-manager-service"  
      ├─► Creates AppRole: "llm-orchestration-service"
      │
      └─► Generates credentials:
           - role_id (static identifier)
           - secret_id (secret credential, renewable)

2. Credential Storage:
   
   Credentials written to shared Docker volumes:
   /agent/credentials/gui_role_id
   /agent/credentials/gui_secret_id
   /agent/credentials/cron_role_id
   /agent/credentials/cron_secret_id
   /agent/credentials/llm_role_id
   /agent/credentials/llm_secret_id

3. Vault Agent Authentication:
   
   vault-agent-gui
      │
      ├─► Reads: role_id + secret_id
      ├─► Authenticates with Vault
      └─► Receives: Vault token (automatically renewed)

4. Token Management:
   
   vault-agent caches token and handles:
   - Automatic renewal before expiration
   - Token rotation on renewal failure
   - Transparent injection into API requests
```

### Role-Based Identity Management

Each service gets its own isolated identity:

1. **gui-service AppRole**
   - Identity: Frontend application
   - Policy: gui-policy
   - Permissions: Read encryption public key only

2. **cron-manager-service AppRole**
   - Identity: Background worker/scheduler
   - Policy: cron-manager-policy
   - Permissions: Full CRUD on secrets + encryption key access

3. **llm-orchestration-service AppRole**
   - Identity: LLM request handler
   - Policy: llm-orchestration-policy
   - Permissions: Read-only access to connection credentials

### Credential Lifecycle

```
Timeline: Credential Generation and Rotation

Day 0 (Initial Setup):
  vault-init: Generate role_id + secret_id
             ↓
  Write to: /agent/credentials/
             ↓
  vault-agent: Authenticate with Vault
             ↓
  Receive: Token (TTL: 1 hour, renewable)

Day 0+: Automatic Token Renewal:
  vault-agent: Monitor token expiration
             ↓
  Before expiry: Request token renewal
             ↓
  Vault: Extend token lifetime (1 hour)
             ↓
  Repeat: Continuous renewal cycle

Container Restart:
  vault-init: Check if Vault is sealed
             ↓
  If unsealed: Validate existing secret_ids
             ↓
  If valid: Reuse existing secret_id (no churn)
  If invalid: Mint new secret_id and write to disk
             ↓
  vault-agent: Re-authenticate with secret_id
             ↓
  New token issued and cached
```

**Security Benefit**: Short-lived tokens with automatic rotation limit the damage from token compromise.

---

## Authorization & Policy Model

### Policy-Based Access Control (PBAC)

Vault enforces authorization through **policies** - declarative rules that define what each identity can access.

#### Policy Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Vault Policy Layer                          │
└────────────────────────────────────────────────────────────────┘

Token (issued to vault-agent-gui)
   │
   ├─► Associated Policy: "gui-policy"
   │
   └─► Allowed Paths:
         secret/data/encryption/public_key     (read, list)
         secret/data/encryption/private_key    (denied)
         secret/data/llm/connections/*         (denied)
         secret/data/embeddings/connections/*  (denied)


Token (issued to vault-agent-cron)
   │
   ├─► Associated Policy: "cron-manager-policy"
   │
   └─► Allowed Paths:
         secret/data/llm/connections/*           (create, read, update, delete, list)
         secret/data/embeddings/connections/*    (create, read, update, delete, list)
         secret/data/encryption/public_key       (read, list)
         secret/data/encryption/private_key      (read, list)


Token (issued to vault-agent-llm)
   │
   ├─► Associated Policy: "llm-orchestration-policy"
   │
   └─► Allowed Paths:
         secret/data/llm/connections/*           (read, list)
         secret/data/embeddings/connections/*    (read, list)
         secret/data/encryption/*                (explicitly denied)
```

### Three-Tier Policy Structure

#### Tier 1: GUI Policy (Minimal Permissions)
**Purpose**: Allow frontend to encrypt user input

**Permissions**:
-  Read: `secret/data/encryption/public_key`
-  List: `secret/metadata/encryption/public_key`

**Denied**:
-  All other paths (deny-by-default)

**Use Case**: Frontend fetches public key to encrypt sensitive credentials (API keys, access keys) before sending to backend.

---

#### Tier 2: CronManager Policy (Secret Management)
**Purpose**: Write secrets to Vault and decrypt sensitive data

**Permissions**:
-  Create/Read/Update/Delete: `secret/data/llm/connections/*`
-  Create/Read/Update/Delete: `secret/data/embeddings/connections/*`
-  List: `secret/metadata/llm/connections/*`
-  List: `secret/metadata/embeddings/connections/*`
-  Read: `secret/data/encryption/public_key` (for verification)
-  Read: `secret/data/encryption/private_key` (for decryption)

**Use Case**: Receives encrypted credentials from GUI, decrypts them using private key, stores plaintext in Vault.

---

#### Tier 3: LLM Orchestration Policy (Read-Only)
**Purpose**: Retrieve credentials to make LLM API calls

**Permissions**:
-  Read: `secret/data/llm/connections/*`
-  Read: `secret/data/embeddings/connections/*`
-  List: `secret/metadata/llm/connections/*`
-  List: `secret/metadata/embeddings/connections/*`

**Explicitly Denied**:
-  Deny: `secret/data/encryption/*` (no access to encryption keys)

**Use Case**: Fetch AWS/Azure credentials to authenticate with LLM providers.

---

### Wildcard Path Patterns

Policies use **glob-style wildcards** (`*`) to match nested paths:

```
Pattern: secret/data/llm/connections/*

Matches:
   secret/data/llm/connections/azure_openai/testing/14
   secret/data/llm/connections/azure_openai/production/gpt-4o
   secret/data/llm/connections/aws_bedrock/testing/42
   secret/data/llm/connections/aws_bedrock/production/claude-3-sonnet

Does NOT match:
   secret/data/embeddings/connections/...
   secret/data/encryption/...
```

**Security Benefit**: One policy rule covers all current and future connection secrets without requiring policy updates.

### Deny-by-Default Security

```
Request Flow with Policy Enforcement:

vault-agent-llm sends request:
  GET /v1/secret/data/encryption/private_key

Vault Policy Engine:
  1. Check token's associated policies: "llm-orchestration-policy"
  2. Evaluate rules in policy:
     - Path: secret/data/encryption/*
     - Capability: deny
  3. Decision:  DENIED

Response: HTTP 403 Forbidden


vault-agent-cron sends request:
  GET /v1/secret/data/encryption/private_key

Vault Policy Engine:
  1. Check token's associated policies: "cron-manager-policy"
  2. Evaluate rules in policy:
     - Path: secret/data/encryption/*
     - Capability: read
  3. Decision:  ALLOWED

Response: HTTP 200 OK + secret data
```

---

## Vault Agent Architecture

### Why Vault Agents?

Traditional approach has security risks:

```
 Direct Vault Access (Insecure):

App Container
  │ env: VAULT_TOKEN=test_token_hvs.abc123xyz...
  │ env: VAULT_ADDR=http://vault:8200
  │
  └─► Direct HTTP call to Vault
       - Token in environment (logged, visible)
       - Token in memory dumps
       - Token in application code
       - No automatic renewal
```

Vault Agent proxy pattern solves these issues:

```
 Vault Agent Proxy (Secure):

App Container
  │ env: VAULT_ADDR=http://vault-agent-llm:8201
  │ (NO VAULT_TOKEN variable)
  │
  └─► HTTP call to Vault Agent (no token)
       │
       └─► Vault Agent injects token automatically
            │
            └─► Vault validates token
                 │
                 └─► Returns secret
```

### Proxy Pattern Benefits

1. **Token Isolation**: Application never sees or handles tokens
2. **Automatic Renewal**: Agent manages token lifecycle transparently
3. **Simplified Code**: No token management logic in application
4. **Audit Trail**: All requests logged through agent layer
5. **Single Point of Auth**: Centralized authentication handling

### Three-Agent Deployment

#### Agent 1: vault-agent-gui
```
Configuration:
  - Listens: 0.0.0.0:8202 (within bykstack network)
  - Auth: AppRole (gui-service)
  - Policy: gui-policy
  - Token Cache: /agent/gui-token/token

Connected Services:
  - GUI (React Frontend)

Token Lifecycle:
  - Token type: periodic (token_period 20m, no max-TTL)
  - Auto-renewal: Every ~13 minutes (~2/3 of period)
  - Re-auth: only on agent restart (never in steady state)
```

#### Agent 2: vault-agent-cron
```
Configuration:
  - Listens: 0.0.0.0:8203 (within bykstack network)
  - Auth: AppRole (cron-manager-service)
  - Policy: cron-manager-policy
  - Token Cache: /agent/cron-token/token

Connected Services:
  - CronManager (Python worker)

Token Lifecycle:
  - Token type: periodic (token_period 30m, no max-TTL)
  - Auto-renewal: Every ~20 minutes (~2/3 of period)
  - Re-auth: only on agent restart (never in steady state)
```

#### Agent 3: vault-agent-llm
```
Configuration:
  - Listens: 0.0.0.0:8201 (within bykstack network)
  - Auth: AppRole (llm-orchestration-service)
  - Policy: llm-orchestration-policy
  - Token Cache: /agent/llm-token/token

Connected Services:
  - LLM Orchestration Service (FastAPI)

Token Lifecycle:
  - Token type: periodic (token_period 1h, no max-TTL)
  - Auto-renewal: Every ~40 minutes (~2/3 of period)
  - Re-auth: only on agent restart (never in steady state)
```

### Token Caching and Auto-Renewal

```
Vault Agent Token Management Cycle:

┌─────────────────────────────────────────────────────────────┐
│                   Token Lifecycle                            │
└─────────────────────────────────────────────────────────────┘

T=0: Initial Authentication
     vault-agent reads credentials
        │
        ├─► POST /v1/auth/approle/login
        │   Body: { role_id, secret_id }
        │
        └─► Receives: { token, period: 3600s, renewable: true }   ← periodic token, no max-TTL
             │
             └─► Cache token in: /agent/llm-token/token


T≈40min: Proactive Renewal (~2/3 of period)
     vault-agent monitors expiration
        │
        ├─► POST /v1/auth/token/renew-self
        │   Header: X-Vault-Token: <current_token>
        │
        └─► Receives: { token, period: 3600s } (same token, period reset)
             │
             └─► Update cache: /agent/llm-token/token
             │
             └─► Repeats forever — a periodic token never hits a max-TTL,
                 so steady-state operation never needs approle/login again.


On agent restart only:
     vault-agent re-reads role_id + secret_id from disk
        │
        ├─► POST /v1/auth/approle/login   (secret_id must still be valid)
        │
        └─► New periodic token issued and cached


Application Request (anytime):
     App sends: GET http://vault-agent-llm:8201/v1/secret/data/llm/...
        │
        ├─► vault-agent intercepts request
        ├─► Injects header: X-Vault-Token: <cached_token>
        ├─► Forwards to: http://vault:8200/v1/secret/data/llm/...
        │
        └─► Returns response to application
```

### Transparent Authentication Injection

When an application makes a request:

```
Step 1: Application Code (No Token)
--------
import requests
response = requests.get(
    "http://vault-agent-llm:8201/v1/secret/data/llm/connections/azure_openai/testing/14"
)
# Note: No X-Vault-Token header sent!


Step 2: Vault Agent Intercepts
--------
vault-agent-llm receives request:
  - Request headers: { User-Agent: python-requests/2.31.0 }
  - No token present


Step 3: Agent Adds Authentication
--------
vault-agent-llm modifies request:
  - Add header: X-Vault-Token: vault_token_test
  - Forward to: http://vault:8200/v1/secret/data/llm/connections/azure_openai/testing/14


Step 4: Vault Validates
--------
Vault server:
  - Checks token validity
  - Looks up associated policy: llm-orchestration-policy
  - Evaluates path permission: secret/data/llm/connections/* → ALLOWED
  - Returns secret data


Step 5: Agent Forwards Response
--------
vault-agent-llm returns to application:
  - Status: 200 OK
  - Body: { "data": { "data": { "endpoint": "...", "api_key": "..." } } }


Application receives response:
  - Thinks it talked directly to Vault
  - Never handled or saw the token 
```

---

## Secret Storage Strategy

### KV v2 Secrets Engine

Vault uses the **Key-Value version 2** (KV v2) secrets engine, which provides:

1. **Versioning**: Every secret write creates a new version
2. **Audit Trail**: Track who changed what and when
3. **Rollback**: Restore previous versions if needed
4. **Soft Delete**: Deleted secrets can be recovered
5. **Metadata**: Store additional context (tags, timestamps)

### Path Hierarchy and Organization

```
vault (root)
└── secret/ (KV v2 mount point)
    ├── llm/
    │   └── connections/
    │       ├── azure_openai/
    │       │   ├── testing/
    │       │   │   ├── 14 → { connection_id, endpoint, api_key, deployment_name }
    │       │   │   ├── 15 → { ... }
    │       │   │   └── 16 → { ... }
    │       │   └── production/
    │       │       ├── gpt-4o → { connection_id, endpoint, api_key, ... }
    │       │       └── gpt-4o-mini → { ... }
    │       └── aws_bedrock/
    │           ├── testing/
    │           │   └── 14 → { connection_id, access_key, secret_key }
    │           └── production/
    │               ├── claude-3-sonnet → { ... }
    │               └── claude-3-opus → { ... }
    │
    ├── embeddings/
    │   └── connections/
    │       ├── azure_openai/
    │       │   └── testing/
    │       │       └── 14 → { connection_id, endpoint, api_key, model }
    │       └── aws_bedrock/
    │           └── testing/
    │               └── 14 → { connection_id, access_key, secret_key, model }
    │
    └── encryption/
        ├── public_key → { key: "-----BEGIN PUBLIC KEY-----...", algorithm: "RSA-OAEP", ... }
        └── private_key → { key: "-----BEGIN PRIVATE KEY-----...", algorithm: "RSA-OAEP", ... }
```

### Path Structure Logic

**Testing Environment:**
```
Pattern: secret/llm/connections/{platform}/{environment}/{connection_id}
Example: secret/llm/connections/azure_openai/testing/14

Why connection_id?
  - Multiple test configurations per platform
  - Easy to create/delete during development
  - Unique identifier for each test setup
```

**Production Environment:**
```
Pattern: secret/llm/connections/{platform}/{environment}/{model}
Example: secret/llm/connections/azure_openai/production/gpt-4o

Why model name?
  - One canonical credential per model
  - Predictable path for application lookups
  - Clear naming convention
```

### Version Control and Audit Trail

KV v2 automatically versions every write:

```
Example: Updating Azure API Key

Version 1 (Initial):
  Path: secret/data/llm/connections/azure_openai/testing/14
  Data: { 
    "connection_id": "14",
    "endpoint": "https://xxx.openai.azure.com/",
    "api_key": "old-key-abc123",
    "deployment_name": "gpt-4o-deployment"
  }
  Metadata: {
    "version": 1,
    "created_time": "2026-01-08T10:30:00Z",
    "created_by": "cron-manager-service"
  }


Version 2 (After Key Rotation):
  Path: secret/data/llm/connections/azure_openai/testing/14
  Data: { 
    "connection_id": "14",
    "endpoint": "https://xxx.openai.azure.com/",
    "api_key": "new-key-xyz789",  ← Updated
    "deployment_name": "gpt-4o-deployment"
  }
  Metadata: {
    "version": 2,
    "created_time": "2026-01-09T14:20:00Z",
    "created_by": "cron-manager-service"
  }


Accessing Versions:
  - Latest: GET /v1/secret/data/llm/connections/azure_openai/testing/14
  - Version 1: GET /v1/secret/data/llm/connections/azure_openai/testing/14?version=1
  - Version 2: GET /v1/secret/data/llm/connections/azure_openai/testing/14?version=2
```

**Security Benefit**: Full audit trail of credential changes with rollback capability.

---

## Access Control Matrix

### Service-to-Secret Mapping

| Service | LLM Connections | Embedding Connections | Public Key | Private Key | Token Lookup |
|---------|-----------------|----------------------|------------|-------------|--------------|
| **GUI (Frontend)** |  Denied |  Denied |  Read |  Denied |  Read |
| **CronManager** |  Full CRUD |  Full CRUD |  Read |  Read |  Read |
| **LLM Service** |  Read |  Read |  Denied |  Denied |  Read |

### Permission Boundaries

#### GUI Service
```
 Allowed Actions:
   - GET secret/data/encryption/public_key
   - LIST secret/metadata/encryption/public_key
   - GET auth/token/lookup-self (verify own token)

 Denied Actions:
   - Any operation on secret/data/llm/*
   - Any operation on secret/data/embeddings/*
   - GET secret/data/encryption/private_key
   - Any write operations

Use Case Flow:
   1. User enters API key in frontend
   2. Frontend fetches public key from Vault
   3. Frontend encrypts API key with public key
   4. Sends encrypted data to backend
```

#### CronManager Service
```
 Allowed Actions:
   - CREATE/READ/UPDATE/DELETE secret/data/llm/connections/*
   - CREATE/READ/UPDATE/DELETE secret/data/embeddings/connections/*
   - LIST secret/metadata/llm/connections/*
   - LIST secret/metadata/embeddings/connections/*
   - GET secret/data/encryption/public_key
   - GET secret/data/encryption/private_key
   - GET auth/token/lookup-self

 Denied Actions:
   - Modify encryption keys (read-only)
   - Access secrets outside defined paths

Use Case Flow:
   1. Receives encrypted credentials from frontend
   2. Fetches private key from Vault
   3. Decrypts credentials locally
   4. Stores plaintext credentials in Vault
   5. Returns success/failure to frontend
```

#### LLM Orchestration Service
```
 Allowed Actions:
   - GET secret/data/llm/connections/*
   - GET secret/data/embeddings/connections/*
   - LIST secret/metadata/llm/connections/*
   - LIST secret/metadata/embeddings/connections/*
   - GET auth/token/lookup-self

 Denied Actions (Explicit Deny):
   - Any operation on secret/data/encryption/* (cannot access keys)
   - Any write operations on secrets
   - Any access to secrets outside defined paths

Use Case Flow:
   1. Receives LLM request from user
   2. Determines required LLM provider (AWS/Azure)
   3. Fetches connection credentials from Vault
   4. Makes authenticated API call to LLM provider
   5. Returns LLM response to user
```

### Least Privilege Enforcement

```
Scenario: LLM Service Attempts Unauthorized Access

Request:
  GET http://vault-agent-llm:8201/v1/secret/data/encryption/private_key

Vault Decision Chain:
  1. Token extracted from request: test_token_hvs.CAESIFS1ZfMfAtwYd9LJ27A1nzg...
  2. Token lookup: Associated policy = "llm-orchestration-policy"
  3. Path evaluation: secret/data/encryption/private_key
  4. Policy rule match:
     path "secret/data/encryption/*" {
       capabilities = ["deny"]
     }
  5. Decision:  DENY

Response:
  Status: 403 Forbidden
  Body: { "errors": ["permission denied"] }

Audit Log:
  [2026-01-09T10:30:00Z] path=secret/data/encryption/private_key 
  action=read identity=llm-orchestration-service result=denied
```

**Security Principle**: Even if LLM service is compromised, attacker cannot access encryption keys.

---

## Initialization & Bootstrapping

### vault-init Container Workflow

The `vault-init` container is a one-time initialization container that sets up Vault on first deployment:

```
┌────────────────────────────────────────────────────────────────┐
│              vault-init Startup Sequence                        │
└────────────────────────────────────────────────────────────────┘

Step 1: Wait for Vault Health
   └─► Retry loop: Check http://vault:8200/v1/sys/health
       └─► Wait until Vault server is responsive

Step 2: Check Vault Status
   └─► Is Vault initialized? 
       ├─► NO → First Time Setup (Steps 3-10)
       └─► YES → Subsequent Deployment (Steps 11-12)

═══════════════════════════════════════════════════════════════════
FIRST TIME DEPLOYMENT
═══════════════════════════════════════════════════════════════════

Step 3: Initialize Vault
   └─► POST /v1/sys/init
       └─► Receives:
           - Unseal keys (5 keys, threshold 3)
           - Root token

Step 4: Unseal Vault
   └─► POST /v1/sys/unseal (3 times with different keys)
       └─► Vault becomes operational

Step 5: Enable KV v2 Secrets Engine
   └─► POST /v1/sys/mounts/secret
       Body: { type: "kv-v2" }

Step 6: Enable AppRole Authentication
   └─► POST /v1/sys/auth/approle
       Body: { type: "approle" }

Step 7: Create Policies
   └─► POST /v1/sys/policies/acl/gui-policy
   └─► POST /v1/sys/policies/acl/cron-manager-policy
   └─► POST /v1/sys/policies/acl/llm-orchestration-policy

Step 8: Create AppRoles
   └─► POST /v1/auth/approle/role/gui-service
   └─► POST /v1/auth/approle/role/cron-manager-service
   └─► POST /v1/auth/approle/role/llm-orchestration-service

Step 9: Generate Credentials
   └─► GET /v1/auth/approle/role/gui-service/role-id
   └─► POST /v1/auth/approle/role/gui-service/secret-id
   └─► Repeat for cron-manager and llm-orchestration

Step 10: Generate RSA Keypair
   └─► openssl genrsa -out private_key.pem 2048
   └─► openssl rsa -pubout -in private_key.pem -out public_key.pem
   └─► POST /v1/secret/data/encryption/public_key
   └─► POST /v1/secret/data/encryption/private_key

Step 11: Write Credentials to Shared Volumes
   └─► /agent/credentials/gui_role_id
   └─► /agent/credentials/gui_secret_id
   └─► /agent/credentials/cron_role_id
   └─► /agent/credentials/cron_secret_id
   └─► /agent/credentials/llm_role_id
   └─► /agent/credentials/llm_secret_id

═══════════════════════════════════════════════════════════════════
SUBSEQUENT DEPLOYMENT (Container Restart)
═══════════════════════════════════════════════════════════════════

Step 12: Check Vault Seal Status
   └─► GET /v1/sys/seal-status
       └─► If unsealed: Skip unseal steps

Step 13: Validate and Reconcile Secret IDs
   └─► For each role (gui, cron-manager, llm-orchestration):
       ├─► Test existing on-disk secret_id via AppRole login
       ├─► If valid: Reuse (no change to credential file)
       └─► If invalid/missing: Mint new secret_id and write to disk

Note: role_ids remain unchanged (static identifiers)
Note: Existing secrets and policies preserved
Note: RSA keypair NOT regenerated (preserved)
Note: Stable secret_ids across restarts reduce credential churn

═══════════════════════════════════════════════════════════════════
COMPLETION
═══════════════════════════════════════════════════════════════════

Step 14: Set File Permissions
   └─► chown vault:vault /agent/credentials/*
   └─► chmod 644 /agent/credentials/*

Step 15: Exit Successfully
   └─► Container stops with exit code 0
   └─► vault-agent containers start (depends_on: service_completed_successfully)
```

### Unseal Key Management

Vault starts in a **sealed** state and must be unsealed before use:

```
Sealed State:
  - Vault knows where data is stored
  - Vault cannot decrypt data (encryption key unknown)
  - All API operations return "Vault is sealed"

Unsealing Process:
  - Vault initialized with Shamir's Secret Sharing
  - Master key split into 5 parts (configurable)
  - Threshold: Any 3 of 5 keys can unseal
  - Each unseal key submitted separately

Security Model:
   No single person can unseal Vault alone
   Compromise of 2 keys is insufficient
   Distributed trust across operators

Current Implementation:
  - All 5 unseal keys stored in vault-data volume
  - Suitable for development/testing
  - Production: Use Vault Auto-Unseal with AWS KMS, Azure Key Vault, etc.
```

**Security Trade-off**: Current setup prioritizes ease of deployment over maximum security. For production, implement auto-unseal with cloud HSM.

### Root Token Handling

The root token has unlimited access to Vault:

```
Root Token Lifecycle:

1. Generation:
   - Created during vault init
   - Full superuser permissions
   - Never expires

2. Current Usage:
   - Used by vault-init script only
   - Performs initial setup tasks
   - Not stored in application containers

3. Security Best Practice:
   - Revoke root token after initialization
   - Use admin policies with limited scope instead
   - Regenerate root token only for emergency recovery

4. Production Recommendation:
   - Revoke: vault token revoke <root_token>
   - Recreate if needed: vault operator generate-root
```

### Credential File Security

```
File Permissions on Shared Volume:

/agent/credentials/
├── gui_role_id          (644 - readable by all agents)
├── gui_secret_id        (644 - readable by all agents)
├── cron_role_id         (644)
├── cron_secret_id       (644)
├── llm_role_id          (644)
└── llm_secret_id        (644)

Owner: vault:vault (UID 100, GID 1000)
Container Access: Read-only mounts in agent containers

Security Considerations:
   Files isolated within Docker volumes (not host filesystem)
   Agent containers mount as read-only
   Only vault-init has write access
   All agents can read all credentials (volume-level isolation only)

```

---

## Security Best Practices Implemented

### 1. No Direct Vault Access from Applications

```
 Implemented Pattern:

Application Container
  └─► Vault Agent Proxy (same network)
       └─► Vault Server (isolated network)

Benefits:
  - Application never handles tokens
  - Token rotation transparent to app
  - Reduced attack surface
  - Centralized authentication audit
```

### 2. Token Environment Variable Isolation

```
 Implemented Pattern:

docker-compose.yml (LLM Service):
  environment:
    - VAULT_ADDR=http://vault-agent-llm:8201
    # NO VAULT_TOKEN variable

```

### 3. Credential File Permissions

```
 Implemented Pattern:

vault-init sets permissions:
  chown vault:vault /agent/credentials/*
  chmod 644 /agent/credentials/*

Vault agents run as:
  user: vault (UID 100)

Why This Matters:
  - Non-root user execution
  - Principle of least privilege
  - Reduced container escape impact
```

### 4. Container User Restrictions

```
 Implemented Pattern:

cron-manager:
  user: "1000:1000"  # Non-root user

vault containers:
  user: vault (implicit)

Benefits:
  - Limits filesystem access
  - Prevents privilege escalation
  - Reduces blast radius of exploits
```

### 5. Health Check Strategies

```
 Implemented Pattern:

Vault Server Health Check:
  test: wget -q -O- http://127.0.0.1:8200/v1/sys/health
  interval: 5s
  retries: 20
  start_period: 10s

Vault Agent Health Check:
  test: test -f /agent/llm-token/token && test -s /agent/llm-token/token
  interval: 10s
  retries: 3
  start_period: 5s

Benefits:
  - Automated service recovery
  - Dependency ordering (agents wait for Vault)
  - Token presence validation
  - Monitoring integration ready
```

---

## Operational Security

### Container Restart Scenarios

#### Scenario 1: Vault Server Restart

```
Event: docker-compose restart vault

Impact:
   Vault data persists (vault-data volume)
   Vault automatically unseals (unseal keys in volume)
   Policies and secrets intact
   AppRole configurations intact

Agent Behavior:
  - Existing tokens remain valid
  - Agents reconnect automatically
  - No re-authentication needed (unless token expired)

Downtime:
  - ~5-10 seconds (health check dependent)
```

#### Scenario 2: Vault Agent Restart

```
Event: docker-compose restart vault-agent-llm

Impact:
   Credentials still available (vault-agent-creds volume)
   Agent re-authenticates automatically
   New token issued and cached

Application Behavior:
  - Brief connection failure during restart
  - Retry logic handles transient errors
  - No manual intervention required

Downtime:
  - ~3-5 seconds (agent startup time)
```

#### Scenario 3: Application Container Restart

```
Event: docker-compose restart llm-orchestration-service

Impact:
   No Vault changes needed
   Vault agent still running
   Tokens still valid

Application Behavior:
  - Reconnects to vault-agent-llm:8201
  - No authentication logic in app
  - Immediate secret access

Downtime:
  - Application startup time only
```

#### Scenario 4: Full System Restart

```
Event: docker-compose down && docker-compose up

Startup Order:
  1. vault (health check: vault ready)
  2. vault-init (runs setup, exits)
  3. vault-agent-* (wait for init completion)
  4. Applications (wait for agents)

vault-init Behavior:
  - Detects Vault already initialized
  - Skips initialization steps
  - Validates existing secret_ids (reuses if still valid)
  - Mints new secret_ids only if existing ones are invalid

Result:
   All services start with validated credentials
   Existing secrets preserved
   No manual intervention needed
   Stable secret_ids reduce unnecessary credential churn
```

### Token Regeneration Strategy

```
Current Implementation:

1. On Every Container Restart:
   └─► vault-init validates existing secret_ids
       ├─► If valid: Reuse (agents continue with same credentials)
       └─► If invalid: Mint new secret_id, agents re-authenticate

2. Token Lifecycle:
   └─► Issue: vault-agent authenticates (periodic token, token_period per role)
   └─► Use: Application makes requests
   └─► Renew: vault-agent renews within the period (~2/3 of period)
   └─► No max-TTL: renewal continues indefinitely
   └─► Re-issue: only on agent restart, via secret_id login

3. Security Benefits:
    Periodic tokens (period 1h LLM, 30m Cron, 20m GUI), renewed continuously
    Steady-state operation never re-runs approle/login (a stale secret_id
      cannot strand a running agent)
    Stable secret_ids (no unnecessary churn on restart)
    Compromised tokens limited to one un-renewed period
```

### Audit Logging Capabilities

Vault provides comprehensive audit logging (not currently enabled in configuration):

```
Enabling Audit Logs:

vault audit enable file file_path=/vault/logs/audit.log

Logged Information:
  - Timestamp of request
  - Client identity (AppRole, token)
  - Request path and method
  - Success or failure
  - Policy evaluation result
  - Response data (hashed for secrets)

Example Audit Entry:
{
  "time": "2026-01-09T10:30:00.123Z",
  "type": "response",
  "auth": {
    "entity_id": "llm-orchestration-service",
    "policies": ["llm-orchestration-policy"]
  },
  "request": {
    "path": "secret/data/llm/connections/azure_openai/testing/14",
    "operation": "read"
  },
  "response": {
    "status": 200
  }
}

Use Cases:
  - Security incident investigation
  - Compliance reporting
  - Anomaly detection
  - Access pattern analysis
```

### Monitoring Integration Points

```
Recommended Monitoring Metrics:

1. Vault Health:
   - Endpoint: http://vault:8200/v1/sys/health
   - Metrics: sealed status, initialized status

2. Token Usage:
   - Endpoint: http://vault:8200/v1/auth/token/lookup-self
   - Metrics: TTL remaining, creation time, policies

3. Secret Access Patterns:
   - Source: Audit logs
   - Metrics: Access frequency, denied requests, unique clients

4. Agent Health:
   - Endpoint: Container health checks
   - Metrics: Token file presence, agent uptime

5. Application Errors:
   - Source: Application logs
   - Metrics: Vault connection failures, 403 errors, timeouts

Integration with Grafana:
  - Loki for log aggregation
  - Prometheus for metrics (add vault exporter)
  - Alertmanager for notifications
```

---

## Architecture Summary

### Security Layers

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Network Isolation                                   │
│  - Vault on internal network only                             │
│  - No external routing                                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Authentication (AppRole)                            │
│  - Machine identities                                         │
│  - Role-based credentials                                     │
│  - Short-lived tokens                                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 3: Authorization (Policies)                            │
│  - Path-based access control                                  │
│  - Least privilege enforcement                                │
│  - Explicit deny rules                                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 4: Proxy Abstraction (Vault Agents)                    │
│  - Token isolation                                            │
│  - Automatic renewal                                          │
│  - Transparent authentication                                 │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 5: Application Layer                                   │
│  - No token handling                                          │
│  - Simple HTTP calls                                          │
│  - Automatic retry logic                                      │
└──────────────────────────────────────────────────────────────┘
```

### Defense in Depth

```
Attack Scenario: Compromised LLM Service Container

Attacker Capabilities:
   Cannot access Vault directly (network isolation)
   Cannot read other service tokens (volume isolation)
   Cannot access encryption keys (policy denial)
   Can read LLM connection credentials (authorized)

Blast Radius:
  - Limited to LLM/embedding connection secrets
  - Cannot modify secrets (read-only policy)
  - Cannot access private encryption key
  - Cannot pivot to other services

Mitigation:
  - Rotate compromised credentials immediately
  - Revoke vault-agent-llm token
  - Regenerate secret_id for llm-orchestration-service
  - Restart llm-orchestration-service container
```

---

## Conclusion

The Vault security architecture implements industry best practices for secrets management:

 **Network Segmentation**: Vault isolated on internal network
 **Strong Authentication**: AppRole with renewable tokens
 **Granular Authorization**: Path-based policies with least privilege
 **Proxy Pattern**: Applications never handle tokens directly
 **Automated Operations**: Self-healing agents and token renewal
 **Audit Capability**: Full request logging available
 **Defense in Depth**: Multiple security layers prevent single point of failure

This architecture provides a secure foundation for managing sensitive credentials in the RAG-Module while maintaining operational simplicity and developer productivity.

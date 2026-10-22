# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The LLM Module is the LLM-orchestration component of the Bürokratt (Estonian government chatbot) stack.
A FastAPI service (`src/`, port 8100) classifies each user query and routes it to one of five workflows
(Service, API-Tool, Context, RAG, OOD). The repo also carries the admin GUI (`GUI/`, React + Vite), a
Node SSE relay (`notification-server/`), Bürokratt DSL for the surrounding Ruuter/DataMapper/Resql/
CronManager containers (`DSL/`), Liquibase migrations (`DSL/Liquibase*/`), Vault bootstrap, Docker
Compose variants and Helm charts (`kubernetes/`).

Start with `docs/ARCHITECTURE.md` and `docs/README.md` (the doc index) before touching a workflow.
`.github/copilot-instructions.md` contains older, RAG-only architecture notes; the tool-classifier
material below supersedes it, but its conventions section is still accurate.

## Commands

Python is pinned to 3.12.10 and managed by `uv`. Always go through `uv run`; never call `python`,
`pytest` or `pip` directly.

```bash
uv python install 3.12.10
uv sync --frozen                      # create .venv from uv.lock (use `uv sync -p 3.12.10` if lock is stale)
uv run pre-commit install             # gitleaks + uv-lock hooks

# Run the API locally (services such as Vault/Redis/Qdrant normally come from docker compose)
uv run uvicorn src.llm_orchestration_service_api:app --reload --port 8100

# Unit tests (fast, no Docker needed)
uv run pytest tests/ -v
uv run pytest tests/test_query_validator.py -v                 # one file
uv run pytest tests/test_tool_classifier.py -k "keyword" -v    # one test by name

# Integration tests: spin up docker-compose-test.yml via testcontainers, need Azure OpenAI secrets
docker compose -f docker-compose-test.yml build
uv run python -m pytest tests/integration_tests/ -v --tb=short --log-cli-level=INFO

# LLM evaluation (deepeval) and red-team suites, also secret-gated
uv run pytest tests/deepeval_tests/standard_tests.py -v --tb=short
uv run pytest tests/deepeval_tests/red_team_tests.py -v --tb=short

# Lint / format / types (all enforced in CI on every PR)
uv run ruff format --check            # `uv run ruff format` to apply; layout-only, safe
uv run ruff check .                   # fix findings by hand; see below
uv run ruff rule ANN204               # explain a rule code
uv run pyright                        # src/ only; tests/ are excluded

# Full stack
docker compose up -d
curl http://localhost:8100/health
```

GUI (`cd GUI`): `npm install --legacy-peer-deps`, `npm run dev` (port 3001), `npm run lint`,
`npm run build`. CI runs `npx eslint "src/**/*.{ts,tsx}" --quiet`.

### Dependency rules

- Add with `uv add "pkg>=x.y,<x.(y+1)"`, remove with `uv remove`, then `uv sync --reinstall`.
- Commit `pyproject.toml` and `uv.lock` together; CI runs `uv sync --frozen` and fails otherwise.
- Never edit `uv.lock` by hand, never introduce a `requirements.txt`, never `uv pip install` for
  permanent deps.

### Lint and type rules that bite

- `ruff check --fix` is forbidden by project policy (autofixes can change control flow). Fix manually.
- Ruff selects `E4 E7 E9 F B T20 N ANN ERA PERF`, line length 88, double quotes. `print()` is a lint
  error in `src/`; use `loguru` (or `LokiLogger`). `ANN` and `T201` are relaxed for `tests/`.
- Pyright runs in `standard` mode on `src/` only. Pydantic models validate at API boundaries
  (`src/models/request_models.py` uses camelCase fields by API contract, exempted from N815).
- Per-file `ANN401` exemptions in `pyproject.toml` exist for DSPy/`Any` pass-throughs; add a new one
  with a comment rather than sprinkling `# type: ignore`.

### Git conventions

- Commit messages must match `^(feat|fix|chore|docs|refactor|style|test)\([0-9]+\): .+$`, e.g.
  `feat(123): add follow-up detector`. The husky `commit-msg` hook enforces this.
- Branches (Sept 2026): `main` is the latest published version; `v1.0.0` collects fixes for it and
  is merged into `main`; `v2.0.0` collects new features. Historic flow was `wip` → `dev` → `main`,
  and several CI triggers still name those branches: integration tests run on PRs to `wip`,
  deepeval on PRs to `wip-eval`, lint/format/pyright/uv-lock on every PR. Pushes to `main` build
  and push images tagged from `release.env` and create a GitHub release. Feature branches usually
  live on a personal fork, so a PR against upstream is cross-fork (no secrets in CI).

## Architecture

### Import paths (important)

The container sets `PYTHONPATH=/app/src` and code runs from the repo root, so modules are imported
under two spellings: `from src.utils.x import ...` and `from utils.x import ...` (likewise
`models.`, `tool_classifier.`, `llm_orchestrator_config.`, `contextual_retrieval.`). Both resolve to
*separate module objects*. Exception classes and singletons must be imported via the same spelling
on both the raising and the catching side (see the note in `src/llm_orchestration_service_api.py`
about `ConfigurationError`). Match the spelling already used in the file you are editing.
`tests/conftest.py` adds both the repo root and `src/` to `sys.path`.

### Request lifecycle

`src/llm_orchestration_service_api.py` exposes `/orchestrate`, `/orchestrate/test` (returns chunks,
used by the GUI test page), `/orchestrate/stream` (SSE), `/orchestrate-eval`, `/embeddings`,
`/generate-context`, `/embedding-models`, `/cache/clear`, `/prompt-config/refresh`, `/health`.
It builds a single `LLMOrchestrationService` (`src/llm_orchestration_service.py`) at startup, along
with the Redis-backed stores and a rate limiter. Since Oct 2026 that class is a thin facade: the
work lives in `src/orchestration/` (`request_preamble.py` validation/language/connection resolution,
`component_factory.py` per-request LLM manager/guardrails/retriever construction,
`guardrails_coordinator.py` input/output checks and the blocked-answer replacement,
`rag_pipeline.py` the RAG-only path, `cost_budget.py`, `inference_storage.py`,
`response_builders.py`) and in `src/vector_indexer/indexer_support.py` (`/embeddings`,
`/generate-context`, `/embedding-models`). Many service methods are one-line delegations kept for
compatibility, so grep for the implementation in those modules, not the facade.

`LLMOrchestrationService.process_orchestration_request` / `stream_orchestration_response`:

1. Language detection, then a `#service,` or `#common_service,` prefix shortcut that executes a multi-step service directly.
2. Cheap query validation (`utils/query_validator.py`) before any component is initialised.
3. Component initialisation per request: `LLMManager` bound to the connection resolved for
   `request.environment`, guardrails, retriever, generator.
4. Input guardrails (NeMo Guardrails via DSPy adapter, `src/guardrails/`).
5. If `TOOL_CLASSIFIER_ENABLED` is true, `ToolClassifier.classify` + `route_to_workflow`; otherwise
   the legacy RAG-only pipeline (`src/orchestration/rag_pipeline.py`). Errors in the classifier
   fall back to RAG.
6. Output guardrails, cost logging to Langfuse, budget update, inference storage via Ruuter.

### Tool classifier and workflows (`src/tool_classifier/`)

- `classifier.py`: hybrid dense (3072-dim embeddings) + sparse (BM25, `sparse_encoder.py`) search
  over indexed intent/endpoint examples in Qdrant. Produces a `ClassificationResult` with a
  `WorkflowType` and metadata that becomes the workflow `context`. The service-intent dense search
  embeds the query with a hard-coded `environment="production"` and no connection id
  (`_get_query_embedding`), so it needs Resql to resolve the production connection; the API-tool
  search (`api_semantic_searcher.py`) uses the request's environment and connection id.
- Layer order is fixed in `enums.py::WORKFLOW_LAYER_ORDER`: SERVICE → API_TOOL_CALLING → CONTEXT →
  RAG → OOD. Every executor subclasses `BaseWorkflow` (`base_workflow.py`) and implements
  `execute_async` and `execute_streaming`. **Returning `None` means "I cannot handle this"** and
  the classifier falls through to the next enabled layer; RAG/OOD must never return `None`.
- `workflows/`: `service_workflow.py` (map to a backend service, multi-step via the `#service,` prefix),
  `api_tool_workflow.py` (agentic parameter collection in `agentic_loop.py` /
  `multi_agentic_loop.py`, intent decomposition, `api_caller.py`, response formatting; session state
  in Redis via `utils/api_tool_session_store.py`, two-tier response cache in `utils/atc_cache_store.py`),
  `context_workflow.py` (greetings and history answers, `context_analyzer.py`,
  `follow_up_detector.py`), `rag_workflow.py` (wraps the legacy RAG pipeline), `ood_workflow.py`.
- Feature flags in `src/llm_orchestrator_config/feature_flags.py` are read from env at import time
  (`TOOL_CLASSIFIER_ENABLED` defaults to **false**, individual workflow flags default to true).

### RAG pipeline

`prompt_refine_manager/prompt_refiner.py` (DSPy, 5 variants + original) → `contextual_retrieval/`
(per-query Qdrant semantic search + BM25, RRF fusion in `rank_fusion.py`, thresholds and top-K in
`config/contextual_retrieval_config.yaml` and `constants.py`) → `response_generator/response_generate.py`
(DSPy, streaming via `dspy.streamify`). Optimised DSPy programs are loaded from
`src/optimization/optimized_modules/` by `optimized_module_loader.py`; the optimizers and datasets
live under `src/optimization/` (see its README).

### LLM providers and secrets (`src/llm_orchestrator_config/`)

- `llm_manager.py`: `LLMManager` is a **process-wide** singleton. The first construction that
  succeeds fixes the environment and `connection_id` for the whole process; later
  `LLMManager(...)` calls return that instance and ignore their arguments. A construction that
  fails (e.g. no connection resolvable) leaves it uninitialised so the next call retries. Model/
  provider defaults come from `config/llm_config.yaml`; credentials come from Vault via
  `vault/secret_resolver.py` (5-minute TTL cache, last-known-good fallback). Providers implement
  `providers/base.py` (Azure OpenAI, AWS Bedrock).
- Use `LLMManager.use_task_local()` (a `dspy.context`) for per-request LM selection; do not call
  `dspy.configure` per request.
- Vault paths end in the connection's `vault_uuid`; the id is fetched from Ruuter by
  `utils/connection_id_fetcher.py`. The service talks to Vault through a vault-agent sidecar
  (`VAULT_ADDR=http://vault-agent-llm:8201`), never with a static token. Credentials are RSA-2048
  encrypted by the GUI before storage and decrypted in CronManager (`utils/decrypt_vault_secrets.py`).
- Admin-editable prompts are fetched from Ruuter by `utils/prompt_config_loader.py` (TTL cache,
  invalidated by `/prompt-config/refresh`).

### Offline indexers (run by CronManager, not by the API)

`src/vector_indexer/main_indexer.py` (knowledge base → Qdrant, `--signed-url`),
`src/api_tool_indexer/main_indexer.py` (API endpoints for the API-Tool workflow) and
`src/intent_data_enrichment/main_enrichment.py` (LLM-enriched intent examples) are CLI scripts
launched by shell scripts in `DSL/CronManager/script/` inside the cron-manager container. They call
back into the API's `/generate-context` (LLM enrichment: prose + "Example queries" that become the
indexed points) and `/embeddings` endpoints, sending `environment=production` and, by default, no
connection id, so the service resolves the production connection via Resql. Both endpoints honour an
explicit `connection_id` in the request. Their pip installs are listed in those shell scripts, so a
new dependency for an indexer must be added there as well as in `pyproject.toml`.

### Surrounding stack

Ruuter (public 8086 / private 8088) is the API gateway; its DSL under `DSL/Ruuter.*/rag-search/`
defines the HTTP routes the GUI and chatbot call, which then hit Resql (SQL in `DSL/Resql/`),
DataMapper (Handlebars templates in `DSL/DMapper/`) or this service. Internal URLs are listed in
`constants.ini`. Adding an API-facing feature usually means touching a Ruuter YAML, a Resql SQL file
and possibly a Liquibase changelog, not just Python. Schema changes go in
`DSL/Liquibase_production/changelog/` with a matching `-rollback.sql`.

### Tests

- `tests/test_*.py` are unit tests and run offline (they mock LLMs, Redis and Qdrant). All pass on
  `v1.0.0` as of Oct 2026 (an earlier `ToolClassifier.classify()` signature drift that failed 25
  of them was fixed upstream). `tests/test_api_tool_eval_suite.py` checks the API-tool deepeval
  suite's own logic against a stub service, no Docker or secrets needed.
- `tests/integration_tests/` uses `testcontainers` to start `docker-compose-test.yml`, bootstraps
  Vault from `test-vault/`, writes a fixed `TEST_VAULT_UUID` connection and exercises indexing,
  inference and connection management end to end. Requires Azure OpenAI secrets in the environment.
- `tests/deepeval_tests/` and `tests/api_tool_eval/` are LLM-quality evaluations, not CI gates for
  ordinary PRs. The deepeval workflow only triggers on PRs to `wip-eval`, and PRs from forks get no
  secrets, so it effectively runs only for branches pushed to the upstream repo.

### Running the deepeval stack locally (`tests/deepeval_tests/`)

The session-scoped `rag_stack` fixture in `tests/deepeval_tests/conftest.py` starts
`docker-compose-eval.yml` via testcontainers, bootstraps Vault (AppRole + a single seeded connection
`evalconnection-1`), downloads pre-computed RAG embeddings from Azure Blob Storage and indexes them.
Everything below was learned the hard way; check it before debugging anything else.

- **Secrets.** Put them in a git-ignored `.env` and run with `uv run --env-file .env ...`
  (values contain `;`, so do not `source` the file). Needed: `AZURE_OPENAI_ENDPOINT`,
  `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` (chat), `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`,
  `AZURE_OPENAI_DEEPEVAL_DEPLOYMENT` (LLM judge), `AZURE_STORAGE_CONNECTION_STRING`,
  `AZURE_STORAGE_CONTAINER_NAME` (`test-embeddings`), `AZURE_STORAGE_BLOB_NAME`
  (`test_embeddings.json`, underscore), plus `SALT`, `ENCRYPTION_KEY`, `NEXTAUTH_SECRET`, which only
  feed the local Langfuse containers (any generated values work). The blob download is mandatory
  even for tests that never use RAG embeddings; the fixture raises without it.
- **Judge model.** Run `uv run deepeval set-azure-openai ...` once (see the workflow file for the
  flags); it writes `.deepeval/.deepeval` in the repo root (git-ignored).
- **Chat deployment must not be a reasoning model.** `providers/azure_openai.py` builds
  `dspy.LM("azure/<deployment>")` with temperature/max_tokens from `llm_config.yaml`; DSPy rejects
  gpt-5/o1/o3/o4 deployments unless temperature is 1.0 and max_tokens ≥ 16000, so every LLM call
  503s. Use a `gpt-4o-mini`-class deployment. The judge deployment may be a reasoning model.
- **Images are not rebuilt.** The fixture runs `docker compose up --wait` without `--build`, so a
  second local run silently uses the previous image. After any `src/` change run
  `docker compose -f docker-compose-eval.yml build llm-orchestration-service` first. CI is unaffected
  (fresh runner).
- **Run artifacts to clean up.** The fixture writes AppRole creds into the *tracked* files
  `test-vault/agents/llm/role_id` and `secret_id` (restore with `git checkout --`), creates
  `test-vault/agent-out/`, downloads the 165 MB `tests/data/test_embeddings.json`, and the API-tool
  tests write `api_tool_test_results.json` to the CWD. All but the two tracked credential files are
  git-ignored.
- **Ports.** Only host ports the fixture itself needs are published; Redis deliberately has none.
  Check `docker compose -f docker-compose-eval.yml config | grep published` against `ss -ltn`.
- **No Ruuter/Resql in the eval stack.** Anything that resolves the production connection via
  `utils/connection_id_fetcher.py` fails there: startup pre-warming of shared guardrails (falls
  back to per-request init, harmless), the service workflow's dense search (hence
  `SERVICE_WORKFLOW_ENABLED=false` in the compose), and formerly `/generate-context` (fixed to
  honour the request's `connection_id`). Requests must always carry
  `connection_id: evalconnection-1`; without one the Vault path builder raises.
- **Eval-only response fields.** With `EVAL_MODE=true`, `/orchestrate-eval` returns
  `retrieval_context` (RAG chunks), `tool_calls` (resolved API-tool calls: name, method, url,
  input_parameters, success, status_code, cache_source) and `workflow` (which classifier layer
  answered). The API-tool deepeval tests score `tool_calls` and assert `workflow` instead of
  parsing the prose answer, which stopped being JSON in May 2026. Caveat: when the output
  guardrail blocks an answer, `GuardrailsCoordinator.handle_output` builds a fresh replacement
  response; `tool_calls` is carried over onto it (fixed after the Estonian intent-switch scenario
  exposed it), `retrieval_context` is not, and the `TestOrchestrationResponse` branch carries
  neither; `workflow` is tagged afterwards on both shapes.
- **Guardrails run on the base prompt.** The newest optimised guardrails module
  `src/optimization/optimized_modules/guardrails/guardrails_optimized_20260424_065431.json` is
  committed without its `_config.yaml` (the older three have one), so the loader logs "Optimized
  module found but no extracted config" and falls back to the base prompt in every environment
  that ships this image. A benign Estonian address answer was blocked by it in the eval.
- **Service logs are truncated.** The fixture dumps `docker logs --tail 5000` for the orchestration
  service (200 for the others) at teardown; 200 hid every agentic-loop diagnosis, so keep it high or
  dump to a file when diagnosing loop behaviour.
- **MinIO.** Docker Hub's `minio/minio` is no longer pullable; the eval compose uses
  `quay.io/minio/minio`. The other compose files and the Helm chart still reference Docker Hub.
- **Diagnosing seeding failures.** Fixture output is loguru → stderr, captured per test and never
  shown for skipped tests. Rerun with `-s` to see the `Seeded endpoint ... (message)` lines and the
  service logs the fixture dumps at teardown.

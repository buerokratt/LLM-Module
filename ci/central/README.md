# Central image builds

The feature branch builds the reviewed Central source on Linux/amd64. Pushes to
`feat/central-opensearch-migration` publish to GHCR with the full commit tag
`sha-<40-character-commit>`. Pull requests and manual runs build without publishing
unless the manual publish input is selected. No workflow deploys to Kubernetes.

`images.json` lists build components and pinned external source contexts. Run
`python3 ci/central/check.py` before pushing. Use `python3 ci/central/build.py
--component NAME --plan` to inspect a recipe. Builds require Docker Buildx and set
OCI source/revision labels; NoOps verifies these labels and resolves image digests.
Private producer packages must grant the NoOps repository Actions read access.
The cluster pull identity also needs read access to every resulting package.

NoOps locks these exact source commits and assembles the operator package. Build
outputs and credentials are ignored; keep local operator bindings outside Git.

## Offline guardrails embeddings

The Central LLM image includes the existing NeMo default MiniLM embedding model,
not a generative model. `guardrails-model.json` pins the source revision and every
artifact checksum, including upstream attribution. Build-time fetching needs no
provider credentials. NeMo uses the exact local model path with local-only loading.
Generation-enabled startup verifies checksums and loads the embedding model before
serving traffic; `/health` exposes `guardrails_embedding_loaded`. Missing/corrupt
artifacts fail startup. Vault/provider configuration is preserved.

CI starts real NeMo embeddings as UID1000 with networking disabled and both model
download methods forbidden, plus missing/corrupt-artifact negatives. Existing241
application regressions run offline. Generation remains opt-in through
`LLM_GENERATION_ENABLED`; provider-backed acceptance is an operator test.

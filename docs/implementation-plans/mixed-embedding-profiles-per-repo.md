# Mixed Embedding Profiles and Non-Blocking Refresh Queue

## Summary

- Add reusable named embedding profiles with per-repo assignment, so one `agent-context` instance can split codebases across two providers while preserving mixed-scope semantic search.
- Keep existing single-provider configs and single-provider indexes working by normalizing the old config form into one default profile and using legacy snapshot fingerprint data as a fallback.
- Make CLI `refresh-one` and MCP `index_codebase` non-blocking by routing both through one service-owned background scheduler that merges new repo requests into the currently active worker instead of spawning back-to-back scans.

## Public Interfaces

- Replace the single global embedding provider setting with named profiles plus repo assignments:

```toml
[embedding]
default_profile = "hosted"

[embedding.profiles.hosted]
provider = "voyage"
model = "voyage-code-3"
[embedding.profiles.hosted.voyage]
api_key_env = "VOYAGE_API_KEY"

[embedding.profiles.local]
provider = "openai"
model = "text-embedding-nomic-embed-text-v1.5"
[embedding.profiles.local.openai]
api_key_env = "LM_STUDIO_API_KEY"
base_url = "http://127.0.0.1:1234/v1"

[[embedding.assignments]]
repo = "/absolute/path/to/repo-a"
profile = "hosted"

[[embedding.assignments]]
repo = "/absolute/path/to/repo-b"
profile = "local"
```

- Keep the current `[embedding] provider/model` shorthand. Normalize it internally to one default named profile so old configs continue to load unchanged.
- Add repo-level embedding metadata to status surfaces: `embeddingProfile`, `configuredEmbeddingFingerprint`, `storedEmbeddingFingerprint`, and `embeddingMismatchReason` when relevant.
- Change `refresh-one` into an enqueue command. Add optional `--listen` defaulting to `127.0.0.1:8765` so the CLI can find the local service when the bind is non-default.
- Add one loopback-only local HTTP enqueue endpoint on the service, used by `refresh-one`. `refresh-one` must exit quickly after enqueue and must return a clear non-zero error if the service is unavailable.
- Keep `refresh-all` and `reindex-all` blocking.

## Key Changes

### Config and Snapshot

In [src/config.rs](/Users/jeremy/dev/agent-context-mcp/src/config.rs) and [src/snapshot.rs](/Users/jeremy/dev/agent-context-mcp/src/snapshot.rs):

- Introduce `EmbeddingProfileConfig` and `EmbeddingAssignment`.
- Resolve each configured repo to one effective profile via explicit assignment or `default_profile`.
- Normalize assignment repo paths relative to the config file, require assigned profiles to exist, reject assignments for repos not present in configured groups, and reject conflicting effective assignments for the same repo across groups.
- Add repo-level `embeddingProfile` and `embeddingFingerprint` fields to snapshot entries.
- Treat the legacy top-level snapshot fingerprint as a backward-compat fallback only when repo-level embedding metadata is absent.

### Engine Behavior

In [src/engine/mod.rs](/Users/jeremy/dev/agent-context-mcp/src/engine/mod.rs):

- Replace the single `EmbeddingClient` with a registry keyed by profile name.
- Cache dimension and fingerprint per profile.
- Resolve repo profile before indexing, and create or recreate that repo's Milvus collections using that profile's dimension.
- Change identity checks from one global embedding fingerprint to repo-scoped compatibility checks, so moving one repo to another profile invalidates only that repo.
- During `search_scope` and `search_symbols`, group repos by effective profile, compute one query embedding per distinct profile, and pass the matching vector into dense and symbol-semantic search for that repo group.
- Keep the current dense-score fusion formula unchanged.
- If one profile is broken, return repo-scoped partial errors while still serving unaffected repos.

### Background Indexing Scheduler

In [src/mcp_server.rs](/Users/jeremy/dev/agent-context-mcp/src/mcp_server.rs) plus CLI handoff:

- Introduce one service-owned `IndexCoordinator` with:
  - `pending` map keyed by repo
  - `running` set keyed by repo
  - one worker-active flag or task handle
- Store per pending repo:
  - repo path
  - merged `force` flag using OR semantics
  - merged execution mode where `ExplicitRefresh` wins over `Standard`
- Route MCP `index_codebase` and local HTTP enqueue requests through the same coordinator.
- Change CLI `refresh-one` to resolve nothing locally except service URL; it should POST the request to the service and render the enqueue result.
- Mark accepted repos as `status="indexing"` with `indexStatus="queued"` immediately, then flip them to `running` when work begins.
- Drain pending repos into a single long-lived background run, index repos one at a time using a repo-scoped internal helper that preserves the stored `force` and `IndexExecutionMode`, then re-check pending before exiting.
- If a new request arrives for a repo still pending, merge it in place and return immediately.
- If a new request arrives for a repo already actively running, return it as `already_running` and do not schedule a second pass.
- If a new request arrives for a different repo while the worker is active, add it to `pending` so the same worker picks it up before exit.
- Keep collection names repo-based; profile changes trigger repo-scoped drop/recreate/reindex, not collection renaming.

### Operator Surfaces and Docs

- Update `doctor` to validate every configured embedding profile, not just one global provider.
- Expand status and MCP structured responses with repo-level embedding metadata and queue state details.
- Update `README.md`, `config.example.toml`, `refresh-one` docs, and MCP tool docs to explain named profiles, per-repo assignment, mixed-provider search, enqueue-only `refresh-one`, and the requirement that the local service be running.

## Test Plan

### Config and Compatibility

- old single-provider config still loads and resolves all repos to one default normalized profile
- named profiles plus assignments parse correctly
- unknown profile, duplicate/conflicting assignment, and assignment to an unconfigured repo are rejected
- old snapshots without repo-level embedding metadata remain compatible under the legacy single-profile setup

### Repo Identity and Indexing

- changing one repo to a different profile marks only that repo incompatible until reindex
- reindexing that repo recreates its collections with the new profile's dimension and stores the new repo-level fingerprint
- mixed-profile scopes still return results from unaffected repos when one profile is misconfigured

### Queueing and Non-Blocking Behavior

- `refresh-one` returns immediately after a successful enqueue and does not wait for indexing completion
- `refresh-one` returns a clear error when the local service is unavailable
- repeated requests for the same pending repo merge without spawning a second worker
- requests for a different repo during an active run are absorbed and processed by the same worker before it exits
- requests for an already-running repo return `already_running` and do not trigger a rerun
- queued repos surface as `indexing` with `indexStatus="queued"` before they become `running`

### MCP and Status Surfaces

- `index_codebase` and CLI enqueue responses report started, queued, merged, and already-running repos consistently
- `doctor` reports all configured profiles and repo-level embedding mismatches
- status and MCP responses serialize `embeddingProfile`, fingerprint fields, and mismatch reason correctly

## Assumptions

- Provider choice is per repo, not per group.
- Mixed-provider scopes remain fully supported via per-profile query fan-out and merged ranking.
- Named profiles are generic, even if the immediate use case is only two providers.
- `refresh-one` and MCP `index_codebase` become enqueue-only; `refresh-all` and `reindex-all` remain blocking.
- The background queue is service-memory only. If the service restarts, pending work is lost and in-progress repos are marked failed the same way interrupted indexing is handled today.
- `refresh-one` requires the local service to be running and will not auto-start it or fall back to inline indexing.

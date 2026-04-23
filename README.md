# agent-context

`agent-context` is a Rust-native MCP server for code intelligence across one or more local repositories.

It combines:

- Milvus for dense semantic retrieval
- a local Tantivy index for lexical and path-aware search
- a local SQLite symbol store for definitions and outlines
- a single shared HTTP MCP endpoint for editors and agents

The result is one local service that agents can use for:

- semantic and hybrid code search
- exact symbol and definition lookup
- indexed file outlines
- multi-repo search across named groups

## Why it exists

Most local code-search setups are either:

- thin wrappers around shell tools
- editor-specific features that do not generalize to MCP clients
- Node-based MCP servers with their own background process churn

`agent-context` keeps the MCP layer native, keeps indexing under explicit local control, and exposes a tool surface that works the same way from Codex, Claude, VS Code/Copilot, or the CLI.

## Scope

V1 is intentionally opinionated:

- macOS-first
- Milvus required
- embedding providers: Voyage, OpenAI, and Ollama
- Homebrew install plus `brew services` is the preferred local service model
- MCP is the primary product interface

This is aimed at power users who are comfortable running Docker and setting API keys in their shell environment.

## Agent-assisted install

Assume an agent is performing setup on behalf of a user.

Required inputs the agent should confirm or infer before running commands:

- the absolute repo paths to index
- the embedding provider: `voyage`, `openai`, or `ollama`
- the provider credential source:
  - `VOYAGE_API_KEY`
  - `OPENAI_API_KEY`
  - or a running Ollama server

The recommended install order is:

1. Install the Homebrew tap and formula.
2. Start Milvus.
3. Run `agent-context init`.
4. Export or validate provider credentials.
5. Run `agent-context doctor`.
6. Start the MCP server with `brew services`.
7. Print and apply client MCP config.
8. Run `refresh-all` or `reindex-all`.
9. Install post-commit hooks for repos that should auto-refresh on commit.

An agent should verify each step before moving on.

## Quickstart

1. Install the binary and verify it is on `PATH`.

   - Preferred:

   ```bash
   brew tap jeremymefford/agent-context-mcp
   brew install agent-context
   ```

   - Release assets: GitHub Releases
   - Fallback: `cargo install --path .`

   Verify:

   ```bash
   agent-context --help
   ```

2. Start Milvus.

   ```bash
   docker compose -f docker/milvus-compose.yml up -d
   ```

3. Write a starter config.

   ```bash
   agent-context init --provider voyage --repo /absolute/path/to/repo
   ```

   Verify the config was written:

   ```bash
   ls -l ~/Library/Application\ Support/agent-context/config.toml
   ```

4. Export credentials if needed.

   ```bash
   export VOYAGE_API_KEY=...
   # or
   export OPENAI_API_KEY=...
   ```

5. Validate the local setup.

   ```bash
   agent-context doctor
   ```

   Do not proceed until `doctor` reports no blocking issues.

   ```bash
   brew services start agent-context
   ```

   Verify:

   ```bash
   brew services list | grep agent-context
   ```

   Health:

   ```bash
   curl http://127.0.0.1:8765/health
   ```

   Manual fallback:

   ```bash
   agent-context serve --listen 127.0.0.1:8765
   ```

7. Connect an MCP client.

   ```bash
   agent-context print-mcp-config --client codex
   ```

8. Index your repos.

   ```bash
   agent-context refresh-all
   # or for a clean rebuild
   agent-context reindex-all
   ```

   Verify with:

   ```bash
   agent-context search workspace "health check"
   ```

## MCP tool surface

The MCP server is the main integration surface.

Current tools:

- `list_scopes`
- `index_codebase`
- `search_code`
- `search_symbols`
- `get_file_outline`
- `explain_search`
- `clear_index`
- `get_indexing_status`

Preferred agent routing:

- `list_scopes` first in unfamiliar workspaces
- `search_symbols` first for exact definition lookup
- `get_file_outline` once the target file is known
- `search_code` for broader semantic or hybrid discovery

## CLI commands

Setup and repair:

- `agent-context init`
- `agent-context doctor`
- `agent-context install-hook <repo>`
- `agent-context print-mcp-config --client codex|claude|copilot`

Indexing and serving:

- `agent-context refresh-one <scope-or-absolute-repo>`
- `agent-context refresh-all`
- `agent-context reindex-all`
- `agent-context search <scope-or-absolute-repo> "<query>"`
- `agent-context list-tools`
- `agent-context serve --listen 127.0.0.1:8765`

## Config model

The canonical config shape is:

```toml
snapshot_path = "~/Library/Application Support/agent-context/state/snapshot.json"
index_root = "~/Library/Application Support/agent-context/index-v1"
default_group = "workspace"

[embedding]
provider = "voyage" # or openai / ollama
model = "voyage-code-3"

[embedding.voyage]
api_key_env = "VOYAGE_API_KEY"
# key_file = "~/Library/Application Support/agent-context/voyage_key"

[milvus]
address = "127.0.0.1:19530"
# token_env = "MILVUS_TOKEN"

[freshness]
# audit_interval_secs = 900

[search]
max_concurrent_requests = 2
max_concurrent_repo_searches = 4
max_concurrent_lexical_tasks = 2
max_concurrent_dense_tasks = 2
max_warm_repos = 4

[[groups]]
id = "workspace"
label = "Workspace"
repos = [
  "/absolute/path/to/repo",
]
```

See the full template in [config.example.toml](/Users/jeremy/.local/share/agent-context/config.example.toml).

## Documentation

- [macOS quickstart](/Users/jeremy/.local/share/agent-context/docs/quickstart-macos.md)
- [Embedding providers](/Users/jeremy/.local/share/agent-context/docs/providers.md)
- [Milvus bootstrap](/Users/jeremy/.local/share/agent-context/docs/milvus.md)
- [MCP client setup](/Users/jeremy/.local/share/agent-context/docs/mcp-clients.md)
- [Troubleshooting](/Users/jeremy/.local/share/agent-context/docs/troubleshooting.md)

## Release and packaging

This repo includes:

- GitHub Actions CI
- GitHub Actions release packaging for:
  - `darwin-arm64`
  - `darwin-x86_64`
- a Homebrew formula template
- a checked-in Milvus Docker Compose bootstrap

See:

- [.github/workflows/ci.yml](/Users/jeremy/.local/share/agent-context/.github/workflows/ci.yml)
- [.github/workflows/release.yml](/Users/jeremy/.local/share/agent-context/.github/workflows/release.yml)
- [packaging/homebrew/agent-context.rb](/Users/jeremy/.local/share/agent-context/packaging/homebrew/agent-context.rb)

## Limitations

- macOS-first only in this release plan
- Milvus is required
- initial indexing depends on an embedding provider
- provider/model changes require `reindex-all`
- indexed search can lag uncommitted filesystem changes until you refresh

## Development

Common local checks:

```bash
cargo fmt
cargo test
cargo clippy --all-targets --all-features --locked -- -D warnings
```

If you are changing index formats or provider behavior, treat that as a rebuild boundary and update the docs and release notes accordingly.

## Agent notes

If you are an agent performing installation:

- prefer explicit absolute paths over relative ones
- run `doctor` before and after service installation
- use `print-mcp-config` instead of hand-writing client snippets
- assume provider/model changes require `reindex-all`
- prefer `install-hook` over manual hook editing

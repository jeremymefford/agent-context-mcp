# macOS Quickstart

This guide is written so an agent can execute it step by step for a user.

Before starting, the agent should know:

- the absolute repo path or paths to index
- the embedding provider to use
- where the provider credential will come from

## 1. Install `agent-context`

Use one of:

- a Homebrew tap formula
- a GitHub release asset
- `cargo install --path .`

Preferred:

```bash
brew tap jeremymefford/agent-context-mcp
brew install agent-context
```

Verify:

```bash
agent-context --help
```

## 2. Start Milvus

From this repo:

```bash
docker compose -f docker/milvus-compose.yml up -d
```

Wait for Milvus health:

```bash
curl http://127.0.0.1:9091/healthz
```

## 3. Create a starter config

```bash
agent-context init --provider voyage --repo /absolute/path/to/repo
```

The default config path is:

```text
~/Library/Application Support/agent-context/config.toml
```

Verify it exists before continuing.

## 4. Export credentials

Voyage:

```bash
export VOYAGE_API_KEY=...
```

OpenAI:

```bash
export OPENAI_API_KEY=...
```

Ollama does not require an API key, but it does require a local Ollama server.

## 5. Validate the environment

```bash
agent-context doctor
```

Do not proceed until `doctor` succeeds.

## 6. Start the MCP server

Preferred:

```bash
brew services start agent-context
```

Verify service registration:

```bash
brew services list | grep agent-context
```

Verify service health:

```bash
curl http://127.0.0.1:8765/health
```

Manual fallbacks:

```bash
agent-context serve --listen 127.0.0.1:8765
```

## 7. Connect your MCP client

Examples:

```bash
agent-context print-mcp-config --client codex
agent-context print-mcp-config --client claude
agent-context print-mcp-config --client copilot
```

## 8. Index and search

Incremental refresh:

```bash
agent-context refresh-all
```

Ad-hoc CLI search:

```bash
agent-context search workspace "graphql schema builder"
```

If search fails because the embedding fingerprint or index format changed, run:

```bash
agent-context reindex-all
```

## 9. Install git hooks

For repos that should refresh on commit:

```bash
agent-context install-hook /absolute/path/to/repo
```

That installs a managed `post-commit` hook block without clobbering unrelated hook content.

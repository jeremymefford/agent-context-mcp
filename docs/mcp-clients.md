# MCP client setup

`agent-context` serves HTTP MCP at:

```text
http://127.0.0.1:8765/mcp
```

If the service is launchd-managed, verify health first:

```bash
curl http://127.0.0.1:8765/health
```

## Codex

Generate:

```bash
agent-context print-mcp-config --client codex
```

Expected shape:

```toml
[mcp_servers.agent-context]
url = "http://127.0.0.1:8765/mcp"
```

## Claude

Generate:

```bash
agent-context print-mcp-config --client claude
```

Expected shape:

```json
{
  "mcpServers": {
    "agent-context": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

## VS Code / Copilot

Generate:

```bash
agent-context print-mcp-config --client copilot
```

Expected shape:

```json
{
  "servers": {
    "agent-context": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

## Agent routing guidance

The MCP server advertises explicit routing instructions, but a client or agent should still prefer:

- `list_scopes` first
- `search_symbols` for exact definitions
- `get_file_outline` for indexed structure
- `search_code` for broader discovery

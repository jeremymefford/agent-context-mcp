# Troubleshooting

## `doctor` fails on backend health

Likely causes:

- Milvus is not running
- the Milvus address is wrong
- Docker is down
- a bearer token is required but not configured

Checks:

```bash
docker compose -f docker/milvus-compose.yml ps
curl http://127.0.0.1:9091/healthz
agent-context doctor
```

## Search or refresh says the local index is incompatible

This usually means one of:

- the embedding provider changed
- the embedding model changed
- the effective embedding dimension changed
- the local search/index format changed

Recovery:

```bash
agent-context reindex-all
```

## Launchd service is installed but health is down

Check:

```bash
launchctl print gui/$(id -u)/dev.agent-context.mcp
tail -n 200 ~/Library/Logs/agent-context-mcp.log
tail -n 200 ~/Library/Logs/agent-context-mcp.err.log
```

Reinstall:

```bash
agent-context uninstall-launchd
agent-context install-launchd
```

## Hooks are not refreshing on commit

Reinstall the managed block:

```bash
agent-context install-hook /absolute/path/to/repo
```

Inspect:

```bash
tail -n 200 ~/Library/Logs/agent-context-hooks.log
```

## Downstream calls hang or take too long

`agent-context` applies request timeouts and bounded retries to Milvus and embedding providers. If you still see delays:

- verify provider reachability independently
- reduce local load on Milvus
- re-run `doctor`

If failures are persistent, prefer explicit retries over assuming the current index state is valid.

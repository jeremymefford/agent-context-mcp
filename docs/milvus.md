# Milvus Bootstrap

`agent-context` requires a running Milvus instance for dense retrieval.

## Local standalone Milvus

This repo ships a checked-in Docker Compose setup:

```bash
docker compose -f docker/milvus-compose.yml up -d
```

Services:

- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

Ports:

- `19530` for Milvus
- `9091` for Milvus health
- `9000` and `9001` for MinIO

## Health checks

Milvus:

```bash
curl http://127.0.0.1:9091/healthz
```

`agent-context`:

```bash
agent-context doctor
```

## Volumes

The compose file persists data under:

```text
./docker/volumes/
```

If you need a clean local reset:

```bash
docker compose -f docker/milvus-compose.yml down
rm -rf docker/volumes
```

That deletes local Milvus state.

## Config

Point `agent-context` at local standalone Milvus with:

```toml
[milvus]
address = "127.0.0.1:19530"
```

If your deployment requires a bearer token:

```toml
[milvus]
address = "127.0.0.1:19530"
token_env = "MILVUS_TOKEN"
```

## Troubleshooting

- If `doctor` reports backend health failures, confirm Docker is running and the compose stack is healthy.
- If memory usage is unexpectedly high, restart the compose stack before assuming index corruption.
- If provider/model configuration changed, Milvus collection dimensions may no longer match; run `agent-context reindex-all`.

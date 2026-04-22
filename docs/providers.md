# Embedding Providers

`agent-context` supports three embedding providers in v1:

- Voyage
- OpenAI
- Ollama

Provider choice is part of local index identity. If you change:

- provider
- model
- effective embedding dimension

you must run:

```bash
agent-context reindex-all
```

## Voyage

```toml
[embedding]
provider = "voyage"
model = "voyage-code-3"

[embedding.voyage]
api_key_env = "VOYAGE_API_KEY"
# key_file = "~/Library/Application Support/agent-context/voyage_key"
```

Notes:

- env vars are preferred
- `key_file` still exists as a compatibility path

## OpenAI

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"

[embedding.openai]
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
```

Notes:

- the default OpenAI base URL assumes the public OpenAI API
- `agent-context` calls the embeddings endpoint directly

## Ollama

```toml
[embedding]
provider = "ollama"
model = "embeddinggemma"

[embedding.ollama]
base_url = "http://127.0.0.1:11434"
```

Notes:

- no API key is required
- the configured model must already be available to the local Ollama server
- the effective embedding dimension is discovered from the provider and becomes part of the stored index identity

## Operational guidance

- Keep model changes intentional.
- Treat provider changes as rebuild boundaries.
- Run `agent-context doctor` after changing provider config.

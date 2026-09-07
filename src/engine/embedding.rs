use crate::config::{EmbeddingConfig, EmbeddingProfileConfig, EmbeddingProvider};
use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::OnceCell;

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(20);
pub const INDEX_TIMEOUT: Duration = Duration::from_secs(120);
const VOYAGE_MAX_BATCH_ITEMS: usize = 1000;
const MAX_RETRIES: usize = 4;

#[derive(Clone)]
pub struct EmbeddingClient {
    inner: Arc<EmbeddingInner>,
    dimension: Arc<OnceCell<usize>>,
}

#[derive(Clone)]
pub struct EmbeddingRegistry {
    clients: Arc<BTreeMap<String, EmbeddingRegistryEntry>>,
}

#[derive(Clone)]
enum EmbeddingRegistryEntry {
    Ready(EmbeddingClient),
    Failed(String),
}

enum EmbeddingInner {
    Voyage(VoyageClient),
    OpenAi(OpenAiClient),
    Ollama(OllamaClient),
}

#[derive(Debug, Clone)]
struct VoyageClient {
    http: reqwest::Client,
    model: String,
}

#[derive(Debug, Clone)]
struct OpenAiClient {
    http: reqwest::Client,
    model: String,
    base_url: String,
}

#[derive(Debug, Clone)]
struct OllamaClient {
    http: reqwest::Client,
    model: String,
    base_url: String,
    dimensions: Option<usize>,
    truncate_dimensions: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CommonEmbeddingsResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OllamaEmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

impl EmbeddingClient {
    pub async fn new(config: &EmbeddingProfileConfig) -> Result<Self> {
        let inner = match config.provider {
            EmbeddingProvider::Voyage => EmbeddingInner::Voyage(VoyageClient::new(
                config.api_key()?.context("missing Voyage API key")?,
                config.model.clone(),
            )?),
            EmbeddingProvider::OpenAi => EmbeddingInner::OpenAi(OpenAiClient::new(
                config.api_key()?.context("missing OpenAI API key")?,
                config.model.clone(),
                config.openai.base_url.clone(),
            )?),
            EmbeddingProvider::Ollama => EmbeddingInner::Ollama(OllamaClient::new(
                config.model.clone(),
                config.ollama.base_url.clone(),
                config.ollama.dimensions,
                config.ollama.truncate_dimensions,
            )?),
        };

        Ok(Self {
            inner: Arc::new(inner),
            dimension: Arc::new(OnceCell::new()),
        })
    }

    pub fn provider_name(&self) -> &'static str {
        match self.inner.as_ref() {
            EmbeddingInner::Voyage(_) => "voyage",
            EmbeddingInner::OpenAi(_) => "openai",
            EmbeddingInner::Ollama(_) => "ollama",
        }
    }

    pub fn model(&self) -> &str {
        match self.inner.as_ref() {
            EmbeddingInner::Voyage(client) => &client.model,
            EmbeddingInner::OpenAi(client) => &client.model,
            EmbeddingInner::Ollama(client) => &client.model,
        }
    }

    pub async fn dimension(&self) -> Result<usize> {
        let dimension = self
            .dimension
            .get_or_try_init(|| async {
                if let Some(dimension) = known_dimension(self.provider_name(), self.model()) {
                    return Ok::<usize, anyhow::Error>(dimension);
                }
                let probe = self
                    .embed_query("__agent_context_dimension_probe__")
                    .await?;
                Ok(probe.len())
            })
            .await?;
        Ok(*dimension)
    }

    pub async fn fingerprint(&self) -> Result<String> {
        let fingerprint = format!(
            "{}:{}:{}",
            self.provider_name(),
            self.model(),
            self.dimension().await?
        );
        match self.inner.as_ref() {
            EmbeddingInner::Ollama(client) => Ok(format!(
                "{fingerprint}:{}",
                ollama_width_fingerprint(client.dimensions, client.truncate_dimensions)
            )),
            EmbeddingInner::Voyage(_) | EmbeddingInner::OpenAi(_) => Ok(fingerprint),
        }
    }

    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let batch_limit = self.document_batch_limit();
        if texts.len() > batch_limit {
            let mut embeddings = Vec::with_capacity(texts.len());
            for batch in texts.chunks(batch_limit) {
                embeddings.extend(self.embed_document_batch(batch).await?);
            }
            return Ok(embeddings);
        }
        self.embed_document_batch(texts).await
    }

    fn document_batch_limit(&self) -> usize {
        match self.inner.as_ref() {
            EmbeddingInner::Voyage(_) => VOYAGE_MAX_BATCH_ITEMS,
            EmbeddingInner::OpenAi(_) | EmbeddingInner::Ollama(_) => usize::MAX,
        }
    }

    async fn embed_document_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match self.inner.as_ref() {
            EmbeddingInner::Voyage(client) => client.embed(texts, Some("document")).await,
            EmbeddingInner::OpenAi(client) => client.embed(texts).await,
            EmbeddingInner::Ollama(client) => client.embed(texts).await,
        }
    }

    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let values = match self.inner.as_ref() {
            EmbeddingInner::Voyage(client) => {
                client.embed(&[text.to_string()], Some("query")).await
            }
            EmbeddingInner::OpenAi(client) => client.embed(&[text.to_string()]).await,
            EmbeddingInner::Ollama(client) => {
                client
                    .embed(&[qwen3_coding_agent_query(&client.model, text)])
                    .await
            }
        }?;
        values
            .into_iter()
            .next()
            .context("embedding provider returned no vector for query")
    }
}

impl EmbeddingRegistry {
    pub async fn new(config: &EmbeddingConfig) -> Result<Self> {
        let mut clients = BTreeMap::new();
        for (name, profile) in config.profiles() {
            let entry = match EmbeddingClient::new(profile).await {
                Ok(client) => EmbeddingRegistryEntry::Ready(client),
                Err(error) => EmbeddingRegistryEntry::Failed(error.to_string()),
            };
            clients.insert(name.clone(), entry);
        }
        Ok(Self {
            clients: Arc::new(clients),
        })
    }

    pub fn client(&self, profile_name: &str) -> Result<&EmbeddingClient> {
        match self
            .clients
            .get(profile_name)
            .with_context(|| format!("embedding profile `{profile_name}` is not initialized"))?
        {
            EmbeddingRegistryEntry::Ready(client) => Ok(client),
            EmbeddingRegistryEntry::Failed(error) => {
                bail!("embedding profile `{profile_name}` is unavailable: {error}")
            }
        }
    }

    pub async fn dimension(&self, profile_name: &str) -> Result<usize> {
        self.client(profile_name)?.dimension().await
    }

    pub async fn fingerprint(&self, profile_name: &str) -> Result<String> {
        self.client(profile_name)?.fingerprint().await
    }

    pub async fn embed_documents(
        &self,
        profile_name: &str,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>> {
        self.client(profile_name)?.embed_documents(texts).await
    }

    pub async fn embed_query(&self, profile_name: &str, text: &str) -> Result<Vec<f32>> {
        self.client(profile_name)?.embed_query(text).await
    }
}

impl VoyageClient {
    fn new(api_key: String, model: String) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("building Voyage auth header")?,
        );

        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
            .build()
            .context("building Voyage HTTP client")?;

        Ok(Self { http, model })
    }

    async fn embed(&self, texts: &[String], input_type: Option<&str>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() > VOYAGE_MAX_BATCH_ITEMS {
            bail!("Voyage batch exceeds max item count: {}", texts.len());
        }

        let url = "https://api.voyageai.com/v1/embeddings";
        let payload = json!({
            "input": texts,
            "model": self.model,
            "input_type": input_type,
        });

        let response = send_with_retry(
            &self.http,
            "Voyage embeddings",
            timeout_for_batch(texts.len()),
            || self.http.post(url).json(&payload),
        )
        .await?;

        let payload: CommonEmbeddingsResponse = response
            .json()
            .await
            .context("decoding Voyage embeddings response")?;
        if payload.data.len() != texts.len() {
            bail!(
                "Voyage returned {} embeddings for {} inputs",
                payload.data.len(),
                texts.len()
            );
        }

        Ok(payload
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }
}

impl OpenAiClient {
    fn new(api_key: String, model: String, base_url: String) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .context("building OpenAI auth header")?,
        );

        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
            .build()
            .context("building OpenAI HTTP client")?;

        Ok(Self {
            http,
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        let payload = json!({
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        });

        let response = send_with_retry(
            &self.http,
            "OpenAI embeddings",
            timeout_for_batch(texts.len()),
            || self.http.post(&url).json(&payload),
        )
        .await?;

        let payload: CommonEmbeddingsResponse = response
            .json()
            .await
            .context("decoding OpenAI embeddings response")?;
        if payload.data.len() != texts.len() {
            bail!(
                "OpenAI returned {} embeddings for {} inputs",
                payload.data.len(),
                texts.len()
            );
        }

        Ok(payload
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }
}

impl OllamaClient {
    fn new(
        model: String,
        base_url: String,
        dimensions: Option<usize>,
        truncate_dimensions: Option<usize>,
    ) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
            .build()
            .context("building Ollama HTTP client")?;

        Ok(Self {
            http,
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
            dimensions,
            truncate_dimensions,
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url);
        let payload = ollama_embed_payload(&self.model, texts, self.dimensions);

        let response = send_with_retry(
            &self.http,
            "Ollama embeddings",
            timeout_for_batch(texts.len()),
            || self.http.post(&url).json(&payload),
        )
        .await?;

        let payload: OllamaEmbedResponse = response
            .json()
            .await
            .context("decoding Ollama embeddings response")?;
        if payload.embeddings.len() != texts.len() {
            bail!(
                "Ollama returned {} embeddings for {} inputs",
                payload.embeddings.len(),
                texts.len()
            );
        }

        truncate_ollama_embeddings(payload.embeddings, self.truncate_dimensions)
    }
}

fn truncate_ollama_embeddings(
    mut embeddings: Vec<Vec<f32>>,
    truncate_dimensions: Option<usize>,
) -> Result<Vec<Vec<f32>>> {
    let Some(truncate_dimensions) = truncate_dimensions else {
        return Ok(embeddings);
    };
    for embedding in &mut embeddings {
        if embedding.len() < truncate_dimensions {
            bail!(
                "Ollama returned {} dimensions, fewer than configured truncate_dimensions {truncate_dimensions}",
                embedding.len()
            );
        }
        embedding.truncate(truncate_dimensions);
    }
    Ok(embeddings)
}

fn ollama_width_fingerprint(
    requested_dimensions: Option<usize>,
    truncate_dimensions: Option<usize>,
) -> String {
    let requested = requested_dimensions
        .map(|dimensions| dimensions.to_string())
        .unwrap_or_else(|| "default".to_string());
    let truncate = truncate_dimensions
        .map(|dimensions| dimensions.to_string())
        .unwrap_or_else(|| "none".to_string());
    format!("request={requested}:truncate={truncate}")
}

const QWEN3_CODING_AGENT_RETRIEVAL_INSTRUCTION: &str = "Given a coding-agent query, retrieve the most relevant code, configuration, documentation, or tests needed to understand, modify, or verify the requested behavior.";

fn qwen3_coding_agent_query(model: &str, query: &str) -> String {
    if model
        .trim()
        .to_ascii_lowercase()
        .starts_with("qwen3-embedding")
    {
        format!("Instruct: {QWEN3_CODING_AGENT_RETRIEVAL_INSTRUCTION}\nQuery: {query}")
    } else {
        query.to_string()
    }
}

fn ollama_embed_payload(
    model: &str,
    texts: &[String],
    dimensions: Option<usize>,
) -> serde_json::Value {
    let mut payload = json!({
        "model": model,
        "input": texts,
        "truncate": true,
    });
    if let Some(dimensions) = dimensions {
        payload["dimensions"] = json!(dimensions);
    }
    payload
}

async fn send_with_retry<F>(
    http: &reqwest::Client,
    operation: &str,
    timeout: Duration,
    mut build_request: F,
) -> Result<reqwest::Response>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let _ = http;
    let mut last_error = None;

    for attempt in 0..MAX_RETRIES {
        let request_id = request_id(operation, attempt);
        let request = build_request()
            .timeout(timeout)
            .header("X-Client-Request-Id", request_id);

        match request.send().await {
            Ok(response) => {
                if response.status().is_success() {
                    return Ok(response);
                }

                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                if is_retryable_response(operation, status, &body) && attempt + 1 < MAX_RETRIES {
                    last_error = Some(format!("{operation} failed with {status}: {body}"));
                    tokio::time::sleep(retry_delay_for(operation, attempt)).await;
                    continue;
                }
                bail!("{operation} failed with {status}: {body}");
            }
            Err(error) => {
                if is_retryable_transport_error(&error) && attempt + 1 < MAX_RETRIES {
                    last_error = Some(format!("{operation} transport error: {error}"));
                    tokio::time::sleep(retry_delay_for(operation, attempt)).await;
                    continue;
                }
                let message = format!("{operation} transport error: {error:#}");
                return Err(error).context(message);
            }
        }
    }

    bail!(
        "{} failed after {} attempts: {}",
        operation,
        MAX_RETRIES,
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
}

fn timeout_for_batch(item_count: usize) -> Duration {
    if item_count <= 4 {
        SEARCH_TIMEOUT
    } else {
        INDEX_TIMEOUT
    }
}

fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    matches!(
        status,
        reqwest::StatusCode::TOO_MANY_REQUESTS
            | reqwest::StatusCode::BAD_GATEWAY
            | reqwest::StatusCode::SERVICE_UNAVAILABLE
            | reqwest::StatusCode::GATEWAY_TIMEOUT
    )
}

fn is_retryable_response(operation: &str, status: reqwest::StatusCode, body: &str) -> bool {
    is_retryable_status(status) || (operation == "Ollama embeddings" && ollama_runtime_error(body))
}

fn ollama_runtime_error(body: &str) -> bool {
    let body = body.to_ascii_lowercase();
    body.contains("connection reset by peer")
        || body.contains("connection refused")
        || (body.contains("/tokenize") && (body.contains("read tcp") || body.contains("eof")))
}

fn is_retryable_transport_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

fn retry_delay(attempt: usize) -> Duration {
    let base_ms = 200_u64.saturating_mul(1_u64 << attempt.min(5));
    let jitter_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_millis() as u64
        % 250;
    Duration::from_millis(base_ms.saturating_add(jitter_ms))
}

fn retry_delay_for(operation: &str, attempt: usize) -> Duration {
    if operation == "Ollama embeddings" {
        // Ollama can briefly lose its model-side tokenizer while returning a 400 to the client.
        let seconds = 2_u64.saturating_mul(1_u64 << attempt.min(3));
        return Duration::from_secs(seconds);
    }
    retry_delay(attempt)
}

fn request_id(operation: &str, attempt: usize) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!(
        "agent-context-{}-{}-{}",
        operation.replace(' ', "-").to_ascii_lowercase(),
        millis,
        attempt
    )
}

fn known_dimension(provider: &str, model: &str) -> Option<usize> {
    match (provider, model) {
        ("voyage", "voyage-large-2") | ("voyage", "voyage-code-2") => Some(1536),
        ("voyage", "voyage-3-lite") => Some(512),
        ("voyage", _) => Some(1024),
        ("openai", "text-embedding-3-small") => Some(1536),
        ("openai", "text-embedding-3-large") => Some(3072),
        ("openai", "text-embedding-ada-002") => Some(1536),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        known_dimension, ollama_embed_payload, ollama_runtime_error, ollama_width_fingerprint,
        qwen3_coding_agent_query, retry_delay, retry_delay_for, truncate_ollama_embeddings,
    };
    use serde_json::json;

    #[test]
    fn knows_openai_embedding_dimensions() {
        assert_eq!(
            known_dimension("openai", "text-embedding-3-small"),
            Some(1536)
        );
        assert_eq!(
            known_dimension("openai", "text-embedding-3-large"),
            Some(3072)
        );
    }

    #[test]
    fn retry_delay_grows_per_attempt() {
        assert!(retry_delay(1) >= retry_delay(0));
    }

    #[test]
    fn retries_transient_ollama_tokenizer_failures_with_longer_backoff() {
        assert!(ollama_runtime_error(
            "Post \\\"http://127.0.0.1:55994/tokenize\\\": read tcp: connection reset by peer"
        ));
        assert!(ollama_runtime_error(
            "Post \\\"http://127.0.0.1:49693/tokenize\\\": EOF"
        ));
        assert!(!ollama_runtime_error("invalid dimensions"));
        assert!(retry_delay_for("Ollama embeddings", 0) > retry_delay(0));
    }

    #[test]
    fn ollama_payload_includes_optional_dimensions() {
        let texts = vec!["hello".to_string()];
        let payload = ollama_embed_payload("qwen3-embedding:8b-q8_0", &texts, Some(1024));

        assert_eq!(
            payload,
            json!({
                "model": "qwen3-embedding:8b-q8_0",
                "input": ["hello"],
                "truncate": true,
                "dimensions": 1024,
            })
        );
    }

    #[test]
    fn truncates_ollama_embeddings_after_receiving_the_requested_width() {
        let embeddings = truncate_ollama_embeddings(vec![vec![1.0; 4096]], Some(1024)).unwrap();

        assert_eq!(embeddings[0].len(), 1024);
        let error = truncate_ollama_embeddings(vec![vec![1.0; 512]], Some(1024)).unwrap_err();
        assert!(error.to_string().contains("fewer than configured"));
    }

    #[test]
    fn ollama_fingerprint_records_requested_and_retained_widths() {
        assert_eq!(
            ollama_width_fingerprint(Some(4096), Some(1024)),
            "request=4096:truncate=1024"
        );
    }

    #[test]
    fn qwen3_queries_are_tuned_for_coding_agent_retrieval() {
        assert_eq!(
            qwen3_coding_agent_query("qwen3-embedding:8b-q8_0", "find indexing retries"),
            "Instruct: Given a coding-agent query, retrieve the most relevant code, configuration, documentation, or tests needed to understand, modify, or verify the requested behavior.\nQuery: find indexing retries"
        );
        assert_eq!(
            qwen3_coding_agent_query("nomic-embed-text", "find indexing retries"),
            "find indexing retries"
        );
    }
}

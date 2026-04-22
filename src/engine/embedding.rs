use crate::config::{EmbeddingConfig, EmbeddingProvider};
use anyhow::{Context, Result, bail};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::OnceCell;

pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(20);
pub const INDEX_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_RETRIES: usize = 4;

#[derive(Clone)]
pub struct EmbeddingClient {
    inner: Arc<EmbeddingInner>,
    dimension: Arc<OnceCell<usize>>,
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
    pub async fn new(config: &EmbeddingConfig) -> Result<Self> {
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
        Ok(format!(
            "{}:{}:{}",
            self.provider_name(),
            self.model(),
            self.dimension().await?
        ))
    }

    pub async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
            EmbeddingInner::Ollama(client) => client.embed(&[text.to_string()]).await,
        }?;
        values
            .into_iter()
            .next()
            .context("embedding provider returned no vector for query")
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
        if texts.len() > 1000 {
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
    fn new(model: String, base_url: String) -> Result<Self> {
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
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/embed", self.base_url);
        let payload = json!({
            "model": self.model,
            "input": texts,
            "truncate": true,
        });

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

        Ok(payload.embeddings)
    }
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
                if is_retryable_status(status) && attempt + 1 < MAX_RETRIES {
                    last_error = Some(format!("{operation} failed with {status}: {body}"));
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                bail!("{operation} failed with {status}: {body}");
            }
            Err(error) => {
                if is_retryable_transport_error(&error) && attempt + 1 < MAX_RETRIES {
                    last_error = Some(format!("{operation} transport error: {error}"));
                    tokio::time::sleep(retry_delay(attempt)).await;
                    continue;
                }
                return Err(error).with_context(|| operation.to_string());
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
    use super::{known_dimension, retry_delay};

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
}

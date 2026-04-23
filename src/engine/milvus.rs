use crate::config::MilvusConfig;
use crate::engine::embedding::{CONNECT_TIMEOUT, INDEX_TIMEOUT, SEARCH_TIMEOUT};
use anyhow::{Context, Result, anyhow, bail};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const COLLECTION_LIMIT_MESSAGE: &str = "[Error]: Your Zilliz Cloud account has hit its collection limit. To continue creating collections, you'll need to expand your capacity. We recommend visiting https://zilliz.com/pricing to explore options for dedicated or serverless clusters.";
const MAX_RETRIES: usize = 4;
#[derive(Debug, Clone)]
pub struct MilvusClient {
    http: reqwest::Client,
    base_url: String,
    database: String,
    collection_presence: Arc<RwLock<HashMap<String, bool>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDocument {
    pub id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub file_extension: String,
    pub metadata: Value,
}

#[derive(Debug, Clone)]
pub struct SearchDocument {
    pub id: String,
    pub content: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub file_extension: String,
    pub metadata: Value,
    pub score: f64,
}

#[derive(Debug, Deserialize)]
struct ApiResponse<T> {
    code: i64,
    data: Option<T>,
    message: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LoadStatePayload {
    #[serde(default, rename = "loadState")]
    load_state: Option<String>,
}

impl MilvusClient {
    pub fn new(config: &MilvusConfig) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));

        if let Some(token) = config.token() {
            let value = HeaderValue::from_str(&format!("Bearer {token}"))
                .context("building Milvus auth header from token")?;
            headers.insert(AUTHORIZATION, value);
        } else if let (Some(username), Some(password)) = (&config.username, &config.password) {
            let value = HeaderValue::from_str(&format!("Bearer {username}:{password}"))
                .context("building Milvus auth header from username/password")?;
            headers.insert(AUTHORIZATION, value);
        }

        let address =
            if config.address.starts_with("http://") || config.address.starts_with("https://") {
                config.address.clone()
            } else {
                format!("http://{}", config.address)
            };

        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .default_headers(headers)
            .build()
            .context("building Milvus HTTP client")?;

        Ok(Self {
            http,
            base_url: format!("{}/v2/vectordb", address.trim_end_matches('/')),
            database: config.database.clone(),
            collection_presence: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn has_collection(&self, collection_name: &str) -> Result<bool> {
        if let Some(exists) = self.cached_collection_presence(collection_name)? {
            return Ok(exists);
        }
        let exists = self.query_collection_presence(collection_name).await?;
        self.set_collection_presence(collection_name, exists)?;
        Ok(exists)
    }

    pub async fn refresh_collection_presence(&self, collection_name: &str) -> Result<bool> {
        let exists = self.query_collection_presence(collection_name).await?;
        self.set_collection_presence(collection_name, exists)?;
        Ok(exists)
    }

    fn cached_collection_presence(&self, collection_name: &str) -> Result<Option<bool>> {
        let cache = self
            .collection_presence
            .read()
            .map_err(|_| anyhow!("Milvus collection presence cache poisoned"))?;
        Ok(cache.get(collection_name).copied())
    }

    fn set_collection_presence(&self, collection_name: &str, exists: bool) -> Result<()> {
        let mut cache = self
            .collection_presence
            .write()
            .map_err(|_| anyhow!("Milvus collection presence cache poisoned"))?;
        cache.insert(collection_name.to_string(), exists);
        Ok(())
    }

    async fn query_collection_presence(&self, collection_name: &str) -> Result<bool> {
        #[derive(Debug, Deserialize)]
        struct HasCollection {
            #[serde(default)]
            has: bool,
        }

        let response: HasCollection = self
            .post(
                "/collections/has",
                self.with_db(json!({
                    "collectionName": collection_name,
                })),
                SEARCH_TIMEOUT,
            )
            .await?;
        Ok(response.has)
    }

    pub async fn healthcheck(&self) -> Result<()> {
        self.has_collection("__agent_context_healthcheck__")
            .await
            .map(|_| ())
    }

    pub async fn ensure_loaded(&self, collection_name: &str) -> Result<()> {
        let state: LoadStatePayload = self
            .post(
                "/collections/get_load_state",
                self.with_db(json!({
                    "collectionName": collection_name,
                })),
                SEARCH_TIMEOUT,
            )
            .await?;

        if state.load_state.as_deref() != Some("LoadStateLoaded") {
            let _: Value = self
                .post(
                    "/collections/load",
                    self.with_db(json!({
                        "collectionName": collection_name,
                    })),
                    INDEX_TIMEOUT,
                )
                .await?;
        }
        Ok(())
    }

    pub async fn drop_collection(&self, collection_name: &str) -> Result<()> {
        let _: Value = self
            .post(
                "/collections/drop",
                self.with_db(json!({
                    "collectionName": collection_name,
                })),
                INDEX_TIMEOUT,
            )
            .await?;
        self.set_collection_presence(collection_name, false)?;
        Ok(())
    }

    pub async fn create_hybrid_collection(
        &self,
        collection_name: &str,
        dimension: usize,
        description: &str,
    ) -> Result<()> {
        let _: Value = self
            .post(
                "/collections/create",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "description": description,
                    "schema": {
                        "autoId": false,
                        "enableDynamicField": false,
                        "fields": [
                            {
                                "fieldName": "id",
                                "dataType": "VarChar",
                                "isPrimary": true,
                                "elementTypeParams": {
                                    "max_length": 512
                                }
                            },
                            {
                                "fieldName": "content",
                                "dataType": "VarChar",
                                "elementTypeParams": {
                                    "max_length": 65535,
                                    "enable_analyzer": true,
                                    "enable_match": true
                                }
                            },
                            {
                                "fieldName": "vector",
                                "dataType": "FloatVector",
                                "elementTypeParams": {
                                    "dim": dimension
                                }
                            },
                            {
                                "fieldName": "sparse_vector",
                                "dataType": "SparseFloatVector"
                            },
                            {
                                "fieldName": "relativePath",
                                "dataType": "VarChar",
                                "elementTypeParams": {
                                    "max_length": 1024
                                }
                            },
                            {
                                "fieldName": "startLine",
                                "dataType": "Int64"
                            },
                            {
                                "fieldName": "endLine",
                                "dataType": "Int64"
                            },
                            {
                                "fieldName": "fileExtension",
                                "dataType": "VarChar",
                                "elementTypeParams": {
                                    "max_length": 32
                                }
                            },
                            {
                                "fieldName": "metadata",
                                "dataType": "VarChar",
                                "elementTypeParams": {
                                    "max_length": 65535
                                }
                            }
                        ],
                        "functions": [
                            {
                                "name": "content_bm25_emb",
                                "type": "BM25",
                                "inputFieldNames": ["content"],
                                "outputFieldNames": ["sparse_vector"],
                                "params": {}
                            }
                        ]
                    }
                })),
                INDEX_TIMEOUT,
            )
            .await?;

        let _: Value = self
            .post(
                "/indexes/create",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "indexParams": [
                        {
                            "fieldName": "vector",
                            "indexName": "vector_index",
                            "metricType": "COSINE",
                            "index_type": "AUTOINDEX"
                        }
                    ]
                })),
                INDEX_TIMEOUT,
            )
            .await?;

        let _: Value = self
            .post(
                "/indexes/create",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "indexParams": [
                        {
                            "fieldName": "sparse_vector",
                            "indexName": "sparse_vector_index",
                            "metricType": "BM25",
                            "index_type": "SPARSE_INVERTED_INDEX"
                        }
                    ]
                })),
                INDEX_TIMEOUT,
            )
            .await?;

        self.set_collection_presence(collection_name, true)?;
        self.ensure_loaded(collection_name).await
    }

    pub async fn insert_documents(
        &self,
        collection_name: &str,
        documents: &[VectorDocument],
    ) -> Result<()> {
        self.ensure_loaded(collection_name).await?;
        let rows = documents
            .iter()
            .map(|document| {
                json!({
                    "id": document.id,
                    "content": document.content,
                    "vector": document.vector,
                    "relativePath": document.relative_path,
                    "startLine": document.start_line,
                    "endLine": document.end_line,
                    "fileExtension": document.file_extension,
                    "metadata": document.metadata.to_string(),
                })
            })
            .collect::<Vec<_>>();

        let _: Value = self
            .post(
                "/entities/insert",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "data": rows,
                })),
                INDEX_TIMEOUT,
            )
            .await?;
        Ok(())
    }

    pub async fn search_dense(
        &self,
        collection_name: &str,
        query_vector: &[f32],
        limit: usize,
        filter: Option<&str>,
    ) -> Result<Vec<SearchDocument>> {
        let mut request = self.with_db(json!({
            "collectionName": collection_name,
            "data": [query_vector],
            "annsField": "vector",
            "limit": limit,
            "outputFields": ["id", "content", "relativePath", "startLine", "endLine", "fileExtension", "metadata"],
            "searchParams": {
                "metricType": "COSINE",
                "params": {}
            }
        }));
        if let Some(filter) = filter.filter(|value| !value.is_empty()) {
            request["filter"] = Value::String(filter.to_string());
        }
        let rows: Vec<Map<String, Value>> = self
            .post("/entities/search", request, SEARCH_TIMEOUT)
            .await?;
        rows.into_iter()
            .map(map_search_row)
            .collect::<Result<Vec<_>>>()
    }

    pub async fn query_ids_for_path(
        &self,
        collection_name: &str,
        relative_path: &str,
    ) -> Result<Vec<String>> {
        self.ensure_loaded(collection_name).await?;
        let escaped = relative_path.replace('\\', "\\\\").replace('"', "\\\"");
        let rows: Vec<Map<String, Value>> = self
            .post(
                "/entities/query",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "filter": format!("relativePath == \"{escaped}\""),
                    "outputFields": ["id"],
                    "limit": 16384,
                    "offset": 0
                })),
                INDEX_TIMEOUT,
            )
            .await?;
        Ok(rows
            .into_iter()
            .filter_map(|row| row.get("id").and_then(as_string))
            .collect())
    }

    pub async fn delete_ids(&self, collection_name: &str, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_loaded(collection_name).await?;
        let filter = format!(
            "id in [{}]",
            ids.iter()
                .map(|id| format!("\"{}\"", id.replace('"', "\\\"")))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _: Value = self
            .post(
                "/entities/delete",
                self.with_db(json!({
                    "collectionName": collection_name,
                    "filter": filter
                })),
                INDEX_TIMEOUT,
            )
            .await?;
        Ok(())
    }

    fn with_db(&self, mut value: Value) -> Value {
        if let Value::Object(ref mut map) = value {
            if !self.database.is_empty() {
                map.insert("dbName".to_string(), Value::String(self.database.clone()));
            } else {
                map.remove("dbName");
            }
        }
        value
    }

    async fn post<T>(&self, endpoint: &str, body: Value, timeout: Duration) -> Result<T>
    where
        T: DeserializeOwned,
    {
        let url = format!("{}{}", self.base_url, endpoint);
        let response = send_with_retry(&self.http, "Milvus API", timeout, || {
            self.http.post(&url).json(&body)
        })
        .await
        .with_context(|| format!("POST {url}"))?;

        let payload: ApiResponse<T> = response
            .json()
            .await
            .with_context(|| format!("decoding Milvus response from {url}"))?;

        if payload.code != 0 && payload.code != 200 {
            let message = payload
                .message
                .unwrap_or_else(|| format!("code {}", payload.code));
            if is_collection_limit_error(&message) {
                bail!("{COLLECTION_LIMIT_MESSAGE}");
            }
            bail!("Milvus API error on {url}: {}", message);
        }

        payload
            .data
            .ok_or_else(|| anyhow!("Milvus response from {url} missing `data`"))
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
        let request = build_request()
            .timeout(timeout)
            .header("X-Client-Request-Id", request_id(operation, attempt));
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

fn is_collection_limit_error(message: &str) -> bool {
    let lowercase = message.to_ascii_lowercase();
    lowercase.contains("exceeded the limit number of collections")
        || (lowercase.contains("collection") && lowercase.contains("limit"))
}

fn map_search_row(mut row: Map<String, Value>) -> Result<SearchDocument> {
    let score = row
        .get("distance")
        .and_then(as_f64)
        .or_else(|| row.get("score").and_then(as_f64))
        .unwrap_or(0.0);
    let metadata = match row.remove("metadata") {
        Some(Value::String(text)) => {
            serde_json::from_str(&text).unwrap_or(Value::Object(Map::new()))
        }
        Some(value) => value,
        None => Value::Object(Map::new()),
    };

    Ok(SearchDocument {
        id: row
            .remove("id")
            .and_then(|value| as_string(&value))
            .context("Milvus search row missing id")?,
        content: row
            .remove("content")
            .and_then(|value| as_string(&value))
            .unwrap_or_default(),
        relative_path: row
            .remove("relativePath")
            .and_then(|value| as_string(&value))
            .unwrap_or_default(),
        start_line: row
            .remove("startLine")
            .and_then(|value| as_u64(&value))
            .unwrap_or(0),
        end_line: row
            .remove("endLine")
            .and_then(|value| as_u64(&value))
            .unwrap_or(0),
        file_extension: row
            .remove("fileExtension")
            .and_then(|value| as_string(&value))
            .unwrap_or_default(),
        metadata,
        score,
    })
}

fn as_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    }
}

fn as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.parse().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{MilvusClient, is_collection_limit_error};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};

    fn test_client() -> MilvusClient {
        MilvusClient {
            http: reqwest::Client::new(),
            base_url: "http://127.0.0.1:19530/v2/vectordb".to_string(),
            database: String::new(),
            collection_presence: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    #[test]
    fn detects_collection_limit_message() {
        assert!(is_collection_limit_error(
            "rpc error: exceeded the limit number of collections for this cluster"
        ));
    }

    #[test]
    fn ignores_unrelated_messages() {
        assert!(!is_collection_limit_error("connection reset by peer"));
    }

    #[test]
    fn collection_presence_cache_tracks_updates() {
        let client = test_client();
        assert_eq!(client.cached_collection_presence("chunks").unwrap(), None);

        client.set_collection_presence("chunks", true).unwrap();
        assert_eq!(
            client.cached_collection_presence("chunks").unwrap(),
            Some(true)
        );

        client.set_collection_presence("chunks", false).unwrap();
        assert_eq!(
            client.cached_collection_presence("chunks").unwrap(),
            Some(false)
        );
    }
}

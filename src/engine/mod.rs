pub mod embedding;
pub mod freshness;
pub mod lexical;
pub mod milvus;
pub mod splitter;
pub mod symbols;

use crate::config::{Config, ResolvedScope, ScopeKind};
use crate::snapshot::{Snapshot, SnapshotEntry, SnapshotStore};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use xxhash_rust::xxh3::Xxh3;

use self::embedding::EmbeddingClient;
use self::freshness::{
    AuditFingerprint, apply_fingerprint, fingerprint_changed, fingerprint_repo,
    merkle_snapshot_path,
};
use self::lexical::{
    ChunkIndexDoc, ChunkSearchHit as LexicalChunkSearchHit, ChunkSearchRequest, LocalIndexStore,
    QueryFlavor, SymbolIndexDoc, SymbolSearchRequest,
};
use self::milvus::{MilvusClient, SearchDocument, VectorDocument};
use self::splitter::{
    CodeChunk, SplitterKind, default_supported_extensions, language_for_extension, split_text,
};
use self::symbols::{IndexedSymbol, OutlineNode, SymbolStore, build_outline, extract_symbols};

const EMBEDDING_BATCH_SIZE: usize = 64;
const CHUNK_LIMIT: usize = 450_000;
const RRF_K: f64 = 100.0;
const CONTENT_HASH_ALGORITHM: &str = "xxh3_128";
const LEGACY_CONTENT_HASH_ALGORITHM: &str = "sha256";
const SNAPSHOT_PROGRESS_WRITE_INTERVAL: Duration = Duration::from_secs(2);
const INDEX_FORMAT_VERSION: &str = "v1";
const SEARCH_ROOT_VERSION: &str = "v1";

thread_local! {
    static LOW_PRIORITY_BLOCKING_THREAD: Cell<bool> = const { Cell::new(false) };
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    config: Config,
    snapshot: SnapshotStore,
    milvus: MilvusClient,
    embedding: EmbeddingClient,
    local_index: LocalIndexStore,
    symbol_store: SymbolStore,
    search_budgets: SearchBudgets,
}

#[derive(Clone)]
struct SearchBudgets {
    requests: Arc<Semaphore>,
    repo_searches: Arc<Semaphore>,
    lexical_tasks: Arc<Semaphore>,
    dense_tasks: Arc<Semaphore>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeIndexResult {
    pub scope: String,
    pub label: String,
    pub has_errors: bool,
    pub repos: Vec<RepoIndexResult>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoIndexResult {
    pub repo: String,
    pub indexed_files: Option<u64>,
    pub total_chunks: Option<u64>,
    pub index_status: Option<String>,
    pub full_reindex: bool,
    pub changes: RepoChangeSummary,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct RepoChangeSummary {
    pub added: u64,
    pub modified: u64,
    pub removed: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeClearResult {
    pub scope: String,
    pub label: String,
    pub repos: Vec<RepoClearResult>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoClearResult {
    pub repo: String,
    pub cleared: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchResponse {
    pub scope: String,
    pub label: String,
    pub partial: bool,
    pub repo_errors: Vec<RepoSearchError>,
    pub plan: SearchPlanSummary,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub repo: String,
    pub repo_label: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub language: String,
    pub score: f64,
    pub match_type: String,
    pub dense_score: Option<f64>,
    pub lexical_score: Option<f64>,
    pub symbol_score: Option<f64>,
    pub indexed_at: Option<String>,
    pub stale: bool,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoSearchError {
    pub repo: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchPlanSummary {
    pub requested_mode: String,
    pub effective_mode: String,
    pub query_kind: String,
    pub dense_weight: f64,
    pub lexical_weight: f64,
    pub symbol_weight: f64,
    pub symbol_lexical_share: f64,
    pub symbol_semantic_share: f64,
    pub dedupe_by_file: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchExplanation {
    pub scope: String,
    pub label: String,
    pub query: String,
    pub plan: SearchPlanSummary,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolSearchResponse {
    pub scope: String,
    pub label: String,
    pub partial: bool,
    pub repo_errors: Vec<RepoSearchError>,
    pub hits: Vec<SymbolSearchHit>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolSearchHit {
    pub symbol_id: String,
    pub repo: String,
    pub repo_label: String,
    pub relative_path: String,
    pub name: String,
    pub kind: String,
    pub container: Option<String>,
    pub language: String,
    pub start_line: u64,
    pub end_line: u64,
    pub score: f64,
    pub lexical_score: Option<f64>,
    pub semantic_score: Option<f64>,
    pub indexed_at: String,
    pub file_hash: String,
    pub stale: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileOutlineResponse {
    pub scope: String,
    pub label: String,
    pub file: String,
    pub matches: Vec<FileOutlineMatch>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FileOutlineMatch {
    pub repo: String,
    pub repo_label: String,
    pub relative_path: String,
    pub language: Option<String>,
    pub indexed_at: Option<String>,
    pub stale: bool,
    pub symbols: Vec<OutlineNode>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchMode {
    Auto,
    Semantic,
    Hybrid,
    Identifier,
    Path,
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: String,
    pub limit: usize,
    pub mode: SearchMode,
    pub extension_filter: Vec<String>,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub file: Option<String>,
    pub dedupe_by_file: bool,
}

#[derive(Debug, Clone)]
pub struct SymbolSearchScopeRequest {
    pub query: String,
    pub limit: usize,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub kind: Option<String>,
    pub container: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum QueryKind {
    NaturalLanguage,
    Identifier,
    Path,
    Mixed,
}

#[derive(Debug, Clone)]
struct SearchPlan {
    requested_mode: SearchMode,
    effective_mode: SearchMode,
    query_kind: QueryKind,
    dense_weight: f64,
    lexical_weight: f64,
    symbol_weight: f64,
    symbol_lexical_share: f64,
    symbol_semantic_share: f64,
    snippet_neighbor_chunks: usize,
    dedupe_by_file: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StatusReport {
    pub scope: String,
    pub label: String,
    pub kind: String,
    pub overall_status: String,
    pub indexed_files: u64,
    pub total_chunks: u64,
    pub repos: Vec<RepoStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identity_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepoStatus {
    pub repo: String,
    pub repo_label: String,
    pub collection_name: String,
    pub status: String,
    pub indexed_files: Option<u64>,
    pub total_chunks: Option<u64>,
    pub index_status: Option<String>,
    pub indexing_percentage: Option<f64>,
    pub last_attempted_percentage: Option<f64>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
struct RepoFile {
    absolute_path: PathBuf,
    hash: String,
}

#[derive(Debug, Clone, Default)]
struct FileDiff {
    added: Vec<String>,
    modified: Vec<String>,
    removed: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct MerkleSnapshot {
    #[serde(default, rename = "fileHashes")]
    file_hashes: Vec<(String, String)>,
    #[serde(default = "default_merkle_hash_algorithm", rename = "hashAlgorithm")]
    hash_algorithm: String,
    #[serde(default, rename = "rootHash")]
    root_hash: Option<String>,
    #[serde(default, rename = "merkleDAG")]
    merkle_dag: Option<Value>,
}

#[derive(Debug, Clone)]
struct RepoSearchHit {
    repo: String,
    relative_path: String,
    start_line: u64,
    end_line: u64,
    language: String,
    indexed_at: Option<String>,
    file_hash: Option<String>,
    content: String,
    dense_score: f64,
    lexical_score: f64,
    symbol_score: f64,
    combined_score: f64,
}

#[derive(Debug, Clone)]
struct RankedSymbolHit {
    symbol_id: String,
    relative_path: String,
    name: String,
    kind: String,
    container: Option<String>,
    language: String,
    start_line: u64,
    end_line: u64,
    indexed_at: String,
    file_hash: String,
    lexical_score: f64,
    semantic_score: f64,
    combined_score: f64,
}

#[derive(Debug, Clone)]
struct PendingChunk {
    chunk: CodeChunk,
    indexed_at: String,
    file_hash: String,
}

#[derive(Debug, Clone)]
struct PendingSymbolDocument {
    symbol: IndexedSymbol,
    basename: String,
    extension: String,
}

#[derive(Debug, Clone)]
struct FileFreshnessProbe {
    repo: PathBuf,
    relative_path: String,
    indexed_file_hash: Option<String>,
    repo_stale_hint: bool,
}

#[derive(Debug, Clone)]
struct RepoSearchOutcome {
    repo: String,
    stale: bool,
    hits: Vec<RepoSearchHit>,
    error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexCompletionStatus {
    Completed,
    LimitReached,
}

impl IndexCompletionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::LimitReached => "limit_reached",
        }
    }
}

impl SearchMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Semantic => "semantic",
            Self::Hybrid => "hybrid",
            Self::Identifier => "identifier",
            Self::Path => "path",
        }
    }
}

impl QueryKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::NaturalLanguage => "natural_language",
            Self::Identifier => "identifier",
            Self::Path => "path",
            Self::Mixed => "mixed",
        }
    }
}

impl SearchPlan {
    fn summary(&self) -> SearchPlanSummary {
        SearchPlanSummary {
            requested_mode: self.requested_mode.as_str().to_string(),
            effective_mode: self.effective_mode.as_str().to_string(),
            query_kind: self.query_kind.as_str().to_string(),
            dense_weight: self.dense_weight,
            lexical_weight: self.lexical_weight,
            symbol_weight: self.symbol_weight,
            symbol_lexical_share: self.symbol_lexical_share,
            symbol_semantic_share: self.symbol_semantic_share,
            dedupe_by_file: self.dedupe_by_file,
        }
    }

    fn lexical_flavor(&self) -> QueryFlavor {
        match self.query_kind {
            QueryKind::NaturalLanguage => QueryFlavor::NaturalLanguage,
            QueryKind::Identifier => QueryFlavor::Identifier,
            QueryKind::Path => QueryFlavor::Path,
            QueryKind::Mixed => QueryFlavor::Mixed,
        }
    }
}

#[derive(Debug, Clone)]
struct ProcessFilesResult {
    indexed_paths: HashSet<String>,
    processed_files: u64,
    total_chunks: u64,
    status: IndexCompletionStatus,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexIdentityStatus {
    pub compatible: bool,
    pub index_format_version: String,
    pub search_root_version: String,
    pub configured_embedding_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_embedding_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

struct ProgressTracker {
    last_write: Instant,
    last_bucket: i32,
}

#[derive(Clone, Copy)]
struct SymbolFusionConfig<'a> {
    query_vector: Option<&'a [f32]>,
    lexical_share: f64,
    semantic_share: f64,
    symbol_collection_exists: bool,
}

#[derive(Clone, Copy)]
struct IndexCollections<'a> {
    chunk: &'a str,
    symbol: Option<&'a str>,
}

impl ProgressTracker {
    fn new() -> Self {
        Self {
            last_write: Instant::now() - SNAPSHOT_PROGRESS_WRITE_INTERVAL,
            last_bucket: -1,
        }
    }

    fn should_persist(&mut self, progress: f64) -> bool {
        let clamped = progress.clamp(0.0, 100.0);
        let bucket = clamped.floor() as i32;
        if bucket > self.last_bucket
            || self.last_write.elapsed() >= SNAPSHOT_PROGRESS_WRITE_INTERVAL
        {
            self.last_bucket = bucket;
            self.last_write = Instant::now();
            return true;
        }
        false
    }
}

impl SearchBudgets {
    fn new(config: &crate::config::SearchConfig) -> Self {
        Self {
            requests: Arc::new(Semaphore::new(config.max_concurrent_requests)),
            repo_searches: Arc::new(Semaphore::new(config.max_concurrent_repo_searches)),
            lexical_tasks: Arc::new(Semaphore::new(config.max_concurrent_lexical_tasks)),
            dense_tasks: Arc::new(Semaphore::new(config.max_concurrent_dense_tasks)),
        }
    }

    async fn acquire_request(&self) -> Result<OwnedSemaphorePermit> {
        self.requests
            .clone()
            .acquire_owned()
            .await
            .context("acquiring global search request budget")
    }

    async fn acquire_repo_search(&self) -> Result<OwnedSemaphorePermit> {
        self.repo_searches
            .clone()
            .acquire_owned()
            .await
            .context("acquiring global repo search budget")
    }

    async fn acquire_lexical(&self) -> Result<OwnedSemaphorePermit> {
        self.lexical_tasks
            .clone()
            .acquire_owned()
            .await
            .context("acquiring global lexical search budget")
    }

    async fn acquire_dense(&self) -> Result<OwnedSemaphorePermit> {
        self.dense_tasks
            .clone()
            .acquire_owned()
            .await
            .context("acquiring global dense search budget")
    }
}

impl Engine {
    pub async fn new(config: &Config) -> Result<Self> {
        let embedding = EmbeddingClient::new(&config.embedding).await?;
        let milvus = MilvusClient::new(&config.milvus)?;
        Ok(Self {
            inner: Arc::new(EngineInner {
                config: config.clone(),
                snapshot: SnapshotStore::new(config.snapshot_path.clone()),
                milvus,
                embedding,
                local_index: LocalIndexStore::new(
                    config.search_index_dir(),
                    config.search.max_warm_repos,
                ),
                symbol_store: SymbolStore::new(config.symbol_db_path()),
                search_budgets: SearchBudgets::new(&config.search),
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    pub async fn embedding_fingerprint(&self) -> Result<String> {
        self.inner.embedding.fingerprint().await
    }

    pub async fn healthcheck(&self) -> Result<()> {
        self.inner.milvus.healthcheck().await?;
        let _ = self.embedding_fingerprint().await?;
        Ok(())
    }

    pub async fn index_identity_status(&self) -> Result<IndexIdentityStatus> {
        let snapshot = self.inner.snapshot.read().await?;
        let configured_embedding_fingerprint = self.embedding_fingerprint().await?;
        Ok(index_identity_status_for_snapshot(
            &snapshot,
            &configured_embedding_fingerprint,
        ))
    }

    pub fn all_scope(&self) -> Result<ResolvedScope> {
        Ok(ResolvedScope {
            kind: ScopeKind::Group,
            id: "all".to_string(),
            label: "All Configured Repos".to_string(),
            repos: self.inner.config.all_repos()?,
        })
    }

    pub async fn mark_interrupted_indexing_failed(&self, reason: &str) -> Result<usize> {
        self.inner
            .snapshot
            .mark_interrupted_indexing_failed(reason)
            .await
    }

    pub async fn mark_scope_indexing(&self, scope: &ResolvedScope) -> Result<()> {
        for repo in &scope.repos {
            let repo_key = repo.display().to_string();
            self.inner
                .snapshot
                .update(|snapshot| {
                    snapshot
                        .codebases
                        .insert(repo_key.clone(), SnapshotEntry::indexing(0.0));
                })
                .await?;
        }
        Ok(())
    }

    pub async fn index_scope(
        &self,
        scope: ResolvedScope,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
    ) -> Result<ScopeIndexResult> {
        self.ensure_index_identity(force).await?;
        self.persist_index_identity().await?;
        let mut results = Vec::new();
        let mut has_errors = false;

        for repo in scope.repos {
            match self
                .index_repo(&repo, force, splitter, custom_extensions, ignore_patterns)
                .await
            {
                Ok(result) => {
                    has_errors |= result.error.is_some();
                    results.push(result);
                }
                Err(error) => {
                    has_errors = true;
                    results.push(RepoIndexResult {
                        repo: repo.display().to_string(),
                        indexed_files: None,
                        total_chunks: None,
                        index_status: Some("failed".to_string()),
                        full_reindex: force,
                        changes: RepoChangeSummary::default(),
                        error: Some(error.to_string()),
                    });
                }
            }
        }

        Ok(ScopeIndexResult {
            scope: scope.id,
            label: scope.label,
            has_errors,
            repos: results,
        })
    }

    pub async fn clear_scope(&self, scope: ResolvedScope) -> Result<ScopeClearResult> {
        let mut repos = Vec::new();
        for repo in scope.repos {
            let repo_label = repo.display().to_string();
            let result = match self.clear_repo(&repo).await {
                Ok(()) => RepoClearResult {
                    repo: repo_label,
                    cleared: true,
                    error: None,
                },
                Err(error) => RepoClearResult {
                    repo: repo_label,
                    cleared: false,
                    error: Some(error.to_string()),
                },
            };
            repos.push(result);
        }
        Ok(ScopeClearResult {
            scope: scope.id,
            label: scope.label,
            repos,
        })
    }

    pub async fn explain_search(
        &self,
        scope: ResolvedScope,
        request: &SearchRequest,
    ) -> Result<SearchExplanation> {
        self.ensure_index_identity(false).await?;
        let plan = plan_search(request);
        Ok(SearchExplanation {
            scope: scope.id,
            label: scope.label,
            query: request.query.clone(),
            plan: plan.summary(),
        })
    }

    async fn acquire_request_budget(&self) -> Result<OwnedSemaphorePermit> {
        self.inner.search_budgets.acquire_request().await
    }

    async fn acquire_repo_search_budget(&self) -> Result<OwnedSemaphorePermit> {
        self.inner.search_budgets.acquire_repo_search().await
    }

    async fn acquire_lexical_budget(&self) -> Result<OwnedSemaphorePermit> {
        self.inner.search_budgets.acquire_lexical().await
    }

    async fn acquire_dense_budget(&self) -> Result<OwnedSemaphorePermit> {
        self.inner.search_budgets.acquire_dense().await
    }

    async fn run_search_lexical_blocking<T, F>(&self, label: &'static str, work: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> Result<T> + Send + 'static,
    {
        let _permit = self.acquire_lexical_budget().await?;
        run_low_priority_blocking(label, work).await
    }

    async fn has_collection_on_search_path(&self, collection_name: &str) -> Result<bool> {
        let _permit = self.acquire_dense_budget().await?;
        self.inner.milvus.has_collection(collection_name).await
    }

    async fn search_symbol_collection_presence(
        &self,
        repos: &[PathBuf],
    ) -> Result<HashMap<String, bool>> {
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let repo_checks = repos.to_vec();
        let mut stream = futures::stream::iter(repo_checks.into_iter().map(|repo| {
            let engine = self.clone();
            async move {
                let _repo_permit = engine.acquire_repo_search_budget().await?;
                let exists = engine
                    .has_collection_on_search_path(&symbol_collection_name(&repo))
                    .await?;
                Ok::<_, anyhow::Error>((repo.display().to_string(), exists))
            }
        }))
        .buffer_unordered(parallelism);

        let mut presence = HashMap::new();
        while let Some(item) = stream.next().await {
            let (repo_key, exists) = item?;
            presence.insert(repo_key, exists);
        }
        Ok(presence)
    }

    pub async fn search_scope(
        &self,
        scope: ResolvedScope,
        request: SearchRequest,
    ) -> Result<SearchResponse> {
        self.ensure_index_identity(false).await?;
        let _request_permit = self.acquire_request_budget().await?;
        let plan = plan_search(&request);
        let query_vector = if plan.dense_weight > 0.0 || plan.symbol_semantic_share > 0.0 {
            let _dense_permit = self.acquire_dense_budget().await?;
            Some(Arc::new(
                self.inner.embedding.embed_query(&request.query).await?,
            ))
        } else {
            None
        };
        let per_repo_limit = (request.limit.max(5) * 4).min(64);
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let filter_expression = build_extension_filter(&request.extension_filter);
        let snapshot = self.inner.snapshot.read().await?;

        let mut stream = futures::stream::iter(scope.repos.into_iter().map(|repo| {
            let engine = self.clone();
            let request = request.clone();
            let plan = plan.clone();
            let query_vector = query_vector.clone();
            let filter_expression = filter_expression.clone();
            let entry = snapshot.codebases.get(&repo.display().to_string()).cloned();
            async move {
                let repo_label = repo.display().to_string();
                let _repo_permit = match engine.acquire_repo_search_budget().await {
                    Ok(permit) => permit,
                    Err(error) => {
                        return RepoSearchOutcome {
                            repo: repo_label,
                            stale: false,
                            hits: Vec::new(),
                            error: Some(error.to_string()),
                        };
                    }
                };
                let stale = engine
                    .repo_is_stale(&repo, entry.as_ref())
                    .await
                    .unwrap_or(false);
                match engine
                    .search_repo(
                        &repo,
                        &request,
                        &plan,
                        query_vector.as_ref().map(|value| value.as_slice()),
                        per_repo_limit,
                        filter_expression.as_deref(),
                    )
                    .await
                {
                    Ok(hits) => RepoSearchOutcome {
                        repo: repo_label,
                        stale,
                        hits,
                        error: None,
                    },
                    Err(error) => RepoSearchOutcome {
                        repo: repo_label,
                        stale,
                        hits: Vec::new(),
                        error: Some(error.to_string()),
                    },
                }
            }
        }))
        .buffer_unordered(parallelism);

        let mut repo_errors = Vec::new();
        let mut merged = Vec::new();
        while let Some(outcome) = stream.next().await {
            if let Some(error) = outcome.error {
                repo_errors.push(RepoSearchError {
                    repo: outcome.repo,
                    error,
                });
                continue;
            }
            for mut hit in outcome.hits {
                hit.combined_score = hit.combined_score.max(0.0);
                merged.push((hit, outcome.stale));
            }
        }

        merged.sort_by(|left, right| {
            right
                .0
                .combined_score
                .partial_cmp(&left.0.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if plan.dedupe_by_file {
            merged = dedupe_repo_search_hits_by_file(merged);
        }
        merged.truncate(request.limit);

        let hits = self.finalize_search_hits(&merged, &request, &plan).await?;

        Ok(SearchResponse {
            scope: scope.id,
            label: scope.label,
            partial: !repo_errors.is_empty(),
            repo_errors,
            plan: plan.summary(),
            hits,
        })
    }

    pub async fn search_symbols(
        &self,
        scope: ResolvedScope,
        request: SymbolSearchScopeRequest,
    ) -> Result<SymbolSearchResponse> {
        self.ensure_index_identity(false).await?;
        let _request_permit = self.acquire_request_budget().await?;
        let flavor = classify_query(&request.query);
        let (lexical_share, semantic_share) = symbol_query_shares(flavor);
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let scope_repos = scope.repos;
        let symbol_collection_presence = if semantic_share > 0.0 {
            self.search_symbol_collection_presence(&scope_repos).await?
        } else {
            HashMap::new()
        };
        let any_symbol_collections = symbol_collection_presence.values().any(|exists| *exists);
        let query_vector = if semantic_share > 0.0 && any_symbol_collections {
            let _dense_permit = self.acquire_dense_budget().await?;
            Some(Arc::new(
                self.inner.embedding.embed_query(&request.query).await?,
            ))
        } else {
            None
        };
        let per_repo_limit = (request.limit.max(5) * 4).min(64);
        let snapshot = self.inner.snapshot.read().await?;

        let mut stream = futures::stream::iter(scope_repos.into_iter().map(|repo| {
            let engine = self.clone();
            let request = request.clone();
            let query_vector = query_vector.clone();
            let entry = snapshot.codebases.get(&repo.display().to_string()).cloned();
            let symbol_collection_exists = symbol_collection_presence
                .get(&repo.display().to_string())
                .copied()
                .unwrap_or(false);
            async move {
                let _repo_permit = engine.acquire_repo_search_budget().await?;
                let stale = engine
                    .repo_is_stale(&repo, entry.as_ref())
                    .await
                    .unwrap_or(false);
                match engine
                    .search_symbol_repo(
                        &repo,
                        &request,
                        flavor,
                        per_repo_limit,
                        SymbolFusionConfig {
                            query_vector: query_vector.as_ref().map(|value| value.as_slice()),
                            lexical_share,
                            semantic_share,
                            symbol_collection_exists,
                        },
                    )
                    .await
                {
                    Ok(hits) => Ok::<_, anyhow::Error>((repo, stale, hits)),
                    Err(error) => Err(anyhow::anyhow!("{}: {error}", repo.display())),
                }
            }
        }))
        .buffer_unordered(parallelism);

        let mut repo_errors = Vec::new();
        let mut hits = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok((repo, stale, repo_hits)) => {
                    for hit in repo_hits {
                        hits.push(SymbolSearchHit {
                            symbol_id: hit.symbol_id,
                            repo: repo.display().to_string(),
                            repo_label: repo_basename(&repo.display().to_string()),
                            relative_path: hit.relative_path,
                            name: hit.name,
                            kind: hit.kind,
                            container: hit.container,
                            language: hit.language,
                            start_line: hit.start_line,
                            end_line: hit.end_line,
                            score: hit.combined_score,
                            lexical_score: (hit.lexical_score > 0.0).then_some(hit.lexical_score),
                            semantic_score: (hit.semantic_score > 0.0)
                                .then_some(hit.semantic_score),
                            indexed_at: hit.indexed_at,
                            file_hash: hit.file_hash,
                            stale,
                        });
                    }
                }
                Err(error) => repo_errors.push(RepoSearchError {
                    repo: error
                        .to_string()
                        .split(':')
                        .next()
                        .unwrap_or("unknown")
                        .to_string(),
                    error: error.to_string(),
                }),
            }
        }

        hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(request.limit);

        let probes = hits
            .iter()
            .map(|hit| FileFreshnessProbe {
                repo: PathBuf::from(&hit.repo),
                relative_path: hit.relative_path.clone(),
                indexed_file_hash: Some(hit.file_hash.clone()),
                repo_stale_hint: hit.stale,
            })
            .collect::<Vec<_>>();
        let fresh_map = self.compute_file_staleness_map(probes).await?;
        for hit in &mut hits {
            let key = file_freshness_key(Path::new(&hit.repo), &hit.relative_path);
            if let Some(stale) = fresh_map.get(&key) {
                hit.stale = *stale;
            }
        }

        Ok(SymbolSearchResponse {
            scope: scope.id,
            label: scope.label,
            partial: !repo_errors.is_empty(),
            repo_errors,
            hits,
        })
    }

    pub async fn get_file_outline(
        &self,
        scope: ResolvedScope,
        file: &str,
    ) -> Result<FileOutlineResponse> {
        self.ensure_index_identity(false).await?;
        let normalized_file = normalize_relative_path(file);
        let snapshot = self.inner.snapshot.read().await?;
        let mut matches = Vec::new();

        for repo in scope.repos {
            let repo_key = repo.display().to_string();
            let symbols = self
                .load_file_outline_symbols(&repo_key, &normalized_file)
                .await?;
            if symbols.is_empty() {
                continue;
            }
            let repo_stale = self
                .repo_is_stale(&repo, snapshot.codebases.get(&repo_key))
                .await
                .unwrap_or(false);
            let stale = self
                .file_is_stale(
                    &repo,
                    &normalized_file,
                    symbols.first().map(|symbol| symbol.file_hash.as_str()),
                    repo_stale,
                )
                .await
                .unwrap_or(repo_stale);
            let indexed_at = symbols.first().map(|symbol| symbol.indexed_at.clone());
            let language = symbols.first().map(|symbol| symbol.language.clone());
            matches.push(FileOutlineMatch {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                relative_path: normalized_file.clone(),
                language,
                indexed_at,
                stale,
                symbols: build_outline(&symbols),
            });
        }

        Ok(FileOutlineResponse {
            scope: scope.id,
            label: scope.label,
            file: normalized_file,
            matches,
        })
    }

    async fn finalize_search_hits(
        &self,
        hits: &[(RepoSearchHit, bool)],
        request: &SearchRequest,
        plan: &SearchPlan,
    ) -> Result<Vec<SearchHit>> {
        let probes = hits
            .iter()
            .map(|(hit, repo_stale)| FileFreshnessProbe {
                repo: PathBuf::from(&hit.repo),
                relative_path: hit.relative_path.clone(),
                indexed_file_hash: hit.file_hash.clone(),
                repo_stale_hint: *repo_stale,
            })
            .collect::<Vec<_>>();
        let stale_map = self.compute_file_staleness_map(probes).await?;

        let mut finalized = Vec::with_capacity(hits.len());
        let mut snippet_cache: HashMap<String, String> = HashMap::new();
        for (hit, repo_stale) in hits {
            let stale = stale_map
                .get(&file_freshness_key(
                    Path::new(&hit.repo),
                    &hit.relative_path,
                ))
                .copied()
                .unwrap_or(*repo_stale);
            let snippet_key = format!("{}:{}", hit.repo, hit.relative_path);
            let content = if let Some(snippet) = snippet_cache.get(&snippet_key) {
                snippet.clone()
            } else {
                let snippet = self
                    .build_hit_snippet(
                        Path::new(&hit.repo),
                        hit,
                        &request.query,
                        plan.snippet_neighbor_chunks,
                    )
                    .await;
                snippet_cache.insert(snippet_key, snippet.clone());
                snippet
            };
            finalized.push(SearchHit {
                repo: hit.repo.clone(),
                repo_label: repo_basename(&hit.repo),
                relative_path: hit.relative_path.clone(),
                start_line: hit.start_line,
                end_line: hit.end_line,
                language: hit.language.clone(),
                score: hit.combined_score,
                match_type: match_type_for_hit(hit),
                dense_score: (hit.dense_score > 0.0).then_some(hit.dense_score),
                lexical_score: (hit.lexical_score > 0.0).then_some(hit.lexical_score),
                symbol_score: (hit.symbol_score > 0.0).then_some(hit.symbol_score),
                indexed_at: hit.indexed_at.clone(),
                stale,
                content,
            });
        }

        Ok(finalized)
    }

    async fn build_hit_snippet(
        &self,
        repo: &Path,
        hit: &RepoSearchHit,
        query: &str,
        neighbor_chunks: usize,
    ) -> String {
        let repo_path = repo.to_path_buf();
        let relative_path = hit.relative_path.clone();
        let target_line = hit.start_line;
        let fallback = build_snippet(query, &hit.content, 900);
        let local_index = self.inner.local_index.clone();
        let context = self
            .run_search_lexical_blocking("search_hit_context", move || {
                let chunks = local_index.chunks_for_file(&repo_path, &relative_path)?;
                Ok(build_chunk_context_snippet(
                    &chunks,
                    target_line,
                    neighbor_chunks,
                    1400,
                ))
            })
            .await;

        match context {
            Ok(Some(context)) => build_snippet(query, &context, 1200),
            _ => fallback,
        }
    }

    async fn compute_file_staleness_map(
        &self,
        probes: Vec<FileFreshnessProbe>,
    ) -> Result<HashMap<String, bool>> {
        self.run_search_lexical_blocking("compute_file_staleness", move || {
            let mut deduped = BTreeMap::new();
            for probe in probes {
                deduped
                    .entry(file_freshness_key(&probe.repo, &probe.relative_path))
                    .or_insert(probe);
            }

            let mut stale = HashMap::with_capacity(deduped.len());
            for (key, probe) in deduped {
                let is_stale = if !probe.repo_stale_hint {
                    false
                } else {
                    let path = probe.repo.join(&probe.relative_path);
                    match probe.indexed_file_hash.as_deref() {
                        Some(indexed_file_hash) => match hash_text_like_file(&path)? {
                            Some(current_hash) => current_hash != indexed_file_hash,
                            None => true,
                        },
                        None => true,
                    }
                };
                stale.insert(key, is_stale);
            }
            Ok(stale)
        })
        .await
    }

    async fn file_is_stale(
        &self,
        repo: &Path,
        relative_path: &str,
        indexed_file_hash: Option<&str>,
        repo_stale_hint: bool,
    ) -> Result<bool> {
        let stale_map = self
            .compute_file_staleness_map(vec![FileFreshnessProbe {
                repo: repo.to_path_buf(),
                relative_path: relative_path.to_string(),
                indexed_file_hash: indexed_file_hash.map(ToString::to_string),
                repo_stale_hint,
            }])
            .await?;
        Ok(stale_map
            .get(&file_freshness_key(repo, relative_path))
            .copied()
            .unwrap_or(repo_stale_hint))
    }

    pub async fn status_scope(&self, scope: ResolvedScope) -> Result<StatusReport> {
        let snapshot = self.inner.snapshot.read().await?;
        let identity_status =
            index_identity_status_for_snapshot(&snapshot, &self.embedding_fingerprint().await?);
        let mut repos = Vec::new();
        let mut indexed_files = 0u64;
        let mut total_chunks = 0u64;

        for repo in scope.repos {
            let repo_key = repo.display().to_string();
            let status = self.status_for_repo(&snapshot, &repo).await?;
            indexed_files += status.indexed_files.unwrap_or(0);
            total_chunks += status.total_chunks.unwrap_or(0);
            repos.push(RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                ..status
            });
        }

        let overall_status = overall_status(&repos);
        Ok(StatusReport {
            scope: scope.id,
            label: scope.label,
            kind: match scope.kind {
                ScopeKind::Repo => "repo".to_string(),
                ScopeKind::Group => "group".to_string(),
            },
            overall_status,
            indexed_files,
            total_chunks,
            repos,
            identity_error: identity_status.reason,
        })
    }

    pub async fn cheap_audit_once(&self) -> Result<Vec<RepoIndexResult>> {
        let repos = self.inner.config.all_repos()?;
        let snapshot = self.inner.snapshot.read().await?;
        let mut refreshed = Vec::new();

        for repo in repos {
            let repo_key = repo.display().to_string();
            let Some(existing) = snapshot.codebases.get(&repo_key).cloned() else {
                continue;
            };
            if existing.status != "indexed" {
                continue;
            }

            let repo_path = repo.clone();
            let Ok(fingerprint) =
                run_low_priority_blocking("fingerprint_repo", move || fingerprint_repo(&repo_path))
                    .await
            else {
                continue;
            };

            if fingerprint_changed(Some(&existing), &fingerprint) {
                let result = self
                    .index_repo(&repo, false, SplitterKind::Ast, &[], &[])
                    .await?;
                refreshed.push(result);
            } else {
                self.record_fingerprint(&repo_key, &fingerprint).await?;
            }
        }

        Ok(refreshed)
    }

    async fn ensure_index_identity(&self, allow_rebuild: bool) -> Result<()> {
        let status = self.index_identity_status().await?;
        if status.compatible || allow_rebuild {
            return Ok(());
        }

        bail!(
            "{} Run `agent-context reindex-all` to rebuild the local indexes.",
            status.reason.unwrap_or_else(|| {
                "local index state is incompatible with the current configuration".to_string()
            })
        );
    }

    async fn persist_index_identity(&self) -> Result<()> {
        let configured_embedding_fingerprint = self.embedding_fingerprint().await?;
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.index_format_version = INDEX_FORMAT_VERSION.to_string();
                snapshot.search_root_version = SEARCH_ROOT_VERSION.to_string();
                snapshot.embedding_fingerprint = Some(configured_embedding_fingerprint.clone());
            })
            .await?;
        Ok(())
    }

    async fn index_repo(
        &self,
        repo: &Path,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
    ) -> Result<RepoIndexResult> {
        validate_repo_path(repo)?;
        let repo_key = repo.display().to_string();
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot
                    .codebases
                    .insert(repo_key.clone(), SnapshotEntry::indexing(0.0));
            })
            .await?;

        let collection_name = collection_name(repo);
        let symbol_collection_name = symbol_collection_name(repo);
        let merkle_path = merkle_snapshot_path(&self.inner.config.merkle_dir(), repo);
        let repo_path = repo.to_path_buf();
        let custom_extensions = custom_extensions.to_vec();
        let ignore_patterns = ignore_patterns.to_vec();
        let current_files = run_low_priority_blocking("scan_repo", move || {
            scan_repo(&repo_path, &custom_extensions, &ignore_patterns)
        })
        .await?;
        let current_hashes = current_files
            .iter()
            .map(|(path, file)| (path.clone(), file.hash.clone()))
            .collect::<BTreeMap<_, _>>();

        let collection_exists = self
            .inner
            .milvus
            .refresh_collection_presence(&collection_name)
            .await?;
        let symbol_collection_exists = self
            .inner
            .milvus
            .refresh_collection_presence(&symbol_collection_name)
            .await?;
        let previous_snapshot = if force || !collection_exists {
            None
        } else {
            load_merkle_snapshot(&merkle_path)
                .await
                .ok()
                .filter(MerkleSnapshot::is_compatible)
        };
        let full_reindex = force || !collection_exists || previous_snapshot.is_none();

        let outcome = async {
            if full_reindex {
                if collection_exists {
                    self.inner.milvus.drop_collection(&collection_name).await?;
                }
                if symbol_collection_exists {
                    self.inner
                        .milvus
                        .drop_collection(&symbol_collection_name)
                        .await?;
                }
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                run_low_priority_blocking("clear_repo_local_index", move || {
                    local_index.clear_repo(&repo_path)
                })
                .await?;
                let symbol_store = self.inner.symbol_store.clone();
                let repo_key_for_symbols = repo_key.clone();
                run_low_priority_blocking("clear_repo_symbols", move || {
                    symbol_store.clear_repo(&repo_key_for_symbols)
                })
                .await?;
                self.inner
                    .milvus
                    .create_hybrid_collection(
                        &collection_name,
                        self.inner.embedding.dimension().await?,
                        &format!("codebasePath:{}", repo.display()),
                    )
                    .await?;
                self.inner
                    .milvus
                    .create_hybrid_collection(
                        &symbol_collection_name,
                        self.inner.embedding.dimension().await?,
                        &format!("symbolCodebasePath:{}", repo.display()),
                    )
                    .await?;
                let files = current_files.values().cloned().collect::<Vec<_>>();
                let processing = self
                    .process_files(
                        repo,
                        &repo_key,
                        IndexCollections {
                            chunk: &collection_name,
                            symbol: Some(&symbol_collection_name),
                        },
                        &files,
                        splitter,
                        current_files.len(),
                    )
                    .await?;
                let indexed_hashes = current_hashes
                    .iter()
                    .filter(|(path, _)| processing.indexed_paths.contains(*path))
                    .map(|(path, hash)| (path.clone(), hash.clone()))
                    .collect::<BTreeMap<_, _>>();
                save_merkle_snapshot(&merkle_path, &indexed_hashes).await?;
                let changes = RepoChangeSummary {
                    added: processing.processed_files,
                    modified: 0,
                    removed: 0,
                };
                Ok::<(ProcessFilesResult, RepoChangeSummary), anyhow::Error>((processing, changes))
            } else {
                let symbol_collection_ready = if symbol_collection_exists {
                    true
                } else {
                    self.inner
                        .milvus
                        .create_hybrid_collection(
                            &symbol_collection_name,
                            self.inner.embedding.dimension().await?,
                            &format!("symbolCodebasePath:{}", repo.display()),
                        )
                        .await?;
                    true
                };
                let previous_snapshot = previous_snapshot.expect("checked above");
                let previous_hashes = previous_snapshot
                    .file_hashes
                    .into_iter()
                    .collect::<BTreeMap<_, _>>();
                let diff = diff_files(&previous_hashes, &current_hashes);

                for relative_path in diff.removed.iter().chain(diff.modified.iter()) {
                    let ids = self
                        .inner
                        .milvus
                        .query_ids_for_path(&collection_name, relative_path)
                        .await?;
                    self.inner.milvus.delete_ids(&collection_name, &ids).await?;
                    if symbol_collection_ready {
                        let symbol_ids = self
                            .inner
                            .milvus
                            .query_ids_for_path(&symbol_collection_name, relative_path)
                            .await?;
                        self.inner
                            .milvus
                            .delete_ids(&symbol_collection_name, &symbol_ids)
                            .await?;
                    }
                }
                let deleted_paths = diff
                    .removed
                    .iter()
                    .chain(diff.modified.iter())
                    .cloned()
                    .collect::<Vec<_>>();
                if !deleted_paths.is_empty() {
                    let repo_path = repo.to_path_buf();
                    let local_index = self.inner.local_index.clone();
                    let paths = deleted_paths.clone();
                    run_low_priority_blocking("delete_local_paths", move || {
                        local_index.delete_paths(&repo_path, &paths)
                    })
                    .await?;
                    let symbol_store = self.inner.symbol_store.clone();
                    let repo_key_for_symbols = repo_key.clone();
                    run_low_priority_blocking("delete_symbol_rows", move || {
                        for path in deleted_paths {
                            symbol_store.delete_file(&repo_key_for_symbols, &path)?;
                        }
                        Ok(())
                    })
                    .await?;
                }

                let to_index = diff
                    .added
                    .iter()
                    .chain(diff.modified.iter())
                    .filter_map(|path| current_files.get(path).cloned())
                    .collect::<Vec<_>>();
                let processing = self
                    .process_files(
                        repo,
                        &repo_key,
                        IndexCollections {
                            chunk: &collection_name,
                            symbol: symbol_collection_ready
                                .then_some(symbol_collection_name.as_str()),
                        },
                        &to_index,
                        splitter,
                        to_index.len(),
                    )
                    .await?;
                let mut persisted_hashes = previous_hashes
                    .into_iter()
                    .filter(|(path, _)| {
                        !diff.removed.contains(path) && !diff.modified.contains(path)
                    })
                    .collect::<BTreeMap<_, _>>();
                for relative_path in &processing.indexed_paths {
                    if let Some(hash) = current_hashes.get(relative_path) {
                        persisted_hashes.insert(relative_path.clone(), hash.clone());
                    }
                }
                save_merkle_snapshot(&merkle_path, &persisted_hashes).await?;
                let changes = RepoChangeSummary {
                    added: diff
                        .added
                        .iter()
                        .filter(|path| processing.indexed_paths.contains(*path))
                        .count() as u64,
                    modified: diff
                        .modified
                        .iter()
                        .filter(|path| processing.indexed_paths.contains(*path))
                        .count() as u64,
                    removed: diff.removed.len() as u64,
                };
                Ok((processing, changes))
            }
        }
        .await;

        match outcome {
            Ok((processing, changes)) => {
                let fingerprint = fingerprint_repo(repo).ok();
                let index_status =
                    if processing.processed_files == 0 && processing.total_chunks == 0 {
                        "empty".to_string()
                    } else {
                        processing.status.as_str().to_string()
                    };
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        let entry = snapshot
                            .codebases
                            .entry(repo_key.clone())
                            .or_insert_with(|| SnapshotEntry::indexing(0.0));
                        *entry = SnapshotEntry::indexed_with_status(
                            Some(processing.processed_files),
                            Some(processing.total_chunks),
                            index_status.clone(),
                        );
                        if let Some(fingerprint) = &fingerprint {
                            apply_fingerprint(entry, fingerprint);
                        }
                    })
                    .await?;

                Ok(RepoIndexResult {
                    repo: repo_key,
                    indexed_files: Some(processing.processed_files),
                    total_chunks: Some(processing.total_chunks),
                    index_status: Some(index_status),
                    full_reindex,
                    changes,
                    error: None,
                })
            }
            Err(error) => {
                let message = error.to_string();
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        let last_progress = snapshot.codebases.get(&repo_key).and_then(|entry| {
                            entry
                                .indexing_percentage
                                .or(entry.last_attempted_percentage)
                        });
                        snapshot.codebases.insert(
                            repo_key.clone(),
                            SnapshotEntry::failed(message.clone(), last_progress),
                        );
                    })
                    .await?;
                Ok(RepoIndexResult {
                    repo: repo_key,
                    indexed_files: None,
                    total_chunks: None,
                    index_status: Some("failed".to_string()),
                    full_reindex,
                    changes: RepoChangeSummary::default(),
                    error: Some(message),
                })
            }
        }
    }

    async fn clear_repo(&self, repo: &Path) -> Result<()> {
        validate_repo_path(repo)?;
        let repo_key = repo.display().to_string();
        let collection_name = collection_name(repo);
        let symbol_collection_name = symbol_collection_name(repo);
        if self
            .inner
            .milvus
            .refresh_collection_presence(&collection_name)
            .await?
        {
            self.inner.milvus.drop_collection(&collection_name).await?;
        }
        if self
            .inner
            .milvus
            .refresh_collection_presence(&symbol_collection_name)
            .await?
        {
            self.inner
                .milvus
                .drop_collection(&symbol_collection_name)
                .await?;
        }
        let repo_path = repo.to_path_buf();
        let local_index = self.inner.local_index.clone();
        run_low_priority_blocking("clear_local_repo_index", move || {
            local_index.clear_repo(&repo_path)
        })
        .await?;
        let repo_key_clone = repo_key.clone();
        let symbol_store = self.inner.symbol_store.clone();
        run_low_priority_blocking("clear_repo_symbol_rows", move || {
            symbol_store.clear_repo(&repo_key_clone)
        })
        .await?;

        let merkle_path = merkle_snapshot_path(&self.inner.config.merkle_dir(), repo);
        if merkle_path.exists() {
            tokio::fs::remove_file(&merkle_path)
                .await
                .with_context(|| format!("removing {}", merkle_path.display()))?;
        }
        self.inner.snapshot.remove(&repo_key).await?;
        Ok(())
    }

    async fn search_repo(
        &self,
        repo: &Path,
        request: &SearchRequest,
        plan: &SearchPlan,
        query_vector: Option<&[f32]>,
        limit: usize,
        filter_expression: Option<&str>,
    ) -> Result<Vec<RepoSearchHit>> {
        let collection_name = collection_name(repo);
        let dense_hits = if let Some(query_vector) = query_vector {
            if plan.dense_weight > 0.0 {
                let _dense_permit = self.acquire_dense_budget().await?;
                if self.inner.milvus.has_collection(&collection_name).await? {
                    self.inner
                        .milvus
                        .search_dense(&collection_name, query_vector, limit, filter_expression)
                        .await?
                        .into_iter()
                        .filter(|hit| search_document_matches(hit, request))
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let lexical_hits = self
            .run_search_lexical_blocking("search_chunk_index", {
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                let request = ChunkSearchRequest {
                    query: request.query.clone(),
                    limit,
                    flavor: plan.lexical_flavor(),
                    path_prefix: request.path_prefix.clone(),
                    language: request.language.clone(),
                    file: request.file.clone(),
                    extension_filter: request.extension_filter.clone(),
                };
                move || local_index.search_chunks(&repo_path, &request)
            })
            .await?;

        let symbol_collection_exists = if plan.symbol_semantic_share > 0.0 {
            self.has_collection_on_search_path(&symbol_collection_name(repo))
                .await?
        } else {
            false
        };

        let symbol_hits = self
            .search_symbol_repo(
                repo,
                &SymbolSearchScopeRequest {
                    query: request.query.clone(),
                    limit,
                    path_prefix: request.path_prefix.clone(),
                    language: request.language.clone(),
                    kind: None,
                    container: None,
                },
                plan.query_kind,
                limit,
                SymbolFusionConfig {
                    query_vector,
                    lexical_share: plan.symbol_lexical_share,
                    semantic_share: plan.symbol_semantic_share,
                    symbol_collection_exists,
                },
            )
            .await?;

        let mut merged: HashMap<String, RepoSearchHit> = HashMap::new();
        accumulate_dense_hits(&mut merged, repo, dense_hits, plan.dense_weight);
        accumulate_lexical_hits(&mut merged, repo, lexical_hits, plan.lexical_weight);
        self.accumulate_symbol_hits(&mut merged, repo, symbol_hits, plan.symbol_weight)
            .await?;

        let mut hits = merged.into_values().collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(hits)
    }

    async fn search_symbol_repo(
        &self,
        repo: &Path,
        request: &SymbolSearchScopeRequest,
        flavor: QueryKind,
        limit: usize,
        fusion: SymbolFusionConfig<'_>,
    ) -> Result<Vec<RankedSymbolHit>> {
        let indexed_hits = self
            .run_search_lexical_blocking("search_symbol_index", {
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                let request = SymbolSearchRequest {
                    query: request.query.clone(),
                    limit,
                    flavor: query_flavor(flavor),
                    path_prefix: request.path_prefix.clone(),
                    language: request.language.clone(),
                    kind: request.kind.clone(),
                    container: request.container.clone(),
                };
                move || local_index.search_symbols(&repo_path, &request)
            })
            .await?;

        let semantic_hits = if fusion.semantic_share > 0.0 && fusion.symbol_collection_exists {
            let collection_name = symbol_collection_name(repo);
            if let Some(query_vector) = fusion.query_vector {
                let _dense_permit = self.acquire_dense_budget().await?;
                self.inner
                    .milvus
                    .search_dense(
                        &collection_name,
                        query_vector,
                        limit.saturating_mul(4),
                        None,
                    )
                    .await?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let mut symbol_ids = indexed_hits
            .iter()
            .map(|hit| hit.symbol_id.clone())
            .collect::<Vec<_>>();
        symbol_ids.extend(
            semantic_hits
                .iter()
                .filter_map(|hit| metadata_string(&hit.metadata, "symbolId")),
        );
        symbol_ids.sort();
        symbol_ids.dedup();
        let authoritative = self
            .run_search_lexical_blocking("load_symbol_rows", {
                let symbol_store = self.inner.symbol_store.clone();
                move || symbol_store.symbols_by_ids(&symbol_ids)
            })
            .await?;
        let authoritative = authoritative
            .into_iter()
            .map(|symbol| (symbol.symbol_id.clone(), symbol))
            .collect::<HashMap<_, _>>();

        let mut merged = HashMap::new();
        accumulate_ranked_symbol_lexical_hits(
            &mut merged,
            &authoritative,
            request,
            indexed_hits,
            fusion.lexical_share,
        );
        accumulate_ranked_symbol_semantic_hits(
            &mut merged,
            &authoritative,
            request,
            semantic_hits,
            fusion.semantic_share,
        );

        let mut hits = merged.into_values().collect::<Vec<_>>();
        hits.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(hits)
    }

    async fn accumulate_symbol_hits(
        &self,
        merged: &mut HashMap<String, RepoSearchHit>,
        repo: &Path,
        symbol_hits: Vec<RankedSymbolHit>,
        weight: f64,
    ) -> Result<()> {
        if weight <= 0.0 {
            return Ok(());
        }

        let max_score = symbol_hits
            .first()
            .map(|hit| hit.combined_score)
            .filter(|score| *score > 0.0)
            .unwrap_or(1.0);
        let mut file_chunk_cache: HashMap<String, Vec<LexicalChunkSearchHit>> = HashMap::new();
        for (rank, hit) in symbol_hits.into_iter().enumerate() {
            if !file_chunk_cache.contains_key(&hit.relative_path) {
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                let relative_path = hit.relative_path.clone();
                let chunks = self
                    .run_search_lexical_blocking("chunks_for_symbol_file", move || {
                        local_index.chunks_for_file(&repo_path, &relative_path)
                    })
                    .await?;
                file_chunk_cache.insert(hit.relative_path.clone(), chunks);
            }
            let chunk_hit = file_chunk_cache
                .get(&hit.relative_path)
                .and_then(|chunks| nearest_chunk_for_line(chunks, hit.start_line));

            let score = fused_source_score(rank, hit.combined_score, max_score, weight);
            if let Some(chunk_hit) = chunk_hit {
                let key = format!("{}:{}", repo.display(), chunk_hit.id);
                let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
                    repo: repo.display().to_string(),
                    relative_path: chunk_hit.relative_path.clone(),
                    start_line: chunk_hit.start_line,
                    end_line: chunk_hit.end_line,
                    language: chunk_hit.language.clone(),
                    indexed_at: Some(chunk_hit.indexed_at.clone()),
                    file_hash: Some(chunk_hit.file_hash.clone()),
                    content: chunk_hit.content.clone(),
                    dense_score: 0.0,
                    lexical_score: 0.0,
                    symbol_score: 0.0,
                    combined_score: 0.0,
                });
                entry.symbol_score += score;
                entry.combined_score += score;
            } else {
                let id = format!("symbol:{}", hit.symbol_id);
                let key = format!("{}:{id}", repo.display());
                let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
                    repo: repo.display().to_string(),
                    relative_path: hit.relative_path.clone(),
                    start_line: hit.start_line,
                    end_line: hit.end_line,
                    language: hit.language.clone(),
                    indexed_at: Some(hit.indexed_at.clone()),
                    file_hash: Some(hit.file_hash.clone()),
                    content: format!("{} {}", hit.kind, hit.name),
                    dense_score: 0.0,
                    lexical_score: 0.0,
                    symbol_score: 0.0,
                    combined_score: 0.0,
                });
                entry.symbol_score += score;
                entry.combined_score += score;
            }
        }
        Ok(())
    }

    async fn load_file_outline_symbols(
        &self,
        repo_key: &str,
        relative_path: &str,
    ) -> Result<Vec<IndexedSymbol>> {
        let repo_key = repo_key.to_string();
        let relative_path = relative_path.to_string();
        let symbol_store = self.inner.symbol_store.clone();
        self.run_search_lexical_blocking("load_file_outline_symbols", move || {
            symbol_store.file_symbols(&repo_key, &relative_path)
        })
        .await
    }

    async fn repo_is_stale(&self, repo: &Path, entry: Option<&SnapshotEntry>) -> Result<bool> {
        let repo_path = repo.to_path_buf();
        let entry = entry.cloned();
        self.run_search_lexical_blocking("repo_stale_check", move || {
            let fingerprint = fingerprint_repo(&repo_path)?;
            Ok(fingerprint_changed(entry.as_ref(), &fingerprint))
        })
        .await
    }

    async fn status_for_repo(&self, snapshot: &Snapshot, repo: &Path) -> Result<RepoStatus> {
        let repo_key = repo.display().to_string();
        let collection_name = collection_name(repo);
        let entry = snapshot.codebases.get(&repo_key).cloned();

        if matches!(
            entry.as_ref().map(|value| value.status.as_str()),
            Some("indexed")
        ) && !self
            .inner
            .milvus
            .refresh_collection_presence(&collection_name)
            .await?
        {
            self.inner.snapshot.remove(&repo_key).await?;
            return Ok(RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                collection_name,
                status: "not_indexed".to_string(),
                indexed_files: None,
                total_chunks: None,
                index_status: None,
                indexing_percentage: None,
                last_attempted_percentage: None,
                error_message: None,
            });
        }

        Ok(match entry {
            Some(entry) => RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                collection_name,
                status: entry.status,
                indexed_files: entry.indexed_files,
                total_chunks: entry.total_chunks,
                index_status: entry.index_status,
                indexing_percentage: entry.indexing_percentage,
                last_attempted_percentage: entry.last_attempted_percentage,
                error_message: entry.error_message,
            },
            None => RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                collection_name,
                status: "not_indexed".to_string(),
                indexed_files: None,
                total_chunks: None,
                index_status: None,
                indexing_percentage: None,
                last_attempted_percentage: None,
                error_message: None,
            },
        })
    }

    async fn record_fingerprint(
        &self,
        repo_key: &str,
        fingerprint: &AuditFingerprint,
    ) -> Result<()> {
        self.inner
            .snapshot
            .update(|snapshot| {
                if let Some(entry) = snapshot.codebases.get_mut(repo_key) {
                    apply_fingerprint(entry, fingerprint);
                }
            })
            .await?;
        Ok(())
    }

    async fn record_indexing_progress(&self, repo_key: &str, progress: f64) -> Result<()> {
        self.inner
            .snapshot
            .update(|snapshot| {
                let entry = snapshot
                    .codebases
                    .entry(repo_key.to_string())
                    .or_insert_with(|| SnapshotEntry::indexing(progress));
                entry.set_indexing_progress(progress);
            })
            .await?;
        Ok(())
    }

    async fn write_file_symbols(
        &self,
        repo: &Path,
        repo_key: &str,
        relative_path: &str,
        symbols: &[IndexedSymbol],
    ) -> Result<()> {
        let symbol_docs = symbols
            .iter()
            .map(|symbol| SymbolIndexDoc {
                symbol_id: symbol.symbol_id.clone(),
                relative_path: symbol.relative_path.clone(),
                basename: basename_for_path(&symbol.relative_path),
                name: symbol.name.clone(),
                kind: symbol.kind.clone(),
                container: symbol.container.clone(),
                language: symbol.language.clone(),
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                indexed_at: symbol.indexed_at.clone(),
                file_hash: symbol.file_hash.clone(),
            })
            .collect::<Vec<_>>();
        let repo_path = repo.to_path_buf();
        let repo_key = repo_key.to_string();
        let relative_path = relative_path.to_string();
        let symbol_store = self.inner.symbol_store.clone();
        let local_index = self.inner.local_index.clone();
        let symbols = symbols.to_vec();
        run_low_priority_blocking("write_file_symbols", move || {
            symbol_store.replace_file_symbols(&repo_key, &relative_path, &symbols)?;
            local_index.replace_symbol_docs(&repo_path, &relative_path, &symbol_docs)?;
            Ok(())
        })
        .await
    }

    async fn process_files(
        &self,
        repo: &Path,
        repo_key: &str,
        collections: IndexCollections<'_>,
        files: &[RepoFile],
        splitter: SplitterKind,
        total_files: usize,
    ) -> Result<ProcessFilesResult> {
        let mut pending_chunks = Vec::new();
        let mut pending_symbols = Vec::new();
        let mut indexed_paths = HashSet::new();
        let mut processed_files = 0u64;
        let mut total_chunks = 0u64;
        let mut status = IndexCompletionStatus::Completed;
        let mut progress_tracker = ProgressTracker::new();

        for file in files {
            let Some(text) = read_utf8_file(&file.absolute_path).await? else {
                continue;
            };
            let relative_path = file
                .absolute_path
                .strip_prefix(repo)
                .unwrap_or(file.absolute_path.as_path())
                .display()
                .to_string()
                .replace('\\', "/");
            let basename = basename_for_path(&relative_path);
            let extension = file
                .absolute_path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| format!(".{value}"))
                .unwrap_or_default();
            let chunks = split_text(&file.absolute_path, &text, splitter)?;
            let chunk_count = chunks.len() as u64;
            if total_chunks + chunk_count > CHUNK_LIMIT as u64 {
                status = IndexCompletionStatus::LimitReached;
                break;
            }
            let indexed_at = crate::snapshot::timestamp();
            let symbols = extract_symbols(
                repo_key,
                &relative_path,
                &file.absolute_path,
                &text,
                &indexed_at,
                &file.hash,
            )?;
            self.write_file_symbols(repo, repo_key, &relative_path, &symbols)
                .await?;
            if collections.symbol.is_some() {
                pending_symbols.extend(symbols.iter().cloned().map(|symbol| {
                    PendingSymbolDocument {
                        symbol,
                        basename: basename.clone(),
                        extension: extension.clone(),
                    }
                }));
                if pending_symbols.len() >= EMBEDDING_BATCH_SIZE {
                    self.flush_symbol_documents(collections.symbol, &mut pending_symbols)
                        .await?;
                }
            }

            for chunk in chunks {
                pending_chunks.push(PendingChunk {
                    chunk,
                    indexed_at: indexed_at.clone(),
                    file_hash: file.hash.clone(),
                });
                if pending_chunks.len() >= EMBEDDING_BATCH_SIZE {
                    self.flush_chunks(repo, collections.chunk, &mut pending_chunks)
                        .await?;
                }
            }

            indexed_paths.insert(relative_path);
            processed_files += 1;
            total_chunks += chunk_count;
            if total_files > 0 {
                let progress = (processed_files as f64 / total_files as f64) * 100.0;
                if progress_tracker.should_persist(progress) {
                    self.record_indexing_progress(repo_key, progress).await?;
                }
            }
        }

        if !pending_chunks.is_empty() {
            self.flush_chunks(repo, collections.chunk, &mut pending_chunks)
                .await?;
        }
        if !pending_symbols.is_empty() {
            self.flush_symbol_documents(collections.symbol, &mut pending_symbols)
                .await?;
        }

        Ok(ProcessFilesResult {
            indexed_paths,
            processed_files,
            total_chunks,
            status,
        })
    }

    async fn flush_chunks(
        &self,
        repo: &Path,
        collection_name: &str,
        pending_chunks: &mut Vec<PendingChunk>,
    ) -> Result<()> {
        let chunks = std::mem::take(pending_chunks);
        if chunks.is_empty() {
            return Ok(());
        }

        let contents = chunks
            .iter()
            .map(|chunk| chunk.chunk.content.clone())
            .collect::<Vec<_>>();
        let embeddings = self.inner.embedding.embed_documents(&contents).await?;

        let documents = chunks
            .into_iter()
            .enumerate()
            .map(
                |(index, pending)| -> Result<(VectorDocument, ChunkIndexDoc)> {
                    let chunk = pending.chunk;
                    let relative_path = chunk
                        .file_path
                        .strip_prefix(repo)
                        .unwrap_or(chunk.file_path.as_path())
                        .display()
                        .to_string()
                        .replace('\\', "/");
                    let basename = basename_for_path(&relative_path);
                    let file_extension = chunk
                        .file_path
                        .extension()
                        .and_then(|value| value.to_str())
                        .map(|value| format!(".{value}"))
                        .unwrap_or_default();
                    let chunk_id = chunk_id(
                        &relative_path,
                        chunk.start_line,
                        chunk.end_line,
                        &chunk.content,
                    );
                    let vector_document = VectorDocument {
                        id: chunk_id.clone(),
                        content: chunk.content.clone(),
                        vector: embeddings
                            .get(index)
                            .cloned()
                            .context("embedding/vector batch mismatch")?,
                        relative_path: relative_path.clone(),
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        file_extension: file_extension.clone(),
                        metadata: json!({
                            "codebasePath": repo.display().to_string(),
                            "language": chunk.language,
                            "chunkIndex": index,
                            "indexedAt": pending.indexed_at,
                            "fileHash": pending.file_hash,
                            "basename": basename,
                        }),
                    };
                    let chunk_document = ChunkIndexDoc {
                        id: chunk_id,
                        relative_path,
                        basename,
                        extension: file_extension,
                        language: chunk.language,
                        content: chunk.content,
                        start_line: chunk.start_line,
                        end_line: chunk.end_line,
                        indexed_at: pending.indexed_at,
                        file_hash: pending.file_hash,
                    };
                    Ok((vector_document, chunk_document))
                },
            )
            .collect::<Result<Vec<_>>>()?;

        let (vector_documents, chunk_documents): (Vec<_>, Vec<_>) = documents.into_iter().unzip();

        self.inner
            .milvus
            .insert_documents(collection_name, &vector_documents)
            .await?;

        let repo_path = repo.to_path_buf();
        let local_index = self.inner.local_index.clone();
        run_low_priority_blocking("write_chunk_lexical_docs", move || {
            local_index.index_chunks(&repo_path, &chunk_documents)
        })
        .await
    }

    async fn flush_symbol_documents(
        &self,
        symbol_collection_name: Option<&str>,
        pending_symbols: &mut Vec<PendingSymbolDocument>,
    ) -> Result<()> {
        let Some(symbol_collection_name) = symbol_collection_name else {
            pending_symbols.clear();
            return Ok(());
        };

        let symbols = std::mem::take(pending_symbols);
        if symbols.is_empty() {
            return Ok(());
        }

        let contents = symbols
            .iter()
            .map(|pending| symbol_semantic_text(&pending.symbol))
            .collect::<Vec<_>>();
        let embeddings = self.inner.embedding.embed_documents(&contents).await?;

        let documents = symbols
            .into_iter()
            .enumerate()
            .map(|(index, pending)| -> Result<VectorDocument> {
                let symbol = pending.symbol;
                Ok(VectorDocument {
                    id: symbol.symbol_id.clone(),
                    content: symbol_semantic_text(&symbol),
                    vector: embeddings
                        .get(index)
                        .cloned()
                        .context("symbol embedding/vector batch mismatch")?,
                    relative_path: symbol.relative_path.clone(),
                    start_line: symbol.start_line,
                    end_line: symbol.end_line,
                    file_extension: pending.extension,
                    metadata: json!({
                        "symbolId": symbol.symbol_id,
                        "name": symbol.name,
                        "kind": symbol.kind,
                        "container": symbol.container,
                        "language": symbol.language,
                        "indexedAt": symbol.indexed_at,
                        "fileHash": symbol.file_hash,
                        "basename": pending.basename,
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        self.inner
            .milvus
            .insert_documents(symbol_collection_name, &documents)
            .await
    }
}

pub fn collection_name(repo: &Path) -> String {
    let digest = md5::compute(repo.display().to_string());
    format!("hybrid_code_chunks_{digest:x}")
        .chars()
        .take(27)
        .collect::<String>()
}

pub fn symbol_collection_name(repo: &Path) -> String {
    let digest = md5::compute(repo.display().to_string());
    format!("hybrid_symbols_{digest:x}")
}

pub fn chunk_id(relative_path: &str, start_line: u64, end_line: u64, content: &str) -> String {
    let digest = xxhash_rust::xxh3::xxh3_128(
        format!("{relative_path}:{start_line}:{end_line}:{content}").as_bytes(),
    );
    format!("chunk_{digest:032x}").chars().take(22).collect()
}

fn symbol_semantic_text(symbol: &IndexedSymbol) -> String {
    let container = symbol.container.as_deref().unwrap_or("root");
    format!(
        "{} {} in {} from {} ({})",
        symbol.kind, symbol.name, container, symbol.relative_path, symbol.language
    )
}

pub fn render_index_text(result: &ScopeIndexResult) -> String {
    let mut lines = vec![format!("Scope: {}", result.label)];
    for repo in &result.repos {
        match &repo.error {
            Some(error) => lines.push(format!("FAIL {}: {}", repo.repo, error)),
            None => lines.push(format!(
                "OK {} files={} chunks={} index_status={} full_reindex={} added={} modified={} removed={}",
                repo.repo,
                repo.indexed_files.unwrap_or(0),
                repo.total_chunks.unwrap_or(0),
                repo.index_status.as_deref().unwrap_or("unknown"),
                repo.full_reindex,
                repo.changes.added,
                repo.changes.modified,
                repo.changes.removed
            )),
        }
    }
    lines.join("\n")
}

pub fn render_clear_text(result: &ScopeClearResult) -> String {
    let mut lines = vec![format!("Scope: {}", result.label)];
    for repo in &result.repos {
        match &repo.error {
            Some(error) => lines.push(format!("FAIL {}: {}", repo.repo, error)),
            None => lines.push(format!("OK {}", repo.repo)),
        }
    }
    lines.join("\n")
}

pub fn render_search_text(result: &SearchResponse) -> String {
    let mut lines = vec![
        format!("Scope: {}", result.label),
        format!(
            "Plan: requested={} effective={} kind={} partial={}",
            result.plan.requested_mode,
            result.plan.effective_mode,
            result.plan.query_kind,
            result.partial
        ),
    ];

    if result.hits.is_empty() {
        lines.push("No results.".to_string());
        if !result.repo_errors.is_empty() {
            for error in &result.repo_errors {
                lines.push(format!("ERR {}: {}", error.repo, error.error));
            }
        }
        return lines.join("\n");
    }

    for (index, hit) in result.hits.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!(
            "{}. {} :: {}:{}-{} score={:.6} matchType={} stale={}",
            index + 1,
            hit.repo_label,
            hit.relative_path,
            hit.start_line,
            hit.end_line,
            hit.score,
            hit.match_type,
            hit.stale
        ));
        lines.push(hit.repo.clone());
        lines.push(truncate_for_display(&hit.content, 800));
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

pub fn render_symbol_search_text(result: &SymbolSearchResponse) -> String {
    let mut lines = vec![format!(
        "Scope: {} partial={}",
        result.label, result.partial
    )];
    if result.hits.is_empty() {
        lines.push("No symbols.".to_string());
    }
    for (index, hit) in result.hits.iter().enumerate() {
        lines.push(format!(
            "{}. {} :: {}:{}-{} {} {} score={:.6} lexical={} semantic={} stale={}",
            index + 1,
            hit.repo_label,
            hit.relative_path,
            hit.start_line,
            hit.end_line,
            hit.kind,
            hit.name,
            hit.score,
            hit.lexical_score.unwrap_or(0.0),
            hit.semantic_score.unwrap_or(0.0),
            hit.stale
        ));
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

pub fn render_outline_text(result: &FileOutlineResponse) -> String {
    let mut lines = vec![format!("Scope: {} file={}", result.label, result.file)];
    if result.matches.is_empty() {
        lines.push("No indexed outline.".to_string());
        return lines.join("\n");
    }
    for entry in &result.matches {
        lines.push(format!(
            "{} :: {} stale={} symbols={}",
            entry.repo_label,
            entry.relative_path,
            entry.stale,
            count_outline_nodes(&entry.symbols)
        ));
    }
    lines.join("\n")
}

pub fn render_search_explanation_text(result: &SearchExplanation) -> String {
    format!(
        "Scope: {}\nquery={}\nrequested={} effective={} kind={} denseWeight={:.2} lexicalWeight={:.2} symbolWeight={:.2} symbolLexicalShare={:.2} symbolSemanticShare={:.2} dedupeByFile={}",
        result.label,
        result.query,
        result.plan.requested_mode,
        result.plan.effective_mode,
        result.plan.query_kind,
        result.plan.dense_weight,
        result.plan.lexical_weight,
        result.plan.symbol_weight,
        result.plan.symbol_lexical_share,
        result.plan.symbol_semantic_share,
        result.plan.dedupe_by_file
    )
}

pub fn render_status_text(result: &StatusReport) -> String {
    let mut lines = vec![
        format!("Scope: {}", result.label),
        format!(
            "Status: {} files={} chunks={}",
            result.overall_status, result.indexed_files, result.total_chunks
        ),
    ];

    for repo in &result.repos {
        let error_suffix = repo
            .error_message
            .as_ref()
            .map(|message| format!(" error={message}"))
            .unwrap_or_default();
        let progress_suffix = repo
            .indexing_percentage
            .map(|progress| format!(" progress={progress:.1}%"))
            .or_else(|| {
                repo.last_attempted_percentage
                    .map(|progress| format!(" last_attempted={progress:.1}%"))
            })
            .unwrap_or_default();
        lines.push(format!(
            "{} status={} index_status={} files={} chunks={}{}{}",
            repo.repo,
            repo.status,
            repo.index_status.as_deref().unwrap_or("unknown"),
            repo.indexed_files.unwrap_or(0),
            repo.total_chunks.unwrap_or(0),
            progress_suffix,
            error_suffix
        ));
    }

    lines.join("\n")
}

fn validate_repo_path(repo: &Path) -> Result<()> {
    if !repo.is_absolute() {
        anyhow::bail!("repo path must be absolute: {}", repo.display());
    }
    if !repo.exists() {
        anyhow::bail!("repo path does not exist: {}", repo.display());
    }
    if !repo.is_dir() {
        anyhow::bail!("repo path is not a directory: {}", repo.display());
    }
    Ok(())
}

fn repo_basename(repo: &str) -> String {
    Path::new(repo)
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| repo.to_string())
}

fn overall_status(repos: &[RepoStatus]) -> String {
    if repos.iter().any(|repo| repo.status == "indexing") {
        return "indexing".to_string();
    }
    if repos.iter().any(|repo| repo.status == "indexfailed") {
        return "indexfailed".to_string();
    }
    if repos.iter().any(|repo| repo.status == "indexed") {
        return "indexed".to_string();
    }
    "not_indexed".to_string()
}

fn weighted_rrf(rank: usize, weight: f64) -> f64 {
    weight / (RRF_K + rank as f64 + 1.0)
}

fn fused_source_score(rank: usize, raw_score: f64, max_raw_score: f64, weight: f64) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let normalized_score = if max_raw_score > 0.0 {
        (raw_score / max_raw_score).clamp(0.0, 1.25)
    } else {
        0.0
    };
    weighted_rrf(rank, weight) + (normalized_score * weight * 0.35)
}

fn accumulate_dense_hits(
    merged: &mut HashMap<String, RepoSearchHit>,
    repo: &Path,
    hits: Vec<SearchDocument>,
    weight: f64,
) {
    if weight <= 0.0 {
        return;
    }
    let repo_key = repo.display().to_string();
    let max_score = hits
        .iter()
        .map(|hit| hit.score)
        .fold(0.0, f64::max)
        .max(1.0);
    for (rank, hit) in hits.into_iter().enumerate() {
        let key = format!("{repo_key}:{}", hit.id);
        let score = fused_source_score(rank, hit.score, max_score, weight);
        let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
            repo: repo_key.clone(),
            relative_path: hit.relative_path.clone(),
            start_line: hit.start_line,
            end_line: hit.end_line,
            language: hit
                .metadata
                .get("language")
                .and_then(Value::as_str)
                .unwrap_or_else(|| language_for_extension(&hit.file_extension))
                .to_string(),
            indexed_at: hit
                .metadata
                .get("indexedAt")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            file_hash: metadata_string(&hit.metadata, "fileHash"),
            content: hit.content.clone(),
            dense_score: 0.0,
            lexical_score: 0.0,
            symbol_score: 0.0,
            combined_score: 0.0,
        });
        entry.dense_score += score;
        entry.combined_score += score;
    }
}

fn accumulate_lexical_hits(
    merged: &mut HashMap<String, RepoSearchHit>,
    repo: &Path,
    hits: Vec<self::lexical::ChunkSearchHit>,
    weight: f64,
) {
    if weight <= 0.0 {
        return;
    }
    let repo_key = repo.display().to_string();
    let max_score = hits
        .iter()
        .map(|hit| hit.score)
        .fold(0.0, f64::max)
        .max(1.0);
    for (rank, hit) in hits.into_iter().enumerate() {
        let key = format!("{repo_key}:{}", hit.id);
        let score = fused_source_score(rank, hit.score, max_score, weight);
        let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
            repo: repo_key.clone(),
            relative_path: hit.relative_path.clone(),
            start_line: hit.start_line,
            end_line: hit.end_line,
            language: hit.language.clone(),
            indexed_at: Some(hit.indexed_at.clone()),
            file_hash: Some(hit.file_hash.clone()),
            content: hit.content.clone(),
            dense_score: 0.0,
            lexical_score: 0.0,
            symbol_score: 0.0,
            combined_score: 0.0,
        });
        entry.lexical_score += score;
        entry.combined_score += score;
    }
}

fn nearest_chunk_for_line(
    chunks: &[LexicalChunkSearchHit],
    line: u64,
) -> Option<LexicalChunkSearchHit> {
    if let Some(chunk) = chunks
        .iter()
        .find(|chunk| chunk.start_line <= line && chunk.end_line >= line)
    {
        return Some(chunk.clone());
    }

    chunks
        .iter()
        .min_by_key(|chunk| distance_to_line(chunk.start_line, chunk.end_line, line))
        .cloned()
}

fn accumulate_ranked_symbol_lexical_hits(
    merged: &mut HashMap<String, RankedSymbolHit>,
    authoritative: &HashMap<String, IndexedSymbol>,
    request: &SymbolSearchScopeRequest,
    hits: Vec<self::lexical::SymbolSearchHit>,
    weight: f64,
) {
    if weight <= 0.0 {
        return;
    }
    let max_score = hits
        .iter()
        .map(|hit| hit.score)
        .fold(0.0, f64::max)
        .max(1.0);
    for (rank, hit) in hits.into_iter().enumerate() {
        let Some(base) = ranked_symbol_hit_from_lexical(authoritative.get(&hit.symbol_id), hit)
        else {
            continue;
        };
        if !symbol_hit_matches_request(&base, request) {
            continue;
        }
        let score = fused_source_score(rank, base.lexical_score, max_score, weight);
        let entry = merged
            .entry(base.symbol_id.clone())
            .or_insert_with(|| RankedSymbolHit {
                lexical_score: 0.0,
                semantic_score: 0.0,
                combined_score: 0.0,
                ..base.clone()
            });
        entry.lexical_score += score;
        entry.combined_score += score;
    }
}

fn accumulate_ranked_symbol_semantic_hits(
    merged: &mut HashMap<String, RankedSymbolHit>,
    authoritative: &HashMap<String, IndexedSymbol>,
    request: &SymbolSearchScopeRequest,
    hits: Vec<SearchDocument>,
    weight: f64,
) {
    if weight <= 0.0 {
        return;
    }
    let max_score = hits
        .iter()
        .map(|hit| hit.score)
        .fold(0.0, f64::max)
        .max(1.0);
    for (rank, hit) in hits.into_iter().enumerate() {
        let Some(base) = ranked_symbol_hit_from_semantic(authoritative, hit) else {
            continue;
        };
        if !symbol_hit_matches_request(&base, request) {
            continue;
        }
        let score = fused_source_score(rank, base.semantic_score, max_score, weight);
        let entry = merged
            .entry(base.symbol_id.clone())
            .or_insert_with(|| RankedSymbolHit {
                lexical_score: 0.0,
                semantic_score: 0.0,
                combined_score: 0.0,
                ..base.clone()
            });
        entry.semantic_score += score;
        entry.combined_score += score;
    }
}

fn ranked_symbol_hit_from_lexical(
    authoritative: Option<&IndexedSymbol>,
    hit: self::lexical::SymbolSearchHit,
) -> Option<RankedSymbolHit> {
    if let Some(symbol) = authoritative {
        return Some(RankedSymbolHit {
            symbol_id: symbol.symbol_id.clone(),
            relative_path: symbol.relative_path.clone(),
            name: symbol.name.clone(),
            kind: symbol.kind.clone(),
            container: symbol.container.clone(),
            language: symbol.language.clone(),
            start_line: symbol.start_line,
            end_line: symbol.end_line,
            indexed_at: symbol.indexed_at.clone(),
            file_hash: symbol.file_hash.clone(),
            lexical_score: hit.score,
            semantic_score: 0.0,
            combined_score: 0.0,
        });
    }

    Some(RankedSymbolHit {
        symbol_id: hit.symbol_id,
        relative_path: hit.relative_path.clone(),
        name: hit.name,
        kind: hit.kind,
        container: hit.container,
        language: hit.language,
        start_line: hit.start_line,
        end_line: hit.end_line,
        indexed_at: hit.indexed_at,
        file_hash: hit.file_hash,
        lexical_score: hit.score,
        semantic_score: 0.0,
        combined_score: 0.0,
    })
}

fn ranked_symbol_hit_from_semantic(
    authoritative: &HashMap<String, IndexedSymbol>,
    hit: SearchDocument,
) -> Option<RankedSymbolHit> {
    let symbol_id = metadata_string(&hit.metadata, "symbolId")?;
    if let Some(symbol) = authoritative.get(&symbol_id) {
        return Some(RankedSymbolHit {
            symbol_id: symbol.symbol_id.clone(),
            relative_path: symbol.relative_path.clone(),
            name: symbol.name.clone(),
            kind: symbol.kind.clone(),
            container: symbol.container.clone(),
            language: symbol.language.clone(),
            start_line: symbol.start_line,
            end_line: symbol.end_line,
            indexed_at: symbol.indexed_at.clone(),
            file_hash: symbol.file_hash.clone(),
            lexical_score: 0.0,
            semantic_score: hit.score,
            combined_score: 0.0,
        });
    }

    Some(RankedSymbolHit {
        symbol_id,
        relative_path: hit.relative_path.clone(),
        name: metadata_string(&hit.metadata, "name")?,
        kind: metadata_string(&hit.metadata, "kind")?,
        container: metadata_string(&hit.metadata, "container"),
        language: metadata_string(&hit.metadata, "language")
            .unwrap_or_else(|| language_for_extension(&hit.file_extension).to_string()),
        start_line: hit.start_line,
        end_line: hit.end_line,
        indexed_at: metadata_string(&hit.metadata, "indexedAt")?,
        file_hash: metadata_string(&hit.metadata, "fileHash")?,
        lexical_score: 0.0,
        semantic_score: hit.score,
        combined_score: 0.0,
    })
}

fn symbol_hit_matches_request(hit: &RankedSymbolHit, request: &SymbolSearchScopeRequest) -> bool {
    if let Some(path_prefix) = &request.path_prefix
        && !normalize_relative_path(&hit.relative_path)
            .starts_with(&normalize_relative_path(path_prefix))
    {
        return false;
    }
    if let Some(language) = &request.language
        && hit.language != language.to_lowercase()
    {
        return false;
    }
    if let Some(kind) = &request.kind
        && hit.kind != kind.to_lowercase()
    {
        return false;
    }
    if let Some(container) = &request.container
        && !hit
            .container
            .as_deref()
            .unwrap_or_default()
            .to_lowercase()
            .contains(&container.to_lowercase())
    {
        return false;
    }
    true
}

#[derive(Debug, Clone, Copy)]
struct QuerySignals {
    word_count: usize,
    stopword_count: usize,
    path_like: bool,
    identifier_like: bool,
    exact_phrase: bool,
}

fn query_signals(query: &str) -> QuerySignals {
    let terms = query.split_whitespace().collect::<Vec<_>>();
    let stopword_count = terms
        .iter()
        .filter(|term| {
            matches!(
                term.to_ascii_lowercase().as_str(),
                "the" | "a" | "an" | "with" | "for" | "and" | "or" | "when" | "where" | "how"
            )
        })
        .count();
    let path_like = query.contains('/')
        || query.contains('\\')
        || query.contains("::")
        || query.split('/').next_back().is_some_and(|segment| {
            segment.contains('.')
                && !segment.starts_with('.')
                && segment.chars().any(|ch| ch.is_ascii_alphabetic())
        });
    let identifier_like = query.contains('_')
        || query.contains('(')
        || query.contains(')')
        || query.contains('<')
        || query.contains('>')
        || query.contains('#')
        || query.contains("->")
        || query.chars().any(|ch| ch.is_ascii_uppercase())
        || (terms.len() == 1
            && query
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '-')));
    let exact_phrase = query.starts_with('"') && query.ends_with('"') && query.len() > 2;

    QuerySignals {
        word_count: terms.len(),
        stopword_count,
        path_like,
        identifier_like,
        exact_phrase,
    }
}

fn symbol_query_shares(kind: QueryKind) -> (f64, f64) {
    match kind {
        QueryKind::NaturalLanguage => (0.35, 0.65),
        QueryKind::Identifier => (0.8, 0.2),
        QueryKind::Path => (0.9, 0.1),
        QueryKind::Mixed => (0.55, 0.45),
    }
}

fn normalize_symbol_shares(
    (mut lexical_share, mut semantic_share): (f64, f64),
    request: &SearchRequest,
) -> (f64, f64) {
    if request.path_prefix.is_some() || request.file.is_some() {
        lexical_share += 0.1;
        semantic_share = (semantic_share - 0.1).max(0.0);
    }
    let total = lexical_share + semantic_share;
    if total <= 0.0 {
        return (1.0, 0.0);
    }
    (lexical_share / total, semantic_share / total)
}

fn classify_query(query: &str) -> QueryKind {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return QueryKind::NaturalLanguage;
    }

    let signals = query_signals(trimmed);
    if signals.path_like {
        return if signals.word_count > 1
            && (signals.identifier_like || signals.stopword_count > 0 || signals.exact_phrase)
        {
            QueryKind::Mixed
        } else {
            QueryKind::Path
        };
    }
    if signals.identifier_like {
        return if signals.word_count > 2 && signals.stopword_count > 0 {
            QueryKind::Mixed
        } else {
            QueryKind::Identifier
        };
    }
    if signals.stopword_count > 0 && signals.word_count >= 3 {
        QueryKind::NaturalLanguage
    } else if signals.word_count > 1 {
        QueryKind::Mixed
    } else {
        QueryKind::Identifier
    }
}

fn query_flavor(kind: QueryKind) -> QueryFlavor {
    match kind {
        QueryKind::NaturalLanguage => QueryFlavor::NaturalLanguage,
        QueryKind::Identifier => QueryFlavor::Identifier,
        QueryKind::Path => QueryFlavor::Path,
        QueryKind::Mixed => QueryFlavor::Mixed,
    }
}

fn plan_search(request: &SearchRequest) -> SearchPlan {
    let query_kind = classify_query(&request.query);
    let effective_mode = match request.mode {
        SearchMode::Auto => match query_kind {
            QueryKind::NaturalLanguage => SearchMode::Semantic,
            QueryKind::Identifier => SearchMode::Identifier,
            QueryKind::Path => SearchMode::Path,
            QueryKind::Mixed => SearchMode::Hybrid,
        },
        mode => mode,
    };

    let (mut dense_weight, mut lexical_weight, mut symbol_weight, snippet_neighbor_chunks) =
        match effective_mode {
            SearchMode::Semantic => (2.0, 0.6, 0.6, 2),
            SearchMode::Hybrid => (1.3, 1.5, 1.2, 2),
            SearchMode::Identifier => (0.45, 2.35, 2.15, 1),
            SearchMode::Path => (0.2, 2.5, 1.1, 1),
            SearchMode::Auto => unreachable!("auto mode should be resolved before planning"),
        };

    if request.path_prefix.is_some() || request.file.is_some() {
        lexical_weight += 0.35;
        dense_weight = (dense_weight - 0.15_f64).max(0.15_f64);
    }
    if request.language.is_some() {
        lexical_weight += 0.1;
        symbol_weight += 0.1;
    }
    if request.file.is_some() {
        symbol_weight += 0.15;
    }

    let (symbol_lexical_share, symbol_semantic_share) =
        normalize_symbol_shares(symbol_query_shares(query_kind), request);

    SearchPlan {
        requested_mode: request.mode,
        effective_mode,
        query_kind,
        dense_weight,
        lexical_weight,
        symbol_weight,
        symbol_lexical_share,
        symbol_semantic_share,
        snippet_neighbor_chunks,
        dedupe_by_file: request.dedupe_by_file,
    }
}

fn search_document_matches(hit: &SearchDocument, request: &SearchRequest) -> bool {
    if let Some(file) = &request.file
        && normalize_relative_path(&hit.relative_path) != normalize_relative_path(file)
    {
        return false;
    }
    if let Some(path_prefix) = &request.path_prefix
        && !normalize_relative_path(&hit.relative_path)
            .starts_with(&normalize_relative_path(path_prefix))
    {
        return false;
    }
    if let Some(language) = &request.language {
        let hit_language = hit
            .metadata
            .get("language")
            .and_then(Value::as_str)
            .unwrap_or_else(|| language_for_extension(&hit.file_extension));
        if hit_language != language {
            return false;
        }
    }
    true
}

fn metadata_string(metadata: &Value, key: &str) -> Option<String> {
    metadata.get(key).and_then(|value| match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    })
}

fn match_type_for_hit(hit: &RepoSearchHit) -> String {
    let sources = [
        (hit.dense_score > 0.0, "semantic"),
        (hit.lexical_score > 0.0, "lexical"),
        (hit.symbol_score > 0.0, "symbol"),
    ]
    .into_iter()
    .filter_map(|(present, label)| present.then_some(label))
    .collect::<Vec<_>>();

    match sources.as_slice() {
        ["semantic"] => "semantic".to_string(),
        ["lexical"] => "lexical".to_string(),
        ["symbol"] => "symbol".to_string(),
        [] => "unknown".to_string(),
        _ => "hybrid".to_string(),
    }
}

fn dedupe_repo_search_hits_by_file(hits: Vec<(RepoSearchHit, bool)>) -> Vec<(RepoSearchHit, bool)> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for hit in hits {
        let key = format!("{}:{}", hit.0.repo, hit.0.relative_path);
        if seen.insert(key) {
            deduped.push(hit);
        }
    }
    deduped
}

fn build_snippet(query: &str, content: &str, max_chars: usize) -> String {
    if content.chars().count() <= max_chars {
        return content.to_string();
    }
    let lowered_content = content.to_lowercase();
    let terms = query
        .split_whitespace()
        .map(|term| term.trim().to_lowercase())
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    if let Some(position) = terms.iter().find_map(|term| lowered_content.find(term)) {
        let start = position.saturating_sub(max_chars / 3);
        let end = (position + (max_chars * 2 / 3)).min(content.len());
        let mut snippet = content
            .get(start..end)
            .map(ToString::to_string)
            .unwrap_or_else(|| content.chars().take(max_chars).collect());
        if start > 0 {
            snippet = format!("...{snippet}");
        }
        if end < content.len() {
            snippet.push_str("...");
        }
        return snippet;
    }
    truncate_for_display(content, max_chars)
}

fn build_chunk_context_snippet(
    chunks: &[self::lexical::ChunkSearchHit],
    target_line: u64,
    neighbor_chunks: usize,
    max_chars: usize,
) -> Option<String> {
    if chunks.is_empty() {
        return None;
    }

    let target_index = chunks
        .iter()
        .position(|chunk| chunk.start_line <= target_line && chunk.end_line >= target_line)
        .unwrap_or_else(|| {
            chunks
                .iter()
                .enumerate()
                .min_by_key(|(_, chunk)| {
                    distance_to_line(chunk.start_line, chunk.end_line, target_line)
                })
                .map(|(index, _)| index)
                .unwrap_or(0)
        });
    let start = target_index.saturating_sub(neighbor_chunks);
    let end = (target_index + neighbor_chunks + 1).min(chunks.len());

    let mut combined = String::new();
    for (index, chunk) in chunks[start..end].iter().enumerate() {
        if index > 0 {
            combined.push_str("\n\n");
        }
        combined.push_str(&chunk.content);
    }

    Some(truncate_for_display(&combined, max_chars))
}

fn file_freshness_key(repo: &Path, relative_path: &str) -> String {
    format!(
        "{}:{}",
        repo.display(),
        normalize_relative_path(relative_path)
    )
}

fn distance_to_line(start_line: u64, end_line: u64, line: u64) -> u64 {
    if (start_line..=end_line).contains(&line) {
        0
    } else if line < start_line {
        start_line.saturating_sub(line)
    } else {
        line.saturating_sub(end_line)
    }
}

fn basename_for_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn normalize_relative_path(path: &str) -> String {
    path.replace('\\', "/").trim_matches('/').to_string()
}

fn index_identity_status_for_snapshot(
    snapshot: &Snapshot,
    configured_embedding_fingerprint: &str,
) -> IndexIdentityStatus {
    let stored_embedding_fingerprint = snapshot.embedding_fingerprint.clone();
    let mut reason = None;

    if snapshot.index_format_version != INDEX_FORMAT_VERSION {
        reason = Some(format!(
            "index format version mismatch: local state is `{}`, expected `{}`.",
            snapshot.index_format_version, INDEX_FORMAT_VERSION
        ));
    } else if snapshot.search_root_version != SEARCH_ROOT_VERSION {
        reason = Some(format!(
            "search root version mismatch: local state is `{}`, expected `{}`.",
            snapshot.search_root_version, SEARCH_ROOT_VERSION
        ));
    } else if let Some(stored) = stored_embedding_fingerprint.as_deref()
        && stored != configured_embedding_fingerprint
    {
        reason = Some(format!(
            "embedding fingerprint mismatch: local state is `{stored}`, current config is `{configured_embedding_fingerprint}`."
        ));
    }

    IndexIdentityStatus {
        compatible: reason.is_none(),
        index_format_version: snapshot.index_format_version.clone(),
        search_root_version: snapshot.search_root_version.clone(),
        configured_embedding_fingerprint: configured_embedding_fingerprint.to_string(),
        stored_embedding_fingerprint,
        reason,
    }
}

fn count_outline_nodes(nodes: &[OutlineNode]) -> usize {
    nodes
        .iter()
        .map(|node| 1 + count_outline_nodes(&node.children))
        .sum()
}

fn truncate_for_display(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut truncated = text.chars().take(max_chars).collect::<String>();
    truncated.push_str("\n...");
    truncated
}

async fn run_low_priority_blocking<T, F>(label: &'static str, work: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(move || {
        lower_current_thread_priority();
        work()
    })
    .await
    .with_context(|| format!("joining blocking task `{label}`"))?
}

fn lower_current_thread_priority() {
    LOW_PRIORITY_BLOCKING_THREAD.with(|state| {
        if state.get() {
            return;
        }

        #[cfg(target_os = "macos")]
        unsafe {
            let _ = libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_BACKGROUND, 0);
        }

        #[cfg(all(unix, not(target_os = "macos")))]
        unsafe {
            let _ = libc::nice(10);
        }

        state.set(true);
    });
}

fn build_extension_filter(extension_filter: &[String]) -> Option<String> {
    if extension_filter.is_empty() {
        return None;
    }

    let quoted = extension_filter
        .iter()
        .map(|extension| {
            format!(
                "\"{}\"",
                extension.replace('\\', "\\\\").replace('"', "\\\"")
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!("fileExtension in [{quoted}]"))
}

fn scan_repo(
    repo: &Path,
    custom_extensions: &[String],
    ignore_patterns: &[String],
) -> Result<BTreeMap<String, RepoFile>> {
    let mut supported_extensions = default_supported_extensions();
    for extension in custom_extensions {
        let normalized = if extension.starts_with('.') {
            extension.clone()
        } else {
            format!(".{extension}")
        };
        supported_extensions.insert(normalized);
    }

    let all_ignore_patterns = collect_ignore_patterns(repo, ignore_patterns)?;
    let ignore_set = build_ignore_set(&all_ignore_patterns)?;
    let mut files = BTreeMap::new();

    let mut builder = WalkBuilder::new(repo);
    builder.hidden(true);
    builder.git_ignore(true);
    builder.git_exclude(true);
    builder.git_global(true);
    builder.follow_links(false);
    builder.require_git(false);

    for entry in builder.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => continue,
        };
        let file_type = match entry.file_type() {
            Some(file_type) if file_type.is_file() => file_type,
            _ => continue,
        };
        let _ = file_type;

        let absolute_path = entry.path().to_path_buf();
        let relative_path = absolute_path
            .strip_prefix(repo)
            .unwrap_or(entry.path())
            .display()
            .to_string()
            .replace('\\', "/");
        if ignore_set.is_match(&relative_path) {
            continue;
        }

        let extension = absolute_path
            .extension()
            .and_then(|value| value.to_str())
            .map(|value| format!(".{value}"))
            .unwrap_or_default();
        if !supported_extensions.contains(&extension) {
            continue;
        }

        let Some(hash) = hash_text_like_file(&absolute_path)? else {
            continue;
        };

        files.insert(
            relative_path.clone(),
            RepoFile {
                absolute_path,
                hash,
            },
        );
    }

    Ok(files)
}

fn build_ignore_set(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(
            Glob::new(pattern).with_context(|| format!("invalid ignore pattern `{pattern}`"))?,
        );
    }
    builder.build().context("building ignore matcher")
}

fn collect_ignore_patterns(repo: &Path, explicit_patterns: &[String]) -> Result<Vec<String>> {
    let mut patterns = explicit_patterns.to_vec();

    if let Ok(entries) = std::fs::read_dir(repo) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
                continue;
            };
            if name.starts_with('.') && name.ends_with("ignore") {
                patterns.extend(read_ignore_file(&path)?);
            }
        }
    }

    let global_ignore = shellexpand::tilde("~/.context/.contextignore").into_owned();
    let global_ignore_path = PathBuf::from(global_ignore);
    if global_ignore_path.exists() {
        patterns.extend(read_ignore_file(&global_ignore_path)?);
    }

    Ok(patterns)
}

fn read_ignore_file(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading ignore file {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToString::to_string)
        .collect())
}

fn hash_text_like_file(path: &Path) -> Result<Option<String>> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };
    let mut reader = std::io::BufReader::new(file);
    let mut sample_buffer = [0u8; 8 * 1024];
    let mut buffer = [0u8; 16 * 1024];
    let mut hasher = Xxh3::new();

    let sample_len = reader
        .read(&mut sample_buffer)
        .with_context(|| format!("reading {}", path.display()))?;
    let sample = &sample_buffer[..sample_len];
    if looks_like_lfs_pointer(sample) || looks_binary(sample) {
        return Ok(None);
    }
    hasher.update(sample);

    loop {
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(Some(format!("{:032x}", hasher.digest128())))
}

fn looks_like_lfs_pointer(sample: &[u8]) -> bool {
    let Ok(text) = std::str::from_utf8(sample) else {
        return false;
    };
    let text = text.trim_start_matches('\u{feff}');
    text.starts_with("version https://git-lfs.github.com/spec/v1")
        && text.contains("\noid sha256:")
        && text.contains("\nsize ")
}

fn looks_binary(sample: &[u8]) -> bool {
    if sample.is_empty() {
        return false;
    }
    if sample.contains(&0) {
        return true;
    }
    std::str::from_utf8(sample).is_err()
}

async fn read_utf8_file(path: &Path) -> Result<Option<String>> {
    let bytes = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(String::from_utf8(bytes).ok())
}

async fn load_merkle_snapshot(path: &Path) -> Result<MerkleSnapshot> {
    let text = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).context("parsing merkle snapshot")
}

async fn save_merkle_snapshot(path: &Path, file_hashes: &BTreeMap<String, String>) -> Result<()> {
    let merkle_dir = path
        .parent()
        .context("merkle snapshot missing parent directory")?;
    tokio::fs::create_dir_all(merkle_dir)
        .await
        .with_context(|| format!("creating {}", merkle_dir.display()))?;

    let serialized = MerkleSnapshot {
        file_hashes: file_hashes
            .iter()
            .map(|(path, hash)| (path.clone(), hash.clone()))
            .collect(),
        hash_algorithm: CONTENT_HASH_ALGORITHM.to_string(),
        root_hash: Some(build_root_hash(file_hashes)),
        merkle_dag: None,
    };
    let text = serde_json::to_string(&serialized).context("serializing merkle snapshot")?;
    tokio::fs::write(path, text)
        .await
        .with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

fn build_root_hash(file_hashes: &BTreeMap<String, String>) -> String {
    let mut hasher = Xxh3::new();
    for (path, hash) in file_hashes {
        hasher.update(path.as_bytes());
        hasher.update(b":");
        hasher.update(hash.as_bytes());
        hasher.update(b"\n");
    }
    format!("{:032x}", hasher.digest128())
}

fn default_merkle_hash_algorithm() -> String {
    LEGACY_CONTENT_HASH_ALGORITHM.to_string()
}

impl MerkleSnapshot {
    fn is_compatible(&self) -> bool {
        self.hash_algorithm == CONTENT_HASH_ALGORITHM
    }
}

fn diff_files(previous: &BTreeMap<String, String>, current: &BTreeMap<String, String>) -> FileDiff {
    let previous_keys = previous.keys().cloned().collect::<HashSet<_>>();
    let current_keys = current.keys().cloned().collect::<HashSet<_>>();

    let mut diff = FileDiff::default();
    for key in current_keys.difference(&previous_keys) {
        diff.added.push(key.clone());
    }
    for key in previous_keys.difference(&current_keys) {
        diff.removed.push(key.clone());
    }
    for key in current_keys.intersection(&previous_keys) {
        if previous.get(key) != current.get(key) {
            diff.modified.push(key.clone());
        }
    }
    diff.added.sort();
    diff.modified.sort();
    diff.removed.sort();
    diff
}

#[cfg(test)]
mod tests {
    use super::{
        CONTENT_HASH_ALGORITHM, LEGACY_CONTENT_HASH_ALGORITHM, MerkleSnapshot, SearchBudgets,
        SearchMode, SearchRequest, build_chunk_context_snippet, build_root_hash, chunk_id,
        classify_query, collection_name, diff_files, hash_text_like_file, plan_search,
        run_low_priority_blocking, scan_repo,
    };
    use crate::config::SearchConfig;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};
    use tokio::sync::Barrier;
    use xxhash_rust::xxh3::xxh3_128;

    fn temp_file_path(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time went backwards")
            .as_nanos();
        std::env::temp_dir().join(format!("agent-context-{name}-{nanos}"))
    }

    #[test]
    fn collection_name_matches_upstream_prefix_and_hash() {
        let collection = collection_name(Path::new("/tmp/example"));
        assert_eq!(collection, "hybrid_code_chunks_89a35363");
    }

    #[test]
    fn chunk_ids_use_xxh3_shape() {
        let value = chunk_id("src/lib.rs", 10, 20, "fn example() {}");
        let expected = format!(
            "chunk_{:032x}",
            xxh3_128("src/lib.rs:10:20:fn example() {}".as_bytes())
        )
        .chars()
        .take(22)
        .collect::<String>();

        assert_eq!(value, expected);
        assert_eq!(value.len(), 22);
    }

    #[test]
    fn diff_detects_added_removed_and_modified_files() {
        let previous = BTreeMap::from([
            ("a.rs".to_string(), "1".to_string()),
            ("b.rs".to_string(), "2".to_string()),
        ]);
        let current = BTreeMap::from([
            ("a.rs".to_string(), "1".to_string()),
            ("b.rs".to_string(), "3".to_string()),
            ("c.rs".to_string(), "4".to_string()),
        ]);

        let diff = diff_files(&previous, &current);
        assert_eq!(diff.added, vec!["c.rs"]);
        assert_eq!(diff.modified, vec!["b.rs"]);
        assert!(diff.removed.is_empty());
    }

    #[test]
    fn hash_skips_git_lfs_pointer_files() {
        let path = temp_file_path("lfs-pointer.rs");
        fs::write(
            &path,
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc123\nsize 42\n",
        )
        .unwrap();

        let result = hash_text_like_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert!(result.is_none());
    }

    #[test]
    fn hash_skips_binary_files() {
        let path = temp_file_path("binary.rs");
        fs::write(&path, b"fn main() {}\0\xff").unwrap();

        let result = hash_text_like_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        assert!(result.is_none());
    }

    #[test]
    fn hash_keeps_text_files() {
        let path = temp_file_path("text.rs");
        let content = "fn main() {\n    println!(\"hello\");\n}\n";
        fs::write(&path, content).unwrap();

        let result = hash_text_like_file(&path).unwrap();
        let _ = fs::remove_file(&path);

        let expected = format!("{:032x}", xxh3_128(content.as_bytes()));
        assert_eq!(result.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn merkle_snapshot_defaults_to_legacy_algorithm_when_missing() {
        let snapshot: MerkleSnapshot =
            serde_json::from_str(r#"{"fileHashes":[["src/lib.rs","abc"]],"rootHash":"deadbeef"}"#)
                .unwrap();

        assert_eq!(snapshot.hash_algorithm, LEGACY_CONTENT_HASH_ALGORITHM);
        assert!(!snapshot.is_compatible());
    }

    #[test]
    fn merkle_snapshot_root_hash_uses_xxh3_128() {
        let file_hashes = BTreeMap::from([
            ("a.rs".to_string(), "111".to_string()),
            ("b.rs".to_string(), "222".to_string()),
        ]);

        let expected_input = "a.rs:111\nb.rs:222\n";
        assert_eq!(
            build_root_hash(&file_hashes),
            format!("{:032x}", xxh3_128(expected_input.as_bytes()))
        );
    }

    #[test]
    fn merkle_snapshot_compatibility_matches_current_algorithm() {
        let snapshot = MerkleSnapshot {
            file_hashes: Vec::new(),
            hash_algorithm: CONTENT_HASH_ALGORITHM.to_string(),
            root_hash: None,
            merkle_dag: None,
        };

        assert!(snapshot.is_compatible());
    }

    #[test]
    fn classify_query_distinguishes_natural_language_identifier_and_path() {
        assert!(matches!(
            classify_query("where is the graphql resolver wired up"),
            super::QueryKind::NaturalLanguage
        ));
        assert!(matches!(
            classify_query("GraphQLResolver"),
            super::QueryKind::Identifier
        ));
        assert!(matches!(
            classify_query("server/crates/api/src/schema.rs"),
            super::QueryKind::Path
        ));
    }

    #[test]
    fn plan_search_biases_file_constrained_queries_toward_lexical() {
        let unconstrained = plan_search(&SearchRequest {
            query: "where is the resolver".to_string(),
            limit: 10,
            mode: SearchMode::Auto,
            extension_filter: Vec::new(),
            path_prefix: None,
            language: None,
            file: None,
            dedupe_by_file: true,
        });
        let constrained = plan_search(&SearchRequest {
            query: "where is the resolver".to_string(),
            limit: 10,
            mode: SearchMode::Auto,
            extension_filter: Vec::new(),
            path_prefix: Some("server/crates/api".to_string()),
            language: None,
            file: Some("server/crates/api/src/schema.rs".to_string()),
            dedupe_by_file: true,
        });

        assert!(constrained.lexical_weight > unconstrained.lexical_weight);
        assert!(constrained.dense_weight < unconstrained.dense_weight);
        assert!(constrained.symbol_lexical_share > unconstrained.symbol_lexical_share);
    }

    #[test]
    fn chunk_context_snippet_includes_neighboring_chunks() {
        let chunks = vec![
            super::lexical::ChunkSearchHit {
                id: "chunk_a".to_string(),
                relative_path: "src/lib.rs".to_string(),
                basename: "lib.rs".to_string(),
                extension: ".rs".to_string(),
                language: "rust".to_string(),
                content: "fn alpha() {}".to_string(),
                start_line: 1,
                end_line: 3,
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "hash".to_string(),
                score: 1.0,
            },
            super::lexical::ChunkSearchHit {
                id: "chunk_b".to_string(),
                relative_path: "src/lib.rs".to_string(),
                basename: "lib.rs".to_string(),
                extension: ".rs".to_string(),
                language: "rust".to_string(),
                content: "fn beta() {}".to_string(),
                start_line: 4,
                end_line: 6,
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "hash".to_string(),
                score: 0.9,
            },
            super::lexical::ChunkSearchHit {
                id: "chunk_c".to_string(),
                relative_path: "src/lib.rs".to_string(),
                basename: "lib.rs".to_string(),
                extension: ".rs".to_string(),
                language: "rust".to_string(),
                content: "fn gamma() {}".to_string(),
                start_line: 7,
                end_line: 9,
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "hash".to_string(),
                score: 0.8,
            },
        ];

        let snippet = build_chunk_context_snippet(&chunks, 5, 1, 200).unwrap();

        assert!(snippet.contains("alpha"));
        assert!(snippet.contains("beta"));
        assert!(snippet.contains("gamma"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_requests_share_repo_search_budget() {
        let budgets = SearchBudgets::new(&SearchConfig {
            max_concurrent_requests: 2,
            max_concurrent_repo_searches: 1,
            max_concurrent_lexical_tasks: 1,
            max_concurrent_dense_tasks: 1,
            max_warm_repos: 1,
        });
        let barrier = Arc::new(Barrier::new(2));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let handles = (0..2)
            .map(|_| {
                let budgets = budgets.clone();
                let barrier = barrier.clone();
                let active = active.clone();
                let max_active = max_active.clone();
                tokio::spawn(async move {
                    let _request = budgets.acquire_request().await.unwrap();
                    barrier.wait().await;
                    let _repo = budgets.acquire_repo_search().await.unwrap();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lexical_tasks_never_exceed_global_limit() {
        let budgets = SearchBudgets::new(&SearchConfig {
            max_concurrent_requests: 3,
            max_concurrent_repo_searches: 3,
            max_concurrent_lexical_tasks: 1,
            max_concurrent_dense_tasks: 2,
            max_warm_repos: 2,
        });
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let handles = (0..3)
            .map(|_| {
                let budgets = budgets.clone();
                let active = active.clone();
                let max_active = max_active.clone();
                tokio::spawn(async move {
                    let _lexical = budgets.acquire_lexical().await.unwrap();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dense_tasks_never_exceed_global_limit() {
        let budgets = SearchBudgets::new(&SearchConfig {
            max_concurrent_requests: 4,
            max_concurrent_repo_searches: 4,
            max_concurrent_lexical_tasks: 2,
            max_concurrent_dense_tasks: 2,
            max_warm_repos: 2,
        });
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        let handles = (0..4)
            .map(|_| {
                let budgets = budgets.clone();
                let active = active.clone();
                let max_active = max_active.clone();
                tokio::spawn(async move {
                    let _dense = budgets.acquire_dense().await.unwrap();
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    max_active.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(30)).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect::<Vec<_>>();

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[test]
    #[ignore = "manual benchmark"]
    fn bench_scan_repo_manual() {
        let repo = std::env::var("CC_INDEXER_BENCH_REPO")
            .expect("set CC_INDEXER_BENCH_REPO to an absolute repo path");
        let path = std::path::PathBuf::from(repo);
        let output_path = std::env::var("CC_INDEXER_BENCH_OUTPUT")
            .unwrap_or_else(|_| "/tmp/agent-context-bench-scan.txt".to_string());

        let runtime = tokio::runtime::Runtime::new().expect("create runtime");
        let mut runs: Vec<(usize, u64, std::time::Duration)> = Vec::new();

        for _ in 0..2 {
            let repo_path = path.clone();
            let started = std::time::Instant::now();
            let files = runtime
                .block_on(run_low_priority_blocking("bench_scan_repo", move || {
                    scan_repo(&repo_path, &[], &[])
                }))
                .expect("scan_repo should succeed");
            let elapsed = started.elapsed();
            let total_bytes = files
                .values()
                .filter_map(|file| {
                    fs::metadata(&file.absolute_path)
                        .ok()
                        .map(|meta| meta.len())
                })
                .sum::<u64>();
            runs.push((files.len(), total_bytes, elapsed));
        }

        let mut lines = Vec::new();
        for (index, (file_count, total_bytes, elapsed)) in runs.into_iter().enumerate() {
            lines.push(format!(
                "bench_scan_repo run={} repo={} files={} bytes={} elapsed_ms={}",
                index + 1,
                path.display(),
                file_count,
                total_bytes,
                elapsed.as_millis()
            ));
        }
        fs::write(&output_path, lines.join("\n") + "\n").expect("write benchmark output");
    }
}

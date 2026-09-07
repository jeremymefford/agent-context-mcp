use crate::config::{Config, ResolvedScope};
use crate::engine::changes::{AnalyzeChangesRequest, ChangeAnalysisResponse};
use crate::engine::impact::{
    ImpactEvidence, ImpactNode, ImpactRequest, ImpactResponse, TracePathRequest, TracePathResponse,
};
use crate::engine::relationships::RepoRelationshipCoverage;
use crate::engine::splitter::SplitterKind;
use crate::engine::symbols::OutlineNode;
use crate::engine::trim::{
    DeadCodeAnalysis, DeadCodeClassification, DeadCodeGroup, DeadCodeRequest, EstimateTrimRequest,
    ExplainRemovalRequest, RemovalExplanation, RemovalValidation, TrimEstimate,
    ValidateRemovalRequest,
};
use crate::engine::{
    EditTargetAnchor, EditTargetReasonCode, EditTargetStatus, Engine, FileOutlineResponse,
    PREPARE_READY_MAX_LINES, PrepareEditTargetRequest, PrepareEditTargetResponse, RepoSearchError,
    SearchHit, SearchMode, SearchPlanSummary, SearchRequest, SearchResponse, SymbolSearchHit,
    SymbolSearchResponse, SymbolSearchScopeRequest, TextSearchHit, TextSearchResponse,
    TextSearchScopeRequest, render_clear_text, render_search_explanation_text, render_status_text,
};
use crate::snapshot::atomic_replace;
use anyhow::{Context, Result, bail};
use axum::{
    Router,
    body::Bytes,
    extract::Request,
    http::StatusCode,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use futures::FutureExt;
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, Content, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities, ServerInfo, Tool, ToolAnnotations,
    },
    service::RequestContext,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, session::local::LocalSessionManager,
        tower::StreamableHttpService,
    },
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::Mutex;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

const INDEX_WORKER_STALE_AFTER: Duration = Duration::from_secs(300);
const INDEX_WORKER_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const BACKGROUND_INDEX_TRANSIENT_RETRIES: u8 = 2;
const MCP_SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);
const SLOW_HTTP_REQUEST: Duration = Duration::from_secs(2);

#[derive(Clone)]
struct NativeServer {
    engine: Engine,
    index_coordinator: Arc<Mutex<IndexCoordinatorState>>,
    index_queue: Arc<IndexQueueStore>,
}

#[derive(Clone, Default)]
struct IndexCoordinatorState {
    pending: BTreeMap<String, PendingIndexRequest>,
    running: BTreeMap<String, u64>,
    active_requests: BTreeMap<String, PendingIndexRequest>,
    recovering: BTreeSet<String>,
    worker: Option<IndexWorkerState>,
    next_worker_generation: u64,
}

#[derive(Clone, Copy)]
struct IndexWorkerState {
    generation: u64,
    started_at: Instant,
    last_heartbeat: Instant,
}

struct WorkerRecovery {
    running_repos: Vec<String>,
    age_secs: u64,
    failure_recorded: bool,
}

impl IndexCoordinatorState {
    fn live_worker_generation(&self, now: Instant) -> Option<u64> {
        self.worker
            .filter(|worker| now.duration_since(worker.last_heartbeat) <= INDEX_WORKER_STALE_AFTER)
            .map(|worker| worker.generation)
    }

    fn begin_stale_worker_recovery(&mut self, now: Instant) -> Option<WorkerRecovery> {
        let worker = self.worker?;
        if now.duration_since(worker.last_heartbeat) <= INDEX_WORKER_STALE_AFTER {
            return None;
        }
        let age_secs = now.duration_since(worker.started_at).as_secs();
        self.worker = None;
        let running_repos = std::mem::take(&mut self.running)
            .into_keys()
            .collect::<Vec<_>>();
        self.recovering.extend(running_repos.iter().cloned());
        Some(WorkerRecovery {
            running_repos,
            age_secs,
            failure_recorded: false,
        })
    }

    fn recovering_repos(&self) -> Vec<String> {
        self.recovering.iter().cloned().collect()
    }

    fn begin_worker_abort_recovery(&mut self, generation: u64) -> Option<Vec<String>> {
        if self.worker.map(|worker| worker.generation) != Some(generation) {
            return None;
        }
        let running_repos = std::mem::take(&mut self.running)
            .into_keys()
            .collect::<Vec<_>>();
        for repo in &running_repos {
            if let Some(mut request) = self.active_requests.remove(repo) {
                request.snapshot_queued = false;
                self.pending.insert(repo.clone(), request);
            }
        }
        self.recovering.extend(running_repos.iter().cloned());
        self.finish_worker(generation);
        Some(running_repos)
    }

    fn finish_recovery(&mut self, repos: &[String]) {
        for repo in repos {
            self.recovering.remove(repo);
        }
    }

    fn ready_pending_exists(&self) -> bool {
        self.pending
            .iter()
            .any(|(repo, request)| request.snapshot_queued && !self.recovering.contains(repo))
    }

    fn unqueued_ready_pending_repos(&self) -> Vec<String> {
        self.pending
            .iter()
            .filter(|(repo, request)| !request.snapshot_queued && !self.recovering.contains(*repo))
            .map(|(repo, _)| repo.clone())
            .collect()
    }

    fn mark_pending_snapshot_queued(&mut self, repos: &[String]) {
        for repo in repos {
            if let Some(request) = self.pending.get_mut(repo) {
                request.snapshot_queued = true;
            }
        }
    }

    fn remove_unqueued_pending(&mut self, repos: &[String]) {
        for repo in repos {
            if self
                .pending
                .get(repo)
                .is_some_and(|request| !request.snapshot_queued)
            {
                self.pending.remove(repo);
            }
        }
    }

    fn ensure_live_worker_for_pending(&mut self, now: Instant) -> Option<u64> {
        if !self.ready_pending_exists() || self.live_worker_generation(now).is_some() {
            return None;
        }
        Some(self.start_worker(now))
    }

    fn start_worker(&mut self, now: Instant) -> u64 {
        self.next_worker_generation = self.next_worker_generation.saturating_add(1);
        let generation = self.next_worker_generation;
        self.worker = Some(IndexWorkerState {
            generation,
            started_at: now,
            last_heartbeat: now,
        });
        generation
    }

    fn heartbeat_worker(&mut self, generation: u64, now: Instant) -> bool {
        if let Some(worker) = self.worker.as_mut()
            && worker.generation == generation
        {
            worker.last_heartbeat = now;
            return true;
        }
        false
    }

    fn finish_worker(&mut self, generation: u64) -> bool {
        if self.worker.map(|worker| worker.generation) == Some(generation) {
            self.worker = None;
            return true;
        }
        false
    }

    fn persisted_queue(&self) -> PersistedIndexQueue {
        PersistedIndexQueue {
            pending: self
                .pending
                .iter()
                .map(|(repo, request)| (repo.clone(), PersistedIndexRequest::from(request)))
                .collect(),
            running: self
                .active_requests
                .iter()
                .map(|(repo, request)| (repo.clone(), PersistedIndexRequest::from(request)))
                .collect(),
        }
    }
}

#[derive(Clone)]
struct PendingIndexRequest {
    repo: PathBuf,
    force: bool,
    explicit_refresh: bool,
    splitter: SplitterKind,
    custom_extensions: Vec<String>,
    ignore_patterns: Vec<String>,
    snapshot_queued: bool,
    retry_attempt: u8,
}

#[derive(Clone, Default, Serialize, Deserialize)]
struct PersistedIndexQueue {
    #[serde(default)]
    pending: BTreeMap<String, PersistedIndexRequest>,
    #[serde(default)]
    running: BTreeMap<String, PersistedIndexRequest>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PersistedIndexRequest {
    repo: PathBuf,
    force: bool,
    explicit_refresh: bool,
    splitter: String,
    #[serde(default)]
    custom_extensions: Vec<String>,
    #[serde(default)]
    ignore_patterns: Vec<String>,
    #[serde(default)]
    retry_attempt: u8,
}

impl From<&PendingIndexRequest> for PersistedIndexRequest {
    fn from(request: &PendingIndexRequest) -> Self {
        Self {
            repo: request.repo.clone(),
            force: request.force,
            explicit_refresh: request.explicit_refresh,
            splitter: match request.splitter {
                SplitterKind::Ast => "ast",
                SplitterKind::LangChain => "langchain",
            }
            .to_string(),
            custom_extensions: request.custom_extensions.clone(),
            ignore_patterns: request.ignore_patterns.clone(),
            retry_attempt: request.retry_attempt,
        }
    }
}

impl PersistedIndexRequest {
    fn into_pending(self) -> Result<PendingIndexRequest> {
        let splitter = match self.splitter.as_str() {
            "ast" => SplitterKind::Ast,
            "langchain" => SplitterKind::LangChain,
            other => bail!("unknown persisted splitter `{other}`"),
        };
        Ok(PendingIndexRequest {
            repo: self.repo,
            force: self.force,
            explicit_refresh: self.explicit_refresh,
            splitter,
            custom_extensions: self.custom_extensions,
            ignore_patterns: self.ignore_patterns,
            snapshot_queued: false,
            retry_attempt: self.retry_attempt,
        })
    }
}

#[derive(Debug)]
struct IndexQueueStore {
    path: PathBuf,
    lock: Mutex<()>,
}

impl IndexQueueStore {
    fn new(snapshot_path: &std::path::Path) -> Self {
        Self {
            path: snapshot_path.with_file_name("index-queue.json"),
            lock: Mutex::new(()),
        }
    }

    async fn load(&self) -> Result<PersistedIndexQueue> {
        let _guard = self.lock.lock().await;
        if !self.path.exists() {
            return Ok(PersistedIndexQueue::default());
        }
        let text = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("reading index queue at {}", self.path.display()))?;
        serde_json::from_str(&text).context("parsing index queue json")
    }

    async fn write(&self, queue: &PersistedIndexQueue) -> Result<()> {
        let _guard = self.lock.lock().await;
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating index queue dir {}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(queue).context("serializing index queue json")?;
        atomic_replace(&self.path, format!("{text}\n").as_bytes())
            .await
            .with_context(|| format!("writing index queue at {}", self.path.display()))
    }
}

fn restore_persisted_index_queue(
    config: &Config,
    queue: PersistedIndexQueue,
) -> (IndexCoordinatorState, BTreeSet<String>) {
    let mut coordinator = IndexCoordinatorState::default();
    let mut resumable = BTreeSet::new();

    for request in queue
        .pending
        .into_values()
        .chain(queue.running.into_values())
    {
        let request = match request.into_pending() {
            Ok(request) => request,
            Err(error) => {
                eprintln!("[warn] dropping invalid persisted index request: {error}");
                continue;
            }
        };
        let repo_key = request.repo.display().to_string();
        match config.is_configured_repo_or_worktree(&request.repo) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!(
                    "[info] dropping persisted index request for unconfigured repo {repo_key}"
                );
                continue;
            }
            Err(error) => {
                eprintln!(
                    "[warn] unable to validate persisted index request for {repo_key}: {error}"
                );
                continue;
            }
        }

        if let Some(existing) = coordinator.pending.get_mut(&repo_key) {
            existing.force |= request.force;
            existing.explicit_refresh |= request.explicit_refresh;
            existing.retry_attempt = existing.retry_attempt.max(request.retry_attempt);
            if !request.custom_extensions.is_empty() {
                existing.custom_extensions = request.custom_extensions;
            }
            if !request.ignore_patterns.is_empty() {
                existing.ignore_patterns = request.ignore_patterns;
            }
        } else {
            coordinator.pending.insert(repo_key.clone(), request);
        }
        resumable.insert(repo_key);
    }

    (coordinator, resumable)
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchCodeArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    extension_filter: Vec<String>,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default = "default_dedupe_by_file")]
    dedupe_by_file: bool,
    #[serde(default)]
    include_diagnostics: bool,
    #[serde(default = "default_include_content")]
    include_content: bool,
    #[serde(default = "default_snippet_chars")]
    snippet_chars: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct IndexCodebaseArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    force: bool,
    #[serde(default = "default_splitter")]
    splitter: String,
    #[serde(default)]
    custom_extensions: Vec<String>,
    #[serde(default)]
    ignore_patterns: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ScopeArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchSymbolsArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    container: Option<String>,
    #[serde(default)]
    include_symbol_id: bool,
    #[serde(default)]
    include_diagnostics: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct AnalyzeImpactArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    symbol_id: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    line: Option<u64>,
    #[serde(default = "default_impact_depth")]
    max_depth: usize,
    #[serde(default = "default_impact_nodes")]
    max_nodes: usize,
    #[serde(default = "default_true")]
    include_tests: bool,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default)]
    include_possible: bool,
    #[serde(default = "default_graph_detail")]
    detail: String,
    #[serde(default = "default_max_evidence_per_node")]
    max_evidence_per_node: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct TracePathArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    from_symbol_id: Option<String>,
    #[serde(default)]
    from_file: Option<String>,
    #[serde(default)]
    from_line: Option<u64>,
    #[serde(default)]
    to_symbol_id: Option<String>,
    #[serde(default)]
    to_file: Option<String>,
    #[serde(default)]
    to_line: Option<u64>,
    #[serde(default = "default_trace_depth")]
    max_depth: usize,
    #[serde(default = "default_trace_paths")]
    max_paths: usize,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default = "default_graph_detail")]
    detail: String,
    #[serde(default = "default_max_evidence_per_node")]
    max_evidence_per_node: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct CoverageArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default = "default_graph_detail")]
    detail: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct AnalyzeChangesArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default = "default_base_ref")]
    base_ref: String,
    #[serde(default)]
    include_untracked: bool,
    #[serde(default = "default_impact_depth")]
    max_depth: usize,
    #[serde(default = "default_impact_nodes")]
    max_nodes: usize,
    #[serde(default = "default_true")]
    include_tests: bool,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default = "default_graph_detail")]
    detail: String,
    #[serde(default = "default_max_evidence_per_node")]
    max_evidence_per_node: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct AnalyzeDeadCodeArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default = "default_dead_code_groups")]
    max_groups: usize,
    #[serde(default = "default_true")]
    include_tests: bool,
    #[serde(default = "default_true")]
    include_risk_candidates: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ExplainRemovalArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    group_id: String,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct ValidateRemovalArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    group_ids: Vec<String>,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default)]
    all_features: bool,
    #[serde(default)]
    include_tests: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct EstimateTrimArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default = "default_min_confidence")]
    min_confidence: u64,
    #[serde(default = "default_dead_code_groups")]
    max_groups: usize,
    #[serde(default = "default_true")]
    include_tests: bool,
    #[serde(default = "default_true")]
    include_risk_candidates: bool,
    #[serde(default)]
    validate: bool,
    #[serde(default = "default_validation_group_limit")]
    validation_group_limit: usize,
    #[serde(default)]
    all_features: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct SearchTextArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    query: String,
    #[serde(default = "default_text_search_limit")]
    limit: usize,
    #[serde(default)]
    path_prefix: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    extension_filter: Vec<String>,
    #[serde(default = "default_case_sensitive")]
    case_sensitive: bool,
    #[serde(default)]
    whole_word: bool,
    #[serde(default = "default_text_search_context_lines")]
    context_lines: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct PrepareEditTargetArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    #[serde(default)]
    file: Option<String>,
    #[serde(default)]
    symbol_id: Option<String>,
    #[serde(default)]
    symbol_name: Option<String>,
    #[serde(default)]
    symbol_kind: Option<String>,
    #[serde(default)]
    symbol_container: Option<String>,
    #[serde(default)]
    line_hint: Option<u64>,
    #[serde(default)]
    query: Option<String>,
    #[serde(default)]
    occurrence: Option<usize>,
    #[serde(default = "default_before_lines")]
    before_lines: usize,
    #[serde(default = "default_after_lines")]
    after_lines: usize,
    #[serde(default = "default_max_lines")]
    max_lines: usize,
    #[serde(default = "default_anchor_count")]
    anchor_count: usize,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
struct FileOutlineArgs {
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    path: Option<String>,
    file: String,
    #[serde(default)]
    detail: Option<String>,
    #[serde(default = "default_outline_depth")]
    max_depth: usize,
    #[serde(default = "default_outline_max_nodes")]
    max_nodes: usize,
    #[serde(default = "default_outline_top_level_limit")]
    top_level_limit: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct IndexLaunchResult {
    scope: String,
    label: String,
    started: bool,
    force: bool,
    queued_repos: Vec<String>,
    merged_repos: Vec<String>,
    already_running: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EnqueueRefreshRequest {
    path: String,
    #[serde(default)]
    force: bool,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    repos: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
struct ListScopesArgs {
    #[serde(default)]
    include_repos: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSummary {
    id: String,
    label: String,
    repo_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    repos: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ListScopesResult {
    default_scope: String,
    groups: Vec<ScopeSummary>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSearchResponse {
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    plan: Option<SearchPlanSummary>,
    hits: Vec<CompactSearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSearchHit {
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_label: Option<String>,
    relative_path: String,
    line: u64,
    #[serde(skip_serializing_if = "String::is_empty")]
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lexical_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    indexed_at: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSymbolSearchResponse {
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    hits: Vec<CompactSymbolHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactSymbolHit {
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_label: Option<String>,
    relative_path: String,
    name: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    container: Option<String>,
    line: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lexical_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    semantic_score: Option<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactTextSearchResponse {
    #[serde(skip_serializing_if = "is_false")]
    partial: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    repo_errors: Vec<RepoSearchError>,
    hits: Vec<CompactTextSearchHit>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactTextSearchHit {
    #[serde(skip_serializing_if = "Option::is_none")]
    repo_label: Option<String>,
    relative_path: String,
    line: u64,
    preview: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactPrepareEditTargetResponse {
    status: EditTargetStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    relative_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    start_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    anchors: Vec<EditTargetAnchor>,
    #[serde(skip_serializing_if = "is_false")]
    unindexed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_id: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    truncated: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    candidates: Vec<CompactEditTargetCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason_code: Option<EditTargetReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suggested_next_tool: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactEditTargetCandidate {
    relative_path: String,
    start_line: u64,
    end_line: u64,
    preview: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutlineDetail {
    Summary,
    Compact,
    Full,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactFileOutlineResponse {
    matches: Vec<CompactFileOutlineMatch>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactFileOutlineMatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    repo: Option<String>,
    #[serde(skip_serializing_if = "is_false")]
    truncated: bool,
    symbols: Vec<CompactOutlineNode>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompactOutlineNode {
    name: String,
    kind: String,
    line: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    children: Vec<CompactOutlineNode>,
}

pub async fn serve(
    config: &Config,
    listen: &str,
    allow_remote_unauthenticated: bool,
) -> Result<()> {
    enforce_loopback_bind(listen, allow_remote_unauthenticated)?;
    let engine = Engine::new(config).await?;
    let index_queue = Arc::new(IndexQueueStore::new(&config.snapshot_path));
    let persisted_queue = match index_queue.load().await {
        Ok(queue) => queue,
        Err(error) => {
            eprintln!("[error] unable to load persisted index queue: {error:#}");
            PersistedIndexQueue::default()
        }
    };
    let (index_coordinator, resumable_repos) =
        restore_persisted_index_queue(config, persisted_queue);
    let interrupted_indexes = engine
        .mark_interrupted_indexing_failed_except(
            "agent-context restarted while indexing was in progress",
            &resumable_repos,
        )
        .await?;
    if interrupted_indexes > 0 {
        eprintln!(
            "[warn] marked {interrupted_indexes} interrupted background index(es) failed during startup recovery"
        );
    }
    let server = NativeServer {
        engine,
        index_coordinator: Arc::new(Mutex::new(index_coordinator)),
        index_queue,
    };
    if !resumable_repos.is_empty() {
        let repos = resumable_repos.into_iter().collect::<Vec<_>>();
        match server.mark_repos_indexing_queued(&repos).await {
            Ok(()) => {
                let worker_generation = {
                    let mut coordinator = server.index_coordinator.lock().await;
                    coordinator.mark_pending_snapshot_queued(&repos);
                    coordinator.ensure_live_worker_for_pending(Instant::now())
                };
                server
                    .persist_index_queue_or_log("startup queue recovery")
                    .await;
                eprintln!(
                    "[info] resumed {} background index request(s) from the persisted queue",
                    repos.len()
                );
                if let Some(generation) = worker_generation {
                    server.spawn_background_index_worker(generation);
                }
            }
            Err(error) => eprintln!(
                "[error] unable to resume persisted index queue; retaining it for the next restart: {error:#}"
            ),
        }
    }
    let cancellation = CancellationToken::new();

    if let Some(interval_secs) = config.freshness.audit_interval_secs {
        let audit_engine = server.engine.clone();
        let audit_token = cancellation.child_token();
        tokio::spawn(async move {
            let duration = std::time::Duration::from_secs(interval_secs);
            loop {
                tokio::select! {
                    _ = audit_token.cancelled() => break,
                    _ = tokio::time::sleep(duration) => {
                        let _ = audit_engine.cheap_audit_once().await;
                    }
                }
            }
        });
    }

    let service = StreamableHttpService::<NativeServer, LocalSessionManager>::new(
        {
            let server = server.clone();
            move || Ok(server.clone())
        },
        Default::default(),
        StreamableHttpServerConfig::default()
            .with_sse_keep_alive(Some(MCP_SSE_KEEP_ALIVE))
            .with_cancellation_token(cancellation.child_token()),
    );

    let router = Router::new()
        .route("/health", get(health))
        .route(
            "/enqueue-refresh",
            post({
                let server = server.clone();
                move |payload| enqueue_refresh(server.clone(), payload)
            }),
        )
        .nest_service("/mcp", service)
        .layer(middleware::from_fn(log_slow_http_request));

    let listener = TcpListener::bind(listen)
        .await
        .with_context(|| format!("binding MCP listener on {listen}"))?;

    let shutdown_token = cancellation.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            wait_for_signal().await;
            shutdown_token.cancel();
        })
        .await
        .context("serving native HTTP MCP endpoint")
}

const SERVER_INSTRUCTIONS: &str = "Start with list_scopes once, then use search_symbols for exact definitions, search_code for discovery, search_text for exact literals instead of shell rg when indexed, and get_file_outline for known-file structure. Check get_indexing_status.features and searchAvailable when a repository lacks an expected retrieval surface: lexical, semantic, and graph indexing can each be disabled per repository, a failed refresh can leave the last committed search index available, and tool warnings identify disabled search channels. Request includeSymbolId when a symbol will feed analyze_impact, trace_path, or prepare_edit_target; graph tools also accept file+line selectors. Use analyze_impact for callers and affected tests, trace_path for a directed dependency explanation, analyze_changes for baseRef-to-working-tree risk, and check_index_coverage before interpreting an empty graph result. For repository trimming, start with analyze_dead_code, inspect candidate groups with explain_removal, validate selected groups in disposable storage with validate_removal, and use estimate_trim for bounded aggregate LOC. Never describe candidateLoc as safely removable; only compilerValidatedLoc has passed the configured compiler check, and public API, macro, feature-gated, plugin, serialization, dynamic-dispatch, unresolved-reference, and external-consumer risks remain explicit blockers. Graph responses use compact agent-ready objects by default; request detail=full only when complete canonical metadata is needed. Follow status and nextActions instead of treating needs_index, symbol_not_found, not_found, or truncation as proof of no dependency. Confidence >=650 is structural; lower values are possible evidence only. Rust and TypeScript are the guaranteed impact languages; repository trim analysis and compiler validation currently target Rust. Use prepare_edit_target only as the final pre-patch step after the exact edit location is known. Relationship analysis requires index format v2 and never rebuilds automatically. scope defaults to the configured default group.";

pub fn tool_list() -> Vec<Tool> {
    vec![
        build_tool(
            "index_codebase",
            "Start background indexing for a configured scope. Worktree scopes refresh only their overlay in overlay mode. Defaults to the configured default group.",
            false,
            index_codebase_schema(),
        ),
        build_tool(
            "search_symbols",
            "Exact symbol and definition lookup. Use this when you know or suspect the definition name.",
            true,
            search_symbols_schema(),
        ),
        build_tool(
            "search_code",
            "Broader code discovery across indexed repos. Snippets are discovery hints; use search_text for exact follow-up once a file or pathPrefix is known.",
            true,
            search_code_schema(),
        ),
        build_tool(
            "search_text",
            "Exact literal confirmation inside a known repo, file, or pathPrefix. Use this instead of shell rg for identifiers, test names, log lines, and exact strings once discovery is done.",
            true,
            search_text_schema(),
        ),
        build_tool(
            "get_file_outline",
            "Return a compact indexed symbol outline summary for a known repo-relative file. Use this for structure instead of broad file reads.",
            true,
            get_file_outline_schema(),
        ),
        build_tool(
            "prepare_edit_target",
            "Final pre-patch step only. Use immediately before editing instead of sed/bat/cat after the exact patch location is known. Not for overview or broad inspection.",
            true,
            prepare_edit_target_schema(),
        ),
        build_graph_tool(
            "analyze_impact",
            "Find callers, transitive dependents, and affected tests for one Rust/TypeScript definition. Pass a symbolId from search_symbols or a repo-relative file+line; returns compact evidence and recovery actions.",
            analyze_impact_schema(),
        ),
        build_graph_tool(
            "trace_path",
            "Explain how one Rust/TypeScript definition depends on another. Each endpoint accepts a symbolId or repo-relative file+line; a not_found status includes coverage-aware next actions.",
            trace_path_schema(),
        ),
        build_graph_tool(
            "analyze_changes",
            "Assess current working-tree Rust/TypeScript declaration changes against a Git base and rank affected code/tests. Returns explicit invalid_base and needs_index states instead of implying safety.",
            analyze_changes_schema(),
        ),
        build_graph_tool(
            "check_index_coverage",
            "Decide whether graph results are trustworthy enough to interpret. Reports readiness, stale files, supported coverage, resolution tiers, and the exact indexing action when refresh is required.",
            coverage_schema(),
        ),
        build_graph_tool(
            "analyze_dead_code",
            "Census Rust declarations that are unreachable, test-only, legacy-looking, or blocked by dynamic/public API risk. Groups cycles into removal units and reports exact declaration LOC without editing the repository.",
            analyze_dead_code_schema(),
        ),
        build_graph_tool(
            "explain_removal",
            "Explain one dead-code group, including its SCC members, exact removal ranges, test reachability, confidence, and blockers.",
            explain_removal_schema(),
        ),
        build_graph_tool(
            "validate_removal",
            "Remove selected dead-code groups only inside a disposable checkout and run a fixed Cargo check. The configured source checkout is never edited.",
            validate_removal_schema(),
        ),
        build_graph_tool(
            "estimate_trim",
            "Aggregate candidate, clean, review-required, risk-excluded, and optionally compiler-validated Rust LOC into a defensible repository trim estimate.",
            estimate_trim_schema(),
        ),
        build_tool(
            "explain_search",
            "Explain how search_code will classify and weight a query.",
            true,
            explain_search_schema(),
        ),
        build_tool(
            "clear_index",
            "Drop the index for a configured scope.",
            false,
            scope_args_schema(
                "Configured group id or repo root. Defaults to the configured default group.",
            ),
        ),
        build_tool(
            "get_indexing_status",
            "Report indexing status for a configured scope.",
            true,
            scope_args_schema(
                "Configured group id or repo root. Defaults to the configured default group.",
            ),
        ),
        build_tool(
            "list_scopes",
            "Preferred first call. Lists configured scopes.",
            true,
            list_scopes_schema(),
        ),
    ]
}

async fn health() -> &'static str {
    "ok"
}

async fn log_slow_http_request(request: Request, next: Next) -> Response {
    let method = request.method().clone();
    let path = request.uri().path().to_string();
    let started = Instant::now();
    let response = next.run(request).await;
    let elapsed = started.elapsed();
    if elapsed >= SLOW_HTTP_REQUEST || !response.status().is_success() {
        eprintln!(
            "[warn] {} HTTP {} {} completed status={} elapsed_ms={}",
            crate::snapshot::timestamp(),
            method,
            path,
            response.status(),
            elapsed.as_millis()
        );
    }
    response
}

async fn enqueue_refresh(server: NativeServer, body: Bytes) -> (StatusCode, String) {
    let payload: EnqueueRefreshRequest = match serde_json::from_slice(&body) {
        Ok(payload) => payload,
        Err(error) => {
            return (
                StatusCode::BAD_REQUEST,
                json!({ "error": error.to_string() }).to_string(),
            );
        }
    };

    let scope = if payload.repos.is_empty() {
        match server
            .engine
            .config()
            .resolve_scope(None, Some(&payload.path))
        {
            Ok(scope) => scope,
            Err(error) => {
                return (
                    StatusCode::BAD_REQUEST,
                    json!({ "error": error.to_string() }).to_string(),
                );
            }
        }
    } else {
        let repos = payload.repos.iter().map(PathBuf::from).collect::<Vec<_>>();
        ResolvedScope {
            kind: match payload.kind.as_deref() {
                Some("repo") if repos.len() == 1 => crate::config::ScopeKind::Repo,
                _ => crate::config::ScopeKind::Group,
            },
            id: payload.scope.unwrap_or_else(|| payload.path.clone()),
            label: payload.label.unwrap_or_else(|| payload.path.clone()),
            repos,
        }
    };

    match server
        .enqueue_scope_indexing(
            scope,
            payload.force,
            true,
            SplitterKind::Ast,
            Vec::new(),
            Vec::new(),
        )
        .await
    {
        Ok(result) => (StatusCode::OK, json!(result).to_string()),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            json!({ "error": error.to_string() }).to_string(),
        ),
    }
}

fn enforce_loopback_bind(listen: &str, allow_remote_unauthenticated: bool) -> Result<()> {
    if allow_remote_unauthenticated || listen_is_loopback(listen) {
        return Ok(());
    }

    bail!(
        "refusing to bind unauthenticated HTTP MCP server to non-loopback address `{listen}`; use --allow-remote-unauthenticated to override"
    );
}

fn listen_is_loopback(listen: &str) -> bool {
    if let Ok(addr) = listen.parse::<SocketAddr>() {
        return addr.ip().is_loopback();
    }

    if let Some(rest) = listen.strip_prefix("localhost:") {
        return !rest.is_empty();
    }

    false
}

async fn wait_for_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        if let Ok(mut stream) = signal(SignalKind::terminate()) {
            let _ = stream.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

impl ServerHandler for NativeServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(SERVER_INSTRUCTIONS)
    }

    #[allow(clippy::manual_async_fn)]
    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<ListToolsResult, McpError>> + Send + '_ {
        async move {
            Ok(ListToolsResult {
                tools: tool_list(),
                next_cursor: None,
                meta: None,
            })
        }
    }

    #[allow(clippy::manual_async_fn)]
    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl Future<Output = Result<CallToolResult, McpError>> + Send + '_ {
        async move {
            let args = request.arguments.unwrap_or_default();
            match request.name.as_ref() {
                "index_codebase" => {
                    let args: IndexCodebaseArgs = parse_args(args)?;
                    let splitter = parse_splitter_kind(&args.splitter).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .enqueue_scope_indexing(
                            scope,
                            args.force,
                            false,
                            splitter,
                            args.custom_extensions,
                            args.ignore_patterns,
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_index_launch_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "search_code" => {
                    let args: SearchCodeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let mode = parse_search_mode(args.mode.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_scope(
                            scope,
                            SearchRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                mode,
                                extension_filter,
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                dedupe_by_file: args.dedupe_by_file,
                                snippet_chars: if args.include_content {
                                    args.snippet_chars.min(1200)
                                } else {
                                    0
                                },
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(search_tool_success(
                        render_search_summary_text(&result),
                        compact_search_response_value(&result, args.include_diagnostics),
                    ))
                }
                "search_symbols" => {
                    let args: SearchSymbolsArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_symbols(
                            scope,
                            SymbolSearchScopeRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                kind: normalize_optional_kind(&args.kind),
                                container: normalize_optional_string(&args.container),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(search_tool_success(
                        render_symbol_search_summary_text(&result),
                        compact_symbol_search_response_value(
                            &result,
                            args.include_symbol_id,
                            args.include_diagnostics,
                        ),
                    ))
                }
                "search_text" => {
                    let args: SearchTextArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .search_text_scope(
                            scope,
                            TextSearchScopeRequest {
                                repo: normalize_optional_string(&args.repo),
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                extension_filter,
                                case_sensitive: args.case_sensitive,
                                whole_word: args.whole_word,
                                context_lines: args.context_lines.min(12),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(search_tool_success(
                        render_text_search_summary_text(&result),
                        compact_text_search_response_value(&result),
                    ))
                }
                "prepare_edit_target" => {
                    let args: PrepareEditTargetArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .prepare_edit_target(
                            scope,
                            PrepareEditTargetRequest {
                                repo: normalize_optional_string(&args.repo),
                                file: normalize_optional_path(&args.file),
                                symbol_id: normalize_optional_string(&args.symbol_id),
                                symbol_name: normalize_optional_string(&args.symbol_name),
                                symbol_kind: normalize_optional_kind(&args.symbol_kind),
                                symbol_container: normalize_optional_string(&args.symbol_container),
                                line_hint: args.line_hint,
                                query: args.query.filter(|value| !value.is_empty()),
                                occurrence: args.occurrence,
                                before_lines: args.before_lines.min(32),
                                after_lines: args.after_lines.min(96),
                                max_lines: args.max_lines.clamp(1, PREPARE_READY_MAX_LINES),
                                anchor_count: args.anchor_count.clamp(1, 3),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_prepare_edit_target_summary_text(&result),
                        serde_json::to_value(compact_prepare_edit_target_response(&result)).ok(),
                    ))
                }
                "analyze_impact" => {
                    let args: AnalyzeImpactArgs = parse_args(args)?;
                    let detail = parse_graph_detail(&args.detail).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .analyze_impact(
                            scope.clone(),
                            ImpactRequest {
                                repo: repo.clone(),
                                symbol_id: normalize_optional_string(&args.symbol_id),
                                file: normalize_optional_path(&args.file),
                                line: args.line,
                                max_depth: args.max_depth,
                                max_nodes: args.max_nodes,
                                include_tests: args.include_tests,
                                min_confidence: args.min_confidence.min(1000),
                                include_possible: args.include_possible,
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            let coverage = self
                                .engine
                                .relationship_readiness(scope.clone(), repo.as_deref())
                                .await
                                .map_err(internal_error)?;
                            let value = json!({
                                "status": "needs_index",
                                "needsIndex": true,
                                "reason": "relationship graph is stale, incomplete, or incompatible",
                                "coverage": coverage,
                                "nextActions": [graph_action(
                                    "index_codebase",
                                    "Refresh this scope before retrying impact analysis.",
                                    json!({"scope": scope.id, "force": false}),
                                )],
                            });
                            return Ok(tool_success(
                                "Impact analysis needs a current relationship index".to_string(),
                                Some(value),
                            ));
                        }
                        Err(error) => {
                            if let Some(value) = graph_recoverable_error_value(
                                &error,
                                "analyze_impact",
                                &scope.id,
                                repo.as_deref(),
                            ) {
                                return Ok(tool_success(
                                    "Impact analysis needs a more precise symbol selector"
                                        .to_string(),
                                    Some(value),
                                ));
                            }
                            return Err(internal_error(error));
                        }
                    };
                    let status = impact_response_status(&result);
                    let summary = format!(
                        "Impact `{status}`: {} direct, {} transitive, {} tests, {} possible dependents, {} possible tests{}",
                        result.direct_dependents.len(),
                        result.transitive_dependents.len(),
                        result.affected_tests.len(),
                        result.possible_dependents.len(),
                        result.possible_tests.len(),
                        if result.truncated { " (truncated)" } else { "" }
                    );
                    Ok(tool_success(
                        summary,
                        Some(impact_response_value(
                            &result,
                            detail,
                            args.max_evidence_per_node,
                        )),
                    ))
                }
                "trace_path" => {
                    let args: TracePathArgs = parse_args(args)?;
                    let detail = parse_graph_detail(&args.detail).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .trace_dependency_path(
                            scope.clone(),
                            TracePathRequest {
                                repo: repo.clone(),
                                from_symbol_id: normalize_optional_string(&args.from_symbol_id),
                                from_file: normalize_optional_path(&args.from_file),
                                from_line: args.from_line,
                                to_symbol_id: normalize_optional_string(&args.to_symbol_id),
                                to_file: normalize_optional_path(&args.to_file),
                                to_line: args.to_line,
                                max_depth: args.max_depth,
                                max_paths: args.max_paths,
                                min_confidence: args.min_confidence.min(1000),
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            let coverage = self
                                .engine
                                .relationship_readiness(scope.clone(), repo.as_deref())
                                .await
                                .map_err(internal_error)?;
                            let value = json!({
                                "status": "needs_index",
                                "needsIndex": true,
                                "reason": "relationship graph is stale, incomplete, or incompatible",
                                "coverage": coverage,
                                "nextActions": [graph_action(
                                    "index_codebase",
                                    "Refresh this scope before retrying path tracing.",
                                    json!({"scope": scope.id, "force": false}),
                                )],
                            });
                            return Ok(tool_success(
                                "Path tracing needs a current relationship index".to_string(),
                                Some(value),
                            ));
                        }
                        Err(error) => {
                            if let Some(value) = graph_recoverable_error_value(
                                &error,
                                "trace_path",
                                &scope.id,
                                repo.as_deref(),
                            ) {
                                return Ok(tool_success(
                                    "Path tracing needs valid from/to symbol selectors".to_string(),
                                    Some(value),
                                ));
                            }
                            return Err(internal_error(error));
                        }
                    };
                    let status = trace_response_status(&result);
                    let summary = format!(
                        "Dependency path `{status}`: {} path(s){}",
                        result.paths.len(),
                        if result.truncated { " (truncated)" } else { "" }
                    );
                    Ok(tool_success(
                        summary,
                        Some(trace_response_value(
                            &result,
                            detail,
                            args.max_evidence_per_node,
                            args.max_depth,
                        )),
                    ))
                }
                "analyze_changes" => {
                    let args: AnalyzeChangesArgs = parse_args(args)?;
                    let detail = parse_graph_detail(&args.detail).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .analyze_changes(
                            scope,
                            AnalyzeChangesRequest {
                                repo: normalize_optional_string(&args.repo),
                                base_ref: args.base_ref,
                                include_untracked: args.include_untracked,
                                max_depth: args.max_depth.clamp(1, 5),
                                max_nodes: args.max_nodes.clamp(1, 250),
                                include_tests: args.include_tests,
                                min_confidence: args.min_confidence.min(1000),
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    let summary = if result.invalid_base {
                        format!(
                            "Change analysis: invalid base ({})",
                            result.reason.as_deref().unwrap_or("unknown reason")
                        )
                    } else if result.needs_index {
                        "Change analysis needs a current index".to_string()
                    } else {
                        format!(
                            "Change analysis: {} files, {} symbols, {} impact roots{}",
                            result.files.len(),
                            result.symbols.len(),
                            result.impacts.len(),
                            if result.truncated { " (truncated)" } else { "" }
                        )
                    };
                    Ok(tool_success(
                        summary,
                        Some(change_response_value(
                            &result,
                            detail,
                            args.max_evidence_per_node,
                        )),
                    ))
                }
                "check_index_coverage" => {
                    let args: CoverageArgs = parse_args(args)?;
                    let detail = parse_graph_detail(&args.detail).map_err(invalid_params)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .relationship_coverage(scope.clone(), args.repo.as_deref())
                        .await
                        .map_err(internal_error)?;
                    let ready = result.iter().all(|coverage| {
                        coverage.graph_status == "ready" && coverage.stale_files.is_empty()
                    });
                    let summary = format!(
                        "Relationship coverage `{}`: {} repositories",
                        if ready { "ready" } else { "needs_index" },
                        result.len()
                    );
                    Ok(tool_success(
                        summary,
                        Some(coverage_response_value(&result, detail, &scope.id)),
                    ))
                }
                "analyze_dead_code" => {
                    let args: AnalyzeDeadCodeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .analyze_dead_code(
                            scope.clone(),
                            DeadCodeRequest {
                                repo: repo.clone(),
                                min_confidence: args.min_confidence.min(1_000),
                                max_groups: args.max_groups.clamp(1, 2_000),
                                include_tests: args.include_tests,
                                include_risk_candidates: args.include_risk_candidates,
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            return trim_needs_index_response(
                                &self.engine,
                                scope,
                                repo.as_deref(),
                                "Dead-code analysis needs a current relationship index",
                            )
                            .await;
                        }
                        Err(error) => return Err(internal_error(error)),
                    };
                    let summary = format!(
                        "Dead-code analysis `{}`: {} groups, {} candidate LOC, {} clean candidate LOC{}",
                        result.status,
                        result.groups.len(),
                        result.totals.candidate_loc,
                        result.totals.clean_candidate_loc,
                        if result.truncated { " (truncated)" } else { "" }
                    );
                    Ok(tool_success(
                        summary,
                        Some(dead_code_response_value(&result)),
                    ))
                }
                "explain_removal" => {
                    let args: ExplainRemovalArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .explain_removal(
                            scope.clone(),
                            ExplainRemovalRequest {
                                repo: repo.clone(),
                                group_id: args.group_id,
                                min_confidence: args.min_confidence.min(1_000),
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            return trim_needs_index_response(
                                &self.engine,
                                scope,
                                repo.as_deref(),
                                "Removal explanation needs a current relationship index",
                            )
                            .await;
                        }
                        Err(error) => return Err(internal_error(error)),
                    };
                    let summary = match result.group.as_ref() {
                        Some(group) => format!(
                            "Removal group found: {} symbols, {} LOC, {} blockers",
                            group.symbols.len(),
                            group.source_loc,
                            group.blockers.len()
                        ),
                        None => "Removal group was not found in the current analysis".to_string(),
                    };
                    Ok(tool_success(
                        summary,
                        Some(removal_explanation_response_value(&result)),
                    ))
                }
                "validate_removal" => {
                    let args: ValidateRemovalArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .validate_removal(
                            scope.clone(),
                            ValidateRemovalRequest {
                                repo: repo.clone(),
                                group_ids: args.group_ids,
                                min_confidence: args.min_confidence.min(1_000),
                                all_features: args.all_features,
                                include_tests: args.include_tests,
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            return trim_needs_index_response(
                                &self.engine,
                                scope,
                                repo.as_deref(),
                                "Removal validation needs a current relationship index",
                            )
                            .await;
                        }
                        Err(error) => return Err(internal_error(error)),
                    };
                    let summary = format!(
                        "Removal validation `{}`: {}/{} groups and {}/{} LOC compiler-validated",
                        result.status,
                        result.validated_group_ids.len(),
                        result.group_ids.len(),
                        result.compiler_validated_loc,
                        result.requested_loc,
                    );
                    Ok(tool_success(
                        summary,
                        Some(removal_validation_response_value(&result)),
                    ))
                }
                "estimate_trim" => {
                    let args: EstimateTrimArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let repo = normalize_optional_string(&args.repo);
                    let result = match self
                        .engine
                        .estimate_trim(
                            scope.clone(),
                            EstimateTrimRequest {
                                repo: repo.clone(),
                                min_confidence: args.min_confidence.min(1_000),
                                max_groups: args.max_groups.clamp(1, 2_000),
                                include_tests: args.include_tests,
                                include_risk_candidates: args.include_risk_candidates,
                                validate: args.validate,
                                validation_group_limit: args.validation_group_limit,
                                all_features: args.all_features,
                            },
                        )
                        .await
                    {
                        Ok(result) => result,
                        Err(error) if graph_requires_index(&error) => {
                            return trim_needs_index_response(
                                &self.engine,
                                scope,
                                repo.as_deref(),
                                "Trim estimation needs a current relationship index",
                            )
                            .await;
                        }
                        Err(error) => return Err(internal_error(error)),
                    };
                    let summary = format!(
                        "Trim estimate: {} candidate LOC, {} clean, {} compiler-validated, {} review-required, {} risk-excluded",
                        result.analysis.totals.candidate_loc,
                        result.analysis.totals.clean_candidate_loc,
                        result.analysis.totals.compiler_validated_loc,
                        result.analysis.totals.review_required_loc,
                        result.analysis.totals.excluded_risk_loc,
                    );
                    Ok(tool_success(
                        summary,
                        Some(trim_estimate_response_value(&result)),
                    ))
                }
                "get_file_outline" => {
                    let args: FileOutlineArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let detail =
                        parse_outline_detail(args.detail.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .get_file_outline(scope, &args.file)
                        .await
                        .map_err(internal_error)?;
                    let result = compact_outline_response(
                        result,
                        OutlineCompactionOptions {
                            detail,
                            max_depth: args.max_depth.clamp(1, 16),
                            max_nodes: args.max_nodes.clamp(1, 512),
                            top_level_limit: args.top_level_limit.clamp(1, 256),
                        },
                    );
                    Ok(tool_success(
                        render_outline_summary_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "explain_search" => {
                    let args: SearchCodeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let extension_filter = normalize_extension_filter(&args.extension_filter)
                        .map_err(invalid_params)?;
                    let mode = parse_search_mode(args.mode.as_deref()).map_err(invalid_params)?;
                    let result = self
                        .engine
                        .explain_search(
                            scope,
                            &SearchRequest {
                                query: args.query,
                                limit: args.limit.clamp(1, 50),
                                mode,
                                extension_filter,
                                path_prefix: normalize_optional_path(&args.path_prefix),
                                language: normalize_optional_language(&args.language),
                                file: normalize_optional_path(&args.file),
                                dedupe_by_file: args.dedupe_by_file,
                                snippet_chars: 0,
                            },
                        )
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_search_explanation_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "clear_index" => {
                    let args: ScopeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .clear_scope(scope)
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_clear_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "get_indexing_status" => {
                    let args: ScopeArgs = parse_args(args)?;
                    let scope = self
                        .engine
                        .config()
                        .resolve_mcp_scope(args.scope.as_deref(), args.path.as_deref())
                        .map_err(invalid_params)?;
                    let result = self
                        .engine
                        .status_scope(scope)
                        .await
                        .map_err(internal_error)?;
                    Ok(tool_success(
                        render_status_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                "list_scopes" => {
                    let args: ListScopesArgs = parse_args(args)?;
                    let result = self.list_scopes(args.include_repos);
                    Ok(tool_success(
                        render_list_scopes_text(&result),
                        serde_json::to_value(result).ok(),
                    ))
                }
                _ => Err(McpError::method_not_found::<
                    rmcp::model::CallToolRequestMethod,
                >()),
            }
        }
    }
}

impl NativeServer {
    async fn persist_index_queue(&self) -> Result<()> {
        let queue = {
            let coordinator = self.index_coordinator.lock().await;
            coordinator.persisted_queue()
        };
        self.index_queue.write(&queue).await
    }

    async fn persist_index_queue_or_log(&self, context: &str) {
        if let Err(error) = self.persist_index_queue().await {
            eprintln!("[error] unable to persist index queue after {context}: {error:#}");
        }
    }

    fn list_scopes(&self, include_repos: bool) -> ListScopesResult {
        let config = self.engine.config();
        ListScopesResult {
            default_scope: config.default_group.clone(),
            groups: config
                .groups
                .iter()
                .map(|group| ScopeSummary {
                    id: group.id.clone(),
                    label: group.label.clone().unwrap_or_else(|| group.id.clone()),
                    repo_count: group.repos.len(),
                    repos: include_repos.then(|| group.repos.clone()),
                })
                .collect(),
        }
    }

    async fn enqueue_scope_indexing(
        &self,
        scope: ResolvedScope,
        force: bool,
        explicit_refresh: bool,
        splitter: SplitterKind,
        custom_extensions: Vec<String>,
        ignore_patterns: Vec<String>,
    ) -> Result<IndexLaunchResult> {
        let mut released_recovery_repos = self.finish_incomplete_recovery_if_possible().await;
        let recovery = self.begin_stale_worker_recovery().await;
        let mut coordinator = self.index_coordinator.lock().await;
        let coordinator_before_enqueue = coordinator.clone();
        let mut queued_repos = Vec::new();
        let mut merged_repos = Vec::new();
        let mut already_running = Vec::new();
        let mut repos_to_mark_queued = Vec::new();

        for repo in &scope.repos {
            let repo_key = repo.display().to_string();
            if coordinator.running.contains_key(&repo_key) {
                already_running.push(repo_key);
            } else if let Some(existing) = coordinator.pending.get_mut(&repo_key) {
                existing.force |= force;
                existing.explicit_refresh |= explicit_refresh;
                if !custom_extensions.is_empty() {
                    existing.custom_extensions = custom_extensions.clone();
                }
                if !ignore_patterns.is_empty() {
                    existing.ignore_patterns = ignore_patterns.clone();
                }
                if !existing.snapshot_queued
                    && !coordinator.recovering.contains(&repo_key)
                    && !repos_to_mark_queued.contains(&repo_key)
                {
                    repos_to_mark_queued.push(repo_key.clone());
                }
                merged_repos.push(repo_key);
            } else {
                let snapshot_queued = false;
                coordinator.pending.insert(
                    repo_key.clone(),
                    PendingIndexRequest {
                        repo: repo.clone(),
                        force,
                        explicit_refresh,
                        splitter,
                        custom_extensions: custom_extensions.clone(),
                        ignore_patterns: ignore_patterns.clone(),
                        snapshot_queued,
                        retry_attempt: 0,
                    },
                );
                if !coordinator.recovering.contains(&repo_key)
                    && !repos_to_mark_queued.contains(&repo_key)
                {
                    repos_to_mark_queued.push(repo_key.clone());
                }
                queued_repos.push(repo_key);
            }
        }
        // Persist before changing the snapshot so a restart can always resume accepted work.
        let queue = coordinator.persisted_queue();
        if let Err(error) = self.index_queue.write(&queue).await {
            *coordinator = coordinator_before_enqueue;
            return Err(error);
        }
        drop(coordinator);

        if let Some(recovery) = recovery.as_ref()
            && recovery.failure_recorded
        {
            self.finish_worker_recovery(&recovery.running_repos).await;
            released_recovery_repos.extend(recovery.running_repos.iter().cloned());
        }

        let recovered_repos_to_mark_queued = if !released_recovery_repos.is_empty() {
            self.recovered_pending_repos_to_mark_queued(&released_recovery_repos)
                .await
        } else {
            Vec::new()
        };
        for repo in recovered_repos_to_mark_queued {
            if !repos_to_mark_queued.contains(&repo) {
                repos_to_mark_queued.push(repo);
            }
        }

        if !repos_to_mark_queued.is_empty()
            && let Err(error) = self.mark_repos_indexing_queued(&repos_to_mark_queued).await
        {
            let generation_to_spawn = {
                let mut coordinator = self.index_coordinator.lock().await;
                coordinator.remove_unqueued_pending(&repos_to_mark_queued);
                coordinator.ensure_live_worker_for_pending(Instant::now())
            };
            self.persist_index_queue_or_log("queue rollback").await;
            if let Some(generation) = generation_to_spawn {
                self.spawn_background_index_worker(generation);
            }
            return Err(error);
        }

        let worker_generation = {
            let mut coordinator = self.index_coordinator.lock().await;
            coordinator.mark_pending_snapshot_queued(&repos_to_mark_queued);
            coordinator.ensure_live_worker_for_pending(Instant::now())
        };
        self.persist_index_queue_or_log("queue acceptance").await;
        if let Some(generation) = worker_generation {
            self.spawn_background_index_worker(generation);
        }

        if queued_repos.is_empty() && merged_repos.is_empty() {
            return Ok(IndexLaunchResult {
                scope: scope.id,
                label: scope.label,
                started: worker_generation.is_some(),
                force,
                queued_repos,
                merged_repos,
                already_running,
            });
        }

        Ok(IndexLaunchResult {
            scope: scope.id,
            label: scope.label,
            started: worker_generation.is_some(),
            force,
            queued_repos,
            merged_repos,
            already_running,
        })
    }

    async fn begin_stale_worker_recovery(&self) -> Option<WorkerRecovery> {
        let recovery = {
            let mut coordinator = self.index_coordinator.lock().await;
            coordinator.begin_stale_worker_recovery(Instant::now())
        };
        let mut recovery = recovery?;

        if recovery.running_repos.is_empty() {
            eprintln!(
                "[warn] replacing stale background indexing worker age={}s with pending work but no running repos",
                recovery.age_secs
            );
        } else {
            eprintln!(
                "[warn] replacing stale background indexing worker age={}s; stale running repos: {}",
                recovery.age_secs,
                recovery.running_repos.join(", ")
            );
            recovery.failure_recorded = self
                .mark_recovered_repos_failed(
                    "stale-background-worker",
                    "stale background worker",
                    &recovery.running_repos,
                    "background indexing worker heartbeat expired",
                )
                .await;
        }

        Some(recovery)
    }

    async fn finish_incomplete_recovery_if_possible(&self) -> Vec<String> {
        let recovering_repos = {
            let coordinator = self.index_coordinator.lock().await;
            coordinator.recovering_repos()
        };
        if recovering_repos.is_empty() {
            return Vec::new();
        }
        if self
            .mark_recovered_repos_failed(
                "incomplete-background-worker-recovery",
                "incomplete background worker recovery",
                &recovering_repos,
                "background indexing worker recovery completed after retry",
            )
            .await
        {
            self.finish_worker_recovery(&recovering_repos).await;
            return recovering_repos;
        }
        Vec::new()
    }

    async fn mark_repos_indexing_queued(&self, repos: &[String]) -> Result<()> {
        if repos.is_empty() {
            return Ok(());
        }
        let scope = ResolvedScope {
            kind: crate::config::ScopeKind::Group,
            id: "background-index-queue".to_string(),
            label: "background index queue".to_string(),
            repos: repos.iter().map(PathBuf::from).collect(),
        };
        self.engine.mark_scope_indexing(&scope).await
    }

    async fn mark_recovered_repos_failed(
        &self,
        scope_id: &str,
        scope_label: &str,
        repos: &[String],
        reason: &str,
    ) -> bool {
        if repos.is_empty() {
            return true;
        }
        let scope = ResolvedScope {
            kind: crate::config::ScopeKind::Group,
            id: scope_id.to_string(),
            label: scope_label.to_string(),
            repos: repos.iter().map(PathBuf::from).collect(),
        };
        if let Err(error) = self.engine.mark_scope_indexing_failed(&scope, reason).await {
            eprintln!("[error] unable to mark recovered background worker repos failed: {error}");
            return false;
        }
        true
    }

    async fn finish_worker_recovery(&self, recovered_repos: &[String]) {
        let mut coordinator = self.index_coordinator.lock().await;
        coordinator.finish_recovery(recovered_repos);
    }

    async fn recovered_pending_repos_to_mark_queued(
        &self,
        recovered_repos: &[String],
    ) -> Vec<String> {
        let coordinator = self.index_coordinator.lock().await;
        if recovered_repos.is_empty() {
            return coordinator.unqueued_ready_pending_repos();
        }
        coordinator
            .unqueued_ready_pending_repos()
            .into_iter()
            .filter(|repo| recovered_repos.contains(repo))
            .collect()
    }

    async fn finish_recovery_queue_pending_and_restart(&self, recovered_repos: &[String]) {
        self.finish_worker_recovery(recovered_repos).await;
        let repos_to_mark_queued = self
            .recovered_pending_repos_to_mark_queued(recovered_repos)
            .await;
        if !repos_to_mark_queued.is_empty()
            && let Err(error) = self.mark_repos_indexing_queued(&repos_to_mark_queued).await
        {
            eprintln!("[error] unable to mark recovered pending repos queued: {error}");
            return;
        }
        let worker_generation = {
            let mut coordinator = self.index_coordinator.lock().await;
            coordinator.mark_pending_snapshot_queued(&repos_to_mark_queued);
            coordinator.ensure_live_worker_for_pending(Instant::now())
        };
        self.persist_index_queue_or_log("background worker recovery")
            .await;
        if let Some(generation) = worker_generation {
            eprintln!(
                "[warn] replacing recovered background indexing worker with generation={generation}"
            );
            self.spawn_background_index_worker(generation);
        }
    }

    fn spawn_background_index_worker(&self, generation: u64) {
        let server = self.clone();
        eprintln!("[info] starting background indexing worker generation={generation}");
        tokio::spawn(async move {
            let result =
                AssertUnwindSafe(server.run_background_index_worker(generation)).catch_unwind();
            if result.await.is_err() {
                eprintln!("[error] background indexing worker generation={generation} panicked");
                server
                    .handle_worker_abort(
                        generation,
                        "background indexing worker panicked before completing",
                    )
                    .await;
            }
        });
    }

    async fn run_background_index_worker(&self, generation: u64) {
        loop {
            let next = {
                let mut coordinator = self.index_coordinator.lock().await;
                if !coordinator.heartbeat_worker(generation, Instant::now()) {
                    eprintln!(
                        "[warn] background indexing worker generation={generation} exiting after replacement"
                    );
                    return;
                }
                let Some(repo_key) = coordinator
                    .pending
                    .iter()
                    .find(|(repo, request)| {
                        request.snapshot_queued && !coordinator.recovering.contains(*repo)
                    })
                    .map(|(repo, _)| repo.clone())
                else {
                    if coordinator.finish_worker(generation) {
                        eprintln!(
                            "[info] background indexing worker generation={generation} drained queue"
                        );
                    }
                    return;
                };
                let request = coordinator
                    .pending
                    .remove(&repo_key)
                    .expect("pending repo exists");
                coordinator.running.insert(repo_key.clone(), generation);
                coordinator
                    .active_requests
                    .insert(repo_key.clone(), request.clone());
                (repo_key, request)
            };

            self.persist_index_queue_or_log("starting background index work")
                .await;

            let (repo_key, request) = next;
            let label = request
                .repo
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_else(|| repo_key.clone());
            let scope = ResolvedScope {
                kind: crate::config::ScopeKind::Repo,
                id: repo_key.clone(),
                label,
                repos: vec![request.repo.clone()],
            };

            eprintln!(
                "[info] background indexing worker generation={generation} indexing {repo_key}"
            );
            let mut heartbeat = tokio::time::interval(INDEX_WORKER_HEARTBEAT_INTERVAL);
            heartbeat.set_missed_tick_behavior(MissedTickBehavior::Delay);
            let index_future = self.engine.index_scope_background(
                scope,
                request.force,
                request.explicit_refresh,
                request.splitter,
                &request.custom_extensions,
                &request.ignore_patterns,
            );
            tokio::pin!(index_future);
            let index_result = loop {
                tokio::select! {
                    _ = heartbeat.tick() => {
                        self.record_worker_heartbeat(generation).await;
                    }
                    result = &mut index_future => {
                        break result;
                    }
                }
            };
            let retry_reason = match &index_result {
                Ok(result) => result
                    .repos
                    .iter()
                    .filter_map(|repo| repo.error.as_deref())
                    .find(|error| retryable_background_index_error(error))
                    .map(str::to_string),
                Err(error) => {
                    let message = format!("{error:#}");
                    retryable_background_index_error(&message).then_some(message)
                }
            };
            match &index_result {
                Err(error) => eprintln!(
                    "[error] background indexing worker generation={generation} failed {repo_key}: {error:#}"
                ),
                Ok(result) if result.has_errors => {
                    eprintln!(
                        "[error] background indexing worker generation={generation} completed {repo_key} with index errors"
                    );
                    for repo in result.repos.iter().filter(|repo| repo.error.is_some()) {
                        eprintln!(
                            "[error] background indexing worker generation={generation} repo={} index_status={} graph_status={} error={}",
                            repo.repo,
                            repo.index_status.as_deref().unwrap_or("unknown"),
                            repo.graph_status.as_deref().unwrap_or("unknown"),
                            repo.error.as_deref().unwrap_or("unknown indexing error")
                        );
                    }
                }
                Ok(_) => eprintln!(
                    "[info] background indexing worker generation={generation} finished {repo_key}"
                ),
            }

            let mut retry_request = retry_reason.and_then(|reason| {
                if request.retry_attempt >= BACKGROUND_INDEX_TRANSIENT_RETRIES {
                    eprintln!(
                        "[error] background indexing worker generation={generation} exhausted transient retries for {repo_key}: {reason}"
                    );
                    return None;
                }
                let mut retry = request.clone();
                retry.retry_attempt = retry.retry_attempt.saturating_add(1);
                retry.snapshot_queued = true;
                eprintln!(
                    "[warn] background indexing worker generation={generation} will retry {repo_key} after transient backend failure (attempt {}/{})",
                    retry.retry_attempt,
                    BACKGROUND_INDEX_TRANSIENT_RETRIES
                );
                Some(retry)
            });

            if retry_request.is_some()
                && let Err(error) = self
                    .mark_repos_indexing_queued(std::slice::from_ref(&repo_key))
                    .await
            {
                eprintln!(
                    "[error] background indexing worker generation={generation} could not queue retry for {repo_key}: {error}"
                );
                retry_request = None;
            }

            let mut coordinator = self.index_coordinator.lock().await;
            if coordinator.running.get(&repo_key).copied() == Some(generation) {
                coordinator.running.remove(&repo_key);
                coordinator.active_requests.remove(&repo_key);
                if let Some(request) = retry_request {
                    coordinator.pending.insert(repo_key.clone(), request);
                }
                coordinator.heartbeat_worker(generation, Instant::now());
            } else {
                eprintln!(
                    "[warn] background indexing worker generation={generation} finished stale repo {repo_key} after replacement"
                );
                coordinator.active_requests.remove(&repo_key);
                drop(coordinator);
                self.persist_index_queue_or_log("late stale worker completion")
                    .await;
                return;
            }
            drop(coordinator);
            self.persist_index_queue_or_log("finishing background index work")
                .await;
        }
    }

    async fn record_worker_heartbeat(&self, generation: u64) {
        let mut coordinator = self.index_coordinator.lock().await;
        coordinator.heartbeat_worker(generation, Instant::now());
    }

    async fn handle_worker_abort(&self, generation: u64, reason: &str) {
        let running_repos = {
            let mut coordinator = self.index_coordinator.lock().await;
            let Some(running_repos) = coordinator.begin_worker_abort_recovery(generation) else {
                return;
            };
            running_repos
        };
        eprintln!(
            "[error] background indexing worker generation={generation} aborted: {reason}; re-queueing {} repo(s)",
            running_repos.len()
        );
        self.persist_index_queue_or_log("background worker abort")
            .await;
        self.finish_recovery_queue_pending_and_restart(&running_repos)
            .await;
    }
}

fn retryable_background_index_error(error: &str) -> bool {
    let error = error.to_ascii_lowercase();
    let transient_transport = error.contains("connection reset by peer")
        || error.contains("connection refused")
        || error.contains("tcp connect error")
        || error.contains("error sending request")
        || error.contains("timed out");
    (error.contains("ollama embeddings")
        && (transient_transport
            || (error.contains("/tokenize")
                && (error.contains("read tcp") || error.contains("eof")))))
        || (error.contains("milvus api") && transient_transport)
}

fn build_tool(
    name: &'static str,
    description: &'static str,
    read_only: bool,
    input_schema: Map<String, Value>,
) -> Tool {
    let mut annotations = ToolAnnotations::default();
    annotations.read_only_hint = Some(read_only);
    annotations.destructive_hint = Some(!read_only);
    annotations.idempotent_hint = Some(read_only);
    annotations.open_world_hint = Some(false);

    let mut tool = Tool::default();
    tool.name = Cow::Borrowed(name);
    tool.description = Some(Cow::Borrowed(description));
    tool.input_schema = Arc::new(input_schema);
    tool.annotations = Some(annotations);
    tool
}

fn build_graph_tool(
    name: &'static str,
    description: &'static str,
    input_schema: Map<String, Value>,
) -> Tool {
    let mut tool = build_tool(name, description, true, input_schema);
    tool.output_schema = Some(Arc::new(graph_output_schema()));
    tool
}

fn parse_args<T>(arguments: Map<String, Value>) -> Result<T, McpError>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(Value::Object(arguments))
        .map_err(|error| invalid_params(anyhow::anyhow!(error)))
}

fn tool_success(text: String, structured: Option<Value>) -> CallToolResult {
    let mut result = CallToolResult::default();
    result.content = vec![Content::text(text)];
    result.structured_content = structured;
    result.is_error = Some(false);
    result
}

fn search_tool_success(summary: String, value: Value) -> CallToolResult {
    if value.is_object() {
        tool_success(summary, Some(value))
    } else {
        let text = serde_json::to_string(&value).unwrap_or(summary);
        tool_success(text, None)
    }
}

fn render_index_launch_text(result: &IndexLaunchResult) -> String {
    let mut lines = vec![format!("Scope: {}", result.label)];
    if result.started {
        lines.push(format!(
            "Started background indexing worker. force={}",
            result.force
        ));
    } else {
        lines.push(
            "Background indexing request accepted without starting a new worker.".to_string(),
        );
    }

    if !result.queued_repos.is_empty() {
        lines.push(format!("Queued: {}", result.queued_repos.join(", ")));
    }
    if !result.merged_repos.is_empty() {
        lines.push(format!("Merged: {}", result.merged_repos.join(", ")));
    }
    if !result.already_running.is_empty() {
        lines.push(format!(
            "Already running: {}",
            result.already_running.join(", ")
        ));
    }

    lines.join("\n")
}

fn render_list_scopes_text(result: &ListScopesResult) -> String {
    let mut lines = vec![format!("Default scope: {}", result.default_scope)];
    if result.groups.is_empty() {
        lines.push("No configured group scopes.".to_string());
        return lines.join("\n");
    }

    for group in &result.groups {
        lines.push(format!(
            "{} ({}) repos={}",
            group.label, group.id, group.repo_count
        ));
    }
    lines.join("\n")
}

fn render_search_summary_text(result: &SearchResponse) -> String {
    let mut lines = vec![format!(
        "hits={} partial={}",
        result.hits.len(),
        result.partial
    )];
    let include_repo_label = search_hits_span_multiple_repos(&result.hits);
    for (index, hit) in result.hits.iter().enumerate() {
        let location = format!("{}:{}", hit.relative_path, hit.start_line);
        if include_repo_label {
            lines.push(format!("{}. {} :: {}", index + 1, hit.repo_label, location));
        } else {
            lines.push(format!("{}. {}", index + 1, location));
        }
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_symbol_search_summary_text(result: &SymbolSearchResponse) -> String {
    let mut lines = vec![format!(
        "symbols={} partial={}",
        result.hits.len(),
        result.partial
    )];
    let include_repo_label = symbol_hits_span_multiple_repos(&result.hits);
    for (index, hit) in result.hits.iter().enumerate() {
        let location = format!("{}:{}", hit.relative_path, hit.start_line);
        if include_repo_label {
            lines.push(format!(
                "{}. {} :: {} {} {}",
                index + 1,
                hit.repo_label,
                location,
                hit.kind,
                hit.name
            ));
        } else {
            lines.push(format!(
                "{}. {} {} {}",
                index + 1,
                location,
                hit.kind,
                hit.name
            ));
        }
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_text_search_summary_text(result: &TextSearchResponse) -> String {
    let mut lines = vec![format!(
        "hits={} partial={}",
        result.hits.len(),
        result.partial
    )];
    if result.hits.is_empty() {
        lines.push("No exact literal hits.".to_string());
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
    }
    lines.join("\n")
}

fn render_prepare_edit_target_summary_text(
    result: &crate::engine::PrepareEditTargetResponse,
) -> String {
    match result.status {
        EditTargetStatus::Ready => format!(
            "Ready {}:{}-{} anchors={}",
            result.relative_path.as_deref().unwrap_or("unknown"),
            result.start_line.unwrap_or(0),
            result.end_line.unwrap_or(0),
            result.anchors.len()
        ),
        EditTargetStatus::Ambiguous => format!(
            "Ambiguous {} candidates={} next={}",
            result.relative_path.as_deref().unwrap_or("target"),
            result.candidates.len(),
            result
                .suggested_next_tool
                .as_deref()
                .unwrap_or("prepare_edit_target")
        ),
        EditTargetStatus::NeedsNarrowing => format!(
            "Needs narrowing {} reason={} next={}",
            result.relative_path.as_deref().unwrap_or("target"),
            result
                .reason_code
                .map(reason_code_label)
                .unwrap_or("unknown"),
            result
                .suggested_next_tool
                .as_deref()
                .unwrap_or("search_text")
        ),
        EditTargetStatus::NotFound => format!(
            "Not found {}",
            result.relative_path.as_deref().unwrap_or("target")
        ),
    }
}

fn reason_code_label(reason: EditTargetReasonCode) -> &'static str {
    match reason {
        EditTargetReasonCode::WindowTooBroad => "window_too_broad",
        EditTargetReasonCode::LargeSymbol => "large_symbol",
        EditTargetReasonCode::MultipleMatches => "multiple_matches",
        EditTargetReasonCode::WeakAnchors => "weak_anchors",
    }
}

fn render_outline_summary_text(result: &CompactFileOutlineResponse) -> String {
    let mut lines = Vec::new();
    if result.matches.is_empty() {
        lines.push("No indexed outline.".to_string());
        return lines.join("\n");
    }
    for entry in &result.matches {
        lines.push(format!(
            "outline topLevelSymbols={} truncated={}",
            entry.symbols.len(),
            entry.truncated
        ));
    }
    lines.join("\n")
}

fn compact_search_response(
    result: &SearchResponse,
    include_diagnostics: bool,
) -> CompactSearchResponse {
    let include_repo_label = search_hits_span_multiple_repos(&result.hits);
    CompactSearchResponse {
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        plan: include_diagnostics.then(|| result.plan.clone()),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactSearchHit {
                repo_label: include_repo_label.then(|| hit.repo_label.clone()),
                relative_path: hit.relative_path.clone(),
                line: hit.start_line,
                content: hit.content.clone(),
                score: include_diagnostics.then_some(round_score(hit.score)),
                dense_score: include_diagnostics
                    .then(|| hit.dense_score.map(round_score))
                    .flatten(),
                lexical_score: include_diagnostics
                    .then(|| hit.lexical_score.map(round_score))
                    .flatten(),
                symbol_score: include_diagnostics
                    .then(|| hit.symbol_score.map(round_score))
                    .flatten(),
                indexed_at: include_diagnostics
                    .then(|| hit.indexed_at.clone())
                    .flatten(),
            })
            .collect(),
    }
}

fn compact_search_response_value(result: &SearchResponse, include_diagnostics: bool) -> Value {
    let compact = compact_search_response(result, include_diagnostics);
    if !compact.partial && compact.repo_errors.is_empty() && compact.plan.is_none() {
        serde_json::to_value(compact.hits).unwrap_or_else(|_| Value::Array(Vec::new()))
    } else {
        serde_json::to_value(compact).unwrap_or(Value::Null)
    }
}

fn compact_symbol_search_response(
    result: &SymbolSearchResponse,
    include_symbol_id: bool,
    include_diagnostics: bool,
) -> CompactSymbolSearchResponse {
    let include_repo_label = symbol_hits_span_multiple_repos(&result.hits);
    CompactSymbolSearchResponse {
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactSymbolHit {
                symbol_id: include_symbol_id.then(|| hit.symbol_id.clone()),
                repo_label: include_repo_label.then(|| hit.repo_label.clone()),
                relative_path: hit.relative_path.clone(),
                name: hit.name.clone(),
                kind: hit.kind.clone(),
                container: hit.container.clone(),
                line: hit.start_line,
                score: include_diagnostics.then_some(round_score(hit.score)),
                lexical_score: include_diagnostics
                    .then(|| hit.lexical_score.map(round_score))
                    .flatten(),
                semantic_score: include_diagnostics
                    .then(|| hit.semantic_score.map(round_score))
                    .flatten(),
            })
            .collect(),
    }
}

fn compact_symbol_search_response_value(
    result: &SymbolSearchResponse,
    include_symbol_id: bool,
    include_diagnostics: bool,
) -> Value {
    let compact = compact_symbol_search_response(result, include_symbol_id, include_diagnostics);
    if !compact.partial && compact.repo_errors.is_empty() {
        serde_json::to_value(compact.hits).unwrap_or_else(|_| Value::Array(Vec::new()))
    } else {
        serde_json::to_value(compact).unwrap_or(Value::Null)
    }
}

fn compact_text_search_response(result: &TextSearchResponse) -> CompactTextSearchResponse {
    let include_repo_label = text_hits_span_multiple_repos(&result.hits);
    CompactTextSearchResponse {
        partial: result.partial,
        repo_errors: result.repo_errors.clone(),
        hits: result
            .hits
            .iter()
            .map(|hit| CompactTextSearchHit {
                repo_label: include_repo_label.then(|| hit.repo_label.clone()),
                relative_path: hit.relative_path.clone(),
                line: hit.start_line,
                preview: hit.preview.clone(),
            })
            .collect(),
    }
}

fn compact_text_search_response_value(result: &TextSearchResponse) -> Value {
    let compact = compact_text_search_response(result);
    if !compact.partial && compact.repo_errors.is_empty() {
        serde_json::to_value(compact.hits).unwrap_or_else(|_| Value::Array(Vec::new()))
    } else {
        serde_json::to_value(compact).unwrap_or(Value::Null)
    }
}

fn search_hits_span_multiple_repos(hits: &[SearchHit]) -> bool {
    hits.iter()
        .map(|hit| hit.repo_label.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1
}

fn symbol_hits_span_multiple_repos(hits: &[SymbolSearchHit]) -> bool {
    hits.iter()
        .map(|hit| hit.repo_label.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1
}

fn text_hits_span_multiple_repos(hits: &[TextSearchHit]) -> bool {
    hits.iter()
        .map(|hit| hit.repo_label.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        > 1
}

fn compact_prepare_edit_target_response(
    result: &PrepareEditTargetResponse,
) -> CompactPrepareEditTargetResponse {
    CompactPrepareEditTargetResponse {
        status: result.status,
        relative_path: result.relative_path.clone(),
        start_line: result.start_line,
        end_line: result.end_line,
        content: result.content.clone(),
        anchors: result.anchors.clone(),
        unindexed: matches!(result.indexed, Some(false)),
        symbol_id: result.symbol_id.clone(),
        truncated: result.truncated.unwrap_or(false),
        candidates: result
            .candidates
            .iter()
            .map(|candidate| CompactEditTargetCandidate {
                relative_path: candidate.relative_path.clone(),
                start_line: candidate.start_line,
                end_line: candidate.end_line,
                preview: candidate.preview.clone(),
            })
            .collect(),
        reason_code: result.reason_code,
        suggested_next_tool: result.suggested_next_tool.clone(),
    }
}

#[derive(Debug, Clone, Copy)]
struct OutlineCompactionOptions {
    detail: OutlineDetail,
    max_depth: usize,
    max_nodes: usize,
    top_level_limit: usize,
}

#[derive(Debug)]
struct OutlineBudget {
    remaining_nodes: usize,
    truncated: bool,
}

fn compact_outline_response(
    result: FileOutlineResponse,
    options: OutlineCompactionOptions,
) -> CompactFileOutlineResponse {
    let include_repo = result.matches.len() > 1;
    CompactFileOutlineResponse {
        matches: result
            .matches
            .into_iter()
            .map(|entry| {
                let mut budget = OutlineBudget {
                    remaining_nodes: options.max_nodes.max(1),
                    truncated: false,
                };
                let symbols = compact_outline_nodes(&entry.symbols, options, 1, true, &mut budget);
                CompactFileOutlineMatch {
                    repo: include_repo.then_some(entry.repo),
                    truncated: budget.truncated
                        || count_compact_outline_nodes(&symbols)
                            < count_outline_nodes(&entry.symbols),
                    symbols,
                }
            })
            .collect(),
    }
}

fn compact_outline_nodes(
    nodes: &[OutlineNode],
    options: OutlineCompactionOptions,
    depth: usize,
    is_top_level: bool,
    budget: &mut OutlineBudget,
) -> Vec<CompactOutlineNode> {
    let limit = if is_top_level {
        options.top_level_limit.max(1)
    } else {
        usize::MAX
    };
    if nodes.len() > limit {
        budget.truncated = true;
    }

    let mut compact = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        if index >= limit {
            break;
        }
        if budget.remaining_nodes == 0 {
            budget.truncated = true;
            break;
        }
        budget.remaining_nodes -= 1;

        let has_children = !node.children.is_empty();
        let children = if should_descend_outline(options, depth) && has_children {
            compact_outline_nodes(&node.children, options, depth + 1, false, budget)
        } else {
            Vec::new()
        };
        compact.push(CompactOutlineNode {
            name: node.name.clone(),
            kind: node.kind.clone(),
            line: node.start_line,
            children,
        });
    }
    compact
}

fn should_descend_outline(options: OutlineCompactionOptions, depth: usize) -> bool {
    match options.detail {
        OutlineDetail::Summary => false,
        OutlineDetail::Compact | OutlineDetail::Full => depth < options.max_depth,
    }
}

fn count_compact_outline_nodes(nodes: &[CompactOutlineNode]) -> usize {
    nodes
        .iter()
        .map(|node| 1 + count_compact_outline_nodes(&node.children))
        .sum()
}

fn count_outline_nodes(nodes: &[OutlineNode]) -> usize {
    nodes
        .iter()
        .map(|node| 1 + count_outline_nodes(&node.children))
        .sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphDetail {
    Compact,
    Full,
}

fn parse_graph_detail(value: &str) -> Result<GraphDetail> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "compact" => Ok(GraphDetail::Compact),
        "full" => Ok(GraphDetail::Full),
        other => bail!("invalid detail `{other}`; expected compact or full"),
    }
}

fn graph_action(tool: &str, reason: &str, arguments: Value) -> Value {
    json!({
        "tool": tool,
        "reason": reason,
        "arguments": arguments,
    })
}

async fn trim_needs_index_response(
    engine: &Engine,
    scope: ResolvedScope,
    repo: Option<&str>,
    summary: &str,
) -> std::result::Result<CallToolResult, McpError> {
    let coverage = engine
        .relationship_readiness(scope.clone(), repo)
        .await
        .map_err(internal_error)?;
    Ok(tool_success(
        summary.to_string(),
        Some(json!({
            "status": "needs_index",
            "needsIndex": true,
            "reason": "relationship graph is stale, incomplete, or incompatible",
            "coverage": coverage,
            "nextActions": [graph_action(
                "index_codebase",
                "Refresh this scope before retrying repository trim analysis.",
                json!({"scope": scope.id, "force": false}),
            )],
        })),
    ))
}

fn graph_value_with_status<T: Serialize>(status: &str, value: &T, actions: Vec<Value>) -> Value {
    let mut value = serde_json::to_value(value).unwrap_or_else(|_| json!({}));
    let object = value
        .as_object_mut()
        .expect("graph responses serialize as objects");
    object.insert("status".to_string(), Value::String(status.to_string()));
    if !actions.is_empty() {
        object.insert("nextActions".to_string(), Value::Array(actions));
    }
    value
}

fn compact_evidence_value(evidence: &ImpactEvidence) -> Value {
    json!({
        "relation": evidence.relation.as_str(),
        "confidence": evidence.confidence,
        "resolution": evidence.resolution,
        "file": evidence.source_path,
        "line": evidence.start_line,
        "endLine": evidence.end_line,
        "text": evidence.text,
    })
}

fn compact_impact_node_value(node: &ImpactNode, max_evidence: usize) -> Value {
    let mut value = json!({
        "name": node.name,
        "kind": node.kind,
        "file": node.relative_path,
        "line": node.start_line,
        "endLine": node.end_line,
        "role": node.source_role,
        "depth": node.depth,
        "score": round_score(node.score),
        "possible": node.possible,
        "evidence": node.evidence.iter().take(max_evidence).map(compact_evidence_value).collect::<Vec<_>>(),
    });
    if let Some(symbol_id) = node.symbol_id.as_ref() {
        value
            .as_object_mut()
            .expect("compact impact node is an object")
            .insert("symbolId".to_string(), Value::String(symbol_id.clone()));
    }
    value
}

fn compact_coverage_value(
    coverage: &RepoRelationshipCoverage,
    include_paths: bool,
    live_audit_verified: bool,
) -> Value {
    let graph_ready = coverage.graph_status == "ready";
    let mut value = json!({
        "repo": coverage.repo,
        "graphStatus": coverage.graph_status,
        "graphReady": graph_ready,
        "liveAudit": if live_audit_verified { "verified" } else { "not_run" },
        "conclusive": graph_ready && live_audit_verified && coverage.stale_files.is_empty(),
        "resolutionPercentage": round_score(coverage.resolution_percentage),
        "files": {
            "supported": coverage.supported_files,
            "unsupported": coverage.unsupported_files,
            "stale": coverage.stale_files.len(),
        },
        "definitions": coverage.definitions,
        "references": {
            "total": coverage.references,
            "definite": coverage.definite,
            "probable": coverage.probable,
            "possible": coverage.possible,
            "unresolved": coverage.unresolved,
        },
        "unstableIdentities": coverage.unstable_identities,
        "languages": coverage.by_language,
    });
    if include_paths {
        let object = value
            .as_object_mut()
            .expect("compact coverage is an object");
        object.insert(
            "staleFiles".to_string(),
            json!(coverage.stale_files.iter().take(50).collect::<Vec<_>>()),
        );
        object.insert(
            "unsupportedPaths".to_string(),
            json!(
                coverage
                    .unsupported_paths
                    .iter()
                    .take(50)
                    .collect::<Vec<_>>()
            ),
        );
    }
    value
}

fn impact_response_status(result: &ImpactResponse) -> &'static str {
    if result.diagnostic.is_some() {
        "unsupported"
    } else if result.truncated {
        "truncated"
    } else if result.direct_dependents.is_empty()
        && result.transitive_dependents.is_empty()
        && result.affected_tests.is_empty()
        && result.possible_dependents.is_empty()
        && result.possible_tests.is_empty()
    {
        "no_dependents"
    } else {
        "ok"
    }
}

fn impact_next_actions(result: &ImpactResponse) -> Vec<Value> {
    let mut actions = Vec::new();
    if result.truncated
        && let Some(symbol_id) = result.root.symbol_id.as_ref()
    {
        actions.push(graph_action(
            "analyze_impact",
            "Traversal was truncated; raise maxDepth or maxNodes only if broader impact is needed.",
            json!({
                "scope": result.scope,
                "repo": result.repo,
                "symbolId": symbol_id,
                "maxDepth": 5,
                "maxNodes": 250,
            }),
        ));
    }
    if impact_response_status(result) == "no_dependents" {
        actions.push(graph_action(
            "check_index_coverage",
            "Confirm that an empty impact result reflects sufficient graph coverage.",
            json!({"scope": result.scope, "repo": result.repo}),
        ));
        actions.push(graph_action(
            "search_code",
            "Look for dynamic, generated, or unsupported-language usage outside the structural graph.",
            json!({"scope": result.scope, "query": result.root.name, "dedupeByFile": true}),
        ));
    }
    actions
}

fn impact_response_value(
    result: &ImpactResponse,
    detail: GraphDetail,
    max_evidence: usize,
) -> Value {
    let status = impact_response_status(result);
    let actions = impact_next_actions(result);
    if detail == GraphDetail::Full {
        let mut value = graph_value_with_status(status, result, actions);
        value
            .as_object_mut()
            .expect("full impact response is an object")
            .insert("coverageAudit".to_string(), json!("verified"));
        return value;
    }
    let max_evidence = max_evidence.clamp(0, 5);
    json!({
        "status": status,
        "scope": result.scope,
        "repo": result.repo,
        "root": compact_impact_node_value(&result.root, max_evidence),
        "counts": {
            "direct": result.direct_dependents.len(),
            "transitive": result.transitive_dependents.len(),
            "tests": result.affected_tests.len(),
            "possible": result.possible_dependents.len() + result.possible_tests.len(),
            "ambiguities": result.ambiguities.len(),
        },
        "directDependents": result.direct_dependents.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        "transitiveDependents": result.transitive_dependents.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        "affectedTests": result.affected_tests.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        "possibleDependents": result.possible_dependents.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        "possibleTests": result.possible_tests.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        "ambiguities": result.ambiguities,
        "coverage": compact_coverage_value(&result.coverage, false, true),
        "diagnostic": result.diagnostic,
        "truncated": result.truncated,
        "truncationNotices": result.truncation_notices,
        "nextActions": actions,
    })
}

fn trace_response_status(result: &TracePathResponse) -> &'static str {
    if result.diagnostic.is_some() {
        "unsupported"
    } else if result.truncated {
        "truncated"
    } else if result.found {
        "found"
    } else {
        "not_found"
    }
}

fn trace_response_value(
    result: &TracePathResponse,
    detail: GraphDetail,
    max_evidence: usize,
    requested_max_depth: usize,
) -> Value {
    let status = trace_response_status(result);
    let mut actions = Vec::new();
    if !result.found {
        actions.push(graph_action(
            "check_index_coverage",
            "A missing path is conclusive only when graph coverage is sufficient.",
            json!({"scope": result.scope, "repo": result.repo}),
        ));
        if requested_max_depth < 10
            && let (Some(from_id), Some(to_id)) =
                (result.from.symbol_id.as_ref(), result.to.symbol_id.as_ref())
        {
            actions.push(graph_action(
                "trace_path",
                "Retry with a wider bound only when a longer structural path is useful.",
                json!({
                    "scope": result.scope,
                    "repo": result.repo,
                    "fromSymbolId": from_id,
                    "toSymbolId": to_id,
                    "maxDepth": (requested_max_depth + 2).min(10),
                }),
            ));
        }
    }
    if detail == GraphDetail::Full {
        let mut value = graph_value_with_status(status, result, actions);
        value
            .as_object_mut()
            .expect("full trace response is an object")
            .insert("coverageAudit".to_string(), json!("verified"));
        return value;
    }
    let max_evidence = max_evidence.clamp(0, 5);
    json!({
        "status": status,
        "scope": result.scope,
        "repo": result.repo,
        "from": compact_impact_node_value(&result.from, 0),
        "to": compact_impact_node_value(&result.to, 0),
        "pathCount": result.paths.len(),
        "paths": result.paths.iter().map(|path| json!({
            "score": round_score(path.score),
            "nodes": path.nodes.iter().map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "ambiguities": result.ambiguities,
        "coverage": compact_coverage_value(&result.coverage, false, true),
        "diagnostic": result.diagnostic,
        "truncated": result.truncated,
        "truncationNotices": result.truncation_notices,
        "nextActions": actions,
    })
}

fn change_response_status(result: &ChangeAnalysisResponse) -> &'static str {
    if result.invalid_base {
        "invalid_base"
    } else if result.needs_index {
        "needs_index"
    } else if result.files.is_empty() {
        "no_changes"
    } else if result.truncated {
        "truncated"
    } else {
        "ok"
    }
}

fn change_response_value(
    result: &ChangeAnalysisResponse,
    detail: GraphDetail,
    max_evidence: usize,
) -> Value {
    let status = change_response_status(result);
    let mut actions = Vec::new();
    if result.needs_index {
        actions.push(graph_action(
            "index_codebase",
            "Refresh the current working-tree index before retrying change analysis.",
            json!({"path": result.repo, "force": false}),
        ));
    } else if result.invalid_base {
        actions.push(graph_action(
            "analyze_changes",
            "Retry with a Git commit, branch, or tag that resolves in this repository.",
            json!({"scope": result.scope, "repo": result.repo, "baseRef": "HEAD"}),
        ));
    }
    if detail == GraphDetail::Full {
        let mut value = graph_value_with_status(status, result, actions);
        value
            .as_object_mut()
            .expect("full change response is an object")
            .insert("coverageAudit".to_string(), json!("verified"));
        return value;
    }
    let max_evidence = max_evidence.clamp(0, 5);
    let symbols = result
        .symbols
        .iter()
        .map(|change| {
            let selected = change.current.as_ref().or(change.old.as_ref());
            json!({
                "change": change.change,
                "symbolId": selected.map(|symbol| symbol.symbol_id.as_str()),
                "name": selected.map(|symbol| symbol.name.as_str()),
                "kind": selected.map(|symbol| symbol.kind.as_str()),
                "qualifiedName": selected.map(|symbol| symbol.qualified_name.as_str()),
                "file": selected.map(|symbol| symbol.relative_path.as_str()),
                "line": selected.map(|symbol| symbol.start_line),
            })
        })
        .collect::<Vec<_>>();
    let impacts = result
        .impacts
        .iter()
        .map(|impact| {
            let analysis = impact.analysis.as_ref();
            json!({
                "change": impact.change,
                "qualifiedName": impact.qualified_name,
                "status": analysis.map(impact_response_status).unwrap_or("possible_only"),
                "counts": analysis.map(|value| json!({
                    "direct": value.direct_dependents.len(),
                    "transitive": value.transitive_dependents.len(),
                    "tests": value.affected_tests.len(),
                    "possible": value.possible_dependents.len() + value.possible_tests.len(),
                })),
                "topDependents": analysis.map(|value| value.direct_dependents.iter().take(10).map(|node| compact_impact_node_value(node, max_evidence)).collect::<Vec<_>>()).unwrap_or_default(),
                "possibleEvidence": impact.possible_evidence.iter().take(max_evidence).map(compact_evidence_value).collect::<Vec<_>>(),
                "note": impact.note,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "status": status,
        "scope": result.scope,
        "repo": result.repo,
        "baseRef": result.base_ref,
        "resolvedBase": result.resolved_base,
        "reason": result.reason,
        "counts": {
            "files": result.files.len(),
            "symbols": result.symbols.len(),
            "impactRoots": result.impacts.len(),
        },
        "files": result.files,
        "symbols": symbols,
        "impacts": impacts,
        "coverage": result.coverage.as_ref().map(|value| compact_coverage_value(value, false, true)),
        "truncated": result.truncated,
        "nextActions": actions,
    })
}

fn coverage_response_value(
    result: &[RepoRelationshipCoverage],
    detail: GraphDetail,
    scope: &str,
) -> Value {
    let ready = result
        .iter()
        .all(|coverage| coverage.graph_status == "ready" && coverage.stale_files.is_empty());
    let status = if ready { "ready" } else { "needs_index" };
    let repositories = if detail == GraphDetail::Full {
        serde_json::to_value(result).unwrap_or_else(|_| json!([]))
    } else {
        Value::Array(
            result
                .iter()
                .map(|coverage| compact_coverage_value(coverage, true, true))
                .collect(),
        )
    };
    let actions = if ready {
        Vec::new()
    } else {
        vec![graph_action(
            "index_codebase",
            "Refresh repositories whose graph status is not ready or whose file hashes are stale.",
            json!({"scope": scope, "force": false}),
        )]
    };
    json!({
        "status": status,
        "ready": ready,
        "repositoryCount": result.len(),
        "repositories": repositories,
        "nextActions": actions,
    })
}

fn compact_dead_code_group_value(group: &DeadCodeGroup) -> Value {
    json!({
        "groupId": group.group_id,
        "classification": group.classification,
        "confidence": group.confidence,
        "sourceLoc": group.source_loc,
        "compilerValidatedLoc": group.compiler_validated_loc,
        "symbolCount": group.symbols.len(),
        "topSymbols": group.symbols.iter().take(5).map(|symbol| json!({
            "symbolId": symbol.symbol_id,
            "name": symbol.name,
            "kind": symbol.kind,
            "file": symbol.relative_path,
            "line": symbol.start_line,
        })).collect::<Vec<_>>(),
        "wholeFiles": group.whole_files,
        "testReferenceCount": group.test_reference_count,
        "incomingPossibleCount": group.incoming_possible_count,
        "blockerCount": group.blockers.len(),
        "blockerKinds": group.blockers.iter().map(|blocker| blocker.kind.as_str()).collect::<BTreeSet<_>>(),
        "rationale": group.rationale,
    })
}

fn safe_validation_candidate(group: &DeadCodeGroup) -> bool {
    group.blockers.is_empty()
        && !matches!(
            group.classification,
            DeadCodeClassification::DynamicOrExternalRisk
                | DeadCodeClassification::IntentionalTestSeam
        )
}

fn dead_code_next_actions(result: &DeadCodeAnalysis) -> Vec<Value> {
    let mut actions = Vec::new();
    if let Some(group) = result.groups.first() {
        actions.push(graph_action(
            "explain_removal",
            "Inspect the largest returned removal group before drawing a conclusion or editing source.",
            json!({
                "scope": result.scope,
                "repo": result.repo,
                "groupId": group.group_id,
            }),
        ));
    }
    if let Some(group) = result
        .groups
        .iter()
        .find(|group| safe_validation_candidate(group))
    {
        actions.push(graph_action(
            "validate_removal",
            "Compiler-check a blocker-free group in a disposable checkout; this does not edit the source repository.",
            json!({
                "scope": result.scope,
                "repo": result.repo,
                "groupIds": [group.group_id],
            }),
        ));
    }
    actions.push(graph_action(
        "estimate_trim",
        "Summarize repository-wide candidate, review-required, risk-excluded, and optionally compiler-validated LOC.",
        json!({
            "scope": result.scope,
            "repo": result.repo,
            "validate": false,
        }),
    ));
    actions
}

fn dead_code_response_value(result: &DeadCodeAnalysis) -> Value {
    json!({
        "status": result.status,
        "scope": result.scope,
        "repo": result.repo,
        "rootCount": result.root_count,
        "rootsTruncated": result.roots_truncated,
        "rootSample": result.roots.iter().take(20).collect::<Vec<_>>(),
        "groupCount": result.group_count,
        "returnedGroupCount": result.groups.len(),
        "totals": result.totals,
        "groups": result.groups.iter().map(compact_dead_code_group_value).collect::<Vec<_>>(),
        "coverage": compact_coverage_value(&result.coverage, false, true),
        "analysisBlockers": result.analysis_blockers,
        "truncated": result.truncated,
        "diagnostic": result.diagnostic,
        "nextActions": dead_code_next_actions(result),
    })
}

fn removal_explanation_response_value(result: &RemovalExplanation) -> Value {
    let mut actions = Vec::new();
    if let Some(group) = result.group.as_ref() {
        if safe_validation_candidate(group) {
            actions.push(graph_action(
                "validate_removal",
                "Compiler-check this blocker-free removal closure in a disposable checkout.",
                json!({
                    "scope": result.scope,
                    "repo": result.repo,
                    "groupIds": [group.group_id],
                }),
            ));
        }
    } else {
        actions.push(graph_action(
            "analyze_dead_code",
            "Refresh the group census because group identifiers are tied to the current graph generation.",
            json!({"scope": result.scope, "repo": result.repo}),
        ));
    }
    json!({
        "status": result.status,
        "scope": result.scope,
        "repo": result.repo,
        "group": result.group,
        "coverage": compact_coverage_value(&result.coverage, false, true),
        "analysisBlockers": result.analysis_blockers,
        "diagnostic": result.diagnostic,
        "nextActions": actions,
    })
}

fn removal_validation_response_value(result: &RemovalValidation) -> Value {
    let mut actions = Vec::new();
    if matches!(result.status.as_str(), "passed" | "partial") {
        actions.push(graph_action(
            "estimate_trim",
            "Recompute aggregate totals and, when desired, validate the next bounded set of blocker-free groups.",
            json!({
                "scope": result.scope,
                "repo": result.repo,
                "validate": false,
            }),
        ));
    }
    if let Some(group_id) = result.rejected_group_ids.first().or_else(|| {
        (result.status == "failed")
            .then(|| result.group_ids.first())
            .flatten()
    }) {
        actions.push(graph_action(
            "explain_removal",
            "Inspect the removal closure and blockers before interpreting compiler diagnostics.",
            json!({
                "scope": result.scope,
                "repo": result.repo,
                "groupId": group_id,
            }),
        ));
    }
    let mut value = serde_json::to_value(result).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.insert("nextActions".to_string(), Value::Array(actions));
    }
    value
}

fn trim_estimate_response_value(result: &TrimEstimate) -> Value {
    let mut analysis = dead_code_response_value(&result.analysis);
    if let Some(object) = analysis.as_object_mut() {
        object.remove("nextActions");
    }
    let validation = result
        .validation
        .as_ref()
        .map(removal_validation_response_value);
    json!({
        "status": result.validation.as_ref().map(|value| value.status.as_str()).unwrap_or(result.analysis.status.as_str()),
        "scope": result.analysis.scope,
        "repo": result.analysis.repo,
        "analysis": analysis,
        "validation": validation,
        "nextActions": dead_code_next_actions(&result.analysis),
    })
}

fn round_score(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn graph_requires_index(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| {
        let message = cause.to_string();
        message.contains("relationship graph is stale")
            || message.contains("relationship graph indexing is disabled")
            || message.contains("relationship overlay is not indexed")
    })
}

fn graph_recoverable_error_value(
    error: &anyhow::Error,
    tool: &str,
    scope: &str,
    _repo: Option<&str>,
) -> Option<Value> {
    let reason = format!("{error:#}");
    let normalized = reason.to_ascii_lowercase();
    let status = if normalized.contains("symbolid was not found")
        || normalized.contains("no indexed symbol contains")
        || normalized.contains("no indexed from symbol contains")
        || normalized.contains("no indexed to symbol contains")
    {
        "symbol_not_found"
    } else if normalized.contains("provide symbolid")
        || normalized.contains("provide fromsymbolid")
        || normalized.contains("provide tosymbolid")
        || normalized.contains("line is required")
    {
        "invalid_selector"
    } else if normalized.contains("requires one repository")
        || normalized.contains("repo hint")
        || normalized.contains("multiple repositories")
    {
        "needs_repo"
    } else {
        return None;
    };
    let mut value = json!({
        "status": status,
        "tool": tool,
        "reason": reason,
        "suggestedNextTool": "search_symbols",
        "suggestedArguments": {
            "scope": scope,
            "includeSymbolId": true,
            "limit": 5,
        },
        "missingArguments": ["query"],
    });
    if status == "needs_repo" {
        value
            .as_object_mut()
            .expect("recoverable graph error is an object")
            .insert(
                "nextActions".to_string(),
                json!([graph_action(
                    "list_scopes",
                    "Resolve the repository before retrying the graph tool.",
                    json!({}),
                )]),
            );
    }
    Some(value)
}

fn invalid_params(error: anyhow::Error) -> McpError {
    McpError::invalid_params(error.to_string(), None)
}

fn internal_error(error: anyhow::Error) -> McpError {
    McpError::internal_error(error.to_string(), None)
}

fn default_limit() -> usize {
    5
}

fn default_true() -> bool {
    true
}
fn default_impact_depth() -> usize {
    2
}
fn default_impact_nodes() -> usize {
    50
}
fn default_trace_depth() -> usize {
    5
}
fn default_trace_paths() -> usize {
    3
}
fn default_min_confidence() -> u64 {
    650
}
fn default_dead_code_groups() -> usize {
    250
}
fn default_validation_group_limit() -> usize {
    25
}
fn default_base_ref() -> String {
    "HEAD".to_string()
}

fn default_graph_detail() -> String {
    "compact".to_string()
}

fn default_max_evidence_per_node() -> usize {
    2
}

fn default_text_search_limit() -> usize {
    10
}

fn default_text_search_context_lines() -> usize {
    1
}

fn default_dedupe_by_file() -> bool {
    true
}

fn default_case_sensitive() -> bool {
    true
}

fn default_include_content() -> bool {
    true
}

fn default_snippet_chars() -> usize {
    220
}

fn default_outline_depth() -> usize {
    1
}

fn default_outline_max_nodes() -> usize {
    12
}

fn default_outline_top_level_limit() -> usize {
    8
}

fn default_before_lines() -> usize {
    8
}

fn default_after_lines() -> usize {
    24
}

fn default_max_lines() -> usize {
    96
}

fn default_anchor_count() -> usize {
    2
}

fn default_splitter() -> String {
    "ast".to_string()
}

fn normalize_extension_filter(extension_filter: &[String]) -> Result<Vec<String>> {
    let mut normalized = extension_filter
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    for extension in &normalized {
        if !(extension.starts_with('.')
            && extension.len() > 1
            && !extension.chars().any(char::is_whitespace))
        {
            anyhow::bail!(
                "invalid extensionFilter entry `{extension}`; use dotted extensions like `.ts` or `.py`"
            );
        }
    }

    normalized.sort();
    normalized.dedup();
    Ok(normalized)
}

fn parse_splitter_kind(value: &str) -> Result<SplitterKind> {
    match value {
        "ast" => Ok(SplitterKind::Ast),
        "langchain" => Ok(SplitterKind::LangChain),
        other => anyhow::bail!("invalid splitter `{other}`; must be `ast` or `langchain`"),
    }
}

fn parse_search_mode(value: Option<&str>) -> Result<SearchMode> {
    match value.unwrap_or("auto") {
        "auto" => Ok(SearchMode::Auto),
        "semantic" => Ok(SearchMode::Semantic),
        "hybrid" => Ok(SearchMode::Hybrid),
        "identifier" => Ok(SearchMode::Identifier),
        "path" => Ok(SearchMode::Path),
        other => anyhow::bail!(
            "invalid mode `{other}`; must be one of `auto`, `semantic`, `hybrid`, `identifier`, or `path`"
        ),
    }
}

fn parse_outline_detail(detail: Option<&str>) -> Result<OutlineDetail> {
    match detail
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("summary")
    {
        "summary" => Ok(OutlineDetail::Summary),
        "compact" => Ok(OutlineDetail::Compact),
        "full" => Ok(OutlineDetail::Full),
        other => anyhow::bail!(
            "invalid detail `{other}`; must be one of `summary`, `compact`, or `full`"
        ),
    }
}

fn normalize_optional_string(value: &Option<String>) -> Option<String> {
    value
        .as_ref()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn normalize_optional_path(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.replace('\\', "/"))
}

fn normalize_optional_language(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.to_lowercase())
}

fn normalize_optional_kind(value: &Option<String>) -> Option<String> {
    normalize_optional_string(value).map(|value| value.to_lowercase())
}

fn nullable_string_schema(description: &str) -> Value {
    json!({
        "description": description,
        "type": ["string", "null"]
    })
}

fn index_codebase_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Start background indexing for a configured scope. In worktree overlay mode, worktree scopes refresh only changed/new overlay files and never trigger an automatic full worktree scan.",
        "properties": {
            "scope": nullable_string_schema("Configured group id, canonical repo root, or worktree root. Defaults to the configured default group."),
            "force": {
                "type": "boolean",
                "default": false,
                "description": "Fully rebuild before indexing."
            },
            "splitter": {
                "type": "string",
                "enum": ["ast", "langchain"],
                "default": "ast",
                "description": "Chunking strategy."
            },
            "customExtensions": {
                "type": "array",
                "default": [],
                "description": "Additional dotted file extensions.",
                "items": {
                    "type": "string"
                }
            },
            "ignorePatterns": {
                "type": "array",
                "default": [],
                "description": "Additional ignore patterns.",
                "items": {
                    "type": "string"
                }
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_code_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Broader code discovery. Use search_symbols first for exact definitions. Treat snippets as discovery hints, then use search_text or get_file_outline once the file is known. Use prepare_edit_target only when the exact patch location is already known.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Natural-language, identifier, or path-like query."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Maximum hits."
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "semantic", "hybrid", "identifier", "path", null],
                "default": "auto",
                "description": "Search mode."
            },
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "file": nullable_string_schema("Repo-relative file path filter."),
            "dedupeByFile": {
                "type": "boolean",
                "default": true,
                "description": "Return one best hit per file."
            },
            "includeContent": {
                "type": "boolean",
                "default": true,
                "description": "Include compact discovery snippets. Use search_text or prepare_edit_target for authoritative follow-up."
            },
            "snippetChars": {
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "maximum": 1200,
                "default": 220,
                "description": "Approximate max discovery snippet characters. Keep this small unless you need broader preview context."
            },
            "includeDiagnostics": {
                "type": "boolean",
                "default": false,
                "description": "Include scores, plan, timestamps, and absolute repo paths."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_symbols_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Exact symbol and definition lookup. Prefer this before raw file reads when the definition name is known or suspected.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Definition name, container, or path-like symbol query."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 5,
                "description": "Maximum hits."
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "kind": nullable_string_schema("Symbol kind filter."),
            "container": nullable_string_schema("Container/module/class filter."),
            "includeSymbolId": {
                "type": "boolean",
                "default": false,
                "description": "Include symbolId only when you plan to hand it directly to prepare_edit_target."
            },
            "includeDiagnostics": {
                "type": "boolean",
                "default": false,
                "description": "Include scores, timestamps, file hashes, and absolute repo paths."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn search_text_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Exact literal confirmation after discovery. Use when the repo, file, or pathPrefix is already known. This is the MCP replacement for narrow shell rg when you need exact strings, identifiers, test names, or log lines.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Absolute configured repo root to narrow a group scope to one repo."),
            "query": {
                "type": "string",
                "description": "Exact literal text to match. Regex is not supported."
            },
            "limit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
                "description": "Maximum hits."
            },
            "file": nullable_string_schema("Repo-relative file path to search."),
            "pathPrefix": nullable_string_schema("Repo-relative path prefix to bound subtree search."),
            "language": nullable_string_schema("Normalized language filter."),
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "caseSensitive": {
                "type": "boolean",
                "default": true,
                "description": "Match case exactly."
            },
            "wholeWord": {
                "type": "boolean",
                "default": false,
                "description": "Require whole-word matches."
            },
            "contextLines": {
                "type": "integer",
                "format": "uint",
                "minimum": 0,
                "maximum": 12,
                "default": 1,
                "description": "Extra surrounding lines in each preview."
            }
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn prepare_edit_target_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Final pre-patch step only. Use after search_symbols, search_code, search_text, or get_file_outline has already identified the exact patch location. Not for overview or broad inspection. If it returns ambiguous or needsNarrowing, go back to search_text or get_file_outline.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Repo root or repo basename when disambiguating within a multi-repo scope."),
            "file": nullable_string_schema("Repo-relative file path. Required unless symbolId is provided. Also used with symbolName for symbol-based prep without carrying symbolId."),
            "symbolId": nullable_string_schema("Indexed symbol id to prepare for editing."),
            "symbolName": nullable_string_schema("Symbol name from search_symbols when preparing by symbol without symbolId."),
            "symbolKind": nullable_string_schema("Optional symbol kind from search_symbols to narrow symbolName matches."),
            "symbolContainer": nullable_string_schema("Optional container/module/class from search_symbols to narrow symbolName matches."),
            "lineHint": {
                "type": ["integer", "null"],
                "format": "uint64",
                "description": "1-based line hint used to narrow a known edit location or symbol result."
            },
            "query": nullable_string_schema("Exact literal text used to narrow a known edit target."),
            "occurrence": {
                "type": ["integer", "null"],
                "format": "uint",
                "minimum": 1,
                "description": "1-based occurrence selector for repeated literal matches when the exact occurrence is already known."
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn analyze_impact_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Find callers, transitive dependents, and affected tests for one Rust/TypeScript definition. Prefer symbolId from search_symbols(includeSymbolId=true); file+line avoids a separate lookup when the source location is already known.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Repo root or basename. Supply this when scope contains multiple repositories."),
            "symbolId": {"type": "string", "minLength": 1, "description": "Root symbol id from search_symbols(includeSymbolId=true)."},
            "file": {"type": "string", "minLength": 1, "description": "Repo-relative source file used with line."},
            "line": {"type": "integer", "format": "uint64", "minimum": 1},
            "maxDepth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            "maxNodes": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
            "includeTests": {"type": "boolean", "default": true},
            "minConfidence": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 650},
            "includePossible": {"type": "boolean", "default": false, "description": "Include confidence 300-649 candidates in separately labeled possible sections."},
            "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact", "description": "compact is optimized for agent decisions; full returns complete canonical metadata."},
            "maxEvidencePerNode": {"type": "integer", "minimum": 0, "maximum": 5, "default": 2, "description": "Maximum evidence edges retained per returned node in compact mode."}
        },
        "anyOf": [
            {"required": ["symbolId"]},
            {"required": ["file", "line"]}
        ]
    }).as_object().cloned().unwrap_or_default()
}

fn trace_path_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Trace the shortest highest-confidence directed dependency paths between two Rust/TypeScript definitions. Each endpoint accepts either a symbolId or a file+line selector.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Repo root or basename. Supply this when scope contains multiple repositories."),
            "fromSymbolId": {"type": "string", "minLength": 1, "description": "Source symbol id from search_symbols(includeSymbolId=true)."},
            "fromFile": {"type": "string", "minLength": 1, "description": "Repo-relative source file used with fromLine."},
            "fromLine": {"type": "integer", "format": "uint64", "minimum": 1},
            "toSymbolId": {"type": "string", "minLength": 1, "description": "Target symbol id from search_symbols(includeSymbolId=true)."},
            "toFile": {"type": "string", "minLength": 1, "description": "Repo-relative target file used with toLine."},
            "toLine": {"type": "integer", "format": "uint64", "minimum": 1},
            "maxDepth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
            "maxPaths": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            "minConfidence": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 650},
            "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact"},
            "maxEvidencePerNode": {"type": "integer", "minimum": 0, "maximum": 5, "default": 2}
        },
        "allOf": [
            {"anyOf": [{"required": ["fromSymbolId"]}, {"required": ["fromFile", "fromLine"]}]},
            {"anyOf": [{"required": ["toSymbolId"]}, {"required": ["toFile", "toLine"]}]}
        ]
    }).as_object().cloned().unwrap_or_default()
}

fn coverage_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Check whether relationship coverage is sufficient to trust an empty or incomplete graph result. Call this before concluding that no dependency exists.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "repo": nullable_string_schema("Optional repo root or basename within a group scope."),
            "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact"}
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn analyze_changes_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Classify Rust/TypeScript declaration changes from a validated Git base to the current working tree, then attach bounded impact evidence. Historical-to-historical comparisons are not supported.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root."),
            "repo": nullable_string_schema("Repo root or basename; required for multi-repo scopes."),
            "baseRef": {"type": "string", "default": "HEAD"},
            "includeUntracked": {"type": "boolean", "default": false},
            "maxDepth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            "maxNodes": {"type": "integer", "minimum": 1, "maximum": 250, "default": 50},
            "includeTests": {"type": "boolean", "default": true},
            "minConfidence": {"type": "integer", "minimum": 0, "maximum": 1000, "default": 650},
            "detail": {"type": "string", "enum": ["compact", "full"], "default": "compact"},
            "maxEvidencePerNode": {"type": "integer", "minimum": 0, "maximum": 5, "default": 2}
        }
    }).as_object().cloned().unwrap_or_default()
}

fn analyze_dead_code_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Perform a read-only Rust reachability census from executable/build roots, public APIs, dynamic registrations, feature gates, and tests. Unreachable cycles are grouped as strongly connected removal units. Results remain advisory until validate_removal passes.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root."),
            "repo": nullable_string_schema("Repo root or basename; required for multi-repo scopes."),
            "minConfidence": {"type": "integer", "minimum": 650, "maximum": 1000, "default": 650},
            "maxGroups": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 250},
            "includeTests": {"type": "boolean", "default": true},
            "includeRiskCandidates": {"type": "boolean", "default": true, "description": "Include public API, macro, feature-gated, plugin, serialization, and dynamic-dispatch candidates as blocked review items."}
        }
    }).as_object().cloned().unwrap_or_default()
}

fn explain_removal_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Explain one group id returned by analyze_dead_code or estimate_trim. Returns SCC members, exact declaration ranges, test-only reachability, possible incoming edges, risk markers, and coverage blockers.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root."),
            "repo": nullable_string_schema("Repo root or basename; required for multi-repo scopes."),
            "groupId": {"type": "string", "minLength": 1},
            "minConfidence": {"type": "integer", "minimum": 650, "maximum": 1000, "default": 650}
        },
        "required": ["groupId"]
    }).as_object().cloned().unwrap_or_default()
}

fn validate_removal_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Validate selected removal groups without editing the configured repository. The tool copies the working tree into temporary storage, verifies an unchanged compiler baseline, adaptively isolates passing groups when a batch fails, runs cargo check --workspace --locked --offline, and removes the temporary checkout.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root."),
            "repo": nullable_string_schema("Repo root or basename; required for multi-repo scopes."),
            "groupIds": {"type": "array", "minItems": 1, "maxItems": 100, "items": {"type": "string", "minLength": 1}},
            "minConfidence": {"type": "integer", "minimum": 650, "maximum": 1000, "default": 650},
            "allFeatures": {"type": "boolean", "default": false},
            "includeTests": {"type": "boolean", "default": false, "description": "Also compile test/example/bench targets. Production validation is the default because test-only groups intentionally have test callers that require migration."}
        },
        "required": ["groupIds"]
    }).as_object().cloned().unwrap_or_default()
}

fn estimate_trim_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Return bounded aggregate Rust trim totals. candidateLoc is advisory, cleanCandidateLoc excludes blocked and dynamic/external-risk groups, reviewRequiredLoc needs human review, excludedRiskLoc is not counted as removable, and compilerValidatedLoc is non-zero only when disposable validation passes.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root."),
            "repo": nullable_string_schema("Repo root or basename; required for multi-repo scopes."),
            "minConfidence": {"type": "integer", "minimum": 650, "maximum": 1000, "default": 650},
            "maxGroups": {"type": "integer", "minimum": 1, "maximum": 2000, "default": 250},
            "includeTests": {"type": "boolean", "default": true},
            "includeRiskCandidates": {"type": "boolean", "default": true},
            "validate": {"type": "boolean", "default": false, "description": "Compile a bounded set of blocker-free groups in one disposable checkout, adaptively isolating the passing subset when a batch fails."},
            "validationGroupLimit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25},
            "allFeatures": {"type": "boolean", "default": false}
        }
    }).as_object().cloned().unwrap_or_default()
}

fn graph_output_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Agent-ready graph result. Inspect status first, then consume evidence or follow nextActions.",
        "properties": {
            "status": {
                "type": "string",
                "description": "Machine-readable outcome such as ok, found, not_found, no_dependents, truncated, needs_index, invalid_base, invalid_selector, symbol_not_found, or unsupported."
            },
            "nextActions": {
                "type": "array",
                "description": "Bounded follow-up tool calls for recoverable or inconclusive outcomes.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string"},
                        "reason": {"type": "string"},
                        "arguments": {"type": "object"}
                    },
                    "required": ["tool", "reason", "arguments"]
                }
            }
        },
        "required": ["status"],
        "additionalProperties": true
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn explain_search_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Explain search_code query planning.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "query": {
                "type": "string",
                "description": "Query to explain."
            },
            "mode": {
                "type": ["string", "null"],
                "enum": ["auto", "semantic", "hybrid", "identifier", "path", null],
                "default": "auto",
                "description": "Search mode."
            },
            "extensionFilter": {
                "type": "array",
                "default": [],
                "description": "Dotted file extensions to include.",
                "items": {
                    "type": "string"
                }
            },
            "pathPrefix": nullable_string_schema("Repo-relative path prefix filter."),
            "language": nullable_string_schema("Normalized language filter."),
            "file": nullable_string_schema("Repo-relative file path filter.")
        },
        "required": ["query"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn get_file_outline_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Return a compact indexed symbol outline summary for a known file. Defaults to a top-level summary and is not a full symbol dump.",
        "properties": {
            "scope": nullable_string_schema("Configured group id or repo root. Defaults to the configured default group."),
            "file": {
                "type": "string",
                "description": "Repo-relative file path."
            },
            "detail": {
                "type": "string",
                "enum": ["summary", "compact", "full"],
                "default": "summary",
                "description": "Outline detail level. `summary` is top-level only, `compact` expands within a small budget, and `full` is opt-in but still capped."
            },
            "maxDepth": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 16,
                "default": 1,
                "description": "Maximum outline tree depth for `compact` or `full` detail."
            },
            "maxNodes": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 512,
                "default": 12,
                "description": "Maximum outline nodes returned across the file."
            },
            "topLevelLimit": {
                "type": "integer",
                "format": "uint",
                "minimum": 1,
                "maximum": 256,
                "default": 8,
                "description": "Maximum top-level declarations returned before truncation."
            }
        },
        "required": ["file"]
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn scope_args_schema(scope_description: &str) -> Map<String, Value> {
    json!({
        "type": "object",
        "properties": {
            "scope": nullable_string_schema(scope_description)
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

fn list_scopes_schema() -> Map<String, Value> {
    json!({
        "type": "object",
        "description": "Lists configured scopes.",
        "properties": {
            "includeRepos": {
                "type": "boolean",
                "default": false,
                "description": "Include repo paths for each scope."
            }
        }
    })
    .as_object()
    .cloned()
    .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{
        GraphDetail, INDEX_WORKER_STALE_AFTER, IndexCoordinatorState, OutlineCompactionOptions,
        OutlineDetail, PendingIndexRequest, SERVER_INSTRUCTIONS, SearchMode,
        analyze_changes_schema, analyze_dead_code_schema, analyze_impact_schema,
        compact_outline_response, compact_prepare_edit_target_response,
        compact_search_response_value, compact_symbol_search_response_value,
        compact_text_search_response_value, coverage_response_value, coverage_schema,
        dead_code_response_value, default_limit, enforce_loopback_bind, estimate_trim_schema,
        explain_removal_schema, get_file_outline_schema, graph_output_schema,
        impact_response_value, list_scopes_schema, listen_is_loopback, normalize_extension_filter,
        parse_search_mode, parse_splitter_kind, prepare_edit_target_schema,
        retryable_background_index_error, search_symbols_schema, search_text_schema,
        search_tool_success, tool_list, trace_path_schema, trace_response_status,
        validate_removal_schema,
    };
    use crate::engine::impact::{ImpactEvidence, ImpactNode, ImpactResponse, TracePathResponse};
    use crate::engine::relationships::{RelationKind, RepoRelationshipCoverage};
    use crate::engine::splitter::SplitterKind;
    use crate::engine::symbols::OutlineNode;
    use crate::engine::trim::{
        DeadCodeAnalysis, DeadCodeClassification, DeadCodeGroup, TrimTotals,
    };
    use crate::engine::{
        AnchorQuality, EditResolutionType, EditTargetAnchor, EditTargetReasonCode,
        EditTargetStatus, FileOutlineMatch, FileOutlineResponse, PrepareEditTargetResponse,
        RepoSearchError, SearchHit, SearchPlanSummary, SearchResponse, SymbolSearchHit,
        SymbolSearchResponse, TextSearchHit, TextSearchResponse,
    };
    use serde_json::{Value, json, to_value};
    use std::path::PathBuf;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    fn pending_request(path: &str) -> PendingIndexRequest {
        PendingIndexRequest {
            repo: PathBuf::from(path),
            force: false,
            explicit_refresh: false,
            splitter: SplitterKind::Ast,
            custom_extensions: Vec::new(),
            ignore_patterns: Vec::new(),
            snapshot_queued: true,
            retry_attempt: 0,
        }
    }

    #[test]
    fn search_limit_default_matches_node_contract() {
        assert_eq!(default_limit(), 5);
    }

    #[test]
    fn loopback_listeners_are_allowed_by_default() {
        assert!(listen_is_loopback("127.0.0.1:8765"));
        assert!(listen_is_loopback("[::1]:8765"));
        assert!(listen_is_loopback("localhost:8765"));
        assert!(enforce_loopback_bind("127.0.0.1:8765", false).is_ok());
    }

    #[test]
    fn remote_listener_requires_explicit_override() {
        let error = enforce_loopback_bind("0.0.0.0:8765", false).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("refusing to bind unauthenticated")
        );
        assert!(enforce_loopback_bind("0.0.0.0:8765", true).is_ok());
    }

    #[test]
    fn extension_filter_normalizes_and_deduplicates() {
        let normalized = normalize_extension_filter(&[
            ".rs".to_string(),
            " .py ".to_string(),
            ".rs".to_string(),
        ])
        .unwrap();

        assert_eq!(normalized, vec![".py".to_string(), ".rs".to_string()]);
    }

    #[test]
    fn extension_filter_rejects_invalid_values() {
        let error = normalize_extension_filter(&["rs".to_string()]).unwrap_err();
        assert!(error.to_string().contains("invalid extensionFilter entry"));
    }

    #[test]
    fn splitter_parser_rejects_unknown_values() {
        let error = parse_splitter_kind("bogus").unwrap_err();
        assert!(error.to_string().contains("invalid splitter"));
    }

    #[test]
    fn splitter_parser_accepts_supported_values() {
        assert!(matches!(
            parse_splitter_kind("ast").unwrap(),
            SplitterKind::Ast
        ));
        assert!(matches!(
            parse_splitter_kind("langchain").unwrap(),
            SplitterKind::LangChain
        ));
    }

    #[test]
    fn background_index_retries_transient_embedding_and_vector_failures() {
        assert!(retryable_background_index_error(
            "Ollama embeddings failed: Post http://127.0.0.1/tokenize: EOF"
        ));
        assert!(retryable_background_index_error(
            "Milvus API: error sending request: tcp connect error: Connection refused"
        ));
        assert!(!retryable_background_index_error(
            "Milvus API: schema mismatch"
        ));
    }

    #[test]
    fn search_mode_parser_defaults_to_auto() {
        assert!(matches!(parse_search_mode(None).unwrap(), SearchMode::Auto));
        assert!(matches!(
            parse_search_mode(Some("identifier")).unwrap(),
            SearchMode::Identifier
        ));
    }

    #[test]
    fn tool_list_includes_scope_discovery() {
        let tools = tool_list();
        assert!(tools.iter().any(|tool| tool.name == "list_scopes"));
        assert!(tools.iter().any(|tool| tool.name == "search_symbols"));
        assert!(tools.iter().any(|tool| tool.name == "search_text"));
        assert!(tools.iter().any(|tool| tool.name == "prepare_edit_target"));
        assert!(tools.iter().any(|tool| tool.name == "get_file_outline"));
        assert!(tools.iter().any(|tool| tool.name == "analyze_impact"));
        assert!(tools.iter().any(|tool| tool.name == "trace_path"));
        assert!(tools.iter().any(|tool| tool.name == "analyze_changes"));
        assert!(tools.iter().any(|tool| tool.name == "check_index_coverage"));
        assert!(tools.iter().any(|tool| tool.name == "analyze_dead_code"));
        assert!(tools.iter().any(|tool| tool.name == "explain_removal"));
        assert!(tools.iter().any(|tool| tool.name == "validate_removal"));
        assert!(tools.iter().any(|tool| tool.name == "estimate_trim"));
        assert!(tools.iter().any(|tool| tool.name == "explain_search"));
    }

    #[test]
    fn relationship_tool_schemas_publish_defaults_and_limits() {
        let impact = Value::Object(analyze_impact_schema());
        let trace = Value::Object(trace_path_schema());
        let changes = Value::Object(analyze_changes_schema());
        let coverage = Value::Object(coverage_schema());

        assert_eq!(impact["properties"]["maxDepth"]["default"], 2);
        assert_eq!(impact["properties"]["maxDepth"]["maximum"], 5);
        assert_eq!(impact["properties"]["maxNodes"]["maximum"], 250);
        assert_eq!(impact["properties"]["minConfidence"]["default"], 650);
        assert_eq!(impact["properties"]["detail"]["default"], "compact");
        assert_eq!(impact["properties"]["maxEvidencePerNode"]["maximum"], 5);
        assert_eq!(impact["anyOf"].as_array().map(Vec::len), Some(2));
        assert_eq!(trace["properties"]["maxDepth"]["default"], 5);
        assert_eq!(trace["properties"]["maxPaths"]["maximum"], 10);
        assert!(trace["properties"].get("fromFile").is_some());
        assert_eq!(trace["allOf"].as_array().map(Vec::len), Some(2));
        assert_eq!(changes["properties"]["baseRef"]["default"], "HEAD");
        assert_eq!(changes["properties"]["includeUntracked"]["default"], false);
        assert_eq!(changes["properties"]["detail"]["default"], "compact");
        assert_eq!(coverage["properties"]["detail"]["default"], "compact");
        assert!(
            coverage["description"]
                .as_str()
                .unwrap()
                .contains("coverage")
        );
    }

    #[test]
    fn relationship_tools_advertise_agent_status_output_schema() {
        let tools = tool_list();
        for name in [
            "analyze_impact",
            "trace_path",
            "analyze_changes",
            "check_index_coverage",
            "analyze_dead_code",
            "explain_removal",
            "validate_removal",
            "estimate_trim",
        ] {
            let tool = tools.iter().find(|tool| tool.name == name).unwrap();
            let schema = tool.output_schema.as_ref().expect("graph output schema");
            assert_eq!(schema["required"], json!(["status"]));
            assert!(schema["properties"].get("nextActions").is_some());
        }
        assert_eq!(graph_output_schema()["required"], json!(["status"]));
    }

    #[test]
    fn trim_tool_schemas_publish_safe_defaults_and_bounds() {
        let analyze = Value::Object(analyze_dead_code_schema());
        let explain = Value::Object(explain_removal_schema());
        let validate = Value::Object(validate_removal_schema());
        let estimate = Value::Object(estimate_trim_schema());

        assert_eq!(analyze["properties"]["minConfidence"]["minimum"], 650);
        assert_eq!(analyze["properties"]["maxGroups"]["maximum"], 2000);
        assert_eq!(
            analyze["properties"]["includeRiskCandidates"]["default"],
            true
        );
        assert_eq!(explain["required"], json!(["groupId"]));
        assert_eq!(validate["required"], json!(["groupIds"]));
        assert_eq!(validate["properties"]["includeTests"]["default"], false);
        assert!(
            validate["description"]
                .as_str()
                .unwrap()
                .contains("without editing")
        );
        assert_eq!(estimate["properties"]["validate"]["default"], false);
        assert_eq!(
            estimate["properties"]["validationGroupLimit"]["maximum"],
            100
        );
    }

    #[test]
    fn dead_code_response_is_compact_and_routes_agents_to_explain_and_validate() {
        let analysis = DeadCodeAnalysis {
            scope: "workspace".to_string(),
            repo: "/repo".to_string(),
            status: "ok".to_string(),
            coverage: RepoRelationshipCoverage {
                graph_status: "ready".to_string(),
                ..RepoRelationshipCoverage::default()
            },
            root_count: 1,
            roots_truncated: false,
            roots: Vec::new(),
            group_count: 1,
            groups: vec![DeadCodeGroup {
                group_id: "dead_fixture".to_string(),
                classification: DeadCodeClassification::Unreachable,
                confidence: 850,
                symbols: Vec::new(),
                removal_ranges: Vec::new(),
                source_loc: 12,
                compiler_validated_loc: 0,
                test_reference_count: 0,
                incoming_possible_count: 0,
                whole_files: vec!["src/dead.rs".to_string()],
                blockers: Vec::new(),
                rationale: vec!["unreachable".to_string()],
            }],
            totals: TrimTotals {
                candidate_loc: 12,
                clean_candidate_loc: 12,
                ..TrimTotals::default()
            },
            analysis_blockers: Vec::new(),
            truncated: false,
            diagnostic: "advisory".to_string(),
        };

        let value = dead_code_response_value(&analysis);
        assert_eq!(value["groupCount"], 1);
        assert_eq!(value["groups"][0]["groupId"], "dead_fixture");
        assert!(value["groups"][0].get("symbols").is_none());
        assert_eq!(value["nextActions"][0]["tool"], "explain_removal");
        assert_eq!(value["nextActions"][1]["tool"], "validate_removal");
    }

    fn impact_node(name: &str, role: &str, evidence_count: usize) -> ImpactNode {
        ImpactNode {
            logical_key: format!("key-{name}"),
            symbol_id: Some(format!("sym-{name}")),
            name: name.to_string(),
            kind: "function".to_string(),
            relative_path: format!("src/{name}.rs"),
            start_line: 10,
            end_line: 12,
            source_role: role.to_string(),
            depth: 1,
            score: 0.9,
            possible: false,
            evidence: (0..evidence_count)
                .map(|index| ImpactEvidence {
                    relation: RelationKind::Calls,
                    confidence: 900,
                    resolution: "same_module_unique".to_string(),
                    source_path: format!("src/caller_{index}.rs"),
                    start_line: 20 + index as u64,
                    end_line: 20 + index as u64,
                    text: format!("target_{index}()"),
                })
                .collect(),
        }
    }

    fn impact_response(direct: Vec<ImpactNode>) -> ImpactResponse {
        ImpactResponse {
            scope: "repo".to_string(),
            repo: "/repo".to_string(),
            root: impact_node("root", "production", 0),
            direct_dependents: direct,
            transitive_dependents: Vec::new(),
            affected_tests: Vec::new(),
            possible_dependents: Vec::new(),
            possible_tests: Vec::new(),
            ambiguities: Vec::new(),
            coverage: RepoRelationshipCoverage {
                repo: "/repo".to_string(),
                graph_status: "ready".to_string(),
                supported_files: 10,
                references: 20,
                definite: 15,
                probable: 2,
                resolution_percentage: 85.0,
                ..RepoRelationshipCoverage::default()
            },
            diagnostic: None,
            truncated: false,
            truncation_notices: Vec::new(),
        }
    }

    #[test]
    fn compact_impact_response_is_bounded_and_chainable() {
        let result = impact_response(vec![impact_node("caller", "production", 3)]);
        let value = impact_response_value(&result, GraphDetail::Compact, 1);
        assert_eq!(value["status"], "ok");
        assert_eq!(value["counts"]["direct"], 1);
        assert_eq!(value["directDependents"][0]["symbolId"], "sym-caller");
        assert_eq!(value["coverage"]["graphReady"], true);
        assert_eq!(value["coverage"]["liveAudit"], "verified");
        assert_eq!(value["coverage"]["conclusive"], true);
        assert_eq!(
            value["directDependents"][0]["evidence"]
                .as_array()
                .map(Vec::len),
            Some(1)
        );
        assert!(value["directDependents"][0].get("logicalKey").is_none());
    }

    #[test]
    fn truncated_trace_never_reports_not_found() {
        let node = impact_node("endpoint", "production", 0);
        let response = TracePathResponse {
            scope: "repo".to_string(),
            repo: "/repo".to_string(),
            from: node.clone(),
            to: node,
            found: false,
            paths: Vec::new(),
            ambiguities: Vec::new(),
            coverage: RepoRelationshipCoverage::default(),
            diagnostic: None,
            truncated: true,
            truncation_notices: vec!["bounded".to_string()],
        };
        assert_eq!(trace_response_status(&response), "truncated");
    }

    #[test]
    fn empty_impact_response_recommends_coverage_and_fallback_search() {
        let value = impact_response_value(&impact_response(Vec::new()), GraphDetail::Compact, 2);
        assert_eq!(value["status"], "no_dependents");
        assert_eq!(value["nextActions"][0]["tool"], "check_index_coverage");
        assert_eq!(value["nextActions"][1]["tool"], "search_code");
    }

    #[test]
    fn coverage_response_wraps_repositories_in_an_object() {
        let coverage = impact_response(Vec::new()).coverage;
        let value = coverage_response_value(&[coverage], GraphDetail::Compact, "repo");
        assert_eq!(value["status"], "ready");
        assert_eq!(value["repositoryCount"], 1);
        assert!(value["repositories"].is_array());
        assert_eq!(value["repositories"][0]["liveAudit"], "verified");
        assert_eq!(value["repositories"][0]["conclusive"], true);
    }

    #[test]
    fn tool_descriptions_route_agents_toward_symbol_search() {
        let tools = tool_list();
        let search_code = tools
            .iter()
            .find(|tool| tool.name == "search_code")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let search_symbols = tools
            .iter()
            .find(|tool| tool.name == "search_symbols")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let search_text = tools
            .iter()
            .find(|tool| tool.name == "search_text")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let prepare_edit_target = tools
            .iter()
            .find(|tool| tool.name == "prepare_edit_target")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();
        let list_scopes = tools
            .iter()
            .find(|tool| tool.name == "list_scopes")
            .and_then(|tool| tool.description.as_deref())
            .unwrap_or_default();

        assert!(search_code.contains("Snippets are discovery hints"));
        assert!(search_symbols.contains("definition name"));
        assert!(search_text.contains("instead of shell rg"));
        assert!(prepare_edit_target.contains("Final pre-patch step only"));
        assert!(list_scopes.contains("Preferred first call"));
    }

    #[test]
    fn tool_list_orders_navigation_before_prepare_edit() {
        let tools = tool_list();
        let names = tools
            .iter()
            .map(|tool| tool.name.as_ref())
            .collect::<Vec<_>>();
        let search_symbols = names
            .iter()
            .position(|name| *name == "search_symbols")
            .expect("search_symbols missing");
        let search_code = names
            .iter()
            .position(|name| *name == "search_code")
            .expect("search_code missing");
        let search_text = names
            .iter()
            .position(|name| *name == "search_text")
            .expect("search_text missing");
        let outline = names
            .iter()
            .position(|name| *name == "get_file_outline")
            .expect("get_file_outline missing");
        let prepare = names
            .iter()
            .position(|name| *name == "prepare_edit_target")
            .expect("prepare_edit_target missing");

        assert!(search_symbols < search_code);
        assert!(search_code < search_text);
        assert!(search_text < outline);
        assert!(outline < prepare);
    }

    #[test]
    fn search_tool_schema_advertises_limit_and_extension_filter_metadata() {
        let tools = tool_list();
        let search_tool = tools
            .iter()
            .find(|tool| tool.name == "search_code")
            .expect("search_code tool missing");
        let schema = Value::Object((*search_tool.input_schema).clone());

        assert_eq!(schema["properties"]["limit"]["default"].as_u64(), Some(5));
        assert_eq!(schema["properties"]["limit"]["maximum"].as_u64(), Some(50));
        assert_eq!(
            schema["properties"]["mode"]["default"].as_str(),
            Some("auto")
        );
        assert!(
            schema["properties"]["extensionFilter"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Dotted file extensions")
        );
        assert!(
            schema["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Treat snippets as discovery hints")
        );
        assert!(schema["properties"].get("path").is_none());
        assert_eq!(
            schema["properties"]["snippetChars"]["default"].as_u64(),
            Some(220)
        );
        assert!(
            schema["properties"]["includeContent"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("authoritative follow-up")
        );
    }

    #[test]
    fn index_tool_schema_advertises_splitter_enum() {
        let tools = tool_list();
        let index_tool = tools
            .iter()
            .find(|tool| tool.name == "index_codebase")
            .expect("index_codebase tool missing");
        let schema = Value::Object((*index_tool.input_schema).clone());
        let enum_values = schema["properties"]["splitter"]["enum"]
            .as_array()
            .expect("splitter enum missing")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>();

        assert_eq!(enum_values, vec!["ast", "langchain"]);
    }

    #[test]
    fn symbol_and_scope_schemas_advertise_preferred_usage() {
        let search_symbols = Value::Object(search_symbols_schema());
        let search_text = Value::Object(search_text_schema());
        let prepare = Value::Object(prepare_edit_target_schema());
        let list_scopes = Value::Object(list_scopes_schema());

        assert!(
            search_symbols["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Exact symbol and definition lookup")
        );
        assert!(
            search_symbols["properties"]["query"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Definition name")
        );
        assert_eq!(
            search_symbols["properties"]["includeSymbolId"]["default"].as_bool(),
            Some(false)
        );
        assert!(
            search_text["description"]
                .as_str()
                .unwrap_or_default()
                .contains("replacement for narrow shell rg")
        );
        assert!(
            search_text["description"]
                .as_str()
                .unwrap_or_default()
                .contains("repo, file, or pathPrefix")
        );
        assert!(
            prepare["properties"]["symbolName"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("search_symbols")
        );
        assert!(
            search_text["properties"]["repo"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("narrow a group scope")
        );
        assert_eq!(
            search_text["properties"]["contextLines"]["default"].as_u64(),
            Some(1)
        );
        assert!(
            prepare["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Final pre-patch step only")
        );
        assert!(
            search_text["properties"]["repo"]["description"]
                .as_str()
                .unwrap_or_default()
                .contains("narrow a group scope")
        );
        assert!(prepare["properties"].get("beforeLines").is_none());
        assert!(prepare["properties"].get("afterLines").is_none());
        assert!(prepare["properties"].get("maxLines").is_none());
        assert!(prepare["properties"].get("anchorCount").is_none());
        assert!(
            list_scopes["description"]
                .as_str()
                .unwrap_or_default()
                .contains("Lists configured scopes")
        );
        let outline = Value::Object(get_file_outline_schema());
        assert_eq!(
            outline["properties"]["detail"]["default"].as_str(),
            Some("summary")
        );
        assert_eq!(
            outline["properties"]["maxDepth"]["default"].as_u64(),
            Some(1)
        );
        assert_eq!(
            outline["properties"]["maxNodes"]["default"].as_u64(),
            Some(12)
        );
        assert_eq!(
            outline["properties"]["topLevelLimit"]["default"].as_u64(),
            Some(8)
        );
        assert!(SERVER_INSTRUCTIONS.contains("search_symbols for exact definitions"));
        assert!(SERVER_INSTRUCTIONS.contains("search_text for exact literals"));
        assert!(SERVER_INSTRUCTIONS.contains("instead of shell rg"));
        assert!(SERVER_INSTRUCTIONS.contains("final pre-patch step"));
    }

    #[test]
    fn compact_text_search_response_stays_small() {
        let response = TextSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![TextSearchHit {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 10,
                preview: "fn build() {}".to_string(),
                stale: false,
            }],
        };

        let json = compact_text_search_response_value(&response);
        assert!(json[0].get("repo").is_none());
        assert!(json[0].get("repoLabel").is_none());
        assert_eq!(json[0]["preview"], "fn build() {}");
    }

    #[test]
    fn compact_text_search_response_keeps_repo_label_for_multi_repo_hits() {
        let response = TextSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![
                TextSearchHit {
                    repo: "/tmp/repo-a".to_string(),
                    repo_label: "repo-a".to_string(),
                    relative_path: "src/lib.rs".to_string(),
                    start_line: 10,
                    end_line: 10,
                    preview: "fn build() {}".to_string(),
                    stale: false,
                },
                TextSearchHit {
                    repo: "/tmp/repo-b".to_string(),
                    repo_label: "repo-b".to_string(),
                    relative_path: "src/main.rs".to_string(),
                    start_line: 20,
                    end_line: 20,
                    preview: "fn main() {}".to_string(),
                    stale: false,
                },
            ],
        };

        let json = compact_text_search_response_value(&response);
        assert_eq!(json[0]["repoLabel"], "repo-a");
        assert_eq!(json[1]["repoLabel"], "repo-b");
    }

    #[test]
    fn search_tool_success_moves_bare_arrays_to_text_content() {
        let value = json!([{ "relativePath": "src/lib.rs", "line": 10 }]);

        let result = search_tool_success("summary".to_string(), value.clone());

        assert!(result.structured_content.is_none());
        let serialized = to_value(&result).unwrap();
        assert_eq!(
            serialized["content"][0]["text"].as_str(),
            Some(serde_json::to_string(&value).unwrap().as_str())
        );
    }

    #[test]
    fn search_tool_success_keeps_objects_as_structured_content() {
        let value = json!({
            "partial": true,
            "repoErrors": [{ "repo": "/tmp/repo", "error": "boom" }],
            "hits": []
        });

        let result = search_tool_success("summary".to_string(), value.clone());

        assert_eq!(result.structured_content, Some(value));
        let serialized = to_value(&result).unwrap();
        assert_eq!(serialized["content"][0]["text"].as_str(), Some("summary"));
    }

    #[test]
    fn compact_prepare_edit_target_response_omits_heavy_diagnostics() {
        let response = PrepareEditTargetResponse {
            status: EditTargetStatus::Ready,
            repo: Some("/tmp/repo".to_string()),
            repo_label: Some("repo".to_string()),
            relative_path: Some("src/lib.rs".to_string()),
            start_line: Some(10),
            end_line: Some(14),
            content: Some("fn build() {}\n".to_string()),
            anchors: vec![EditTargetAnchor {
                line: 10,
                text: "fn build() {".to_string(),
                unique_in_file: true,
            }],
            anchor_quality: Some(AnchorQuality::Strong),
            resolution_type: Some(EditResolutionType::Literal),
            file_hash: Some("abc".to_string()),
            indexed: Some(true),
            stale: Some(false),
            indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
            indexed_file_hash: Some("abc".to_string()),
            symbol_id: Some("sym_123".to_string()),
            truncated: Some(false),
            candidates: Vec::new(),
            symbol_start_line: None,
            symbol_end_line: None,
            reason_code: None,
            suggested_next_tool: None,
        };

        let json = to_value(compact_prepare_edit_target_response(&response)).unwrap();
        assert!(json.get("repoLabel").is_none());
        assert!(json.get("fileHash").is_none());
        assert!(json.get("indexedAt").is_none());
        assert!(json.get("indexedFileHash").is_none());
        assert!(json.get("stale").is_none());
        assert!(json.get("anchorQuality").is_none());
        assert!(json.get("resolutionType").is_none());
        assert!(json.get("unindexed").is_none());
        assert!(json.get("truncated").is_none());
        assert!(json.get("symbolStartLine").is_none());
        assert!(json.get("symbolEndLine").is_none());
        assert_eq!(json["anchors"][0]["text"], "fn build() {");
    }

    #[test]
    fn compact_prepare_edit_target_response_keeps_guidance_for_non_ready_states() {
        let response = PrepareEditTargetResponse {
            status: EditTargetStatus::NeedsNarrowing,
            repo: Some("/tmp/repo".to_string()),
            repo_label: Some("repo".to_string()),
            relative_path: Some("src/lib.rs".to_string()),
            start_line: None,
            end_line: None,
            content: None,
            anchors: Vec::new(),
            anchor_quality: None,
            resolution_type: Some(EditResolutionType::Symbol),
            file_hash: Some("abc".to_string()),
            indexed: Some(true),
            stale: Some(false),
            indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
            indexed_file_hash: Some("abc".to_string()),
            symbol_id: Some("sym_123".to_string()),
            truncated: Some(true),
            candidates: Vec::new(),
            symbol_start_line: Some(10),
            symbol_end_line: Some(80),
            reason_code: Some(EditTargetReasonCode::LargeSymbol),
            suggested_next_tool: Some("get_file_outline".to_string()),
        };

        let json = to_value(compact_prepare_edit_target_response(&response)).unwrap();
        assert_eq!(json["reasonCode"], "large_symbol");
        assert_eq!(json["suggestedNextTool"], "get_file_outline");
        assert!(json.get("content").is_none());
    }

    #[test]
    fn compact_search_response_omits_diagnostics_by_default() {
        let response = SearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            plan: SearchPlanSummary {
                requested_mode: "auto".to_string(),
                effective_mode: "hybrid".to_string(),
                query_kind: "mixed".to_string(),
                dense_weight: 1.0,
                lexical_weight: 1.0,
                symbol_weight: 1.0,
                symbol_lexical_share: 0.5,
                symbol_semantic_share: 0.5,
                dedupe_by_file: true,
            },
            hits: vec![SearchHit {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                start_line: 10,
                end_line: 12,
                language: "rust".to_string(),
                score: 0.42,
                match_type: "hybrid".to_string(),
                dense_score: Some(0.1),
                lexical_score: Some(0.2),
                symbol_score: Some(0.3),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                content: "fn example() {}".to_string(),
            }],
        };

        let value = compact_search_response_value(&response, false);
        let hit = &value[0];

        assert!(hit.get("repoLabel").is_none());
        assert!(hit.get("language").is_none());
        assert!(hit.get("matchType").is_none());
        assert!(hit.get("score").is_none());
        assert!(hit.get("denseScore").is_none());
        assert!(hit.get("indexedAt").is_none());
        assert!(hit.get("stale").is_none());
        assert_eq!(hit["relativePath"].as_str(), Some("src/lib.rs"));
        assert_eq!(hit["line"].as_u64(), Some(10));
        assert_eq!(hit["content"].as_str(), Some("fn example() {}"));
    }

    #[test]
    fn compact_search_response_keeps_repo_label_for_multi_repo_hits() {
        let response = SearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            plan: SearchPlanSummary {
                requested_mode: "auto".to_string(),
                effective_mode: "hybrid".to_string(),
                query_kind: "mixed".to_string(),
                dense_weight: 1.0,
                lexical_weight: 1.0,
                symbol_weight: 1.0,
                symbol_lexical_share: 0.5,
                symbol_semantic_share: 0.5,
                dedupe_by_file: true,
            },
            hits: vec![
                SearchHit {
                    repo: "/tmp/repo-a".to_string(),
                    repo_label: "repo-a".to_string(),
                    relative_path: "src/lib.rs".to_string(),
                    start_line: 10,
                    end_line: 12,
                    language: "rust".to_string(),
                    score: 0.42,
                    match_type: "hybrid".to_string(),
                    dense_score: Some(0.1),
                    lexical_score: Some(0.2),
                    symbol_score: Some(0.3),
                    indexed_at: None,
                    stale: false,
                    content: "fn example() {}".to_string(),
                },
                SearchHit {
                    repo: "/tmp/repo-b".to_string(),
                    repo_label: "repo-b".to_string(),
                    relative_path: "src/main.rs".to_string(),
                    start_line: 20,
                    end_line: 20,
                    language: "rust".to_string(),
                    score: 0.41,
                    match_type: "hybrid".to_string(),
                    dense_score: None,
                    lexical_score: None,
                    symbol_score: None,
                    indexed_at: None,
                    stale: false,
                    content: "fn main() {}".to_string(),
                },
            ],
        };

        let value = compact_search_response_value(&response, false);
        assert_eq!(value[0]["repoLabel"], "repo-a");
        assert_eq!(value[1]["repoLabel"], "repo-b");
    }

    #[test]
    fn compact_symbol_search_response_omits_repo_label_for_single_repo_hits() {
        let response = SymbolSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![SymbolSearchHit {
                symbol_id: "sym_123".to_string(),
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                name: "build".to_string(),
                kind: "function".to_string(),
                language: "rust".to_string(),
                container: None,
                start_line: 10,
                end_line: 12,
                score: 0.9,
                lexical_score: Some(0.4),
                semantic_score: Some(0.5),
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "abc".to_string(),
                stale: false,
            }],
        };

        let value = compact_symbol_search_response_value(&response, false, false);
        assert!(value[0].get("symbolId").is_none());
        assert!(value[0].get("repoLabel").is_none());
    }

    #[test]
    fn compact_symbol_search_response_keeps_repo_label_for_multi_repo_hits() {
        let response = SymbolSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![
                SymbolSearchHit {
                    symbol_id: "sym_123".to_string(),
                    repo: "/tmp/repo-a".to_string(),
                    repo_label: "repo-a".to_string(),
                    relative_path: "src/lib.rs".to_string(),
                    name: "build".to_string(),
                    kind: "function".to_string(),
                    language: "rust".to_string(),
                    container: None,
                    start_line: 10,
                    end_line: 12,
                    score: 0.9,
                    lexical_score: Some(0.4),
                    semantic_score: Some(0.5),
                    indexed_at: "2026-01-01T00:00:00Z".to_string(),
                    file_hash: "abc".to_string(),
                    stale: false,
                },
                SymbolSearchHit {
                    symbol_id: "sym_456".to_string(),
                    repo: "/tmp/repo-b".to_string(),
                    repo_label: "repo-b".to_string(),
                    relative_path: "src/main.rs".to_string(),
                    name: "main".to_string(),
                    kind: "function".to_string(),
                    language: "rust".to_string(),
                    container: None,
                    start_line: 20,
                    end_line: 22,
                    score: 0.8,
                    lexical_score: Some(0.3),
                    semantic_score: Some(0.5),
                    indexed_at: "2026-01-01T00:00:00Z".to_string(),
                    file_hash: "def".to_string(),
                    stale: false,
                },
            ],
        };

        let value = compact_symbol_search_response_value(&response, false, false);
        assert_eq!(value[0]["repoLabel"], "repo-a");
        assert_eq!(value[1]["repoLabel"], "repo-b");
    }

    #[test]
    fn compact_symbol_search_response_includes_symbol_id_when_requested() {
        let response = SymbolSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: false,
            repo_errors: Vec::new(),
            hits: vec![SymbolSearchHit {
                symbol_id: "sym_123".to_string(),
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                name: "build".to_string(),
                kind: "function".to_string(),
                language: "rust".to_string(),
                container: None,
                start_line: 10,
                end_line: 12,
                score: 0.9,
                lexical_score: Some(0.4),
                semantic_score: Some(0.5),
                indexed_at: "2026-01-01T00:00:00Z".to_string(),
                file_hash: "abc".to_string(),
                stale: false,
            }],
        };

        let value = compact_symbol_search_response_value(&response, true, false);
        assert_eq!(value[0]["symbolId"], "sym_123");
    }

    #[test]
    fn compact_search_response_keeps_box_when_metadata_is_present() {
        let response = SearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: true,
            repo_errors: vec![RepoSearchError {
                repo: "/tmp/repo".to_string(),
                error: "boom".to_string(),
            }],
            plan: SearchPlanSummary {
                requested_mode: "auto".to_string(),
                effective_mode: "hybrid".to_string(),
                query_kind: "mixed".to_string(),
                dense_weight: 1.0,
                lexical_weight: 1.0,
                symbol_weight: 1.0,
                symbol_lexical_share: 0.5,
                symbol_semantic_share: 0.5,
                dedupe_by_file: true,
            },
            hits: Vec::new(),
        };

        let value = compact_search_response_value(&response, true);
        assert!(value.is_object());
        assert_eq!(value["partial"], true);
        assert!(value.get("hits").is_some());
    }

    #[test]
    fn compact_text_search_response_keeps_box_when_metadata_is_present() {
        let response = TextSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: true,
            repo_errors: vec![RepoSearchError {
                repo: "/tmp/repo".to_string(),
                error: "boom".to_string(),
            }],
            hits: Vec::new(),
        };

        let value = compact_text_search_response_value(&response);
        assert!(value.is_object());
        assert_eq!(value["partial"], true);
        assert!(value.get("hits").is_some());
    }

    #[test]
    fn compact_symbol_search_response_keeps_box_when_metadata_is_present() {
        let response = SymbolSearchResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            partial: true,
            repo_errors: vec![RepoSearchError {
                repo: "/tmp/repo".to_string(),
                error: "boom".to_string(),
            }],
            hits: Vec::new(),
        };

        let value = compact_symbol_search_response_value(&response, false, false);
        assert!(value.is_object());
        assert_eq!(value["partial"], true);
        assert!(value.get("hits").is_some());
    }

    #[test]
    fn compact_outline_response_prunes_to_requested_depth() {
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                symbols: vec![OutlineNode {
                    symbol_id: "root".to_string(),
                    name: "Root".to_string(),
                    kind: "struct".to_string(),
                    container: None,
                    language: "rust".to_string(),
                    start_line: 1,
                    end_line: 20,
                    children: vec![OutlineNode {
                        symbol_id: "child".to_string(),
                        name: "child".to_string(),
                        kind: "method".to_string(),
                        container: Some("Root".to_string()),
                        language: "rust".to_string(),
                        start_line: 5,
                        end_line: 8,
                        children: vec![OutlineNode {
                            symbol_id: "grandchild".to_string(),
                            name: "grandchild".to_string(),
                            kind: "function".to_string(),
                            container: Some("child".to_string()),
                            language: "rust".to_string(),
                            start_line: 6,
                            end_line: 7,
                            children: Vec::new(),
                        }],
                    }],
                }],
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Compact,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        assert_eq!(compact.matches[0].symbols[0].children.len(), 1);
        assert!(
            compact.matches[0].symbols[0].children[0]
                .children
                .is_empty()
        );
        assert_eq!(compact.matches[0].symbols[0].line, 1);
        assert!(compact.matches[0].truncated);
    }

    #[test]
    fn compact_outline_response_defaults_to_summary_budget() {
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: Some("2026-01-01T00:00:00Z".to_string()),
                stale: false,
                symbols: vec![
                    OutlineNode {
                        symbol_id: "root".to_string(),
                        name: "Root".to_string(),
                        kind: "struct".to_string(),
                        container: None,
                        language: "rust".to_string(),
                        start_line: 1,
                        end_line: 20,
                        children: vec![OutlineNode {
                            symbol_id: "child".to_string(),
                            name: "child".to_string(),
                            kind: "method".to_string(),
                            container: Some("Root".to_string()),
                            language: "rust".to_string(),
                            start_line: 5,
                            end_line: 8,
                            children: Vec::new(),
                        }],
                    },
                    OutlineNode {
                        symbol_id: "other".to_string(),
                        name: "Other".to_string(),
                        kind: "enum".to_string(),
                        container: None,
                        language: "rust".to_string(),
                        start_line: 30,
                        end_line: 40,
                        children: Vec::new(),
                    },
                ],
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        let match_entry = &compact.matches[0];
        assert!(match_entry.repo.is_none());
        assert!(match_entry.truncated);
        assert!(match_entry.symbols[0].children.is_empty());
        assert_eq!(match_entry.symbols[0].line, 1);
    }

    #[test]
    fn compact_outline_response_caps_top_level_nodes() {
        let symbols = (0..4)
            .map(|index| OutlineNode {
                symbol_id: format!("sym_{index}"),
                name: format!("Item{index}"),
                kind: "function".to_string(),
                container: None,
                language: "rust".to_string(),
                start_line: index + 1,
                end_line: index + 1,
                children: Vec::new(),
            })
            .collect();
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![FileOutlineMatch {
                repo: "/tmp/repo".to_string(),
                repo_label: "repo".to_string(),
                relative_path: "src/lib.rs".to_string(),
                language: Some("rust".to_string()),
                indexed_at: None,
                stale: false,
                symbols,
            }],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 2,
            },
        );

        assert_eq!(compact.matches[0].symbols.len(), 2);
        assert_eq!(compact.matches[0].symbols[0].line, 1);
        assert!(compact.matches[0].truncated);
    }

    #[test]
    fn compact_outline_response_includes_repo_when_multiple_matches_exist() {
        let match_entry = FileOutlineMatch {
            repo: "/tmp/repo-a".to_string(),
            repo_label: "repo-a".to_string(),
            relative_path: "src/lib.rs".to_string(),
            language: Some("rust".to_string()),
            indexed_at: None,
            stale: false,
            symbols: vec![OutlineNode {
                symbol_id: "root".to_string(),
                name: "Root".to_string(),
                kind: "struct".to_string(),
                container: None,
                language: "rust".to_string(),
                start_line: 1,
                end_line: 10,
                children: Vec::new(),
            }],
        };
        let result = FileOutlineResponse {
            scope: "workspace".to_string(),
            label: "Workspace".to_string(),
            file: "src/lib.rs".to_string(),
            matches: vec![
                match_entry.clone(),
                FileOutlineMatch {
                    repo: "/tmp/repo-b".to_string(),
                    repo_label: "repo-b".to_string(),
                    ..match_entry
                },
            ],
        };

        let compact = compact_outline_response(
            result,
            OutlineCompactionOptions {
                detail: OutlineDetail::Summary,
                max_depth: 2,
                max_nodes: 64,
                top_level_limit: 32,
            },
        );

        assert_eq!(compact.matches[0].repo.as_deref(), Some("/tmp/repo-a"));
        assert_eq!(compact.matches[1].repo.as_deref(), Some("/tmp/repo-b"));
    }

    #[test]
    fn coordinator_starts_worker_for_pending_without_live_worker() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/repo".to_string(), pending_request("/tmp/repo"));

        let generation = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(generation, Some(1));
        assert_eq!(
            coordinator.worker.map(|worker| (
                worker.generation,
                worker.started_at,
                worker.last_heartbeat
            )),
            Some((1, now, now))
        );
    }

    #[test]
    fn coordinator_does_not_start_duplicate_fresh_worker() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/repo".to_string(), pending_request("/tmp/repo"));
        assert_eq!(coordinator.ensure_live_worker_for_pending(now), Some(1));

        let duplicate = coordinator.ensure_live_worker_for_pending(now + Duration::from_secs(1));

        assert_eq!(duplicate, None);
        assert_eq!(coordinator.worker.map(|worker| worker.generation), Some(1));
    }

    #[test]
    fn coordinator_replaces_stale_worker_for_pending_work() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/repo".to_string(), pending_request("/tmp/repo"));
        let first =
            coordinator.start_worker(now - INDEX_WORKER_STALE_AFTER - Duration::from_secs(1));
        coordinator
            .running
            .insert("/tmp/running".to_string(), first);

        let recovery = coordinator
            .begin_stale_worker_recovery(now)
            .expect("stale worker should enter recovery");

        assert_eq!(recovery.running_repos, vec!["/tmp/running".to_string()]);
        assert_eq!(recovery.age_secs, INDEX_WORKER_STALE_AFTER.as_secs() + 1);
        assert_eq!(coordinator.worker.map(|worker| worker.generation), None);
        assert!(coordinator.running.is_empty());
        assert!(coordinator.recovering.contains("/tmp/running"));

        coordinator.finish_recovery(&recovery.running_repos);
        let replacement = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(replacement, Some(2));
        assert_eq!(coordinator.worker.map(|worker| worker.generation), Some(2));
    }

    #[test]
    fn coordinator_keeps_stale_active_work_persisted_for_restart_recovery() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        let generation =
            coordinator.start_worker(now - INDEX_WORKER_STALE_AFTER - Duration::from_secs(1));
        coordinator
            .running
            .insert("/tmp/running".to_string(), generation);
        coordinator
            .active_requests
            .insert("/tmp/running".to_string(), pending_request("/tmp/running"));

        let recovery = coordinator
            .begin_stale_worker_recovery(now)
            .expect("stale worker should enter recovery");

        assert_eq!(recovery.running_repos, vec!["/tmp/running".to_string()]);
        assert!(
            coordinator
                .persisted_queue()
                .running
                .contains_key("/tmp/running")
        );
    }

    #[test]
    fn coordinator_keeps_fresh_running_worker_alive() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        let generation = coordinator.start_worker(now);
        coordinator
            .running
            .insert("/tmp/running".to_string(), generation);

        let recovery = coordinator.begin_stale_worker_recovery(now + Duration::from_secs(1));

        assert!(recovery.is_none());
        assert_eq!(
            coordinator.running.get("/tmp/running").copied(),
            Some(generation)
        );
        assert_eq!(
            coordinator.live_worker_generation(now + Duration::from_secs(1)),
            Some(generation)
        );
    }

    #[test]
    fn coordinator_recovers_stale_worker_with_only_pending_work() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/repo".to_string(), pending_request("/tmp/repo"));
        coordinator.start_worker(now - INDEX_WORKER_STALE_AFTER - Duration::from_secs(1));

        let recovery = coordinator
            .begin_stale_worker_recovery(now)
            .expect("stale worker should be replaced even without running repos");
        let replacement = coordinator.ensure_live_worker_for_pending(now);

        assert!(recovery.running_repos.is_empty());
        assert_eq!(replacement, Some(2));
    }

    #[test]
    fn coordinator_does_not_start_worker_for_unqueued_pending_snapshot() {
        let now = Instant::now();
        let mut request = pending_request("/tmp/repo");
        request.snapshot_queued = false;
        let mut coordinator = IndexCoordinatorState::default();
        coordinator.pending.insert("/tmp/repo".to_string(), request);

        let generation = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(generation, None);
        assert_eq!(
            coordinator.unqueued_ready_pending_repos(),
            vec!["/tmp/repo".to_string()]
        );

        coordinator.mark_pending_snapshot_queued(&["/tmp/repo".to_string()]);
        let generation = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(generation, Some(1));
    }

    #[test]
    fn coordinator_removes_only_unqueued_pending_after_mark_queued_failure() {
        let mut unqueued = pending_request("/tmp/unqueued");
        unqueued.snapshot_queued = false;
        let queued = pending_request("/tmp/queued");
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/unqueued".to_string(), unqueued);
        coordinator
            .pending
            .insert("/tmp/queued".to_string(), queued);

        coordinator
            .remove_unqueued_pending(&["/tmp/unqueued".to_string(), "/tmp/queued".to_string()]);

        assert!(!coordinator.pending.contains_key("/tmp/unqueued"));
        assert!(coordinator.pending.contains_key("/tmp/queued"));
    }

    #[test]
    fn coordinator_keeps_recovered_running_repo_blocked_until_finished() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/running".to_string(), pending_request("/tmp/running"));
        let generation =
            coordinator.start_worker(now - INDEX_WORKER_STALE_AFTER - Duration::from_secs(1));
        coordinator
            .running
            .insert("/tmp/running".to_string(), generation);

        let recovery = coordinator
            .begin_stale_worker_recovery(now)
            .expect("stale running repo should enter recovery");
        let replacement = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(replacement, None);
        assert!(coordinator.recovering.contains("/tmp/running"));

        coordinator.finish_recovery(&recovery.running_repos);
        let replacement = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(replacement, Some(2));
    }

    #[test]
    fn coordinator_merges_force_before_restarting_recovered_pending_worker() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .pending
            .insert("/tmp/repo".to_string(), pending_request("/tmp/repo"));
        coordinator.start_worker(now - INDEX_WORKER_STALE_AFTER - Duration::from_secs(1));

        let recovery = coordinator
            .begin_stale_worker_recovery(now)
            .expect("stale worker should be recovered before enqueue classification");
        assert!(recovery.running_repos.is_empty());

        let existing = coordinator
            .pending
            .get_mut("/tmp/repo")
            .expect("pending repo should still be mergeable");
        existing.force |= true;
        existing.explicit_refresh |= true;

        coordinator.finish_recovery(&recovery.running_repos);
        let generation = coordinator.ensure_live_worker_for_pending(now);

        assert_eq!(generation, Some(2));
        let request = coordinator
            .pending
            .get("/tmp/repo")
            .expect("worker has not drained pending request in this unit test");
        assert!(request.force);
        assert!(request.explicit_refresh);
    }

    #[test]
    fn coordinator_persists_active_work_separately_from_pending() {
        let mut coordinator = IndexCoordinatorState::default();
        let mut running = pending_request("/tmp/running");
        running.force = true;
        running.retry_attempt = 1;
        coordinator
            .active_requests
            .insert("/tmp/running".to_string(), running);
        coordinator
            .pending
            .insert("/tmp/pending".to_string(), pending_request("/tmp/pending"));

        let journal = coordinator.persisted_queue();

        assert!(journal.pending.contains_key("/tmp/pending"));
        let running = journal.running.get("/tmp/running").unwrap();
        assert!(running.force);
        assert_eq!(running.retry_attempt, 1);
    }

    #[test]
    fn coordinator_requeues_active_work_after_worker_abort() {
        let now = Instant::now();
        let mut coordinator = IndexCoordinatorState::default();
        let generation = coordinator.start_worker(now);
        let request = pending_request("/tmp/running");
        coordinator
            .running
            .insert("/tmp/running".to_string(), generation);
        coordinator
            .active_requests
            .insert("/tmp/running".to_string(), request);

        let recovered = coordinator
            .begin_worker_abort_recovery(generation)
            .expect("active worker should enter recovery");

        assert_eq!(recovered, vec!["/tmp/running".to_string()]);
        assert!(coordinator.running.is_empty());
        assert!(coordinator.active_requests.is_empty());
        assert!(!coordinator.pending["/tmp/running"].snapshot_queued);
    }

    #[tokio::test]
    async fn index_queue_store_round_trips_running_requests() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-context-index-queue-{nanos}"));
        let store = super::IndexQueueStore::new(&root.join("snapshot.json"));
        let mut coordinator = IndexCoordinatorState::default();
        coordinator
            .active_requests
            .insert("/tmp/running".to_string(), pending_request("/tmp/running"));

        store.write(&coordinator.persisted_queue()).await.unwrap();
        let restored = store.load().await.unwrap();

        assert!(restored.running.contains_key("/tmp/running"));
        tokio::fs::remove_dir_all(root).await.unwrap();
    }

    #[tokio::test]
    async fn independent_index_queue_stores_do_not_collide_on_temp_files() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("agent-context-index-queue-race-{nanos}"));
        let snapshot_path = root.join("snapshot.json");
        let first = super::IndexQueueStore::new(&snapshot_path);
        let second = super::IndexQueueStore::new(&snapshot_path);
        let mut first_coordinator = IndexCoordinatorState::default();
        first_coordinator
            .pending
            .insert("/tmp/first".to_string(), pending_request("/tmp/first"));
        let mut second_coordinator = IndexCoordinatorState::default();
        second_coordinator
            .pending
            .insert("/tmp/second".to_string(), pending_request("/tmp/second"));
        let first_queue = first_coordinator.persisted_queue();
        let second_queue = second_coordinator.persisted_queue();

        let (first_result, second_result) =
            tokio::join!(first.write(&first_queue), second.write(&second_queue));
        first_result.unwrap();
        second_result.unwrap();

        let restored = first.load().await.unwrap();
        assert!(
            restored.pending.contains_key("/tmp/first")
                || restored.pending.contains_key("/tmp/second")
        );
        tokio::fs::remove_dir_all(root).await.unwrap();
    }
}

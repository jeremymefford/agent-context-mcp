pub mod embedding;
pub mod freshness;
pub mod lexical;
pub mod live_files;
pub mod milvus;
pub mod splitter;
pub mod symbols;

use crate::config::{Config, ResolvedScope, ScopeKind, WorktreeMode, WorktreeResolution};
use crate::snapshot::{Snapshot, SnapshotEntry, SnapshotStore, WorktreeSnapshotEntry};
use anyhow::{Context, Result, bail};
use futures::StreamExt;
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::WalkBuilder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use xxhash_rust::xxh3::Xxh3;

use self::embedding::EmbeddingRegistry;
use self::freshness::{
    AuditFingerprint, apply_fingerprint, fingerprint_changed, fingerprint_repo,
    merkle_snapshot_path,
};
use self::lexical::{
    ChunkIndexDoc, ChunkSearchHit as LexicalChunkSearchHit, ChunkSearchRequest, LocalIndexStore,
    QueryFlavor, SymbolIndexDoc, SymbolSearchRequest,
};
use self::live_files::{LiveFileSnapshot, LiveFileStore, TextMatch};
use self::milvus::{MilvusClient, SearchDocument, VectorDocument};
use self::splitter::{
    CodeChunk, SplitterKind, default_supported_extensions, language_for_extension, split_text,
};
use self::symbols::{IndexedSymbol, OutlineNode, SymbolStore, build_outline, extract_symbols};

const EMBEDDING_BATCH_SIZE: usize = 64;
const SYMBOL_INDEX_REPLACEMENT_BATCH_SIZE: usize = 64;
const CHUNK_LIMIT: usize = 450_000;
const RRF_K: f64 = 100.0;
const CONTENT_HASH_ALGORITHM: &str = "xxh3_128";
const LEGACY_CONTENT_HASH_ALGORITHM: &str = "sha256";
const SNAPSHOT_PROGRESS_WRITE_INTERVAL: Duration = Duration::from_secs(2);
const INDEX_FORMAT_VERSION: &str = "v1";
const SEARCH_ROOT_VERSION: &str = "v1";
const CHUNK_COLLECTION_PREFIX: &str = "hybrid_code_chunks_";
const SYMBOL_COLLECTION_PREFIX: &str = "hybrid_symbols_";
const OVERLAY_CHUNK_COLLECTION_PREFIX: &str = "hybrid_code_overlay_";
const OVERLAY_SYMBOL_COLLECTION_PREFIX: &str = "hybrid_symbols_overlay_";
const OVERLAY_STORAGE_ROOT_PREFIX: &str = "/__agent_context_overlay";
const WORKTREE_EMBEDDING_INHERIT: &str = "inherit";
const VECTOR_FLUSH_FILE_CHANGE_THRESHOLD: u64 = 32;
const REMOVED_REPO_INDEX_STATUS: &str = "removed";
const LIVE_FILE_CACHE_LIMIT: usize = 64;
const SEARCH_TEXT_SHORTLIST_LIMIT: usize = 64;
const SEARCH_TEXT_FALLBACK_MAX_FILES: usize = 64;
const SEARCH_TEXT_PREVIEW_CHARS: usize = 180;
pub(crate) const PREPARE_READY_MAX_LINES: usize = 160;
const PREPARE_AMBIGUOUS_PREVIEW_CHARS: usize = 120;
const PREPARE_AMBIGUOUS_PREVIEW_BEFORE_LINES: u64 = 1;
const PREPARE_AMBIGUOUS_PREVIEW_AFTER_LINES: u64 = 1;
const EXPLICIT_REFRESH_MAX_FILE_PREPARE_PARALLELISM: usize = 8;

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
    embedding: EmbeddingRegistry,
    local_index: LocalIndexStore,
    live_files: LiveFileStore,
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
pub struct VectorHygieneReport {
    pub expected_collections: usize,
    pub agent_context_collections: Vec<String>,
    pub stale_collections: Vec<String>,
    pub loaded_collections: Vec<String>,
    pub recommended_loaded_collection_limit: usize,
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
pub struct TextSearchResponse {
    pub scope: String,
    pub label: String,
    pub partial: bool,
    pub repo_errors: Vec<RepoSearchError>,
    pub hits: Vec<TextSearchHit>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TextSearchHit {
    pub repo: String,
    pub repo_label: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub preview: String,
    pub stale: bool,
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
pub struct PrepareEditTargetResponse {
    pub status: EditTargetStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relative_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub anchors: Vec<EditTargetAnchor>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_quality: Option<AnchorQuality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution_type: Option<EditResolutionType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stale: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indexed_file_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<EditTargetCandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_start_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol_end_line: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<EditTargetReasonCode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_next_tool: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum EditTargetStatus {
    Ready,
    Ambiguous,
    NeedsNarrowing,
    NotFound,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditTargetReasonCode {
    WindowTooBroad,
    LargeSymbol,
    MultipleMatches,
    WeakAnchors,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EditResolutionType {
    Symbol,
    LineHint,
    Literal,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AnchorQuality {
    Strong,
    Weak,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EditTargetAnchor {
    pub line: u64,
    pub text: String,
    pub unique_in_file: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EditTargetCandidate {
    pub repo: String,
    pub repo_label: String,
    pub relative_path: String,
    pub start_line: u64,
    pub end_line: u64,
    pub preview: String,
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
    pub snippet_chars: usize,
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

#[derive(Debug, Clone)]
pub struct TextSearchScopeRequest {
    pub repo: Option<String>,
    pub query: String,
    pub limit: usize,
    pub path_prefix: Option<String>,
    pub language: Option<String>,
    pub file: Option<String>,
    pub extension_filter: Vec<String>,
    pub case_sensitive: bool,
    pub whole_word: bool,
    pub context_lines: usize,
}

#[derive(Debug, Clone)]
pub struct PrepareEditTargetRequest {
    pub repo: Option<String>,
    pub file: Option<String>,
    pub symbol_id: Option<String>,
    pub symbol_name: Option<String>,
    pub symbol_kind: Option<String>,
    pub symbol_container: Option<String>,
    pub line_hint: Option<u64>,
    pub query: Option<String>,
    pub occurrence: Option<usize>,
    pub before_lines: usize,
    pub after_lines: usize,
    pub max_lines: usize,
    pub anchor_count: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_profile: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configured_embedding_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stored_embedding_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_mismatch_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repo_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canonical_repo_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub changed_files: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deleted_files: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlay_mismatch_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct StaleIndexingRepo {
    pub repo: String,
    pub repo_label: String,
    pub index_status: Option<String>,
    pub last_updated: Option<String>,
    pub age_secs: Option<u64>,
}

#[derive(Debug, Clone)]
struct RepoFile {
    absolute_path: PathBuf,
    hash: String,
    bytes: u64,
}

#[derive(Debug, Clone, Default)]
struct FileDiff {
    added: Vec<String>,
    modified: Vec<String>,
    removed: Vec<String>,
}

#[derive(Debug, Clone)]
struct RepoContext {
    requested_root: PathBuf,
    canonical_root: PathBuf,
    overlay: Option<WorktreeOverlayContext>,
}

#[derive(Debug, Clone, Default)]
struct QueryProfileUsage {
    canonical_repos: BTreeSet<String>,
    overlay_repos: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct WorktreeOverlayContext {
    resolution: WorktreeResolution,
    storage_root: PathBuf,
    repo_key: String,
    chunk_collection: String,
    symbol_collection: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(rename_all = "camelCase")]
struct OverlayIndexState {
    canonical_root: String,
    worktree_root: String,
    repo_identity: String,
    overlay_id: String,
    #[serde(default)]
    replaced_paths: Vec<String>,
    #[serde(default)]
    deleted_paths: Vec<String>,
    #[serde(default)]
    indexed_hashes: Vec<(String, String)>,
    #[serde(default)]
    changed_files: u64,
    #[serde(default)]
    deleted_files: u64,
    #[serde(default)]
    overlay_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    overlay_status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    embedding_fingerprint: Option<String>,
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

type LiveFileRequestCache = HashMap<String, Arc<LiveFileSnapshot>>;

#[derive(Debug, Clone)]
struct IndexedFileMetadata {
    indexed_at: Option<String>,
    indexed_file_hash: Option<String>,
}

#[derive(Debug, Clone)]
struct ResolvedEditSymbol {
    repo: PathBuf,
    symbol: IndexedSymbol,
}

#[derive(Clone)]
struct ReadyEditTarget {
    snapshot: Arc<LiveFileSnapshot>,
    start_line: u64,
    end_line: u64,
    resolution_type: EditResolutionType,
    symbol_id: Option<String>,
    indexed_metadata: IndexedFileMetadata,
    truncated: bool,
    symbol_signature_line: Option<u64>,
    query: Option<String>,
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
struct PendingSymbolIndexReplacement {
    relative_path: String,
    documents: Vec<SymbolIndexDoc>,
}

#[derive(Debug, Clone)]
struct PreparedRepoFile {
    relative_path: String,
    basename: String,
    extension: String,
    indexed_at: String,
    file_hash: String,
    chunks: Vec<CodeChunk>,
    symbols: Vec<IndexedSymbol>,
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
    warnings: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct SymbolRepoSearchOutcome {
    repo: PathBuf,
    stale: bool,
    hits: Vec<RankedSymbolHit>,
    warnings: Vec<String>,
    error: Option<String>,
}

#[derive(Debug, Clone)]
struct SearchContextResult<T> {
    hits: Vec<T>,
    warnings: Vec<String>,
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

fn index_status_for_coverage(
    processing_status: IndexCompletionStatus,
    coverage: IndexCoverage,
) -> String {
    if coverage.indexed_files == 0 && coverage.total_chunks == 0 {
        "empty".to_string()
    } else if processing_status == IndexCompletionStatus::LimitReached {
        IndexCompletionStatus::LimitReached.as_str().to_string()
    } else {
        IndexCompletionStatus::Completed.as_str().to_string()
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
    status: IndexCompletionStatus,
}

#[derive(Debug, Clone, Copy)]
struct IndexCoverage {
    indexed_files: u64,
    total_chunks: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexIdentityStatus {
    pub compatible: bool,
    pub index_format_version: String,
    pub search_root_version: String,
    pub configured_embedding_fingerprints: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone)]
struct RepoEmbeddingIdentityStatus {
    profile_name: String,
    configured_fingerprint: Option<String>,
    stored_fingerprint: Option<String>,
    reason: Option<String>,
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

#[derive(Clone, Copy)]
struct ProcessFilesPlan<'a> {
    repo: &'a Path,
    storage_repo: &'a Path,
    repo_key: &'a str,
    profile_name: &'a str,
    collections: IndexCollections<'a>,
    splitter: SplitterKind,
    total_files: usize,
    mode: IndexExecutionMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexExecutionMode {
    Standard,
    ExplicitRefresh,
}

impl IndexExecutionMode {
    fn file_prepare_parallelism(self) -> usize {
        match self {
            Self::Standard => 1,
            Self::ExplicitRefresh => std::thread::available_parallelism()
                .map(|count| {
                    count
                        .get()
                        .clamp(2, EXPLICIT_REFRESH_MAX_FILE_PREPARE_PARALLELISM)
                })
                .unwrap_or(4),
        }
    }
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
        let embedding = EmbeddingRegistry::new(&config.embedding).await?;
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
                live_files: LiveFileStore::new(LIVE_FILE_CACHE_LIMIT),
                symbol_store: SymbolStore::new(config.symbol_db_path()),
                search_budgets: SearchBudgets::new(&config.search),
            }),
        })
    }

    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    fn embedding_profile_name_for_repo<'a>(&'a self, repo: &Path) -> Result<&'a str> {
        self.inner.config.embedding.profile_name_for_repo(repo)
    }

    async fn embedding_fingerprint_for_repo(&self, repo: &Path) -> Result<String> {
        let profile_name = self.embedding_profile_name_for_repo(repo)?;
        self.inner.embedding.fingerprint(profile_name).await
    }

    async fn embedding_dimension_for_repo(&self, repo: &Path) -> Result<usize> {
        let profile_name = self.embedding_profile_name_for_repo(repo)?;
        self.inner.embedding.dimension(profile_name).await
    }

    fn repo_context(&self, repo: &Path) -> Result<RepoContext> {
        if self.inner.config.worktrees.mode == WorktreeMode::Overlay
            && let Some(resolution) = self.inner.config.worktree_resolution(repo)?
        {
            let overlay_id = resolution.overlay_id.clone();
            return Ok(RepoContext {
                requested_root: resolution.worktree_root.clone(),
                canonical_root: resolution.canonical_root.clone(),
                overlay: Some(WorktreeOverlayContext {
                    resolution,
                    storage_root: overlay_storage_root(&overlay_id),
                    repo_key: overlay_repo_key(&overlay_id),
                    chunk_collection: overlay_collection_name(&overlay_id),
                    symbol_collection: overlay_symbol_collection_name(&overlay_id),
                }),
            });
        }

        Ok(RepoContext {
            requested_root: repo.to_path_buf(),
            canonical_root: repo.to_path_buf(),
            overlay: None,
        })
    }

    fn overlay_profile_name<'a>(&'a self, ctx: &'a RepoContext) -> Result<&'a str> {
        if ctx.overlay.is_none() {
            return self.embedding_profile_name_for_repo(&ctx.canonical_root);
        }
        let configured = self.inner.config.worktrees.embedding_profile.as_str();
        if configured == WORKTREE_EMBEDDING_INHERIT {
            self.embedding_profile_name_for_repo(&ctx.canonical_root)
        } else {
            Ok(configured)
        }
    }

    async fn overlay_fingerprint(&self, ctx: &RepoContext) -> Result<String> {
        let profile_name = self.overlay_profile_name(ctx)?;
        self.inner.embedding.fingerprint(profile_name).await
    }

    async fn overlay_dimension(&self, ctx: &RepoContext) -> Result<usize> {
        let profile_name = self.overlay_profile_name(ctx)?;
        self.inner.embedding.dimension(profile_name).await
    }

    fn overlay_state_path(&self, overlay_id: &str) -> PathBuf {
        self.inner
            .config
            .index_root
            .join("overlays")
            .join(format!("{overlay_id}.json"))
    }

    async fn load_overlay_state(
        &self,
        overlay: &WorktreeOverlayContext,
    ) -> Result<Option<OverlayIndexState>> {
        self.load_overlay_state_by_id(&overlay.resolution.overlay_id)
            .await
    }

    async fn load_overlay_state_by_id(
        &self,
        overlay_id: &str,
    ) -> Result<Option<OverlayIndexState>> {
        let path = self.overlay_state_path(overlay_id);
        if !path.exists() {
            return Ok(None);
        }
        let raw = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("reading overlay state {}", path.display()))?;
        let state = serde_json::from_str(&raw)
            .with_context(|| format!("parsing overlay state {}", path.display()))?;
        Ok(Some(state))
    }

    async fn save_overlay_state(&self, state: &OverlayIndexState) -> Result<()> {
        let path = self.overlay_state_path(&state.overlay_id);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("creating overlay state dir {}", parent.display()))?;
        }
        let raw = serde_json::to_string_pretty(state)?;
        tokio::fs::write(&path, raw)
            .await
            .with_context(|| format!("writing overlay state {}", path.display()))
    }

    async fn remove_overlay_state_if_present(&self, overlay_id: &str) -> Result<()> {
        let path = self.overlay_state_path(overlay_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| format!("removing {}", path.display())),
        }
    }

    async fn configured_embedding_fingerprints(&self) -> Result<BTreeMap<String, String>> {
        let mut fingerprints = BTreeMap::new();
        for profile_name in self.inner.config.embedding.profiles().map(|(name, _)| name) {
            fingerprints.insert(
                profile_name.clone(),
                self.inner.embedding.fingerprint(profile_name).await?,
            );
        }
        Ok(fingerprints)
    }

    async fn repo_embedding_identity_status(
        &self,
        snapshot: &Snapshot,
        repo: &Path,
    ) -> Result<RepoEmbeddingIdentityStatus> {
        let repo_key = repo.display().to_string();
        let entry = snapshot.codebases.get(&repo_key);
        let profile_name = self.embedding_profile_name_for_repo(repo)?.to_string();
        let configured_fingerprint = self.inner.embedding.fingerprint(&profile_name).await.ok();
        Ok(repo_embedding_identity_status_for_snapshot(
            snapshot,
            entry,
            &profile_name,
            configured_fingerprint,
        ))
    }

    pub async fn healthcheck(&self) -> Result<()> {
        self.inner.milvus.healthcheck().await?;
        let _ = self.configured_embedding_fingerprints().await?;
        Ok(())
    }

    pub async fn index_identity_status(&self) -> Result<IndexIdentityStatus> {
        let snapshot = self.inner.snapshot.read().await?;
        let configured_embedding_fingerprints = self.configured_embedding_fingerprints().await?;
        Ok(index_identity_status_for_snapshot(
            &snapshot,
            self.inner.config.embedding.default_profile_name(),
            &configured_embedding_fingerprints,
        ))
    }

    pub async fn vector_hygiene_report(&self) -> Result<VectorHygieneReport> {
        let repos = self.inner.config.all_repos()?;
        let snapshot = self.inner.snapshot.read().await?;
        let expected = expected_vector_collections(&repos, &snapshot);
        let mut agent_context_collections = self
            .inner
            .milvus
            .list_collections()
            .await?
            .into_iter()
            .filter(|collection| is_agent_context_vector_collection(collection))
            .collect::<Vec<_>>();
        agent_context_collections.sort();

        let stale_collections = stale_vector_collections(&agent_context_collections, &expected);
        let mut loaded_collections = Vec::new();
        for collection in &agent_context_collections {
            if self
                .inner
                .milvus
                .collection_load_state(collection)
                .await?
                .as_deref()
                == Some("LoadStateLoaded")
            {
                loaded_collections.push(collection.clone());
            }
        }

        Ok(VectorHygieneReport {
            expected_collections: expected.len(),
            agent_context_collections,
            stale_collections,
            loaded_collections,
            recommended_loaded_collection_limit: self
                .inner
                .config
                .search
                .max_warm_repos
                .saturating_mul(2),
        })
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

    pub async fn stale_indexing_repos(
        &self,
        stale_after: Duration,
    ) -> Result<Vec<StaleIndexingRepo>> {
        let snapshot = self.inner.snapshot.read().await?;
        let now = chrono::Utc::now();
        let mut stale = Vec::new();

        for (repo, entry) in snapshot.codebases {
            if entry.status != "indexing" {
                continue;
            }
            let age_secs = snapshot_entry_age_secs(entry.last_updated.as_deref(), now);
            if age_secs.is_some_and(|age| age < stale_after.as_secs()) {
                continue;
            }
            stale.push(StaleIndexingRepo {
                repo: repo.clone(),
                repo_label: repo_basename(&repo),
                index_status: entry.index_status,
                last_updated: entry.last_updated,
                age_secs,
            });
        }

        for (repo, entry) in snapshot.worktrees {
            if entry.status != "indexing" {
                continue;
            }
            let age_secs = snapshot_entry_age_secs(entry.last_updated.as_deref(), now);
            if age_secs.is_some_and(|age| age < stale_after.as_secs()) {
                continue;
            }
            stale.push(StaleIndexingRepo {
                repo: repo.clone(),
                repo_label: repo_basename(&repo),
                index_status: entry.overlay_status,
                last_updated: entry.last_updated,
                age_secs,
            });
        }

        stale.sort_by(|left, right| left.repo_label.cmp(&right.repo_label));
        Ok(stale)
    }

    pub async fn mark_scope_indexing_failed(
        &self,
        scope: &ResolvedScope,
        reason: &str,
    ) -> Result<()> {
        for repo in &scope.repos {
            let ctx = self.repo_context(repo)?;
            if let Some(overlay) = ctx.overlay.as_ref() {
                let worktree_key = overlay.resolution.worktree_root.display().to_string();
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        if let Some(entry) = snapshot.worktrees.get_mut(&worktree_key)
                            && entry.status == "indexing"
                        {
                            entry.status = "indexfailed".to_string();
                            entry.overlay_status = Some("failed".to_string());
                            entry.overlay_mismatch_reason = Some(reason.to_string());
                            entry.last_updated = Some(crate::snapshot::timestamp());
                        }
                    })
                    .await?;
                continue;
            }

            let repo_key = repo.display().to_string();
            self.inner
                .snapshot
                .update(|snapshot| {
                    if let Some(entry) = snapshot.codebases.get_mut(&repo_key)
                        && entry.status == "indexing"
                    {
                        *entry = SnapshotEntry::failed(
                            reason.to_string(),
                            entry
                                .indexing_percentage
                                .or(entry.last_attempted_percentage),
                            entry.embedding_profile.clone(),
                            entry.embedding_fingerprint.clone(),
                        );
                    }
                })
                .await?;
        }
        Ok(())
    }

    pub async fn mark_scope_indexing(&self, scope: &ResolvedScope) -> Result<()> {
        for repo in &scope.repos {
            let ctx = self.repo_context(repo)?;
            if let Some(overlay) = ctx.overlay.as_ref() {
                let profile_name = self.overlay_profile_name(&ctx).ok().map(str::to_string);
                let embedding_fingerprint = self.overlay_fingerprint(&ctx).await.ok();
                let worktree_key = overlay.resolution.worktree_root.display().to_string();
                let canonical_root = overlay.resolution.canonical_root.display().to_string();
                let repo_identity = overlay.resolution.repo_identity.clone();
                let overlay_id = overlay.resolution.overlay_id.clone();
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        snapshot.worktrees.insert(
                            worktree_key.clone(),
                            WorktreeSnapshotEntry::indexing(
                                canonical_root.clone(),
                                repo_identity.clone(),
                                overlay_id.clone(),
                                profile_name.clone(),
                                embedding_fingerprint.clone(),
                            ),
                        );
                    })
                    .await?;
                continue;
            }
            let repo_key = repo.display().to_string();
            let profile_name = self
                .embedding_profile_name_for_repo(repo)
                .ok()
                .map(str::to_string);
            self.inner
                .snapshot
                .update(|snapshot| {
                    snapshot.codebases.insert(
                        repo_key.clone(),
                        SnapshotEntry::indexing(0.0, "queued", profile_name.clone(), None),
                    );
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
        self.index_scope_with_mode(
            scope,
            force,
            splitter,
            custom_extensions,
            ignore_patterns,
            IndexExecutionMode::Standard,
        )
        .await
    }

    #[allow(dead_code)]
    pub async fn index_scope_explicit_refresh(
        &self,
        scope: ResolvedScope,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
    ) -> Result<ScopeIndexResult> {
        self.index_scope_with_mode(
            scope,
            force,
            splitter,
            custom_extensions,
            ignore_patterns,
            IndexExecutionMode::ExplicitRefresh,
        )
        .await
    }

    pub async fn index_scope_background(
        &self,
        scope: ResolvedScope,
        force: bool,
        explicit_refresh: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
    ) -> Result<ScopeIndexResult> {
        self.index_scope_with_mode(
            scope,
            force,
            splitter,
            custom_extensions,
            ignore_patterns,
            if explicit_refresh {
                IndexExecutionMode::ExplicitRefresh
            } else {
                IndexExecutionMode::Standard
            },
        )
        .await
    }

    async fn index_scope_with_mode(
        &self,
        scope: ResolvedScope,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
        mode: IndexExecutionMode,
    ) -> Result<ScopeIndexResult> {
        let mut results = Vec::new();
        let mut has_errors = false;
        let mut existing_repos = Vec::new();

        for repo in scope.repos {
            if repo.is_absolute() && !repo.exists() {
                match self.clear_removed_worktree_overlay(&repo, force).await {
                    Ok(Some(result)) => results.push(result),
                    Ok(None) => match self.clear_removed_repo(&repo, force).await {
                        Ok(result) => results.push(result),
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
                    },
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
            } else {
                existing_repos.push(repo);
            }
        }

        if !existing_repos.is_empty() {
            self.ensure_index_identity(force).await?;
            self.persist_index_identity().await?;
        }

        for repo in existing_repos {
            match self
                .index_repo(
                    &repo,
                    force,
                    splitter,
                    custom_extensions,
                    ignore_patterns,
                    mode,
                )
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

    async fn query_profile_usages_for_contexts(
        &self,
        scope_contexts: &[RepoContext],
        blocked_repos: &HashMap<String, String>,
    ) -> Result<BTreeMap<String, QueryProfileUsage>> {
        let mut usages = BTreeMap::<String, QueryProfileUsage>::new();
        for ctx in scope_contexts {
            let repo_key = ctx.requested_root.display().to_string();
            if blocked_repos.contains_key(&repo_key) {
                continue;
            }
            let canonical_profile = self
                .embedding_profile_name_for_repo(&ctx.canonical_root)?
                .to_string();
            usages
                .entry(canonical_profile)
                .or_default()
                .canonical_repos
                .insert(repo_key.clone());

            let Some(overlay) = ctx.overlay.as_ref() else {
                continue;
            };
            let overlay_searchable = match self.load_overlay_state(overlay).await {
                Ok(Some(state)) => overlay_state_has_search_index(&state),
                Ok(None) | Err(_) => false,
            };
            if overlay_searchable {
                usages
                    .entry(self.overlay_profile_name(ctx)?.to_string())
                    .or_default()
                    .overlay_repos
                    .insert(repo_key);
            }
        }
        Ok(usages)
    }

    async fn embed_query_vectors_for_usages(
        &self,
        profile_usages: BTreeMap<String, QueryProfileUsage>,
        query: &str,
        blocked_repos: &mut HashMap<String, String>,
    ) -> Result<(HashMap<String, Arc<Vec<f32>>>, HashMap<String, String>)> {
        let mut query_vectors = HashMap::new();
        let mut overlay_query_errors = HashMap::new();
        for (profile_name, usage) in profile_usages {
            let _dense_permit = self.acquire_dense_budget().await?;
            match self.inner.embedding.embed_query(&profile_name, query).await {
                Ok(vector) => {
                    query_vectors.insert(profile_name, Arc::new(vector));
                }
                Err(error) => {
                    let message = error.to_string();
                    apply_query_embedding_failure(
                        &profile_name,
                        &usage,
                        &message,
                        blocked_repos,
                        &mut overlay_query_errors,
                    );
                }
            }
        }
        Ok((query_vectors, overlay_query_errors))
    }

    pub async fn search_scope(
        &self,
        scope: ResolvedScope,
        request: SearchRequest,
    ) -> Result<SearchResponse> {
        let _request_permit = self.acquire_request_budget().await?;
        let plan = plan_search(&request);
        let per_repo_limit = (request.limit.max(5) * 4).min(64);
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let filter_expression = build_extension_filter(&request.extension_filter);
        let snapshot = self.inner.snapshot.read().await?;
        let scope_contexts = scope
            .repos
            .iter()
            .map(|repo| self.repo_context(repo))
            .collect::<Result<Vec<_>>>()?;
        let mut blocked_repos = HashMap::new();
        let mut query_vectors: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
        let mut overlay_query_errors: HashMap<String, String> = HashMap::new();

        for ctx in &scope_contexts {
            let identity = self
                .repo_embedding_identity_status(&snapshot, &ctx.canonical_root)
                .await?;
            if let Some(reason) = identity.reason {
                blocked_repos.insert(ctx.requested_root.display().to_string(), reason);
            }
        }

        if plan.dense_weight > 0.0 || plan.symbol_semantic_share > 0.0 {
            let usages = self
                .query_profile_usages_for_contexts(&scope_contexts, &blocked_repos)
                .await?;
            let (vectors, errors) = self
                .embed_query_vectors_for_usages(usages, &request.query, &mut blocked_repos)
                .await?;
            query_vectors = vectors;
            overlay_query_errors = errors;
        }

        let mut stream = futures::stream::iter(scope_contexts.into_iter().map(|ctx| {
            let engine = self.clone();
            let request = request.clone();
            let plan = plan.clone();
            let canonical_query_vector = engine
                .embedding_profile_name_for_repo(&ctx.canonical_root)
                .ok()
                .and_then(|profile| query_vectors.get(profile).cloned());
            let overlay_query_vector = ctx.overlay.as_ref().and_then(|_| {
                engine
                    .overlay_profile_name(&ctx)
                    .ok()
                    .and_then(|profile| query_vectors.get(profile).cloned())
            });
            let repo_error = blocked_repos
                .get(&ctx.requested_root.display().to_string())
                .cloned();
            let filter_expression = filter_expression.clone();
            let entry = snapshot
                .codebases
                .get(&ctx.canonical_root.display().to_string())
                .cloned();
            async move {
                let repo_label = ctx.requested_root.display().to_string();
                if let Some(error) = repo_error {
                    return RepoSearchOutcome {
                        repo: repo_label,
                        stale: false,
                        hits: Vec::new(),
                        warnings: Vec::new(),
                        error: Some(error),
                    };
                }
                let _repo_permit = match engine.acquire_repo_search_budget().await {
                    Ok(permit) => permit,
                    Err(error) => {
                        return RepoSearchOutcome {
                            repo: repo_label,
                            stale: false,
                            hits: Vec::new(),
                            warnings: Vec::new(),
                            error: Some(error.to_string()),
                        };
                    }
                };
                let stale = engine
                    .repo_is_stale(&ctx.requested_root, entry.as_ref())
                    .await
                    .unwrap_or(false);
                match engine
                    .search_repo_context(
                        &ctx,
                        &request,
                        &plan,
                        canonical_query_vector
                            .as_ref()
                            .map(|value| value.as_slice()),
                        overlay_query_vector.as_ref().map(|value| value.as_slice()),
                        per_repo_limit,
                        filter_expression.as_deref(),
                    )
                    .await
                {
                    Ok(result) => RepoSearchOutcome {
                        repo: repo_label,
                        stale,
                        hits: result.hits,
                        warnings: result.warnings,
                        error: None,
                    },
                    Err(error) => RepoSearchOutcome {
                        repo: repo_label,
                        stale,
                        hits: Vec::new(),
                        warnings: Vec::new(),
                        error: Some(error.to_string()),
                    },
                }
            }
        }))
        .buffer_unordered(parallelism);

        let mut repo_errors = overlay_query_errors
            .into_iter()
            .map(|(repo, error)| RepoSearchError { repo, error })
            .collect::<Vec<_>>();
        let mut repo_error_repos = repo_errors
            .iter()
            .map(|error| error.repo.clone())
            .collect::<HashSet<_>>();
        let mut merged = Vec::new();
        while let Some(outcome) = stream.next().await {
            if let Some(error) = outcome.error {
                if repo_error_repos.insert(outcome.repo.clone()) {
                    repo_errors.push(RepoSearchError {
                        repo: outcome.repo,
                        error,
                    });
                }
                continue;
            }
            for warning in outcome.warnings {
                if repo_error_repos.insert(outcome.repo.clone()) {
                    repo_errors.push(RepoSearchError {
                        repo: outcome.repo.clone(),
                        error: warning,
                    });
                }
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
        let _request_permit = self.acquire_request_budget().await?;
        let flavor = classify_query(&request.query);
        let (lexical_share, semantic_share) = symbol_query_shares(flavor);
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let scope_contexts = scope
            .repos
            .iter()
            .map(|repo| self.repo_context(repo))
            .collect::<Result<Vec<_>>>()?;
        let per_repo_limit = (request.limit.max(5) * 4).min(64);
        let snapshot = self.inner.snapshot.read().await?;
        let mut blocked_repos = HashMap::new();
        let mut query_vectors: HashMap<String, Arc<Vec<f32>>> = HashMap::new();
        let mut overlay_query_errors: HashMap<String, String> = HashMap::new();

        for ctx in &scope_contexts {
            let identity = self
                .repo_embedding_identity_status(&snapshot, &ctx.canonical_root)
                .await?;
            if let Some(reason) = identity.reason {
                blocked_repos.insert(ctx.requested_root.display().to_string(), reason);
            }
        }

        if semantic_share > 0.0 {
            let usages = self
                .query_profile_usages_for_contexts(&scope_contexts, &blocked_repos)
                .await?;
            let (vectors, errors) = self
                .embed_query_vectors_for_usages(usages, &request.query, &mut blocked_repos)
                .await?;
            query_vectors = vectors;
            overlay_query_errors = errors;
        }

        let mut stream = futures::stream::iter(scope_contexts.into_iter().map(|ctx| {
            let engine = self.clone();
            let request = request.clone();
            let canonical_query_vector = engine
                .embedding_profile_name_for_repo(&ctx.canonical_root)
                .ok()
                .and_then(|profile| query_vectors.get(profile).cloned());
            let overlay_query_vector = ctx.overlay.as_ref().and_then(|_| {
                engine
                    .overlay_profile_name(&ctx)
                    .ok()
                    .and_then(|profile| query_vectors.get(profile).cloned())
            });
            let repo_error = blocked_repos
                .get(&ctx.requested_root.display().to_string())
                .cloned();
            let entry = snapshot
                .codebases
                .get(&ctx.canonical_root.display().to_string())
                .cloned();
            async move {
                if let Some(error) = repo_error {
                    return SymbolRepoSearchOutcome {
                        repo: ctx.requested_root,
                        stale: false,
                        hits: Vec::new(),
                        warnings: Vec::new(),
                        error: Some(error),
                    };
                }
                let _repo_permit = match engine.acquire_repo_search_budget().await {
                    Ok(permit) => permit,
                    Err(error) => {
                        return SymbolRepoSearchOutcome {
                            repo: ctx.requested_root,
                            stale: false,
                            hits: Vec::new(),
                            warnings: Vec::new(),
                            error: Some(error.to_string()),
                        };
                    }
                };
                let stale = engine
                    .repo_is_stale(&ctx.requested_root, entry.as_ref())
                    .await
                    .unwrap_or(false);
                match engine
                    .search_symbol_context(
                        &ctx,
                        &request,
                        flavor,
                        per_repo_limit,
                        canonical_query_vector
                            .as_ref()
                            .map(|value| value.as_slice()),
                        overlay_query_vector.as_ref().map(|value| value.as_slice()),
                        lexical_share,
                        semantic_share,
                    )
                    .await
                {
                    Ok(result) => SymbolRepoSearchOutcome {
                        repo: ctx.requested_root,
                        stale,
                        hits: result.hits,
                        warnings: result.warnings,
                        error: None,
                    },
                    Err(error) => SymbolRepoSearchOutcome {
                        repo: ctx.requested_root,
                        stale,
                        hits: Vec::new(),
                        warnings: Vec::new(),
                        error: Some(error.to_string()),
                    },
                }
            }
        }))
        .buffer_unordered(parallelism);

        let mut repo_errors = overlay_query_errors
            .into_iter()
            .map(|(repo, error)| RepoSearchError { repo, error })
            .collect::<Vec<_>>();
        let mut repo_error_repos = repo_errors
            .iter()
            .map(|error| error.repo.clone())
            .collect::<HashSet<_>>();
        let mut hits = Vec::new();
        while let Some(outcome) = stream.next().await {
            let repo_key = outcome.repo.display().to_string();
            if let Some(error) = outcome.error {
                if repo_error_repos.insert(repo_key.clone()) {
                    repo_errors.push(RepoSearchError {
                        repo: repo_key,
                        error,
                    });
                }
                continue;
            }
            for warning in outcome.warnings {
                if repo_error_repos.insert(repo_key.clone()) {
                    repo_errors.push(RepoSearchError {
                        repo: repo_key.clone(),
                        error: warning,
                    });
                }
            }
            for hit in outcome.hits {
                hits.push(SymbolSearchHit {
                    symbol_id: hit.symbol_id,
                    repo: repo_key.clone(),
                    repo_label: repo_basename(&repo_key),
                    relative_path: hit.relative_path,
                    name: hit.name,
                    kind: hit.kind,
                    container: hit.container,
                    language: hit.language,
                    start_line: hit.start_line,
                    end_line: hit.end_line,
                    score: hit.combined_score,
                    lexical_score: (hit.lexical_score > 0.0).then_some(hit.lexical_score),
                    semantic_score: (hit.semantic_score > 0.0).then_some(hit.semantic_score),
                    indexed_at: hit.indexed_at,
                    file_hash: hit.file_hash,
                    stale: outcome.stale,
                });
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

    pub async fn search_text_scope(
        &self,
        scope: ResolvedScope,
        request: TextSearchScopeRequest,
    ) -> Result<TextSearchResponse> {
        let scope = self.narrow_scope_to_repo(scope, request.repo.as_deref())?;
        if request.query.trim().is_empty() {
            bail!("search_text requires a non-empty `query`");
        }
        if !text_search_has_bounded_target(&scope, &request) {
            bail!("search_text requires `file`, `pathPrefix`, or a repo-bounded scope");
        }
        let _request_permit = self.acquire_request_budget().await?;
        let parallelism = self.inner.config.search.max_concurrent_repo_searches.max(1);
        let scope_repos = scope.repos;

        let mut stream = futures::stream::iter(scope_repos.into_iter().map(|repo| {
            let engine = self.clone();
            let request = request.clone();
            let repo_name = repo.display().to_string();
            async move {
                let result = async {
                    let _repo_permit = engine.acquire_repo_search_budget().await?;
                    engine.search_text_repo(&repo, &request).await
                }
                .await;
                (repo_name, result)
            }
        }))
        .buffer_unordered(parallelism);

        let mut repo_errors = Vec::new();
        let mut hits = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                (_repo, Ok(repo_hits)) => hits.extend(repo_hits),
                (repo, Err(error)) => repo_errors.push(RepoSearchError {
                    repo,
                    error: error.to_string(),
                }),
            }
        }

        hits.sort_by(|left, right| {
            left.relative_path
                .cmp(&right.relative_path)
                .then(left.start_line.cmp(&right.start_line))
                .then(left.repo_label.cmp(&right.repo_label))
        });
        hits.truncate(request.limit);

        Ok(TextSearchResponse {
            scope: scope.id,
            label: scope.label,
            partial: !repo_errors.is_empty(),
            repo_errors,
            hits,
        })
    }

    pub async fn prepare_edit_target(
        &self,
        scope: ResolvedScope,
        request: PrepareEditTargetRequest,
    ) -> Result<PrepareEditTargetResponse> {
        let has_symbol_locator = request
            .symbol_name
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty());
        if request.symbol_id.is_none() && request.file.is_none() {
            bail!("prepare_edit_target requires `symbolId` or `file`");
        }
        if request.file.is_some()
            && request.line_hint.is_none()
            && request.query.is_none()
            && !has_symbol_locator
        {
            bail!(
                "prepare_edit_target requires `lineHint`, `query`, or `symbolName` when `file` is provided"
            );
        }
        let _request_permit = self.acquire_request_budget().await?;
        if let Some(symbol_id) = &request.symbol_id {
            return self
                .prepare_symbol_edit_target(scope, symbol_id, &request)
                .await;
        }
        if has_symbol_locator {
            return self.prepare_named_symbol_edit_target(scope, &request).await;
        }
        self.prepare_file_edit_target(scope, &request).await
    }

    pub async fn get_file_outline(
        &self,
        scope: ResolvedScope,
        file: &str,
    ) -> Result<FileOutlineResponse> {
        let normalized_file = normalize_relative_path(file);
        let snapshot = self.inner.snapshot.read().await?;
        let mut matches = Vec::new();

        for repo in scope.repos {
            let ctx = self.repo_context(&repo)?;
            let repo_key = repo.display().to_string();
            let canonical_key = ctx.canonical_root.display().to_string();
            let symbols = self
                .load_file_outline_symbols_for_repo(&repo, &normalized_file)
                .await?;
            if symbols.is_empty() {
                continue;
            }
            let repo_stale = self
                .repo_is_stale(&repo, snapshot.codebases.get(&canonical_key))
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

    async fn search_text_repo(
        &self,
        repo: &Path,
        request: &TextSearchScopeRequest,
    ) -> Result<Vec<TextSearchHit>> {
        let candidate_files = self.search_text_candidate_files(repo, request).await?;
        let mut cache = LiveFileRequestCache::new();
        let mut hits = Vec::new();

        for relative_path in candidate_files {
            let snapshot = match self.load_live_file(repo, &relative_path, &mut cache).await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    if request.file.is_none() {
                        continue;
                    }
                    if !live_file_exists(repo, &relative_path)? {
                        continue;
                    }
                    return Err(error);
                }
            };
            let matches = snapshot.find_literal_matches(
                &request.query,
                request.case_sensitive,
                request.whole_word,
                request.limit.saturating_mul(4).max(8),
            )?;
            for matched in matches {
                let preview_start = matched
                    .start_line
                    .saturating_sub(request.context_lines as u64)
                    .max(1);
                let preview_end =
                    (matched.end_line + request.context_lines as u64).min(snapshot.total_lines());
                let preview = snapshot
                    .slice_lines(preview_start, preview_end)
                    .map(|text| compact_preview(text, SEARCH_TEXT_PREVIEW_CHARS))
                    .unwrap_or_default();
                hits.push(TextSearchHit {
                    repo: repo.display().to_string(),
                    repo_label: repo_basename(&repo.display().to_string()),
                    relative_path: snapshot.relative_path.clone(),
                    start_line: matched.start_line,
                    end_line: matched.end_line,
                    preview,
                    stale: false,
                });
                if hits.len() >= request.limit.saturating_mul(4) {
                    break;
                }
            }
            if hits.len() >= request.limit.saturating_mul(4) {
                break;
            }
        }

        Ok(hits)
    }

    async fn search_text_candidate_files(
        &self,
        repo: &Path,
        request: &TextSearchScopeRequest,
    ) -> Result<Vec<String>> {
        if let Some(file) = &request.file {
            return Ok(vec![normalize_relative_path(file)]);
        }

        if search_text_scans_live_repo(request) {
            let language = request.language.clone();
            let extension_filter = request.extension_filter.clone();
            let repo_path = repo.to_path_buf();
            return self
                .run_search_lexical_blocking("search_text_repo_candidates", move || {
                    collect_live_candidate_files(
                        &repo_path,
                        None,
                        language.as_deref(),
                        &extension_filter,
                    )
                })
                .await;
        }

        let files = self
            .search_text_index_candidate_files(repo, request)
            .await?;
        if !files.is_empty() {
            return Ok(files);
        }

        let path_prefix = request.path_prefix.clone();
        let language = request.language.clone();
        let extension_filter = request.extension_filter.clone();
        let repo_path = repo.to_path_buf();
        let fallback: Vec<String> = self
            .run_search_lexical_blocking("search_text_fallback_candidates", move || {
                collect_live_candidate_files(
                    &repo_path,
                    path_prefix.as_deref(),
                    language.as_deref(),
                    &extension_filter,
                )
            })
            .await?;
        validate_search_text_fallback_size(request, fallback.len())?;
        Ok(fallback)
    }

    async fn search_text_index_candidate_files(
        &self,
        repo: &Path,
        request: &TextSearchScopeRequest,
    ) -> Result<Vec<String>> {
        let ctx = self.repo_context(repo)?;
        let chunk_request = text_search_chunk_request(request);
        let canonical_hits = self
            .run_search_lexical_blocking("search_text_shortlist", {
                let repo_path = ctx.canonical_root.clone();
                let local_index = self.inner.local_index.clone();
                let request = chunk_request.clone();
                move || local_index.search_chunks(&repo_path, &request)
            })
            .await?;
        let canonical_files = canonical_hits
            .into_iter()
            .map(|hit| hit.relative_path)
            .collect::<Vec<_>>();

        let Some(overlay) = ctx.overlay.as_ref() else {
            return Ok(dedup_relative_paths(canonical_files));
        };
        let overlay_state = match self.load_overlay_state(overlay).await {
            Ok(Some(state)) => state,
            Ok(None) | Err(_) => return Ok(dedup_relative_paths(canonical_files)),
        };

        let suppressed_paths = if overlay_state_suppresses_canonical(&overlay_state) {
            overlay_suppressed_paths(&overlay_state)
        } else {
            BTreeSet::new()
        };
        let overlay_files = if overlay_state_has_search_index(&overlay_state) {
            let overlay_hits = self
                .run_search_lexical_blocking("search_text_overlay_shortlist", {
                    let repo_path = overlay.storage_root.clone();
                    let local_index = self.inner.local_index.clone();
                    let request = chunk_request;
                    move || local_index.search_chunks(&repo_path, &request)
                })
                .await?;
            overlay_hits
                .into_iter()
                .map(|hit| hit.relative_path)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        Ok(merge_text_candidate_files(
            canonical_files,
            overlay_files,
            &suppressed_paths,
        ))
    }

    fn narrow_scope_to_repo(
        &self,
        scope: ResolvedScope,
        repo: Option<&str>,
    ) -> Result<ResolvedScope> {
        let Some(repo) = repo.map(str::trim).filter(|value| !value.is_empty()) else {
            return Ok(scope);
        };

        let repo_scope = self.inner.config.resolve_mcp_scope(Some(repo), None)?;
        let repo_path = repo_scope
            .repos
            .into_iter()
            .next()
            .context("repo scope resolved without repos")?;
        if !scope.repos.iter().any(|candidate| candidate == &repo_path) {
            bail!("repo `{repo}` is not part of scope `{}`", scope.id);
        }

        Ok(ResolvedScope {
            kind: ScopeKind::Repo,
            id: repo_path.display().to_string(),
            label: repo_scope.label,
            repos: vec![repo_path],
        })
    }

    async fn prepare_symbol_edit_target(
        &self,
        scope: ResolvedScope,
        symbol_id: &str,
        request: &PrepareEditTargetRequest,
    ) -> Result<PrepareEditTargetResponse> {
        let Some(repo) = self.resolve_prepare_repo(&scope, request.repo.as_deref(), true)? else {
            return Ok(PrepareEditTargetResponse {
                status: EditTargetStatus::NotFound,
                repo: None,
                repo_label: None,
                relative_path: None,
                start_line: None,
                end_line: None,
                content: None,
                anchors: Vec::new(),
                anchor_quality: None,
                resolution_type: None,
                file_hash: None,
                indexed: None,
                stale: None,
                indexed_at: None,
                indexed_file_hash: None,
                symbol_id: Some(symbol_id.to_string()),
                truncated: None,
                candidates: Vec::new(),
                symbol_start_line: None,
                symbol_end_line: None,
                reason_code: None,
                suggested_next_tool: None,
            });
        };

        let Some(resolved) = self.resolve_symbol_in_repo(&repo, symbol_id).await? else {
            return Ok(PrepareEditTargetResponse {
                status: EditTargetStatus::NotFound,
                repo: Some(repo.display().to_string()),
                repo_label: Some(repo_basename(&repo.display().to_string())),
                relative_path: None,
                start_line: None,
                end_line: None,
                content: None,
                anchors: Vec::new(),
                anchor_quality: None,
                resolution_type: None,
                file_hash: None,
                indexed: Some(false),
                stale: Some(false),
                indexed_at: None,
                indexed_file_hash: None,
                symbol_id: Some(symbol_id.to_string()),
                truncated: None,
                candidates: Vec::new(),
                symbol_start_line: None,
                symbol_end_line: None,
                reason_code: None,
                suggested_next_tool: None,
            });
        };

        let mut cache = LiveFileRequestCache::new();
        let snapshot = self
            .load_live_file(&resolved.repo, &resolved.symbol.relative_path, &mut cache)
            .await?;
        let indexed_metadata = IndexedFileMetadata {
            indexed_at: Some(resolved.symbol.indexed_at.clone()),
            indexed_file_hash: Some(resolved.symbol.file_hash.clone()),
        };

        if let Some(query) = request.query.as_deref() {
            let matches = snapshot
                .find_literal_matches(query, true, false, usize::MAX)?
                .into_iter()
                .filter(|matched| {
                    matched.start_line >= resolved.symbol.start_line
                        && matched.end_line <= resolved.symbol.end_line
                })
                .collect::<Vec<_>>();
            if matches.is_empty() {
                return Ok(not_found_prepare_response(
                    &resolved.repo,
                    Some(&resolved.symbol.relative_path),
                    Some(symbol_id),
                ));
            }
            if let Some(ready) = self.pick_ready_match_target(
                &snapshot,
                &resolved.symbol,
                &indexed_metadata,
                &matches,
                request,
                EditResolutionType::Literal,
            )? {
                return self.build_ready_prepare_response(ready, request).await;
            }
            return Ok(self.build_ambiguous_prepare_response(
                &resolved.repo,
                &snapshot,
                matches,
                request,
            ));
        }

        let symbol_lines = span_line_count(resolved.symbol.start_line, resolved.symbol.end_line);
        if symbol_lines <= PREPARE_READY_MAX_LINES as u64 {
            return self
                .build_ready_prepare_response(
                    ReadyEditTarget {
                        snapshot,
                        start_line: resolved.symbol.start_line,
                        end_line: resolved.symbol.end_line,
                        resolution_type: EditResolutionType::Symbol,
                        symbol_id: Some(resolved.symbol.symbol_id.clone()),
                        indexed_metadata,
                        truncated: false,
                        symbol_signature_line: Some(resolved.symbol.start_line),
                        query: None,
                    },
                    request,
                )
                .await;
        }

        Ok(self.build_large_symbol_prepare_response(
            &snapshot,
            &indexed_metadata,
            Some(resolved.symbol.symbol_id.clone()),
            Some(EditResolutionType::Symbol),
            resolved.symbol.start_line,
            resolved.symbol.end_line,
        ))
    }

    async fn prepare_named_symbol_edit_target(
        &self,
        scope: ResolvedScope,
        request: &PrepareEditTargetRequest,
    ) -> Result<PrepareEditTargetResponse> {
        let file = request
            .file
            .as_deref()
            .map(normalize_relative_path)
            .context("prepare_edit_target requires `file` when `symbolId` is not provided")?;
        let symbol_name = request
            .symbol_name
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context("prepare_edit_target requires `symbolName` when resolving by symbol")?;
        let repos = self.resolve_prepare_repos(&scope, request.repo.as_deref())?;
        let mut resolved = Vec::new();

        for repo in repos {
            let symbols = self
                .load_file_outline_symbols_for_repo(&repo, &file)
                .await?;
            let matches = symbols
                .into_iter()
                .filter(|symbol| symbol.name == symbol_name)
                .filter(|symbol| {
                    request
                        .symbol_kind
                        .as_deref()
                        .is_none_or(|kind| symbol.kind == kind)
                })
                .filter(|symbol| {
                    request
                        .symbol_container
                        .as_deref()
                        .is_none_or(|container| symbol.container.as_deref() == Some(container))
                })
                .collect::<Vec<_>>();
            if matches.is_empty() {
                continue;
            }
            if let Some(line_hint) = request.line_hint {
                if let Some(symbol) = matches
                    .iter()
                    .find(|symbol| line_hint >= symbol.start_line && line_hint <= symbol.end_line)
                {
                    resolved.push(ResolvedEditSymbol {
                        repo: repo.clone(),
                        symbol: symbol.clone(),
                    });
                    continue;
                }
            }
            if matches.len() == 1 {
                resolved.push(ResolvedEditSymbol {
                    repo: repo.clone(),
                    symbol: matches.into_iter().next().expect("single symbol match"),
                });
                continue;
            }
            return Ok(named_symbol_ambiguous_prepare_response(
                &repo, &file, matches,
            ));
        }

        match resolved.len() {
            0 => Ok(not_found_prepare_response_for_file(&file)),
            1 => {
                self.prepare_symbol_edit_target(
                    ResolvedScope {
                        kind: ScopeKind::Repo,
                        id: resolved[0].repo.display().to_string(),
                        label: resolved[0].repo.display().to_string(),
                        repos: vec![resolved[0].repo.clone()],
                    },
                    &resolved[0].symbol.symbol_id,
                    request,
                )
                .await
            }
            _ => Ok(ambiguous_prepare_response_from_symbols(&file, &resolved)),
        }
    }

    async fn prepare_file_edit_target(
        &self,
        scope: ResolvedScope,
        request: &PrepareEditTargetRequest,
    ) -> Result<PrepareEditTargetResponse> {
        let file = request
            .file
            .as_deref()
            .map(normalize_relative_path)
            .context("prepare_edit_target requires `file` when `symbolId` is not provided")?;
        let repos = self.resolve_prepare_repos(&scope, request.repo.as_deref())?;
        let mut candidates = Vec::new();
        let mut cache = LiveFileRequestCache::new();

        for repo in repos {
            let _repo_permit = self.acquire_repo_search_budget().await?;
            let snapshot = match self.load_live_file(&repo, &file, &mut cache).await {
                Ok(snapshot) => snapshot,
                Err(_) => continue,
            };
            if let Some(query) = request.query.as_deref() {
                let matches = snapshot
                    .find_literal_matches(query, true, false, usize::MAX)?
                    .into_iter()
                    .collect::<Vec<_>>();
                if matches.is_empty() {
                    continue;
                }
                let indexed_metadata = self
                    .indexed_file_metadata(&repo, &snapshot.relative_path)
                    .await?;
                let file_symbols = self
                    .load_file_outline_symbols_for_repo(&repo, &snapshot.relative_path)
                    .await?;
                if let Some(ready) = self.pick_ready_file_match_target(
                    &snapshot,
                    indexed_metadata.clone(),
                    &file_symbols,
                    &matches,
                    request,
                )? {
                    candidates.push(ready);
                } else {
                    return Ok(
                        self.build_ambiguous_prepare_response(&repo, &snapshot, matches, request)
                    );
                }
                continue;
            }

            let line_hint = request.line_hint.context(
                "prepare_edit_target requires `lineHint` or `query` when `file` is provided",
            )?;
            let indexed_metadata = self
                .indexed_file_metadata(&repo, &snapshot.relative_path)
                .await?;
            let file_symbols = self
                .load_file_outline_symbols_for_repo(&repo, &snapshot.relative_path)
                .await?;
            if let Some(symbol) = narrowest_covering_symbol(&file_symbols, line_hint) {
                let symbol_lines = span_line_count(symbol.start_line, symbol.end_line);
                if symbol_lines <= PREPARE_READY_MAX_LINES as u64 {
                    candidates.push(ReadyEditTarget {
                        snapshot: snapshot.clone(),
                        start_line: symbol.start_line,
                        end_line: symbol.end_line,
                        resolution_type: EditResolutionType::Symbol,
                        symbol_id: Some(symbol.symbol_id.clone()),
                        indexed_metadata,
                        truncated: false,
                        symbol_signature_line: Some(symbol.start_line),
                        query: None,
                    });
                } else {
                    return Ok(self.build_large_symbol_prepare_response(
                        &snapshot,
                        &indexed_metadata,
                        Some(symbol.symbol_id.clone()),
                        Some(EditResolutionType::LineHint),
                        symbol.start_line,
                        symbol.end_line,
                    ));
                }
            } else {
                let (start_line, end_line, truncated) = bounded_window(
                    snapshot.total_lines(),
                    line_hint,
                    line_hint,
                    request.before_lines,
                    request.after_lines,
                    request.max_lines,
                );
                candidates.push(ReadyEditTarget {
                    snapshot,
                    start_line,
                    end_line,
                    resolution_type: EditResolutionType::LineHint,
                    symbol_id: None,
                    indexed_metadata,
                    truncated,
                    symbol_signature_line: None,
                    query: None,
                });
            }
        }

        if candidates.is_empty() {
            return Ok(not_found_prepare_response_for_file(&file));
        }
        if candidates.len() == 1 {
            return self
                .build_ready_prepare_response(candidates.remove(0), request)
                .await;
        }
        Ok(PrepareEditTargetResponse {
            status: EditTargetStatus::Ambiguous,
            repo: None,
            repo_label: None,
            relative_path: Some(file),
            start_line: None,
            end_line: None,
            content: None,
            anchors: Vec::new(),
            anchor_quality: None,
            resolution_type: None,
            file_hash: None,
            indexed: None,
            stale: None,
            indexed_at: None,
            indexed_file_hash: None,
            symbol_id: None,
            truncated: None,
            candidates: candidates
                .into_iter()
                .map(|candidate| EditTargetCandidate {
                    repo: candidate.snapshot.repo.display().to_string(),
                    repo_label: repo_basename(&candidate.snapshot.repo.display().to_string()),
                    relative_path: candidate.snapshot.relative_path.clone(),
                    start_line: candidate.start_line,
                    end_line: candidate.end_line,
                    preview: candidate
                        .snapshot
                        .slice_lines(candidate.start_line, candidate.end_line)
                        .map(|text| compact_preview(text, SEARCH_TEXT_PREVIEW_CHARS))
                        .unwrap_or_default(),
                })
                .collect(),
            symbol_start_line: None,
            symbol_end_line: None,
            reason_code: Some(EditTargetReasonCode::MultipleMatches),
            suggested_next_tool: Some("prepare_edit_target".to_string()),
        })
    }

    async fn build_ready_prepare_response(
        &self,
        ready: ReadyEditTarget,
        request: &PrepareEditTargetRequest,
    ) -> Result<PrepareEditTargetResponse> {
        let content = ready
            .snapshot
            .slice_lines(ready.start_line, ready.end_line)
            .map(ToString::to_string)
            .context("selected edit target lines are unavailable")?;
        let anchors = select_edit_anchors(
            &ready.snapshot,
            ready.start_line,
            ready.end_line,
            request.anchor_count,
            ready.query.as_deref(),
            ready.symbol_signature_line,
        );
        if let Some((reason_code, suggested_next_tool)) =
            ready_target_reason(ready.start_line, ready.end_line, &anchors)
        {
            return Ok(self.build_needs_narrowing_prepare_response(
                &ready.snapshot,
                &ready.indexed_metadata,
                ready.symbol_id,
                Some(ready.resolution_type),
                Some(ready.start_line),
                Some(ready.end_line),
                Some(reason_code),
                Some(suggested_next_tool.to_string()),
                None,
                None,
                Some(ready.truncated),
            ));
        }
        let anchor_quality = if anchors.iter().any(|anchor| anchor.unique_in_file) {
            AnchorQuality::Strong
        } else {
            AnchorQuality::Weak
        };
        let indexed = ready.indexed_metadata.indexed_file_hash.is_some();
        let stale = ready
            .indexed_metadata
            .indexed_file_hash
            .as_deref()
            .is_some_and(|hash| hash != ready.snapshot.file_hash);
        Ok(PrepareEditTargetResponse {
            status: EditTargetStatus::Ready,
            repo: Some(ready.snapshot.repo.display().to_string()),
            repo_label: Some(repo_basename(&ready.snapshot.repo.display().to_string())),
            relative_path: Some(ready.snapshot.relative_path.clone()),
            start_line: Some(ready.start_line),
            end_line: Some(ready.end_line),
            content: Some(content),
            anchors,
            anchor_quality: Some(anchor_quality),
            resolution_type: Some(ready.resolution_type),
            file_hash: Some(ready.snapshot.file_hash.clone()),
            indexed: Some(indexed),
            stale: Some(stale),
            indexed_at: ready.indexed_metadata.indexed_at,
            indexed_file_hash: ready.indexed_metadata.indexed_file_hash,
            symbol_id: ready.symbol_id,
            truncated: Some(ready.truncated),
            candidates: Vec::new(),
            symbol_start_line: None,
            symbol_end_line: None,
            reason_code: None,
            suggested_next_tool: None,
        })
    }

    fn build_large_symbol_prepare_response(
        &self,
        snapshot: &Arc<LiveFileSnapshot>,
        indexed_metadata: &IndexedFileMetadata,
        symbol_id: Option<String>,
        resolution_type: Option<EditResolutionType>,
        symbol_start_line: u64,
        symbol_end_line: u64,
    ) -> PrepareEditTargetResponse {
        self.build_needs_narrowing_prepare_response(
            snapshot,
            indexed_metadata,
            symbol_id,
            resolution_type,
            None,
            None,
            Some(EditTargetReasonCode::LargeSymbol),
            Some("get_file_outline".to_string()),
            Some(symbol_start_line),
            Some(symbol_end_line),
            Some(true),
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "prepare-edit narrowing responses need to bundle guidance plus optional symbol and span metadata"
    )]
    fn build_needs_narrowing_prepare_response(
        &self,
        snapshot: &Arc<LiveFileSnapshot>,
        indexed_metadata: &IndexedFileMetadata,
        symbol_id: Option<String>,
        resolution_type: Option<EditResolutionType>,
        start_line: Option<u64>,
        end_line: Option<u64>,
        reason_code: Option<EditTargetReasonCode>,
        suggested_next_tool: Option<String>,
        symbol_start_line: Option<u64>,
        symbol_end_line: Option<u64>,
        truncated: Option<bool>,
    ) -> PrepareEditTargetResponse {
        PrepareEditTargetResponse {
            status: EditTargetStatus::NeedsNarrowing,
            repo: Some(snapshot.repo.display().to_string()),
            repo_label: Some(repo_basename(&snapshot.repo.display().to_string())),
            relative_path: Some(snapshot.relative_path.clone()),
            start_line,
            end_line,
            content: None,
            anchors: Vec::new(),
            anchor_quality: None,
            resolution_type,
            file_hash: Some(snapshot.file_hash.clone()),
            indexed: Some(indexed_metadata.indexed_file_hash.is_some()),
            stale: indexed_metadata
                .indexed_file_hash
                .as_deref()
                .map(|hash| hash != snapshot.file_hash),
            indexed_at: indexed_metadata.indexed_at.clone(),
            indexed_file_hash: indexed_metadata.indexed_file_hash.clone(),
            symbol_id,
            truncated,
            candidates: Vec::new(),
            symbol_start_line,
            symbol_end_line,
            reason_code,
            suggested_next_tool,
        }
    }

    async fn load_live_file(
        &self,
        repo: &Path,
        relative_path: &str,
        cache: &mut LiveFileRequestCache,
    ) -> Result<Arc<LiveFileSnapshot>> {
        let normalized = normalize_relative_path(relative_path);
        let key = file_freshness_key(repo, &normalized);
        if let Some(snapshot) = cache.get(&key) {
            return Ok(snapshot.clone());
        }
        let repo_path = repo.to_path_buf();
        let live_files = self.inner.live_files.clone();
        let snapshot = self
            .run_search_lexical_blocking("load_live_file", move || {
                live_files.load_snapshot(&repo_path, &normalized)
            })
            .await?;
        cache.insert(key, snapshot.clone());
        Ok(snapshot)
    }

    async fn index_lookup_target(
        &self,
        repo: &Path,
        relative_path: &str,
    ) -> Result<(PathBuf, String)> {
        let ctx = self.repo_context(repo)?;
        let normalized_path = normalize_relative_path(relative_path);
        if let Some(overlay) = ctx.overlay.as_ref() {
            if let Ok(Some(state)) = self.load_overlay_state(overlay).await {
                if overlay_lookup_uses_overlay(&state, &normalized_path) {
                    return Ok((overlay.storage_root.clone(), overlay.repo_key.clone()));
                }
            }
        }
        Ok((
            ctx.canonical_root.clone(),
            ctx.canonical_root.display().to_string(),
        ))
    }

    async fn load_file_outline_symbols_for_repo(
        &self,
        repo: &Path,
        relative_path: &str,
    ) -> Result<Vec<IndexedSymbol>> {
        let (repo_path, repo_key) = self.index_lookup_target(repo, relative_path).await?;
        let _ = repo_path;
        self.load_file_outline_symbols(&repo_key, relative_path)
            .await
    }

    async fn indexed_file_metadata(
        &self,
        repo: &Path,
        relative_path: &str,
    ) -> Result<IndexedFileMetadata> {
        let (repo_path, repo_key) = self.index_lookup_target(repo, relative_path).await?;
        self.indexed_file_metadata_from_index(&repo_path, &repo_key, relative_path)
            .await
    }

    async fn indexed_file_metadata_from_index(
        &self,
        repo: &Path,
        repo_key: &str,
        relative_path: &str,
    ) -> Result<IndexedFileMetadata> {
        let repo_path = repo.to_path_buf();
        let normalized_path = normalize_relative_path(relative_path);
        let chunk_path = normalized_path.clone();
        let local_index = self.inner.local_index.clone();
        let chunk_metadata = self
            .run_search_lexical_blocking("indexed_file_chunk_metadata", move || {
                let chunks = local_index.chunks_for_file(&repo_path, &chunk_path)?;
                Ok::<_, anyhow::Error>(
                    chunks
                        .first()
                        .map(|chunk| IndexedFileMetadata {
                            indexed_at: Some(chunk.indexed_at.clone()),
                            indexed_file_hash: Some(chunk.file_hash.clone()),
                        })
                        .unwrap_or(IndexedFileMetadata {
                            indexed_at: None,
                            indexed_file_hash: None,
                        }),
                )
            })
            .await?;
        if chunk_metadata.indexed_file_hash.is_some() {
            return Ok(chunk_metadata);
        }

        let repo_key = repo_key.to_string();
        let symbol_path = normalized_path.clone();
        let symbol_store = self.inner.symbol_store.clone();
        self.run_search_lexical_blocking("indexed_file_symbol_metadata", move || {
            let symbols = symbol_store.file_symbols(&repo_key, &symbol_path)?;
            Ok::<_, anyhow::Error>(
                symbols
                    .first()
                    .map(|symbol| IndexedFileMetadata {
                        indexed_at: Some(symbol.indexed_at.clone()),
                        indexed_file_hash: Some(symbol.file_hash.clone()),
                    })
                    .unwrap_or(IndexedFileMetadata {
                        indexed_at: None,
                        indexed_file_hash: None,
                    }),
            )
        })
        .await
    }

    async fn resolve_symbol_in_repo(
        &self,
        repo: &Path,
        symbol_id: &str,
    ) -> Result<Option<ResolvedEditSymbol>> {
        let ctx = self.repo_context(repo)?;
        let mut repo_keys = Vec::new();
        if let Some(overlay) = ctx.overlay.as_ref() {
            repo_keys.push(overlay.repo_key.clone());
            repo_keys.push(ctx.canonical_root.display().to_string());
        } else {
            repo_keys.push(repo.display().to_string());
        }
        let symbol_id = symbol_id.to_string();
        for repo_key in repo_keys {
            let symbol_store = self.inner.symbol_store.clone();
            let symbol_id = symbol_id.clone();
            let symbol = self
                .run_search_lexical_blocking("resolve_symbol_for_edit", move || {
                    symbol_store.symbol_by_id(&repo_key, &symbol_id)
                })
                .await?;
            if let Some(symbol) = symbol {
                return Ok(Some(ResolvedEditSymbol {
                    repo: repo.to_path_buf(),
                    symbol,
                }));
            }
        }
        Ok(None)
    }

    fn resolve_prepare_repos(
        &self,
        scope: &ResolvedScope,
        repo_hint: Option<&str>,
    ) -> Result<Vec<PathBuf>> {
        if let Some(repo) = self.resolve_prepare_repo(scope, repo_hint, false)? {
            return Ok(vec![repo]);
        }
        Ok(scope.repos.clone())
    }

    fn resolve_prepare_repo(
        &self,
        scope: &ResolvedScope,
        repo_hint: Option<&str>,
        require_single: bool,
    ) -> Result<Option<PathBuf>> {
        if let Some(repo_hint) = repo_hint {
            let hint = repo_hint.trim();
            let matches = scope
                .repos
                .iter()
                .filter(|repo| {
                    repo.display().to_string() == hint
                        || repo
                            .file_name()
                            .and_then(|value| value.to_str())
                            .is_some_and(|value| value == hint)
                })
                .cloned()
                .collect::<Vec<_>>();
            return match matches.len() {
                0 => bail!("repo `{hint}` is not part of the selected scope"),
                1 => Ok(matches.into_iter().next()),
                _ => bail!("repo `{hint}` is ambiguous within the selected scope"),
            };
        }
        if !require_single {
            return Ok(None);
        }
        match scope.repos.as_slice() {
            [repo] => Ok(Some(repo.clone())),
            _ => bail!("repo is required when resolving `symbolId` within a multi-repo scope"),
        }
    }

    fn pick_ready_match_target(
        &self,
        snapshot: &Arc<LiveFileSnapshot>,
        symbol: &IndexedSymbol,
        indexed_metadata: &IndexedFileMetadata,
        matches: &[TextMatch],
        request: &PrepareEditTargetRequest,
        resolution_type: EditResolutionType,
    ) -> Result<Option<ReadyEditTarget>> {
        let Some(selected) = select_text_match(matches, request.line_hint, request.occurrence)
        else {
            return Ok(None);
        };
        if symbol_fits_ready_window(symbol, selected, request.max_lines) {
            return Ok(Some(ReadyEditTarget {
                snapshot: snapshot.clone(),
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                resolution_type,
                symbol_id: Some(symbol.symbol_id.clone()),
                indexed_metadata: indexed_metadata.clone(),
                truncated: false,
                symbol_signature_line: Some(symbol.start_line),
                query: request.query.clone(),
            }));
        }
        let (start_line, end_line, truncated) = bounded_window(
            snapshot.total_lines(),
            selected.start_line,
            selected.end_line,
            request.before_lines,
            request.after_lines,
            request.max_lines,
        );
        Ok(Some(ReadyEditTarget {
            snapshot: snapshot.clone(),
            start_line,
            end_line,
            resolution_type,
            symbol_id: Some(symbol.symbol_id.clone()),
            indexed_metadata: indexed_metadata.clone(),
            truncated: truncated || start_line != symbol.start_line || end_line != symbol.end_line,
            symbol_signature_line: Some(symbol.start_line),
            query: request.query.clone(),
        }))
    }

    fn pick_ready_file_match_target(
        &self,
        snapshot: &Arc<LiveFileSnapshot>,
        indexed_metadata: IndexedFileMetadata,
        file_symbols: &[IndexedSymbol],
        matches: &[TextMatch],
        request: &PrepareEditTargetRequest,
    ) -> Result<Option<ReadyEditTarget>> {
        let Some(selected) = select_text_match(matches, request.line_hint, request.occurrence)
        else {
            return Ok(None);
        };
        let covering_symbol = narrowest_covering_symbol(file_symbols, selected.start_line)
            .filter(|symbol| selected.end_line <= symbol.end_line);
        if let Some(symbol) = covering_symbol {
            if symbol_fits_ready_window(symbol, selected, request.max_lines) {
                return Ok(Some(ReadyEditTarget {
                    snapshot: snapshot.clone(),
                    start_line: symbol.start_line,
                    end_line: symbol.end_line,
                    resolution_type: EditResolutionType::Literal,
                    symbol_id: Some(symbol.symbol_id.clone()),
                    indexed_metadata,
                    truncated: false,
                    symbol_signature_line: Some(symbol.start_line),
                    query: request.query.clone(),
                }));
            }
        }
        let (start_line, end_line, truncated) = bounded_window(
            snapshot.total_lines(),
            selected.start_line,
            selected.end_line,
            request.before_lines,
            request.after_lines,
            request.max_lines,
        );
        Ok(Some(ReadyEditTarget {
            snapshot: snapshot.clone(),
            start_line,
            end_line,
            resolution_type: EditResolutionType::Literal,
            symbol_id: covering_symbol.map(|symbol| symbol.symbol_id.clone()),
            indexed_metadata,
            truncated: truncated
                || covering_symbol.is_some_and(|symbol| {
                    start_line != symbol.start_line || end_line != symbol.end_line
                }),
            symbol_signature_line: covering_symbol.map(|symbol| symbol.start_line),
            query: request.query.clone(),
        }))
    }

    fn build_ambiguous_prepare_response(
        &self,
        repo: &Path,
        snapshot: &Arc<LiveFileSnapshot>,
        matches: Vec<TextMatch>,
        _request: &PrepareEditTargetRequest,
    ) -> PrepareEditTargetResponse {
        let candidates = matches
            .into_iter()
            .map(|matched| {
                let preview_start = matched
                    .start_line
                    .saturating_sub(PREPARE_AMBIGUOUS_PREVIEW_BEFORE_LINES)
                    .max(1);
                let preview_end = (matched.end_line + PREPARE_AMBIGUOUS_PREVIEW_AFTER_LINES)
                    .min(snapshot.total_lines());
                EditTargetCandidate {
                    repo: repo.display().to_string(),
                    repo_label: repo_basename(&repo.display().to_string()),
                    relative_path: snapshot.relative_path.clone(),
                    start_line: matched.start_line,
                    end_line: matched.end_line,
                    preview: snapshot
                        .slice_lines(preview_start, preview_end)
                        .map(|text| compact_preview(text, PREPARE_AMBIGUOUS_PREVIEW_CHARS))
                        .unwrap_or_default(),
                }
            })
            .collect();
        PrepareEditTargetResponse {
            status: EditTargetStatus::Ambiguous,
            repo: Some(repo.display().to_string()),
            repo_label: Some(repo_basename(&repo.display().to_string())),
            relative_path: Some(snapshot.relative_path.clone()),
            start_line: None,
            end_line: None,
            content: None,
            anchors: Vec::new(),
            anchor_quality: None,
            resolution_type: None,
            file_hash: None,
            indexed: None,
            stale: None,
            indexed_at: None,
            indexed_file_hash: None,
            symbol_id: None,
            truncated: None,
            candidates,
            symbol_start_line: None,
            symbol_end_line: None,
            reason_code: Some(EditTargetReasonCode::MultipleMatches),
            suggested_next_tool: Some("prepare_edit_target".to_string()),
        }
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
            let content = if request.snippet_chars == 0 {
                String::new()
            } else if let Some(snippet) = snippet_cache.get(&snippet_key) {
                snippet.clone()
            } else {
                let snippet = self
                    .build_hit_snippet(
                        Path::new(&hit.repo),
                        hit,
                        &request.query,
                        plan.snippet_neighbor_chunks,
                        request.snippet_chars,
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
        max_chars: usize,
    ) -> String {
        if max_chars == 0 {
            return String::new();
        }
        let repo_path = repo.to_path_buf();
        let relative_path = hit.relative_path.clone();
        let target_line = hit.start_line;
        let fallback = build_snippet(query, &hit.content, max_chars);
        let local_index = self.inner.local_index.clone();
        let context_chars = max_chars.saturating_mul(2).clamp(max_chars, 1200);
        let context = self
            .run_search_lexical_blocking("search_hit_context", move || {
                let chunks = local_index.chunks_for_file(&repo_path, &relative_path)?;
                Ok(build_chunk_context_snippet(
                    &chunks,
                    target_line,
                    neighbor_chunks,
                    context_chars,
                ))
            })
            .await;

        match context {
            Ok(Some(context)) => build_snippet(query, &context, max_chars),
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
        let identity_status = index_identity_status_for_snapshot(
            &snapshot,
            self.inner.config.embedding.default_profile_name(),
            &self.configured_embedding_fingerprints().await?,
        );
        let mut repos = Vec::new();
        let mut indexed_files = 0u64;
        let mut total_chunks = 0u64;

        for repo in scope.repos {
            let ctx = self.repo_context(&repo)?;
            let repo_key = repo.display().to_string();
            let status = if ctx.overlay.is_some() {
                self.status_for_worktree_overlay(&snapshot, &ctx).await?
            } else {
                self.status_for_repo(&snapshot, &repo).await?
            };
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
            if repo.is_absolute() && !repo.exists() {
                refreshed.push(self.clear_removed_repo(&repo, false).await?);
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
                    .index_repo(
                        &repo,
                        false,
                        SplitterKind::Ast,
                        &[],
                        &[],
                        IndexExecutionMode::Standard,
                    )
                    .await?;
                refreshed.push(result);
            } else {
                self.record_fingerprint(&repo_key, &fingerprint).await?;
            }
        }

        Ok(refreshed)
    }

    #[allow(dead_code)]
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

    #[allow(dead_code)]
    async fn persist_index_identity(&self) -> Result<()> {
        let configured_embedding_fingerprints = self.configured_embedding_fingerprints().await?;
        let default_profile = self
            .inner
            .config
            .embedding
            .default_profile_name()
            .to_string();
        let legacy_fingerprint = configured_embedding_fingerprints
            .get(&default_profile)
            .cloned();
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.index_format_version = INDEX_FORMAT_VERSION.to_string();
                snapshot.search_root_version = SEARCH_ROOT_VERSION.to_string();
                snapshot.embedding_fingerprint = legacy_fingerprint.clone();
            })
            .await?;
        Ok(())
    }

    async fn fail_worktree_overlay(
        &self,
        overlay: &WorktreeOverlayContext,
        worktree_key: &str,
        canonical_key: &str,
        message: String,
        profile_name: Option<String>,
        embedding_fingerprint: Option<String>,
        force: bool,
        changes: RepoChangeSummary,
        index_status: &str,
    ) -> Result<RepoIndexResult> {
        let failed_state = failed_overlay_state(
            overlay,
            worktree_key,
            canonical_key,
            profile_name.clone(),
            embedding_fingerprint.clone(),
        );
        if self.save_overlay_state(&failed_state).await.is_err() {
            self.remove_overlay_state_if_present(&overlay.resolution.overlay_id)
                .await?;
        }
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.worktrees.insert(
                    worktree_key.to_string(),
                    WorktreeSnapshotEntry::failed(
                        canonical_key.to_string(),
                        overlay.resolution.repo_identity.clone(),
                        overlay.resolution.overlay_id.clone(),
                        message.clone(),
                        profile_name.clone(),
                        embedding_fingerprint.clone(),
                    ),
                );
            })
            .await?;
        Ok(RepoIndexResult {
            repo: worktree_key.to_string(),
            indexed_files: None,
            total_chunks: None,
            index_status: Some(index_status.to_string()),
            full_reindex: force,
            changes,
            error: Some(message),
        })
    }

    async fn index_worktree_overlay(
        &self,
        ctx: RepoContext,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
        mode: IndexExecutionMode,
    ) -> Result<RepoIndexResult> {
        let Some(overlay) = ctx.overlay.as_ref() else {
            bail!("internal error: missing worktree overlay context");
        };
        let worktree_key = ctx.requested_root.display().to_string();
        let canonical_key = ctx.canonical_root.display().to_string();
        let profile_name = match self.overlay_profile_name(&ctx) {
            Ok(name) => name.to_string(),
            Err(error) => {
                return self
                    .fail_worktree_overlay(
                        overlay,
                        &worktree_key,
                        &canonical_key,
                        error.to_string(),
                        None,
                        None,
                        force,
                        RepoChangeSummary::default(),
                        "failed",
                    )
                    .await;
            }
        };
        let configured_embedding_fingerprint = match self.overlay_fingerprint(&ctx).await {
            Ok(fingerprint) => fingerprint,
            Err(error) => {
                return self
                    .fail_worktree_overlay(
                        overlay,
                        &worktree_key,
                        &canonical_key,
                        error.to_string(),
                        Some(profile_name.clone()),
                        None,
                        force,
                        RepoChangeSummary::default(),
                        "failed",
                    )
                    .await;
            }
        };
        let configured_embedding_dimension = match self.overlay_dimension(&ctx).await {
            Ok(dimension) => dimension,
            Err(error) => {
                return self
                    .fail_worktree_overlay(
                        overlay,
                        &worktree_key,
                        &canonical_key,
                        error.to_string(),
                        Some(profile_name.clone()),
                        Some(configured_embedding_fingerprint.clone()),
                        force,
                        RepoChangeSummary::default(),
                        "failed",
                    )
                    .await;
            }
        };

        if !ctx.canonical_root.exists() {
            let message = format!(
                "canonical repo `{}` is missing",
                ctx.canonical_root.display()
            );
            return self
                .fail_worktree_overlay(
                    overlay,
                    &worktree_key,
                    &canonical_key,
                    message,
                    Some(profile_name.clone()),
                    Some(configured_embedding_fingerprint.clone()),
                    false,
                    RepoChangeSummary::default(),
                    "canonical_missing",
                )
                .await;
        }

        {
            let snapshot = self.inner.snapshot.read().await?;
            let canonical_entry = snapshot.codebases.get(&canonical_key);
            if !matches!(
                canonical_entry.map(|entry| entry.status.as_str()),
                Some("indexed")
            ) {
                let message = format!(
                    "canonical repo `{}` is not indexed; index it before refreshing worktree overlays",
                    ctx.canonical_root.display()
                );
                drop(snapshot);
                return self
                    .fail_worktree_overlay(
                        overlay,
                        &worktree_key,
                        &canonical_key,
                        message,
                        Some(profile_name.clone()),
                        Some(configured_embedding_fingerprint.clone()),
                        false,
                        RepoChangeSummary::default(),
                        "canonical_missing",
                    )
                    .await;
            }
        }

        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.worktrees.insert(
                    worktree_key.clone(),
                    WorktreeSnapshotEntry::indexing(
                        canonical_key.clone(),
                        overlay.resolution.repo_identity.clone(),
                        overlay.resolution.overlay_id.clone(),
                        Some(profile_name.clone()),
                        Some(configured_embedding_fingerprint.clone()),
                    ),
                );
                if let Some(entry) = snapshot.worktrees.get_mut(&worktree_key) {
                    entry.overlay_status = Some("running".to_string());
                }
            })
            .await?;

        let mut attempted_changes = RepoChangeSummary::default();
        let outcome = async {
            self.remove_overlay_state_if_present(&overlay.resolution.overlay_id)
                .await?;
            let canonical_root = ctx.canonical_root.clone();
            let worktree_root = ctx.requested_root.clone();
            let canonical_custom_extensions = custom_extensions.to_vec();
            let canonical_ignore_patterns = ignore_patterns.to_vec();
            let canonical_files = run_low_priority_blocking("scan_canonical_for_overlay", move || {
                scan_repo(
                    &canonical_root,
                    &canonical_custom_extensions,
                    &canonical_ignore_patterns,
                )
            })
            .await?;
            let worktree_custom_extensions = custom_extensions.to_vec();
            let worktree_ignore_patterns = ignore_patterns.to_vec();
            let worktree_files = run_low_priority_blocking("scan_worktree_overlay", move || {
                scan_repo(
                    &worktree_root,
                    &worktree_custom_extensions,
                    &worktree_ignore_patterns,
                )
            })
            .await?;

            let live_canonical_hashes = canonical_files
                .iter()
                .map(|(path, file)| (path.clone(), file.hash.clone()))
                .collect::<BTreeMap<_, _>>();
            let canonical_merkle_path =
                merkle_snapshot_path(&self.inner.config.merkle_dir(), &ctx.canonical_root);
            let canonical_hashes = load_merkle_snapshot(&canonical_merkle_path)
                .await
                .ok()
                .filter(MerkleSnapshot::is_compatible)
                .map(|snapshot| snapshot.file_hashes.into_iter().collect::<BTreeMap<_, _>>())
                .unwrap_or(live_canonical_hashes);
            let worktree_hashes = worktree_files
                .iter()
                .map(|(path, file)| (path.clone(), file.hash.clone()))
                .collect::<BTreeMap<_, _>>();
            let diff = diff_files(&canonical_hashes, &worktree_hashes);
            let to_index = diff
                .added
                .iter()
                .chain(diff.modified.iter())
                .filter_map(|path| worktree_files.get(path).cloned())
                .collect::<Vec<_>>();
            let overlay_bytes = to_index
                .iter()
                .map(|file| file.bytes)
                .fold(0u64, u64::saturating_add);
            let changed_files = to_index.len() as u64;
            let deleted_files = diff.removed.len() as u64;
            let changes = RepoChangeSummary {
                added: diff.added.len() as u64,
                modified: diff.modified.len() as u64,
                removed: deleted_files,
            };
            attempted_changes = changes.clone();
            let over_cap = changed_files as usize > self.inner.config.worktrees.max_overlay_files
                || overlay_bytes > self.inner.config.worktrees.max_overlay_bytes;

            if over_cap {
                self.clear_worktree_overlay_indexes(overlay).await?;
                let state = OverlayIndexState {
                    canonical_root: canonical_key.clone(),
                    worktree_root: worktree_key.clone(),
                    repo_identity: overlay.resolution.repo_identity.clone(),
                    overlay_id: overlay.resolution.overlay_id.clone(),
                    replaced_paths: diff.modified.clone(),
                    deleted_paths: diff.removed.clone(),
                    indexed_hashes: Vec::new(),
                    changed_files,
                    deleted_files,
                    overlay_bytes,
                    overlay_status: Some("too_large".to_string()),
                    embedding_profile: Some(profile_name.clone()),
                    embedding_fingerprint: Some(configured_embedding_fingerprint.clone()),
                };
                self.save_overlay_state(&state).await?;
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        snapshot.worktrees.insert(
                            worktree_key.clone(),
                            WorktreeSnapshotEntry::indexed(
                                canonical_key.clone(),
                                overlay.resolution.repo_identity.clone(),
                                overlay.resolution.overlay_id.clone(),
                                "too_large",
                                changed_files,
                                deleted_files,
                                overlay_bytes,
                                Some(profile_name.clone()),
                                Some(configured_embedding_fingerprint.clone()),
                            ),
                        );
                        if let Some(entry) = snapshot.worktrees.get_mut(&worktree_key) {
                            entry.overlay_mismatch_reason = Some(format!(
                                "overlay has {changed_files} changed/new files and {overlay_bytes} bytes, exceeding configured caps"
                            ));
                        }
                    })
                    .await?;
                return Ok(RepoIndexResult {
                    repo: worktree_key.clone(),
                    indexed_files: Some(0),
                    total_chunks: Some(0),
                    index_status: Some("too_large".to_string()),
                    full_reindex: false,
                    changes,
                    error: None,
                });
            }

            self.clear_worktree_overlay_indexes(overlay).await?;
            if !to_index.is_empty() {
                self.inner
                    .milvus
                    .create_hybrid_collection(
                        &overlay.chunk_collection,
                        configured_embedding_dimension,
                        &format!("codebasePath:{}", ctx.requested_root.display()),
                    )
                    .await?;
                self.inner
                    .milvus
                    .create_hybrid_collection(
                        &overlay.symbol_collection,
                        configured_embedding_dimension,
                        &format!("symbolCodebasePath:{}", ctx.requested_root.display()),
                    )
                    .await?;
            }

            let processing = self
                .process_files(
                    ProcessFilesPlan {
                        repo: &ctx.requested_root,
                        storage_repo: &overlay.storage_root,
                        repo_key: &overlay.repo_key,
                        profile_name: &profile_name,
                        collections: IndexCollections {
                            chunk: &overlay.chunk_collection,
                            symbol: (!to_index.is_empty())
                                .then_some(overlay.symbol_collection.as_str()),
                        },
                        splitter,
                        total_files: 0,
                        mode,
                    },
                    &to_index,
                )
                .await?;
            let indexed_hashes = worktree_hashes
                .iter()
                .filter(|(path, _)| processing.indexed_paths.contains(*path))
                .map(|(path, hash)| (path.clone(), hash.clone()))
                .collect::<Vec<_>>();
            let state = OverlayIndexState {
                canonical_root: canonical_key.clone(),
                worktree_root: worktree_key.clone(),
                repo_identity: overlay.resolution.repo_identity.clone(),
                overlay_id: overlay.resolution.overlay_id.clone(),
                replaced_paths: diff.modified.clone(),
                deleted_paths: diff.removed.clone(),
                indexed_hashes,
                changed_files,
                deleted_files,
                overlay_bytes,
                overlay_status: Some("completed".to_string()),
                embedding_profile: Some(profile_name.clone()),
                embedding_fingerprint: Some(configured_embedding_fingerprint.clone()),
            };
            let overlay_storage_root = overlay.storage_root.clone();
            let local_index = self.inner.local_index.clone();
            let chunk_coverage =
                run_low_priority_blocking("count_overlay_chunk_coverage", move || {
                    local_index.chunk_coverage(&overlay_storage_root)
                })
                .await?;
            let coverage = IndexCoverage {
                indexed_files: chunk_coverage.indexed_files,
                total_chunks: chunk_coverage.total_chunks,
            };
            let index_status = index_status_for_coverage(processing.status, coverage);
            if !to_index.is_empty()
                && let Err(error) = self
                    .maintain_vector_collections(
                        &[
                            overlay.chunk_collection.clone(),
                            overlay.symbol_collection.clone(),
                        ],
                        true,
                    )
                    .await
            {
                bail!("vector maintenance failed: {error}");
            }
            self.save_overlay_state(&state).await?;
            self.inner
                .snapshot
                .update(|snapshot| {
                    snapshot.worktrees.insert(
                        worktree_key.clone(),
                        WorktreeSnapshotEntry::indexed(
                            canonical_key.clone(),
                            overlay.resolution.repo_identity.clone(),
                            overlay.resolution.overlay_id.clone(),
                            index_status.clone(),
                            changed_files,
                            deleted_files,
                            overlay_bytes,
                            Some(profile_name.clone()),
                            Some(configured_embedding_fingerprint.clone()),
                        ),
                    );
                })
                .await?;
            Ok::<_, anyhow::Error>(RepoIndexResult {
                repo: worktree_key.clone(),
                indexed_files: Some(coverage.indexed_files),
                total_chunks: Some(coverage.total_chunks),
                index_status: Some(index_status),
                full_reindex: false,
                changes,
                error: None,
            })
        }
        .await;

        match outcome {
            Ok(result) => Ok(result),
            Err(error) => {
                self.fail_worktree_overlay(
                    overlay,
                    &worktree_key,
                    &canonical_key,
                    error.to_string(),
                    Some(profile_name),
                    Some(configured_embedding_fingerprint),
                    force,
                    attempted_changes,
                    "failed",
                )
                .await
            }
        }
    }

    async fn index_repo(
        &self,
        repo: &Path,
        force: bool,
        splitter: SplitterKind,
        custom_extensions: &[String],
        ignore_patterns: &[String],
        mode: IndexExecutionMode,
    ) -> Result<RepoIndexResult> {
        validate_absolute_repo_path(repo)?;
        if !repo.exists() {
            return self.clear_removed_repo(repo, force).await;
        }
        validate_repo_path(repo)?;
        let ctx = self.repo_context(repo)?;
        if ctx.overlay.is_some() {
            return self
                .index_worktree_overlay(
                    ctx,
                    force,
                    splitter,
                    custom_extensions,
                    ignore_patterns,
                    mode,
                )
                .await;
        }
        let repo_key = repo.display().to_string();
        let profile_name = self.embedding_profile_name_for_repo(repo)?.to_string();
        let configured_embedding_fingerprint = self.embedding_fingerprint_for_repo(repo).await?;
        let configured_embedding_dimension = self.embedding_dimension_for_repo(repo).await?;
        let snapshot_before = self.inner.snapshot.read().await?;
        let repo_identity = repo_embedding_identity_status_for_snapshot(
            &snapshot_before,
            snapshot_before.codebases.get(&repo_key),
            &profile_name,
            Some(configured_embedding_fingerprint.clone()),
        );
        drop(snapshot_before);
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.codebases.insert(
                    repo_key.clone(),
                    SnapshotEntry::indexing(
                        0.0,
                        "running",
                        Some(profile_name.clone()),
                        Some(configured_embedding_fingerprint.clone()),
                    ),
                );
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
        let previous_snapshot = if force || !collection_exists || repo_identity.reason.is_some() {
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
                        configured_embedding_dimension,
                        &format!("codebasePath:{}", repo.display()),
                    )
                    .await?;
                self.inner
                    .milvus
                    .create_hybrid_collection(
                        &symbol_collection_name,
                        configured_embedding_dimension,
                        &format!("symbolCodebasePath:{}", repo.display()),
                    )
                    .await?;
                let files = current_files.values().cloned().collect::<Vec<_>>();
                let processing = self
                    .process_files(
                        ProcessFilesPlan {
                            repo,
                            storage_repo: repo,
                            repo_key: &repo_key,
                            profile_name: &profile_name,
                            collections: IndexCollections {
                                chunk: &collection_name,
                                symbol: Some(&symbol_collection_name),
                            },
                            splitter,
                            total_files: current_files.len(),
                            mode,
                        },
                        &files,
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
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                let chunk_coverage =
                    run_low_priority_blocking("count_repo_chunk_coverage", move || {
                        local_index.chunk_coverage(&repo_path)
                    })
                    .await?;
                let coverage = IndexCoverage {
                    indexed_files: chunk_coverage.indexed_files,
                    total_chunks: chunk_coverage.total_chunks,
                };
                Ok::<(ProcessFilesResult, RepoChangeSummary, IndexCoverage), anyhow::Error>((
                    processing, changes, coverage,
                ))
            } else {
                let symbol_collection_ready = if symbol_collection_exists {
                    true
                } else {
                    self.inner
                        .milvus
                        .create_hybrid_collection(
                            &symbol_collection_name,
                            configured_embedding_dimension,
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
                        ProcessFilesPlan {
                            repo,
                            storage_repo: repo,
                            repo_key: &repo_key,
                            profile_name: &profile_name,
                            collections: IndexCollections {
                                chunk: &collection_name,
                                symbol: symbol_collection_ready
                                    .then_some(symbol_collection_name.as_str()),
                            },
                            splitter,
                            total_files: to_index.len(),
                            mode,
                        },
                        &to_index,
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
                let repo_path = repo.to_path_buf();
                let local_index = self.inner.local_index.clone();
                let chunk_coverage =
                    run_low_priority_blocking("count_repo_chunk_coverage", move || {
                        local_index.chunk_coverage(&repo_path)
                    })
                    .await?;
                let coverage = IndexCoverage {
                    indexed_files: chunk_coverage.indexed_files,
                    total_chunks: chunk_coverage.total_chunks,
                };
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
                Ok((processing, changes, coverage))
            }
        }
        .await;

        match outcome {
            Ok((processing, changes, coverage)) => {
                if vector_release_needed(full_reindex, &changes)
                    && let Err(error) = self
                        .maintain_vector_collections(
                            &[collection_name.clone(), symbol_collection_name.clone()],
                            vector_flush_needed(full_reindex, &changes),
                        )
                        .await
                {
                    let message = format!("vector maintenance failed: {error}");
                    self.inner
                        .snapshot
                        .update(|snapshot| {
                            let last_progress =
                                snapshot.codebases.get(&repo_key).and_then(|entry| {
                                    entry
                                        .indexing_percentage
                                        .or(entry.last_attempted_percentage)
                                });
                            snapshot.codebases.insert(
                                repo_key.clone(),
                                SnapshotEntry::failed(
                                    message.clone(),
                                    last_progress,
                                    Some(profile_name.clone()),
                                    Some(configured_embedding_fingerprint.clone()),
                                ),
                            );
                        })
                        .await?;
                    return Ok(RepoIndexResult {
                        repo: repo_key,
                        indexed_files: None,
                        total_chunks: None,
                        index_status: Some("failed".to_string()),
                        full_reindex,
                        changes,
                        error: Some(message),
                    });
                }
                let fingerprint = fingerprint_repo(repo).ok();
                let index_status = index_status_for_coverage(processing.status, coverage);
                self.inner
                    .snapshot
                    .update(|snapshot| {
                        let entry =
                            snapshot
                                .codebases
                                .entry(repo_key.clone())
                                .or_insert_with(|| {
                                    SnapshotEntry::indexing(
                                        0.0,
                                        "running",
                                        Some(profile_name.clone()),
                                        Some(configured_embedding_fingerprint.clone()),
                                    )
                                });
                        *entry = SnapshotEntry::indexed_with_status(
                            Some(coverage.indexed_files),
                            Some(coverage.total_chunks),
                            index_status.clone(),
                            Some(profile_name.clone()),
                            Some(configured_embedding_fingerprint.clone()),
                        );
                        if let Some(fingerprint) = &fingerprint {
                            apply_fingerprint(entry, fingerprint);
                        }
                    })
                    .await?;
                self.invalidate_worktree_overlays_for_canonical(&repo_key)
                    .await?;

                Ok(RepoIndexResult {
                    repo: repo_key,
                    indexed_files: Some(coverage.indexed_files),
                    total_chunks: Some(coverage.total_chunks),
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
                            SnapshotEntry::failed(
                                message.clone(),
                                last_progress,
                                Some(profile_name.clone()),
                                Some(configured_embedding_fingerprint.clone()),
                            ),
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

    async fn clear_removed_repo(&self, repo: &Path, force: bool) -> Result<RepoIndexResult> {
        let removed_files = self.previous_indexed_file_count(repo).await;
        self.clear_repo_indexes(repo).await?;
        Ok(removed_repo_index_result(repo, removed_files, force))
    }

    async fn clear_removed_worktree_overlay(
        &self,
        repo: &Path,
        force: bool,
    ) -> Result<Option<RepoIndexResult>> {
        let repo_key = repo.display().to_string();
        let entry = {
            let snapshot = self.inner.snapshot.read().await?;
            snapshot.worktrees.get(&repo_key).cloned()
        };
        let Some(entry) = entry else {
            return Ok(None);
        };

        let overlay = worktree_overlay_context_from_snapshot(repo, &entry);
        self.clear_worktree_overlay_indexes(&overlay).await?;
        let state_path = self.overlay_state_path(&entry.overlay_id);
        if state_path.exists() {
            tokio::fs::remove_file(&state_path)
                .await
                .with_context(|| format!("removing {}", state_path.display()))?;
        }
        self.inner
            .snapshot
            .update(|snapshot| {
                snapshot.worktrees.remove(&repo_key);
            })
            .await?;

        let removed_files = entry.changed_files.unwrap_or(0);
        Ok(Some(removed_repo_index_result(repo, removed_files, force)))
    }

    async fn previous_indexed_file_count(&self, repo: &Path) -> u64 {
        let merkle_path = merkle_snapshot_path(&self.inner.config.merkle_dir(), repo);
        if !merkle_path.exists() {
            return 0;
        }
        load_merkle_snapshot(&merkle_path)
            .await
            .map(|snapshot| snapshot.file_hashes.len() as u64)
            .unwrap_or(0)
    }

    async fn clear_repo(&self, repo: &Path) -> Result<()> {
        validate_absolute_repo_path(repo)?;
        let ctx = self.repo_context(repo)?;
        if let Some(overlay) = ctx.overlay.as_ref() {
            self.clear_worktree_overlay_indexes(overlay).await?;
            let state_path = self.overlay_state_path(&overlay.resolution.overlay_id);
            if state_path.exists() {
                tokio::fs::remove_file(&state_path)
                    .await
                    .with_context(|| format!("removing {}", state_path.display()))?;
            }
            let worktree_key = overlay.resolution.worktree_root.display().to_string();
            self.inner
                .snapshot
                .update(|snapshot| {
                    snapshot.worktrees.remove(&worktree_key);
                })
                .await?;
            return Ok(());
        }
        self.clear_repo_indexes(repo).await
    }

    async fn clear_repo_indexes(&self, repo: &Path) -> Result<()> {
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

    async fn clear_worktree_overlay_indexes(&self, overlay: &WorktreeOverlayContext) -> Result<()> {
        if self
            .inner
            .milvus
            .refresh_collection_presence(&overlay.chunk_collection)
            .await?
        {
            self.inner
                .milvus
                .drop_collection(&overlay.chunk_collection)
                .await?;
        }
        if self
            .inner
            .milvus
            .refresh_collection_presence(&overlay.symbol_collection)
            .await?
        {
            self.inner
                .milvus
                .drop_collection(&overlay.symbol_collection)
                .await?;
        }
        let storage_root = overlay.storage_root.clone();
        let local_index = self.inner.local_index.clone();
        run_low_priority_blocking("clear_overlay_local_index", move || {
            local_index.clear_repo(&storage_root)
        })
        .await?;
        let overlay_repo_key = overlay.repo_key.clone();
        let symbol_store = self.inner.symbol_store.clone();
        run_low_priority_blocking("clear_overlay_symbol_rows", move || {
            symbol_store.clear_repo(&overlay_repo_key)
        })
        .await
    }

    async fn invalidate_worktree_overlays_for_canonical(&self, canonical_key: &str) -> Result<()> {
        let snapshot = self.inner.snapshot.read().await?;
        let affected = snapshot
            .worktrees
            .iter()
            .filter(|(_, entry)| entry.canonical_root == canonical_key)
            .filter(|(_, entry)| entry.status != "indexing")
            .map(|(worktree, entry)| (worktree.clone(), entry.overlay_id.clone()))
            .collect::<Vec<_>>();
        drop(snapshot);

        if affected.is_empty() {
            return Ok(());
        }

        let reason = "canonical index refreshed; refresh this worktree overlay".to_string();
        self.inner
            .snapshot
            .update(|snapshot| {
                for (worktree_key, _) in &affected {
                    if let Some(entry) = snapshot.worktrees.get_mut(worktree_key) {
                        entry.overlay_status = Some("stale".to_string());
                        entry.overlay_mismatch_reason = Some(reason.clone());
                        entry.last_updated = Some(crate::snapshot::timestamp());
                    }
                }
            })
            .await?;

        for (_, overlay_id) in affected {
            if let Some(mut state) = self.load_overlay_state_by_id(&overlay_id).await? {
                state.overlay_status = Some("stale".to_string());
                self.save_overlay_state(&state).await?;
            }
        }

        Ok(())
    }

    async fn maintain_vector_collections(&self, collections: &[String], flush: bool) -> Result<()> {
        let mut collections = collections.to_vec();
        collections.sort();
        collections.dedup();

        if flush {
            for collection in &collections {
                self.inner
                    .milvus
                    .flush_collection(collection)
                    .await
                    .with_context(|| format!("flushing Milvus collection {collection}"))?;
            }
        }

        for collection in &collections {
            self.inner
                .milvus
                .release_collection_if_loaded(collection)
                .await
                .with_context(|| format!("releasing Milvus collection {collection}"))?;
        }

        Ok(())
    }

    async fn search_repo_context(
        &self,
        ctx: &RepoContext,
        request: &SearchRequest,
        plan: &SearchPlan,
        canonical_query_vector: Option<&[f32]>,
        overlay_query_vector: Option<&[f32]>,
        limit: usize,
        filter_expression: Option<&str>,
    ) -> Result<SearchContextResult<RepoSearchHit>> {
        let canonical_key = ctx.canonical_root.display().to_string();
        let mut hits = self
            .search_repo_index(
                &ctx.requested_root,
                &ctx.canonical_root,
                &canonical_key,
                &collection_name(&ctx.canonical_root),
                &symbol_collection_name(&ctx.canonical_root),
                request,
                plan,
                canonical_query_vector,
                limit,
                filter_expression,
            )
            .await?;

        let Some(overlay) = ctx.overlay.as_ref() else {
            return Ok(SearchContextResult {
                hits,
                warnings: Vec::new(),
            });
        };

        let mut warnings = Vec::new();
        let overlay_state = match self.load_overlay_state(overlay).await {
            Ok(state) => state,
            Err(error) => {
                warnings.push(overlay_state_load_warning(&error));
                None
            }
        };
        let suppressed_paths = overlay_state
            .as_ref()
            .filter(|state| overlay_state_suppresses_canonical(state))
            .map(overlay_suppressed_paths)
            .unwrap_or_default();
        if !suppressed_paths.is_empty() {
            hits.retain(|hit| !suppressed_paths.contains(&hit.relative_path));
        }

        let overlay_healthy = overlay_state
            .as_ref()
            .is_some_and(overlay_state_has_search_index);
        if overlay_healthy {
            let overlay_hits = self
                .search_repo_index(
                    &ctx.requested_root,
                    &overlay.storage_root,
                    &overlay.repo_key,
                    &overlay.chunk_collection,
                    &overlay.symbol_collection,
                    request,
                    plan,
                    overlay_query_vector,
                    limit,
                    filter_expression,
                )
                .await?;
            hits.extend(overlay_hits);
        }

        hits.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(SearchContextResult { hits, warnings })
    }

    #[allow(clippy::too_many_arguments)]
    async fn search_repo_index(
        &self,
        display_repo: &Path,
        storage_repo: &Path,
        symbol_repo_key: &str,
        collection_name: &str,
        symbol_collection_name: &str,
        request: &SearchRequest,
        plan: &SearchPlan,
        query_vector: Option<&[f32]>,
        limit: usize,
        filter_expression: Option<&str>,
    ) -> Result<Vec<RepoSearchHit>> {
        let dense_hits = if let Some(query_vector) = query_vector {
            if plan.dense_weight > 0.0 {
                let _dense_permit = self.acquire_dense_budget().await?;
                if self.inner.milvus.has_collection(collection_name).await? {
                    self.inner
                        .milvus
                        .search_dense(collection_name, query_vector, limit, filter_expression)
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
                let repo_path = storage_repo.to_path_buf();
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
            self.has_collection_on_search_path(symbol_collection_name)
                .await?
        } else {
            false
        };

        let symbol_hits = self
            .search_symbol_repo(
                storage_repo,
                symbol_repo_key,
                symbol_collection_name,
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
        accumulate_dense_hits(&mut merged, display_repo, dense_hits, plan.dense_weight);
        accumulate_lexical_hits(&mut merged, display_repo, lexical_hits, plan.lexical_weight);
        self.accumulate_symbol_hits(
            &mut merged,
            display_repo,
            storage_repo,
            symbol_hits,
            plan.symbol_weight,
        )
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

    async fn search_symbol_context(
        &self,
        ctx: &RepoContext,
        request: &SymbolSearchScopeRequest,
        flavor: QueryKind,
        limit: usize,
        canonical_query_vector: Option<&[f32]>,
        overlay_query_vector: Option<&[f32]>,
        lexical_share: f64,
        semantic_share: f64,
    ) -> Result<SearchContextResult<RankedSymbolHit>> {
        let canonical_collection = symbol_collection_name(&ctx.canonical_root);
        let canonical_collection_exists = if semantic_share > 0.0 {
            self.has_collection_on_search_path(&canonical_collection)
                .await?
        } else {
            false
        };
        let canonical_key = ctx.canonical_root.display().to_string();
        let mut hits = self
            .search_symbol_repo(
                &ctx.canonical_root,
                &canonical_key,
                &canonical_collection,
                request,
                flavor,
                limit,
                SymbolFusionConfig {
                    query_vector: canonical_query_vector,
                    lexical_share,
                    semantic_share,
                    symbol_collection_exists: canonical_collection_exists,
                },
            )
            .await?;

        let Some(overlay) = ctx.overlay.as_ref() else {
            return Ok(SearchContextResult {
                hits,
                warnings: Vec::new(),
            });
        };

        let mut warnings = Vec::new();
        let overlay_state = match self.load_overlay_state(overlay).await {
            Ok(state) => state,
            Err(error) => {
                warnings.push(overlay_state_load_warning(&error));
                None
            }
        };
        let suppressed_paths = overlay_state
            .as_ref()
            .filter(|state| overlay_state_suppresses_canonical(state))
            .map(overlay_suppressed_paths)
            .unwrap_or_default();
        if !suppressed_paths.is_empty() {
            hits.retain(|hit| !suppressed_paths.contains(&hit.relative_path));
        }

        let overlay_healthy = overlay_state
            .as_ref()
            .is_some_and(overlay_state_has_search_index);
        if overlay_healthy {
            let overlay_collection_exists = if semantic_share > 0.0 {
                self.has_collection_on_search_path(&overlay.symbol_collection)
                    .await?
            } else {
                false
            };
            let overlay_hits = self
                .search_symbol_repo(
                    &overlay.storage_root,
                    &overlay.repo_key,
                    &overlay.symbol_collection,
                    request,
                    flavor,
                    limit,
                    SymbolFusionConfig {
                        query_vector: overlay_query_vector,
                        lexical_share,
                        semantic_share,
                        symbol_collection_exists: overlay_collection_exists,
                    },
                )
                .await?;
            hits.extend(overlay_hits);
        }

        hits.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);
        Ok(SearchContextResult { hits, warnings })
    }

    async fn search_symbol_repo(
        &self,
        storage_repo: &Path,
        symbol_repo_key: &str,
        symbol_collection_name: &str,
        request: &SymbolSearchScopeRequest,
        flavor: QueryKind,
        limit: usize,
        fusion: SymbolFusionConfig<'_>,
    ) -> Result<Vec<RankedSymbolHit>> {
        let indexed_hits = self
            .run_search_lexical_blocking("search_symbol_index", {
                let repo_path = storage_repo.to_path_buf();
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
            if let Some(query_vector) = fusion.query_vector {
                let _dense_permit = self.acquire_dense_budget().await?;
                self.inner
                    .milvus
                    .search_dense(
                        symbol_collection_name,
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
                let repo_key = symbol_repo_key.to_string();
                move || symbol_store.symbols_by_repo_and_ids(&repo_key, &symbol_ids)
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
        display_repo: &Path,
        storage_repo: &Path,
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
                let repo_path = storage_repo.to_path_buf();
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
                let key = format!("{}:{}", display_repo.display(), chunk_hit.id);
                let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
                    repo: display_repo.display().to_string(),
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
                let key = format!("{}:{id}", display_repo.display());
                let entry = merged.entry(key).or_insert_with(|| RepoSearchHit {
                    repo: display_repo.display().to_string(),
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
        let identity = self.repo_embedding_identity_status(snapshot, repo).await?;

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
                embedding_profile: Some(identity.profile_name),
                configured_embedding_fingerprint: identity.configured_fingerprint,
                stored_embedding_fingerprint: identity.stored_fingerprint,
                embedding_mismatch_reason: identity.reason,
                repo_type: None,
                canonical_repo_label: None,
                overlay_status: None,
                changed_files: None,
                deleted_files: None,
                overlay_bytes: None,
                overlay_mismatch_reason: None,
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
                embedding_profile: entry.embedding_profile.or(Some(identity.profile_name)),
                configured_embedding_fingerprint: identity.configured_fingerprint,
                stored_embedding_fingerprint: identity.stored_fingerprint,
                embedding_mismatch_reason: identity.reason,
                repo_type: None,
                canonical_repo_label: None,
                overlay_status: None,
                changed_files: None,
                deleted_files: None,
                overlay_bytes: None,
                overlay_mismatch_reason: None,
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
                embedding_profile: Some(identity.profile_name),
                configured_embedding_fingerprint: identity.configured_fingerprint,
                stored_embedding_fingerprint: identity.stored_fingerprint,
                embedding_mismatch_reason: identity.reason,
                repo_type: None,
                canonical_repo_label: None,
                overlay_status: None,
                changed_files: None,
                deleted_files: None,
                overlay_bytes: None,
                overlay_mismatch_reason: None,
            },
        })
    }

    async fn status_for_worktree_overlay(
        &self,
        snapshot: &Snapshot,
        ctx: &RepoContext,
    ) -> Result<RepoStatus> {
        let Some(overlay) = ctx.overlay.as_ref() else {
            bail!("internal error: missing worktree overlay context");
        };
        let repo_key = ctx.requested_root.display().to_string();
        let canonical_key = ctx.canonical_root.display().to_string();
        let entry = snapshot.worktrees.get(&repo_key).cloned();
        let profile_name = self.overlay_profile_name(ctx)?.to_string();
        let configured_fingerprint = self.inner.embedding.fingerprint(&profile_name).await.ok();
        let stored_fingerprint = entry
            .as_ref()
            .and_then(|entry| entry.embedding_fingerprint.clone());
        let fingerprint_mismatch = match (
            configured_fingerprint.as_deref(),
            stored_fingerprint.as_deref(),
        ) {
            (Some(configured), Some(stored)) if configured != stored => Some(format!(
                "overlay embedding fingerprint mismatch: local state is `{stored}`, current config is `{configured}`"
            )),
            _ => None,
        };
        let overlay_mismatch_reason = entry
            .as_ref()
            .and_then(|entry| entry.overlay_mismatch_reason.clone())
            .or_else(|| fingerprint_mismatch.clone());
        let coverage = {
            let storage_root = overlay.storage_root.clone();
            let local_index = self.inner.local_index.clone();
            self.run_search_lexical_blocking("overlay_status_chunk_coverage", move || {
                local_index.chunk_coverage(&storage_root)
            })
            .await
            .ok()
        };
        let default_overlay_status = if snapshot.codebases.contains_key(&canonical_key) {
            "not_indexed"
        } else {
            "canonical_missing"
        };

        Ok(match entry {
            Some(entry) => RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                collection_name: overlay.chunk_collection.clone(),
                status: entry.status,
                indexed_files: coverage
                    .as_ref()
                    .map(|coverage| coverage.indexed_files)
                    .or(entry.changed_files),
                total_chunks: coverage.as_ref().map(|coverage| coverage.total_chunks),
                index_status: entry.overlay_status.clone(),
                indexing_percentage: None,
                last_attempted_percentage: None,
                error_message: entry.overlay_mismatch_reason.clone(),
                embedding_profile: entry.embedding_profile.or(Some(profile_name)),
                configured_embedding_fingerprint: configured_fingerprint,
                stored_embedding_fingerprint: stored_fingerprint,
                embedding_mismatch_reason: fingerprint_mismatch,
                repo_type: Some("worktree_overlay".to_string()),
                canonical_repo_label: Some(repo_basename(&canonical_key)),
                overlay_status: entry.overlay_status,
                changed_files: entry.changed_files,
                deleted_files: entry.deleted_files,
                overlay_bytes: entry.overlay_bytes,
                overlay_mismatch_reason,
            },
            None => RepoStatus {
                repo: repo_key.clone(),
                repo_label: repo_basename(&repo_key),
                collection_name: overlay.chunk_collection.clone(),
                status: "not_indexed".to_string(),
                indexed_files: None,
                total_chunks: None,
                index_status: Some(default_overlay_status.to_string()),
                indexing_percentage: None,
                last_attempted_percentage: None,
                error_message: None,
                embedding_profile: Some(profile_name),
                configured_embedding_fingerprint: configured_fingerprint,
                stored_embedding_fingerprint: None,
                embedding_mismatch_reason: None,
                repo_type: Some("worktree_overlay".to_string()),
                canonical_repo_label: Some(repo_basename(&canonical_key)),
                overlay_status: Some(default_overlay_status.to_string()),
                changed_files: None,
                deleted_files: None,
                overlay_bytes: None,
                overlay_mismatch_reason,
            },
        })
    }

    pub async fn prune_stale_vector_collections(&self) -> Result<Vec<String>> {
        let report = self.vector_hygiene_report().await?;
        for collection in &report.stale_collections {
            self.inner
                .milvus
                .drop_collection(collection)
                .await
                .with_context(|| format!("dropping stale Milvus collection {collection}"))?;
        }
        Ok(report.stale_collections)
    }

    pub async fn release_loaded_vector_collections(&self) -> Result<Vec<String>> {
        let report = self.vector_hygiene_report().await?;
        for collection in &report.loaded_collections {
            self.inner
                .milvus
                .release_collection_if_loaded(collection)
                .await
                .with_context(|| format!("releasing loaded Milvus collection {collection}"))?;
        }
        Ok(report.loaded_collections)
    }

    pub async fn drop_vector_collections(&self, collections: &[String]) -> Result<Vec<String>> {
        let mut dropped = Vec::new();
        for collection in collections {
            self.inner
                .milvus
                .drop_collection(collection)
                .await
                .with_context(|| format!("dropping Milvus collection {collection}"))?;
            dropped.push(collection.clone());
        }
        Ok(dropped)
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

    async fn record_indexing_progress(
        &self,
        repo: &Path,
        repo_key: &str,
        progress: f64,
    ) -> Result<()> {
        let profile_name = self
            .embedding_profile_name_for_repo(repo)
            .ok()
            .map(str::to_string);
        let configured_embedding_fingerprint = self.embedding_fingerprint_for_repo(repo).await.ok();
        self.inner
            .snapshot
            .update(|snapshot| {
                let entry = snapshot
                    .codebases
                    .entry(repo_key.to_string())
                    .or_insert_with(|| {
                        SnapshotEntry::indexing(
                            progress,
                            "running",
                            profile_name.clone(),
                            configured_embedding_fingerprint.clone(),
                        )
                    });
                entry.embedding_profile = profile_name.clone();
                entry.embedding_fingerprint = configured_embedding_fingerprint.clone();
                entry.set_indexing_progress(progress, "running");
            })
            .await?;
        Ok(())
    }

    async fn write_file_symbols(
        &self,
        repo_key: &str,
        relative_path: &str,
        symbols: &[IndexedSymbol],
    ) -> Result<()> {
        let repo_key = repo_key.to_string();
        let relative_path = relative_path.to_string();
        let symbol_store = self.inner.symbol_store.clone();
        let symbols = symbols.to_vec();
        run_low_priority_blocking("write_file_symbols", move || {
            symbol_store.replace_file_symbols(&repo_key, &relative_path, &symbols)?;
            Ok(())
        })
        .await
    }

    async fn flush_symbol_index_replacements(
        &self,
        repo: &Path,
        pending_replacements: &mut Vec<PendingSymbolIndexReplacement>,
    ) -> Result<()> {
        let replacements = std::mem::take(pending_replacements)
            .into_iter()
            .map(|replacement| (replacement.relative_path, replacement.documents))
            .collect::<Vec<_>>();
        if replacements.is_empty() {
            return Ok(());
        }
        let repo_path = repo.to_path_buf();
        let local_index = self.inner.local_index.clone();
        run_low_priority_blocking("write_symbol_lexical_docs", move || {
            local_index.replace_symbol_docs_batch(&repo_path, &replacements)
        })
        .await
    }

    async fn process_files(
        &self,
        plan: ProcessFilesPlan<'_>,
        files: &[RepoFile],
    ) -> Result<ProcessFilesResult> {
        let mut pending_chunks = Vec::new();
        let mut pending_symbols = Vec::new();
        let mut pending_symbol_index_replacements = Vec::new();
        let mut indexed_paths = HashSet::new();
        let mut processed_files = 0u64;
        let mut total_chunks = 0u64;
        let mut status = IndexCompletionStatus::Completed;
        let mut progress_tracker = ProgressTracker::new();

        let repo_path = plan.repo.to_path_buf();
        let repo_key_owned = plan.repo_key.to_string();
        let work_items = files.to_vec();
        let mut prepared_files = futures::stream::iter(work_items.into_iter().map(|file| {
            let engine = self.clone();
            let repo = repo_path.clone();
            let repo_key = repo_key_owned.clone();
            async move {
                engine
                    .prepare_repo_file(&repo, &repo_key, file, plan.splitter)
                    .await
            }
        }))
        .buffer_unordered(plan.mode.file_prepare_parallelism());

        while let Some(prepared) = prepared_files.next().await {
            let Some(prepared) = prepared? else {
                continue;
            };
            let chunk_count = prepared.chunks.len() as u64;
            if total_chunks + chunk_count > CHUNK_LIMIT as u64 {
                status = IndexCompletionStatus::LimitReached;
                break;
            }
            self.write_file_symbols(plan.repo_key, &prepared.relative_path, &prepared.symbols)
                .await?;
            let symbol_index_docs = prepared
                .symbols
                .iter()
                .map(|symbol| SymbolIndexDoc {
                    symbol_id: symbol.symbol_id.clone(),
                    relative_path: symbol.relative_path.clone(),
                    basename: prepared.basename.clone(),
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
            pending_symbol_index_replacements.push(PendingSymbolIndexReplacement {
                relative_path: prepared.relative_path.clone(),
                documents: symbol_index_docs,
            });
            if pending_symbol_index_replacements.len() >= SYMBOL_INDEX_REPLACEMENT_BATCH_SIZE {
                self.flush_symbol_index_replacements(
                    plan.storage_repo,
                    &mut pending_symbol_index_replacements,
                )
                .await?;
            }
            if plan.collections.symbol.is_some() {
                pending_symbols.extend(prepared.symbols.iter().cloned().map(|symbol| {
                    PendingSymbolDocument {
                        symbol,
                        basename: prepared.basename.clone(),
                        extension: prepared.extension.clone(),
                    }
                }));
                if pending_symbols.len() >= EMBEDDING_BATCH_SIZE {
                    self.flush_symbol_documents(
                        plan.profile_name,
                        plan.collections.symbol,
                        &mut pending_symbols,
                    )
                    .await?;
                }
            }

            for chunk in prepared.chunks {
                pending_chunks.push(PendingChunk {
                    chunk,
                    indexed_at: prepared.indexed_at.clone(),
                    file_hash: prepared.file_hash.clone(),
                });
                if pending_chunks.len() >= EMBEDDING_BATCH_SIZE {
                    self.flush_chunks(
                        plan.repo,
                        plan.storage_repo,
                        plan.profile_name,
                        plan.collections.chunk,
                        &mut pending_chunks,
                    )
                    .await?;
                }
            }

            indexed_paths.insert(prepared.relative_path);
            processed_files += 1;
            total_chunks += chunk_count;
            if plan.total_files > 0 {
                let progress = (processed_files as f64 / plan.total_files as f64) * 100.0;
                if progress_tracker.should_persist(progress) {
                    self.record_indexing_progress(plan.repo, plan.repo_key, progress)
                        .await?;
                }
            }
        }

        if !pending_chunks.is_empty() {
            self.flush_chunks(
                plan.repo,
                plan.storage_repo,
                plan.profile_name,
                plan.collections.chunk,
                &mut pending_chunks,
            )
            .await?;
        }
        if !pending_symbols.is_empty() {
            self.flush_symbol_documents(
                plan.profile_name,
                plan.collections.symbol,
                &mut pending_symbols,
            )
            .await?;
        }
        if !pending_symbol_index_replacements.is_empty() {
            self.flush_symbol_index_replacements(
                plan.storage_repo,
                &mut pending_symbol_index_replacements,
            )
            .await?;
        }

        Ok(ProcessFilesResult {
            indexed_paths,
            processed_files,
            status,
        })
    }

    async fn prepare_repo_file(
        &self,
        repo: &Path,
        repo_key: &str,
        file: RepoFile,
        splitter: SplitterKind,
    ) -> Result<Option<PreparedRepoFile>> {
        let Some(text) = read_utf8_file(&file.absolute_path).await? else {
            return Ok(None);
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
        let indexed_at = crate::snapshot::timestamp();

        let absolute_path = file.absolute_path.clone();
        let file_hash = file.hash.clone();
        let repo_key = repo_key.to_string();
        let relative_path_for_task = relative_path.clone();
        let indexed_at_for_task = indexed_at.clone();
        let text_for_task = text;
        let file_hash_for_task = file_hash.clone();
        let (chunks, symbols) = tokio::task::spawn_blocking(move || {
            let chunks = split_text(&absolute_path, &text_for_task, splitter)?;
            let symbols = extract_symbols(
                &repo_key,
                &relative_path_for_task,
                &absolute_path,
                &text_for_task,
                &indexed_at_for_task,
                &file_hash_for_task,
            )?;
            Ok::<_, anyhow::Error>((chunks, symbols))
        })
        .await
        .context("joining file preparation task")??;

        Ok(Some(PreparedRepoFile {
            relative_path,
            basename,
            extension,
            indexed_at,
            file_hash,
            chunks,
            symbols,
        }))
    }

    async fn flush_chunks(
        &self,
        repo: &Path,
        storage_repo: &Path,
        profile_name: &str,
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
        let embeddings = self
            .inner
            .embedding
            .embed_documents(profile_name, &contents)
            .await?;

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

        let repo_path = storage_repo.to_path_buf();
        let local_index = self.inner.local_index.clone();
        run_low_priority_blocking("write_chunk_lexical_docs", move || {
            local_index.index_chunks(&repo_path, &chunk_documents)
        })
        .await
    }

    async fn flush_symbol_documents(
        &self,
        profile_name: &str,
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
        let embeddings = self
            .inner
            .embedding
            .embed_documents(profile_name, &contents)
            .await?;

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

fn overlay_collection_name(overlay_id: &str) -> String {
    format!("{OVERLAY_CHUNK_COLLECTION_PREFIX}{overlay_id}")
}

fn overlay_symbol_collection_name(overlay_id: &str) -> String {
    format!("{OVERLAY_SYMBOL_COLLECTION_PREFIX}{overlay_id}")
}

fn overlay_storage_root(overlay_id: &str) -> PathBuf {
    PathBuf::from(OVERLAY_STORAGE_ROOT_PREFIX).join(overlay_id)
}

fn overlay_repo_key(overlay_id: &str) -> String {
    format!("overlay:{overlay_id}")
}

fn worktree_overlay_context_from_snapshot(
    worktree_root: &Path,
    entry: &WorktreeSnapshotEntry,
) -> WorktreeOverlayContext {
    WorktreeOverlayContext {
        resolution: WorktreeResolution {
            worktree_root: worktree_root.to_path_buf(),
            canonical_root: PathBuf::from(&entry.canonical_root),
            repo_identity: entry.repo_identity.clone(),
            overlay_id: entry.overlay_id.clone(),
        },
        storage_root: overlay_storage_root(&entry.overlay_id),
        repo_key: overlay_repo_key(&entry.overlay_id),
        chunk_collection: overlay_collection_name(&entry.overlay_id),
        symbol_collection: overlay_symbol_collection_name(&entry.overlay_id),
    }
}

fn failed_overlay_state(
    overlay: &WorktreeOverlayContext,
    worktree_key: &str,
    canonical_key: &str,
    embedding_profile: Option<String>,
    embedding_fingerprint: Option<String>,
) -> OverlayIndexState {
    OverlayIndexState {
        canonical_root: canonical_key.to_string(),
        worktree_root: worktree_key.to_string(),
        repo_identity: overlay.resolution.repo_identity.clone(),
        overlay_id: overlay.resolution.overlay_id.clone(),
        replaced_paths: Vec::new(),
        deleted_paths: Vec::new(),
        indexed_hashes: Vec::new(),
        changed_files: 0,
        deleted_files: 0,
        overlay_bytes: 0,
        overlay_status: Some("failed".to_string()),
        embedding_profile,
        embedding_fingerprint,
    }
}

fn overlay_suppressed_paths(state: &OverlayIndexState) -> BTreeSet<String> {
    state
        .replaced_paths
        .iter()
        .chain(state.deleted_paths.iter())
        .cloned()
        .collect()
}

fn overlay_state_suppresses_canonical(state: &OverlayIndexState) -> bool {
    matches!(
        state.overlay_status.as_deref(),
        None | Some("completed") | Some("empty") | Some("too_large")
    )
}

fn overlay_state_has_search_index(state: &OverlayIndexState) -> bool {
    matches!(
        state.overlay_status.as_deref(),
        None | Some("completed") | Some("empty")
    )
}

fn overlay_state_has_indexed_path(state: &OverlayIndexState, normalized_path: &str) -> bool {
    overlay_state_has_search_index(state)
        && state
            .indexed_hashes
            .iter()
            .any(|(path, _)| path == normalized_path)
}

fn overlay_lookup_uses_overlay(state: &OverlayIndexState, normalized_path: &str) -> bool {
    overlay_state_has_indexed_path(state, normalized_path)
        || (overlay_state_suppresses_canonical(state)
            && overlay_suppressed_paths(state).contains(normalized_path))
}

fn overlay_state_load_warning(error: &anyhow::Error) -> String {
    format!("overlay state unavailable: {error}")
}

fn expected_vector_collections(repos: &[PathBuf], snapshot: &Snapshot) -> HashSet<String> {
    let mut expected = repos
        .iter()
        .flat_map(|repo| [collection_name(repo), symbol_collection_name(repo)])
        .collect::<HashSet<_>>();
    for worktree in snapshot.worktrees.values() {
        if worktree.status == "indexed" || worktree.status == "indexing" {
            expected.insert(overlay_collection_name(&worktree.overlay_id));
            expected.insert(overlay_symbol_collection_name(&worktree.overlay_id));
        }
    }
    expected
}

fn stale_vector_collections(existing: &[String], expected: &HashSet<String>) -> Vec<String> {
    existing
        .iter()
        .filter(|collection| is_agent_context_vector_collection(collection))
        .filter(|collection| !expected.contains(*collection))
        .cloned()
        .collect()
}

fn is_agent_context_vector_collection(collection: &str) -> bool {
    collection.starts_with(CHUNK_COLLECTION_PREFIX)
        || collection.starts_with(SYMBOL_COLLECTION_PREFIX)
        || collection.starts_with(OVERLAY_CHUNK_COLLECTION_PREFIX)
        || collection.starts_with(OVERLAY_SYMBOL_COLLECTION_PREFIX)
}

fn vector_release_needed(full_reindex: bool, changes: &RepoChangeSummary) -> bool {
    full_reindex || vector_changed_file_count(changes) > 0
}

fn vector_flush_needed(full_reindex: bool, changes: &RepoChangeSummary) -> bool {
    full_reindex || vector_changed_file_count(changes) >= VECTOR_FLUSH_FILE_CHANGE_THRESHOLD
}

fn vector_changed_file_count(changes: &RepoChangeSummary) -> u64 {
    changes
        .added
        .saturating_add(changes.modified)
        .saturating_add(changes.removed)
}

fn removed_repo_index_result(repo: &Path, removed_files: u64, force: bool) -> RepoIndexResult {
    RepoIndexResult {
        repo: repo.display().to_string(),
        indexed_files: None,
        total_chunks: None,
        index_status: Some(REMOVED_REPO_INDEX_STATUS.to_string()),
        full_reindex: force,
        changes: RepoChangeSummary {
            added: 0,
            modified: 0,
            removed: removed_files,
        },
        error: None,
    }
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
            "{}. {} :: {}:{}-{} score={:.6}",
            index + 1,
            hit.repo_label,
            hit.relative_path,
            hit.start_line,
            hit.end_line,
            hit.score
        ));
        lines.push(hit.repo.clone());
        lines.push(truncate_for_display(&hit.content, 320));
    }
    for error in &result.repo_errors {
        lines.push(format!("ERR {}: {}", error.repo, error.error));
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
        let overlay_suffix = if repo.repo_type.as_deref() == Some("worktree_overlay") {
            format!(
                " repo_type=worktree_overlay canonical={} overlay_status={} changed={} deleted={} overlay_bytes={}{}",
                repo.canonical_repo_label.as_deref().unwrap_or("unknown"),
                repo.overlay_status.as_deref().unwrap_or("unknown"),
                repo.changed_files.unwrap_or(0),
                repo.deleted_files.unwrap_or(0),
                repo.overlay_bytes.unwrap_or(0),
                repo.overlay_mismatch_reason
                    .as_ref()
                    .map(|reason| format!(" overlay_reason={reason}"))
                    .unwrap_or_default()
            )
        } else {
            String::new()
        };
        lines.push(format!(
            "{} status={} index_status={} files={} chunks={}{}{}{}",
            repo.repo,
            repo.status,
            repo.index_status.as_deref().unwrap_or("unknown"),
            repo.indexed_files.unwrap_or(0),
            repo.total_chunks.unwrap_or(0),
            progress_suffix,
            error_suffix,
            overlay_suffix
        ));
    }

    lines.join("\n")
}

fn validate_repo_path(repo: &Path) -> Result<()> {
    validate_absolute_repo_path(repo)?;
    if !repo.exists() {
        anyhow::bail!("repo path does not exist: {}", repo.display());
    }
    if !repo.is_dir() {
        anyhow::bail!("repo path is not a directory: {}", repo.display());
    }
    Ok(())
}

fn validate_absolute_repo_path(repo: &Path) -> Result<()> {
    if !repo.is_absolute() {
        anyhow::bail!("repo path must be absolute: {}", repo.display());
    }
    Ok(())
}

fn repo_basename(repo: &str) -> String {
    Path::new(repo)
        .file_name()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_else(|| repo.to_string())
}

fn snapshot_entry_age_secs(
    last_updated: Option<&str>,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<u64> {
    let last_updated = last_updated?;
    let parsed = chrono::DateTime::parse_from_rfc3339(last_updated).ok()?;
    let age = now.signed_duration_since(parsed.with_timezone(&chrono::Utc));
    Some(age.num_seconds().max(0) as u64)
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
        QueryKind::Identifier => (1.0, 0.0),
        QueryKind::Path => (1.0, 0.0),
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
            SearchMode::Identifier => (0.0, 2.35, 2.15, 1),
            SearchMode::Path => (0.0, 2.5, 1.1, 1),
            SearchMode::Auto => unreachable!("auto mode should be resolved before planning"),
        };

    if request.path_prefix.is_some() || request.file.is_some() {
        lexical_weight += 0.35;
        if dense_weight > 0.0 {
            dense_weight = (dense_weight - 0.15_f64).max(0.15_f64);
        }
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

fn compact_preview(content: &str, max_chars: usize) -> String {
    truncate_for_display(content.trim_matches(['\r', '\n']), max_chars)
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
    default_profile_name: &str,
    configured_embedding_fingerprints: &BTreeMap<String, String>,
) -> IndexIdentityStatus {
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
    } else if let Some(stored) = snapshot.embedding_fingerprint.as_deref()
        && configured_embedding_fingerprints
            .get(default_profile_name)
            .is_some_and(|configured| configured != stored)
    {
        reason = Some(format!(
            "legacy embedding fingerprint mismatch: local state is `{stored}`, current default profile `{default_profile_name}` is `{}`.",
            configured_embedding_fingerprints
                .get(default_profile_name)
                .cloned()
                .unwrap_or_default()
        ));
    }

    IndexIdentityStatus {
        compatible: reason.is_none(),
        index_format_version: snapshot.index_format_version.clone(),
        search_root_version: snapshot.search_root_version.clone(),
        configured_embedding_fingerprints: configured_embedding_fingerprints.clone(),
        reason,
    }
}

fn repo_embedding_identity_status_for_snapshot(
    snapshot: &Snapshot,
    entry: Option<&SnapshotEntry>,
    configured_profile_name: &str,
    configured_fingerprint: Option<String>,
) -> RepoEmbeddingIdentityStatus {
    let stored_profile = entry
        .and_then(|value| value.embedding_profile.clone())
        .unwrap_or_else(|| configured_profile_name.to_string());
    let stored_fingerprint = entry
        .and_then(|value| value.embedding_fingerprint.clone())
        .or_else(|| snapshot.embedding_fingerprint.clone());
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
    } else if stored_profile != configured_profile_name {
        reason = Some(format!(
            "embedding profile mismatch: local state is `{stored_profile}`, current config is `{configured_profile_name}`."
        ));
    } else if let (Some(stored), Some(configured)) = (
        stored_fingerprint.as_deref(),
        configured_fingerprint.as_deref(),
    ) && stored != configured
    {
        reason = Some(format!(
            "embedding fingerprint mismatch: local state is `{stored}`, current config is `{configured}`."
        ));
    }

    RepoEmbeddingIdentityStatus {
        profile_name: configured_profile_name.to_string(),
        configured_fingerprint,
        stored_fingerprint,
        reason,
    }
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
        let bytes = entry.metadata().map(|metadata| metadata.len()).unwrap_or(0);

        files.insert(
            relative_path.clone(),
            RepoFile {
                absolute_path,
                hash,
                bytes,
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

fn text_search_has_bounded_target(scope: &ResolvedScope, request: &TextSearchScopeRequest) -> bool {
    request.file.is_some() || request.path_prefix.is_some() || scope.repos.len() == 1
}

fn text_search_chunk_request(request: &TextSearchScopeRequest) -> ChunkSearchRequest {
    ChunkSearchRequest {
        query: request.query.clone(),
        limit: SEARCH_TEXT_SHORTLIST_LIMIT,
        flavor: query_flavor(classify_query(&request.query)),
        path_prefix: request.path_prefix.clone(),
        language: request.language.clone(),
        file: None,
        extension_filter: request.extension_filter.clone(),
    }
}

fn dedup_relative_paths(mut files: Vec<String>) -> Vec<String> {
    files.sort();
    files.dedup();
    files
}

fn merge_text_candidate_files(
    canonical_files: Vec<String>,
    overlay_files: Vec<String>,
    suppressed_paths: &BTreeSet<String>,
) -> Vec<String> {
    let mut files = canonical_files
        .into_iter()
        .filter(|path| !suppressed_paths.contains(path))
        .collect::<Vec<_>>();
    files.extend(overlay_files);
    dedup_relative_paths(files)
}

fn validate_search_text_fallback_size(
    request: &TextSearchScopeRequest,
    file_count: usize,
) -> Result<()> {
    if request.path_prefix.is_some() && file_count > SEARCH_TEXT_FALLBACK_MAX_FILES {
        bail!(
            "search_text pathPrefix matched {file_count} files; narrow `file` or `pathPrefix` before retrying"
        );
    }
    Ok(())
}

fn apply_query_embedding_failure(
    profile_name: &str,
    usage: &QueryProfileUsage,
    message: &str,
    blocked_repos: &mut HashMap<String, String>,
    overlay_query_errors: &mut HashMap<String, String>,
) {
    for repo in &usage.canonical_repos {
        blocked_repos.insert(repo.clone(), message.to_string());
    }
    let overlay_message =
        format!("overlay semantic search unavailable for profile `{profile_name}`: {message}");
    for repo in &usage.overlay_repos {
        if !blocked_repos.contains_key(repo) {
            overlay_query_errors.insert(repo.clone(), overlay_message.clone());
        }
    }
}

fn search_text_scans_live_repo(request: &TextSearchScopeRequest) -> bool {
    request.file.is_none() && request.path_prefix.is_none()
}

fn collect_live_candidate_files(
    repo: &Path,
    path_prefix: Option<&str>,
    language: Option<&str>,
    extension_filter: &[String],
) -> Result<Vec<String>> {
    let normalized_prefix = path_prefix
        .map(normalize_relative_path)
        .filter(|prefix| !prefix.is_empty());
    let walk_root = if let Some(prefix) = normalized_prefix.as_deref() {
        let prefix_path = repo.join(prefix);
        if prefix_path.is_dir() {
            prefix_path
        } else {
            prefix_path
                .parent()
                .filter(|parent| parent.is_dir())
                .map(Path::to_path_buf)
                .unwrap_or_else(|| repo.to_path_buf())
        }
    } else {
        repo.to_path_buf()
    };

    let mut files = Vec::new();
    let mut builder = WalkBuilder::new(&walk_root);
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
        let Some(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_file() {
            continue;
        }

        let absolute_path = entry.path();
        let Ok(relative_path) = absolute_path.strip_prefix(repo) else {
            continue;
        };
        let relative_path = relative_path.display().to_string().replace('\\', "/");
        if !path_matches_prefix(&relative_path, normalized_prefix.as_deref()) {
            continue;
        }
        if !path_matches_live_filters(&relative_path, language, extension_filter) {
            continue;
        }
        if hash_text_like_file(absolute_path)?.is_none() {
            continue;
        }
        files.push(relative_path);
    }

    files.sort();
    files.dedup();
    Ok(files)
}

fn path_matches_prefix(relative_path: &str, prefix: Option<&str>) -> bool {
    match prefix {
        None => true,
        Some(prefix) => relative_path == prefix || relative_path.starts_with(&format!("{prefix}/")),
    }
}

fn live_file_exists(repo: &Path, relative_path: &str) -> Result<bool> {
    let canonical_repo = repo
        .canonicalize()
        .with_context(|| format!("resolving repo root {}", repo.display()))?;
    let normalized_relative_path = normalize_relative_path(relative_path);
    if normalized_relative_path.is_empty() {
        return Ok(false);
    }

    let requested_path = canonical_repo.join(&normalized_relative_path);
    let metadata = match std::fs::symlink_metadata(&requested_path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("reading metadata for {}", requested_path.display()));
        }
    };
    if !metadata.is_file() {
        return Ok(false);
    }

    let canonical_file = requested_path
        .canonicalize()
        .with_context(|| format!("resolving file {}", requested_path.display()))?;
    if !canonical_file.starts_with(&canonical_repo) {
        bail!(
            "file `{normalized_relative_path}` escapes repo root {}",
            canonical_repo.display()
        );
    }

    Ok(true)
}

fn path_matches_live_filters(
    relative_path: &str,
    language: Option<&str>,
    extension_filter: &[String],
) -> bool {
    let extension = relative_path_extension(relative_path);
    if !extension_filter.is_empty() && !extension_filter.iter().any(|value| value == &extension) {
        return false;
    }
    if language.is_some_and(|expected| language_for_extension(&extension) != expected) {
        return false;
    }
    !extension.is_empty() && default_supported_extensions().contains(&extension)
}

fn relative_path_extension(relative_path: &str) -> String {
    Path::new(relative_path)
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| format!(".{value}"))
        .unwrap_or_default()
}

fn not_found_prepare_response(
    repo: &Path,
    relative_path: Option<&str>,
    symbol_id: Option<&str>,
) -> PrepareEditTargetResponse {
    PrepareEditTargetResponse {
        status: EditTargetStatus::NotFound,
        repo: Some(repo.display().to_string()),
        repo_label: Some(repo_basename(&repo.display().to_string())),
        relative_path: relative_path.map(ToString::to_string),
        start_line: None,
        end_line: None,
        content: None,
        anchors: Vec::new(),
        anchor_quality: None,
        resolution_type: None,
        file_hash: None,
        indexed: None,
        stale: None,
        indexed_at: None,
        indexed_file_hash: None,
        symbol_id: symbol_id.map(ToString::to_string),
        truncated: None,
        candidates: Vec::new(),
        symbol_start_line: None,
        symbol_end_line: None,
        reason_code: None,
        suggested_next_tool: None,
    }
}

fn not_found_prepare_response_for_file(file: &str) -> PrepareEditTargetResponse {
    PrepareEditTargetResponse {
        status: EditTargetStatus::NotFound,
        repo: None,
        repo_label: None,
        relative_path: Some(file.to_string()),
        start_line: None,
        end_line: None,
        content: None,
        anchors: Vec::new(),
        anchor_quality: None,
        resolution_type: None,
        file_hash: None,
        indexed: None,
        stale: None,
        indexed_at: None,
        indexed_file_hash: None,
        symbol_id: None,
        truncated: None,
        candidates: Vec::new(),
        symbol_start_line: None,
        symbol_end_line: None,
        reason_code: None,
        suggested_next_tool: None,
    }
}

fn named_symbol_ambiguous_prepare_response(
    repo: &Path,
    file: &str,
    matches: Vec<IndexedSymbol>,
) -> PrepareEditTargetResponse {
    PrepareEditTargetResponse {
        status: EditTargetStatus::Ambiguous,
        repo: Some(repo.display().to_string()),
        repo_label: Some(repo_basename(&repo.display().to_string())),
        relative_path: Some(file.to_string()),
        start_line: None,
        end_line: None,
        content: None,
        anchors: Vec::new(),
        anchor_quality: None,
        resolution_type: Some(EditResolutionType::Symbol),
        file_hash: None,
        indexed: Some(true),
        stale: Some(false),
        indexed_at: None,
        indexed_file_hash: None,
        symbol_id: None,
        truncated: Some(matches.len() > 8),
        candidates: matches
            .into_iter()
            .take(8)
            .map(|symbol| EditTargetCandidate {
                repo: repo.display().to_string(),
                repo_label: repo_basename(&repo.display().to_string()),
                relative_path: symbol.relative_path.clone(),
                start_line: symbol.start_line,
                end_line: symbol.end_line,
                preview: match symbol.container.as_deref() {
                    Some(container) => {
                        format!("{} {} in {}", symbol.kind, symbol.name, container)
                    }
                    None => format!("{} {}", symbol.kind, symbol.name),
                },
            })
            .collect(),
        symbol_start_line: None,
        symbol_end_line: None,
        reason_code: Some(EditTargetReasonCode::MultipleMatches),
        suggested_next_tool: Some("prepare_edit_target".to_string()),
    }
}

fn ambiguous_prepare_response_from_symbols(
    file: &str,
    resolved: &[ResolvedEditSymbol],
) -> PrepareEditTargetResponse {
    PrepareEditTargetResponse {
        status: EditTargetStatus::Ambiguous,
        repo: None,
        repo_label: None,
        relative_path: Some(file.to_string()),
        start_line: None,
        end_line: None,
        content: None,
        anchors: Vec::new(),
        anchor_quality: None,
        resolution_type: Some(EditResolutionType::Symbol),
        file_hash: None,
        indexed: Some(true),
        stale: Some(false),
        indexed_at: None,
        indexed_file_hash: None,
        symbol_id: None,
        truncated: Some(resolved.len() > 8),
        candidates: resolved
            .iter()
            .take(8)
            .map(|entry| EditTargetCandidate {
                repo: entry.repo.display().to_string(),
                repo_label: repo_basename(&entry.repo.display().to_string()),
                relative_path: entry.symbol.relative_path.clone(),
                start_line: entry.symbol.start_line,
                end_line: entry.symbol.end_line,
                preview: match entry.symbol.container.as_deref() {
                    Some(container) => format!(
                        "{} {} in {}",
                        entry.symbol.kind, entry.symbol.name, container
                    ),
                    None => format!("{} {}", entry.symbol.kind, entry.symbol.name),
                },
            })
            .collect(),
        symbol_start_line: None,
        symbol_end_line: None,
        reason_code: Some(EditTargetReasonCode::MultipleMatches),
        suggested_next_tool: Some("prepare_edit_target".to_string()),
    }
}

fn span_line_count(start_line: u64, end_line: u64) -> u64 {
    end_line.saturating_sub(start_line).saturating_add(1)
}

fn ready_target_reason(
    start_line: u64,
    end_line: u64,
    anchors: &[EditTargetAnchor],
) -> Option<(EditTargetReasonCode, &'static str)> {
    if span_line_count(start_line, end_line) > PREPARE_READY_MAX_LINES as u64 {
        return Some((EditTargetReasonCode::WindowTooBroad, "search_text"));
    }
    if !anchors.iter().any(|anchor| anchor.unique_in_file) {
        return Some((EditTargetReasonCode::WeakAnchors, "search_text"));
    }
    None
}

fn bounded_window(
    total_lines: u64,
    focus_start: u64,
    focus_end: u64,
    before_lines: usize,
    after_lines: usize,
    max_lines: usize,
) -> (u64, u64, bool) {
    if total_lines == 0 {
        return (0, 0, false);
    }

    let max_lines = max_lines.max(1) as u64;
    let focus_start = focus_start.clamp(1, total_lines);
    let focus_end = focus_end.clamp(focus_start, total_lines);
    let mut start_line = focus_start.saturating_sub(before_lines as u64).max(1);
    let mut end_line = (focus_end + after_lines as u64).min(total_lines);
    let mut truncated = false;

    while span_line_count(start_line, end_line) > max_lines {
        truncated = true;
        let extra_before = focus_start.saturating_sub(start_line);
        let extra_after = end_line.saturating_sub(focus_end);
        if extra_after > extra_before && end_line > focus_end {
            end_line = end_line.saturating_sub(1);
        } else if start_line < focus_start {
            start_line = start_line.saturating_add(1);
        } else if end_line > focus_end {
            end_line = end_line.saturating_sub(1);
        } else {
            break;
        }
    }

    (start_line, end_line, truncated)
}

fn narrowest_covering_symbol(symbols: &[IndexedSymbol], line: u64) -> Option<&IndexedSymbol> {
    symbols
        .iter()
        .filter(|symbol| symbol.start_line <= line && symbol.end_line >= line)
        .min_by(|left, right| {
            span_line_count(left.start_line, left.end_line)
                .cmp(&span_line_count(right.start_line, right.end_line))
                .then(left.start_line.cmp(&right.start_line))
                .then(left.end_line.cmp(&right.end_line))
                .then(left.name.cmp(&right.name))
        })
}

fn symbol_fits_ready_window(symbol: &IndexedSymbol, selected: TextMatch, max_lines: usize) -> bool {
    selected.start_line >= symbol.start_line
        && selected.end_line <= symbol.end_line
        && span_line_count(symbol.start_line, symbol.end_line) <= max_lines as u64
        && span_line_count(symbol.start_line, symbol.end_line) <= PREPARE_READY_MAX_LINES as u64
}

fn select_text_match(
    matches: &[TextMatch],
    line_hint: Option<u64>,
    occurrence: Option<usize>,
) -> Option<TextMatch> {
    if matches.is_empty() {
        return None;
    }

    if let Some(occurrence) = occurrence {
        return occurrence
            .checked_sub(1)
            .and_then(|index| matches.get(index).copied());
    }

    if let Some(line_hint) = line_hint {
        let mut ranked = matches
            .iter()
            .copied()
            .map(|matched| {
                let distance = if line_hint < matched.start_line {
                    matched.start_line - line_hint
                } else {
                    line_hint.saturating_sub(matched.end_line)
                };
                (matched, distance)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then(left.0.start_line.cmp(&right.0.start_line))
                .then(left.0.end_line.cmp(&right.0.end_line))
        });
        let first = ranked.first().copied()?;
        if ranked.get(1).is_some_and(|second| second.1 == first.1) {
            return None;
        }
        return Some(first.0);
    }

    (matches.len() == 1).then_some(matches[0])
}

fn select_edit_anchors(
    snapshot: &LiveFileSnapshot,
    start_line: u64,
    end_line: u64,
    anchor_count: usize,
    query: Option<&str>,
    symbol_signature_line: Option<u64>,
) -> Vec<EditTargetAnchor> {
    let anchor_count = anchor_count.clamp(1, 3);
    let mut line_counts = HashMap::new();
    for line in 1..=snapshot.total_lines() {
        let Some(text) = snapshot.line_text(line) else {
            continue;
        };
        let normalized = normalize_anchor_text(text);
        if normalized.is_empty() {
            continue;
        }
        *line_counts.entry(normalized.to_string()).or_insert(0usize) += 1;
    }

    let mut ranked = Vec::new();
    for line in start_line..=end_line {
        let Some(text) = snapshot.line_text(line) else {
            continue;
        };
        let normalized = normalize_anchor_text(text);
        if !is_anchor_candidate(normalized) {
            continue;
        }
        let unique_in_file = line_counts.get(normalized).copied().unwrap_or(0) == 1;
        let query_match = query.is_some_and(|needle| !needle.is_empty() && text.contains(needle));
        let signature_match = symbol_signature_line == Some(line);
        let structural_bonus = normalized
            .chars()
            .any(|character| matches!(character, '=' | ':' | '(' | ')' | '[' | ']' | '{' | '}'))
            as i32;
        let score = (unique_in_file as i32 * 1000)
            + (query_match as i32 * 600)
            + (signature_match as i32 * 500)
            + (normalized.chars().any(char::is_alphanumeric) as i32 * 100)
            + (structural_bonus * 25)
            - (line.saturating_sub(start_line) as i32);
        ranked.push((score, line, normalized.to_string(), unique_in_file));
    }

    ranked.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    ranked.truncate(anchor_count);
    ranked.sort_by_key(|(_, line, _, _)| *line);

    ranked
        .into_iter()
        .map(|(_, line, text, unique_in_file)| EditTargetAnchor {
            line,
            text,
            unique_in_file,
        })
        .collect()
}

fn normalize_anchor_text(text: &str) -> &str {
    text.trim_end_matches(['\r', '\n']).trim()
}

fn is_anchor_candidate(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    if matches!(
        text,
        "{" | "}" | "(" | ")" | "[" | "]" | "," | ");" | "};" | "];"
    ) {
        return false;
    }
    text.chars().any(char::is_alphanumeric)
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
        CONTENT_HASH_ALGORITHM, EditTargetAnchor, EditTargetReasonCode, IndexCompletionStatus,
        IndexCoverage, IndexExecutionMode, LEGACY_CONTENT_HASH_ALGORITHM, MerkleSnapshot,
        OverlayIndexState, QueryProfileUsage, SearchBudgets, SearchMode, SearchRequest, TextMatch,
        TextSearchScopeRequest, apply_query_embedding_failure, bounded_window,
        build_chunk_context_snippet, build_root_hash, chunk_id, classify_query,
        collect_live_candidate_files, collection_name, diff_files, expected_vector_collections,
        failed_overlay_state, hash_text_like_file, index_identity_status_for_snapshot,
        index_status_for_coverage, is_agent_context_vector_collection, live_file_exists,
        merge_text_candidate_files, narrowest_covering_symbol, overlay_lookup_uses_overlay,
        overlay_state_has_indexed_path, overlay_state_has_search_index, overlay_state_load_warning,
        overlay_state_suppresses_canonical, plan_search, ready_target_reason,
        removed_repo_index_result, run_low_priority_blocking, scan_repo,
        search_text_scans_live_repo, select_edit_anchors, select_text_match,
        stale_vector_collections, symbol_collection_name, symbol_fits_ready_window,
        text_search_has_bounded_target, validate_absolute_repo_path,
        validate_search_text_fallback_size, vector_flush_needed, vector_release_needed,
        worktree_overlay_context_from_snapshot,
    };
    use crate::config::{ResolvedScope, ScopeKind, SearchConfig};
    use crate::engine::live_files::LiveFileStore;
    use crate::engine::symbols::IndexedSymbol;
    use crate::snapshot::{Snapshot, WorktreeSnapshotEntry};
    use std::collections::{BTreeMap, BTreeSet, HashMap};
    use std::fs;
    use std::path::{Path, PathBuf};
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

    fn temp_dir(name: &str) -> PathBuf {
        let path = temp_file_path(name);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn select_text_match_requires_disambiguation_for_repeated_hits() {
        let matches = vec![
            TextMatch {
                start_byte: 0,
                end_byte: 4,
                start_line: 10,
                end_line: 10,
            },
            TextMatch {
                start_byte: 10,
                end_byte: 14,
                start_line: 20,
                end_line: 20,
            },
        ];

        assert!(select_text_match(&matches, None, None).is_none());
        assert_eq!(
            select_text_match(&matches, None, Some(2)).map(|matched| matched.start_line),
            Some(20)
        );
    }

    #[test]
    fn bounded_window_keeps_focus_line_and_caps_span() {
        let (start_line, end_line, truncated) = bounded_window(100, 40, 40, 10, 10, 5);

        assert!(truncated);
        assert!(start_line <= 40);
        assert!(end_line >= 40);
        assert_eq!(end_line - start_line + 1, 5);
    }

    #[test]
    fn narrowest_covering_symbol_prefers_inner_symbol() {
        let outer = IndexedSymbol {
            symbol_id: "outer".to_string(),
            repo: "/tmp/repo".to_string(),
            relative_path: "src/lib.rs".to_string(),
            name: "outer".to_string(),
            kind: "function".to_string(),
            container: None,
            language: "rust".to_string(),
            start_line: 10,
            end_line: 30,
            indexed_at: "2026-01-01T00:00:00Z".to_string(),
            file_hash: "hash".to_string(),
            parent_symbol_id: None,
        };
        let inner = IndexedSymbol {
            symbol_id: "inner".to_string(),
            repo: "/tmp/repo".to_string(),
            relative_path: "src/lib.rs".to_string(),
            name: "inner".to_string(),
            kind: "function".to_string(),
            container: Some("outer".to_string()),
            language: "rust".to_string(),
            start_line: 18,
            end_line: 22,
            indexed_at: "2026-01-01T00:00:00Z".to_string(),
            file_hash: "hash".to_string(),
            parent_symbol_id: Some("outer".to_string()),
        };

        let symbols = [outer, inner];
        let selected = narrowest_covering_symbol(&symbols, 20).unwrap();
        assert_eq!(selected.symbol_id, "inner");
    }

    #[test]
    fn symbol_ready_window_requires_selected_match_to_fit_budget() {
        let symbol = IndexedSymbol {
            symbol_id: "helper".to_string(),
            repo: "/tmp/repo".to_string(),
            relative_path: "src/lib.rs".to_string(),
            name: "helper".to_string(),
            kind: "function".to_string(),
            container: None,
            language: "rust".to_string(),
            start_line: 10,
            end_line: 80,
            indexed_at: "2026-01-01T00:00:00Z".to_string(),
            file_hash: "hash".to_string(),
            parent_symbol_id: None,
        };
        let selected = TextMatch {
            start_byte: 120,
            end_byte: 132,
            start_line: 42,
            end_line: 42,
        };

        assert!(symbol_fits_ready_window(&symbol, selected, 96));
        assert!(!symbol_fits_ready_window(&symbol, selected, 32));
    }

    #[test]
    fn select_edit_anchors_prefers_unique_query_line() {
        let repo = temp_dir("anchors-repo");
        let file = repo.join("src").join("lib.rs");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(
            &file,
            "fn alpha() {\n    let shared = 1;\n    let target = build_value();\n    let shared = 1;\n}\n",
        )
        .unwrap();

        let snapshot = LiveFileStore::new(4)
            .load_snapshot(&repo, "src/lib.rs")
            .unwrap();
        let anchors = select_edit_anchors(&snapshot, 1, 5, 2, Some("build_value"), Some(1));

        assert!(!anchors.is_empty());
        assert!(anchors.iter().any(|anchor: &EditTargetAnchor| {
            anchor.text.contains("build_value") && anchor.unique_in_file
        }));
    }

    #[test]
    fn ready_target_reason_requires_narrow_windows() {
        let anchors = vec![EditTargetAnchor {
            line: 100,
            text:
                "let exact_key = download_client_item_identity(client_id, download_client_item_id);"
                    .to_string(),
            unique_in_file: true,
        }];

        assert_eq!(ready_target_reason(100, 259, &anchors), None);
        assert_eq!(
            ready_target_reason(100, 260, &anchors),
            Some((EditTargetReasonCode::WindowTooBroad, "search_text"))
        );
        assert_eq!(
            ready_target_reason(1001, 1161, &anchors),
            Some((EditTargetReasonCode::WindowTooBroad, "search_text"))
        );
    }

    #[test]
    fn ready_target_reason_requires_unique_anchors() {
        let anchors = vec![EditTargetAnchor {
            line: 42,
            text: "}".to_string(),
            unique_in_file: false,
        }];

        assert_eq!(
            ready_target_reason(40, 44, &anchors),
            Some((EditTargetReasonCode::WeakAnchors, "search_text"))
        );
    }

    #[test]
    fn text_search_bounded_target_allows_repo_scopes_without_prefix() {
        let scope = ResolvedScope {
            kind: ScopeKind::Repo,
            id: "/tmp/repo".to_string(),
            label: "repo".to_string(),
            repos: vec![PathBuf::from("/tmp/repo")],
        };
        let request = TextSearchScopeRequest {
            repo: None,
            query: "needle".to_string(),
            limit: 10,
            path_prefix: None,
            language: None,
            file: None,
            extension_filter: Vec::new(),
            case_sensitive: true,
            whole_word: false,
            context_lines: 1,
        };

        assert!(text_search_has_bounded_target(&scope, &request));
    }

    #[test]
    fn text_search_bounded_target_requires_hint_for_group_scope() {
        let scope = ResolvedScope {
            kind: ScopeKind::Group,
            id: "workspace".to_string(),
            label: "Workspace".to_string(),
            repos: vec![PathBuf::from("/tmp/repo-a"), PathBuf::from("/tmp/repo-b")],
        };
        let request = TextSearchScopeRequest {
            repo: None,
            query: "needle".to_string(),
            limit: 10,
            path_prefix: None,
            language: None,
            file: None,
            extension_filter: Vec::new(),
            case_sensitive: true,
            whole_word: false,
            context_lines: 1,
        };

        assert!(!text_search_has_bounded_target(&scope, &request));
    }

    #[test]
    fn repo_bounded_text_search_scans_live_repo() {
        let repo_wide = TextSearchScopeRequest {
            repo: None,
            query: "needle".to_string(),
            limit: 10,
            path_prefix: None,
            language: None,
            file: None,
            extension_filter: Vec::new(),
            case_sensitive: true,
            whole_word: false,
            context_lines: 1,
        };
        let file_scoped = TextSearchScopeRequest {
            file: Some("src/lib.rs".to_string()),
            ..repo_wide.clone()
        };
        let subtree_scoped = TextSearchScopeRequest {
            path_prefix: Some("src".to_string()),
            ..repo_wide.clone()
        };

        assert!(search_text_scans_live_repo(&repo_wide));
        assert!(!search_text_scans_live_repo(&file_scoped));
        assert!(!search_text_scans_live_repo(&subtree_scoped));
    }

    #[test]
    fn text_search_candidate_merge_uses_overlay_and_suppresses_replaced_paths() {
        let suppressed_paths = BTreeSet::from(["src/changed.rs".to_string()]);
        let merged = merge_text_candidate_files(
            vec!["src/changed.rs".to_string(), "src/unchanged.rs".to_string()],
            vec!["src/changed.rs".to_string(), "src/new.rs".to_string()],
            &suppressed_paths,
        );

        assert_eq!(
            merged,
            vec![
                "src/changed.rs".to_string(),
                "src/new.rs".to_string(),
                "src/unchanged.rs".to_string(),
            ]
        );
    }

    #[test]
    fn text_search_fallback_errors_when_path_prefix_is_too_broad() {
        let request = TextSearchScopeRequest {
            repo: None,
            query: "needle".to_string(),
            limit: 10,
            path_prefix: Some("src".to_string()),
            language: None,
            file: None,
            extension_filter: Vec::new(),
            case_sensitive: true,
            whole_word: false,
            context_lines: 1,
        };

        let error =
            validate_search_text_fallback_size(&request, super::SEARCH_TEXT_FALLBACK_MAX_FILES + 1)
                .unwrap_err()
                .to_string();
        assert!(error.contains("pathPrefix matched"));
        assert!(validate_search_text_fallback_size(&request, 1).is_ok());
    }

    #[test]
    fn query_embedding_failure_blocks_canonical_but_only_partials_overlay() {
        let mut usage = QueryProfileUsage::default();
        usage.canonical_repos.insert("/tmp/canonical".to_string());
        usage.overlay_repos.insert("/tmp/overlay".to_string());
        let mut blocked_repos = HashMap::new();
        let mut overlay_errors = HashMap::new();

        apply_query_embedding_failure(
            "local",
            &usage,
            "provider unavailable",
            &mut blocked_repos,
            &mut overlay_errors,
        );

        assert_eq!(
            blocked_repos.get("/tmp/canonical").map(String::as_str),
            Some("provider unavailable")
        );
        assert!(overlay_errors["/tmp/overlay"].contains("overlay semantic search unavailable"));
    }

    #[test]
    fn query_embedding_failure_does_not_add_overlay_partial_for_blocked_repo() {
        let mut usage = QueryProfileUsage::default();
        usage.canonical_repos.insert("/tmp/repo".to_string());
        usage.overlay_repos.insert("/tmp/repo".to_string());
        let mut blocked_repos = HashMap::new();
        let mut overlay_errors = HashMap::new();

        apply_query_embedding_failure(
            "shared",
            &usage,
            "provider unavailable",
            &mut blocked_repos,
            &mut overlay_errors,
        );

        assert!(blocked_repos.contains_key("/tmp/repo"));
        assert!(!overlay_errors.contains_key("/tmp/repo"));
    }

    #[test]
    fn explicit_refresh_uses_parallel_file_preparation() {
        assert_eq!(IndexExecutionMode::Standard.file_prepare_parallelism(), 1);
        assert!(IndexExecutionMode::ExplicitRefresh.file_prepare_parallelism() >= 2);
    }

    #[test]
    fn collect_live_candidate_files_respects_prefix_and_language() {
        let repo = temp_dir("candidates-repo");
        let prefix = repo.join("src").join("pipeline");
        fs::create_dir_all(&prefix).unwrap();
        fs::write(prefix.join("worker.rs"), "fn build_worker() {}\n").unwrap();
        fs::write(prefix.join("worker.py"), "def build_worker():\n    pass\n").unwrap();
        fs::write(repo.join("src").join("other.rs"), "fn other() {}\n").unwrap();
        fs::write(
            repo.join("src").join("pipeline-two.rs"),
            "fn other_pipeline() {}\n",
        )
        .unwrap();

        let rust_only =
            collect_live_candidate_files(&repo, Some("src/pipeline"), Some("rust"), &[]).unwrap();
        let explicit_python =
            collect_live_candidate_files(&repo, Some("src/pipeline"), None, &[String::from(".py")])
                .unwrap();
        let repo_wide = collect_live_candidate_files(&repo, None, Some("rust"), &[]).unwrap();

        assert_eq!(rust_only, vec!["src/pipeline/worker.rs".to_string()]);
        assert_eq!(explicit_python, vec!["src/pipeline/worker.py".to_string()]);
        assert_eq!(
            repo_wide,
            vec![
                "src/other.rs".to_string(),
                "src/pipeline-two.rs".to_string(),
                "src/pipeline/worker.rs".to_string(),
            ]
        );
    }

    #[test]
    fn live_file_exists_distinguishes_missing_paths_from_escape_attempts() {
        let repo = temp_dir("live-file-exists-repo");
        let src_dir = repo.join("src");
        fs::create_dir_all(&src_dir).unwrap();
        fs::write(src_dir.join("lib.rs"), "fn main() {}\n").unwrap();

        assert!(live_file_exists(&repo, "src/lib.rs").unwrap());
        assert!(!live_file_exists(&repo, "src/missing.rs").unwrap());

        let outside = temp_file_path("outside.rs");
        fs::write(&outside, "fn outside() {}\n").unwrap();
        let escaped = format!("../{}", outside.file_name().unwrap().to_string_lossy());
        assert!(live_file_exists(&repo, &escaped).is_err());

        let _ = fs::remove_file(outside);
    }

    #[test]
    fn collection_name_matches_upstream_prefix_and_hash() {
        let collection = collection_name(Path::new("/tmp/example"));
        assert_eq!(collection, "hybrid_code_chunks_89a35363");
    }

    #[test]
    fn vector_collection_helpers_identify_expected_and_stale_collections() {
        let repos = vec![PathBuf::from("/tmp/example"), PathBuf::from("/tmp/another")];
        let expected = expected_vector_collections(&repos, &Snapshot::default());
        assert!(expected.contains(&collection_name(Path::new("/tmp/example"))));
        assert!(expected.contains(&symbol_collection_name(Path::new("/tmp/example"))));

        let stale = stale_vector_collections(
            &[
                collection_name(Path::new("/tmp/example")),
                "hybrid_code_chunks_deadbeef".to_string(),
                "unrelated_collection".to_string(),
            ],
            &expected,
        );

        assert_eq!(stale, vec!["hybrid_code_chunks_deadbeef"]);
        assert!(is_agent_context_vector_collection(
            "hybrid_symbols_deadbeefdeadbeef"
        ));
        assert!(!is_agent_context_vector_collection("not_agent_context"));
    }

    #[test]
    fn vector_release_runs_for_reindex_or_actual_changes() {
        assert!(vector_release_needed(
            true,
            &super::RepoChangeSummary::default()
        ));
        assert!(vector_release_needed(
            false,
            &super::RepoChangeSummary {
                added: 0,
                modified: 1,
                removed: 0,
            }
        ));
        assert!(!vector_release_needed(
            false,
            &super::RepoChangeSummary::default()
        ));
    }

    #[test]
    fn vector_flush_runs_only_for_reindex_or_larger_batches() {
        assert!(vector_flush_needed(
            true,
            &super::RepoChangeSummary::default()
        ));
        assert!(!vector_flush_needed(
            false,
            &super::RepoChangeSummary {
                added: 1,
                modified: 0,
                removed: 0,
            }
        ));
        assert!(vector_flush_needed(
            false,
            &super::RepoChangeSummary {
                added: 16,
                modified: 16,
                removed: 0,
            }
        ));
    }

    #[test]
    fn removed_repo_index_result_reports_cleanup_without_error() {
        let result = removed_repo_index_result(Path::new("/tmp/deleted-repo"), 42, false);

        assert_eq!(result.repo, "/tmp/deleted-repo");
        assert_eq!(result.index_status.as_deref(), Some("removed"));
        assert_eq!(result.changes.removed, 42);
        assert!(result.indexed_files.is_none());
        assert!(result.total_chunks.is_none());
        assert!(result.error.is_none());
    }

    #[test]
    fn deleted_worktree_overlay_context_can_be_rebuilt_from_snapshot_entry() {
        let entry = WorktreeSnapshotEntry::indexed(
            "/tmp/canonical",
            "common-dir",
            "overlay123",
            "completed",
            2,
            1,
            128,
            Some("local".to_string()),
            Some("local:model:1024".to_string()),
        );
        let overlay =
            worktree_overlay_context_from_snapshot(Path::new("/tmp/missing-worktree"), &entry);

        assert_eq!(
            overlay.resolution.worktree_root,
            PathBuf::from("/tmp/missing-worktree")
        );
        assert_eq!(
            overlay.resolution.canonical_root,
            PathBuf::from("/tmp/canonical")
        );
        assert_eq!(overlay.resolution.repo_identity, "common-dir");
        assert_eq!(overlay.resolution.overlay_id, "overlay123");
        assert_eq!(overlay.repo_key, "overlay:overlay123");
        assert_eq!(
            overlay.storage_root,
            PathBuf::from(super::OVERLAY_STORAGE_ROOT_PREFIX).join("overlay123")
        );
        assert!(overlay.chunk_collection.contains("overlay123"));
        assert!(overlay.symbol_collection.contains("overlay123"));
    }

    #[test]
    fn overlay_lookup_gates_stale_and_too_large_state() {
        let completed = OverlayIndexState {
            overlay_status: Some("completed".to_string()),
            indexed_hashes: vec![("src/changed.rs".to_string(), "hash".to_string())],
            ..OverlayIndexState::default()
        };
        assert!(overlay_state_has_indexed_path(&completed, "src/changed.rs"));
        assert!(overlay_lookup_uses_overlay(&completed, "src/changed.rs"));

        let stale = OverlayIndexState {
            overlay_status: Some("stale".to_string()),
            indexed_hashes: vec![("src/changed.rs".to_string(), "hash".to_string())],
            replaced_paths: vec!["src/changed.rs".to_string()],
            ..OverlayIndexState::default()
        };
        assert!(!overlay_state_has_indexed_path(&stale, "src/changed.rs"));
        assert!(!overlay_lookup_uses_overlay(&stale, "src/changed.rs"));

        let too_large = OverlayIndexState {
            overlay_status: Some("too_large".to_string()),
            replaced_paths: vec!["src/changed.rs".to_string()],
            ..OverlayIndexState::default()
        };
        assert!(!overlay_state_has_indexed_path(
            &too_large,
            "src/changed.rs"
        ));
        assert!(overlay_lookup_uses_overlay(&too_large, "src/changed.rs"));
    }

    #[test]
    fn failed_overlay_state_does_not_search_or_suppress_canonical() {
        let entry = WorktreeSnapshotEntry::indexed(
            "/tmp/canonical",
            "common-dir",
            "overlay123",
            "completed",
            2,
            1,
            128,
            Some("local".to_string()),
            Some("local:model:1024".to_string()),
        );
        let overlay = worktree_overlay_context_from_snapshot(Path::new("/tmp/worktree"), &entry);
        let failed = failed_overlay_state(
            &overlay,
            "/tmp/worktree",
            "/tmp/canonical",
            Some("local".to_string()),
            Some("local:model:1024".to_string()),
        );

        assert_eq!(failed.overlay_status.as_deref(), Some("failed"));
        assert!(!overlay_state_has_search_index(&failed));
        assert!(!overlay_state_suppresses_canonical(&failed));
        assert!(!overlay_lookup_uses_overlay(&failed, "src/changed.rs"));
    }

    #[test]
    fn text_candidate_merge_keeps_canonical_when_overlay_unavailable() {
        let merged = merge_text_candidate_files(
            vec!["src/a.rs".to_string(), "src/b.rs".to_string()],
            Vec::new(),
            &BTreeSet::new(),
        );

        assert_eq!(merged, vec!["src/a.rs".to_string(), "src/b.rs".to_string()]);
    }

    #[test]
    fn overlay_state_load_warning_is_repo_scoped_and_actionable() {
        let error = anyhow::anyhow!("parsing overlay state /tmp/state.json");
        let warning = overlay_state_load_warning(&error);

        assert!(warning.contains("overlay state unavailable"));
        assert!(warning.contains("parsing overlay state"));
    }

    #[test]
    fn absolute_validation_allows_missing_paths_for_cleanup() {
        assert!(validate_absolute_repo_path(Path::new("/tmp/deleted-repo")).is_ok());
        assert!(validate_absolute_repo_path(Path::new("relative/repo")).is_err());
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
    fn identity_status_uses_actual_default_profile_for_legacy_fingerprint() {
        let snapshot = Snapshot {
            format_version: "v3".to_string(),
            index_format_version: "v1".to_string(),
            search_root_version: "v1".to_string(),
            embedding_fingerprint: Some("hosted-fingerprint".to_string()),
            ..Snapshot::default()
        };
        let configured = BTreeMap::from([
            ("aaa-local".to_string(), "local-fingerprint".to_string()),
            ("zzz-hosted".to_string(), "hosted-fingerprint".to_string()),
        ]);

        let status = index_identity_status_for_snapshot(&snapshot, "zzz-hosted", &configured);

        assert!(status.compatible);
        assert!(status.reason.is_none());
    }

    #[test]
    fn identity_status_mismatch_mentions_default_profile() {
        let snapshot = Snapshot {
            format_version: "v3".to_string(),
            index_format_version: "v1".to_string(),
            search_root_version: "v1".to_string(),
            embedding_fingerprint: Some("old-fingerprint".to_string()),
            ..Snapshot::default()
        };
        let configured = BTreeMap::from([
            ("aaa-local".to_string(), "local-fingerprint".to_string()),
            ("zzz-hosted".to_string(), "hosted-fingerprint".to_string()),
        ]);

        let status = index_identity_status_for_snapshot(&snapshot, "zzz-hosted", &configured);

        assert!(!status.compatible);
        assert!(
            status
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("zzz-hosted")
        );
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
            snippet_chars: 360,
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
            snippet_chars: 360,
        });

        assert!(constrained.lexical_weight > unconstrained.lexical_weight);
        assert!(constrained.dense_weight < unconstrained.dense_weight);
        assert!(constrained.symbol_lexical_share > unconstrained.symbol_lexical_share);
    }

    #[test]
    fn plan_search_keeps_identifier_and_path_queries_off_dense_backends() {
        for query in ["GraphQLResolver", "server/crates/api/src/schema.rs"] {
            let plan = plan_search(&SearchRequest {
                query: query.to_string(),
                limit: 10,
                mode: SearchMode::Auto,
                extension_filter: Vec::new(),
                path_prefix: None,
                language: None,
                file: None,
                dedupe_by_file: true,
                snippet_chars: 360,
            });

            assert_eq!(plan.dense_weight, 0.0);
            assert_eq!(plan.symbol_semantic_share, 0.0);
            assert!(plan.symbol_lexical_share > 0.0);
        }
    }

    #[test]
    fn index_status_only_reports_limit_when_processing_hits_chunk_limit() {
        let coverage = IndexCoverage {
            indexed_files: 561,
            total_chunks: 16_448,
        };

        assert_eq!(
            index_status_for_coverage(IndexCompletionStatus::Completed, coverage),
            "completed"
        );
        assert_eq!(
            index_status_for_coverage(IndexCompletionStatus::LimitReached, coverage),
            "limit_reached"
        );
        assert_eq!(
            index_status_for_coverage(
                IndexCompletionStatus::Completed,
                IndexCoverage {
                    indexed_files: 0,
                    total_chunks: 0,
                },
            ),
            "empty"
        );
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
